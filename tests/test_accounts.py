"""Accounts, sessions and saved documents: sign up / log in / log out, owner-only access to saved documents,
guest documents unchanged, library listing with stats, rename/delete, settings, expiry rules, password storage."""
import importlib
import os
import sqlite3
import sys
import time

import pytest

from conftest import LOREM, PDF, ROOT, paragraph, text
import accounts


@pytest.fixture
def app_mod(tmp, monkeypatch):
    monkeypatch.setenv("PARSEFLOW_DATA_DIR", str(tmp / "data"))
    sys.modules.pop("app", None)
    sys.path.insert(0, str(ROOT / "backend"))
    return importlib.import_module("app")


def _client(app_mod):
    from fastapi.testclient import TestClient
    return TestClient(app_mod.app)


def _pdf(tmp, name="doc.pdf", title="Saved Title"):
    return PDF().page(lambda p: (text(p, 72, 90, title, 20, "hebo"), paragraph(p, 72, 130, 450, LOREM))).save(tmp / name)


def _upload(c, f):
    r = c.post("/api/process", files={"file": (f.name, f.read_bytes(), "application/pdf")}).json()
    for _ in range(300):
        if c.get(f"/api/status/{r['document_id']}").json().get("status") != "running":
            break
        time.sleep(0.05)
    return r


def _signup(c, email="ada@example.com", pw="correct horse", name="Ada"):
    return c.post("/api/auth/signup", json=dict(name=name, email=email, password=pw))


def test_signup_login_logout_and_validation(app_mod):
    c = _client(app_mod)
    assert c.get("/api/auth/me").json() == {"user": None}
    assert _signup(c, pw="short").json()["error"]["code"] == "PF-401"
    assert _signup(c, email="not-an-email").status_code == 400
    r = _signup(c)
    assert r.status_code == 200 and r.json()["user"]["email"] == "ada@example.com"
    cookie = r.headers["set-cookie"].lower()
    assert "pf_session=" in cookie and "httponly" in cookie and "samesite=lax" in cookie
    assert c.get("/api/auth/me").json()["user"]["name"] == "Ada"
    assert _signup(c).status_code == 409                           # duplicate email
    c.post("/api/auth/logout")
    assert c.get("/api/auth/me").json()["user"] is None
    assert c.post("/api/auth/login", json=dict(email="ADA@example.com", password="nope")).status_code == 401
    assert c.post("/api/auth/login", json=dict(email="ADA@example.com", password="correct horse")).json()["user"]["name"] == "Ada"


def test_passwords_and_sessions_are_not_stored_in_clear(app_mod):
    c = _client(app_mod)
    r = _signup(c)
    token = r.cookies.get("pf_session")
    db = sqlite3.connect(app_mod.DATA / "parseflow.db")
    pw = db.execute("SELECT pw FROM users").fetchone()[0]
    tokens = [t for (t,) in db.execute("SELECT token FROM sessions")]
    assert pw.startswith("scrypt$") and "correct horse" not in pw
    assert token and token not in tokens and len(tokens) == 1
    assert accounts.check_password("correct horse", pw) and not accounts.check_password("correct horsE", pw)


def test_login_rate_limit(app_mod):
    c = _client(app_mod)
    _signup(c); c.post("/api/auth/logout")
    codes = [c.post("/api/auth/login", json=dict(email="ada@example.com", password="bad")).status_code for _ in range(9)]
    assert codes[:8] == [401] * 8 and codes[8] == 429


def test_saved_documents_are_private_and_listed(tmp, app_mod):
    a, b, guest = _client(app_mod), _client(app_mod), _client(app_mod)
    _signup(a); _signup(b, email="bob@example.com", name="Bob")
    r = _upload(a, _pdf(tmp))
    did = r["document_id"]
    assert r["saved"] is True
    lib = a.get("/api/my/documents").json()
    d = lib["documents"][0]
    assert d["id"] == did and d["status"] == "complete" and d["stats"]["blocks"] >= 2 and d["stats"]["pages"] == 1
    assert lib["usage"] == dict(documents=1, pages=1, bytes=d["size"])
    for path in (f"/api/document/{did}", f"/api/document/{did}/markdown", f"/api/status/{did}", f"/api/page/{did}/1.png",
                 f"/api/thumb/{did}/1.jpg", f"/api/ask/{did}?q=title"):
        assert a.get(path).status_code == 200, path
        assert b.get(path).status_code == 404, path                 # another account
        assert guest.get(path).status_code == 404, path             # not logged in
    assert b.get("/api/my/documents").json()["documents"] == []
    assert guest.get("/api/my/documents").status_code == 401
    assert b.delete(f"/api/my/documents/{did}").status_code == 404 and (app_mod.DATA / did).exists()


def test_guest_uploads_still_work_and_are_not_saved(tmp, app_mod):
    g = _client(app_mod)
    r = _upload(g, _pdf(tmp, "guest.pdf"))
    assert r["saved"] is False
    assert g.get(f"/api/document/{r['document_id']}").status_code == 200
    assert _client(app_mod).get(f"/api/document/{r['document_id']}").status_code == 404   # private to the uploading browser
    assert g.get("/api/my/documents").status_code == 401


def test_rename_delete_and_expiry(tmp, app_mod, monkeypatch):
    c = _client(app_mod)
    _signup(c)
    saved = _upload(c, _pdf(tmp, "a.pdf"))["document_id"]
    guest = _upload(_client(app_mod), _pdf(tmp, "g.pdf"))["document_id"]
    assert c.patch(f"/api/my/documents/{saved}", json=dict(filename="Quarterly report.pdf")).json()["document"]["filename"] == "Quarterly report.pdf"
    assert c.patch(f"/api/my/documents/{saved}", json=dict(filename="  ")).status_code == 400
    # expiry: guest uploads older than the TTL are removed, saved ones are kept
    old = time.time() - 3 * 86400
    for d in (saved, guest):
        os.utime(app_mod.DATA / d, (old, old))
    app_mod._cleanup_old()
    assert (app_mod.DATA / saved).exists() and not (app_mod.DATA / guest).exists()
    assert c.delete(f"/api/my/documents/{saved}").json()["ok"]
    assert not (app_mod.DATA / saved).exists() and c.get("/api/my/documents").json()["documents"] == []


def test_settings_password_change_and_account_deletion(tmp, app_mod):
    c, other = _client(app_mod), _client(app_mod)
    _signup(c)
    other.post("/api/auth/login", json=dict(email="ada@example.com", password="correct horse"))
    assert other.get("/api/auth/me").json()["user"]
    assert c.patch("/api/account", json=dict(name="Ada L.")).json()["user"]["name"] == "Ada L."
    assert c.post("/api/account/password", json=dict(current="wrong", new="new password 1")).status_code == 401
    assert c.post("/api/account/password", json=dict(current="correct horse", new="new password 1")).json()["ok"]
    assert c.get("/api/auth/me").json()["user"]                      # this session stays
    assert other.get("/api/auth/me").json()["user"] is None          # other sessions are signed out
    did = _upload(c, _pdf(tmp))["document_id"]
    assert c.post("/api/account/delete", json=dict(password="correct horse")).status_code == 401
    r = c.post("/api/account/delete", json=dict(password="new password 1")).json()
    assert r == dict(ok=True, deleted_documents=1) and not (app_mod.DATA / did).exists()
    assert c.get("/api/auth/me").json()["user"] is None
    assert c.post("/api/auth/login", json=dict(email="ada@example.com", password="new password 1")).status_code == 401


def test_interrupted_processing_marked_failed_on_restart(tmp):
    st = accounts.Store(tmp / "x.db")
    u = st.signup("Ada", "ada@example.com", "correct horse")
    st.add_document("abcdefabcdef", u["id"], "a.pdf", "PDF", 10, 1)
    st2 = accounts.Store(tmp / "x.db")
    d = st2.documents(u["id"])[0]
    assert d["status"] == "failed" and "interrupted" in d["error"]


def test_env_example_loads_and_every_number_parses(app_mod):
    env = {}
    app_mod.load_env(ROOT / ".env.example", env)
    assert env["PARSEFLOW_JOB_TTL_HOURS"] == "24" and env["MAX_PAGES"] == "0" and env["MAX_UPLOAD_MB"] == "0" and env["PARSEFLOW_ISOLATE"] == "1" and env["PARSEFLOW_SECURE_COOKIE"] == "0"
    for k, v in env.items():
        assert "#" not in v, k
    for k in ("MAX_PAGES", "MAX_UPLOAD_MB", "PARSEFLOW_JOB_TTL_HOURS", "PARSEFLOW_SESSION_DAYS", "PARSEFLOW_WORKERS", "OCR_TIMEOUT", "ASK_TIMEOUT"):
        float(env[k])
    env2 = {}
    p = ROOT / "tests" / "_tmp_env"
    p.write_text("A=1   # note\n  # whole-line comment\nB=x#y\nC=\n")
    try:
        app_mod.load_env(p, env2)
    finally:
        p.unlink()
    assert env2 == {"A": "1", "B": "x#y"}
