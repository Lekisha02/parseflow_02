"""Email verification, password reset (emails land in the outbox without SMTP), optional verification before login,
private guest documents with an owner-controlled share link, status read from disk, HTTPS cookies behind a proxy,
and the PostgreSQL statement translation."""
import importlib
import re
import sys
import time

from conftest import LOREM, PDF, ROOT, paragraph, text
import accounts


def _app(tmp, monkeypatch, **env):
    monkeypatch.setenv("PARSEFLOW_DATA_DIR", str(tmp / "data"))
    monkeypatch.delenv("SMTP_HOST", raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    sys.modules.pop("app", None)
    sys.path.insert(0, str(ROOT / "backend"))
    return importlib.import_module("app")


def _client(app):
    from fastapi.testclient import TestClient
    return TestClient(app.app)


def _last_link(app, kind):
    from email import message_from_bytes, policy
    mails = sorted((app.DATA / "outbox").glob("*.eml"), key=lambda p: p.stat().st_mtime_ns)
    body = message_from_bytes(mails[-1].read_bytes(), policy=policy.default).get_content()
    return re.search(r"#%s=([A-Za-z0-9_-]+)" % kind, body).group(1), body


def _signup(c, email="ada@example.com", pw="correct horse"):
    return c.post("/api/auth/signup", json=dict(name="Ada", email=email, password=pw))


def test_email_verification_flow(tmp, monkeypatch):
    app = _app(tmp, monkeypatch, PARSEFLOW_PUBLIC_URL="https://parseflow.example.org")
    c = _client(app)
    r = _signup(c).json()
    assert r["user"]["verified"] is False and r["verification_sent"] is True
    token, body = _last_link(app, "verify")
    assert "https://parseflow.example.org/#verify=" in body and "Confirm" in body
    other = _client(app)
    assert other.post("/api/auth/verify", json=dict(token="nope")).status_code == 400
    v = other.post("/api/auth/verify", json=dict(token=token))
    assert v.status_code == 200 and v.json()["user"]["verified"] is True and other.get("/api/auth/me").json()["user"]["email"] == "ada@example.com"
    assert other.post("/api/auth/verify", json=dict(token=token)).status_code == 400          # single use
    assert app.health()["email"] == "outbox"


def test_password_reset_flow(tmp, monkeypatch):
    app = _app(tmp, monkeypatch)
    a, b = _client(app), _client(app)
    _signup(a)
    assert b.post("/api/auth/reset/request", json=dict(email="nobody@example.com")).json() == {"ok": True}  # no enumeration
    n0 = len(list((app.DATA / "outbox").glob("*.eml")))
    assert b.post("/api/auth/reset/request", json=dict(email="ADA@example.com")).json() == {"ok": True}
    assert len(list((app.DATA / "outbox").glob("*.eml"))) == n0 + 1
    token, body = _last_link(app, "reset")
    assert b.post("/api/auth/reset", json=dict(token=token, password="short")).status_code == 400
    r = b.post("/api/auth/reset", json=dict(token=token, password="a brand new pw"))
    assert r.status_code == 200 and r.json()["user"]["verified"] is True
    assert a.get("/api/auth/me").json()["user"] is None                                     # other sessions signed out
    assert b.post("/api/auth/reset", json=dict(token=token, password="another one!")).status_code == 400
    assert _client(app).post("/api/auth/login", json=dict(email="ada@example.com", password="a brand new pw")).status_code == 200
    assert _client(app).post("/api/auth/login", json=dict(email="ada@example.com", password="correct horse")).status_code == 401


def test_verification_can_be_required(tmp, monkeypatch):
    app = _app(tmp, monkeypatch, PARSEFLOW_REQUIRE_VERIFIED="1")
    c = _client(app)
    r = _signup(c).json()
    assert r["needs_verification"] and r["user"] is None and c.get("/api/auth/me").json()["user"] is None
    assert c.post("/api/auth/login", json=dict(email="ada@example.com", password="correct horse")).status_code == 403
    token, _ = _last_link(app, "verify")
    assert c.post("/api/auth/verify", json=dict(token=token)).status_code == 200
    assert c.post("/api/auth/login", json=dict(email="ada@example.com", password="correct horse")).status_code == 200


def _wait(c, did):
    for _ in range(400):
        if c.get(f"/api/status/{did}").json().get("status") in ("complete", "failed"):
            return
        time.sleep(0.05)


def test_guest_documents_are_private_until_shared(tmp, monkeypatch):
    app = _app(tmp, monkeypatch, PARSEFLOW_ISOLATE="0")
    owner, stranger = _client(app), _client(app)
    f = PDF().page(lambda p: (text(p, 72, 90, "Private", 20, "hebo"), paragraph(p, 72, 130, 450, LOREM))).save(tmp / "g.pdf")
    did = owner.post("/api/process", files={"file": ("g.pdf", f.read_bytes(), "application/pdf")}).json()["document_id"]
    assert "pf_guest" in owner.cookies
    _wait(owner, did)
    for path in (f"/api/document/{did}", f"/api/page/{did}/1.png", f"/api/document/{did}/markdown", f"/api/status/{did}"):
        assert owner.get(path).status_code == 200 and stranger.get(path).status_code == 404, path
    assert stranger.post(f"/api/document/{did}/share", json=dict(shared=True)).status_code == 404
    owner.post(f"/api/ask/{did}", json=dict(question="What is private?"))
    s = owner.post(f"/api/document/{did}/share", json=dict(shared=True)).json()
    assert s["shared"] and s["link"].endswith(f"/#doc={did}")
    assert stranger.get(f"/api/document/{did}").status_code == 200
    assert stranger.get(f"/api/document/{did}/access").json() == dict(owner=False, shared=True, saved=False)
    assert stranger.get(f"/api/chat/{did}").json()["messages"] == []                       # the owner's chat stays private
    stranger.post(f"/api/ask/{did}", json=dict(question="Stranger question"))
    assert [m.get("text") for m in owner.get(f"/api/chat/{did}").json()["messages"] if m["role"] == "user"] == ["What is private?"]
    assert stranger.delete(f"/api/chat/{did}").status_code == 404
    owner.post(f"/api/document/{did}/share", json=dict(shared=False))
    assert stranger.get(f"/api/document/{did}").status_code == 404
    # after a restart (no job in memory) the status comes from the files
    app.JOBS.clear()
    assert owner.get(f"/api/status/{did}").json()["status"] == "complete"


def test_https_cookie_behind_a_proxy(tmp, monkeypatch):
    app = _app(tmp, monkeypatch)
    c = _client(app)
    r = c.post("/api/auth/signup", json=dict(name="A", email="a@example.com", password="correct horse"), headers={"x-forwarded-proto": "https"})
    assert "secure" in r.headers["set-cookie"].lower()
    r = c.post("/api/auth/signup", json=dict(name="B", email="b@example.com", password="correct horse"))
    assert "secure" not in r.headers["set-cookie"].lower()


def test_postgres_statement_translation_and_old_databases(tmp):
    assert accounts.pg_sql("SELECT * FROM users WHERE email=? AND x LIKE 'a%'") == "SELECT * FROM users WHERE email=%s AND x LIKE 'a%%'"
    import sqlite3
    db = tmp / "old.db"
    c = sqlite3.connect(db)
    c.executescript("CREATE TABLE users(id INTEGER PRIMARY KEY, email TEXT UNIQUE NOT NULL, name TEXT NOT NULL, pw TEXT NOT NULL, created REAL NOT NULL);")
    c.execute("INSERT INTO users(email,name,pw,created) VALUES('o@example.com','Old',?,1)", (accounts.hash_password("correct horse"),))
    c.commit(); c.close()
    st = accounts.Store(db)                                  # a database from an earlier version gains the new column
    assert st.public(st.login("o@example.com", "correct horse"))["verified"] is False


def test_smtp_delivery(tmp, monkeypatch):
    """The SMTP path, against a minimal local SMTP server."""
    import socketserver
    import threading
    got = []

    class H(socketserver.StreamRequestHandler):
        def handle(self):
            w = lambda s: self.wfile.write((s + "\r\n").encode())
            w("220 stub ready")
            data, in_data = [], False
            for raw in self.rfile:
                line = raw.decode().rstrip("\r\n")
                if in_data:
                    if line == ".":
                        in_data = False; got.append("\n".join(data)); w("250 queued")
                    else:
                        data.append(line)
                    continue
                cmd = line.split(" ")[0].upper()
                if cmd in ("EHLO", "HELO"):
                    w("250 stub")
                elif cmd == "DATA":
                    in_data = True; w("354 go")
                elif cmd == "QUIT":
                    w("221 bye"); break
                else:
                    w("250 ok")

    srv = socketserver.ThreadingTCPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        app = _app(tmp, monkeypatch, SMTP_HOST="127.0.0.1", SMTP_PORT=str(srv.server_address[1]), SMTP_SECURITY="none",
                   SMTP_FROM="ParseFlow <pf@example.org>")
        monkeypatch.setenv("SMTP_HOST", "127.0.0.1")
        c = _client(app)
        _signup(c)
        assert got and "Subject: Confirm your ParseFlow email address" in got[0] and "To: ada@example.com" in got[0]
        assert not (app.DATA / "outbox").exists()
    finally:
        srv.shutdown()
