"""Local accounts, sessions and saved documents for the ParseFlow web app.

Everything stays on the ParseFlow server: one SQLite file (PARSEFLOW_DATA_DIR/parseflow.db), standard library only.

* Passwords: scrypt (n=2^14, r=8, p=1) with a per-user random salt; never stored or logged in clear.
* Sessions: a random 256-bit token in an HttpOnly, SameSite=Lax cookie; only its SHA-256 is stored, so a copy of
  the database does not contain usable sessions. Sessions expire after PARSEFLOW_SESSION_DAYS (default 30).
* Documents: a row per upload made while logged in. Guest uploads have no row and expire after
  PARSEFLOW_JOB_TTL_HOURS as before; saved documents stay until their owner deletes them (or the account).
* Email verification and password reset: single-use random tokens (only their SHA-256 is stored) that expire
  (24 h to verify, 1 h to reset). A reset signs out every session.
* Database: SQLite by default; PostgreSQL when DATABASE_URL=postgresql://... is set (needs `psycopg`), for running
  several ParseFlow servers against one database.
"""
import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
import threading
import time

EMAIL = re.compile(r"^[^@\s]{1,64}@[^@\s]{1,190}\.[^@\s]{2,24}$")
MIN_PASSWORD = 8
SESSION_DAYS = float(os.getenv("PARSEFLOW_SESSION_DAYS", "30"))
COOKIE = "pf_session"

SCHEMA = """
CREATE TABLE IF NOT EXISTS users(
  id INTEGER PRIMARY KEY, email TEXT UNIQUE NOT NULL, name TEXT NOT NULL, pw TEXT NOT NULL, created REAL NOT NULL);
CREATE TABLE IF NOT EXISTS sessions(
  token TEXT PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE, created REAL NOT NULL, expires REAL NOT NULL);
CREATE TABLE IF NOT EXISTS documents(
  id TEXT PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE, filename TEXT NOT NULL,
  type TEXT, size INTEGER, pages INTEGER, created REAL NOT NULL, status TEXT NOT NULL, stats TEXT, error TEXT);
CREATE INDEX IF NOT EXISTS documents_user ON documents(user_id, created);
CREATE TABLE IF NOT EXISTS tokens(
  token TEXT PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE, kind TEXT NOT NULL, expires REAL NOT NULL);
"""

PG_SCHEMA = """
CREATE TABLE IF NOT EXISTS users(
  id BIGSERIAL PRIMARY KEY, email TEXT UNIQUE NOT NULL, name TEXT NOT NULL, pw TEXT NOT NULL, created DOUBLE PRECISION NOT NULL,
  verified DOUBLE PRECISION);
CREATE TABLE IF NOT EXISTS sessions(
  token TEXT PRIMARY KEY, user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE, created DOUBLE PRECISION NOT NULL,
  expires DOUBLE PRECISION NOT NULL);
CREATE TABLE IF NOT EXISTS documents(
  id TEXT PRIMARY KEY, user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE, filename TEXT NOT NULL,
  type TEXT, size BIGINT, pages INTEGER, created DOUBLE PRECISION NOT NULL, status TEXT NOT NULL, stats TEXT, error TEXT);
CREATE INDEX IF NOT EXISTS documents_user ON documents(user_id, created);
CREATE TABLE IF NOT EXISTS tokens(
  token TEXT PRIMARY KEY, user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE, kind TEXT NOT NULL,
  expires DOUBLE PRECISION NOT NULL);
"""
VERIFY_HOURS, RESET_HOURS = 24, 1


def pg_sql(sql):
    """SQLite-style statement -> PostgreSQL ('?' placeholders -> '%s')."""
    return sql.replace("%", "%%").replace("?", "%s")


class AuthError(Exception):
    def __init__(self, code, message, status=400):
        super().__init__(message)
        self.code, self.message, self.status = code, message, status


def hash_password(pw, salt=None):
    salt = salt or secrets.token_bytes(16)
    h = hashlib.scrypt(pw.encode("utf-8"), salt=salt, n=2 ** 14, r=8, p=1, dklen=32)
    return "scrypt$%s$%s" % (salt.hex(), h.hex())


def check_password(pw, stored):
    try:
        kind, salt, h = stored.split("$")
        return kind == "scrypt" and hmac.compare_digest(hash_password(pw, bytes.fromhex(salt)), stored)
    except Exception:
        return False


def _tok(token):
    return hashlib.sha256(token.encode()).hexdigest()


class Store:
    def __init__(self, path, url=None):
        self.path = str(path)
        self.url = url if url is not None else os.getenv("DATABASE_URL", "")
        self.pg = self.url.startswith(("postgres://", "postgresql://"))
        self.lock = threading.Lock()
        self.fails = {}                    # email -> [timestamps of failed logins]
        self.mails = {}                    # email -> [timestamps of verification / reset mails sent]
        if self.pg:
            import psycopg                 # optional dependency, only for DATABASE_URL=postgresql://...
            self.Integrity = (psycopg.errors.UniqueViolation,)
            with self._db() as c:
                for stmt in [x for x in PG_SCHEMA.split(";") if x.strip()]:
                    c.execute(stmt)
                c.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS verified DOUBLE PRECISION")
                c.commit()
        else:
            self.Integrity = (sqlite3.IntegrityError,)
            with self._db() as c:
                c.executescript(SCHEMA)
                cols = {r[1] for r in c.execute("PRAGMA table_info(users)")}
                if "verified" not in cols:      # databases made by earlier versions
                    c.execute("ALTER TABLE users ADD COLUMN verified REAL")
        # documents that were processing when the server stopped: the app queues them again (and resets this)
        self._q("UPDATE documents SET status='failed', error='Processing was interrupted (server restart).' WHERE status='processing'", write=True)

    def _db(self):
        if self.pg:
            import psycopg
            from psycopg.rows import dict_row
            return psycopg.connect(self.url, row_factory=dict_row)
        c = sqlite3.connect(self.path, timeout=10, check_same_thread=False)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA foreign_keys=ON")
        return c

    def _q(self, sql, args=(), one=False, write=False):
        with self.lock:
            c = self._db()
            try:
                cur = c.execute(pg_sql(sql) if self.pg else sql, args)
                rows = cur.fetchall() if cur.description else []
                if write:
                    c.commit()
                return (rows[0] if rows else None) if one else rows
            finally:
                c.close()

    # ---------------------------------------------------------------------------------------- users
    @staticmethod
    def public(u):
        if not u:
            return None
        v = u["verified"] if "verified" in u.keys() else None
        return dict(id=u["id"], email=u["email"], name=u["name"], created=u["created"], verified=bool(v))

    def signup(self, name, email, password):
        name, email = (name or "").strip()[:80], (email or "").strip().lower()
        if not name:
            raise AuthError("PF-401", "Please enter your name.")
        if not EMAIL.match(email):
            raise AuthError("PF-401", "Please enter a valid email address.")
        if len(password or "") < MIN_PASSWORD:
            raise AuthError("PF-401", f"Use a password of at least {MIN_PASSWORD} characters.")
        try:
            self._q("INSERT INTO users(email,name,pw,created) VALUES(?,?,?,?)", (email, name, hash_password(password), time.time()), write=True)
        except self.Integrity:
            raise AuthError("PF-409", "An account with this email already exists. Log in instead.", 409)
        return self._q("SELECT * FROM users WHERE email=?", (email,), one=True)

    def login(self, email, password):
        email = (email or "").strip().lower()
        now = time.time()
        recent = [t for t in self.fails.get(email, []) if now - t < 600]
        if len(recent) >= 8:
            raise AuthError("PF-429", "Too many failed attempts. Wait a few minutes and try again.", 429)
        u = self._q("SELECT * FROM users WHERE email=?", (email,), one=True)
        if not u or not check_password(password or "", u["pw"]):
            self.fails[email] = recent + [now]
            raise AuthError("PF-401", "Email or password is incorrect.", 401)
        self.fails.pop(email, None)
        return u

    def update_name(self, uid, name):
        name = (name or "").strip()[:80]
        if not name:
            raise AuthError("PF-401", "Please enter your name.")
        self._q("UPDATE users SET name=? WHERE id=?", (name, uid), write=True)
        return self._q("SELECT * FROM users WHERE id=?", (uid,), one=True)

    def change_password(self, uid, current, new, keep_token=None):
        u = self._q("SELECT * FROM users WHERE id=?", (uid,), one=True)
        if not u or not check_password(current or "", u["pw"]):
            raise AuthError("PF-401", "Your current password is incorrect.", 401)
        if len(new or "") < MIN_PASSWORD:
            raise AuthError("PF-401", f"Use a password of at least {MIN_PASSWORD} characters.")
        self._q("UPDATE users SET pw=? WHERE id=?", (hash_password(new), uid), write=True)
        # sign out every other session
        self._q("DELETE FROM sessions WHERE user_id=? AND token<>?", (uid, _tok(keep_token) if keep_token else ""), write=True)

    def delete_user(self, uid, password):
        u = self._q("SELECT * FROM users WHERE id=?", (uid,), one=True)
        if not u or not check_password(password or "", u["pw"]):
            raise AuthError("PF-401", "Your password is incorrect.", 401)
        ids = [r["id"] for r in self._q("SELECT id FROM documents WHERE user_id=?", (uid,))]
        self._q("DELETE FROM users WHERE id=?", (uid,), write=True)
        return ids

    # ---------------------------------------------------------------------------------------- email tokens
    def _rate_mail(self, email):
        now = time.time()
        recent = [t for t in self.mails.get(email, []) if now - t < 3600]
        if len(recent) >= 5:
            raise AuthError("PF-429", "Too many emails were sent to this address. Try again later.", 429)
        self.mails[email] = recent + [now]

    def new_token(self, uid, kind):
        """A single-use link token ('verify' or 'reset'); earlier tokens of the same kind stop working."""
        token = secrets.token_urlsafe(32)
        hours = VERIFY_HOURS if kind == "verify" else RESET_HOURS
        self._q("DELETE FROM tokens WHERE user_id=? AND kind=?", (uid, kind), write=True)
        self._q("INSERT INTO tokens(token,user_id,kind,expires) VALUES(?,?,?,?)", (_tok(token), uid, kind, time.time() + hours * 3600), write=True)
        return token

    def use_token(self, token, kind):
        """The user id for a valid token (which is then used up), else AuthError."""
        if not token:
            raise AuthError("PF-410", "This link is not valid.", 400)
        r = self._q("SELECT * FROM tokens WHERE token=? AND kind=?", (_tok(token), kind), one=True)
        if not r or r["expires"] < time.time():
            raise AuthError("PF-410", "This link has expired or was already used. Ask for a new one.", 400)
        self._q("DELETE FROM tokens WHERE token=?", (_tok(token),), write=True)
        return r["user_id"]

    def verification_token(self, uid):
        u = self._q("SELECT * FROM users WHERE id=?", (uid,), one=True)
        if not u:
            raise AuthError("PF-404", "No such account.", 404)
        self._rate_mail(u["email"])
        return u, self.new_token(uid, "verify")

    def verify(self, token):
        uid = self.use_token(token, "verify")
        self._q("UPDATE users SET verified=? WHERE id=?", (time.time(), uid), write=True)
        return self._q("SELECT * FROM users WHERE id=?", (uid,), one=True)

    def reset_token(self, email):
        """(user, token) for a reset email, or (None, None) when there is no such account (the caller answers the
        same either way, so the form does not reveal which emails have accounts)."""
        email = (email or "").strip().lower()
        u = self._q("SELECT * FROM users WHERE email=?", (email,), one=True)
        if not u:
            return None, None
        self._rate_mail(email)
        return u, self.new_token(u["id"], "reset")

    def reset_password(self, token, new):
        if len(new or "") < MIN_PASSWORD:
            raise AuthError("PF-401", f"Use a password of at least {MIN_PASSWORD} characters.")
        uid = self.use_token(token, "reset")
        # the link reached the inbox, so the address is proven as well
        self._q("UPDATE users SET pw=?, verified=COALESCE(verified, ?) WHERE id=?", (hash_password(new), time.time(), uid), write=True)
        self._q("DELETE FROM sessions WHERE user_id=?", (uid,), write=True)
        self.fails.clear()
        return self._q("SELECT * FROM users WHERE id=?", (uid,), one=True)

    # ---------------------------------------------------------------------------------------- sessions
    def new_session(self, uid):
        token = secrets.token_urlsafe(32)
        now = time.time()
        self._q("DELETE FROM sessions WHERE expires<?", (now,), write=True)
        self._q("INSERT INTO sessions(token,user_id,created,expires) VALUES(?,?,?,?)", (_tok(token), uid, now, now + SESSION_DAYS * 86400), write=True)
        return token

    def user_for(self, token):
        if not token:
            return None
        return self._q("SELECT u.* FROM sessions s JOIN users u ON u.id=s.user_id WHERE s.token=? AND s.expires>?", (_tok(token), time.time()), one=True)

    def end_session(self, token):
        if token:
            self._q("DELETE FROM sessions WHERE token=?", (_tok(token),), write=True)

    # ---------------------------------------------------------------------------------------- documents
    def add_document(self, doc_id, uid, filename, typ, size, pages):
        self._q("INSERT INTO documents(id,user_id,filename,type,size,pages,created,status) VALUES(?,?,?,?,?,?,?,?)",
                (doc_id, uid, filename, typ, size, pages, time.time(), "processing"), write=True)

    def finish_document(self, doc_id, status, stats=None, error=None):
        self._q("UPDATE documents SET status=?, stats=?, error=? WHERE id=?",
                (status, json.dumps(stats) if stats is not None else None, error, doc_id), write=True)

    def owner(self, doc_id):
        r = self._q("SELECT user_id FROM documents WHERE id=?", (doc_id,), one=True)
        return r["user_id"] if r else None

    def owned_ids(self):
        return {r["id"] for r in self._q("SELECT id FROM documents")}

    @staticmethod
    def doc_public(r):
        return dict(id=r["id"], filename=r["filename"], type=r["type"], size=r["size"], pages=r["pages"], created=r["created"],
                    status=r["status"], stats=json.loads(r["stats"]) if r["stats"] else None, error=r["error"])

    def documents(self, uid):
        return [self.doc_public(r) for r in self._q("SELECT * FROM documents WHERE user_id=? ORDER BY created DESC", (uid,))]

    def document(self, uid, doc_id):
        r = self._q("SELECT * FROM documents WHERE id=? AND user_id=?", (doc_id, uid), one=True)
        return self.doc_public(r) if r else None

    def rename(self, uid, doc_id, filename):
        filename = re.sub(r"[\x00-\x1f]", "", (filename or "").strip())[:160]
        if not filename:
            raise AuthError("PF-400", "The name cannot be empty.")
        self._q("UPDATE documents SET filename=? WHERE id=? AND user_id=?", (filename, doc_id, uid), write=True)

    def delete_document(self, uid, doc_id):
        self._q("DELETE FROM documents WHERE id=? AND user_id=?", (doc_id, uid), write=True)


def doc_stats(doc):
    """Library summary of a processed document (what the 'My documents' list and filters show)."""
    bl = doc.get("blocks", [])
    pages = doc.get("pages", [])
    flagged = sum(1 for b in bl if b.get("verification") != "verified")
    return dict(pages=len(pages), blocks=len(bl), tables=sum(b.get("type") == "table" for b in bl),
                equations=sum(b.get("type") == "equation" for b in bl) + sum(1 for b in bl if b.get("inline_math")),
                figures=sum(b.get("type") in ("figure", "chart", "diagram") for b in bl),
                handwriting=sum(str(b.get("type", "")).startswith("handwritten") for b in bl),
                scanned=sum(1 for p in pages if p.get("route") in ("ocr", "embedded_ocr")),
                verified=len(bl) - flagged, flagged=flagged,
                furniture=sum(1 for b in bl if (b.get("furniture") or {}).get("excluded")),
                seconds=(doc.get("timings") or {}).get("total"), warnings=len(doc.get("warnings") or []))
