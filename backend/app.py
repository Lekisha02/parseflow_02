import os, re, sys, json, time, uuid, shutil, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Query, Request, Body
from fastapi.responses import JSONResponse, FileResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
sys.path.insert(0, str(Path(__file__).parent))
ROOT = Path(__file__).resolve().parent.parent
def load_env(path, env=os.environ):
    """KEY=value lines (blank lines, '# comments' and 'KEY=value  # comment' allowed); existing variables win."""
    try: lines = Path(path).read_text().splitlines()
    except FileNotFoundError: return
    for line in lines:
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            v = re.sub(r"\s+#.*$", "", v).strip()
            if v: env.setdefault(k.strip(), v)
load_env(ROOT/".env")
from processing import pipeline, ocr, vision, recognizers, runner
from processing import markdown as mdgen
from processing import ask as askmod
from processing import assistant, block_ask, docfiles, engines, formats, llm
import accounts, mailer

VERSION = "2.6.0"
def _init_data():
    p = Path(os.getenv("PARSEFLOW_DATA_DIR", "/tmp/parseflow_data" if os.getenv("VERCEL") else str(ROOT/"data")))
    try:
        p.mkdir(parents=True, exist_ok=True)
        return p
    except OSError:
        p = Path("/tmp/parseflow_data")
        p.mkdir(parents=True, exist_ok=True)
        return p
DATA = _init_data()
MAX_BYTES = int(float(os.getenv("MAX_UPLOAD_MB", "0") or 0)*1024*1024)        # 0 = no limit
ISOLATE = os.getenv("PARSEFLOW_ISOLATE", "1").lower() not in ("0", "false", "no", "off")   # one worker process per document
JOB_TIMEOUT = float(os.getenv("PARSEFLOW_JOB_TIMEOUT", "0") or 0)                       # seconds; 0 = no limit
JOB_TTL = float(os.getenv("PARSEFLOW_JOB_TTL_HOURS", "24"))*3600
ALLOWED = set(formats.SUPPORTED); DIRECT = {"pdf", "png", "jpg", "jpeg"}
JOBS = {}; POOL = ThreadPoolExecutor(int(os.getenv("PARSEFLOW_WORKERS", "2")))
app = FastAPI(title="ParseFlow", version=VERSION)
STORE = accounts.Store(DATA/"parseflow.db")
SECURE_COOKIE = os.getenv("PARSEFLOW_SECURE_COOKIE", "0").lower() in ("1", "true", "yes")
# X-Forwarded-Host/-Proto are used to build links only behind a proxy you trust (they can be forged otherwise);
# set PARSEFLOW_PUBLIC_URL in production so emailed links never depend on request headers
TRUST_PROXY = os.getenv("PARSEFLOW_TRUST_PROXY", "0").lower() in ("1", "true", "yes")
REQUIRE_VERIFIED = os.getenv("PARSEFLOW_REQUIRE_VERIFIED", "0").lower() in ("1", "true", "yes")
GUEST_COOKIE = "pf_guest"

def err(code, title, msg, status=400): return JSONResponse(dict(error=dict(code=code, title=title, message=msg)), status_code=status)
def _safe_id(doc_id): return bool(re.fullmatch(r"[a-f0-9]{12}", doc_id or ""))

def _cleanup_old():
    """Delete guest documents older than PARSEFLOW_JOB_TTL_HOURS. Documents saved to an account are kept until
    their owner deletes them."""
    if JOB_TTL <= 0: return
    now = time.time(); owned = STORE.owned_ids()
    for d in DATA.iterdir():
        try:
            if d.is_dir() and d.name not in owned and now - d.stat().st_mtime > JOB_TTL:
                shutil.rmtree(d, True); JOBS.pop(d.name, None)
        except OSError: pass

_LLM_TEST = dict(at=0.0)


@app.post("/api/llm/test")
def llm_test():
    """A real one-line request to the configured AI model (and an image check when it takes images)."""
    now = time.time()
    if now - _LLM_TEST["at"] < 5:
        return err("PF-429", "Too Many Requests", "Wait a few seconds before testing again.", 429)
    _LLM_TEST["at"] = now
    return llm.selftest()


@app.get("/api/health")
def health():
    try:
        import pymupdf; pm = pymupdf.VersionBind
    except Exception: pm = None
    return dict(ok=True, version=VERSION, email=mailer.mode(), require_verified=REQUIRE_VERIFIED, ocr=ocr.available(), vision=vision.info(), ask_llm=llm.info(), local_models=engines.info(), math_recognizer=recognizers.info(), pymupdf=pm,
                max_upload_mb=MAX_BYTES//(1024*1024), max_pages=int(os.getenv("MAX_PAGES", "0") or 0),
                formats=dict(supported=sorted(formats.SUPPORTED), libreoffice=bool(formats.soffice()),
                             need_libreoffice=sorted((formats.WORD | formats.SLIDES | {"xls", "ods"}) - formats.NATIVE_OK)))

# ------------------------------------------------------------------------------------------- accounts
def _user(request):
    return STORE.user_for(request.cookies.get(accounts.COOKIE))

def _access(doc_id):
    try: return json.loads((DATA/doc_id/"access.json").read_text())
    except Exception: return {}

def _hash(t): return accounts._tok(t) if t else ""

def _is_owner(request, doc_id):
    """The account that saved the document, or the browser that uploaded it as a guest."""
    if not _safe_id(doc_id): return False
    owner = STORE.owner(doc_id)
    if owner is not None:
        u = _user(request); return bool(u and u["id"] == owner)
    a = _access(doc_id)
    if not a.get("guest"): return True                     # uploaded before guest documents were private
    return _hash(request.cookies.get(GUEST_COOKIE)) == a["guest"]

def _guard(request, doc_id):
    """A document is visible to its owner (account, or the guest browser that uploaded it) and, once its owner
    turns sharing on, to anyone with the link (read-only)."""
    if not _safe_id(doc_id): return err("PF-404", "Not Found", "Document not available.", 404)
    if _is_owner(request, doc_id) or _access(doc_id).get("shared"): return None
    return err("PF-404", "Not Found", "Document not available.", 404)

def _secure(request):
    """Mark cookies Secure on HTTPS, including HTTPS ended at a reverse proxy (a forged header only makes the cookie
    stricter)."""
    if SECURE_COOKIE or request.url.scheme == "https": return True
    return request.headers.get("x-forwarded-proto", "").split(",")[0].strip().lower() == "https"

def _base_url(request):
    b = os.getenv("PARSEFLOW_PUBLIC_URL", "").rstrip("/")
    if b: return b
    if TRUST_PROXY:
        proto = request.headers.get("x-forwarded-proto", "").split(",")[0].strip() or request.url.scheme
        host = request.headers.get("x-forwarded-host", "").split(",")[0].strip() or request.url.netloc
        return f"{proto}://{host}"
    return f"{request.url.scheme}://{request.url.netloc}"

def _session_response(user, token, request=None, **extra):
    r = JSONResponse(dict(user=STORE.public(user), **extra))
    r.set_cookie(accounts.COOKIE, token, max_age=int(accounts.SESSION_DAYS * 86400), httponly=True, samesite="lax",
                 secure=_secure(request) if request is not None else SECURE_COOKIE, path="/")
    return r

def _send_verification(request, u):
    """Best effort: a failed email never fails sign-up (the user can ask again from Settings)."""
    try:
        u, token = STORE.verification_token(u["id"])
        link = f"{_base_url(request)}/#verify={token}"
        mailer.send(u["email"], "Confirm your ParseFlow email address",
                    f"Hello {u['name']},\n\nConfirm your email address for ParseFlow by opening this link (valid for 24 hours):\n\n{link}\n\n"
                    "If you did not create this account, ignore this email.\n", DATA/"outbox")
        return True
    except Exception:
        return False

def _auth_err(e): return err(e.code, "Account", e.message, e.status)

def _need_user(request):
    u = _user(request)
    return (u, None) if u else (None, err("PF-401", "Not Logged In", "Log in to use your library.", 401))

def _remove_doc_files(doc_id):
    if _safe_id(doc_id): shutil.rmtree(DATA/doc_id, True); JOBS.pop(doc_id, None)

@app.post("/api/auth/signup")
def signup(request: Request, payload: dict = Body(...)):
    try: u = STORE.signup(payload.get("name"), payload.get("email"), payload.get("password"))
    except accounts.AuthError as e: return _auth_err(e)
    sent = _send_verification(request, u)
    if REQUIRE_VERIFIED:
        return JSONResponse(dict(user=None, needs_verification=True, email=u["email"], verification_sent=sent))
    return _session_response(u, STORE.new_session(u["id"]), request, verification_sent=sent)

@app.post("/api/auth/login")
def login(request: Request, payload: dict = Body(...)):
    try: u = STORE.login(payload.get("email"), payload.get("password"))
    except accounts.AuthError as e: return _auth_err(e)
    if REQUIRE_VERIFIED and not u["verified"]:
        return err("PF-403", "Account", "Confirm your email address first: open the link we sent you (or ask for a new one).", 403)
    return _session_response(u, STORE.new_session(u["id"]), request)

@app.post("/api/auth/verify")
def verify_email(request: Request, payload: dict = Body(...)):
    """Open the emailed link: the address is confirmed and the browser is logged in."""
    try: u = STORE.verify(str(payload.get("token") or ""))
    except accounts.AuthError as e: return _auth_err(e)
    return _session_response(u, STORE.new_session(u["id"]), request)

@app.post("/api/auth/verify/resend")
def verify_resend(request: Request, payload: dict = Body(default={})):
    u = _user(request)
    if not u and payload.get("email"):
        u = STORE._q("SELECT * FROM users WHERE email=?", (str(payload["email"]).strip().lower(),), one=True)
    if u and not u["verified"]:
        try: STORE._rate_mail(u["email"] + ":resend")
        except accounts.AuthError as e: return _auth_err(e)
        _send_verification(request, u)
    return dict(ok=True)                     # same answer whether or not the address has an account

@app.post("/api/auth/reset/request")
def reset_request(request: Request, payload: dict = Body(...)):
    """Email a password-reset link (valid 1 hour). The answer never reveals whether the address has an account."""
    try: u, token = STORE.reset_token(payload.get("email"))
    except accounts.AuthError as e: return _auth_err(e)
    if u:
        try:
            mailer.send(u["email"], "Reset your ParseFlow password",
                        f"Hello {u['name']},\n\nSomeone (hopefully you) asked to reset the password of this ParseFlow account. "
                        f"Open this link within one hour to choose a new password:\n\n{_base_url(request)}/#reset={token}\n\n"
                        "If you did not ask for this, ignore this email; your password stays the same.\n", DATA/"outbox")
        except Exception:
            pass
    return dict(ok=True)

@app.post("/api/auth/reset")
def reset_password(request: Request, payload: dict = Body(...)):
    """Choose a new password with the emailed token. Every other session is signed out; this browser is logged in."""
    try: u = STORE.reset_password(str(payload.get("token") or ""), payload.get("password"))
    except accounts.AuthError as e: return _auth_err(e)
    return _session_response(u, STORE.new_session(u["id"]), request)

@app.post("/api/auth/logout")
def logout(request: Request):
    STORE.end_session(request.cookies.get(accounts.COOKIE))
    r = JSONResponse(dict(ok=True)); r.delete_cookie(accounts.COOKIE, path="/"); return r

@app.get("/api/auth/me")
def me(request: Request):
    return dict(user=STORE.public(_user(request)))

@app.patch("/api/account")
def account_update(request: Request, payload: dict = Body(...)):
    u, e = _need_user(request)
    if e: return e
    try: return dict(user=STORE.public(STORE.update_name(u["id"], payload.get("name"))))
    except accounts.AuthError as x: return _auth_err(x)

@app.post("/api/account/password")
def account_password(request: Request, payload: dict = Body(...)):
    u, e = _need_user(request)
    if e: return e
    try: STORE.change_password(u["id"], payload.get("current"), payload.get("new"), keep_token=request.cookies.get(accounts.COOKIE))
    except accounts.AuthError as x: return _auth_err(x)
    return dict(ok=True)

@app.post("/api/account/delete")
def account_delete(request: Request, payload: dict = Body(...)):
    u, e = _need_user(request)
    if e: return e
    try: ids = STORE.delete_user(u["id"], payload.get("password"))
    except accounts.AuthError as x: return _auth_err(x)
    for i in ids: _remove_doc_files(i)
    r = JSONResponse(dict(ok=True, deleted_documents=len(ids))); r.delete_cookie(accounts.COOKIE, path="/"); return r

@app.get("/api/my/documents")
def my_documents(request: Request):
    u, e = _need_user(request)
    if e: return e
    docs = STORE.documents(u["id"])
    for d in docs:
        j = JOBS.get(d["id"])
        if d["status"] == "processing" and j:
            run = [s for s in j["prog"].stages if s["state"] == "running"]
            if j["prog"].status == "queued":
                ahead = sum(1 for k, o in list(JOBS.items()) if k != d["id"] and o["prog"].status == "queued" and o["started"] < j["started"])
                d["progress"] = "Queued" + (f" · {ahead} ahead" if ahead else " · next")
            else:
                d["progress"] = (run[-1]["name"].title() + (" · " + run[-1]["detail"] if run[-1]["detail"] else "")) if run else None
    return dict(documents=docs, usage=dict(documents=len(docs), pages=sum(d["pages"] or 0 for d in docs), bytes=sum(d["size"] or 0 for d in docs)))

@app.patch("/api/my/documents/{doc_id}")
def my_document_rename(doc_id: str, request: Request, payload: dict = Body(...)):
    u, e = _need_user(request)
    if e: return e
    if not STORE.document(u["id"], doc_id): return err("PF-404", "Not Found", "Document not available.", 404)
    try: STORE.rename(u["id"], doc_id, payload.get("filename"))
    except accounts.AuthError as x: return _auth_err(x)
    return dict(document=STORE.document(u["id"], doc_id))

@app.delete("/api/my/documents/{doc_id}")
def my_document_delete(doc_id: str, request: Request):
    u, e = _need_user(request)
    if e: return e
    if not STORE.document(u["id"], doc_id): return err("PF-404", "Not Found", "Document not available.", 404)
    if JOBS.get(doc_id) and JOBS[doc_id]["prog"].status == "running":
        return err("PF-409", "Still Processing", "Wait until processing finishes, then delete the document.", 409)
    STORE.delete_document(u["id"], doc_id); _remove_doc_files(doc_id)
    return dict(ok=True)

# ------------------------------------------------------------------------------------------- documents
@app.post("/api/process")
async def process(request: Request, file: UploadFile = File(...)):
    _cleanup_old()
    user = _user(request)
    name = os.path.basename(file.filename or "upload")
    ext = name.lower().rsplit(".", 1)[-1] if "." in name else ""
    if ext not in ALLOWED:
        return err("PF-204", "Unsupported Document", "ParseFlow could not process this file type. Supported: " + formats.LABEL + ".")
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name)[-120:] or "upload"
    doc_id = uuid.uuid4().hex[:12]; d = DATA/doc_id; d.mkdir()
    # PDFs and images are read as they are; other formats are kept as the original and converted next to it
    src = d/(("source_" if ext in DIRECT else "original_")+doc_id+"_"+safe)
    size = 0
    try:
        with open(src, "wb") as f:
            while chunk := await file.read(1024*1024):
                size += len(chunk)
                if MAX_BYTES and size > MAX_BYTES:
                    f.close(); shutil.rmtree(d, True)
                    return err("PF-206", "File Too Large", f"Maximum size is {MAX_BYTES//(1024*1024)} MB.", 413)
                f.write(chunk)
    except Exception:
        shutil.rmtree(d, True); return err("PF-500", "Upload Failed", "The upload could not be stored.", 500)
    if ext not in DIRECT:
        try:
            src = Path(formats.prepare(str(src), name, str(d), doc_id).path)
        except formats.ConvertError as e:
            shutil.rmtree(d, True); return JSONResponse(dict(error=e.dict()), status_code=400)
        except Exception:
            shutil.rmtree(d, True); return err("PF-208", "Conversion Failed", "The file could not be converted for reading.")
    try: pages = pipeline.page_count(str(src), name)
    except pipeline.PFError as e: shutil.rmtree(d, True); return JSONResponse(dict(error=e.dict()), status_code=400)
    except Exception: shutil.rmtree(d, True); return err("PF-201", "Corrupt Document", "The file could not be read.")
    gtoken = None
    if user: STORE.add_document(doc_id, user["id"], name, ext.upper(), size, pages)
    else:
        # a guest document belongs to the browser that uploaded it (a private cookie) until its owner shares it
        gtoken = request.cookies.get(GUEST_COOKIE) or accounts.secrets.token_urlsafe(32)
        (d/"access.json").write_text(json.dumps(dict(guest=_hash(gtoken), shared=os.getenv("PARSEFLOW_GUEST_LINKS", "private") == "open")))
    _submit(doc_id, src, name, user["id"] if user else None)
    r = JSONResponse(dict(document_id=doc_id, filename=name, size=size, type=ext.upper(), pages=pages, saved=bool(user)))
    if gtoken and request.cookies.get(GUEST_COOKIE) != gtoken:
        r.set_cookie(GUEST_COOKIE, gtoken, max_age=365 * 86400, httponly=True, samesite="lax", secure=_secure(request), path="/")
    return r

def _submit(doc_id, src, name, uid=None):
    """Queue a document for processing. PARSEFLOW_WORKERS documents run at once; the rest wait their turn (status
    'queued' with their place in line)."""
    d = DATA/doc_id
    prog = pipeline.Progress(); prog.status = "queued"
    JOBS[doc_id] = dict(prog=prog, doc=None, started=time.time())
    user = dict(id=uid) if uid else None
    def run():
        prog.status = "running"
        try:
            if ISOLATE:
                r = runner.run_isolated(str(src), name, doc_id, str(d), prog, timeout=JOB_TIMEOUT or None)
                prog.stages = r.get("stages") or prog.stages
                if r.get("status") == "complete":
                    prog.status = "complete"
                    if user: STORE.finish_document(doc_id, "complete", r.get("stats"))
                else:
                    prog.status, prog.error = "failed", r.get("error")
                    if user: STORE.finish_document(doc_id, "failed", error=(r.get("error") or {}).get("message"))
                return
            doc = pipeline.process(str(src), name, doc_id, str(d), prog, save_images="needed")
            assistant.prepare(doc, str(d))
            try: docfiles.write_variants(str(d), doc)
            except Exception: pass
            prog.status = "complete"
            if user: STORE.finish_document(doc_id, "complete", accounts.doc_stats(doc))
        except pipeline.PFError as e:
            prog.status, prog.error = "failed", e.dict()
            if user: STORE.finish_document(doc_id, "failed", error=e.message)
        except Exception as e:
            prog.status, prog.error = "failed", dict(code="PF-500", title="Processing Error", message=f"Unexpected failure ({type(e).__name__}); the server is still running.")
            if user: STORE.finish_document(doc_id, "failed", error=prog.error["message"])
        finally:
            if prog.status == "failed":         # a real failure is final: not retried after a restart
                try: (d/"failed.json").write_text(json.dumps(prog.error or {}))
                except OSError: pass
    POOL.submit(run)

def _resume_interrupted():
    """After a restart: documents whose processing was cut off (an upload with no document.json yet) are queued
    again; saved slices (checkpoint/) mean they continue from where they stopped instead of page 1."""
    if os.getenv("PARSEFLOW_RESUME", "1").lower() in ("0", "false", "no", "off"): return []
    resumed = []
    for d in sorted(DATA.iterdir(), key=lambda p: p.stat().st_mtime if p.exists() else 0):
        try:
            if not (d.is_dir() and _safe_id(d.name)) or (d/"document.json").exists() or (d/"failed.json").exists() or d.name in JOBS: continue
            src = formats.source_file(d)
            if src is None: continue
            src = Path(src)
            name = formats.original_name(d) or src.name.split("_", 2)[-1]
            uid = STORE.owner(d.name)
            if uid: STORE.finish_document(d.name, "processing")
            _submit(d.name, src, name, uid); resumed.append(d.name)
        except Exception:
            continue
    return resumed

RESUMED = _resume_interrupted()

@app.get("/api/status/{doc_id}")
def status(doc_id: str, request: Request):
    if (g := _guard(request, doc_id)): return g
    j = JOBS.get(doc_id)
    if not j:
        # not started by this server process (a restart, or another server sharing the data folder): read the disk
        d = DATA/doc_id
        if (d/"document.json").exists(): return dict(status="complete", stages=[dict(name=n, state="done", detail="") for n in pipeline.STAGES], error=None, timings={})
        try:
            pj = json.loads((d/"progress.json").read_text())
            if (d/"failed.json").exists() and pj.get("status") != "failed": pj["status"] = "failed"
            return dict(status=pj.get("status", "running"), stages=pj.get("stages") or [], error=pj.get("error"), timings=pj.get("timings") or {})
        except Exception: pass
        if (d/"failed.json").exists():
            return dict(status="failed", stages=[], error=json.loads((d/"failed.json").read_text() or "{}") or None, timings={})
        return err("PF-404", "Unknown Document", "No such processing job.", 404)
    p = j["prog"]
    out = dict(status=p.status, stages=p.stages, error=p.error, timings=getattr(p, "timings", {}))
    if p.status == "queued":
        out["queue_position"] = 1 + sum(1 for k, o in list(JOBS.items()) if k != doc_id and o["prog"].status == "queued" and o["started"] < j["started"])
    return out

@app.get("/api/document/{doc_id}")
def document(doc_id: str, request: Request, lite: str = Query("0")):
    """The Digital Twin JSON. lite=1 (or lite=auto for a large document): every block without the heavy provenance fields and the Markdown (what the
    viewer needs to open a very large document; /blocks gives the rest per page). Sent gzip-compressed when the
    client accepts it."""
    if (g := _guard(request, doc_id)): return g
    p = DATA/doc_id/"document.json"
    if not _safe_id(doc_id) or not p.exists(): return err("PF-404", "Not Ready", "Document not available.", 404)
    # lite=auto: the light copy only when the full JSON is large (LITE_MB, default 8 MB)
    lite = lite in ("1", "true", "yes") or (lite == "auto" and p.stat().st_size > float(os.getenv("PARSEFLOW_LITE_MB", "8")) * 1e6)
    try: docfiles.ensure(str(DATA/doc_id))
    except Exception:
        if lite: return err("PF-500", "Server Error", "The document could not be prepared for viewing.", 500)
        return FileResponse(p, media_type="application/json")
    gz = DATA/doc_id/("document.lite.json.gz" if lite else "document.json.gz")
    if "gzip" in request.headers.get("accept-encoding", "").lower():
        return FileResponse(gz, media_type="application/json", headers={"Content-Encoding": "gzip", "Vary": "Accept-Encoding"})
    if not lite: return FileResponse(p, media_type="application/json")
    import gzip as _gz
    return Response(_gz.decompress(gz.read_bytes()), media_type="application/json")

@app.get("/api/document/{doc_id}/access")
def document_access(doc_id: str, request: Request):
    if (g := _guard(request, doc_id)): return g
    return dict(owner=_is_owner(request, doc_id), shared=bool(_access(doc_id).get("shared")), saved=STORE.owner(doc_id) is not None)

@app.post("/api/document/{doc_id}/share")
def document_share(doc_id: str, request: Request, payload: dict = Body(...)):
    """Owner only: shared=true lets anyone with the link open the document (read-only); false makes it private."""
    if not _safe_id(doc_id) or not (DATA/doc_id).exists() or not _is_owner(request, doc_id):
        return err("PF-404", "Not Found", "Document not available.", 404)
    a = _access(doc_id); a["shared"] = bool(payload.get("shared"))
    tmp = DATA/doc_id/"access.json.tmp"; tmp.write_text(json.dumps(a)); os.replace(tmp, DATA/doc_id/"access.json")
    return dict(shared=a["shared"], link=f"{_base_url(request)}/#doc={doc_id}")

@app.get("/api/document/{doc_id}/blocks")
def document_blocks(doc_id: str, request: Request, page: int = Query(0), pages: str = Query("")):
    """Full blocks (with evidence_detail, raw_text, transforms) of one page or a range 'a-b' (at most 50 pages)."""
    if (g := _guard(request, doc_id)): return g
    if not _safe_id(doc_id) or not (DATA/doc_id/"document.json").exists(): return err("PF-404", "Not Ready", "Document not available.", 404)
    want = [page] if page else []
    if (m := re.fullmatch(r"(\d{1,5})(?:-(\d{1,5}))?", pages or "")):
        a = int(m.group(1)); b = int(m.group(2) or a)
        want = list(range(min(a, b), min(max(a, b), min(a, b) + 49) + 1))
    if not want: return err("PF-400", "Bad Request", "Give page=N or pages=a-b.")
    return dict(pages=want, blocks=docfiles.page_blocks(str(DATA/doc_id), want))

@app.get("/api/document/{doc_id}/markdown")
def document_markdown(doc_id: str, request: Request, include_furniture: bool = Query(False), annotate: bool = Query(True)):
    """Canonical backend Markdown. include_furniture=1 restores running headers/footers; annotate=0 drops HTML-comment warnings."""
    if (g := _guard(request, doc_id)): return g
    p = DATA/doc_id/"document.json"
    if not _safe_id(doc_id) or not p.exists(): return err("PF-404", "Not Ready", "Document not available.", 404)
    if not include_furniture and annotate:
        try:
            docfiles.ensure(str(DATA/doc_id))
            return FileResponse(DATA/doc_id/"document.md", media_type="text/markdown; charset=utf-8")
        except Exception: pass
    doc = json.loads(p.read_text())
    return PlainTextResponse(mdgen.render(doc, include_furniture=include_furniture, annotate=annotate), media_type="text/markdown; charset=utf-8")

@app.get("/api/ask/{doc_id}")
def ask_document(doc_id: str, request: Request, q: str = Query("", max_length=500)):
    """Extractive answer to a question about a processed document: parts copied from blocks, with evidence."""
    if (g := _guard(request, doc_id)): return g
    p = DATA/doc_id/"document.json"
    if not _safe_id(doc_id) or not p.exists(): return err("PF-404", "Not Ready", "Document not available.", 404)
    key = (doc_id, p.stat().st_mtime)
    doc = None if key in askmod._CACHE else json.loads(p.read_text())
    return askmod.answer(doc, q, key=key)

@app.post("/api/ask/{doc_id}")
def ask_assistant(doc_id: str, request: Request, payload: dict = Body(...)):
    """Ask assistant: {question, history:[{role, content, chunk_ids?}]} -> answer (LLM when configured, else
    passages quoted from the document) with page citations."""
    if (g := _guard(request, doc_id)): return g
    p = DATA/doc_id/"document.json"
    if not p.exists(): return err("PF-404", "Not Ready", "Document not available.", 404)
    mt = p.stat().st_mtime
    key = (doc_id, mt)
    doc = None if key in assistant._CACHE else json.loads(p.read_text())
    hist = payload.get("history") if isinstance(payload.get("history"), list) else []
    q = str(payload.get("question") or "")[:2000]
    r = assistant.respond(doc, q, history=hist[-12:], key=key, folder=str(DATA/doc_id), doc_mtime=mt)
    if q.strip() and payload.get("save", True) and _is_owner(request, doc_id):
        _chat_append(doc_id, [dict(role="user", text=q, at=time.time()), dict(role="assistant", s=r, at=time.time())])
    return r

@app.post("/api/ask/{doc_id}/block/{block_id}")
def ask_block(doc_id: str, block_id: str, request: Request, payload: dict = Body(...)):
    """Ask about one block: {question, history} -> answer grounded in the block, its image, caption, surrounding text,
    extracted structure and related pages; extraction confidence is reported separately from the answer."""
    if (g := _guard(request, doc_id)): return g
    p = DATA/doc_id/"document.json"
    if not p.exists(): return err("PF-404", "Not Ready", "Document not available.", 404)
    if not re.fullmatch(r"[A-Za-z0-9_]{1,40}", block_id or ""): return err("PF-404", "Not Found", "No such block.", 404)
    mt = p.stat().st_mtime
    key = (doc_id, mt)
    doc = None if key in assistant._CACHE else json.loads(p.read_text())
    hist = payload.get("history") if isinstance(payload.get("history"), list) else []
    q = str(payload.get("question") or "")[:2000]
    r = block_ask.respond(doc, block_id, q, history=hist[-12:], key=key, folder=str(DATA/doc_id), doc_mtime=mt)
    if r.get("error"): return err(r["error"]["code"], r["error"]["title"], r["error"]["message"], 404)
    if q.strip() and payload.get("save", True) and _is_owner(request, doc_id):
        _chat_append(doc_id, [dict(role="user", text=q, block=block_id, at=time.time()), dict(role="assistant", s=r, block=block_id, at=time.time())])
    return r

CHAT_LOCK = threading.Lock()

def _chat_path(doc_id):
    return DATA/doc_id/"chat.json"

def _chat_load(doc_id):
    try: return json.loads(_chat_path(doc_id).read_text()).get("messages", [])
    except Exception: return []

def _chat_append(doc_id, msgs):
    """The conversation is kept with the document (same access rules), so it survives a reload or another device."""
    with CHAT_LOCK:
        m = (_chat_load(doc_id) + msgs)[-200:]
        tmp = _chat_path(doc_id).with_suffix(".tmp")
        tmp.write_text(json.dumps(dict(messages=m))); os.replace(tmp, _chat_path(doc_id))

@app.get("/api/chat/{doc_id}")
def chat_history(doc_id: str, request: Request):
    """The saved Ask conversation for a document: [{role:'user', text} | {role:'assistant', s:<answer>}]."""
    if (g := _guard(request, doc_id)): return g
    if not _safe_id(doc_id) or not (DATA/doc_id).exists(): return err("PF-404", "Not Found", "Document not available.", 404)
    block = request.query_params.get("block")
    msgs = _chat_load(doc_id) if _is_owner(request, doc_id) else []
    return dict(messages=[m for m in msgs if (m.get("block") == block if block else not m.get("block"))])

@app.delete("/api/chat/{doc_id}")
def chat_clear(doc_id: str, request: Request):
    if (g := _guard(request, doc_id)): return g
    if not _safe_id(doc_id) or not (DATA/doc_id).exists() or not _is_owner(request, doc_id): return err("PF-404", "Not Found", "Document not available.", 404)
    block = request.query_params.get("block")
    with CHAT_LOCK:
        keep = [m for m in _chat_load(doc_id) if ((m.get("block") != block) if block else bool(m.get("block")))]
        tmp = _chat_path(doc_id).with_suffix(".tmp"); tmp.write_text(json.dumps(dict(messages=keep))); os.replace(tmp, _chat_path(doc_id))
    return dict(ok=True)

RENDER_LOCK = threading.Lock()
THUMB_W = 180

def _source(doc_id):
    d = DATA/doc_id
    src = formats.source_file(d) if d.is_dir() else None
    return d, (Path(src) if src else None)

def _page_image(doc_id, n, thumb=False):
    """Page bitmaps are rendered on first request and cached next to the document (a 400-page upload is not
    rasterised up front). Full pages are PNG at the viewer scale; thumbnails are small JPEGs."""
    d, src = _source(doc_id)
    out = d/(f"thumb_{n}.jpg" if thumb else f"page_{n}.png")
    if out.exists(): return out
    full = d/f"page_{n}.png"
    if src is None or not (d/"document.json").exists(): return None
    with RENDER_LOCK:
        if out.exists(): return out
        from PIL import Image
        name = src.name.split("_", 2)[-1]
        if thumb and full.exists():
            im = Image.open(full); im.load()
        else:
            kind, s = pipeline.open_source(str(src), name)
            try:
                if kind == "pdf":
                    if not 1 <= n <= s.page_count: return None
                    w = s[n - 1].rect.width or 612
                    im = pipeline.render_page(s, n, zoom=(THUMB_W * 2 / w) if thumb else None)
                else:
                    if n != 1: return None
                    im = s
            finally:
                if kind == "pdf": s.close()
        tmp = out.with_suffix(".tmp")
        if thumb:
            im = im.convert("RGB"); im.thumbnail((THUMB_W * 2, THUMB_W * 4)); im.save(tmp, "JPEG", quality=72)
        else:
            im.save(tmp, "PNG", compress_level=1)
        os.replace(tmp, out)
    return out

@app.get("/api/page/{doc_id}/{n}.png")
def page_png(doc_id: str, n: int, request: Request):
    if (g := _guard(request, doc_id)): return g
    p = _page_image(doc_id, n)
    if p is None: return err("PF-404", "Not Found", "Page image not found.", 404)
    return FileResponse(p, media_type="image/png", headers={"Cache-Control": "private, max-age=86400"})

@app.get("/api/thumb/{doc_id}/{n}.jpg")
def page_thumb(doc_id: str, n: int, request: Request):
    if (g := _guard(request, doc_id)): return g
    p = _page_image(doc_id, n, thumb=True)
    if p is None: return err("PF-404", "Not Found", "Page image not found.", 404)
    return FileResponse(p, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=86400"})

def _mount_ui():
    candidates = [
        ROOT / "frontend",
        ROOT / "public",
        Path(__file__).parent / "frontend",
        Path(__file__).parent.parent / "public",
        Path("frontend"),
        Path("public"),
    ]
    for c in candidates:
        if c.is_dir() and (c / "index.html").is_file():
            try:
                app.mount("/", StaticFiles(directory=str(c), html=True), name="ui")
                return
            except Exception:
                pass
    for c in candidates:
        if c.is_dir():
            try:
                app.mount("/", StaticFiles(directory=str(c), html=True), name="ui")
                return
            except Exception:
                pass
    from fastapi.responses import HTMLResponse
    @app.get("/")
    def _fallback_ui():
        return HTMLResponse("<!DOCTYPE html><html><head><title>ParseFlow</title></head><body><h1>ParseFlow API is running</h1><p><a href='/api/health'>Check /api/health</a></p></body></html>")

_mount_ui()
