"""Vercel serverless entrypoint for ParseFlow FastAPI API routes."""
import json
import sys
import traceback
from pathlib import Path

_APP = None
_ERROR = None

try:
    ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "backend"))
    from backend.app import app as backend_app
    _APP = backend_app
except Exception:
    _ERROR = traceback.format_exc()


async def app(scope, receive, send):
    if _APP is not None and not _ERROR:
        await _APP(scope, receive, send)
        return

    tb_lines = _ERROR.splitlines() if _ERROR else ["Unknown error"]
    error_msg = tb_lines[-1] if tb_lines else "Serverless Startup Error"
    payload = json.dumps({
        "ok": False,
        "error": "Serverless Startup Error",
        "details": error_msg,
        "traceback": tb_lines,
    })
    body = payload.encode("utf-8")
    await send({
        "type": "http.response.start",
        "status": 500,
        "headers": [
            [b"content-type", b"application/json; charset=utf-8"],
            [b"content-length", str(len(body)).encode("ascii")],
        ],
    })
    await send({
        "type": "http.response.body",
        "body": body,
    })
