"""Outgoing email for account verification and password reset.

    SMTP_HOST, SMTP_PORT (587), SMTP_USER, SMTP_PASSWORD, SMTP_FROM, SMTP_SECURITY (starttls | ssl | none)
    PARSEFLOW_PUBLIC_URL    the address users open ParseFlow at (links in emails); default: the request's own URL

Without SMTP_HOST the message is not sent: it is written to PARSEFLOW_DATA_DIR/outbox/ and printed to the server log,
so a local or hackathon install still works (open the link from there). /api/health reports which mode is active.
"""
import os
import smtplib
import ssl
import sys
import time
from email.message import EmailMessage
from pathlib import Path


def mode():
    return "smtp" if os.getenv("SMTP_HOST") else "outbox"


def send(to, subject, text, outbox_dir):
    """Send one plain-text email. Returns 'smtp' or the path of the outbox file. Raises on SMTP failure."""
    msg = EmailMessage()
    msg["From"] = os.getenv("SMTP_FROM") or os.getenv("SMTP_USER") or "ParseFlow <no-reply@localhost>"
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(text)
    if mode() == "smtp":
        host, port = os.getenv("SMTP_HOST"), int(os.getenv("SMTP_PORT", "587"))
        sec = (os.getenv("SMTP_SECURITY") or ("ssl" if port == 465 else "starttls")).lower()
        ctx = ssl.create_default_context()
        if sec == "ssl":
            s = smtplib.SMTP_SSL(host, port, context=ctx, timeout=20)
        else:
            s = smtplib.SMTP(host, port, timeout=20)
            if sec == "starttls":
                s.starttls(context=ctx)
        try:
            if os.getenv("SMTP_USER"):
                s.login(os.getenv("SMTP_USER"), os.getenv("SMTP_PASSWORD", ""))
            s.send_message(msg)
        finally:
            try:
                s.quit()
            except Exception:
                pass
        return "smtp"
    d = Path(outbox_dir)
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{time.strftime('%Y%m%d-%H%M%S')}-{abs(hash((to, subject, time.time()))) % 10**8:08d}.eml"
    p.write_bytes(bytes(msg))
    print(f"[parseflow] email not sent (no SMTP_HOST); saved to {p}\n  To: {to}\n  Subject: {subject}\n{text}", file=sys.stderr)
    return str(p)
