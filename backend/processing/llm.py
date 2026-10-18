"""Text LLM access for the Ask assistant. Server-side only; nothing is called unless configured.

ASK_PROVIDER   anthropic | gemini | openai | ollama | none   (default: the vision provider if a vision key is set, else none)
ASK_API_KEY    key for the provider (falls back to VISION_API_KEY / ANTHROPIC_API_KEY / GEMINI_API_KEY / OPENAI_API_KEY)
ASK_MODEL      model name (defaults: claude-sonnet-5-5, gemini-2.5-flash, gpt-4o-mini, llama3.1)
ASK_BASE_URL   OpenAI-compatible endpoint (default https://api.openai.com/v1; ollama: http://localhost:11434/v1).
               Any OpenAI-compatible server works (Ollama, LM Studio, vLLM, llama.cpp server, Groq, OpenRouter ...),
               so the assistant can run fully offline with a local model.
"""
import base64
import os

import httpx

DEFAULT_MODEL = {"anthropic": "claude-sonnet-5-5", "gemini": "gemini-3.5-flash", "openai": "gpt-4o-mini", "ollama": "llama3.1"}


def provider():
    p = (os.getenv("ASK_PROVIDER") or "").strip().lower()
    if p:
        return p
    if os.getenv("VISION_API_KEY") or os.getenv("ANTHROPIC_API_KEY") or os.getenv("GEMINI_API_KEY"):
        return (os.getenv("VISION_PROVIDER") or "gemini").lower()
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    return "none"


def key():
    p = provider()
    env = {"anthropic": "ANTHROPIC_API_KEY", "gemini": "GEMINI_API_KEY", "openai": "OPENAI_API_KEY"}.get(p, "")
    return os.getenv("ASK_API_KEY") or os.getenv("VISION_API_KEY") or (os.getenv(env) if env else "") or ""


def model():
    return os.getenv("ASK_MODEL") or DEFAULT_MODEL.get(provider(), "")


def base_url():
    if provider() == "ollama":
        return (os.getenv("ASK_BASE_URL") or "http://localhost:11434/v1").rstrip("/")
    return (os.getenv("ASK_BASE_URL") or "https://api.openai.com/v1").rstrip("/")


def available():
    p = provider()
    if p in ("none", "", "off"):
        return False
    if p == "ollama":
        return True
    if p == "openai" and os.getenv("ASK_BASE_URL"):
        return True          # local OpenAI-compatible servers often need no key
    return bool(key())


def vision_ok():
    """Can the configured model look at images? ASK_VISION=1/0 forces it; auto: the hosted providers (Anthropic,
    Gemini, OpenAI) can; a local/OpenAI-compatible server only when ASK_VISION=1 (the model must be a vision model,
    e.g. llava or qwen2.5-vl in Ollama)."""
    v = (os.getenv("ASK_VISION") or "auto").strip().lower()
    if not available() or v in ("0", "no", "off", "false"):
        return False
    if v in ("1", "yes", "on", "true"):
        return True
    p = provider()
    return p in ("anthropic", "gemini") or (p == "openai" and not os.getenv("ASK_BASE_URL"))


def info():
    return dict(configured=available(), provider=provider(), model=model() if available() else None, vision=vision_ok())


class LLMError(Exception):
    pass


def chat(system, messages, max_tokens=1600, temperature=0.2, timeout=None, client=None, images=None):
    """messages: [{"role": "user"|"assistant", "content": str}]. images: [(mime, bytes)] attached to the last user
    message (only when vision_ok()). Returns the reply text. Raises LLMError."""
    if not available():
        raise LLMError("no language model configured")
    p = provider()
    timeout = timeout or float(os.getenv("ASK_TIMEOUT", "120"))
    http = client or httpx
    imgs = [(m, base64.b64encode(b).decode()) for m, b in (images or [])]
    try:
        if p == "anthropic":
            msgs = [dict(m) for m in messages]
            if imgs:
                msgs[-1]["content"] = [{"type": "image", "source": {"type": "base64", "media_type": m, "data": d}} for m, d in imgs] + \
                    [{"type": "text", "text": msgs[-1]["content"]}]
            r = http.post("https://api.anthropic.com/v1/messages", timeout=timeout,
                          headers={"x-api-key": key(), "anthropic-version": "2023-06-01", "content-type": "application/json"},
                          json={"model": model(), "max_tokens": max_tokens, "temperature": temperature, "system": system,
                                "messages": msgs})
            _check(r)
            return "".join(c.get("text", "") for c in r.json().get("content", []) if c.get("type") == "text").strip()
        if p == "gemini":
            contents = [{"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]} for m in messages]
            if imgs:
                contents[-1]["parts"] += [{"inline_data": {"mime_type": m, "data": d}} for m, d in imgs]
            r = http.post(f"https://generativelanguage.googleapis.com/v1beta/models/{model()}:generateContent", timeout=timeout,
                          headers={"x-goog-api-key": key()},
                          json={"systemInstruction": {"parts": [{"text": system}]}, "contents": contents,
                                "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens}})
            _check(r)
            return _gemini_text(r.json())
        # openai-compatible (openai, ollama, lm studio, vllm, ...)
        headers = {"content-type": "application/json"}
        if key():
            headers["authorization"] = "Bearer " + key()
        msgs = [dict(m) for m in messages]
        if imgs:
            msgs[-1]["content"] = [{"type": "text", "text": msgs[-1]["content"]}] + \
                [{"type": "image_url", "image_url": {"url": f"data:{m};base64,{d}"}} for m, d in imgs]
        r = http.post(base_url() + "/chat/completions", timeout=timeout, headers=headers,
                      json={"model": model(), "temperature": temperature, "max_tokens": max_tokens,
                            "messages": [{"role": "system", "content": system}] + msgs})
        _check(r)
        return (r.json()["choices"][0]["message"]["content"] or "").strip()
    except LLMError:
        raise
    except httpx.TimeoutException:
        raise LLMError("the language model did not answer in time")
    except httpx.HTTPError as e:
        raise LLMError(f"could not reach the language model ({type(e).__name__})")
    except (KeyError, IndexError, ValueError, TypeError):
        raise LLMError("the language model returned an unexpected response")


def _gemini_text(j):
    """The answer text of a Gemini response. Thinking models may spend the output budget before writing; a blocked
    prompt or an empty answer is reported with its reason instead of as a parsing error."""
    cands = j.get("candidates") or []
    if not cands:
        why = (j.get("promptFeedback") or {}).get("blockReason")
        raise LLMError("the model returned no answer" + (f" (blocked: {why})" if why else ""))
    parts = (cands[0].get("content") or {}).get("parts") or []
    text = "".join(pt.get("text", "") for pt in parts if not pt.get("thought")).strip()
    if not text:
        fr = cands[0].get("finishReason") or "unknown"
        hint = " - raise ASK_MAX_TOKENS (thinking models use part of it before answering)" if fr == "MAX_TOKENS" else ""
        raise LLMError(f"the model returned an empty answer (finish reason {fr}){hint}")
    return text


def _check(r):
    if r.status_code >= 400:
        msg = ""
        try:
            j = r.json()
            msg = (j.get("error") or {}).get("message") if isinstance(j.get("error"), dict) else str(j.get("error") or "")
        except Exception:
            pass
        raise LLMError(f"language model HTTP {r.status_code}" + (f": {msg[:200]}" if msg else ""))


def selftest(client=None):
    """One tiny real request to the configured model (and, if it takes images, one with a picture of a digit), to
    show whether answers will work: reachability, the key, the model name, vision. Never raises."""
    import time
    out = dict(configured=available(), provider=provider(), model=model() if available() else None, vision=vision_ok())
    if not available():
        out.update(ok=False, error="No AI model configured (set ASK_PROVIDER and ASK_API_KEY, or ASK_PROVIDER=ollama).")
        return out
    t = time.time()
    try:
        r = chat("You are a connection test. Reply with exactly the word OK.", [dict(role="user", content="ping")],
                 max_tokens=512, timeout=60, client=client)
        out.update(ok=bool(r and r.strip()), reply=(r or "")[:80], seconds=round(time.time() - t, 2))
    except LLMError as e:
        out.update(ok=False, error=str(e), seconds=round(time.time() - t, 2))
        return out
    if vision_ok():
        try:
            import io
            from PIL import Image, ImageDraw, ImageFont
            im = Image.new("RGB", (120, 120), "white")
            d = ImageDraw.Draw(im)
            try:
                f = ImageFont.load_default(size=90)
            except TypeError:
                f = ImageFont.load_default()
            d.text((30, 10), "7", fill="black", font=f)
            buf = io.BytesIO(); im.save(buf, "PNG")
            t2 = time.time()
            r2 = chat("You read images.", [dict(role="user", content="Which digit is in the image? Answer with the digit only.")],
                      max_tokens=512, timeout=60, client=client, images=[("image/png", buf.getvalue())])
            out.update(vision_reply=(r2 or "")[:40], vision_ok=bool(r2 and "7" in r2), vision_seconds=round(time.time() - t2, 2))
        except LLMError as e:
            out.update(vision_ok=False, vision_error=str(e))
    return out
