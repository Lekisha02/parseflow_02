"""Optional neural embeddings for the Ask assistant's retrieval (on top of the built-in keyword + LSA search).

    ASK_EMBED_MODEL     embedding model name, e.g. nomic-embed-text (Ollama), text-embedding-3-small (OpenAI);
                        empty = off (default)
    ASK_EMBED_BASE_URL  OpenAI-compatible endpoint (default: ASK_BASE_URL, else Ollama's http://localhost:11434/v1)
    ASK_EMBED_API_KEY   key if the endpoint needs one (default: ASK_API_KEY / OPENAI_API_KEY)

Chunk vectors are computed once per document (in batches) and saved next to it as ask_emb.npz; each question then
costs one embedding call. With a local Ollama model this stays fully offline.
"""
import os

import httpx


def model():
    return (os.getenv("ASK_EMBED_MODEL") or "").strip()


def available():
    return bool(model())


def _base():
    return (os.getenv("ASK_EMBED_BASE_URL") or os.getenv("ASK_BASE_URL") or "http://localhost:11434/v1").rstrip("/")


def _key():
    return os.getenv("ASK_EMBED_API_KEY") or os.getenv("ASK_API_KEY") or os.getenv("OPENAI_API_KEY") or ""


def embed(texts, client=None, timeout=60):
    """List of vectors for texts (OpenAI-compatible /embeddings)."""
    http = client or httpx
    headers = {"content-type": "application/json"}
    if _key():
        headers["authorization"] = "Bearer " + _key()
    r = http.post(_base() + "/embeddings", json={"model": model(), "input": texts}, headers=headers, timeout=timeout)
    if r.status_code >= 400:
        raise RuntimeError(f"embedding endpoint HTTP {r.status_code}")
    data = sorted(r.json()["data"], key=lambda d: d.get("index", 0))
    return [d["embedding"] for d in data]


def _chunk_vectors(ix):
    import numpy as np
    if ix.emb is not None:
        return ix.emb
    folder = getattr(ix, "folder", None)
    path = os.path.join(folder, "ask_emb.npz") if folder else None
    if path and os.path.exists(path):
        try:
            z = np.load(path, allow_pickle=False)
            if str(z["model"]) == model() and z["vecs"].shape[0] == len(ix.chunks):
                ix.emb = z["vecs"]
                return ix.emb
        except Exception:
            pass
    cap = int(os.getenv("ASK_EMBED_MAX_CHUNKS", "20000"))
    texts = [c["section"] + "\n" + c["text"][:2000] for c in ix.chunks[:cap]]
    vecs = []
    for k in range(0, len(texts), 64):
        vecs += embed(texts[k:k + 64])
    m = np.array(vecs, np.float32)
    m /= (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)
    if len(m) < len(ix.chunks):
        m = np.vstack([m, np.zeros((len(ix.chunks) - len(m), m.shape[1]), np.float32)])
    ix.emb = m
    if path:
        try:
            np.savez_compressed(path, vecs=m, model=np.array(model()))
        except Exception:
            pass
    return m


def similarity(ix, text):
    """Cosine similarity of text to every chunk, or None when embeddings are off or the endpoint fails (retrieval
    then simply uses keywords + LSA)."""
    import time
    if not available() or not ix.chunks or not text.strip():
        return None
    if time.time() - getattr(ix, "_emb_failed", 0) < 300:
        return None                       # the endpoint failed a moment ago: do not retry on every question
    try:
        import numpy as np
        m = _chunk_vectors(ix)
        q = np.array(embed([text])[0], np.float32)
        q /= (np.linalg.norm(q) + 1e-9)
        return m @ q
    except Exception:
        ix.emb = None
        ix._emb_failed = time.time()
        return None
