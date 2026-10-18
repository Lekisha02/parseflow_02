"""Files written next to document.json so a very large document is served in pieces.

    document.json          the full Digital Twin (unchanged; the canonical output)
    document.json.gz       the same, gzip-compressed (served as-is to browsers that accept gzip: ~8x smaller)
    document.lite.json.gz  what the viewer needs to open the document: every block, without the heavy provenance
                           fields (evidence_detail, raw_text, transforms) and without the Markdown; the viewer
                           fetches those per page when a block is inspected
    blocks.jsonl           one full block per line, in page order
    blocks.idx.json        {page: [byte offset, byte length]} into blocks.jsonl
    document.md            the default Markdown (served without parsing the JSON again)

A 1,312-page book: document.json 25 MB -> the viewer downloads ~1.3 MB (lite, gzip) to open it.
"""
import gzip
import json
import os

HEAVY = ("evidence_detail", "raw_text", "transforms")
VARIANTS = ("document.json.gz", "document.lite.json.gz", "blocks.jsonl", "blocks.idx.json", "document.md")


def lite(doc):
    out = {k: v for k, v in doc.items() if k not in ("markdown", "router")}
    out["blocks"] = [{k: v for k, v in b.items() if k not in HEAVY} for b in doc.get("blocks", [])]
    out["lite"] = True
    out["markdown"] = None
    return out


def _atomic(path, data, mode="wb"):
    tmp = path + ".tmp"
    with open(tmp, mode) as f:
        f.write(data)
    os.replace(tmp, path)


def write_variants(folder, doc=None):
    """Write the compressed, lite and per-page files via streaming (safe on 512MB RAM)."""
    p = os.path.join(folder, "document.json")
    if doc is None:
        with open(p, encoding="utf-8") as f:
            doc = json.load(f)

    # 1. Stream document.json.gz in 64KB chunks (uses 64KB RAM instead of buffering the whole file)
    gz_path = os.path.join(folder, "document.json.gz")
    tmp_gz = gz_path + ".tmp"
    with open(p, "rb") as f_in, gzip.open(tmp_gz, "wb", compresslevel=6) as f_out:
        while chunk := f_in.read(65536):
            f_out.write(chunk)
    os.replace(tmp_gz, gz_path)

    # 2. Stream document.lite.json.gz directly to gzip (avoids large in-memory string buffer)
    lite_path = os.path.join(folder, "document.lite.json.gz")
    tmp_lite = lite_path + ".tmp"
    with gzip.open(tmp_lite, "wt", encoding="utf-8", compresslevel=6) as f_out:
        json.dump(lite(doc), f_out, ensure_ascii=True)
    os.replace(tmp_lite, lite_path)

    # 3. Stream blocks.jsonl line-by-line (avoids joining tens of thousands of strings in RAM)
    jsonl_path = os.path.join(folder, "blocks.jsonl")
    tmp_jsonl = jsonl_path + ".tmp"
    idx, off = {}, 0
    with open(tmp_jsonl, "wb") as f_out:
        for b in sorted(doc.get("blocks", []), key=lambda b: (b.get("page", 0), b.get("reading_order", 0))):
            line = (json.dumps(b, ensure_ascii=True) + "\n").encode()
            pg = str(b.get("page", 0))
            if pg not in idx:
                idx[pg] = [off, 0]
            idx[pg][1] += len(line)
            f_out.write(line)
            off += len(line)
    os.replace(tmp_jsonl, jsonl_path)

    _atomic(os.path.join(folder, "blocks.idx.json"), json.dumps(idx).encode())
    _atomic(os.path.join(folder, "document.md"), (doc.get("markdown") or "").encode("utf-8"))
    return True


def fresh(folder):
    """True when every variant exists and is newer than document.json."""
    p = os.path.join(folder, "document.json")
    try:
        t = os.stat(p).st_mtime
        return all(os.stat(os.path.join(folder, v)).st_mtime >= t for v in VARIANTS)
    except OSError:
        return False


def ensure(folder):
    """Variants for documents processed before they existed (written once, on first request)."""
    if not fresh(folder) and os.path.exists(os.path.join(folder, "document.json")):
        write_variants(folder)


def page_blocks(folder, pages):
    """Full blocks of the given pages, read straight from blocks.jsonl."""
    ensure(folder)
    with open(os.path.join(folder, "blocks.idx.json"), encoding="utf-8") as f:
        idx = json.load(f)
    out = []
    with open(os.path.join(folder, "blocks.jsonl"), "rb") as f:
        for pg in pages:
            r = idx.get(str(pg))
            if not r:
                continue
            f.seek(r[0])
            out += [json.loads(line) for line in f.read(r[1]).splitlines() if line.strip()]
    return out
