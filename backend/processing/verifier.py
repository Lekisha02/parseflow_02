"""Evidence verification: EXTRACT -> VALIDATE -> ACCEPT or FLAG.

Rules (stricter than v1):
  * A block is "verified" only with evidence that does not come from the extractor that produced it, OR when
    the content is read directly from the document's encoded text layer (no recognition step) and the
    text-layer quality checks pass. That basis is recorded explicitly as "document-encoded", not "independent".
  * Two outputs derived from the same underlying extraction never count as independent (e.g. table cells
    checked against the same text layer they were read from, or two OCR passes of the same engine).
  * OCR confidence and model self-reported confidence are recorded, never sufficient alone.
  * A formatting check or a plausible-looking output never upgrades a block.
  * The original extraction is never overwritten: raw_text / transforms keep the trail.

Statuses: verified | review (uncertain) | failed.
Each block gets: verification, evidence (short string, v1-compatible) and evidence_detail (structured).
"""
import re

from . import ocr
from .schema import OK, REVIEW, verdict


def _norm(s):
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


def _toks(s):
    return set(re.findall(r"[a-z0-9]{2,}", (s or "").lower()))


def _detail(basis, independent, signals, method):
    return dict(basis=basis, independent=bool(independent), signals=list(signals), confidence_method=method)


def verify(b, page_info, crop):
    """page_info: dict(width, height, norm_text, text_quality, hidden_text_norm, ocr_ok); crop: callable(bbox)->PIL image.
    Mutates and returns the block."""
    c, ex = b["confidence"], b["extractor"]
    x0, y0, x1, y1 = b["bbox"]
    signals, v = [], verdict(c)
    detail = None
    if x0 < -2 or y0 < -2 or x1 > page_info["width"] + 2 or y1 > page_info["height"] + 2 or x1 <= x0 or y1 <= y0:
        v = "failed"
        b["warnings"].append("Bounding box outside page bounds.")
        detail = _detail("geometry check failed", False, ["bbox outside page"], b.get("confidence_type", ""))
    elif ex == "digital_text_parser":
        q = page_info.get("text_quality", 1.0)
        signals.append(f"text-layer quality {q:.2f}")
        if not (b.get("content") or "").strip() and b["type"] not in ("figure",):
            v = "failed"; signals.append("empty content")
        elif q < 0.6:
            v = "review"; b["warnings"].append("Text layer looks unreliable (unmapped glyphs / implausible words).")
        elif b["type"] == "equation":
            v = "verified" if (not b.get("reconstructed") and c >= OK) else "review"
            if v == "review":
                signals.append("LaTeX structure reconstructed from glyph geometry (inferred, not encoded)")
        else:
            v = "verified" if c >= OK else verdict(c)
        detail = _detail("document-encoded text layer (no recognition step)", False, signals, "text-layer quality checks")
    elif ex == "table_engine":
        tb = b.get("cells") or []
        geometric = b.get("table_method") == "ruled_grid"
        if geometric:
            signals.append("cell grid defined by vector ruling lines (independent of text)")
        else:
            signals.append(f"column alignment score {b.get('alignment', 0):.2f}")
        empties = sum(1 for x in tb if not x.get("text"))
        if empties:
            signals.append(f"{empties} empty cell(s) kept empty")
        q = page_info.get("text_quality", 1.0)
        if q < 0.6:
            v = "review"; b["warnings"].append("Table text comes from an unreliable text layer.")
        elif geometric and c >= OK:
            v = "verified"
        else:
            v = "review" if c >= REVIEW else "failed"
            if not geometric:
                b["warnings"].append("Borderless table: structure inferred from text alignment; review cell boundaries.")
        detail = _detail("cell text from the encoded text layer; structure from " + ("ruling geometry" if geometric else "alignment heuristics"),
                         geometric, signals, "heuristic (estimated)")
    elif ex == "ocr":
        signals.append(f"mean OCR word confidence {c:.2f} (self-reported by engine)")
        indep = False
        hidden = page_info.get("hidden_text_norm") or ""
        if hidden and b.get("content"):
            toks = _toks(b["content"])
            hid = _toks(page_info.get("hidden_text_raw", ""))
            if toks:
                ov = len(toks & hid) / len(toks)
                signals.append(f"agreement with the PDF's embedded OCR layer {ov:.0%} (independent recognizer)")
                indep = ov >= 0.8
        if not (b.get("content") or "").strip() and b["type"] != "figure":
            v = "failed"
        elif indep and c >= REVIEW:
            v = "verified"
        else:
            v = "review" if c >= REVIEW else "failed"
            if v == "review":
                b["warnings"].append("OCR result without independent confirmation — review against the source image.")
        detail = _detail("OCR (Tesseract)", indep, signals, "mean word confidence")
    elif ex == "raster_diagram":
        st = b.get("structure") or {}
        signals.append("outlines, lines and arrowheads found in the image pixels; labels by OCR")
        signals.append("structure: " + (", ".join(x.get("type", "") for x in st.get("parts", [])) or "none"))
        ag = b.get("agreement")
        indep = False
        if ag:
            signals.append(ag["summary"])
            indep = ag.get("edges_total", 0) >= 3 and ag.get("edges_confirmed", 0) >= 0.9 * ag["edges_total"]
        signals += list(st.get("notes") or [])
        v = "review" if c >= REVIEW else "failed"
        detail = _detail("image geometry + OCR" + (" checked against an AI vision model's reading" if ag else ""), indep, signals,
                         "geometry heuristic (estimated)")
    elif ex == "vector_diagram":
        st = b.get("structure") or {}
        kinds = [x.get("type") for x in st.get("parts", [])]
        signals.append("shapes, lines and arrowheads read from the PDF's vector drawing commands; labels from the text layer")
        signals.append("structure: " + (", ".join(kinds) or "none"))
        signals += list(st.get("notes") or [])
        # how the drawing maps to nodes/edges is a geometric heuristic, so the interpretation is always for review
        v = "review" if c >= REVIEW else "failed"
        detail = _detail("vector drawing geometry + text layer (no recognition model)", False, signals,
                         "geometry heuristic (estimated)")
    elif ex == "fallback":
        v = "failed"
        detail = _detail("no extractor available", False, ["region preserved, not interpreted"], "heuristic (estimated)")
    else:  # model-based (vision_*, math_vision, math_model)
        indep = False
        if not b.get("content") and not b.get("latex"):
            v = "failed"; signals.append("model returned no content")
        else:
            signals.append("model self-reported confidence")
            if b["type"] == "table" and b.get("rows") and page_info.get("norm_text"):
                cells = [x for r in b["rows"] for x in r if x]
                fr = sum(1 for x in cells if _norm(x) and _norm(x) in page_info["norm_text"]) / max(1, len(cells))
                signals.append(f"cells found in the page text layer {fr:.0%} (independent of the model)")
                indep = fr >= 0.9
            if b["type"] in ("handwritten_text", "handwritten_code", "paragraph", "heading", "list") and page_info.get("ocr_ok"):
                try:
                    oc, ot = ocr.quick_conf(crop(b["bbox"]))
                    a, t = _toks(ot), _toks(b["content"])
                    if t and a:
                        ov = len(a & t) / len(t)
                        signals.append(f"independent OCR token overlap {ov:.0%}")
                        indep = indep or (ov >= 0.5 and oc >= 0.6)
                except Exception:
                    pass
            v = "verified" if (indep and c >= OK) else ("review" if c >= REVIEW else "failed")
        detail = _detail("vision / recognition model", indep, signals, "model (self-reported)")
    b["verification"] = v
    b["evidence_detail"] = detail
    b["evidence"] = ("independent: " if detail["independent"] else "") + "; ".join(detail["signals"]) if detail["signals"] else detail["basis"]
    if v != "verified" and not b["warnings"]:
        b["warnings"].append("Confidence below verification threshold (0.85) — review source." if v == "review" else "Low confidence — review source.")
    if v == "review" and ex.startswith(("vision", "math")) and not detail["independent"] and \
            not any("self-reported" in w for w in b["warnings"]):
        b["warnings"].append("Model self-reported confidence; no independent evidence available.")
    return b
