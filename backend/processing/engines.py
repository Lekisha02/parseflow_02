"""Optional local models for what Tesseract and the text layer cannot do: another OCR engine, handwriting, and the
data of chart *images*. All are off by default, run on this machine only, and never download weights; a missing
model is reported through info() and the region stays flagged, exactly as before.

    PARSEFLOW_OCR_ENGINE          tesseract (default) | paddleocr | surya | command
    PARSEFLOW_OCR_CMD             command: executable taking an image path, printing JSON
                                  [{"text": "...", "conf": 0.93, "box": [x0, y0, x1, y1]}, ...]  (pixels, one entry per word)
    PARSEFLOW_HANDWRITING         none (default) | trocr | command
    PARSEFLOW_HANDWRITING_CMD     command: image path in, text out (or JSON {"text": ..., "conf": ...})
    PARSEFLOW_CHART_MODEL         none (default) | deplot | command
    PARSEFLOW_CHART_CMD           command: image path in, CSV/TSV table (first row = header) or JSON {"chart_type", "rows"} out
    PARSEFLOW_MODEL_DIR           folder holding local weights: paddleocr/ (det, rec, cls), trocr/ (a Hugging Face
                                  checkpoint folder), deplot/ (a Hugging Face checkpoint folder). Surya uses its own cache.

The "command" plugins make any local model usable without touching ParseFlow (a Python script, a CLI, a container).
The built-in adapters (PaddleOCR, Surya, TrOCR, DePlot) follow those projects' documented local-inference APIs; they
are loaded only when selected and only from local files.
"""
import csv
import io
import json
import os
import subprocess
import tempfile
import threading

_LOCK = threading.Lock()
_MODELS = {}
_ERRORS = {}


def _env(k, default="none"):
    return (os.getenv(k) or default).strip().lower()


def _dir(sub):
    return os.path.join(os.getenv("PARSEFLOW_MODEL_DIR", ""), sub)


def _run_cmd(cmd, img, timeout=120):
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img.save(f.name); path = f.name
    try:
        r = subprocess.run(cmd.split() + [path], capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0:
            raise RuntimeError((r.stderr or "").strip()[:200] or f"exit code {r.returncode}")
        return r.stdout
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def _model(kind, loader):
    with _LOCK:
        if kind in _MODELS:
            return _MODELS[kind]
        try:
            _MODELS[kind] = loader()
            _ERRORS.pop(kind, None)
        except Exception as e:                  # missing package or weights: reported, never fatal
            _MODELS[kind] = None
            _ERRORS[kind] = f"{type(e).__name__}: {e}"[:300]
        return _MODELS[kind]


def _need_dir(path):
    if not os.path.isdir(path) or not os.listdir(path):
        raise FileNotFoundError(f"model folder {path!r} is missing or empty (weights are never downloaded automatically)")
    return path


# ------------------------------------------------------------------------------------------------- OCR engines
def ocr_engine():
    return _env("PARSEFLOW_OCR_ENGINE", "tesseract")


def ocr_ready():
    e = ocr_engine()
    if e == "tesseract":
        return False
    if e == "command":
        return bool(os.getenv("PARSEFLOW_OCR_CMD"))
    return _ocr_model() is not None


def _ocr_model():
    e = ocr_engine()
    if e == "paddleocr":
        def load():
            from paddleocr import PaddleOCR
            d = _need_dir(_dir("paddleocr"))
            return PaddleOCR(det_model_dir=os.path.join(d, "det"), rec_model_dir=os.path.join(d, "rec"),
                             cls_model_dir=os.path.join(d, "cls"), use_angle_cls=True, show_log=False)
        return _model("paddleocr", load)
    if e == "surya":
        def load():
            from surya.recognition import RecognitionPredictor
            from surya.detection import DetectionPredictor
            return (RecognitionPredictor(), DetectionPredictor())
        return _model("surya", load)
    return None


def ocr_words(img):
    """Words from the configured non-Tesseract engine, in Tesseract's word format; None to fall back to Tesseract."""
    e = ocr_engine()
    try:
        if e == "command" and os.getenv("PARSEFLOW_OCR_CMD"):
            raw = json.loads(_run_cmd(os.getenv("PARSEFLOW_OCR_CMD"), img))
            items = [(w["text"], float(w.get("conf", 0.9)), w["box"]) for w in raw if str(w.get("text", "")).strip()]
        elif e == "paddleocr":
            m = _ocr_model()
            if m is None:
                return None
            import numpy as np
            res = m.ocr(np.array(img.convert("RGB")), cls=True) or []
            items = []
            for line in (res[0] or []):
                quad, (txt, conf) = line
                xs, ys = [p[0] for p in quad], [p[1] for p in quad]
                items += _split_line(txt, float(conf), (min(xs), min(ys), max(xs), max(ys)))
        elif e == "surya":
            m = _ocr_model()
            if m is None:
                return None
            rec, det = m
            pred = rec([img.convert("RGB")], det_predictor=det)[0]
            items = []
            for tl in pred.text_lines:
                items += _split_line(tl.text, float(tl.confidence or 0.8), tuple(tl.bbox))
        else:
            return None
    except Exception as ex:
        _ERRORS[e] = f"{type(ex).__name__}: {ex}"[:300]
        return None
    out = []
    lines = sorted({round(b[1] / 4) for _, _, b in items})
    for k, (t, c, b) in enumerate(items):
        ln = lines.index(round(b[1] / 4))
        out.append(dict(text=t, conf=max(0.0, min(1.0, c)), box=tuple(b), key=(1, ln), line=ln, h=b[3] - b[1], block=1, par=ln))
    return out


def _split_line(text, conf, box):
    """Line-level engines -> word boxes (widths in proportion to word length)."""
    ws = text.split()
    if not ws:
        return []
    tot = sum(len(w) for w in ws) + len(ws) - 1
    x0, y0, x1, y1 = box
    cur, out = x0, []
    for w in ws:
        wd = (x1 - x0) * len(w) / max(1, tot)
        out.append((w, conf, (cur, y0, cur + wd, y1)))
        cur += wd + (x1 - x0) / max(1, tot)
    return out


# ------------------------------------------------------------------------------------------------- handwriting
def handwriting_name():
    return _env("PARSEFLOW_HANDWRITING")


def handwriting_ready():
    n = handwriting_name()
    if n == "command":
        return bool(os.getenv("PARSEFLOW_HANDWRITING_CMD"))
    if n == "trocr":
        return _trocr() is not None
    return False


def _trocr():
    def load():
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        d = _need_dir(_dir("trocr"))
        return (TrOCRProcessor.from_pretrained(d, local_files_only=True), VisionEncoderDecoderModel.from_pretrained(d, local_files_only=True))
    return _model("trocr", load)


def read_handwriting(img):
    """(text, confidence) for a handwriting crop, or None. Line by line for TrOCR (it reads one line at a time)."""
    n = handwriting_name()
    try:
        if n == "command" and os.getenv("PARSEFLOW_HANDWRITING_CMD"):
            out = _run_cmd(os.getenv("PARSEFLOW_HANDWRITING_CMD"), img).strip()
            if out.startswith("{"):
                j = json.loads(out)
                return (j.get("text") or "").strip() or None, float(j.get("conf", 0.6))
            return (out or None), 0.6
        if n == "trocr":
            m = _trocr()
            if m is None:
                return None
            proc, model = m
            texts = []
            for line in _line_crops(img):
                pix = proc(images=line.convert("RGB"), return_tensors="pt").pixel_values
                ids = model.generate(pix, max_new_tokens=128)
                texts.append(proc.batch_decode(ids, skip_special_tokens=True)[0].strip())
            t = "\n".join(x for x in texts if x)
            return (t or None), 0.6
    except Exception as ex:
        _ERRORS[n] = f"{type(ex).__name__}: {ex}"[:300]
    return None


def _line_crops(img):
    """Split a crop into text lines at empty horizontal bands (ink projection)."""
    import numpy as np
    g = np.array(img.convert("L"))
    ink = (g < min(170, np.median(g) * 0.65)).sum(axis=1)
    rows, start = [], None
    for y, v in enumerate(ink):
        if v > 1 and start is None:
            start = y
        elif v <= 1 and start is not None:
            if y - start > 6:
                rows.append((start, y))
            start = None
    if start is not None and len(ink) - start > 6:
        rows.append((start, len(ink)))
    if not rows:
        return [img]
    return [img.crop((0, max(0, a - 4), img.width, min(img.height, b + 4))) for a, b in rows]


# ------------------------------------------------------------------------------------------------- chart images
def chart_name():
    return _env("PARSEFLOW_CHART_MODEL")


def chart_ready():
    n = chart_name()
    if n == "command":
        return bool(os.getenv("PARSEFLOW_CHART_CMD"))
    if n == "deplot":
        return _deplot() is not None
    return False


def _deplot():
    def load():
        from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor
        d = _need_dir(_dir("deplot"))
        return (Pix2StructProcessor.from_pretrained(d, local_files_only=True),
                Pix2StructForConditionalGeneration.from_pretrained(d, local_files_only=True))
    return _model("deplot", load)


def _table_text(txt):
    """CSV / TSV / DePlot's ' | ' + <0x0A> linearised table -> rows."""
    txt = txt.replace("<0x0A>", "\n").strip()
    lines = [l for l in txt.splitlines() if l.strip()]
    if not lines:
        return None
    if all(" | " in l for l in lines[1:] or lines):
        rows = [[c.strip() for c in l.split(" | ")] for l in lines]
    else:
        dialect = "excel-tab" if "\t" in lines[0] else "excel"
        rows = [[c.strip() for c in r] for r in csv.reader(io.StringIO("\n".join(lines)), dialect=dialect)]
    rows = [r for r in rows if any(r)]
    if rows and rows[0] and rows[0][0].upper().startswith("TITLE"):
        rows = rows[1:]
    return rows if len(rows) >= 2 and max(len(r) for r in rows) >= 2 else None


def read_chart(img):
    """dict(chart_type, rows) for a chart image, or None."""
    n = chart_name()
    try:
        if n == "command" and os.getenv("PARSEFLOW_CHART_CMD"):
            out = _run_cmd(os.getenv("PARSEFLOW_CHART_CMD"), img).strip()
            if out.startswith("{"):
                j = json.loads(out)
                rows = j.get("rows")
                return dict(chart_type=j.get("chart_type") or "chart", rows=rows) if rows else None
            rows = _table_text(out)
            return dict(chart_type="chart", rows=rows) if rows else None
        if n == "deplot":
            m = _deplot()
            if m is None:
                return None
            proc, model = m
            inp = proc(images=img.convert("RGB"), text="Generate underlying data table of the figure below:", return_tensors="pt")
            ids = model.generate(**inp, max_new_tokens=512)
            rows = _table_text(proc.decode(ids[0], skip_special_tokens=True))
            return dict(chart_type="chart", rows=rows) if rows else None
    except Exception as ex:
        _ERRORS[n] = f"{type(ex).__name__}: {ex}"[:300]
    return None


def info():
    return dict(ocr_engine=ocr_engine(), ocr_ready=ocr_ready() if ocr_engine() != "tesseract" else True,
                handwriting=handwriting_name(), handwriting_ready=handwriting_ready(),
                chart_model=chart_name(), chart_ready=chart_ready(), errors=dict(_ERRORS))
