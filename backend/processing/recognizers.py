"""Optional, pluggable local formula recognizers for *image* math (scanned equations).

Digital PDFs never need these: mathtex.py rebuilds LaTeX from the text layer. For scans, a recognizer is
called only on detected equation regions, never on whole pages.

Configuration (all optional; default is "none" = offline, no model, equations from scans are kept as OCR text and flagged):

    PARSEFLOW_MATH_RECOGNIZER = none | pix2tex | rapid_latex_ocr | command
    PARSEFLOW_MATH_MODEL_DIR  = directory holding the model weights (required for pix2tex / rapid_latex_ocr)
    PARSEFLOW_MATH_CMD        = for "command": an executable taking an image path, printing LaTeX on stdout
                                (lets you plug in any local model without touching ParseFlow code)

Weights are never downloaded at runtime. If the configured weights are missing, recognize() returns None and
the reason is reported through info()["error"] and block warnings.

Licensing / requirements (see README "Formula recognition"):
  pix2tex (LaTeX-OCR, MIT)          pip install "pix2tex[cli]" ; needs PyTorch (CPU OK, ~1-3 s per equation)
                                     weights.pth + image_resizer.pth from the project's GitHub release, placed in MODEL_DIR
  rapid_latex_ocr (Apache-2.0)      pip install rapid_latex_ocr ; ONNX Runtime only (no torch)
                                     encoder.onnx, decoder.onnx, image_resizer.onnx, tokenizer.json in MODEL_DIR
"""
import os
import subprocess
import tempfile
import threading

_LOCK = threading.Lock()
_STATE = dict(name=None, model=None, error=None)


def name():
    return os.getenv("PARSEFLOW_MATH_RECOGNIZER", "none").strip().lower()


def info():
    _load()
    return dict(name=name(), ready=_STATE["model"] is not None or (name() == "command" and bool(os.getenv("PARSEFLOW_MATH_CMD"))),
                error=_STATE["error"])


def _need(paths):
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError("missing model file(s): " + ", ".join(missing) + " (weights are never downloaded automatically)")


def _load():
    n = name()
    with _LOCK:
        if _STATE["name"] == n:
            return
        _STATE.update(name=n, model=None, error=None)
        d = os.getenv("PARSEFLOW_MATH_MODEL_DIR", "")
        try:
            if n in ("", "none"):
                return
            if n == "pix2tex":
                _need([os.path.join(d, "weights.pth"), os.path.join(d, "image_resizer.pth")])
                from munch import Munch                     # noqa: imported lazily, optional dependency
                from pix2tex.cli import LatexOCR
                args = Munch(config=os.path.join(d, "config.yaml") if os.path.exists(os.path.join(d, "config.yaml")) else None,
                             checkpoint=os.path.join(d, "weights.pth"), no_cuda=True, no_resize=False)
                if args.config is None:
                    del args["config"]
                _STATE["model"] = LatexOCR(args)
            elif n == "rapid_latex_ocr":
                files = dict(image_resizer_path="image_resizer.onnx", encoder_path="encoder.onnx", decoder_path="decoder.onnx", tokenizer_json="tokenizer.json")
                paths = {k: os.path.join(d, v) for k, v in files.items()}
                _need(paths.values())
                from rapid_latex_ocr import LaTeXOCR
                _STATE["model"] = LaTeXOCR(**paths)
            elif n == "command":
                if not os.getenv("PARSEFLOW_MATH_CMD"):
                    raise ValueError("PARSEFLOW_MATH_CMD is not set")
            else:
                raise ValueError(f"unknown recognizer '{n}'")
        except Exception as e:      # missing package / weights: report, never crash, never download
            _STATE["error"] = f"{type(e).__name__}: {e}"
            _STATE["model"] = None


def recognize(img, timeout=30):
    """img: PIL image of one equation region. Returns (latex, confidence) or None when unavailable/failed."""
    _load()
    n = name()
    try:
        if n == "command" and os.getenv("PARSEFLOW_MATH_CMD"):
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                img.save(f.name)
            try:
                r = subprocess.run([os.getenv("PARSEFLOW_MATH_CMD"), f.name], capture_output=True, text=True, timeout=timeout)
            finally:
                os.unlink(f.name)
            out = r.stdout.strip()
            return (out, 0.6) if r.returncode == 0 and out else None
        m = _STATE["model"]
        if m is None:
            return None
        if n == "pix2tex":
            return (m(img), 0.6)
        if n == "rapid_latex_ocr":
            import io
            buf = io.BytesIO(); img.convert("RGB").save(buf, "PNG")
            res = m(buf.getvalue())
            latex = res[0] if isinstance(res, (tuple, list)) else res
            return (latex, 0.6) if latex else None
    except Exception as e:
        _STATE["error"] = f"{type(e).__name__}: {e}"
    return None
