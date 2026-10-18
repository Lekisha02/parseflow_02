"""Every input format becomes something the extraction pipeline already reads well: a PDF (with a real text layer
wherever the source had text) or a page image.

    prepare(src, filename, out_dir, doc_id) -> Prepared(path, kind, fmt, method, notes)

* PDF, PNG, JPG/JPEG are used as they are (photos are turned upright from their EXIF orientation).
* Other images (TIFF incl. multi-page, BMP, GIF, WebP, HEIC when pillow-heif is installed) become PNG, or a PDF
  of images for several frames.
* Word (.docx, .doc, .odt, .rtf), PowerPoint (.pptx, .ppt, .odp) and Excel-type files (.xlsx, .xlsm, .xls, .ods,
  .csv, .tsv): LibreOffice converts documents and slides when it is installed (exact layout and pagination).
  Without it, .docx, .pptx, .xlsx/.xlsm, .csv/.tsv are rendered by ParseFlow itself from their content (python-docx,
  python-pptx, openpyxl): text stays text, tables become ruled tables, images and chart data are kept. Spreadsheets
  are always rendered by ParseFlow from the cell values (a LibreOffice print of a wide sheet splits rows across
  pages). The legacy binary formats (.doc, .ppt, .xls) and OpenDocument need LibreOffice.
* Text, Markdown and HTML are typeset into a PDF; EPUB, XPS, FB2, MOBI, CBZ and SVG are converted by MuPDF.

The converted PDF is stored as the document's source (`source_<id>_<name>.pdf`), next to the untouched original
(`original_<id>_<name>`), so the viewer, page images, Ask-about-this-block crops and resume-after-restart all work
on it. Nothing is guessed: a format that cannot be converted here fails with a clear message.
"""
import csv
import datetime as _dt
import glob
import html
import io
import os
import re
import shutil
import subprocess
import tempfile
import uuid
from dataclasses import dataclass, field

import pymupdf

PDF_EXT = {"pdf"}
IMAGE_DIRECT = {"png", "jpg", "jpeg"}
IMAGE_CONVERT = {"tif", "tiff", "bmp", "gif", "webp", "heic", "heif", "jfif", "jp2", "pbm", "pgm", "ppm", "pnm"}
WORD = {"docx", "doc", "odt", "rtf", "docm", "dotx", "wpd"}
SLIDES = {"pptx", "ppt", "odp", "pptm", "ppsx", "pps", "potx", "key"}
SHEETS = {"xlsx", "xlsm", "xls", "ods", "csv", "tsv", "xltx"}
TEXT = {"txt", "md", "markdown", "html", "htm", "xhtml", "json", "xml", "log"}
MUPDF = {"epub", "xps", "oxps", "fb2", "mobi", "cbz", "svg"}
SUPPORTED = PDF_EXT | IMAGE_DIRECT | IMAGE_CONVERT | WORD | SLIDES | SHEETS | TEXT | MUPDF
NATIVE_OK = {"docx", "docm", "dotx", "pptx", "pptm", "ppsx", "potx", "xlsx", "xlsm", "xltx", "csv", "tsv"}

LABEL = "PDF, Word (DOCX/DOC/ODT/RTF), PowerPoint (PPTX/PPT/ODP), Excel (XLSX/XLS/ODS/CSV), images (PNG/JPG/TIFF/BMP/GIF/WebP), text/Markdown/HTML, EPUB"


class ConvertError(Exception):
    def __init__(self, code, title, message):
        super().__init__(message)
        self.code, self.title, self.message = code, title, message

    def dict(self):
        return dict(code=self.code, title=self.title, message=self.message)


@dataclass
class Prepared:
    path: str
    kind: str              # pdf | image
    fmt: str               # the original format, upper case (DOCX, PPTX, ...)
    method: str            # as-is | libreoffice | native | mupdf | image
    notes: list = field(default_factory=list)


def ext_of(name):
    name = os.path.basename(str(name or ""))
    return name.lower().rsplit(".", 1)[-1] if "." in name else ""


def sniff(path):
    """The real type from the first bytes (a renamed file is not trusted by its extension alone)."""
    try:
        with open(path, "rb") as f:
            head = f.read(4096)
    except OSError:
        return None
    if head.startswith(b"%PDF"):
        return "pdf"
    if head.startswith(b"\x89PNG"):
        return "png"
    if head[:3] == b"\xff\xd8\xff":
        return "jpg"
    if head[:4] in (b"II*\x00", b"MM\x00*"):
        return "tiff"
    if head[:2] == b"BM":
        return "bmp"
    if head[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "webp"
    if head[:4] == b"PK\x03\x04":
        import zipfile
        try:
            with zipfile.ZipFile(path) as z:
                names = set(z.namelist())
                mt = z.read("mimetype").decode("ascii", "ignore").strip() if "mimetype" in names else ""
        except Exception:
            return "zip"
        if any(n.startswith("word/") for n in names):
            return "docx"
        if any(n.startswith("ppt/") for n in names):
            return "pptx"
        if any(n.startswith("xl/") for n in names):
            return "xlsx"
        if "opendocument.text" in mt:
            return "odt"
        if "opendocument.presentation" in mt:
            return "odp"
        if "opendocument.spreadsheet" in mt:
            return "ods"
        if "epub" in mt:
            return "epub"
        return "zip"
    if head[:8] == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1":
        return "ole"                       # legacy .doc / .ppt / .xls
    if head.startswith(b"{\\rtf"):
        return "rtf"
    return None


def _fail_type(name):
    raise ConvertError("PF-204", "Unsupported Document", f"ParseFlow cannot read “{name}”. Supported: {LABEL}.")


def check_supported(name):
    if ext_of(name) not in SUPPORTED:
        _fail_type(name)


# ------------------------------------------------------------------------------------------------ entry point
def prepare(src, filename, out_dir, doc_id="doc"):
    """Return a Prepared pointing at a PDF or image the pipeline reads. Re-uses an earlier conversion. What was
    done is recorded in <out_dir>/conversion.json for the pipeline (format, method, notes)."""
    import json
    meta = os.path.join(out_dir, "conversion.json")
    if os.path.exists(meta):
        try:
            m = json.load(open(meta))
            if m.get("original") == os.path.basename(src) and os.path.exists(m["path"]):
                return Prepared(m["path"], m["kind"], m["fmt"], m["method"], m.get("notes") or [])
        except Exception:
            pass
    p = _prepare(src, filename, out_dir, doc_id)
    if p.method != "as-is":
        try:
            with open(meta, "w") as f:
                json.dump(dict(original=os.path.basename(src), path=p.path, kind=p.kind, fmt=p.fmt, method=p.method,
                               notes=p.notes, filename=os.path.basename(filename or src)), f)
        except OSError:
            pass
    return p


def source_file(folder):
    """The file the pipeline reads for a stored document: the converted file when there is one, else source_*."""
    import json
    folder = str(folder)
    try:
        m = json.load(open(os.path.join(folder, "conversion.json")))
        p = m.get("path") or ""
        p = p if os.path.isabs(p) else os.path.join(folder, p)
        if os.path.exists(os.path.join(folder, os.path.basename(p))):
            return os.path.join(folder, os.path.basename(p))
    except Exception:
        pass
    try:
        c = sorted(f for f in os.listdir(folder) if f.startswith("source_"))
    except OSError:
        return None
    return os.path.join(folder, c[0]) if c else None


def original_name(folder):
    import json
    try:
        return json.load(open(os.path.join(str(folder), "conversion.json"))).get("filename")
    except Exception:
        return None


def conversion_info(out_dir, path):
    """The conversion record for a prepared file, or None."""
    import json
    try:
        m = json.load(open(os.path.join(out_dir, "conversion.json")))
        return m if os.path.basename(m.get("path", "")) == os.path.basename(path) else None
    except Exception:
        return None


def _prepare(src, filename, out_dir, doc_id="doc"):
    src = str(src)
    name = os.path.basename(str(filename or src))
    ext = ext_of(name)
    if os.path.getsize(src) == 0:
        raise ConvertError("PF-205", "Empty Document", "The file contains no data.")
    real = sniff(src)
    # a mislabelled file is read as what it really is (a PDF named .docx, a PNG named .jpg)
    if real in ("pdf", "png", "jpg") and ext not in IMAGE_DIRECT | PDF_EXT:
        ext = real
    if real == "ole" and ext in ("docx", "pptx", "xlsx"):
        ext = {"docx": "doc", "pptx": "ppt", "xlsx": "xls"}[ext]
    if real in ("docx", "pptx", "xlsx", "odt", "odp", "ods", "epub") and ext not in (WORD | SLIDES | SHEETS | MUPDF):
        ext = real
    if ext not in SUPPORTED:
        _fail_type(name)
    # a damaged Office file must not be 'converted' as if it were plain text
    zipped = {"docx", "docm", "dotx", "pptx", "pptm", "ppsx", "potx", "xlsx", "xlsm", "xltx", "odt", "odp", "ods", "epub"}
    if ext in zipped and real not in ("docx", "pptx", "xlsx", "odt", "odp", "ods", "epub", "zip"):
        raise ConvertError("PF-201", "Corrupt Document", f"This {ext.upper()} file is damaged or not really a {ext.upper()} file.")
    if ext in ("doc", "ppt", "xls") and real not in ("ole", "rtf", "docx", "pptx", "xlsx"):
        raise ConvertError("PF-201", "Corrupt Document", f"This {ext.upper()} file is damaged or not really a {ext.upper()} file.")
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name)[-120:] or "upload"
    fmt = ext.upper().replace("JPEG", "JPG") if ext != "jpeg" else "JPEG"
    if ext in PDF_EXT:
        return Prepared(src, "pdf", "PDF", "as-is")
    if ext in IMAGE_DIRECT:
        return _image_direct(src, out_dir, doc_id, safe, fmt)
    target = os.path.join(out_dir, f"source_{doc_id}_{safe}.pdf")
    if os.path.exists(target) and os.path.getsize(target) > 0:
        return Prepared(target, "pdf", fmt, "cached")
    if ext in IMAGE_CONVERT:
        return _image_convert(src, out_dir, doc_id, safe, fmt, ext)
    if ext in MUPDF:
        return _mupdf(src, target, fmt, ext)
    if ext in TEXT:
        return _text(src, target, fmt, ext)
    if ext in SHEETS:
        return _sheet(src, target, fmt, ext, name)
    mode = os.getenv("PARSEFLOW_OFFICE", "auto").lower()       # auto | libreoffice | native
    lo = soffice() if mode != "native" else None
    if lo:
        try:
            return _libreoffice(lo, src, target, fmt, ext)
        except ConvertError:
            if ext not in NATIVE_OK:
                raise
    if ext in NATIVE_OK and mode != "libreoffice":
        if ext in ("docx", "docm", "dotx"):
            return _docx(src, target, fmt)
        return _pptx(src, target, fmt)
    raise ConvertError("PF-207", "Converter Not Installed",
                       f"{fmt} files need LibreOffice to be converted (install it from libreoffice.org, then restart "
                       "ParseFlow). DOCX, PPTX, XLSX and CSV work without it.")


# ------------------------------------------------------------------------------------------------ images
def _image_direct(src, out_dir, doc_id, safe, fmt):
    from PIL import Image, ImageOps
    try:
        im = Image.open(src)
        exif = im.getexif() if hasattr(im, "getexif") else {}
        orient = exif.get(0x0112, 1) if exif else 1
    except Exception:
        raise ConvertError("PF-201", "Corrupt Document", "The image could not be decoded.")
    if orient in (None, 1):
        return Prepared(src, "image", fmt, "as-is")
    out = os.path.join(out_dir, f"upright_{doc_id}_{os.path.splitext(safe)[0]}.png")
    ImageOps.exif_transpose(im).convert("RGB").save(out)
    return Prepared(out, "image", fmt, "image", [f"Photo turned upright from its EXIF orientation ({orient})."])


def _image_convert(src, out_dir, doc_id, safe, fmt, ext):
    from PIL import Image, ImageOps, ImageSequence
    if ext in ("heic", "heif"):
        try:
            import pillow_heif
            pillow_heif.register_heif_opener()
        except Exception:
            raise ConvertError("PF-207", "Converter Not Installed", "HEIC photos need the pillow-heif package (pip install pillow-heif).")
    try:
        im = Image.open(src)
        frames = [ImageOps.exif_transpose(f.copy()).convert("RGB") for f in ImageSequence.Iterator(im)]
    except Exception:
        raise ConvertError("PF-201", "Corrupt Document", "The image could not be decoded.")
    if not frames:
        raise ConvertError("PF-205", "Empty Document", "The image has no frames.")
    if ext == "gif":
        frames = frames[:1]                       # an animated GIF: its first frame
    if len(frames) == 1:
        out = os.path.join(out_dir, f"source_{doc_id}_{os.path.splitext(safe)[0]}.png")
        frames[0].save(out)
        return Prepared(out, "image", fmt, "image", [f"{fmt} image converted to PNG."])
    # several pages (a scanned multi-page TIFF): one PDF page per frame, sized from its resolution
    out = os.path.join(out_dir, f"source_{doc_id}_{safe}.pdf")
    doc = pymupdf.open()
    try:
        dpi = float(im.info.get("dpi", (200, 200))[0]) or 200.0
    except (TypeError, ValueError):
        dpi = 200.0
    dpi = dpi if 50 <= dpi <= 1200 else 200.0
    for fr in frames:
        buf = io.BytesIO(); fr.save(buf, "PNG")
        w, h = float(fr.width * 72 / dpi), float(fr.height * 72 / dpi)
        pg = doc.new_page(width=w, height=h)
        pg.insert_image(pg.rect, stream=buf.getvalue())
    doc.save(out); doc.close()
    return Prepared(out, "pdf", fmt, "image", [f"Multi-page {fmt}: {len(frames)} pages, each read as a scanned page."])


# ------------------------------------------------------------------------------------------------ LibreOffice
def soffice():
    env = os.getenv("PARSEFLOW_SOFFICE")
    cands = [env] if env else []
    cands += [shutil.which("soffice"), shutil.which("libreoffice"),
              "/Applications/LibreOffice.app/Contents/MacOS/soffice",
              r"C:\Program Files\LibreOffice\program\soffice.exe", r"C:\Program Files (x86)\LibreOffice\program\soffice.exe"]
    return next((c for c in cands if c and os.path.exists(c)), None)


def _run_lo(lo, src, outdir, to):
    prof = tempfile.mkdtemp(prefix="pf_lo_")
    try:
        cmd = [lo, f"-env:UserInstallation=file://{prof if prof.startswith('/') else '/' + prof.replace(os.sep, '/')}",
               "--headless", "--norestore", "--nologo", "--convert-to", to, "--outdir", outdir, src]
        r = subprocess.run(cmd, capture_output=True, timeout=float(os.getenv("PARSEFLOW_CONVERT_TIMEOUT", "180")))
        stem = os.path.splitext(os.path.basename(src))[0]
        want = to.split(":")[0]
        out = os.path.join(outdir, f"{stem}.{want}")
        if not os.path.exists(out):
            raise ConvertError("PF-208", "Conversion Failed",
                               "LibreOffice could not convert this file (" + (r.stderr.decode("utf-8", "ignore").strip()[:200] or f"exit {r.returncode}") + ").")
        return out
    except subprocess.TimeoutExpired:
        raise ConvertError("PF-208", "Conversion Failed", "LibreOffice took too long to convert this file (PARSEFLOW_CONVERT_TIMEOUT).")
    finally:
        shutil.rmtree(prof, True)


def _libreoffice(lo, src, target, fmt, ext):
    work = tempfile.mkdtemp(prefix="pf_conv_")
    try:
        inp = os.path.join(work, "input." + ext)          # a plain name: LibreOffice dislikes some characters
        shutil.copy(src, inp)
        out = _run_lo(lo, inp, work, "pdf")
        _check_pdf(out, fmt)
        shutil.move(out, target)
    finally:
        shutil.rmtree(work, True)
    notes = [f"Converted from {fmt} to PDF with LibreOffice (layout and pagination as LibreOffice renders them)."]
    if ext in ("pptx", "pptm", "ppsx", "potx"):
        n = _append_chart_data(src, target)
        if n:
            notes.append(f"The data of {n} chart(s) is added on extra pages at the end, read from the presentation file "
                         "(a drawn chart only shows its bars).")
    return Prepared(target, "pdf", fmt, "libreoffice", notes)


def _charts_of_pptx(src):
    """[(slide number, title, header row, rows)] for every chart in a .pptx, from its embedded data."""
    try:
        import pptx
        prs = pptx.Presentation(src)
    except Exception:
        return []
    out = []

    def walk(shapes):
        for sh in shapes:
            if getattr(sh, "shape_type", None) == pptx.enum.shapes.MSO_SHAPE_TYPE.GROUP:
                yield from walk(sh.shapes)
            elif getattr(sh, "has_chart", False) and sh.has_chart:
                yield sh.chart
    for k, slide in enumerate(prs.slides, start=1):
        for ch in walk(slide.shapes):
            try:
                cats = [str(c) for c in ch.plots[0].categories]
                series = [(s.name, list(s.values)) for p_ in ch.plots for s in p_.series]
                title = ch.chart_title.text_frame.text if ch.has_title else ""
                rows = [[c] + [_cell_text(v[i]) if i < len(v) else "" for _, v in series] for i, c in enumerate(cats)]
                out.append((k, title, [""] + [n for n, _ in series], rows))
            except Exception:
                continue
    return out


def _append_chart_data(src, pdf_path):
    charts = _charts_of_pptx(src)
    if not charts:
        return 0
    d = pymupdf.open(pdf_path)
    W_, H_ = (d[0].rect.width, d[0].rect.height) if d.page_count else (720, 540)
    for k, title, hdr, rows in charts:
        pg = d.new_page(width=W_, height=H_)
        h_ = (f"<h3>Data of the chart on slide {k}" + (f" ({_esc(title)})" if title else "") + "</h3><table><tr>" +
              "".join(f"<td><b>{_esc(x)}</b></td>" for x in hdr) + "</tr>" +
              "".join("<tr>" + "".join(f"<td>{_esc(x)}</td>" for x in r) + "</tr>" for r in rows) + "</table>")
        pg.insert_htmlbox(pg.rect + (36, 36, -36, -36), h_, css=CSS, scale_low=0)
    tmp = pdf_path + ".tmp"
    d.save(tmp)
    d.close()
    os.replace(tmp, pdf_path)
    return len(charts)


def _check_pdf(path, fmt):
    try:
        d = pymupdf.open(path)
        n = d.page_count
        d.close()
    except Exception:
        n = 0
    if not n:
        raise ConvertError("PF-208", "Conversion Failed", f"The {fmt} file converted to an empty document.")


# ------------------------------------------------------------------------------------------------ typesetting
CSS = """
* { font-family: sans-serif; }
body { font-size: 10.5pt; line-height: 1.35; }
h1 { font-size: 20pt; margin: 10pt 0 6pt 0; } h2 { font-size: 16pt; margin: 9pt 0 5pt 0; }
h3 { font-size: 13.5pt; margin: 8pt 0 4pt 0; } h4, h5, h6 { font-size: 11.5pt; margin: 6pt 0 3pt 0; }
p { margin: 0 0 6pt 0; } li { margin: 0 0 2pt 0; }
table { border-collapse: collapse; margin: 6pt 0 8pt 0; width: 100%; }
td, th { border: 0.8px solid black; padding: 2pt 4pt; vertical-align: top; }
th { font-weight: bold; }
pre, code { font-family: monospace; font-size: 9pt; }
pre { white-space: pre-wrap; margin: 0 0 6pt 0; }
img { max-width: 100%; }
"""


def typeset(html_text, target, archive_dir=None, page=(612, 792), css=CSS, margin=54):
    """HTML -> PDF with MuPDF's Story: real text, ruled tables, images; pages are filled in order."""
    arch = pymupdf.Archive(archive_dir) if archive_dir else None
    story = pymupdf.Story(html=html_text, user_css=css, archive=arch)
    mediabox = pymupdf.Rect(0, 0, *page)
    where = mediabox + (margin, margin, -margin, -margin)
    writer = pymupdf.DocumentWriter(target)
    more, pages = 1, 0
    while more:
        dev = writer.begin_page(mediabox)
        more, _ = story.place(where)
        story.draw(dev)
        writer.end_page()
        pages += 1
        if pages > 20000:
            break
    writer.close()
    return pages


def _esc(s):
    return html.escape(str(s), quote=False)


def _text(src, target, fmt, ext):
    raw = open(src, "rb").read()
    text = None
    for enc in ("utf-8-sig", "utf-16", "cp1252", "latin-1"):
        try:
            text = raw.decode(enc)
            if enc == "utf-16" and not raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
                continue
            break
        except UnicodeDecodeError:
            continue
    text = text or raw.decode("utf-8", "replace")
    if ext in ("md", "markdown"):
        try:
            import markdown
            body = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists"])
        except ImportError:
            body = _mini_markdown(text)
        note = "Markdown typeset to PDF (headings, lists, tables and code kept)."
    elif ext in ("html", "htm", "xhtml"):
        body = re.sub(r"(?is)<(script|style|noscript)[^>]*>.*?</\1>", "", text)
        body = re.sub(r"(?is)<img[^>]+src=[\"']https?://[^>]*>", "", body)     # no network fetches
        note = "HTML typeset to PDF (scripts and remote images left out)."
    elif ext == "json":
        import json
        try:
            text = json.dumps(json.loads(text), indent=2, ensure_ascii=False)
        except Exception:
            pass
        body = f"<pre>{_esc(text)}</pre>"
        note = "JSON typeset to PDF."
    else:
        paras = re.split(r"\n\s*\n", text.replace("\r\n", "\n"))
        body = "".join(f"<pre>{_esc(p)}</pre>" if re.search(r"^ {2,}\S|\t", p, re.M) else f"<p>{_esc(p).replace(chr(10), ' ')}</p>"
                       for p in paras if p.strip())
        note = "Plain text typeset to PDF."
    typeset(f"<html><body>{body}</body></html>", target)
    return Prepared(target, "pdf", fmt, "native", [note])


def _mini_markdown(t):
    out = []
    for line in t.splitlines():
        m = re.match(r"(#{1,6})\s+(.*)", line)
        if m:
            out.append(f"<h{len(m.group(1))}>{_esc(m.group(2))}</h{len(m.group(1))}>")
        elif re.match(r"\s*[-*+]\s+", line):
            out.append(f"<ul><li>{_esc(re.sub(r'^\s*[-*+]\s+', '', line))}</li></ul>")
        elif line.strip():
            out.append(f"<p>{_esc(line)}</p>")
    return "\n".join(out)


def _mupdf(src, target, fmt, ext):
    try:
        d = pymupdf.open(src, filetype=ext if ext != "oxps" else "xps")
        pdf = d.convert_to_pdf()
        d.close()
    except Exception as e:
        raise ConvertError("PF-201", "Corrupt Document", f"The {fmt} file could not be read ({type(e).__name__}).")
    with open(target, "wb") as f:
        f.write(pdf)
    _check_pdf(target, fmt)
    return Prepared(target, "pdf", fmt, "mupdf", [f"{fmt} converted to PDF with MuPDF."])


# ------------------------------------------------------------------------------------------------ spreadsheets
def _cell_text(v, number_format=""):
    if v is None:
        return ""
    if isinstance(v, bool):
        return "TRUE" if v else "FALSE"
    if isinstance(v, _dt.datetime):
        return v.strftime("%Y-%m-%d %H:%M") if (v.hour or v.minute) else v.strftime("%Y-%m-%d")
    if isinstance(v, (_dt.date, _dt.time)):
        return v.isoformat()
    if isinstance(v, float):
        if "%" in (number_format or ""):
            return f"{v * 100:.{max(0, (number_format.split('.')[-1].count('0') if '.' in number_format else 0))}f}%"
        if v.is_integer() and abs(v) < 1e15:
            return str(int(v))
        return f"{v:.10g}"
    return str(v)


def _grid_html(title, rows, merges=None, max_cols=14):
    """rows: list of lists of strings (already trimmed). merges: {(r, c): (rowspan, colspan)} and covered cells."""
    merges = merges or {}
    covered = set()
    for (r, c), (rs, cs) in merges.items():
        for i in range(r, r + rs):
            for j in range(c, c + cs):
                if (i, j) != (r, c):
                    covered.add((i, j))
    ncol = max((len(r) for r in rows), default=0)
    groups = [(0, ncol)] if ncol <= max_cols else [(s, min(ncol, s + max_cols)) for s in range(0, ncol, max_cols)]
    parts = []
    for gi, (a, b) in enumerate(groups):
        cap = title + (f" — columns {_col_name(a)}–{_col_name(b - 1)}" if len(groups) > 1 else "")
        parts.append(f"<h2>{_esc(cap)}</h2><table>")
        for ri, row in enumerate(rows):
            tds = []
            for ci in range(a, b):
                if (ri, ci) in covered:
                    continue
                v = row[ci] if ci < len(row) else ""
                rs, cs = merges.get((ri, ci), (1, 1))
                cs = min(cs, b - ci)
                attr = (f' rowspan="{rs}"' if rs > 1 else "") + (f' colspan="{cs}"' if cs > 1 else "")
                tds.append(f"<td{attr}>{_esc(v) if v != '' else '&nbsp;'}</td>")
            parts.append("<tr>" + "".join(tds) + "</tr>")
        parts.append("</table>")
    return "".join(parts)


def _col_name(i):
    s = ""
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        s = chr(65 + r) + s
    return s


def _trim(rows):
    while rows and not any(str(x).strip() for x in rows[-1]):
        rows.pop()
    width = max((max((k + 1 for k, x in enumerate(r) if str(x).strip()), default=0) for r in rows), default=0)
    return [list(r[:width]) + [""] * (width - len(r[:width])) for r in rows]


def _sheet(src, target, fmt, ext, name=None):
    notes, sheets = [], []
    if ext in ("csv", "tsv"):
        raw = open(src, "rb").read().decode("utf-8-sig", "replace")
        try:
            dialect = csv.Sniffer().sniff(raw[:20000], delimiters=",;\t|") if ext == "csv" else csv.excel_tab
        except csv.Error:
            dialect = csv.excel
        rows = _trim([list(r) for r in csv.reader(io.StringIO(raw), dialect)])
        sheets.append((os.path.splitext(name or os.path.basename(src))[0] or "Data", rows, {}))
    else:
        path = src
        tmp = None
        if ext in ("xls", "ods"):
            lo = soffice()
            if not lo:
                raise ConvertError("PF-207", "Converter Not Installed",
                                   f"{fmt} files need LibreOffice (libreoffice.org); XLSX and CSV work without it.")
            tmp = tempfile.mkdtemp(prefix="pf_conv_")
            inp = os.path.join(tmp, "input." + ext)
            shutil.copy(src, inp)
            path = _run_lo(lo, inp, tmp, "xlsx")
            notes.append(f"{fmt} converted to XLSX with LibreOffice, then read cell by cell.")
        try:
            import openpyxl
        except ImportError:
            raise ConvertError("PF-207", "Converter Not Installed", "Excel files need the openpyxl package (pip install openpyxl).")
        def load(pth):
            try:
                return (openpyxl.load_workbook(pth, data_only=True), openpyxl.load_workbook(pth, data_only=False))
            except Exception as e:
                raise ConvertError("PF-201", "Corrupt Document", f"The spreadsheet could not be opened ({type(e).__name__}).")
        wb, wbf = load(path)

        def uncached(wb, wbf):
            n = 0
            for ws, wf in zip(wb.worksheets, wbf.worksheets):
                for row in wf.iter_rows():
                    for c in row:
                        if isinstance(c.value, str) and c.value.startswith("=") and ws[c.coordinate].value is None:
                            n += 1
            return n
        formulas = sum(1 for wf in wbf.worksheets for row in wf.iter_rows() for c in row if isinstance(c.value, str) and c.value.startswith("="))
        missing = uncached(wb, wbf) if formulas else 0
        if missing:
            # saved by a program that does not calculate (a script, some exporters): let LibreOffice compute them
            lo = soffice()
            if lo:
                tmp = tmp or tempfile.mkdtemp(prefix="pf_conv_")
                inp = os.path.join(tmp, "recalc." + ("xlsx" if path.lower().endswith(("xlsx", "xlsm")) else ext))
                shutil.copy(path, inp)
                outd = os.path.join(tmp, "recalc_out")
                os.makedirs(outd, exist_ok=True)
                try:
                    path = _run_lo(lo, inp, outd, "xlsx")
                    wb, wbf = load(path)
                    notes.append(f"{missing} formula cell(s) had no saved result; LibreOffice calculated them.")
                    missing = uncached(wb, wbf)
                except ConvertError:
                    pass
        for ws, wf in zip(wb.worksheets, wbf.worksheets):
            if getattr(ws, "sheet_state", "visible") != "visible":
                notes.append(f"Hidden sheet “{ws.title}” included.")
            rows = []
            for row in ws.iter_rows():
                vals = []
                for c in row:
                    v = c.value
                    if v is None:
                        f = wf[c.coordinate].value
                        if isinstance(f, str) and f.startswith("="):
                            vals.append(f"{f} (not calculated)")
                            continue
                    vals.append(_cell_text(v, getattr(c, "number_format", "")))
                rows.append(vals)
            rows = _trim(rows)
            merges = {}
            for rng in (ws.merged_cells.ranges if getattr(ws, "merged_cells", None) else []):
                r0, c0 = rng.min_row - 1, rng.min_col - 1
                if r0 < len(rows):
                    merges[(r0, c0)] = (rng.max_row - rng.min_row + 1, rng.max_col - rng.min_col + 1)
            sheets.append((ws.title, rows, merges))
        if formulas:
            notes.append(f"{formulas} formula cell(s): their calculated values are shown" +
                         (f"; {missing} could not be calculated here and show the formula." if missing else "."))
        if tmp:
            shutil.rmtree(tmp, True)
    body = []
    for title, rows, merges in sheets:
        if not rows:
            body.append(f"<h2>{_esc(title)}</h2><p>(empty sheet)</p>")
            continue
        body.append(_grid_html(title, rows, merges))
    ncol = max((max((len(r) for r in rows), default=0) for _, rows, _ in sheets), default=0)
    size = 9 if ncol <= 8 else 8 if ncol <= 11 else 7
    css = CSS + f"body {{ font-size: {size}pt; }} td, th {{ padding: 1.5pt 3pt; }}"
    typeset("<html><body>" + "".join(body) + "</body></html>", target, page=(792, 612), css=css, margin=36)
    notes.insert(0, f"{fmt} read cell by cell ({len(sheets)} sheet{'s' if len(sheets) != 1 else ''}); each sheet is a ruled table.")
    return Prepared(target, "pdf", fmt, "native", notes)


# ------------------------------------------------------------------------------------------------ Word (native)
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
M = "{http://schemas.openxmlformats.org/officeDocument/2006/math}"


def _docx(src, target, fmt):
    try:
        import docx
    except ImportError:
        raise ConvertError("PF-207", "Converter Not Installed", "Word files need LibreOffice or the python-docx package (pip install python-docx).")
    try:
        d = docx.Document(src)
    except Exception as e:
        raise ConvertError("PF-201", "Corrupt Document", f"The Word file could not be opened ({type(e).__name__}).")
    work = tempfile.mkdtemp(prefix="pf_docx_")
    imgs = {}

    def image(rid):
        if rid in imgs:
            return imgs[rid]
        try:
            part = d.part.related_parts[rid]
            ext = os.path.splitext(part.partname)[1].lower() or ".png"
            if ext not in (".png", ".jpg", ".jpeg", ".gif", ".bmp"):
                return None
            name = f"img{len(imgs)}{ext}"
            with open(os.path.join(work, name), "wb") as f:
                f.write(part.blob)
            imgs[rid] = name
            return name
        except Exception:
            return None

    def runs_html(p_el):
        out = []
        for el in p_el.iter():
            if el.tag == W + "r":
                rpr = el.find(W + "rPr")
                t = "".join((x.text or "") if x.tag == W + "t" else "\t" if x.tag == W + "tab" else "\n" if x.tag in (W + "br", W + "cr") else ""
                            for x in el)
                s = _esc(t).replace("\n", "<br/>")
                if rpr is not None and s:
                    va = rpr.find(W + "vertAlign")
                    if rpr.find(W + "b") is not None and rpr.find(W + "b").get(W + "val") not in ("0", "false"):
                        s = f"<b>{s}</b>"
                    if rpr.find(W + "i") is not None and rpr.find(W + "i").get(W + "val") not in ("0", "false"):
                        s = f"<i>{s}</i>"
                    if va is not None and va.get(W + "val") == "superscript":
                        s = f"<sup>{s}</sup>"
                    elif va is not None and va.get(W + "val") == "subscript":
                        s = f"<sub>{s}</sub>"
                out.append(s)
                for blip in el.iter(A + "blip"):
                    n = image(blip.get(R + "embed"))
                    if n:
                        out.append(f'<img src="{n}"/>')
            elif el.tag == M + "oMath" and el.getparent().tag != M + "oMathPara":
                out.append("<i>" + _esc("".join(x.text or "" for x in el.iter(M + "t"))) + "</i>")
        return "".join(out)

    def para(p_el):
        pstyle = p_el.find(f"{W}pPr/{W}pStyle")
        sid = pstyle.get(W + "val") if pstyle is not None else ""
        try:
            sname = d.styles.get_by_id(sid, docx.enum.style.WD_STYLE_TYPE.PARAGRAPH).name if sid else ""
        except Exception:
            sname = sid
        body = runs_html(p_el)
        if not body.strip():
            return None, None
        m = re.match(r"(?i)heading\s*(\d)", sname or "")
        if m:
            return "h", f"<h{min(6, int(m.group(1)))}>{body}</h{min(6, int(m.group(1)))}>"
        if (sname or "").lower() == "title":
            return "h", f"<h1>{body}</h1>"
        if p_el.find(f"{W}pPr/{W}numPr") is not None or re.match(r"(?i)list", sname or ""):
            return ("ol" if re.search(r"(?i)number", sname or "") else "ul"), f"<li>{body}</li>"
        return "p", f"<p>{body}</p>"

    def table(t_el):
        rows = t_el.findall(W + "tr")
        grid = []          # per row: list of [colstart, span, vmerge, html]
        for tr in rows:
            col, cells = 0, []
            for tc in tr.findall(W + "tc"):
                pr = tc.find(W + "tcPr")
                span = int(pr.find(W + "gridSpan").get(W + "val")) if pr is not None and pr.find(W + "gridSpan") is not None else 1
                vm = pr.find(W + "vMerge") if pr is not None else None
                vmerge = None if vm is None else (vm.get(W + "val") or "continue")
                inner = "".join(h for _, h in (para(p) for p in tc.findall(W + "p")) if h) or "&nbsp;"
                inner = re.sub(r"</?p>", "", inner) if inner.count("<p>") == 1 else inner
                cells.append([col, span, vmerge, inner])
                col += span
            grid.append(cells)
        out = ["<table>"]
        for ri, cells in enumerate(grid):
            tds = []
            for col, span, vmerge, inner in cells:
                if vmerge == "continue":
                    continue
                rs = 1
                if vmerge == "restart":
                    for nxt in grid[ri + 1:]:
                        c = next((x for x in nxt if x[0] == col), None)
                        if c is not None and c[2] == "continue":
                            rs += 1
                        else:
                            break
                attr = (f' colspan="{span}"' if span > 1 else "") + (f' rowspan="{rs}"' if rs > 1 else "")
                tds.append(f"<td{attr}>{inner}</td>")
            out.append("<tr>" + "".join(tds) + "</tr>")
        out.append("</table>")
        return "".join(out)

    html_parts, open_list = [], None
    for el in d.element.body.iterchildren():
        if el.tag == W + "p":
            kind, h = para(el)
            if h is None:
                continue
            if kind in ("ul", "ol"):
                if open_list != kind:
                    if open_list:
                        html_parts.append(f"</{open_list}>")
                    html_parts.append(f"<{kind}>")
                    open_list = kind
                html_parts.append(h)
                continue
            if open_list:
                html_parts.append(f"</{open_list}>"); open_list = None
            html_parts.append(h)
        elif el.tag == W + "tbl":
            if open_list:
                html_parts.append(f"</{open_list}>"); open_list = None
            html_parts.append(table(el))
    if open_list:
        html_parts.append(f"</{open_list}>")
    sec = d.sections[0] if d.sections else None
    page = (sec.page_width.pt, sec.page_height.pt) if sec is not None and sec.page_width else (612, 792)
    try:
        typeset("<html><body>" + "".join(html_parts) + "</body></html>", target, archive_dir=work, page=page)
    finally:
        shutil.rmtree(work, True)
    _check_pdf(target, fmt)
    return Prepared(target, "pdf", fmt, "native",
                    [f"{fmt} rendered by ParseFlow from its content (headings, lists, tables with merged cells, images); "
                     "page breaks and exact layout differ from Word. Install LibreOffice for Word's own layout."])


# ------------------------------------------------------------------------------------------------ PowerPoint (native)
def _pptx(src, target, fmt):
    try:
        import pptx
        from pptx.util import Emu
    except ImportError:
        raise ConvertError("PF-207", "Converter Not Installed", "PowerPoint files need LibreOffice or the python-pptx package (pip install python-pptx).")
    try:
        prs = pptx.Presentation(src)
    except Exception as e:
        raise ConvertError("PF-201", "Corrupt Document", f"The PowerPoint file could not be opened ({type(e).__name__}).")
    pt = lambda emu: (emu or 0) / 12700.0
    W_, H_ = pt(prs.slide_width) or 720, pt(prs.slide_height) or 540
    doc = pymupdf.open()
    charts = notes_pages = 0

    def text_html(tf, default):
        out = []
        for p in tf.paragraphs:
            runs = []
            for r in p.runs:
                s = _esc(r.text)
                if not s:
                    continue
                f = r.font
                if f.bold:
                    s = f"<b>{s}</b>"
                if f.italic:
                    s = f"<i>{s}</i>"
                size = f.size.pt if f.size else None
                runs.append(f'<span style="font-size:{size:.1f}pt">{s}</span>' if size else s)
            if runs:
                bullet = "• " if p.level and p.level > 0 else ""
                out.append(f'<p style="margin:0 0 3pt {p.level * 14}pt">{bullet}{"".join(runs)}</p>')
        return f'<div style="font-size:{default}pt">' + "".join(out) + "</div>" if out else ""

    def place(page, shape, ox=0.0, oy=0.0, sx=1.0, sy=1.0):
        nonlocal charts
        try:
            x, y = ox + pt(shape.left) * sx, oy + pt(shape.top) * sy
            w, h = pt(shape.width) * sx, pt(shape.height) * sy
        except Exception:
            return
        rect = pymupdf.Rect(x, y, x + max(w, 4), y + max(h, 4)) & page.rect
        if rect.is_empty:
            return
        st = shape.shape_type
        if st == pptx.enum.shapes.MSO_SHAPE_TYPE.GROUP:
            xfrm = shape._element.grpSpPr.find(A + "xfrm")
            try:
                ch_off, ch_ext = xfrm.find(A + "chOff"), xfrm.find(A + "chExt")
                cx, cy = int(ch_off.get("x")), int(ch_off.get("y"))
                cw, chh = int(ch_ext.get("cx")) or 1, int(ch_ext.get("cy")) or 1
                gsx, gsy = (shape.width / cw) * sx, (shape.height / chh) * sy
                gox, goy = x - pt(cx) * gsx, y - pt(cy) * gsy
            except Exception:
                gsx, gsy, gox, goy = sx, sy, ox, oy
            for s in shape.shapes:
                place(page, s, gox, goy, gsx, gsy)
            return
        if getattr(shape, "has_table", False) and shape.has_table:
            rows = [[_esc(c.text) or "&nbsp;" for c in r.cells] for r in shape.table.rows]
            h_ = "<table>" + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows) + "</table>"
            page.insert_htmlbox(rect, h_, css=CSS + "body{font-size:11pt}", scale_low=0)
            return
        if getattr(shape, "has_chart", False) and shape.has_chart:
            ch = shape.chart
            try:
                cats = [str(c) for c in ch.plots[0].categories]
                series = [(s.name, list(s.values)) for p_ in ch.plots for s in p_.series]
                hdr = "<tr><td>&nbsp;</td>" + "".join(f"<td><b>{_esc(n)}</b></td>" for n, _ in series) + "</tr>"
                body = "".join("<tr><td>" + _esc(c) + "</td>" + "".join(f"<td>{_cell_text(v[i]) if i < len(v) else ''}</td>" for _, v in series) + "</tr>"
                               for i, c in enumerate(cats))
                title = _esc(ch.chart_title.text_frame.text) if ch.has_title else ""
                page.insert_htmlbox(rect, f"<p><b>Data of the chart{(' (' + title + ')') if title else ''}</b></p><table>{hdr}{body}</table>",
                                    css=CSS + "body{font-size:10pt}", scale_low=0)
                charts += 1
            except Exception:
                pass
            return
        if st == pptx.enum.shapes.MSO_SHAPE_TYPE.PICTURE or hasattr(shape, "image"):
            try:
                page.insert_image(rect, stream=shape.image.blob, keep_proportion=True)
            except Exception:
                pass
        if getattr(shape, "has_text_frame", False) and shape.has_text_frame and shape.text_frame.text.strip():
            default = 18
            try:
                if shape.is_placeholder and "TITLE" in str(shape.placeholder_format.type):
                    default = 32
            except Exception:
                pass
            hh = text_html(shape.text_frame, default)
            if hh:
                page.insert_htmlbox(rect, hh, css="* { font-family: sans-serif; }", scale_low=0)

    for k, slide in enumerate(prs.slides):
        page = doc.new_page(width=W_, height=H_)
        for shape in slide.shapes:
            place(page, shape)
        if slide.has_notes_slide and slide.notes_slide.notes_text_frame is not None:
            nt = slide.notes_slide.notes_text_frame.text.strip()
            if nt:
                npg = doc.new_page(width=W_, height=H_)
                npg.insert_htmlbox(npg.rect + (36, 36, -36, -36),
                                   f"<h3>Speaker notes — slide {k + 1}</h3>" + "".join(f"<p>{_esc(x)}</p>" for x in nt.splitlines() if x.strip()),
                                   css=CSS, scale_low=0)
                notes_pages += 1
    if not doc.page_count:
        raise ConvertError("PF-205", "Empty Document", "The presentation has no slides.")
    doc.save(target)
    doc.close()
    notes = [f"{fmt} rendered by ParseFlow slide by slide (text boxes, tables, pictures at their positions); "
             "themes and backgrounds are not drawn. Install LibreOffice for PowerPoint's own look."]
    if charts:
        notes.append(f"{charts} chart(s) kept as their data tables.")
    if notes_pages:
        notes.append(f"Speaker notes of {notes_pages} slide(s) follow their slide as a separate page.")
    return Prepared(target, "pdf", fmt, "native", notes)


def annotate(blocks, conv):
    """Use what is known about the original format. A slide's title is its first short line of text (presentations
    give every title the same size, so size rarity alone does not find them)."""
    if (conv.get("fmt") or "") not in {x.upper() for x in SLIDES}:
        return
    by_page = {}
    for b in blocks:
        by_page.setdefault(b.get("page"), []).append(b)
    for page, bs in by_page.items():
        bs = sorted(bs, key=lambda b: b.get("reading_order", 0))
        first = next((b for b in bs if b.get("type") in ("paragraph", "heading", "list") and b.get("content")), None)
        if first is None or first.get("type") == "heading":
            continue
        t = str(first["content"]).strip()
        if first["type"] == "paragraph" and len(t) <= 120 and "\n" not in t and not t.endswith((".", ",", ";")):
            first["type"] = "heading"; first["level"] = 2
            first.setdefault("transforms", []).append("slide title: first line of text on the slide")
