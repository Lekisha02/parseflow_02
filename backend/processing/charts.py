"""Data from vector charts in digital PDFs, without a model.

A chart drawn as vector graphics (most charts exported from Excel, matplotlib, R, PowerPoint ...) keeps its exact
geometry: every bar is a filled rectangle, every line series a path, and the axis tick labels are text. The values
can therefore be measured instead of guessed:

  1. value axis   - numeric tick labels in one column (vertical bars, lines) or one row (horizontal bars) give a
                    linear map from page position to value (least squares; rejected unless it fits closely)
  2. bars         - filled rectangles standing on a common base line; grouped into series by fill colour
  3. lines        - stroked paths with at least three vertices that are not axis or grid lines
  4. categories   - the text labels under (or beside) the bars / at the line vertices
  5. legend       - a small swatch (rectangle or short line) with a text label next to it names a series

The result is a table (category x series) plus the chart type. Values are measurements from the drawing:
they are rounded to the precision of the tick labels and the block is marked for review, never 'verified'.
Charts that do not fit these patterns return None and stay figures.
"""
import re
import statistics

NUM = re.compile(r"^[-−–]?\(?[$€£¥]?\s*[-−–]?\d[\d,]*(?:\.\d+)?\s*%?\)?$")


def _num(t):
    t = t.strip()
    if not NUM.match(t):
        return None
    neg = t.startswith(("-", "−", "–", "(")) and not t.startswith("((")
    v = re.sub(r"[^\d.]", "", t)
    try:
        x = float(v)
    except ValueError:
        return None
    return -x if neg else x


def _decimals(texts):
    d = 0
    for t in texts:
        m = re.search(r"\.(\d+)", t)
        if m:
            d = max(d, len(m.group(1)))
    return d


def _inside(b, r, tol=1.0):
    return b[0] >= r[0] - tol and b[1] >= r[1] - tol and b[2] <= r[2] + tol and b[3] <= r[3] + tol


def _fit(pairs):
    """Least-squares v = a*pos + b; returns (a, b, max relative residual) or None."""
    if len(pairs) < 2:
        return None
    xs = [p for p, _ in pairs]; ys = [v for _, v in pairs]
    mx, my = statistics.mean(xs), statistics.mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx < 1e-6:
        return None
    a = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    b = my - a * mx
    span = (max(ys) - min(ys)) or 1.0
    res = max(abs(a * x + b - y) for x, y in zip(xs, ys)) / span
    return a, b, res


def _ticks(drawings, orient):
    """Positions of short tick marks: horizontal stubs (y axis) or vertical stubs (x axis)."""
    out = []
    for d in drawings or ():
        for it in d.get("items") or ():
            if it[0] != "l":
                continue
            p, q = it[1], it[2]
            if orient == "y" and abs(p.y - q.y) < 0.3 and 1 < abs(p.x - q.x) < 10:
                out.append(((p.y + q.y) / 2, min(p.x, q.x)))
            elif orient == "x" and abs(p.x - q.x) < 0.3 and 1 < abs(p.y - q.y) < 10:
                out.append(((p.x + q.x) / 2, min(p.y, q.y)))
    return out


def _axis(labels, orient, drawings=None):
    """Numeric tick labels stacked in a column (orient 'y') or lined up in a row (orient 'x') -> value map. Where
    the chart draws tick marks, their exact positions replace the (approximately centred) label positions."""
    nums = [(s, _num(s["text"])) for s in labels]
    nums = [(s, v) for s, v in nums if v is not None]
    best = None
    groups = {}
    for s, v in nums:
        key = round((s["bbox"][2] if orient == "y" else (s["bbox"][1] + s["bbox"][3]) / 2) / 4)
        groups.setdefault(key, []).append((s, v))
    for g in groups.values():
        if len(g) < 3 or len({v for _, v in g}) < 3:
            continue
        ticks = _ticks(drawings, orient)
        pos = []
        for s, v in g:
            c = (s["bbox"][1] + s["bbox"][3]) / 2 if orient == "y" else (s["bbox"][0] + s["bbox"][2]) / 2
            half = (s["bbox"][3] - s["bbox"][1]) / 2 if orient == "y" else (s["bbox"][2] - s["bbox"][0]) / 2 + 2
            t = [tp for tp, _ in ticks if abs(tp - c) <= half + 1]
            pos.append((min(t, key=lambda tp: abs(tp - c)) if t else c, v))
        f = _fit(pos)
        if f and f[2] < 0.02 and abs(f[0]) > 1e-9:
            if best is None or len(g) > len(best[1]):
                best = (f, g)
    if not best:
        return None
    (a, b, _), g = best
    return dict(a=a, b=b, labels=[s for s, _ in g], decimals=_decimals([s["text"] for s, _ in g]),
                edge=max(s["bbox"][2] for s, _ in g) if orient == "y" else min(s["bbox"][1] for s, _ in g))


def _colour(f):
    return tuple(round(c, 2) for c in f) if f else None


def _legend(drawings, labels, region):
    """{colour: name} from swatches with a text label right next to them."""
    out = {}
    for d in drawings:
        r = d.get("rect")
        col = _colour(d.get("fill") or d.get("color"))
        if r is None or col is None:
            continue
        small = r.width < 0.08 * (region[2] - region[0]) and r.height < 0.05 * (region[3] - region[1]) and max(r.width, r.height) > 2
        if not small:
            continue
        cy = (r.y0 + r.y1) / 2
        nxt = [s for s in labels if 0 <= s["bbox"][0] - r.x1 < 12 and s["bbox"][1] - 3 <= cy <= s["bbox"][3] + 3]
        if nxt:
            out.setdefault(col, min(nxt, key=lambda s: s["bbox"][0])["text"].strip())
    return out


def vector_chart(region, drawings, segs):
    """region: figure box (PDF points); drawings: page.get_drawings(); segs: text segments (with .bbox/.text) on
    the page. Returns dict(chart_type, columns, rows, axes, notes) or None."""
    labels = [dict(bbox=tuple(s.bbox), text=s.text) for s in segs if _inside(s.bbox, region, 60)]
    dr = [d for d in drawings if d.get("rect") is not None and _inside(tuple(d["rect"]), region, 2)]
    if len(dr) < 3 or len(labels) < 3:
        return None
    W, H = region[2] - region[0], region[3] - region[1]
    legend = _legend(dr, labels, region)
    fills = [d for d in dr if d.get("fill") and len(d.get("items") or []) == 1 and d["items"][0][0] == "re"
             and _colour(d["fill"]) not in ((1.0, 1.0, 1.0),) and not (d["rect"].width > 0.9 * W and d["rect"].height > 0.9 * H)]
    swatch = lambda r: r.width < 0.08 * W and r.height < 0.05 * H and abs(r.width - r.height) < 0.5 * max(r.width, r.height) and \
        any(_colour(d.get("fill")) in legend for d in dr if d.get("rect") is r)
    bars = [d for d in fills if not swatch(d["rect"])]
    for orient in ("v", "h"):
        res = _bars(bars, labels, legend, region, orient, dr)
        if res:
            return res
    return _lines(dr, labels, legend, region)


def _cat_label(labels, x, y, orient, axis_labels, base):
    """Category label for a bar / point at (x, y): the nearest non-numeric (or non-axis) text past the base line."""
    cands = []
    for s in labels:
        if s in axis_labels:
            continue
        b = s["bbox"]
        if orient == "v":
            if b[1] < base - 1:
                continue
            d = abs((b[0] + b[2]) / 2 - x) + 0.3 * abs(b[1] - base)
            if b[0] - 6 <= x <= b[2] + 6:
                cands.append((d, s))
        else:
            if b[2] > base + 1:
                continue
            d = abs((b[1] + b[3]) / 2 - y) + 0.3 * abs(base - b[2])
            if b[1] - 4 <= y <= b[3] + 4:
                cands.append((d, s))
    if not cands:
        # grouped bars: the group's label sits under the group, not under each bar - take the nearest label
        near = []
        for s in labels:
            if s in axis_labels:
                continue
            b = s["bbox"]
            if orient == "v" and b[1] >= base - 1 and b[1] - base < 30:
                near.append((abs((b[0] + b[2]) / 2 - x), s))
            elif orient == "h" and b[2] <= base + 1 and base - b[2] < 80:
                near.append((abs((b[1] + b[3]) / 2 - y), s))
        if near:
            d, s = min(near, key=lambda t: t[0])
            span = (s["bbox"][2] - s["bbox"][0]) if orient == "v" else (s["bbox"][3] - s["bbox"][1])
            if d < max(30.0, 1.5 * span):
                return s["text"].strip()
        return None
    return min(cands, key=lambda t: t[0])[1]["text"].strip()


def _bars(bars, labels, legend, region, orient, bars_all=()):
    if len(bars) < 2:
        return None
    ax = _axis(labels, "y" if orient == "v" else "x", [d for d in bars_all])
    if not ax:
        return None
    a, b = ax["a"], ax["b"]
    zero = -b / a                                       # page position of value 0
    rects = [d["rect"] for d in bars]
    # bars stand on a common base: one edge at the zero line (or at the plot's edge when the axis does not start at 0)
    if orient == "v":
        ends = [r.y1 for r in rects]
    else:
        ends = [r.x0 for r in rects]
    base = statistics.median(ends)
    shared = sum(1 for e in ends if abs(e - base) < 1.5) >= 0.6 * len(ends)
    if not shared and abs(base - zero) > 3:
        return None
    base_pos = base if shared else zero               # the drawn base line (tick labels are only approximately centred)
    on_base = [d for d in bars if (abs((d["rect"].y1 if orient == "v" else d["rect"].x0) - base_pos) < 1.5
                                   or abs((d["rect"].y0 if orient == "v" else d["rect"].x1) - base_pos) < 1.5)]
    if len(on_base) < 2:
        return None
    # thin long rectangles are axis / grid lines drawn as filled boxes, not bars
    on_base = [d for d in on_base if min(d["rect"].width, d["rect"].height) > 1.2]
    if len(on_base) < 2:
        return None
    dec = ax["decimals"]
    series = {}
    for d in on_base:
        r = d["rect"]
        col = _colour(d["fill"])
        if orient == "v":
            far = r.y0 if abs(r.y1 - base_pos) < abs(r.y0 - base_pos) else r.y1
            val = a * far + b
            cat = _cat_label(labels, (r.x0 + r.x1) / 2, None, "v", ax["labels"], max(r.y1, base_pos))
            key = (r.x0 + r.x1) / 2
        else:
            far = r.x1 if abs(r.x0 - base_pos) < abs(r.x1 - base_pos) else r.x0
            val = a * far + b
            cat = _cat_label(labels, None, (r.y0 + r.y1) / 2, "h", ax["labels"], min(r.x0, base_pos))
            key = (r.y0 + r.y1) / 2
        series.setdefault(col, []).append(dict(key=key, cat=cat, val=round(val, dec + 1 if dec < 3 else dec)))
    # categories: bars of different series that share a label (grouped bars) or the same position
    cats = []
    for col, items in series.items():
        for it in items:
            name = it["cat"] or f"#{len(cats) + 1}"
            if name not in cats:
                cats.append(name)
    order = {c: min(it["key"] for items in series.values() for it in items if (it["cat"] or "") == c) if any((it["cat"] or "") == c for items in series.values() for it in items) else 1e9 for c in cats}
    cats.sort(key=lambda c: order[c])
    cols = list(series)
    names = [legend.get(c) or (f"Series {k + 1}" if len(cols) > 1 else "Value") for k, c in enumerate(cols)]
    rows = []
    for c in cats:
        row = [c]
        for col in cols:
            v = next((it["val"] for it in series[col] if (it["cat"] or "") == c), None)
            row.append("" if v is None else _fmt(v, dec))
        rows.append(row)
    if all(r[0].startswith("#") for r in rows):
        return None                                      # no category labels: not confident it is a bar chart
    used = [l["bbox"] for l in ax["labels"]] + [l["bbox"] for l in labels if l["text"].strip() in set(cats) | set(names)]
    return dict(chart_type="bar" if orient == "v" else "horizontal bar", columns=["Category"] + names, rows=rows, used=used,
                axes=dict(value_axis="y" if orient == "v" else "x", ticks=[s["text"] for s in ax["labels"]]),
                notes=[f"values measured from {len(on_base)} bar(s) against {len(ax['labels'])} axis tick labels"])


def _fmt(v, dec):
    return f"{v:.{dec}f}" if dec else (str(int(round(v))) if abs(v - round(v)) < 0.05 else f"{v:.1f}")


def _lines(dr, labels, legend, region):
    ax = _axis(labels, "y", dr)
    if not ax:
        return None
    a, b = ax["a"], ax["b"]
    W = region[2] - region[0]
    series = []
    for d in dr:
        items = d.get("items") or []
        if d.get("fill") or len(items) < 2 or any(it[0] != "l" for it in items):
            continue
        pts = [(items[0][1].x, items[0][1].y)] + [(it[2].x, it[2].y) for it in items]
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        if max(ys) - min(ys) < 1.0 or max(xs) - min(xs) < 0.3 * W:
            continue                                     # flat: axis or grid line
        if any(x2 < x1 - 0.5 for (x1, _), (x2, _) in zip(pts, pts[1:])):
            continue                                     # not a function of x
        series.append((_colour(d.get("color")), pts))
    if not series:
        return None
    dec = ax["decimals"]
    # categories at the x positions of the vertices (labels under the plot), else the vertex x itself
    xs = sorted({round(p[0], 1) for _, pts in series for p in pts})
    base = max(s["bbox"][3] for s in ax["labels"])
    cats = []
    for x in xs:
        lab = _cat_label(labels, x, None, "v", ax["labels"], base - 2)
        cats.append(lab or f"x={x:.0f}")
    if sum(1 for c in cats if not c.startswith("x=")) < 0.5 * len(cats):
        return None
    names = [legend.get(col) or (f"Series {k + 1}" if len(series) > 1 else "Value") for k, (col, _) in enumerate(series)]
    rows = []
    for x, c in zip(xs, cats):
        row = [c]
        for _, pts in series:
            p = next((p for p in pts if abs(p[0] - x) < 0.6), None)
            row.append("" if p is None else _fmt(round(a * p[1] + b, dec + 1), dec))
        rows.append(row)
    used = [l["bbox"] for l in ax["labels"]] + [l["bbox"] for l in labels if l["text"].strip() in set(cats) | set(names)]
    return dict(chart_type="line", columns=["Category"] + names, rows=rows, used=used,
                axes=dict(value_axis="y", ticks=[s["text"] for s in ax["labels"]]),
                notes=[f"values measured at {len(xs)} vertices of {len(series)} line(s) against {len(ax['labels'])} axis tick labels"])
