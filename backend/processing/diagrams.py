"""Structure of diagrams drawn as vector graphics in digital PDFs: trees, graphs, flowcharts, arrays / stacks / queues,
linked cells. Read from the drawing itself, without a model, so the facts are exact:

    nodes   closed shapes - circles/ellipses, rectangles, rounded boxes, diamonds - with the text inside them as label
    edges   strokes (and thin filled strokes, as TeX/Illustrator often draw lines) whose two ends touch two nodes;
            a small filled triangle at an end gives the direction
    labels  text outside nodes: on an edge (weights, yes/no), above/below a cell (indices), or at the tail of an arrow
            that points at a node (pointers such as S.top = 4 or Q.head = 7)

and then interpreted:

    sequence      >= 3 boxes of one height side by side (an array, stack or queue): cells in order, their indices, cell
                  colours (e.g. occupied vs empty), pointers into it
    tree          connected, edges = nodes - 1, every edge going down: root, parent/children (left/right for binary
                  trees), leaves, height, pre-/in-/post-/level-order traversals, heap / binary-search-tree property
    flowchart     directed graph with a decision (diamond) or start/end terminals: steps from the start, branches and
                  their labels
    graph         anything else connected: vertices, edges (directed or not, weights), degrees

Nothing is guessed: a node without readable text is reported as unlabelled, an edge whose ends do not both touch a
node is not reported, and `notes` say what could not be read. describe() turns the structure into plain sentences
(used for search, for the Ask assistant's context, and for offline answers).
"""
import math
import re
from collections import defaultdict, deque

MIN_NODE, MAX_NODE = 6.0, 140.0


# ------------------------------------------------------------------------------------------- geometry helpers
def _pt(p):
    return (float(p.x), float(p.y)) if hasattr(p, "x") else (float(p[0]), float(p[1]))


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _bbox(pts):
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return (min(xs), min(ys), max(xs), max(ys))


def _area(poly):
    s = 0.0
    for (x0, y0), (x1, y1) in zip(poly, poly[1:] + poly[:1]):
        s += x0 * y1 - x1 * y0
    return abs(s) / 2


def _center(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def _inside(p, b, pad=0.0):
    return b[0] - pad <= p[0] <= b[2] + pad and b[1] - pad <= p[1] <= b[3] + pad


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def _node_dist(p, n):
    """Distance from a point to a node's outline (ellipses measured as ellipses)."""
    b = n["bbox"]
    if n["shape"] == "ellipse":
        cx, cy = _center(b); rx, ry = max(1e-3, (b[2] - b[0]) / 2), max(1e-3, (b[3] - b[1]) / 2)
        r = math.hypot((p[0] - cx) / rx, (p[1] - cy) / ry)
        return max(0.0, (r - 1.0) * min(rx, ry))
    dx = max(b[0] - p[0], 0.0, p[0] - b[2]); dy = max(b[1] - p[1], 0.0, p[1] - b[3])
    return math.hypot(dx, dy)


def _seg_dist(p, a, b):
    vx, vy = b[0] - a[0], b[1] - a[1]
    L = vx * vx + vy * vy
    t = 0.0 if L == 0 else max(0.0, min(1.0, ((p[0] - a[0]) * vx + (p[1] - a[1]) * vy) / L))
    return _dist(p, (a[0] + t * vx, a[1] + t * vy))


# ------------------------------------------------------------------------------------------- primitives
def _subpaths(d):
    """[(kind, points)] per subpath: kind 'poly' (lines), 'curve' (beziers), 'mixed', 'rect'."""
    out, cur, kinds = [], [], set()

    def flush():
        if cur:
            out.append(("curve" if kinds == {"c"} else "poly" if kinds == {"l"} else "mixed", list(cur)))

    for it in d.get("items", []):
        op = it[0]
        if op == "re":
            flush(); cur, kinds = [], set()
            r = it[1]
            out.append(("rect", [(r.x0, r.y0), (r.x1, r.y0), (r.x1, r.y1), (r.x0, r.y1)]))
            continue
        if op == "qu":
            flush(); cur, kinds = [], set()
            q = it[1]
            qp = [_pt(q.ul), _pt(q.ur), _pt(q.lr), _pt(q.ll)]
            if getattr(q, "is_rectangular", True):
                out.append(("rect", qp))
            else:                                   # a rotated quad (an unfilled diamond, a rhombus): keep its corners
                out.append(("poly", qp + [qp[0]]))
            continue
        pts = [_pt(p) for p in it[1:]]
        if cur and _dist(cur[-1], pts[0]) > 0.5:
            flush(); cur, kinds = [], set()
        if not cur:
            cur.append(pts[0])
        if op == "l":
            cur.append(pts[-1]); kinds.add("l")
        elif op == "c":
            cur.extend(pts[1:]); kinds.add("c")
    flush()
    return out


def _primitives(drawings, region, page_area):
    """Closed shapes, line segments and arrowheads of the drawings inside region."""
    shapes, segs, heads = [], [], []
    for di, d in enumerate(drawings):
        r = d.get("rect")
        if r is None or r.width * r.height > 0.5 * page_area:
            continue
        if not (r.x1 >= region[0] - 4 and r.x0 <= region[2] + 4 and r.y1 >= region[1] - 4 and r.y0 <= region[3] + 4):
            continue
        filled, stroked = d.get("fill") is not None, d.get("color") is not None and "s" in (d.get("type") or "")
        fill = tuple(round(c, 2) for c in d["fill"]) if d.get("fill") else None
        for kind, pts in _subpaths(d):
            if len(pts) < 2:
                continue
            bb = _bbox(pts); w, h = bb[2] - bb[0], bb[3] - bb[1]
            closed = _dist(pts[0], pts[-1]) < 1.0 or kind == "rect" or (filled and len(pts) >= 3)
            if kind == "rect":
                if MIN_NODE <= w <= MAX_NODE * 3 and MIN_NODE <= h <= MAX_NODE:
                    shapes.append(dict(shape="rect", bbox=bb, fill=fill))
                elif w < 2.5 or h < 2.5:                         # a rule drawn as a thin rectangle
                    segs.append(dict(pts=[(bb[0], (bb[1] + bb[3]) / 2), (bb[2], (bb[1] + bb[3]) / 2)] if w >= h else
                                     [((bb[0] + bb[2]) / 2, bb[1]), ((bb[0] + bb[2]) / 2, bb[3])], curved=False))
                continue
            if kind == "curve" and closed and MIN_NODE <= w <= MAX_NODE * 2 and MIN_NODE <= h <= MAX_NODE and 0.2 <= w / max(h, 1e-3) <= 5:
                shapes.append(dict(shape="ellipse", bbox=bb, fill=fill)); continue
            if kind == "mixed" and closed and MIN_NODE <= w <= MAX_NODE * 3 and MIN_NODE <= h <= MAX_NODE:
                shapes.append(dict(shape="rounded", bbox=bb, fill=fill)); continue
            if kind == "poly" and closed:
                poly = pts[:-1] if _dist(pts[0], pts[-1]) < 1.0 else pts
                A = _area(poly)
                far = max(((a, b) for i, a in enumerate(poly) for b in poly[i + 1:]), key=lambda t: _dist(*t), default=None)
                L = _dist(*far) if far else 0
                if len(poly) == 3 and max(w, h) <= 18 and not filled and A > 4:
                    # an outlined triangle at a line end: UML generalisation ('is a kind of')
                    tip = max(poly, key=lambda q: sum(_dist(q, o) for o in poly))
                    rest = [q for q in poly if q is not tip]
                    heads.append(dict(tip=tip, back=(sum(q[0] for q in rest) / 2, sum(q[1] for q in rest) / 2), kind="hollow", poly=list(poly))); continue
                if len(poly) in (3, 4) and max(w, h) <= 12 and filled and not (len(poly) == 4 and A > 0.6 * w * h):
                    # arrowhead (a triangle, or a notched 4-point head): the tip is the vertex farthest from the others
                    tip = max(poly, key=lambda q: sum(_dist(q, o) for o in poly))
                    rest = [q for q in poly if q is not tip]
                    heads.append(dict(tip=tip, back=(sum(q[0] for q in rest) / len(rest), sum(q[1] for q in rest) / len(rest)))); continue
                if L > 4 and A / L < 2.2:                         # a line drawn as a thin filled polygon
                    segs.append(dict(pts=list(far), curved=False)); continue
                if len(poly) == 4 and MIN_NODE <= w <= MAX_NODE * 3 and MIN_NODE <= h <= MAX_NODE:
                    axis = all(abs(a[0] - b[0]) < 0.8 or abs(a[1] - b[1]) < 0.8 for a, b in zip(poly, poly[1:] + poly[:1]))
                    cx, cy = _center(bb)
                    mids = all(abs(p[0] - cx) < 0.8 or abs(p[1] - cy) < 0.8 for p in poly)
                    if axis:
                        shapes.append(dict(shape="rect", bbox=bb, fill=fill)); continue
                    if mids:
                        shapes.append(dict(shape="diamond", bbox=bb, fill=fill)); continue
                if len(poly) >= 5 and MIN_NODE <= w <= MAX_NODE and MIN_NODE <= h <= MAX_NODE and A > 0.5 * w * h:
                    shapes.append(dict(shape="ellipse" if A < 0.85 * w * h else "rect", bbox=bb, fill=fill)); continue
            if not closed or stroked:
                # open strokes: straight pieces as segments; a curve or polyline as one connection end to end
                if kind == "poly":
                    segs.append(dict(pts=pts, curved=False))          # every vertex may touch a node (1 -> 2 -> 4)
                elif _dist(pts[0], pts[-1]) > 3:
                    mid = (((pts[0][0] + 3 * pts[1][0] + 3 * pts[2][0] + pts[3][0]) / 8, (pts[0][1] + 3 * pts[1][1] + 3 * pts[2][1] + pts[3][1]) / 8)
                           if len(pts) == 4 else pts[len(pts) // 2])
                    segs.append(dict(pts=[pts[0], pts[-1]], curved=True, mid=mid, drawing=di))
                elif kind in ("curve", "mixed") and len(pts) >= 4:      # a loop leaving and re-entering one node
                    apex = (((pts[0][0] + 3 * pts[1][0] + 3 * pts[2][0] + pts[3][0]) / 8, (pts[0][1] + 3 * pts[1][1] + 3 * pts[2][1] + pts[3][1]) / 8)
                            if len(pts) == 4 else max(pts, key=lambda q: _dist(q, pts[0])))
                    segs.append(dict(pts=[pts[0], pts[-1]], curved=True, mid=apex, loop=True))
    return shapes, _join_pieces(segs, shapes), heads


def _join_pieces(segs, shapes):
    """Some PDF producers write one curved stroke as several pieces in shuffled order (CLRS draws a self-loop as
    three beziers listed 2, 1, 3). Join curved pieces of the same drawing where one ends exactly where the next
    begins, only two piece ends meet there (not several edges at one node), and that joint is not on a node's
    outline (two edges that meet at a node stay two edges)."""
    pieces = [g for g in segs if g.get("drawing") is not None and g.get("curved") and not g.get("loop")]
    if len(pieces) < 2:
        return segs
    ends = [q for g in pieces for q in (g["pts"][0], g["pts"][-1])]

    def joint_ok(q):
        if sum(1 for e in ends if _dist(e, q) < 0.5) != 2:
            return False
        return all(_node_dist(q, dict(shape=sh["shape"], bbox=sh["bbox"])) >= 2.0 for sh in shapes)
    chains = [dict(g, pts=list(g["pts"]), via=[g["mid"]]) for g in pieces]
    merged = True
    while merged:
        merged = False
        for a in chains:
            for b in chains:
                if a is b or a["drawing"] != b["drawing"]:
                    continue
                if _dist(a["pts"][-1], b["pts"][0]) < 0.5 and joint_ok(a["pts"][-1]):
                    a["via"] += [a["pts"][-1]] + b["via"]
                    a["pts"] = [a["pts"][0], b["pts"][-1]]
                    chains.remove(b)
                    merged = True
                    break
            if merged:
                break
    out = [g for g in segs if not any(g is p for p in pieces)]
    for c in chains:
        via = c.pop("via")
        if len(via) > 1:                       # joined: the point farthest from the ends is the apex / label point
            c["mid"] = max(via, key=lambda q: min(_dist(q, c["pts"][0]), _dist(q, c["pts"][-1])))
            if _dist(c["pts"][0], c["pts"][-1]) <= 3:
                c["loop"] = True
        out.append(c)
    return out


def _dedupe(shapes):
    out = []
    def same(a, b):
        if _iou(a["bbox"], b["bbox"]) > 0.8:
            return True
        # a highlight ring drawn around a node (concentric, a little larger) is the same node
        wa, wb = a["bbox"][2] - a["bbox"][0], b["bbox"][2] - b["bbox"][0]
        return _dist(_center(a["bbox"]), _center(b["bbox"])) < 2.0 and 0.7 <= wa / max(wb, 1e-3) <= 1.43 and a["shape"] == b["shape"]
    for s in sorted(shapes, key=lambda s: -(s["fill"] is not None)):
        o = next((o for o in out if same(s, o)), None)
        if o is not None:
            dw = (s["bbox"][2] - s["bbox"][0]) - (o["bbox"][2] - o["bbox"][0])
            if s["fill"] is None and o.get("fill") is not None and dw < -2 and s["shape"] == "ellipse":
                o["double"] = True                # a second, smaller outline inside a filled circle: accepting state
            elif s["fill"] is None and o.get("fill") is not None:
                o["ring"] = True                  # outlined in a highlight colour
            elif s["fill"] is None and o.get("fill") is None and abs(dw) > 2:
                o["double"] = True                # two concentric outlines: an accepting state in an automaton
            continue
        out.append(s)
    # a frame around several nodes is not a node
    return [s for s in out if sum(1 for o in out if o is not s and _inside(_center(o["bbox"]), s["bbox"])
                                  and (o["bbox"][2] - o["bbox"][0]) < (s["bbox"][2] - s["bbox"][0])) < 2]


def _phrase(w0, free, gap=6.0):
    """The words on w0's line that touch it on either side (gaps under `gap` points): 'S.top = 4'."""
    line = sorted([w for w in free if abs(_center(w["bbox"])[1] - _center(w0["bbox"])[1]) < 3], key=lambda w: w["bbox"][0])
    k = line.index(w0)
    lo = hi = k
    while lo > 0 and line[lo]["bbox"][0] - line[lo - 1]["bbox"][2] < gap:
        lo -= 1
    while hi + 1 < len(line) and line[hi + 1]["bbox"][0] - line[hi]["bbox"][2] < gap:
        hi += 1
    return re.sub(r"\b([A-Za-z])\. (?=[a-z])", r"\1.", " ".join(w["text"] for w in line[lo:hi + 1]))   # 'S. top' -> 'S.top'


def page_words(page):
    """Words with boxes from ParseFlow's own text layer (broken font maps repaired: '=' not 'D')."""
    from . import textlayer
    segs, _ = textlayer.read_segments(page)
    out = []
    for sg in segs:
        if not sg.horizontal:
            # text set along a slanted edge (weights such as '12' rotated 45 degrees): one word per run, with the
            # box of its glyphs
            t = "".join(c.c for c in sg.chars).strip()
            if t and len(t) <= 20:
                cs = [c for c in sg.chars if c.c.strip()]
                out.append((min(c.x0 for c in cs), min(c.y0 for c in cs), max(c.x1 for c in cs), max(c.y1 for c in cs), t))
            continue
        cur = []
        for ch in list(sg.chars) + [None]:
            if ch is None or not ch.c.strip():
                if cur:
                    out.append((min(c.x0 for c in cur), min(c.y0 for c in cur), max(c.x1 for c in cur), max(c.y1 for c in cur),
                                "".join(c.c for c in cur)))
                cur = []
            else:
                if cur and (ch.x0 - cur[-1].x1 > 0.25 * max(ch.size, 1) or
                            abs(ch.size - cur[-1].size) > 0.2 * max(ch.size, cur[-1].size, 1)):   # a sub/superscript starts
                    out.append((min(c.x0 for c in cur), min(c.y0 for c in cur), max(c.x1 for c in cur), max(c.y1 for c in cur),
                                "".join(c.c for c in cur)))
                    cur = []
                cur.append(ch)
    return out


def _label(ws):
    """The text inside one shape, line by line. Smaller words set below / above a line are subscripts / superscripts
    of the word before them ('(a0, a4)' is written a_0, a_4; 'ω' with '2n' under it is ω_2n), and touching words are
    not split."""
    if not ws:
        return None
    if len(ws) == 1:
        return ws[0]["text"]
    H = max(w["bbox"][3] - w["bbox"][1] for w in ws)
    big = sorted((w for w in ws if w["bbox"][3] - w["bbox"][1] >= 0.8 * H), key=lambda w: (w["bbox"][1] + w["bbox"][3]) / 2)
    lines = []                                   # [centre y, [words]]
    for w in big:
        cy = (w["bbox"][1] + w["bbox"][3]) / 2
        if lines and abs(cy - lines[-1][0]) < 0.5 * H:
            lines[-1][1].append(w)
        else:
            lines.append([cy, [w]])
    for w in sorted((w for w in ws if w["bbox"][3] - w["bbox"][1] < 0.8 * H), key=lambda w: w["bbox"][0]):
        cy = (w["bbox"][1] + w["bbox"][3]) / 2
        # a small word written right after a word (touching it) is its sub/superscript; one standing under or over a
        # word (a size field under a key, as in CLRS's augmented trees) is a line of its own
        att = [L for L in lines if any(-1.0 <= w["bbox"][0] - o["bbox"][2] < 1.5 for o in L[1]) and abs(cy - L[0]) < 0.9 * H]
        if att:
            min(att, key=lambda L: abs(cy - L[0]))[1].append(w)
        else:
            same = [L for L in lines if abs(cy - L[0]) < 0.3 * H]
            if same:
                same[0][1].append(w)
            else:
                lines.append([cy, [w]])
    lines.sort(key=lambda L: L[0])
    out_lines = []
    for mid, lw in lines:
        out, prev = "", None
        for w in sorted(lw, key=lambda w: w["bbox"][0]):
            h = w["bbox"][3] - w["bbox"][1]
            cy = (w["bbox"][1] + w["bbox"][3]) / 2
            touching = prev is not None and w["bbox"][0] - prev["bbox"][2] < 1.2
            if h < 0.8 * H and prev is not None and touching and abs(cy - mid) > 0.12 * H:
                out += ("_" if cy > mid else "^") + w["text"]
            else:
                out += ("" if touching or not out else " ") + w["text"]
            prev = w
        out_lines.append(out)
    return " ".join(out_lines)


# ------------------------------------------------------------------------------------------- extraction
def extract(page, region, words=None):
    """Structure of the vector diagram inside region (x0, y0, x1, y1 in PDF points), or None when there is none."""
    try:
        drawings = page.get_drawings()
    except Exception:
        return None
    pr = page.rect
    shapes, segs, heads = _primitives(drawings, region, pr.width * pr.height)
    if len([s for s in shapes if _inside(_center(s["bbox"]), region, 6)]) < 2:
        return None
    if words is None:
        try:
            words = page_words(page)
        except Exception:
            words = page.get_text("words")
    return build(shapes, segs, heads, region, words)


def build(shapes, segs, heads, region, words, source="vector drawing (exact geometry)"):
    """Nodes, edges, labels and pointers from primitives (closed shapes, strokes, arrowheads) and words, all in PDF
    points; then the interpretation. Used for vector drawings and for diagrams read from images."""
    ws = [dict(text=w[4], bbox=(w[0], w[1], w[2], w[3])) for w in words if _inside(((w[0] + w[2]) / 2, (w[1] + w[3]) / 2), region, 30)]
    heads = list(heads)
    kept = []
    for sh in shapes:
        b = sh["bbox"]
        if sh["shape"] == "diamond" and max(b[2] - b[0], b[3] - b[1]) <= 16 and not any(_inside(_center(w["bbox"]), b) for w in ws):
            # a small empty diamond at a line end: UML aggregation (outlined) or composition (filled)
            dark = sh.get("fill") is not None and sum(sh["fill"]) / 3 < 0.5
            heads.append(dict(tip=_center(b), back=_center(b), kind="composition" if dark else "aggregation", box=b))
            continue
        kept.append(sh)
    nodes = _dedupe([s for s in kept if _inside(_center(s["bbox"]), region, 6)])
    if len(nodes) < 2:
        return None
    for n in nodes:
        b = n["bbox"]
        flat = [sg for sg in segs if not sg.get("curved") and len(sg["pts"]) == 2 and abs(sg["pts"][0][1] - sg["pts"][1][1]) < 1.0
                and all(_inside(q, b, 1) for q in sg["pts"])]
        if n["shape"] == "rect":
            # a box divided by full-width lines: a UML class (name | attributes | operations)
            cuts = sorted({round(sg["pts"][0][1], 1) for sg in flat if abs(sg["pts"][0][0] - sg["pts"][1][0]) >= 0.85 * (b[2] - b[0])
                           and b[1] + 3 < sg["pts"][0][1] < b[3] - 3})
            if cuts:
                n["cuts"] = cuts
        elif n["shape"] == "ellipse":
            # an underlined attribute name: the key of an entity in an ER diagram
            if any(abs(sg["pts"][0][0] - sg["pts"][1][0]) >= 6 for sg in flat):
                n["underlined"] = True
    for k, n in enumerate(nodes):
        n["id"] = k
        if n.pop("nil_px", False):
            n["nil"] = True                              # a NIL slash seen in an image
    # labels inside nodes (the smallest node containing the word)
    free = []
    inside = defaultdict(list)
    for w in ws:
        c = _center(w["bbox"])
        hit = [n for n in nodes if _inside(c, n["bbox"], -0.5)]
        if hit:
            inside[min(hit, key=lambda n: (n["bbox"][2] - n["bbox"][0]) * (n["bbox"][3] - n["bbox"][1]))["id"]].append(w)
        else:
            free.append(w)
    for n in nodes:
        n["label"] = _label(inside.get(n["id"], []))
        if n.get("cuts"):
            ys = [n["bbox"][1]] + n["cuts"] + [n["bbox"][3]]
            comp = []
            for a_, b_ in zip(ys, ys[1:]):
                part = [w for w in inside.get(n["id"], []) if a_ <= _center(w["bbox"])[1] < b_]
                lines = defaultdict(list)
                for w in part:
                    lines[round(_center(w["bbox"])[1] / 3)].append(w)
                comp.append([_label(v) for _, v in sorted(lines.items())])
            n["compartments"] = comp
            n["label"] = " ".join(comp[0]) if comp and comp[0] else n["label"]
    # a short tag written right next to a node (a vertex name such as s, t; an index; a black-height) belongs to it
    round_nodes = [n for n in nodes if n["shape"] == "ellipse" and (n["bbox"][3] - n["bbox"][1]) <= 40]
    keep = []
    for w in free:
        if round_nodes and len(w["text"]) <= 3:
            c = _center(w["bbox"])
            n = min(round_nodes, key=lambda m: _node_dist(c, m))
            if _node_dist(c, n) < min(0.55 * (n["bbox"][3] - n["bbox"][1]) + 2, 12) and n.get("tag") is None:
                n["tag"] = w["text"]; continue
        keep.append(w)
    free = keep
    # edges
    edges, pointers = [], []
    def head_at(p):
        for h in heads:
            if h.get("poly"):
                # an outlined triangle: the line meets the middle of one side; the opposite corner is the tip
                P = h["poly"]
                for k in range(3):
                    m = ((P[k][0] + P[(k + 1) % 3][0]) / 2, (P[k][1] + P[(k + 1) % 3][1]) / 2)
                    if _dist(m, p) < 4.5:
                        return dict(h, tip=P[(k + 2) % 3], back=m)
        hs = [h for h in heads if not h.get("poly") and (_dist(h["tip"], p) < 6 or _dist(h["back"], p) < 4.5 or (h.get("box") and _inside(p, h["box"], 2)))]
        return min(hs, key=lambda h: _dist(h["back"], p)) if hs else None

    tol = lambda n: max(3.5, 0.3 * min(n["bbox"][2] - n["bbox"][0], n["bbox"][3] - n["bbox"][1]))
    for sg in segs:                                     # a diagonal stroke across a box marks it NIL
        if len(sg["pts"]) == 2:
            a, b = sg["pts"]
            box = next((n for n in nodes if n["shape"] == "rect" and _inside(a, n["bbox"], 1) and _inside(b, n["bbox"], 1)), None)
            if box is not None and abs(a[0] - b[0]) > 2 and abs(a[1] - b[1]) > 2:
                box["nil"] = True
    for sg in segs:
        pts = list(sg["pts"])
        if len(pts) < 2 or _dist(pts[0], pts[-1]) < 3 and len(pts) == 2:
            continue
        hA, hB = head_at(pts[0]), head_at(pts[-1])
        if hA is not None:
            pts[0] = hA["tip"]                          # an arrow ends where its head points
        if hB is not None:
            pts[-1] = hB["tip"]
        hits = []                                       # (vertex index, node) for every vertex that touches a node
        for k, p in enumerate(pts):
            n = min(nodes, key=lambda m: _node_dist(p, m))
            extra = 4 if (k == 0 and hA is not None) or (k == len(pts) - 1 and hB is not None) else 0
            if _node_dist(p, n) <= tol(n) + extra:
                if not hits or hits[-1][1] is not n:
                    hits.append((k, n))
        if sg.get("curved") and len(pts) == 2:
            na_ = min(nodes, key=lambda m: _node_dist(pts[0], m)); nb_ = min(nodes, key=lambda m: _node_dist(pts[1], m))
            if na_ is nb_ and _node_dist(pts[0], na_) <= tol(na_) + 4 and _node_dist(pts[1], nb_) <= tol(nb_) + 4 and \
                    not _inside(sg["mid"], na_["bbox"]):
                edges.append(dict(a=na_["id"], b=na_["id"], directed=True, p=(pts[0], pts[1]), mid=sg["mid"], curved=True, loop=True))
                continue
        for (k1, n1), (k2, n2) in zip(hits, hits[1:]):
            a, b = pts[k1], pts[k2]
            mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
            if _inside(mid, n1["bbox"]) and _inside(mid, n2["bbox"]):
                continue
            to_b = hB is not None and k2 == len(pts) - 1
            to_a = hA is not None and k1 == 0
            directed = to_a != to_b
            src, dst = (n2, n1) if (to_a and not to_b) else (n1, n2)
            kind = (hB if (to_b and not to_a) else hA if (to_a and not to_b) else None) or {}
            edges.append(dict(a=src["id"], b=dst["id"], directed=directed, p=(a, b), curved=sg["curved"], mid=sg.get("mid"),
                              head=kind.get("kind", "arrow") if directed else None))
        if len(hits) <= 1:
            # an arrow pointing at a node from free text: a pointer such as 'S.top = 4'
            for head, end, other in ((hA, 0, len(pts) - 1), (hB, len(pts) - 1, 0)):
                if head is None or not hits or hits[0][0] != end:
                    continue
                tail = pts[other]
                near = sorted(free, key=lambda w: _node_dist(tail, dict(shape="rect", bbox=w["bbox"])))
                if near and _node_dist(tail, dict(shape="rect", bbox=near[0]["bbox"])) < 14:
                    pointers.append(dict(text=_phrase(near[0], free), node=hits[0][1]["id"]))
                elif min(_node_dist(tail, m) for m in nodes) > 10:
                    hits[0][1]["entry"] = True            # an arrow from nowhere: the start state of an automaton
    # one edge per pair (a filled line and its outline are drawn twice)
    seen, uniq = set(), []
    for e in edges:
        k = (e["a"], e["b"], round(e["mid"][0]), round(e["mid"][1])) if e.get("mid") else \
            (e["a"], e["b"]) if e["directed"] else tuple(sorted((e["a"], e["b"])))
        if k not in seen:
            seen.add(k); uniq.append(e)
    edges = uniq
    # edge labels: short phrases (words touching on one line) closest to the middle part of one edge (weights, yes/no,
    # 'I/O done')
    phrases, taken = [], set()
    for w in sorted(free, key=lambda w: (round(w["bbox"][1]), w["bbox"][0])):
        if id(w) in taken:
            continue
        line = sorted([x for x in free if abs(_center(x["bbox"])[1] - _center(w["bbox"])[1]) < 3 and id(x) not in taken], key=lambda x: x["bbox"][0])
        grp = [w]
        for x in line:
            if x["bbox"][0] > grp[-1]["bbox"][2] and x["bbox"][0] - grp[-1]["bbox"][2] < 5:
                grp.append(x)
        taken.update(id(x) for x in grp)
        phrases.append(dict(text=" ".join(x["text"] for x in grp), bbox=(min(x["bbox"][0] for x in grp), min(x["bbox"][1] for x in grp),
                                                                       max(x["bbox"][2] for x in grp), max(x["bbox"][3] for x in grp))))
    # a label set on two lines ('Evaluation' over 'Time Θ(n lg n)', 'I/O' over 'done') is one phrase: short pieces
    # stacked directly above each other and centred on each other
    phrases.sort(key=lambda p: (p["bbox"][1], p["bbox"][0]))
    stacked, gone = [], set()
    for i, a in enumerate(phrases):
        if i in gone:
            continue
        cur = dict(a)
        for j in range(i + 1, len(phrases)):
            b = phrases[j]
            if j in gone:
                continue
            A, B = cur["bbox"], b["bbox"]
            h = max(A[3] - A[1], B[3] - B[1], 1)
            gap = B[1] - A[3]
            ov = min(A[2], B[2]) - max(A[0], B[0])
            centred = abs((A[0] + A[2]) / 2 - (B[0] + B[2]) / 2) < 0.35 * max(A[2] - A[0], B[2] - B[0])
            numeric = re.fullmatch(r"[−–-]?[\d.∞]+", cur["text"]) and re.fullmatch(r"[−–-]?[\d.∞]+", b["text"])
            short = min(len(cur["text"].strip()), len(b["text"].strip())) <= 2   # stacked symbols ('a' over 'b') are separate labels
            if -0.2 * h <= gap <= 0.45 * h and (ov > 0.5 * min(A[2] - A[0], B[2] - B[0]) or centred) and not numeric and not short \
                    and len(cur["text"]) + len(b["text"]) <= 40:
                cur = dict(text=cur["text"] + " " + b["text"], bbox=(min(A[0], B[0]), A[1], max(A[2], B[2]), B[3]), lines=cur.get("lines", 1) + 1)
                gone.add(j)
        stacked.append(cur)
    phrases = stacked
    cands = []
    for w in phrases:
        if len(w["text"]) > (40 if w.get("lines", 1) > 1 else 16) or re.fullmatch(r"\(?[a-z]\)", w["text"]):
            continue
        c = _center(w["bbox"])
        ds = []
        for e in edges:
            a, b = e["p"]
            m = e.get("mid")
            d = min(_seg_dist(c, a, m), _seg_dist(c, m, b)) if m else _seg_dist(c, a, b)
            if e.get("loop"):
                d = _dist(c, m) - 3
            if _dist(c, a) > 6 and _dist(c, b) > 6:
                ds.append((d, id(e), e))
        ds.sort(key=lambda t: t[0])
        for k, (d, _, e) in enumerate(ds[:3]):
            nxt = ds[k + 1][0] if k + 1 < len(ds) else None
            if d < 12 or (d < 18 and (nxt is None or nxt > 1.8 * d)):
                cands.append((d, k, id(w), w, e))
    # one phrase per edge, closest pairs first: a label already claimed by a nearer edge is not added to another
    # edge, and a second phrase does not pile onto an edge that has its label ('a a')
    done_w, done_e = set(), set()
    for d, k, wid, w, e in sorted(cands, key=lambda t: (t[1], t[0])):
        if wid in done_w or id(e) in done_e:
            continue
        e["label"] = w["text"]
        done_w.add(wid); done_e.add(id(e))
    # second pass, for edges still without a label: a phrase further away (up to 30 pt) that is clearly nearer to this
    # edge than to any other edge or node, and that no other edge took
    for w in phrases:
        if id(w) in done_w or len(w["text"]) > 16 or re.fullmatch(r"\(?[a-z]\)", w["text"]):
            continue
        c = _center(w["bbox"])
        ds = []
        for e in edges:
            a, b = e["p"]
            m = e.get("mid")
            d = min(_seg_dist(c, a, m), _seg_dist(c, m, b)) if m else _seg_dist(c, a, b)
            ds.append((d, e))
        ds.sort(key=lambda t: t[0])
        if not ds or ds[0][0] > 30 or id(ds[0][1]) in done_e:
            continue
        near_node = min((_node_dist(c, n) for n in nodes), default=99)
        if (len(ds) == 1 or ds[1][0] > 1.6 * ds[0][0]) and near_node > 0.8 * ds[0][0]:
            e = ds[0][1]
            e["label"] = w["text"]; e["label_far"] = True
            done_w.add(id(w)); done_e.add(id(e))
    st = _interpret(nodes, _merge_parallel(edges), pointers, free)
    if st:
        st["source"] = source
    return st


def _merge_parallel(edges):
    """A highlighted edge is often drawn twice (the edge, then a thick coloured stroke over it), and the copy may lose
    its arrowhead or its label. Between the same two nodes: a plain line next to an arrow is the same edge (its
    label moves to the arrow), and an unlabelled arrow next to a labelled one in the same direction is a copy.
    Two arrows in opposite directions, and several plain lines (a multigraph), are kept."""
    pair = defaultdict(list)
    for e in edges:
        if e["a"] != e["b"]:
            pair[frozenset((e["a"], e["b"]))].append(e)
    drop = set()
    for g in pair.values():
        arrows = [e for e in g if e["directed"]]
        if not arrows or len(g) < 2:
            continue
        for u in [e for e in g if not e["directed"]]:
            tgt = next((d for d in arrows if not d.get("label")), None) if u.get("label") else None
            if tgt is not None:
                tgt["label"] = u["label"]
            elif u.get("label") and all(d.get("label") for d in arrows):
                continue                                  # a labelled line beside labelled arrows: keep it
            drop.add(id(u))
        for d in arrows:
            if not d.get("label") and any(o is not d and o["a"] == d["a"] and o["b"] == d["b"] and o.get("label") for o in arrows):
                drop.add(id(d))
    return [e for e in edges if id(e) not in drop]


# ------------------------------------------------------------------------------------------- interpretation
def _runs(nodes, used, vertical, lo, hi):
    """Runs of lo..hi boxes of one size touching side by side (rows, or columns when vertical)."""
    out = []
    boxes = [n for n in nodes if n["shape"] in ("rect", "rounded") and n["id"] not in used]
    groups = defaultdict(list)
    for n in boxes:
        c = _center(n["bbox"])
        groups[round((c[0] if vertical else c[1]) / 3)].append(n)
    size = lambda m: (m["bbox"][2] - m["bbox"][0]) if vertical else (m["bbox"][3] - m["bbox"][1])
    for g in groups.values():
        g.sort(key=lambda n: n["bbox"][1] if vertical else n["bbox"][0])
        run = [g[0]] if g else []
        for n in g[1:] + [None]:
            p = run[-1] if run else None
            ok = n is not None and p is not None
            if ok:
                gap = (n["bbox"][1] - p["bbox"][3]) if vertical else (n["bbox"][0] - p["bbox"][2])
                ok = -1.5 <= gap <= (1.5 if hi <= 3 else 3.0) and abs(size(n) - size(p)) <= 0.2 * max(size(p), 1)
            if ok:
                run.append(n); continue
            if lo <= len(run) <= hi:
                out.append(dict(vertical=vertical, cells=[m["id"] for m in run]))
                used.update(m["id"] for m in run)
            run = [n] if n is not None else []
    return out


def _sequences(nodes, used, edges=()):
    """Arrays (rows of >= 3 boxes), then records (2-3 boxes side by side: a linked-list node such as [key | next]),
    then columns of >= 3 boxes (an array drawn vertically, e.g. the Adj array of adjacency lists). A row of exactly
    three boxes joined by arrows to other small rows is a record ([prev | key | next] of a doubly linked list)."""
    rows = _runs(nodes, used, False, 3, 10 ** 6)
    small = [r for r in rows if len(r["cells"]) == 3]
    where = {i: k for k, r in enumerate(small) for i in r["cells"]}
    linked = set()
    for e in edges:
        ka, kb = where.get(e["a"]), where.get(e["b"])
        if ka is not None and kb is not None and ka != kb:
            linked |= {ka, kb}
    recs3 = [small[k] for k in sorted(linked)]
    rows = [r for r in rows if not any(r is x for x in recs3)]
    recs = recs3 + _runs(nodes, used, False, 2, 3)
    cols = _runs(nodes, used, True, 3, 10 ** 6)
    return rows + cols, recs


def _annotate_cells(seq, nodes, free):
    """Index numbers or text right above/below (or beside, for columns) each cell."""
    by = {n["id"]: n for n in nodes}
    cells = [by[i] for i in seq["cells"]]
    marks = []
    for c in cells:
        b = c["bbox"]
        if seq["vertical"]:
            near = [w for w in free if abs(_center(w["bbox"])[1] - _center(b)[1]) < (b[3] - b[1]) / 2 and 0 < b[0] - w["bbox"][2] < 14]
        else:
            near = [w for w in free if abs(_center(w["bbox"])[0] - _center(b)[0]) < (b[2] - b[0]) / 2 and 0 < b[1] - w["bbox"][3] < 12]
        marks.append(" ".join(w["text"] for w in near) or None)
    seq["indices"] = marks if sum(1 for m in marks if m) >= 0.6 * len(marks) else None
    if not seq["vertical"]:
        # text under the row: under a cell's middle (a label) or under a cell boundary (a time axis, as in a Gantt chart)
        row_y = max(c["bbox"][3] for c in cells)
        under = [w for w in free if 0 <= w["bbox"][1] - row_y < 14 and cells[0]["bbox"][0] - 8 <= _center(w["bbox"])[0] <= cells[-1]["bbox"][2] + 8]
        edges_x = [cells[0]["bbox"][0]] + [c["bbox"][2] for c in cells]
        bounds = [None] * len(edges_x)
        below = [None] * len(cells)
        for w in under:
            cx = _center(w["bbox"])[0]
            k = min(range(len(edges_x)), key=lambda i: abs(edges_x[i] - cx))
            if abs(edges_x[k] - cx) <= 4:
                bounds[k] = w["text"]
            else:
                j = next((j for j, c in enumerate(cells) if c["bbox"][0] <= cx <= c["bbox"][2]), None)
                if j is not None:
                    below[j] = ((below[j] + " ") if below[j] else "") + w["text"]
        if sum(1 for b in bounds if b) >= max(2, 0.6 * len(bounds)):
            seq["boundaries"] = bounds
        if sum(1 for b in below if b) >= 0.6 * len(below):
            seq["below"] = below
    # a name written just before the row ('S', 'Q', 'A')
    b0 = cells[0]["bbox"]
    name = [w for w in free if abs(_center(w["bbox"])[1] - _center(b0)[1]) < 6 and 0 < b0[0] - w["bbox"][2] < 16 and len(w["text"]) <= 6] \
        if not seq["vertical"] else []
    if name:
        w0 = name[-1]
        # with the letter it is attached to ('c' + subscript '0' -> c_0)
        group = [w for w in free if 0 <= w0["bbox"][0] - w["bbox"][2] < 1.5 and abs(_center(w["bbox"])[1] - _center(w0["bbox"])[1]) < 8] + [w0]
        seq["name"] = _label(group)
    else:
        seq["name"] = None
    fills = [c["fill"] for c in cells]
    if len(set(fills)) > 1:
        groups, cur = [], None
        for k, f in enumerate(fills):
            if cur and cur["fill"] == f:
                cur["to"] = k
            else:
                cur = dict(fill=f, frm=k, to=k); groups.append(cur)
        seq["fill_groups"] = [dict(fill=_colour(g["fill"]), cells=(g["frm"], g["to"])) for g in groups]
    return seq


def _colour(f):
    if f is None:
        return "no fill"
    r, g, b = f[:3]
    if r > 0.9 and g > 0.9 and b > 0.9:
        return "white"
    if r > 0.85 and g > 0.75 and b < 0.8:
        return "tan/yellow"
    if b > 0.75 and r < 0.8:
        return "blue"
    if g > r and g > b:
        return "green"
    if r > g and r > b:
        return "red/orange" if g > 0.3 else "red"
    if r < 0.3 and g < 0.3 and b < 0.3:
        return "black/dark"
    return "grey" if abs(r - g) < 0.1 and abs(g - b) < 0.1 else f"rgb({r:.2f},{g:.2f},{b:.2f})"


def _num(s):
    try:
        return float(str(s).replace("−", "-"))
    except (TypeError, ValueError):
        return None


def _tree(comp, nodes, edges):
    by = {n["id"]: n for n in nodes}
    es = [e for e in edges if e["a"] in comp and e["b"] in comp]
    if len(es) != len(comp) - 1:
        return None
    parent, kids, elab = {}, defaultdict(list), {}
    for e in es:
        A, B = by[e["a"]], by[e["b"]]
        if e["directed"]:
            p, c = A, B
        else:
            p, c = (A, B) if _center(A["bbox"])[1] < _center(B["bbox"])[1] - 2 else (B, A) if _center(B["bbox"])[1] < _center(A["bbox"])[1] - 2 else (None, None)
        if p is None or c["id"] in parent:
            return None
        parent[c["id"]] = p["id"]; kids[p["id"]].append(c["id"])
        if e.get("label"):
            elab[(p["id"], c["id"])] = e["label"]          # 0/1 on a Huffman or trie edge, a weight
    roots = [i for i in comp if i not in parent]
    if len(roots) != 1:
        return None
    root = roots[0]
    for k in kids:
        kids[k].sort(key=lambda i: _center(by[i]["bbox"])[0])
    # a chain of arrows (s -> x -> y) is a path, not a binary tree with sides; a chain of plain lines drawn as a
    # tree (a degenerate BST) keeps its left/right children
    binary = all(len(v) <= 2 for v in kids.values()) and (any(len(v) == 2 for v in kids.values()) or not any(e["directed"] for e in es))
    lab = lambda i: by[i]["label"] if by[i]["label"] is not None else f"(node {i}, unlabelled)"
    if binary:
        # an only child straight below its parent has no side: then this is not drawn as a binary tree
        for p_, ch in kids.items():
            if len(ch) == 1:
                w = by[p_]["bbox"][2] - by[p_]["bbox"][0]
                if abs(_center(by[ch[0]]["bbox"])[0] - _center(by[p_]["bbox"])[0]) < 0.3 * w:
                    binary = False
    side = {}
    if binary:
        for p, ch in kids.items():
            if len(ch) == 2:
                side[ch[0]], side[ch[1]] = "left", "right"
            else:
                side[ch[0]] = "left" if _center(by[ch[0]]["bbox"])[0] < _center(by[p]["bbox"])[0] else "right"

    def pre(i):
        return [i] + [x for c in kids.get(i, []) for x in pre(c)]

    def post(i):
        return [x for c in kids.get(i, []) for x in post(c)] + [i]

    def ino(i):
        ch = kids.get(i, [])
        L = [c for c in ch if side.get(c) == "left"]; R = [c for c in ch if side.get(c) == "right"]
        return [x for c in L for x in ino(c)] + [i] + [x for c in R for x in ino(c)]
    level, q, depth = [], deque([root]), {root: 0}
    while q:
        i = q.popleft(); level.append(i)
        for c in kids.get(i, []):
            depth[c] = depth[i] + 1; q.append(c)
    level_pos = lambda i: level.index(i)
    t = dict(type="binary tree" if binary else "tree", root=lab(root), nodes=len(comp), height=max(depth.values()),
             leaves=[lab(i) for i in level if not kids.get(i)],
             children=[[lab(p), [(side.get(c) + ": " if binary else "") + lab(c) + (f" (edge label {elab[(p, c)]})" if elab.get((p, c)) else "")
                                 for c in ch]] for p, ch in sorted(kids.items(), key=lambda t: level_pos(t[0]))],
             preorder=[lab(i) for i in pre(root)], postorder=[lab(i) for i in post(root)], level_order=[lab(i) for i in level])
    if binary:
        t["inorder"] = [lab(i) for i in ino(root)]
    vals = {i: _num(by[i]["label"]) for i in comp}
    if all(v is not None for v in vals.values()):
        props = []
        if all(vals[parent[c]] >= vals[c] for c in parent):
            props.append("max-heap property holds (every parent ≥ its children)")
        if all(vals[parent[c]] <= vals[c] for c in parent):
            props.append("min-heap property holds (every parent ≤ its children)")
        if binary:
            seq = [vals[i] for i in ino(root)]
            if all(a <= b for a, b in zip(seq, seq[1:])):
                props.append("binary-search-tree property holds (in-order keys are sorted)")
        t["properties"] = props or ["neither heap nor binary-search-tree order holds for the keys as read"]
    # B-tree: nodes holding several keys, each internal node with one child more than it has keys
    keys = {i: (by[i]["label"] or "").split() for i in comp}
    if any(len(k) >= 2 for k in keys.values()) and all(keys.values()):
        internal = [i for i in comp if kids.get(i)]
        if internal and all(len(kids[i]) == len(keys[i]) + 1 for i in internal):
            t["type"] = "B-tree"
            t.pop("inorder", None)
            key = lambda x: (0, _num(x)) if _num(x) is not None else (1, x.lower())
            bad = []
            for i in comp:
                ks = keys[i]
                if any(key(a) > key(b) for a, b in zip(ks, ks[1:])):
                    bad.append(f"keys of node [{' '.join(ks)}] are not in order")
            def subtree(i):
                return [k for j in pre(i) for k in keys[j]]
            for i in internal:
                ks = keys[i]
                for ci, c in enumerate(kids[i]):
                    lo = ks[ci - 1] if ci > 0 else None
                    hi = ks[ci] if ci < len(ks) else None
                    for k in subtree(c):
                        if (lo is not None and key(k) < key(lo)) or (hi is not None and key(k) > key(hi)):
                            bad.append(f"key {k} under [{' '.join(ks)}] (child {ci + 1}) is outside {lo or '−∞'}…{hi or '+∞'}")
                            break
            leaf_depths = {depth[i] for i in comp if not kids.get(i)}
            if len(leaf_depths) > 1:
                bad.append("leaves are at different depths")
            t["btree"] = dict(nodes=[[lab(i), keys[i]] for i in level], max_keys=max(len(k) for k in keys.values()),
                              min_keys=min(len(keys[i]) for i in comp if i != root),
                              in_order_keys=[k for i in _btree_inorder(root, kids, keys) for k in [i]],
                              property=("B-tree order holds (keys sorted in every node, each subtree between its separating keys, all leaves at one depth)"
                                        if not bad else "B-tree order does not hold: " + "; ".join(bad[:4])))
    # labelled edges along root-to-leaf paths spell something: trie words, Huffman / prefix codes
    if elab and all(len(str(v)) <= 3 for v in elab.values()):
        paths = []
        def walk(i, acc):
            if not kids.get(i):
                paths.append((lab(i), acc))
            for c in kids.get(i, []):
                walk(c, acc + str(elab.get((i, c), "?")))
        walk(root, "")
        if all(set(v) <= {"0", "1"} for v in elab.values()):
            t["codes"] = [f"{leaf}: {code}" for leaf, code in paths]
        else:
            t["paths"] = [f"{code} → {leaf}" for leaf, code in paths]
    return t


def _class_diagram(comp, es, by, lab):
    classes = []
    for i in sorted(comp, key=lambda i: (_center(by[i]["bbox"])[1], _center(by[i]["bbox"])[0])):
        c = by[i].get("compartments") or [[by[i]["label"] or ""]]
        classes.append(dict(name=" ".join(c[0]) or lab(i), attributes=c[1] if len(c) > 1 else [], operations=c[2] if len(c) > 2 else []))
    name = lambda i: " ".join((by[i].get("compartments") or [[lab(i)]])[0]) or lab(i)
    rel = []
    for e in es:
        a, b, h = name(e["a"]), name(e["b"]), e.get("head")
        if h == "hollow":
            rel.append(f"{a} is a kind of {b} (generalisation)")
        elif h == "composition":
            rel.append(f"{b} is composed of {a} (composition: the part belongs to one whole)")
        elif h == "aggregation":
            rel.append(f"{b} has {a} (aggregation)")
        elif e["directed"]:
            rel.append(f"{a} → {b} (association / dependency)" + (f", labelled {e['label']}" if e.get("label") else ""))
        else:
            rel.append(f"{a} — {b} (association)" + (f", labelled {e['label']}" if e.get("label") else ""))
    return dict(type="class diagram", classes=classes, relationships=rel)


def _er_diagram(comp, es, by, lab):
    shp = lambda i: by[i]["shape"]
    nbr = defaultdict(list)
    for e in es:
        nbr[e["a"]].append((e["b"], e.get("label"))); nbr[e["b"]].append((e["a"], e.get("label")))
    ents = [i for i in comp if shp(i) == "rect"]
    rels = [i for i in comp if shp(i) == "diamond"]
    wordy = lambda i: bool(re.search(r"[A-Za-z]{2}", by[i]["label"] or ""))
    # entities and relationships are named with words; boxes and diamonds holding numbers or nothing are something else
    if not ents or not rels or not any(wordy(i) for i in ents) or not any(wordy(i) for i in rels):
        return None
    def attrs(i):
        out = []
        for j, _ in nbr[i]:
            if shp(j) == "ellipse":
                a = lab(j) + (" (key)" if by[j].get("underlined") else "") + (" (multi-valued)" if by[j].get("double") else "")
                sub = [lab(k) for k, _ in nbr[j] if k != i and shp(k) == "ellipse"]
                out.append(a + (f" made of {', '.join(sub)}" if sub else ""))
        return out
    entities = [dict(name=lab(i), weak=bool(by[i].get("double")), attributes=attrs(i)) for i in ents]
    relationships = []
    for r in rels:
        sides = [f"{lab(j)}" + (f" ({c})" if c else "") for j, c in nbr[r] if shp(j) == "rect"]
        relationships.append(dict(name=lab(r), identifying=bool(by[r].get("double")), connects=sides, attributes=attrs(r)))
    return dict(type="ER diagram", entities=entities, relationships=relationships)


def _btree_inorder(i, kids, keys):
    ch = kids.get(i, [])
    out = []
    for ci, k in enumerate(keys[i]):
        if ci < len(ch):
            out += _btree_inorder(ch[ci], kids, keys)
        out.append(k)
    if len(ch) > len(keys[i]):
        out += _btree_inorder(ch[-1], kids, keys)
    return out


def _colours(comp, by):
    """Node fill colours when they differ (red-black trees, highlighted vertices): {colour: [labels]}."""
    g = defaultdict(list)
    for i in sorted(comp, key=lambda i: (_center(by[i]["bbox"])[1], _center(by[i]["bbox"])[0])):
        g[_colour(by[i]["fill"])].append(by[i]["label"] if by[i]["label"] is not None else f"(node {i})")
    return dict(g) if len(g) > 1 else None


def _overlapping(comp, by):
    """True when shapes of a component overlap each other substantially: graph and tree nodes never do."""
    ids = list(comp)
    for k, a in enumerate(ids):
        A = by[a]["bbox"]
        for b in ids[k + 1:]:
            B = by[b]["bbox"]
            ix = max(0.0, min(A[2], B[2]) - max(A[0], B[0])) * max(0.0, min(A[3], B[3]) - max(A[1], B[1]))
            small = min((A[2] - A[0]) * (A[3] - A[1]), (B[2] - B[0]) * (B[3] - B[1]))
            if small > 0 and ix > 0.15 * small:
                return True
    return False


def _interpret(nodes, edges, pointers, free):
    by = {n["id"]: n for n in nodes}
    blank = overlap = 0
    used = set()
    parts = []
    # arrays / stacks / queues, and records (multi-field boxes of linked structures)
    seqs, recs = _sequences(nodes, used, edges)
    rec_of = {}
    for k, r in enumerate(recs):
        for i in r["cells"]:
            rec_of[i] = k
    cell_of = {}
    for k, sq in enumerate(seqs):
        for j, i in enumerate(sq["cells"]):
            cell_of[i] = (k, j)
    for seq in seqs:
        _annotate_cells(seq, nodes, free)
        cellset = set(seq["cells"])
        ptr = [dict(text=p["text"], cell=seq["cells"].index(p["node"]) + 1) for p in pointers if p["node"] in cellset]
        links = [e for e in edges if e["a"] in cellset and e["b"] in cellset]
        part = dict(type="sequence", orientation="column" if seq["vertical"] else "row", name=seq["name"],
                    cells=[by[i]["label"] for i in seq["cells"]], indices=seq["indices"], pointers=ptr)
        if seq.get("boundaries"):
            part["boundaries"] = seq["boundaries"]
        if seq.get("below"):
            part["below"] = seq["below"]
        if seq.get("fill_groups"):
            part["fill_groups"] = [dict(fill=g["fill"], cells=f"{g['cells'][0] + 1}–{g['cells'][1] + 1}") for g in seq["fill_groups"]]
        if links:
            part["links_between_cells"] = len(links)
        parts.append(part)
    # linked lists: arrows from a record (or an array cell) to the next record
    if recs:
        def rlabel(k):
            f = []
            for i in recs[k]["cells"]:
                n = by[i]
                f.append("NIL" if n.get("nil") else (n["label"] if n["label"] is not None else "·"))
            return "[" + " | ".join(f) + "]"
        outs, has_in = defaultdict(list), set()
        starts = []
        for e in edges:
            if e["b"] not in rec_of or not e["directed"]:
                continue
            tgt = rec_of[e["b"]]
            if e["a"] in rec_of and rec_of[e["a"]] != tgt:
                if tgt not in outs[rec_of[e["a"]]]:
                    outs[rec_of[e["a"]]].append(tgt)
                has_in.add(tgt)
            elif e["a"] in cell_of:
                k, j = cell_of[e["a"]]
                starts.append((f"{seqs[k].get('name') or 'array'} cell {j + 1}" + (f" ({seqs[k]['indices'][j]})" if seqs[k].get("indices") and seqs[k]["indices"][j] else ""), tgt))
                has_in.add(tgt)
            elif e["a"] not in rec_of:
                starts.append((by[e["a"]]["label"] or "a node", tgt)); has_in.add(tgt)
        for p in pointers:
            if p["node"] in rec_of:
                starts.append((p["text"], rec_of[p["node"]])); has_in.add(rec_of[p["node"]])
        starts += [(None, k) for k in range(len(recs)) if k not in has_in]
        if not starts and recs:                 # every node has an incoming arrow (circular or doubly linked)
            starts = [(None, min(range(len(recs)), key=lambda k: (_center(by[recs[k]["cells"][0]]["bbox"])[0])))]
        lists = []
        for frm, k in starts:
            chain, vis, prev, cycle, back = [], set(), None, False, 0
            while k is not None and k not in vis:
                vis.add(k); chain.append(rlabel(k))
                if prev is not None and prev in outs[k]:
                    back += 1                       # this node also points back to the one before it
                cand = [t for t in outs[k] if t != prev]
                nk = next((t for t in cand if t not in vis), None)
                if nk is None and any(t in vis for t in cand):
                    cycle = True
                prev, k = k, nk
            lists.append(dict(start=frm, nodes=chain, cycle=cycle,
                              doubly=len(chain) > 1 and back == len(chain) - 1))
        blank_lists = [L for L in lists if not L["start"] and all(re.fullmatch(r"\[[·|\s]*\]", x) for x in L["nodes"])]
        lists = [L for L in lists if L not in blank_lists]
        blank += len(blank_lists)               # boxes with nothing in them (shaded areas, decorations)
        if lists:
            parts.append(dict(type="linked lists", lists=lists))
        used.update(rec_of)
    # connected components of the remaining nodes
    adj = defaultdict(set)
    for e in edges:
        if e["a"] in used and e["b"] in used:
            continue
        adj[e["a"]].add(e["b"]); adj[e["b"]].add(e["a"])
    seen, blank_ids = set(), set()
    for n in nodes:
        if n["id"] in seen or n["id"] in used or not adj.get(n["id"]):
            continue
        comp, q = set(), [n["id"]]
        while q:
            i = q.pop()
            if i in comp:
                continue
            comp.add(i); q.extend(adj[i] - comp)
        seen |= comp
        if len(comp) < 2:
            continue
        es = [e for e in edges if e["a"] in comp and e["b"] in comp]
        if all(by[i]["label"] is None and not by[i].get("tag") for i in comp):
            blank += 1                          # no text anywhere (gate symbols, decorations): nothing to interpret
            blank_ids |= comp
            continue
        if _overlapping(comp, by):
            overlap += 1                        # shapes lying over each other (a Venn / set diagram), not nodes
            blank_ids |= comp
            continue
        named = sum(1 for i in comp if by[i].get("tag") and not re.fullmatch(r"\d+", by[i]["tag"])) >= 0.6 * len(comp)

        def lab(i):
            v = by[i]["label"] if by[i]["label"] is not None else None
            if named and by[i].get("tag"):
                return f"{by[i]['tag']} ({v})" if v is not None else by[i]["tag"]
            return v if v is not None else f"(node {i}, unlabelled)"
        shapes = {by[i]["shape"] for i in comp}
        directed = any(e["directed"] for e in es)
        terminals = [i for i in comp if re.fullmatch(r"(?i)(start|begin|end|stop|exit|return.*)", by[i]["label"] or "")]
        if any(by[i].get("compartments") for i in comp) or any(e.get("head") in ("hollow", "aggregation", "composition") for e in es):
            parts.append(_class_diagram(comp, es, by, lab))
            continue
        if not directed and "diamond" in shapes and "rect" in shapes:
            er = _er_diagram(comp, es, by, lab)
            if er:
                parts.append(er)
                continue
        labelled = sum(1 for e in es if (e.get("label") or "").strip())
        if directed and any(by[i].get("double") or by[i].get("entry") for i in comp) and shapes <= {"ellipse"} \
                and labelled >= max(1, 0.5 * len(es)):                 # a state diagram labels its transitions
            trans = defaultdict(list)
            for e in es:
                syms = [x for x in re.split(r"[,\s]+", e.get("label") or "") if x] or ["?"]
                for x in syms:
                    trans[(lab(e["a"]), x)].append(lab(e["b"]))
            start = [lab(i) for i in comp if by[i].get("entry")]
            acc = [lab(i) for i in comp if by[i].get("double")]
            alphabet = sorted({x for (_, x) in trans if x not in ("?", "ε", "λ", "e")})
            det = all(len(v) == 1 for v in trans.values()) and not any(x in ("ε", "λ") for (_, x) in trans)
            parts.append(dict(type="automaton", states=[lab(i) for i in sorted(comp, key=lambda i: _center(by[i]["bbox"])[0])],
                              start=start[0] if len(start) == 1 else None, accepting=acc, alphabet=alphabet,
                              deterministic=det and len(start) == 1,
                              transitions=[[a, x, sorted(v)] for (a, x), v in sorted(trans.items())],
                              unlabelled_transitions=sum(1 for (a, x) in trans if x == "?")))
            continue
        if directed and ("diamond" in shapes or terminals):
            indeg = defaultdict(int)
            for e in es:
                indeg[e["b"]] += 1
            start = next((i for i in terminals if re.match(r"(?i)start|begin", by[i]["label"] or "")), None)
            start = start if start is not None else next((i for i in sorted(comp, key=lambda i: _center(by[i]["bbox"])[1]) if indeg[i] == 0), min(comp))
            order, q2, vis = [], deque([start]), set()
            while q2:
                i = q2.popleft()
                if i in vis:
                    continue
                vis.add(i); order.append(i)
                q2.extend(sorted((e["b"] for e in es if e["a"] == i), key=lambda j: (_center(by[j]["bbox"])[1], _center(by[j]["bbox"])[0])))
            steps = []
            for i in order:
                outs = [e for e in es if e["a"] == i]
                kind = "decision" if by[i]["shape"] == "diamond" else "start/end" if by[i]["shape"] in ("ellipse", "rounded") or i in terminals else "step"
                steps.append(dict(node=lab(i), kind=kind, next=[(e.get("label") + " → " if e.get("label") else "") + lab(e["b"]) for e in outs]))
            parts.append(dict(type="flowchart", start=lab(start), steps=steps,
                              unreached=[lab(i) for i in comp if i not in vis]))
            continue
        t = _tree(comp, nodes, edges)
        if t:
            tags = [(by[i]["label"], by[i]["tag"]) for i in sorted(comp, key=lambda i: (_center(by[i]["bbox"])[1], _center(by[i]["bbox"])[0])) if by[i].get("tag")]
            if tags:
                t["numbers_beside_nodes"] = [f"{a}: {b}" for a, b in tags]
            cols = _colours(comp, by)
            if cols:
                t["node_colours"] = cols
            parts.append(t)
            continue
        pairs = {tuple(sorted((e["a"], e["b"]))) for e in es if e["a"] != e["b"]}
        if directed and len(pairs) == len(comp) - 1 and len(es) > len(pairs):
            # every link drawn as a pair of arrows (child pointer down, parent pointer up): a linked tree
            t = _tree(comp, nodes, [dict(a=a, b=b, directed=False) for a, b in pairs])
            if t:
                t["drawn_as"] = ("linked nodes whose parent–child links are drawn as arrows both ways (child pointers "
                                 "and parent pointers); parent/child is taken from the vertical layout")
                cols = _colours(comp, by)
                if cols:
                    t["node_colours"] = cols
                parts.append(t)
                continue
        deg = defaultdict(int)
        for e in es:
            deg[e["a"]] += 1; deg[e["b"]] += 1
        cols = _colours(comp, by)
        parts.append(dict(type="directed graph" if directed else "graph", **({"node_colours": cols} if cols else {}), vertices=[lab(i) for i in sorted(comp, key=lambda i: (lab(i)))],
                          edges=[f"{lab(e['a'])} {'→' if e['directed'] else '—'} {lab(e['b'])}" + (f" ({e['label']})" if e.get("label") else "") for e in es],
                          degrees={lab(i): deg[i] for i in comp}))
    if not parts:
        return None
    unl = sum(1 for n in nodes if n["label"] is None and n["id"] in seen - blank_ids)   # empty array cells are not unread text
    notes = []
    if overlap:
        notes.append(f"{overlap} group(s) of overlapping shapes (for example a Venn or set diagram) were not read as "
                     "nodes and edges")
    if blank:
        notes.append(f"{blank} group(s) of connected shapes have no text inside any shape and were not interpreted (what they "
                     "stand for cannot be read from the drawing)")
    if unl:
        notes.append(f"{unl} shape(s) have no readable text inside (shown as unlabelled; their meaning is not guessed)")
    loose = [w["text"] for w in free if len(w["text"]) <= 20][:40]
    return dict(parts=parts, notes=notes, text_outside_shapes=loose, source="vector drawing (exact geometry)")


# ------------------------------------------------------------------------------------------- plain-language description
def estimate(st):
    """An estimated (not measured) confidence for a structure read from a drawing: it starts at 0.9 and drops for
    each kind of gap the reader itself reported (shapes without readable text, transitions without a symbol, an
    automaton without a start state, flowchart steps not reachable from the start). It is shown as 'estimated'."""
    c = 0.9
    if any("no readable text" in n for n in st.get("notes") or []):
        c -= 0.1
    for p in st.get("parts", []):
        if p.get("type") == "automaton" and (p.get("unlabelled_transitions") or not p.get("start")):
            c -= 0.1
        if p.get("type") == "flowchart" and p.get("unreached"):
            c -= 0.1
    return round(max(0.6, c), 2)


def describe(st):
    """Sentences stating what the structure shows (facts only; no interpretation beyond the computed properties)."""
    if not st:
        return ""
    out = []
    for p in st.get("parts", []):
        t = p["type"]
        if t == "sequence":
            nm = f" {p['name']}" if p.get("name") else ""
            cells = [c if c is not None else "(empty)" for c in p["cells"]]
            s = f"A {p['orientation']} of {len(cells)} cells{nm} (an array; also how stacks and queues are drawn): " + ", ".join(cells) + "."
            if p.get("indices"):
                s += (" Labels beside the cells: " if p["orientation"] == "column" else " Positions above the cells: ") + ", ".join(i or "?" for i in p["indices"]) + "."
            if p.get("fill_groups"):
                s += " Cell colours: " + "; ".join(f"cells {g['cells']} {g['fill']}" for g in p["fill_groups"]) + "."
            for ptr in p.get("pointers", []):
                s += f" An arrow labelled “{ptr['text']}” points at cell {ptr['cell']}."
            if p.get("boundaries"):
                bd = p["boundaries"]
                s += " Values under the cell boundaries (a time line, as in a Gantt chart): " + ", ".join(
                    f"{c if c is not None else '(empty)'} from {bd[k] or '?'} to {bd[k + 1] or '?'}" for k, c in enumerate(cells)) + "."
            if p.get("below"):
                s += " Text under the cells: " + ", ".join(b or "?" for b in p["below"]) + "."
            out.append(s)
        elif t in ("tree", "binary tree", "B-tree"):
            s = f"A {t} with {p['nodes']} nodes, root {p['root']}, height {p['height']}. "
            s += "Children: " + "; ".join(f"{k} → {', '.join(v)}" for k, v in p["children"]) + ". "
            s += "Leaves: " + ", ".join(p["leaves"]) + ". "
            s += "Pre-order: " + ", ".join(p["preorder"]) + ". "
            if p.get("inorder"):
                s += "In-order: " + ", ".join(p["inorder"]) + ". "
            s += "Post-order: " + ", ".join(p["postorder"]) + ". Level order: " + ", ".join(p["level_order"]) + "."
            if p.get("properties"):
                s += " " + "; ".join(p["properties"]).capitalize() + "."
            if p.get("node_colours"):
                s += " Node colours: " + "; ".join(f"{c}: {', '.join(v)}" for c, v in p["node_colours"].items()) + "."
            if p.get("numbers_beside_nodes"):
                s += " Text beside nodes: " + ", ".join(p["numbers_beside_nodes"]) + "."
            if p.get("drawn_as"):
                s += " Drawn as " + p["drawn_as"] + "."
            if p.get("btree"):
                bt = p["btree"]
                s += (f" Keys per node: " + "; ".join(f"[{' '.join(k)}]" for _, k in bt["nodes"]) + f" (at most {bt['max_keys']} keys per node). "
                      f"Keys in order: {', '.join(bt['in_order_keys'])}. {bt['property']}.")
            if p.get("codes"):
                s += " Codes read along the 0/1 edges from the root: " + "; ".join(p["codes"]) + "."
            if p.get("paths"):
                s += " Edge labels along each root-to-leaf path: " + "; ".join(p["paths"]) + "."
            out.append(s)
        elif t == "class diagram":
            s = "A UML class diagram. " + " ".join(
                f"Class {c['name']}" + (f" with attributes {', '.join(c['attributes'])}" if c["attributes"] else "") +
                (f" and operations {', '.join(c['operations'])}" if c["operations"] else "") + "." for c in p["classes"])
            if p["relationships"]:
                s += " Relationships: " + "; ".join(p["relationships"]) + "."
            out.append(s)
        elif t == "ER diagram":
            s = "An entity–relationship diagram. " + " ".join(
                f"{'Weak entity' if e['weak'] else 'Entity'} {e['name']}" + (f" with attributes {', '.join(e['attributes'])}" if e["attributes"] else "") + "."
                for e in p["entities"])
            s += " " + " ".join(f"{'Identifying relationship' if r['identifying'] else 'Relationship'} {r['name']} connects {' and '.join(r['connects']) or 'nothing readable'}"
                                + (f" (attributes {', '.join(r['attributes'])})" if r["attributes"] else "") + "." for r in p["relationships"])
            out.append(s)
        elif t == "flowchart":
            s = f"A flowchart starting at {p['start']}. " + " ".join(
                f"{st_['node']} ({st_['kind']})" + (" → " + "; ".join(st_["next"]) if st_["next"] else " (end)") + "." for st_ in p["steps"])
            if p.get("unreached"):
                s += " Not reached from the start: " + ", ".join(p["unreached"]) + "."
            out.append(s)
        elif t == "automaton":
            s = (f"A {'deterministic' if p['deterministic'] else 'nondeterministic'} finite automaton with states "
                 f"{', '.join(p['states'])}; start state {p['start'] or 'not identified'}; accepting state(s) "
                 f"{', '.join(p['accepting']) or 'none identified'}; alphabet {{{', '.join(p['alphabet'])}}}. Transitions: "
                 + "; ".join(f"δ({a}, {x}) = {', '.join(v)}" for a, x, v in p["transitions"]) + ".")
            if p.get("unlabelled_transitions"):
                s += f" {p['unlabelled_transitions']} transition(s) have no readable symbol."
            out.append(s)
        elif t == "linked lists":
            for L in p["lists"]:
                out.append((f"From {L['start']}: " if L["start"] else "A linked list: ") + (" ⇄ " if L.get("doubly") else " → ").join(L["nodes"])
                           + (" (doubly linked: each node also points back to the previous one)" if L.get("doubly") else "")
                           + (" (the last node points back to an earlier node: circular / a cycle)" if L["cycle"] else "") + ".")
        else:
            out.append(f"A {t} with {len(p['vertices'])} vertices ({', '.join(p['vertices'])}) and {len(p['edges'])} edges: " + "; ".join(p["edges"]) + "."
                       + (" Vertex colours: " + "; ".join(f"{c}: {', '.join(v)}" for c, v in p["node_colours"].items()) + "." if p.get("node_colours") else ""))
    seqs = [p for p in st.get("parts", []) if p["type"] == "sequence" and p["orientation"] == "row"]
    for k, (a, b) in enumerate(zip(seqs, seqs[1:])):
        if len(a["cells"]) == len(b["cells"]) and a["cells"] != b["cells"]:
            diff = [i for i, (x, y) in enumerate(zip(a["cells"], b["cells"])) if x != y]
            if len(diff) == 2 and a["cells"][diff[0]] == b["cells"][diff[1]] and a["cells"][diff[1]] == b["cells"][diff[0]]:
                out.append(f"From row {k + 1} to row {k + 2}: cells {diff[0] + 1} and {diff[1] + 1} are swapped.")
            elif diff:
                out.append(f"From row {k + 1} to row {k + 2}: cells {', '.join(str(i + 1) for i in diff)} change "
                           f"({', '.join(str(a['cells'][i] or '(empty)') for i in diff)} → {', '.join(str(b['cells'][i] or '(empty)') for i in diff)}).")
    for n in st.get("notes", []):
        out.append("Note: " + n + ".")
    return " ".join(out)


# ------------------------------------------------------------------------------------------- exact computations
def run_automaton(part, word):
    """Trace a word through an extracted automaton (subset simulation, so NFAs work too). Returns the trace or None
    when the automaton is not usable (no start state, or a symbol with no readable transitions)."""
    if not part or part.get("type") != "automaton" or not part.get("start"):
        return None
    delta = defaultdict(set)
    for a, x, v in part["transitions"]:
        delta[(a, x)].update(v)

    def close(S):
        S, stack = set(S), list(S)
        while stack:
            q = stack.pop()
            for eps in ("ε", "λ"):
                for r in delta.get((q, eps), ()):
                    if r not in S:
                        S.add(r); stack.append(r)
        return S
    cur = close({part["start"]})
    steps = [dict(read=None, states=sorted(cur))]
    for ch in word:
        cur = close({r for q in cur for r in delta.get((q, ch), ())})
        steps.append(dict(read=ch, states=sorted(cur)))
        if not cur:
            break
    acc = bool(cur & set(part["accepting"]))
    return dict(word=word, steps=steps, accepted=acc, dead=not cur)


def graph_facts(part, src=None, dst=None):
    """Connectivity, cycles and (weighted) shortest paths of an extracted graph."""
    import heapq
    if not part or part.get("type") not in ("graph", "directed graph"):
        return None
    directed = part["type"] == "directed graph"
    adj = defaultdict(list)
    for e in part["edges"]:
        m = re.match(r"^(.*?) (→|—) (.*?)(?: \((.*)\))?$", e)
        if not m:
            continue
        a, b, w = m.group(1), m.group(3), m.group(4)
        try:
            wt = float(str(w).replace("–", "-").replace("−", "-")) if w is not None else 1.0
        except ValueError:
            wt = 1.0
        adj[a].append((b, wt))
        if not directed:
            adj[b].append((a, wt))
    V = part["vertices"]
    out = {}
    # cycles
    color, has_cycle = {}, False

    def dfs(u, parent):
        nonlocal has_cycle
        color[u] = 1
        for v, _ in adj[u]:
            if color.get(v) == 1 and (directed or v != parent):
                has_cycle = True
            elif v not in color:
                dfs(v, u)
        color[u] = 2
    for v in V:
        if v not in color:
            dfs(v, None)
    out["has_cycle"] = has_cycle
    seen, comps = set(), 0
    und = defaultdict(set)
    for a in adj:
        for b, _ in adj[a]:
            und[a].add(b); und[b].add(a)
    for v in V:
        if v in seen:
            continue
        comps += 1
        st = [v]
        while st:
            x = st.pop()
            if x in seen:
                continue
            seen.add(x); st.extend(und[x] - seen)
    out["connected_components"] = comps
    if src and dst:
        neg = any(w < 0 for a in adj for _, w in adj[a])
        dist, prev = {src: 0.0}, {}
        if not neg:
            pq = [(0.0, src)]
            while pq:
                d, u = heapq.heappop(pq)
                if d > dist.get(u, float("inf")):
                    continue
                for v, w in adj[u]:
                    if d + w < dist.get(v, float("inf")):
                        dist[v] = d + w; prev[v] = u; heapq.heappush(pq, (d + w, v))
        else:                                   # Bellman-Ford for negative weights
            for _ in range(len(V) - 1):
                for u in list(adj):
                    for v, w in adj[u]:
                        if u in dist and dist[u] + w < dist.get(v, float("inf")):
                            dist[v] = dist[u] + w; prev[v] = u
        if dst in dist:
            path, x = [dst], dst
            while x != src:
                x = prev[x]; path.append(x)
            out["shortest_path"] = dict(path=list(reversed(path)), length=dist[dst], weighted=any(w != 1.0 for a in adj for _, w in adj[a]))
        else:
            out["shortest_path"] = dict(path=None, note=f"no path from {src} to {dst} in the extracted edges")
    return out


# ------------------------------------------------------------------------------------------- AI-model readings
_REL = re.compile(r"^\s*(.+?)\s*(?:-+\s*\(?([^>()]*?)\)?\s*-*>|→|->|—|--|–)\s*(.+?)\s*$")


def from_model(elements, relationships):
    """A vision model's list of nodes and edges ('30 -left-> 20', 'A -> B', 'x — y') as a structure, so the same
    interpretation and computations run on it. Positions are unknown, so only directed edges can form a tree."""
    rels = [str(r) for r in (relationships or []) if str(r).strip()]
    if not rels:
        return None
    names, edges = {}, []

    def nid(t):
        t = t.strip().strip(".;,")
        if t not in names:
            names[t] = len(names)
        return names[t]
    for r in rels:
        m = _REL.match(r)
        if not m:
            continue
        a, lbl, b = m.group(1), (m.group(2) or "").strip(), m.group(3)
        if len(a) > 60 or len(b) > 60:
            continue
        directed = "->" in r or "→" in r or ">" in r
        side = lbl if lbl in ("left", "right") else None
        edges.append(dict(a=nid(a), b=nid(b), directed=directed, label=None if side else (lbl or None), p=((0, 0), (0, 0)), side=side))
    if not edges:
        return None
    for t in elements or []:
        nid(str(t).split(" (")[0])
    # positions: a child marked 'left' sits left of its parent, 'right' to the right, one level lower
    pos = {i: [0.0, 0.0] for i in names.values()}
    for _ in range(len(names)):
        for e in edges:
            if e["directed"]:
                pos[e["b"]][1] = max(pos[e["b"]][1], pos[e["a"]][1] + 40)
                if e["side"]:
                    pos[e["b"]][0] = pos[e["a"]][0] + (-30 if e["side"] == "left" else 30) * (0.6 ** (pos[e["a"]][1] / 40))
    nodes = [dict(id=i, shape="ellipse", bbox=(pos[i][0] - 10, pos[i][1] - 10, pos[i][0] + 10, pos[i][1] + 10), label=t, fill=None)
             for t, i in names.items()]
    st = _interpret(nodes, edges, [], [])
    if st:
        st["source"] = "AI vision model (its own reading of the image, not verified by geometry)"
    return st


def _edge_set(st):
    out = set()
    norm = lambda s: re.sub(r"\s+", "", str(s or "")).lower().split("(")[0]
    for p in (st or {}).get("parts", []):
        for e in p.get("edges") or []:
            m = re.match(r"^(.*?) (→|—) (.*?)(?: \((.*)\))?$", e)
            if m:
                out.add(frozenset((norm(m.group(1)), norm(m.group(3)))))
        for parent, kids in p.get("children") or []:
            for k in kids:
                out.add(frozenset((norm(parent), norm(re.sub(r"^(left|right): ", "", k).split(" (edge label")[0]))))
        for a, x, tos in p.get("transitions") or []:
            for b in tos:
                out.add(frozenset((norm(a), norm(b))))
    return {e for e in out if all(e) and not any("unlabelled" in x for x in e)}


def agreement(image_st, model_st):
    """How far an AI model's reading agrees with the structure read from the pixels (edges between named nodes)."""
    a, b = _edge_set(image_st), _edge_set(model_st)
    both = a & b
    return dict(edges_total=len(b), edges_confirmed=len(both), only_model=sorted(" — ".join(sorted(e)) for e in b - a)[:20],
                only_image=sorted(" — ".join(sorted(e)) for e in a - b)[:20],
                summary=f"{len(both)} of {len(b)} edges in the AI model's reading are also in the image geometry"
                        + (f"; {len(a - b)} edge(s) seen in the image are missing from the model's reading" if a - b else ""))
