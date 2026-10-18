"""Vector-diagram understanding: the structure read from a PDF's drawing commands (nodes, edges, labels, pointers)
and the deterministic facts computed from it (traversals, heap/BST checks, automaton runs, shortest paths).
Every figure is drawn here with PyMuPDF so the expected answer is known exactly; the real-textbook checks on CLRS
figures live in the manual evaluation notes because that book cannot be shipped with the tests."""
import math

import pymupdf
import pytest

from processing import diagrams

BLACK = (0, 0, 0)


def _head(p, b, ang, L=7):
    l = (b[0] - L * math.cos(ang - 0.4), b[1] - L * math.sin(ang - 0.4))
    r = (b[0] - L * math.cos(ang + 0.4), b[1] - L * math.sin(ang + 0.4))
    p.draw_polyline([b, l, r, b], color=BLACK, fill=BLACK, closePath=True)


def _arrow(p, a, b, label=None, off=(4, -3), size=9):
    p.draw_line(a, b, color=BLACK, width=1)
    _head(p, b, math.atan2(b[1] - a[1], b[0] - a[0]))
    if label:
        p.insert_text(((a[0] + b[0]) / 2 + off[0], (a[1] + b[1]) / 2 + off[1]), label, fontsize=size)


def _circle(p, c, r, t, double=False):
    p.draw_circle(c, r, color=BLACK, fill=(1, 1, 1))
    if double:
        p.draw_circle(c, r - 3, color=BLACK)
    p.insert_text((c[0] - 2.8 * len(t), c[1] + 3.5), t, fontsize=10)


def _box(p, r, t=None, fill=(1, 1, 1)):
    p.draw_rect(pymupdf.Rect(*r), color=BLACK, fill=fill)
    if t is not None:
        p.insert_text((r[0] + (r[2] - r[0]) / 2 - 2.8 * len(t), r[1] + (r[3] - r[1]) / 2 + 3.5), t, fontsize=10)


def _page(draw):
    doc = pymupdf.open()
    p = doc.new_page(width=612, height=792)
    draw(p)
    return doc, p


def _part(st, kind):
    assert st, "no structure was read"
    got = [x for x in st["parts"] if x["type"] == kind]
    assert got, [x["type"] for x in st["parts"]]
    return got[0]


def _tree_page(p, keys):
    """A complete binary tree drawn level by level (keys in level order); edges join circle borders."""
    pos = {}
    for i, k in enumerate(keys):
        lvl = int(math.log2(i + 1))
        idx = i + 1 - 2 ** lvl
        span = 400 / 2 ** lvl
        pos[i] = (106 + span * (idx + 0.5), 100 + 60 * lvl)
    for i in range(1, len(keys)):
        a, b = pos[(i - 1) // 2], pos[i]
        d = math.dist(a, b)
        u = ((b[0] - a[0]) / d, (b[1] - a[1]) / d)
        p.draw_line((a[0] + 14 * u[0], a[1] + 14 * u[1]), (b[0] - 14 * u[0], b[1] - 14 * u[1]), color=BLACK)
    for i, k in enumerate(keys):
        _circle(p, pos[i], 14, str(k))


def test_heap_tree_traversals_and_property():
    doc, p = _page(lambda p: _tree_page(p, [16, 14, 10, 8, 7, 9, 3]))
    t = _part(diagrams.extract(p, (60, 60, 560, 260)), "binary tree")
    assert t["root"] == "16" and t["nodes"] == 7 and t["height"] == 2
    assert t["level_order"] == ["16", "14", "10", "8", "7", "9", "3"]
    assert t["preorder"] == ["16", "14", "8", "7", "10", "9", "3"]
    assert t["inorder"] == ["8", "14", "7", "16", "9", "10", "3"]
    assert t["postorder"] == ["8", "7", "14", "9", "3", "10", "16"]
    assert sorted(t["leaves"]) == ["3", "7", "8", "9"]
    assert any("max-heap property holds" in x for x in t["properties"])
    assert not any("binary-search-tree" in x for x in t["properties"])


def test_bst_property_and_description():
    doc, p = _page(lambda p: _tree_page(p, [8, 3, 10, 1, 6, 9, 14]))
    st = diagrams.extract(p, (60, 60, 560, 260))
    t = _part(st, "binary tree")
    assert t["inorder"] == ["1", "3", "6", "8", "9", "10", "14"]
    assert any("binary-search-tree property holds" in x for x in t["properties"])
    text = diagrams.describe(st)
    assert "root 8" in text.replace("root is 8", "root 8") or "8" in text
    assert "1, 3, 6, 8, 9, 10, 14" in text or "1 3 6 8 9 10 14" in text or "['1', '3'" in text


def test_stack_cells_and_top_pointer():
    def draw(p):
        for k, v in enumerate(["15", "6", "2", "9"]):
            _box(p, (100 + 30 * k, 120, 130 + 30 * k, 145), v)
        for k in range(2):
            _box(p, (220 + 30 * k, 120, 250 + 30 * k, 145))
        for k in range(6):
            p.insert_text((112 + 30 * k, 112), str(k + 1), fontsize=8)
        p.insert_text((180, 185), "S.top = 4", fontsize=9)
        _arrow(p, (197, 176), (197, 147))
    doc, p = _page(draw)
    s = _part(diagrams.extract(p, (80, 95, 320, 195)), "sequence")
    assert s["cells"][:4] == ["15", "6", "2", "9"] and s["cells"][4:] == [None, None]
    assert s["indices"] == ["1", "2", "3", "4", "5", "6"]
    assert s["pointers"] and s["pointers"][0]["cell"] == 4 and "top" in s["pointers"][0]["text"]


def test_queue_head_and_tail_pointers():
    def draw(p):
        vals = [None, None, "15", "6", "9", "8", None]
        for k, v in enumerate(vals):
            _box(p, (100 + 30 * k, 120, 130 + 30 * k, 145), v)
        p.insert_text((150, 185), "Q.head", fontsize=9)
        _arrow(p, (175, 176), (175, 147))
        p.insert_text((270, 185), "Q.tail", fontsize=9)
        _arrow(p, (295, 176), (295, 147))
    doc, p = _page(draw)
    s = _part(diagrams.extract(p, (80, 95, 340, 195)), "sequence")
    assert s["cells"] == [None, None, "15", "6", "9", "8", None]
    ptr = {x["text"]: x["cell"] for x in s["pointers"]}
    assert ptr == {"Q.head": 3, "Q.tail": 7}


def test_singly_linked_list_with_nil():
    def draw(p):
        xs = [100, 200, 300]
        for x, v in zip(xs, ["9", "16", "4"]):
            _box(p, (x, 120, x + 30, 145), v)
            _box(p, (x + 30, 120, x + 50, 145))
        for x in xs[:2]:
            _arrow(p, (x + 40, 132.5), (x + 100, 132.5))
        p.draw_line((330, 145), (350, 120), color=BLACK)            # NIL slash in the last next field
        p.insert_text((40, 136), "L.head", fontsize=9)
        _arrow(p, (75, 132.5), (100, 132.5))
    doc, p = _page(draw)
    ll = _part(diagrams.extract(p, (30, 100, 380, 165)), "linked lists")
    chains = [x for x in ll["lists"] if len(x["nodes"]) == 3]
    assert chains, ll
    assert [n.split(" | ")[0].strip("[") for n in chains[0]["nodes"]] == ["9", "16", "4"]
    assert "NIL" in chains[0]["nodes"][-1] and chains[0]["cycle"] is False
    assert chains[0]["start"] == "L.head"


def _graph(p):
    pts = {"s": (100, 200), "a": (220, 120), "b": (220, 280), "t": (340, 200)}
    for n, c in pts.items():
        _circle(p, c, 15, n)

    def e(u, v, w, off):
        a, b = pts[u], pts[v]
        d = math.dist(a, b)
        x = ((b[0] - a[0]) / d, (b[1] - a[1]) / d)
        _arrow(p, (a[0] + 15 * x[0], a[1] + 15 * x[1]), (b[0] - 15 * x[0], b[1] - 15 * x[1]), w, off)
    e("s", "a", "4", (-12, -4)); e("s", "b", "1", (-14, 10)); e("b", "a", "2", (5, 0))
    e("a", "t", "1", (6, -4)); e("b", "t", "5", (6, 10))


def test_weighted_directed_graph_and_shortest_path():
    doc, p = _page(_graph)
    g = _part(diagrams.extract(p, (70, 90, 370, 310)), "directed graph")
    assert sorted(g["vertices"]) == ["a", "b", "s", "t"]
    assert sorted(g["edges"]) == sorted(["s → a (4)", "s → b (1)", "b → a (2)", "a → t (1)", "b → t (5)"])
    f = diagrams.graph_facts(g, "s", "t")
    assert f["shortest_path"]["path"] == ["s", "b", "a", "t"] and f["shortest_path"]["length"] == 4
    assert f["has_cycle"] is False and f["connected_components"] == 1


def test_flowchart_decision_branches():
    def draw(p):
        p.draw_oval(pymupdf.Rect(250, 80, 350, 110), color=BLACK); p.insert_text((285, 99), "Start", fontsize=10)
        _box(p, (240, 140, 360, 170)); p.insert_text((265, 159), "Read n", fontsize=10)
        dm = [(300, 200), (360, 235), (300, 270), (240, 235)]
        p.draw_polyline(dm + [dm[0]], color=BLACK, closePath=True); p.insert_text((272, 239), "n > 0?", fontsize=8)
        _box(p, (150, 300, 250, 330)); p.insert_text((165, 319), "Positive", fontsize=10)
        _box(p, (350, 300, 450, 330)); p.insert_text((367, 319), "Not positive", fontsize=10)
        p.draw_oval(pymupdf.Rect(250, 370, 350, 400), color=BLACK); p.insert_text((289, 389), "End", fontsize=10)
        _arrow(p, (300, 110), (300, 140)); _arrow(p, (300, 170), (300, 200))
        _arrow(p, (240, 235), (200, 300), "yes", (4, 0)); _arrow(p, (360, 235), (400, 300), "no", (4, 0))
        _arrow(p, (200, 330), (290, 370)); _arrow(p, (400, 330), (310, 370))
    doc, p = _page(draw)
    f = _part(diagrams.extract(p, (130, 70, 470, 410)), "flowchart")
    assert f["start"] == "Start" and not f["unreached"]
    dec = [s for s in f["steps"] if s["kind"] == "decision"]
    assert len(dec) == 1 and sorted(dec[0]["next"]) == ["no → Not positive", "yes → Positive"]
    assert [s["node"] for s in f["steps"]][:3] == ["Start", "Read n", "n > 0?"]


def _dfa(p):
    def loop(c, r, label):
        a, b = (c[0] - 6, c[1] - r + 1), (c[0] + 6, c[1] - r + 1)
        p.draw_bezier(a, (c[0] - 18, c[1] - r - 26), (c[0] + 18, c[1] - r - 26), b, color=BLACK)
        _head(p, b, math.atan2(b[1] - (c[1] - r - 26), b[0] - (c[0] + 18)))
        p.insert_text((c[0] - 3, c[1] - r - 22), label, fontsize=9)
    q0, q1 = (180, 130), (320, 130)
    _circle(p, q0, 18, "q0"); _circle(p, q1, 18, "q1", double=True)
    _arrow(p, (120, 130), (162, 130))
    _arrow(p, (198, 124), (302, 124), "1", (0, -4)); _arrow(p, (302, 138), (198, 138), "0", (0, 12))
    loop(q0, 18, "0"); loop(q1, 18, "1")


def test_dfa_transitions_and_runs():
    doc, p = _page(_dfa)
    a = _part(diagrams.extract(p, (100, 70, 360, 170)), "automaton")
    assert a["start"] == "q0" and a["accepting"] == ["q1"] and a["deterministic"]
    assert sorted(map(tuple, ([x, y, tuple(z)] for x, y, z in a["transitions"]))) == sorted(
        [("q0", "0", ("q0",)), ("q0", "1", ("q1",)), ("q1", "0", ("q0",)), ("q1", "1", ("q1",))])
    assert diagrams.run_automaton(a, "0101")["accepted"] is True
    r = diagrams.run_automaton(a, "110")
    assert r["accepted"] is False and [s["states"] for s in r["steps"]] == [["q0"], ["q1"], ["q1"], ["q0"]]
    assert diagrams.run_automaton(a, "2")["dead"] is True


def test_gantt_chart_time_boundaries():
    def draw(p):
        x = 100
        for name, t0, t1 in (("P1", 0, 4), ("P2", 4, 7), ("P3", 7, 12)):
            w = (t1 - t0) * 20
            _box(p, (x, 240, x + w, 265), name, fill=(0.85, 0.9, 1))
            p.insert_text((x - 3, 278), str(t0), fontsize=9)
            x += w
        p.insert_text((x - 5, 278), "12", fontsize=9)
    doc, p = _page(draw)
    st = diagrams.extract(p, (80, 225, 360, 290))
    s = _part(st, "sequence")
    assert s["cells"] == ["P1", "P2", "P3"] and s.get("boundaries") == ["0", "4", "7", "12"]
    text = diagrams.describe(st)
    assert "P1" in text and "0" in text and "12" in text


def test_sorting_trace_rows_are_diffed():
    def draw(p):
        for row, vals in enumerate(([5, 1, 4, 2], [1, 5, 4, 2])):
            for k, v in enumerate(vals):
                _box(p, (100 + 22 * k, 520 + row * 40, 122 + 22 * k, 542 + row * 40), str(v), fill=(1, 0.95, 0.8))
    doc, p = _page(draw)
    st = diagrams.extract(p, (80, 505, 220, 600))
    rows = [x["cells"] for x in st["parts"] if x["type"] == "sequence"]
    assert rows == [["5", "1", "4", "2"], ["1", "5", "4", "2"]]
    assert "swap" in diagrams.describe(st).lower() or "exchang" in diagrams.describe(st).lower()


def test_unreadable_labels_are_reported_not_invented():
    def draw(p):
        _tree_page(p, [1, 2, 3])
        p.draw_circle((150, 400), 14, color=BLACK)                  # a 3-node tree further down, one key unreadable
        p.draw_circle((110, 460), 14, color=BLACK); p.insert_text((106, 464), "4", fontsize=10)
        p.draw_circle((190, 460), 14, color=BLACK); p.insert_text((186, 464), "9", fontsize=10)
        p.draw_line((142, 411), (118, 449), color=BLACK); p.draw_line((158, 411), (182, 449), color=BLACK)
        p.draw_circle((350, 400), 14, color=BLACK)                  # two shapes with no text at all
        p.draw_circle((350, 470), 14, color=BLACK)
        p.draw_line((350, 414), (350, 456), color=BLACK)
    doc, p = _page(draw)
    st = diagrams.extract(p, (90, 380, 380, 490))
    assert any("no readable text" in n for n in st["notes"]) and any("no text inside any shape" in n for n in st["notes"])
    assert len(st["parts"]) == 1                                     # the text-free pair is not interpreted
    t = st["parts"][0]
    assert t["root"].endswith("unlabelled)") and sorted(t["leaves"]) == ["4", "9"]
    assert diagrams.estimate(st) < 0.9


def test_plain_text_region_is_not_a_diagram():
    doc, p = _page(lambda p: p.insert_text((72, 100), "Just a sentence with no drawing at all.", fontsize=11))
    assert diagrams.extract(p, (60, 80, 400, 120)) is None


@pytest.mark.parametrize("word,ok", [("1", True), ("10", False), ("", False), ("0011", True)])
def test_dfa_language_ends_in_one(word, ok):
    doc, p = _page(_dfa)
    a = _part(diagrams.extract(p, (100, 70, 360, 170)), "automaton")
    assert diagrams.run_automaton(a, word)["accepted"] is ok


def test_self_loop_drawn_as_shuffled_pieces_and_unlabelled_digraph_is_not_an_automaton():
    """CLRS-style: a loop stroke written as three beziers in the order 2, 1, 3; no edge has a symbol, so it is a
    directed graph (an automaton needs labelled transitions)."""
    def draw(p):
        a, b = (100, 100), (200, 100)
        _circle(p, a, 12, "1"); _circle(p, b, 12, "2")
        _arrow(p, (112, 100), (188, 100))
        sh = p.new_shape()
        P0, P1, P2, P3 = (209, 94), (222, 100), (209, 106), (200, 112)
        sh.draw_bezier(P1, (222, 104), (216, 106), P2)            # piece 2 first
        sh.draw_bezier(P0, (216, 94), (222, 96), P1)              # then piece 1
        sh.draw_bezier(P2, (204, 106), (202, 109), P3)            # then piece 3
        sh.finish(color=BLACK, closePath=False); sh.commit()
        _head(p, P0, math.atan2(94 - 96, 209 - 216))
    doc, p = _page(draw)
    st = diagrams.extract(p, (80, 80, 240, 125))
    g = _part(st, "directed graph")
    assert sorted(g["edges"]) == ["1 → 2", "2 → 2"]
    assert not [x for x in st["parts"] if x["type"] == "automaton"]


def test_tree_drawn_with_child_and_parent_pointers():
    def draw(p):
        pos = {"6": (300, 100), "5": (220, 160), "7": (380, 160)}
        for k, c in pos.items():
            _box(p, (c[0] - 15, c[1] - 12, c[0] + 15, c[1] + 12), k, fill=(1, 0.9, 0.7))
        for child in ("5", "7"):
            (x0, y0), (x1, y1) = pos["6"], pos[child]
            s = 1 if x1 > x0 else -1
            _arrow(p, (x0 + 4 * s, y0 + 12), (x1 - 12 * s, y1 - 12))        # child pointer
            _arrow(p, (x1 + 2 * s, y1 - 12), (x0 + 14 * s, y0 + 12))         # parent pointer, drawn beside it
    doc, p = _page(draw)
    t = _part(diagrams.extract(p, (180, 70, 420, 190)), "binary tree")
    assert t["root"] == "6" and t["inorder"] == ["5", "6", "7"] and "parent pointer" in t["drawn_as"]


def test_doubly_linked_list_of_three_field_records():
    """CLRS 10.3 style: [prev | key | next] records joined by arrows both ways, NIL slashes at both ends."""
    def draw(p):
        xs = [100, 200, 300]
        for x, v in zip(xs, ["9", "16", "4"]):
            _box(p, (x, 120, x + 18, 140)); _box(p, (x + 18, 120, x + 42, 140), v); _box(p, (x + 42, 120, x + 60, 140))
        p.draw_line((100, 140), (118, 120), color=BLACK)                # prev of the first node is NIL
        p.draw_line((342, 140), (360, 120), color=BLACK)                # next of the last node is NIL
        for x in xs[:2]:
            _arrow(p, (x + 51, 126), (x + 100, 126))                     # next
            _arrow(p, (x + 109, 134), (x + 60, 134))                     # prev
        p.insert_text((40, 134), "L.head", fontsize=9)
        _arrow(p, (75, 130), (100, 130))
    doc, p = _page(draw)
    st = diagrams.extract(p, (30, 100, 380, 160))
    ll = _part(st, "linked lists")["lists"]
    main = max(ll, key=lambda L: len(L["nodes"]))
    assert main["start"] == "L.head" and main["doubly"] and not main["cycle"]
    assert [n.split(" | ")[1] for n in main["nodes"]] == ["9", "16", "4"]
    assert main["nodes"][0].startswith("[NIL") and main["nodes"][-1].endswith("NIL]")
    assert "doubly linked" in diagrams.describe(st)


def test_subscripted_node_labels_and_arrow_chains():
    def draw(p):
        pos = [(150, 100), (150, 170), (150, 240)]
        for k, c in enumerate(pos):
            p.draw_circle(c, 16, color=BLACK)
            p.insert_text((c[0] - 7, c[1] + 3), "a", fontsize=11)
            p.insert_text((c[0] - 1.2, c[1] + 6), str(k), fontsize=7)    # subscript, smaller and lower
        _arrow(p, (150, 116), (150, 154)); _arrow(p, (150, 186), (150, 224))
    doc, p = _page(draw)
    t = _part(diagrams.extract(p, (120, 70, 180, 270)), "tree")         # a chain of arrows: not a binary tree
    assert t["preorder"] == ["a_0", "a_1", "a_2"] and "inorder" not in t


def test_venn_diagram_is_not_read_as_a_graph():
    def draw(p):
        p.draw_circle((200, 200), 50, color=BLACK); p.insert_text((170, 200), "A", fontsize=11)
        p.draw_circle((260, 200), 50, color=BLACK); p.insert_text((280, 200), "B", fontsize=11)
        p.draw_line((150, 270), (310, 270), color=BLACK)
    doc, p = _page(draw)
    st = diagrams.extract(p, (140, 140, 320, 280))
    assert st is None or not any(x["type"] in ("graph", "directed graph", "tree", "binary tree") for x in st["parts"])


def test_highlighted_edge_drawn_twice_is_one_edge():
    def draw(p):
        _circle(p, (100, 100), 14, "u"); _circle(p, (220, 100), 14, "v"); _circle(p, (160, 180), 14, "w")
        p.draw_line((114, 100), (206, 100), color=(0.3, 0.7, 1), width=4)     # highlight under the edge, no head
        _arrow(p, (114, 100), (206, 100), "5", (0, -5))
        _arrow(p, (210, 110), (170, 168), "2", (6, 0))
        _arrow(p, (150, 168), (110, 112), "3", (-12, 0))
    doc, p = _page(draw)
    g = _part(diagrams.extract(p, (80, 80, 240, 200)), "directed graph")
    assert sorted(g["edges"]) == ["u → v (5)", "v → w (2)", "w → u (3)"]
    assert diagrams.graph_facts(g)["has_cycle"] is True


def test_rotated_stacked_and_farther_labels():
    """Weights written along slanted edges (rotated text), a two-line edge label, and a label set a little further
    from its edge than usual are all assigned, each to one edge only."""
    def draw(p):
        _circle(p, (100, 300), 14, "s"); _circle(p, (200, 200), 14, "a"); _circle(p, (300, 300), 14, "t")
        _circle(p, (200, 400), 14, "b")
        _arrow(p, (110, 290), (190, 210))
        pt = pymupdf.Point(140, 240)
        p.insert_text(pt, "12", fontsize=10, morph=(pt, pymupdf.Matrix(-45)))      # along the s->a edge
        _arrow(p, (210, 210), (290, 290))
        pt2 = pymupdf.Point(250, 232)
        p.insert_text(pt2, "7", fontsize=10, morph=(pt2, pymupdf.Matrix(45)))
        _arrow(p, (114, 300), (286, 300))
        p.insert_text((178, 290), "Evaluation", fontsize=8)                      # two-line label over s->t
        p.insert_text((176, 299), "Time O(n)", fontsize=8)
        _arrow(p, (110, 310), (190, 390))
        p.insert_text((118, 362), "5", fontsize=10)                              # ~20 pt from its edge
    doc, p = _page(draw)
    g = _part(diagrams.extract(p, (80, 180, 320, 420)), "directed graph")
    assert sorted(g["edges"]) == sorted(["s → a (12)", "a → t (7)", "s → t (Evaluation Time O(n))", "s → b (5)"])


def test_btree_keys_and_order():
    def draw(p):
        def node(x, y, keys):
            w = 16 * len(keys) + 8
            p.draw_rect(pymupdf.Rect(x - w / 2, y - 10, x + w / 2, y + 10), color=BLACK, fill=(1, 0.9, 0.7), radius=0.3)
            for k, t in enumerate(keys):
                p.insert_text((x - w / 2 + 6 + 16 * k, y + 4), t, fontsize=10)
            return (x, y)
        r = node(300, 100, ["M"])
        a, b = node(200, 160, ["D", "H"]), node(400, 160, ["Q", "T", "X"])
        leaves = [node(x, 230, ks) for x, ks in ((110, ["B", "C"]), (180, ["F", "G"]), (250, ["J", "K", "L"]),
                                                  (330, ["N", "P"]), (390, ["R", "S"]), (450, ["V", "W"]), (510, ["Y", "Z"]))]
        for c in (a, b):
            p.draw_line((r[0], r[1] + 10), (c[0], c[1] - 10), color=BLACK)
        for c in leaves[:3]:
            p.draw_line((a[0], a[1] + 10), (c[0], c[1] - 10), color=BLACK)
        for c in leaves[3:]:
            p.draw_line((b[0], b[1] + 10), (c[0], c[1] - 10), color=BLACK)
    doc, p = _page(draw)
    st = diagrams.extract(p, (60, 80, 560, 250))
    t = _part(st, "B-tree")
    assert t["btree"]["in_order_keys"] == list("BCDFGHJKLMNPQRSTVWXYZ")
    assert t["btree"]["property"].startswith("B-tree order holds")
    assert "Keys per node" in diagrams.describe(st)


def test_huffman_codes_from_edge_labels():
    def draw(p):
        pos = {"100": (300, 80), "a:45": (220, 150), "55": (380, 150), "c:12": (330, 220), "d:16": (430, 220)}
        for k, c in pos.items():
            _box(p, (c[0] - 22, c[1] - 10, c[0] + 22, c[1] + 10), k)
        for (u, v), lbl in ((("100", "a:45"), "0"), (("100", "55"), "1"), (("55", "c:12"), "0"), (("55", "d:16"), "1")):
            (x0, y0), (x1, y1) = pos[u], pos[v]
            p.draw_line((x0, y0 + 10), (x1, y1 - 10), color=BLACK)
            p.insert_text(((x0 + x1) / 2 + (-12 if x1 < x0 else 6), (y0 + y1) / 2), lbl, fontsize=9)
    doc, p = _page(draw)
    t = _part(diagrams.extract(p, (180, 60, 480, 240)), "binary tree")
    assert sorted(t["codes"]) == sorted(["a:45: 0", "c:12: 10", "d:16: 11"])


def test_er_diagram():
    def draw(p):
        _box(p, (80, 200, 160, 230), "Student"); _box(p, (360, 200, 440, 230), "Course")
        dm = [(260, 190), (300, 215), (260, 240), (220, 215)]
        p.draw_polyline(dm + [dm[0]], color=BLACK, closePath=True); p.insert_text((243, 219), "Enrols", fontsize=9)
        p.draw_oval(pymupdf.Rect(60, 120, 130, 150), color=BLACK); p.insert_text((75, 139), "roll_no", fontsize=9)
        p.draw_line((75, 142), (115, 142), color=BLACK)                        # underlined: the key
        p.draw_oval(pymupdf.Rect(140, 120, 200, 150), color=BLACK); p.insert_text((155, 139), "name", fontsize=9)
        p.draw_oval(pymupdf.Rect(370, 120, 440, 150), color=BLACK); p.insert_text((385, 139), "title", fontsize=9)
        for a, b in (((95, 150), (110, 200)), ((170, 150), (140, 200)), ((405, 150), (400, 200)), ((160, 215), (220, 215)), ((300, 215), (360, 215))):
            p.draw_line(a, b, color=BLACK)
        p.insert_text((185, 210), "N", fontsize=9); p.insert_text((330, 210), "M", fontsize=9)
    doc, p = _page(draw)
    er = _part(diagrams.extract(p, (50, 110, 450, 250)), "ER diagram")
    ents = {e["name"]: e for e in er["entities"]}
    assert set(ents) == {"Student", "Course"}
    assert "roll_no (key)" in ents["Student"]["attributes"] and "name" in ents["Student"]["attributes"]
    assert ents["Course"]["attributes"] == ["title"]
    r = er["relationships"][0]
    assert r["name"] == "Enrols" and sorted(r["connects"]) == ["Course (M)", "Student (N)"]


def test_uml_class_diagram():
    def draw(p):
        def cls(x, y, name, attrs, ops):
            h = 16 * (1 + len(attrs) + len(ops)) + 8
            p.draw_rect(pymupdf.Rect(x, y, x + 140, y + h), color=BLACK)
            yy = y + 13
            p.insert_text((x + 40, yy), name, fontsize=10); yy += 6
            p.draw_line((x, yy), (x + 140, yy), color=BLACK); yy += 12
            for a in attrs:
                p.insert_text((x + 6, yy), a, fontsize=9); yy += 14
            p.draw_line((x, yy - 9), (x + 140, yy - 9), color=BLACK); yy += 4
            for o in ops:
                p.insert_text((x + 6, yy), o, fontsize=9); yy += 14
            return pymupdf.Rect(x, y, x + 140, y + h)
        a = cls(200, 60, "Shape", ["- colour: str"], ["+ area(): float"])
        b = cls(100, 220, "Circle", ["- r: float"], ["+ area(): float"])
        c = cls(320, 220, "Square", ["- side: float"], ["+ area(): float"])
        for k in (b, c):
            tip = ((a.x0 + a.x1) / 2 + (k.x0 - 200) / 4, a.y1)
            p.draw_line(((k.x0 + k.x1) / 2, k.y0), (tip[0], tip[1] + 12), color=BLACK)
            p.draw_polyline([(tip[0], tip[1]), (tip[0] - 7, tip[1] + 12), (tip[0] + 7, tip[1] + 12), (tip[0], tip[1])], color=BLACK, closePath=True)
    doc, p = _page(draw)
    cd = _part(diagrams.extract(p, (90, 50, 470, 320)), "class diagram")
    cls = {c["name"]: c for c in cd["classes"]}
    assert set(cls) == {"Shape", "Circle", "Square"}
    assert cls["Shape"]["attributes"] == ["- colour: str"] and cls["Shape"]["operations"] == ["+ area(): float"]
    assert sorted(cd["relationships"]) == ["Circle is a kind of Shape (generalisation)", "Square is a kind of Shape (generalisation)"]
