"""Diagrams that are images (scans, screenshots, pictures in a PDF): the structure is recovered from the pixels with
the same interpretation as vector figures. Figures are drawn here, rasterized, and read back without the PDF's
drawing commands; geometry is tested with the known words, and end to end with OCR when Tesseract is installed."""
import io

import pymupdf
import pytest
from PIL import Image

import test_diagrams as T
from conftest import HAS_TESSERACT
from processing import diagrams, diagrams_raster as R, pipeline


def _raster(draw, region, zoom=2.0, oracle=True):
    doc, p = T._page(draw)
    pix = p.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), clip=pymupdf.Rect(*region), alpha=False)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    words = None
    if oracle:
        words = [((w[0] - region[0]) * zoom, (w[1] - region[1]) * zoom, (w[2] - region[0]) * zoom, (w[3] - region[1]) * zoom, w[4], 1.0)
                 for w in diagrams.page_words(p)]
    return R.extract(img, origin=region[:2], zoom=zoom, words=words)


def test_tree_from_pixels():
    st = _raster(lambda p: T._tree_page(p, [16, 14, 10, 8, 7, 9, 3]), (60, 60, 560, 260))
    t = T._part(st, "binary tree")
    assert t["level_order"] == ["16", "14", "10", "8", "7", "9", "3"] and any("max-heap" in x for x in t["properties"])
    assert "image" in st["source"]


def test_weighted_digraph_from_pixels():
    g = T._part(_raster(T._graph, (70, 90, 370, 310)), "directed graph")
    assert sorted(g["edges"]) == sorted(["s → a (4)", "s → b (1)", "b → a (2)", "a → t (1)", "b → t (5)"])
    assert diagrams.graph_facts(g, "s", "t")["shortest_path"]["path"] == ["s", "b", "a", "t"]


def test_dfa_from_pixels_double_circle_and_loops():
    a = T._part(_raster(T._dfa, (100, 70, 360, 170)), "automaton")
    assert a["start"] == "q0" and a["accepting"] == ["q1"]
    assert sorted(map(tuple, ([x, y, tuple(z)] for x, y, z in a["transitions"]))) == sorted(
        [("q0", "0", ("q0",)), ("q0", "1", ("q1",)), ("q1", "0", ("q0",)), ("q1", "1", ("q1",))])
    assert diagrams.run_automaton(a, "0101")["accepted"] is True


def test_filled_nodes_without_outlines():
    """Nodes drawn as coloured discs with no outline (the CLRS style) and thin anti-aliased edges."""
    def draw(p):
        pos = {"s": (100, 100), "v1": (220, 60), "v2": (220, 140), "t": (340, 100)}
        for k, c in pos.items():
            p.draw_circle(c, 14, color=None, fill=(1, 0.88, 0.7))
            p.insert_text((c[0] - 3 * len(k), c[1] + 3.5), k, fontsize=10)
        for u, v, w in (("s", "v1", "16"), ("s", "v2", "13"), ("v1", "t", "12"), ("v2", "t", "14")):
            (x0, y0), (x1, y1) = pos[u], pos[v]
            import math
            d = math.dist((x0, y0), (x1, y1)); ux, uy = (x1 - x0) / d, (y1 - y0) / d
            p.draw_line((x0 + 14 * ux, y0 + 14 * uy), (x1 - 20 * ux, y1 - 20 * uy), color=(0, 0, 0), width=0.6)
            T._head(p, (x1 - 14 * ux, y1 - 14 * uy), math.atan2(uy, ux))
            p.insert_text(((x0 + x1) / 2, (y0 + y1) / 2 - 5), w, fontsize=9)
    g = T._part(_raster(draw, (70, 30, 370, 170), zoom=150 / 72), "directed graph")
    assert sorted(g["edges"]) == sorted(["s → v1 (16)", "s → v2 (13)", "v1 → t (12)", "v2 → t (14)"])


def test_venn_and_blank_shapes_are_not_graphs():
    def draw(p):
        p.draw_circle((200, 200), 50, color=(0, 0, 0)); p.insert_text((170, 200), "A", fontsize=11)
        p.draw_circle((260, 200), 50, color=(0, 0, 0)); p.insert_text((280, 200), "B", fontsize=11)
    st = _raster(draw, (140, 140, 320, 260))
    assert st is None or not any(x["type"] in ("graph", "directed graph", "tree", "binary tree") for x in st["parts"])


@pytest.mark.ocr
def test_tree_and_graph_with_ocr():
    t = T._part(_raster(lambda p: T._tree_page(p, [16, 14, 10, 8, 7, 9, 3]), (60, 60, 560, 260), oracle=False), "binary tree")
    assert t["level_order"] == ["16", "14", "10", "8", "7", "9", "3"]
    g = T._part(_raster(T._graph, (70, 90, 370, 310), oracle=False), "directed graph")
    assert {e.split(" (")[-1] for e in g["edges"]} == {"4)", "1)", "2)", "5)"}      # every weight read by OCR
    assert len(g["edges"]) == 5


@pytest.mark.ocr
def test_picture_of_a_tree_inside_a_pdf_becomes_a_diagram_block(tmp, monkeypatch):
    for k in ("VISION_API_KEY", "VISION_PROVIDER"):
        monkeypatch.delenv(k, raising=False)
    doc, p = T._page(lambda p: T._tree_page(p, [8, 3, 10, 1, 6, 9, 14]))
    pix = p.get_pixmap(matrix=pymupdf.Matrix(2, 2), clip=pymupdf.Rect(60, 60, 560, 260), alpha=False)
    out = pymupdf.open(); pg = out.new_page(width=612, height=792)
    pg.insert_text((72, 60), "Figure 1 A binary search tree (scanned).", fontsize=10)
    pg.insert_image(pymupdf.Rect(72, 80, 542, 268), stream=pix.tobytes("png"))
    f = tmp / "pic.pdf"; out.save(f)
    d = pipeline.process(str(f), "pic.pdf", "picpicpicpic", str(tmp / "o"), pipeline.Progress(), save_images=False)
    dg = [b for b in d["blocks"] if b["type"] == "diagram" and b["extractor"] == "raster_diagram"]
    assert dg, [(b["type"], b["extractor"]) for b in d["blocks"]]
    t = T._part(dg[0]["structure"], "binary tree")
    assert t["inorder"] == ["1", "3", "6", "8", "9", "10", "14"]
    assert dg[0]["verification"] == "review" and "estimated" in dg[0]["confidence_type"]


def test_model_reading_as_structure_and_agreement():
    m = diagrams.from_model(["30", "20", "40", "10"], ["30 -left-> 20", "30 -right-> 40", "20 -left-> 10"])
    t = T._part(m, "binary tree")
    assert t["inorder"] == ["10", "20", "30", "40"] and "AI vision model" in m["source"]
    img = _raster(lambda p: T._tree_page(p, [30, 20, 40, 10]), (60, 60, 560, 260))
    ag = diagrams.agreement(img, m)
    assert ag["edges_total"] == 3 and ag["edges_confirmed"] == 3
    wrong = diagrams.from_model([], ["30 -left-> 20", "30 -right-> 50"])
    ag2 = diagrams.agreement(img, wrong)
    assert ag2["edges_confirmed"] == 1 and "30 — 50" in " ".join(ag2["only_model"])
