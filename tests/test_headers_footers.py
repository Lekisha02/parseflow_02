"""Running header / footer / page-number classification (document level, evidence-based, configurable)."""
from conftest import LOREM, PDF, paragraph, text
from processing import headers_footers as hf


def _doc(tmp, n=3, header=True, footer=True, first_title=False, footnote=False, section_top=False):
    pdf = PDF()
    for i in range(n):
        def draw(p, i=i):
            if header:
                text(p, 72, 36, "Journal of Synthetic Tests, Vol. 12, Issue 3", 8)
            if first_title and i == 0:
                text(p, 72, 60, "A One-Off Article Title", 20, "hebo")
            if section_top and i == 1:
                text(p, 72, 60, "2. Methods", 13, "hebo")
            y = paragraph(p, 72, 110, 450, (LOREM + " ") * 6)
            if footnote:
                text(p, 72, 700, "1 This footnote explains a detail of the method and must stay in the text body.", 8)
            if footer:
                text(p, 290, 760, f"Page {i + 1} of {n}", 9)
        pdf.page(draw)
    return pdf.save(tmp / f"hf_{n}_{header}_{footer}_{first_title}_{footnote}_{section_top}.pdf")


def test_repeated_header_and_page_numbers_removed(tmp, convert):
    d = convert(_doc(tmp))
    md = d["markdown"]
    assert "Journal of Synthetic Tests" not in md
    assert "Page 1 of 3" not in md and "Page 3 of 3" not in md
    assert LOREM[:60] in md
    # kept in the JSON with bbox, role, score and reasons
    fur = [b for b in d["blocks"] if (b.get("furniture") or {}).get("excluded")]
    assert len(fur) >= 6
    for b in fur:
        assert b["furniture"]["reasons"] and b["bbox"] and b["role"] in ("header", "footer", "page_number")
    assert any("repeats on" in r for b in fur for r in b["furniture"]["reasons"])


def test_first_page_title_footnote_and_section_heading_kept(tmp, convert):
    md = convert(_doc(tmp, first_title=True, footnote=True, section_top=True))["markdown"]
    assert "A One-Off Article Title" in md
    assert "This footnote explains a detail" in md
    assert "2. Methods" in md


def test_document_without_running_headers_keeps_everything(tmp, convert):
    d = convert(_doc(tmp, n=2, header=False, footer=False))
    assert not any((b.get("furniture") or {}).get("excluded") for b in d["blocks"])


def test_removal_is_configurable_and_restorable(tmp, convert, monkeypatch):
    f = _doc(tmp, n=2)
    d = convert(f)
    from processing import markdown as md
    assert "Journal of Synthetic Tests" not in d["markdown"]
    assert "Journal of Synthetic Tests" in md.render(d, include_furniture=True)
    monkeypatch.setenv("PARSEFLOW_REMOVE_FURNITURE", "0")
    d2 = convert(f)
    assert "Journal of Synthetic Tests" in d2["markdown"]


def test_single_page_signals():
    page = dict(width=612, height=792, items=[
        dict(id="pn", bbox=(300, 760, 312, 770), text="12", size=9, kind="text"),
        dict(id="dl", bbox=(150, 4, 470, 14), text="Downloaded from http://example.org on July 12, 2017", size=8, kind="text"),
        dict(id="body", bbox=(72, 100, 540, 700), text=LOREM * 5, size=10, kind="text"),
        dict(id="ft1", bbox=(72, 740, 160, 750), text="Revision: 2.4", size=8, kind="text"),
        dict(id="ft2", bbox=(280, 740, 350, 750), text="P/N 119-036", size=8, kind="text"),
        dict(id="ft3", bbox=(470, 740, 540, 750), text="10 June 2019", size=8, kind="text"),
    ])
    r = hf.classify([page], 10)
    for k in ("pn", "dl", "ft1", "ft2", "ft3"):
        assert r[k]["excluded"], (k, r.get(k))
    assert "body" not in r
    assert r["pn"]["role"] == "page_number"


def test_page_number_patterns():
    for s in ["12", "- 12 -", "Page 3 of 20", "xii", "p. 7", "3/20"]:
        assert hf.PAGE_NUM.match(s), s
    for s in ["2020 was a year", "Table 3", "12 apples were sold"]:
        assert not hf.PAGE_NUM.match(s), s


def test_no_removal_without_structural_signal():
    # a short line at the very top that continues the previous page's paragraph (not isolated, no repetition)
    page = dict(width=612, height=792, items=[
        dict(id="cont", bbox=(72, 50, 300, 60), text="published in 2019 by the authors.", size=10, kind="text"),
        dict(id="body", bbox=(72, 61, 540, 700), text=LOREM * 5, size=10, kind="text"),
    ])
    r = hf.classify([page], 10)
    assert not r.get("cont", {}).get("excluded")
