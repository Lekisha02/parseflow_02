"""Table detection and structure reconstruction."""
import re

from conftest import LOREM, PDF, paragraph, text
from processing import tables as T


def _tables(d):
    return [b for b in d["blocks"] if b["type"] == "table"]


def _grid(page, x0, y0, colw, rowh, nrows, ncols, merged=()):
    """Draw a ruled grid; merged = [(row, col0, col1)] suppresses inner verticals inside that row span."""
    for r in range(nrows + 1):
        page.draw_line((x0, y0 + r * rowh), (x0 + colw * ncols, y0 + r * rowh))
    for c in range(ncols + 1):
        for r in range(nrows):
            if any(m[0] == r and m[1] < c <= m[2] for m in merged):
                continue
            page.draw_line((x0 + c * colw, y0 + r * rowh), (x0 + c * colw, y0 + (r + 1) * rowh))


def test_ruled_table_with_merged_header_cell(tmp, convert):
    def draw(p):
        _grid(p, 72, 100, 110, 20, 4, 3, merged=[(0, 1, 2)])
        text(p, 78, 114, "Region")
        text(p, 200, 114, "Revenue ($M)")              # spans columns 1-2
        for i, (a, b, c) in enumerate([("", "FY23", "FY24"), ("Americas", "18.0", "19.1"), ("APAC", "14.4", "-17.9")], 1):
            for j, v in enumerate((a, b, c)):
                if v:
                    text(p, 78 + 110 * j, 114 + 20 * i, v)
    d = convert(PDF().page(draw).save(tmp / "ruled.pdf"))
    tb = _tables(d)
    assert len(tb) == 1
    b = tb[0]
    assert b["table_method"] == "ruled_grid"
    span = [c for c in b["cells"] if c["text"] == "Revenue ($M)"][0]
    assert span["colspan"] == 2
    assert "<table>" in b["markdown"] and 'colspan="2"' in b["markdown"]      # merged cell needs HTML
    assert "-17.9" in b["markdown"]                                          # sign/precision preserved
    empty = [c for c in b["cells"] if c["row"] == 1 and c["col"] == 0][0]
    assert empty["text"] == "" and empty["filled"] is False                  # empty cell stays empty
    assert b["verification"] == "verified"                                   # grid geometry is independent of the text


def test_borderless_numeric_table_markdown(tmp, convert):
    rows = [("Model", "Accuracy", "F1", "Latency"), ("Baseline", "71.2", "0.69", "12 ms"), ("Ours", "84.9", "0.83", "15 ms"),
            ("Ours + OCR", "86.1", "0.85", "41 ms"), ("Oracle", "92.0", "0.91", "n/a")]

    def draw(p):
        paragraph(p, 72, 80, 450, LOREM)
        for i, r in enumerate(rows):
            for j, v in enumerate(r):
                text(p, 72 + 120 * j, 140 + 16 * i, v)
        paragraph(p, 72, 240, 450, LOREM)
    d = convert(PDF().page(draw).save(tmp / "borderless.pdf"))
    tb = _tables(d)
    assert len(tb) == 1 and tb[0]["table_method"] == "aligned_text"
    md = tb[0]["markdown"]
    assert md.splitlines()[0] == "| Model | Accuracy | F1 | Latency |"
    assert "| Ours + OCR | 86.1 | 0.85 | 41 ms |" in md
    assert tb[0]["verification"] == "review"            # structure inferred from alignment: not independently verified
    assert LOREM[:40] in d["markdown"]


def test_two_column_prose_is_not_a_table(tmp, convert):
    def draw(p):
        paragraph(p, 72, 100, 220, LOREM * 3)
        paragraph(p, 320, 100, 220, LOREM * 3)
    assert not _tables(convert(PDF().page(draw).save(tmp / "prose.pdf")))


def test_header_rows_and_underline_span(tmp, convert):
    def draw(p):
        text(p, 330, 100, "Year Ended")
        p.draw_line((250, 104), (500, 104))
        for j, v in enumerate(["Jan 2025", "Jan 2024", "Jan 2023"]):
            text(p, 260 + 90 * j, 116, v)
        data = [("Cost of revenue", "$", "178", "$", "141", "$", "138"), ("Research", "", "3,423", "", "2,532", "", "1,892"),
                ("Sales", "", "1,136", "", "876", "", "680")]
        for i, r in enumerate(data):
            text(p, 72, 136 + 16 * i, r[0])
            for k in range(3):
                if r[1 + 2 * k]:
                    text(p, 250 + 90 * k, 136 + 16 * i, r[1 + 2 * k])
                text(p, 290 + 90 * k, 136 + 16 * i, r[2 + 2 * k])
    d = convert(PDF().page(draw).save(tmp / "hdr.pdf"))
    b = _tables(d)[0]
    year = [c for c in b["cells"] if c["text"] == "Year Ended"][0]
    assert year["header"] and year["colspan"] == 3
    texts = {c["text"] for c in b["cells"]}
    assert "$178" in texts and "$" not in texts          # currency column folded into its value, not dropped


def test_markdown_escaping_and_line_breaks():
    tb = T.from_rows([["Name", "Value"], ["a|b", "x\ny"], ["back\\slash", ""]])
    md = T.to_markdown(tb)
    assert "a\\|b" in md
    assert "x y" in md and md.count("\n") == 3          # in-cell line break does not create a new row
    assert "back\\\\slash" in md
    assert md.splitlines()[3].endswith("|  |")          # empty cell preserved


def test_html_serialisation_escapes_and_spans():
    cells = [T.Cell(0, 0, "A & B", (0, 0, 1, 1), 1, 2, True), T.Cell(1, 0, "<x>", (0, 0, 1, 1)), T.Cell(1, 1, "2", (0, 0, 1, 1))]
    tb = T.Table((0, 0, 1, 1), 2, 2, cells, "test", 0.9)
    h = T.serialize(tb)
    assert h.startswith("<table>") and 'colspan="2"' in h and "A &amp; B" in h and "&lt;x&gt;" in h and "<th" in h


def test_never_invents_values():
    tb = T.from_rows([["h1", "h2", "h3"], ["1", "", "3"]])
    g = tb.grid()
    assert g[1] == ["1", "", "3"]
