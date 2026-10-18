"""Document-level classification of running headers, footers, page numbers and margin stamps
("page furniture").

A fixed top/bottom margin is never the removal rule on its own. Each candidate gets a score from several
independent signals and a list of human-readable reasons:

  + position: inside the top/bottom band of the page (stronger at the extreme edge), or a rotated stamp in a side margin
  + isolation: separated from the body text by a gap larger than normal line spacing
  + repetition: the same text (digits normalised) recurs near the same position on other pages
  + page-number patterns: "12", "- 12 -", "xii", "Page 3 of 20", "p. 7", "12 | Journal ..."
  + publication furniture patterns: URLs, DOI/ISSN, (c)/copyright, "Downloaded from", "Download date",
    journal volume/issue lines, "Published by", repository boilerplate
  + font smaller than body text, short text
  - long text, captions, footnotes (marker + sentence), section headings, large-font titles (one-off content)

Only candidates scoring >= REMOVE are excluded from the body Markdown; everything stays in the JSON with its
bbox, role, score and reasons, and removal can be switched off (PARSEFLOW_REMOVE_FURNITURE=0 / CLI flag).
"""
import os
import re
from collections import defaultdict

REMOVE = 1.0

PAGE_NUM = re.compile(r"""^\s*(?:
      [-–—‒(\[]?\s*\d{1,4}\s*[-–—‒)\]]?            # 12, -12-, (12)
    | (?:page|pg\.?|p\.|seite|página|pagina|strona)\s*\d{1,4}(?:\s*(?:of|/|de|von|z)\s*\d{1,4})?   # Page 3 of 20
    | \d{1,4}\s*(?:/|of)\s*\d{1,4}                                               # 3/20, 3 of 20
    | [ivxlcdm]{1,7}                                                            # roman numerals
    )\s*$""", re.I | re.X)
NUM_WITH_TITLE = re.compile(r"^\s*(\d{1,4})\s*[|•·:]\s*\S.{0,120}$|^.{1,120}\S\s*[|•·:]\s*(\d{1,4})\s*$")
FURNITURE = re.compile(r"""(
      https?://|www\.|\.(?:com|org|net|edu|gov|ac\.[a-z]{2})\b|doi\s*:|\bdoi\.org|\bissn\b|\bisbn\b|e-?issn
    | ©|\(c\)\s*\d{4}|copyright|all\s+rights\s+reserved|licen[cs]ed?\s+under|creative\s+commons|open\s+access
    | download(?:ed)?\s+(?:from|date|on|by)|downloaded|retrieved\s+from|accessed\s+(?:on|from)
    | \bvol(?:ume)?\.?\s*\d|\bissue\s*\d|\bno\.\s*\d|\bpp\.\s*\d|\bjournal\b|\bproceedings\b|\btransactions\b
    | published\s+(?:by|online)|preprint|arxiv:|manuscript|author\s+manuscript|submitted\s+to|under\s+review
    | \brevision\b|\brev\.\s*\d|\bp/n\b|\bpart\s+no\b|confidential|proprietary|draft
    | @[a-z0-9-]+\.[a-z]|\bpage\s+\d+
    | \b(?:19|20)\d{2}\b.*\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2},?\s+(?:19|20)\d{2}
    | \b\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+(?:19|20)\d{2}
    )""", re.I | re.X)
FOOTNOTE = re.compile(r"^\s*(?:\d{1,3}|[*†‡§¶]|\[\d+\]|[a-z])\s*[\)\.]?\s*[A-ZÀ-ɏ].{20,}")
CAPTION = re.compile(r"^\s*(fig(?:ure)?|table|chart|plate|scheme)\.?\s*\d", re.I)
SECTION = re.compile(r"^\s*(?:\d+(?:\.\d+)*\.?|[IVX]+\.|[A-Z]\.)\s+[A-Z][^\s]*(?:\s+\S+){0,8}\s*$")


def _norm(t):
    t = re.sub(r"\d+", "#", (t or "").lower())
    t = re.sub(r"[\W_]+", " ", t).strip()
    # a page number on the outer edge moves sides between even and odd pages ('310 Chapter 17' / 'Chapter 17 311')
    return re.sub(r"^(?:# )+|(?: #)+$", "", t).strip() or t


def _rep_hits(rep, k, zone, pi, cy_rel, n_pages):
    """Other pages with the same normalised text in the same band at (nearly) the same height. In long documents a
    running header only spans its chapter, so the requirement is capped at 4 pages instead of 30% of the document."""
    hits = len({p for p, c in rep[(k, zone)] if p != pi and abs(c - cy_rel) < 0.03})
    need = min(0.3 * (n_pages - 1), 4)
    return hits, hits >= 1 and hits >= need


def enabled():
    return os.getenv("PARSEFLOW_REMOVE_FURNITURE", "1").lower() not in ("0", "false", "no", "off")


def classify(pages, body_size):
    """pages: list of dict(width, height, items=[dict(id, bbox(pt), text, size, rotated(bool), kind)]).
    Returns {item_id: dict(role, score, reasons, excluded)} for every item that scored > 0.3."""
    n_pages = len(pages)
    # repetition index: normalised text -> set of (page, zone)
    rep = defaultdict(set)
    for pi, p in enumerate(pages):
        H = p["height"]
        for it in p["items"]:
            cy = (it["bbox"][1] + it["bbox"][3]) / 2
            zone = "top" if cy < 0.2 * H else "bottom" if cy > 0.8 * H else "mid"
            k = _norm(it["text"])
            if k and zone != "mid":
                rep[(k, zone)].add((pi, round(cy / H, 3)))

    out = {}
    for pi, p in enumerate(pages):
        W, H, items = p["width"], p["height"], p["items"]
        body = [it for it in items if it.get("kind") == "text"]
        # rows of edge-band items sharing a baseline (header/footer lines are often 2-3 short pieces)
        rowmates = {}
        edge = [it for it in body if it["bbox"][3] <= 0.16 * H or it["bbox"][1] >= 0.84 * H]
        for it in edge:
            cy = (it["bbox"][1] + it["bbox"][3]) / 2
            h = max(1.0, it["bbox"][3] - it["bbox"][1])
            rowmates[id(it)] = [o for o in edge if o is not it and abs((o["bbox"][1] + o["bbox"][3]) / 2 - cy) < 0.5 * h]
        for it in items:
            if it.get("kind") != "text":
                continue
            x0, y0, x1, y1 = it["bbox"]
            text = (it["text"] or "").strip()
            if not text:
                continue
            words = len(text.split())
            cy = (y0 + y1) / 2
            score, reasons, role = 0.0, [], None
            strong = False      # at least one structural signal is required for removal
            top = y1 <= 0.12 * H or (y0 <= 0.07 * H and y1 <= 0.16 * H)
            bottom = y0 >= 0.88 * H or (y1 >= 0.93 * H and y0 >= 0.84 * H)
            if not top and not bottom and words <= 12 and PAGE_NUM.match(text):
                others = [o for o in body if o is not it]
                # pages whose content ends (or starts) far from the paper edge: the last/first isolated line
                # of the content is the edge for page-number purposes
                below = [o for o in others if o["bbox"][1] >= y1 - 1]
                above = [o for o in others if o["bbox"][3] <= y0 + 1]
                gap_above = min((y0 - o["bbox"][3] for o in above), default=H)
                gap_below = min((o["bbox"][1] - y1 for o in below), default=H)
                if others and not below and y0 >= 0.45 * H and gap_above > 1.3 * body_size:
                    bottom = True
                elif others and not above and y1 <= 0.55 * H and gap_below > 1.3 * body_size:
                    top = True
            side = it.get("rotated") and (x1 <= 0.12 * W or x0 >= 0.88 * W)
            if it.get("rotated") and not side and (x1 <= 0.2 * W or x0 >= 0.8 * W):
                side = True
            if not (top or bottom or side):
                continue
            role = "header" if top else "footer" if bottom else "margin"
            if side:
                score += 0.7; reasons.append("rotated text in the side margin"); strong = True
            else:
                score += 0.35; reasons.append(f"in the {'top' if top else 'bottom'} band of the page")
                if (top and y1 <= 0.06 * H) or (bottom and y0 >= 0.94 * H):
                    score += 0.2; reasons.append("at the extreme page edge"); strong = True
                # isolation from body text
                others = [o for o in body if o is not it]
                if top:
                    below = [o["bbox"][1] - y1 for o in others if o["bbox"][1] >= y1 - 1]
                    gap = min(below) if below else H
                else:
                    above = [y0 - o["bbox"][3] for o in others if o["bbox"][3] <= y0 + 1]
                    gap = min(above) if above else H
                if gap > 1.3 * body_size:
                    score += 0.3; reasons.append(f"separated from body text by {gap:.0f}pt"); strong = True
            mates = rowmates.get(id(it), [])
            with_pnum = False
            if mates and not side:
                row = [it] + mates
                ext = max(o["bbox"][2] for o in row) - min(o["bbox"][0] for o in row)
                if any(PAGE_NUM.match((o["text"] or "").strip()) for o in mates) and words <= 12:
                    score += 0.6; reasons.append("shares its line with a page number"); strong = True; with_pnum = True
                elif all(len((o["text"] or "").split()) <= 8 for o in row) and ext >= 0.45 * W:
                    score += 0.4; reasons.append(f"one of {len(row)} short items spread across an edge line"); strong = True
            # page numbers
            if PAGE_NUM.match(text) and words <= 6:
                score += 1.0; reasons.append("page-number pattern"); role = "page_number"; strong = True
            elif NUM_WITH_TITLE.match(text) and words <= 16:
                score += 0.5; reasons.append("number beside a running title")
            # repetition across pages
            k = _norm(text)
            zone = "top" if cy < 0.2 * H else "bottom" if cy > 0.8 * H else "mid"
            repeats = False
            if n_pages >= 2 and k and zone != "mid":
                hits, repeats = _rep_hits(rep, k, zone, pi, cy / H, n_pages)
                if repeats:
                    score += 0.8; reasons.append(f"repeats on {hits} other page(s)"); strong = True
            if FURNITURE.search(text):
                score += 0.55; reasons.append("publication/boilerplate pattern (URL, DOI, copyright, journal, date, revision...)")
            sz = it.get("size") or body_size
            if sz <= 0.92 * body_size:
                score += 0.25; reasons.append("smaller than body text")
            if words <= 14:
                score += 0.15; reasons.append("short")
            # protective signals: one-off content near the edges
            if words > 40:
                score -= 1.0; reasons.append("long text (likely content)")
            elif words > 25:
                score -= 0.45; reasons.append("fairly long text")
            if CAPTION.match(text):
                score -= 1.0; reasons.append("caption")
            if bottom and FOOTNOTE.match(text) and not FURNITURE.search(text) and not repeats:
                score -= 0.8; reasons.append("footnote (marker + sentence)")
            if sz >= 1.3 * body_size and top and not FURNITURE.search(text) and not PAGE_NUM.match(text) and not with_pnum and not repeats:
                score -= 0.8; reasons.append("large one-off text (title/heading)")
            elif n_pages >= 3 and top and sz >= 1.05 * body_size and not repeats and not FURNITURE.search(text) \
                    and not PAGE_NUM.match(text) and not NUM_WITH_TITLE.match(text) and not with_pnum:
                # long document: a heading-sized line that appears on no other page is that page's title, not a running head
                score -= 0.35; reasons.append("heading-sized text that does not repeat on other pages (page title)")
            if SECTION.match(text) and not PAGE_NUM.match(text) and not FURNITURE.search(text) and top:
                score -= 0.5; reasons.append("looks like a section heading")
            if not strong and score >= REMOVE:
                score = REMOVE - 0.05; reasons.append("no structural signal (isolation, repetition, page number, margin) - kept")
            if score > 0.3:
                out[it["id"]] = dict(role=role, score=round(score, 2), reasons=reasons, excluded=bool(score >= REMOVE and enabled()))
    return out
