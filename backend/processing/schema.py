"""Canonical Digital Twin schema helpers. Every processor emits blocks through make_block().

Block contract (all keys are always present; extra keys are optional and additive):
    block_id, type, content, page, bbox [x0,y0,x1,y1] in page-image pixels (PDF points * Z),
    reading_order, confidence, confidence_type, extractor, verification, relationships, warnings

Verification statuses (kept from v1 so the frontend and existing JSON stay compatible):
    "verified" - independent evidence (or document-encoded text with passing quality checks) supports it
    "review"   - uncertain: plausible but not independently confirmed; a human should check it
    "failed"   - extraction failed or contradicted by evidence; the region is preserved, not guessed

Optional evidence-trail keys (added in v2, all additive):
    evidence           short human-readable summary (string, as in v1)
    evidence_detail    dict(basis=..., independent=bool, signals=[...], confidence_method=...)
    raw_text           the extractor's original output before any normalisation
    transforms         list of normalisation / reconstruction steps applied to raw_text
    role               "body" | "header" | "footer" | "page_number" | "margin" (page furniture)
    furniture          dict(score, reasons, excluded) when the block was classified as page furniture
    cells              structured table cells (see tables.py)
    latex, math_display for equations
"""
OK, REVIEW = 0.85, 0.60
Z = 2.0   # page-image pixels per PDF point (bbox coordinate space shared with the frontend viewer)


def verdict(conf):
    return "verified" if conf >= OK else "review" if conf >= REVIEW else "failed"


def make_block(bid, type_, page, bbox, conf, ctype, extractor, content=None, **extra):
    b = dict(block_id=bid, type=type_, content=content, page=page,
             bbox=[int(round(v)) for v in bbox], reading_order=0,
             confidence=round(float(conf), 2), confidence_type=ctype, extractor=extractor,
             verification="review", relationships=[], warnings=[])
    b.update({k: v for k, v in extra.items() if v is not None})
    return b


def area(r):  # r = (x0,y0,x1,y1)
    return max(0, r[2]-r[0]) * max(0, r[3]-r[1])


def inter(a, b):
    return area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))


def overlap_frac(a, b):
    """fraction of a covered by b"""
    return inter(a, b) / area(a) if area(a) else 0


def union(a, b):
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def scale(r, z=Z):
    return tuple(v * z for v in r)
