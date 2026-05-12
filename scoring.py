"""
scoring.py
Opportunity scoring for Review2Roadmap.

Each theme is scored on five dimensions (all normalized to a 1–10 scale):

  Frequency      (25%) — how many customers report this issue?
  Severity       (30%) — how bad is the customer impact?
  Business Impact(20%) — how broadly does it affect the customer base?
  Confidence     (15%) — how consistently / reliably is it reported?
  NPS Risk       (10%) — how likely is this to hurt Net Promoter Score?

Final score = weighted sum, rounded to one decimal place.
Themes are then bucketed into P1 / P2 / P3 by score percentile.
"""

# Lookup tables for categorical fields → numeric weights
_SEVERITY_MAP = {"low": 2, "medium": 5, "high": 8, "critical": 10}
_IMPACT_MAP = {"individual": 3, "neighborhood": 6, "widespread": 10}


# ── Individual dimension scorers ──────────────────────────────────────────────

def _frequency(review_count: int, max_count: int) -> float:
    if max_count == 0:
        return 1.0
    return 1.0 + 9.0 * (review_count / max_count)


def _severity(reviews: list[dict]) -> float:
    if not reviews:
        return 1.0
    weights = [_SEVERITY_MAP.get(r.get("severity", "low"), 2) for r in reviews]
    return sum(weights) / len(weights)


def _business_impact(reviews: list[dict]) -> float:
    if not reviews:
        return 1.0
    scores = [_IMPACT_MAP.get(r.get("customer_impact", "individual"), 3) for r in reviews]
    return sum(scores) / len(scores)


def _confidence(review_count: int) -> float:
    # Each additional review raises confidence; caps at 10 around 7+ reviews.
    return min(10.0, 1.0 + (review_count - 1) * 1.5)


def _nps_risk(reviews: list[dict]) -> float:
    if not reviews:
        return 1.0
    high_sev_neg = sum(
        1 for r in reviews
        if r.get("severity") in ("high", "critical") and r.get("sentiment") == "negative"
    )
    proportion = high_sev_neg / len(reviews)
    return 1.0 + 9.0 * proportion


# ── Public API ────────────────────────────────────────────────────────────────

def score_theme(theme: dict, max_review_count: int) -> dict:
    """Add all dimension scores + final opportunity_score to a theme dict."""
    reviews = theme.get("reviews", [])
    n = theme.get("review_count", len(reviews))

    f = _frequency(n, max_review_count)
    s = _severity(reviews)
    b = _business_impact(reviews)
    c = _confidence(n)
    nps = _nps_risk(reviews)

    total = round(0.25 * f + 0.30 * s + 0.20 * b + 0.15 * c + 0.10 * nps, 1)

    return {
        **theme,
        "frequency_score": round(f, 1),
        "severity_score": round(s, 1),
        "business_impact_score": round(b, 1),
        "confidence_score": round(c, 1),
        "nps_risk_score": round(nps, 1),
        "opportunity_score": total,
    }


def rank_themes(themes: list[dict]) -> list[dict]:
    """Score, sort, and assign P1/P2/P3 priority labels to a list of themes."""
    if not themes:
        return themes

    max_count = max(t.get("review_count", 0) for t in themes) or 1
    scored = [score_theme(t, max_count) for t in themes]
    scored.sort(key=lambda t: t["opportunity_score"], reverse=True)

    n = len(scored)
    for i, t in enumerate(scored):
        if i < max(1, round(n * 0.33)):
            t["priority"] = "P1"
        elif i < max(2, round(n * 0.67)):
            t["priority"] = "P2"
        else:
            t["priority"] = "P3"

    return scored
