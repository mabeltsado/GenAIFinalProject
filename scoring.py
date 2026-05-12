"""
scoring.py
Opportunity scoring model for Review2Roadmap.

Each product opportunity theme is scored across five dimensions on a 1–5 scale.
The five scores are combined using goal-specific weights to produce a final
priority score on a 0–100 scale.

Dimensions
----------
  frequency_score      — how many customers report this theme?
  severity_score       — how bad is the customer impact on average?
  nps_risk_score       — how low is the average NPS for this theme?
  business_impact_score— how directly does this theme affect the selected goal?
  confidence_score     — how consistently / reliably is the theme reported?

Goal-specific weights
---------------------
Different product goals shift emphasis across dimensions. For example,
"Reduce churn" weights NPS risk and severity heavily, while
"Find quick wins" weights confidence and business impact more.

Public API
----------
  calculate_priority_score(theme, product_goal)
      → {priority_score, score_breakdown, priority_level}

  explain_score(theme, product_goal)
      → plain-English string explaining why a theme was ranked where it was

  rank_themes(themes, product_goal)
      → themes sorted by priority_score, with labels added in-place
"""

# ── Goal-specific weights ─────────────────────────────────────────────────────
# All five weights in each goal profile must sum to exactly 1.0.

GOAL_WEIGHTS: dict[str, dict[str, float]] = {
    "Reduce churn": {
        "frequency":       0.20,
        "severity":        0.25,
        "nps_risk":        0.25,
        "business_impact": 0.20,
        "confidence":      0.10,
    },
    "Improve customer satisfaction": {
        "frequency":       0.25,
        "severity":        0.20,
        "nps_risk":        0.20,
        "business_impact": 0.20,
        "confidence":      0.15,
    },
    "Reduce support tickets": {
        "frequency":       0.30,
        "severity":        0.15,
        "nps_risk":        0.15,
        "business_impact": 0.25,
        "confidence":      0.15,
    },
    "Identify urgent bugs or service failures": {
        "frequency":       0.15,
        "severity":        0.35,
        "nps_risk":        0.15,
        "business_impact": 0.25,
        "confidence":      0.10,
    },
    "Find quick wins for the product roadmap": {
        "frequency":       0.20,
        "severity":        0.15,
        "nps_risk":        0.10,
        "business_impact": 0.25,
        "confidence":      0.30,
    },
}

# Fallback when an unrecognised goal is passed
_DEFAULT_GOAL = "Improve customer satisfaction"

# ── Business impact lookup table ──────────────────────────────────────────────
# Maps (product_goal, opportunity_type) → business impact score 1–5.
# A score of 5 means this opportunity type is a direct driver of the goal.
# A score of 1 means it is largely unrelated to the goal.

_BUSINESS_IMPACT_TABLE: dict[str, dict[str, int]] = {
    "Reduce churn": {
        "retention risk":        5,  # directly about customers leaving
        "outage communication":  4,  # prolonged outages drive churn
        "support experience":    4,  # poor service = cancellations
        "high bill investigation":4, # shock bills trigger switching
        "renewal education":     4,  # customers leave at contract end
        "billing transparency":  3,
        "pricing clarity":       3,
        "payment reliability":   3,
        "plan education":        2,
        "app improvement":       2,
        "green energy messaging":1,
    },
    "Improve customer satisfaction": {
        "support experience":    5,  # support quality is the #1 CSAT driver
        "outage communication":  5,
        "billing transparency":  4,
        "high bill investigation":4,
        "app improvement":       4,
        "payment reliability":   4,
        "pricing clarity":       3,
        "plan education":        3,
        "green energy messaging":3,
        "renewal education":     3,
        "retention risk":        3,
    },
    "Reduce support tickets": {
        "billing transparency":  5,  # billing is the #1 inbound call driver
        "high bill investigation":5,
        "app improvement":       5,  # self-service deflects calls
        "payment reliability":   4,
        "plan education":        4,
        "outage communication":  4,  # outage calls spike without proactive comms
        "pricing clarity":       3,
        "renewal education":     3,
        "support experience":    3,  # fixing support UX reduces repeat contacts
        "retention risk":        2,
        "green energy messaging":1,
    },
    "Identify urgent bugs or service failures": {
        "outage communication":  5,  # operational failure
        "app improvement":       5,  # broken product = bug
        "payment reliability":   5,  # payment failures = service failure
        "high bill investigation":4, # billing bugs
        "support experience":    3,
        "billing transparency":  3,
        "pricing clarity":       2,
        "retention risk":        2,
        "renewal education":     1,
        "plan education":        1,
        "green energy messaging":1,
    },
    "Find quick wins for the product roadmap": {
        "billing transparency":  5,  # adding a bill breakdown = fast content win
        "app improvement":       5,  # many small UX fixes ship quickly
        "plan education":        4,  # FAQ / explainer content
        "green energy messaging":4,  # content marketing
        "pricing clarity":       4,  # FAQ update
        "outage communication":  3,  # notification setup = medium effort
        "payment reliability":   3,
        "renewal education":     3,
        "support experience":    2,  # hard to fix support culture quickly
        "high bill investigation":2,
        "retention risk":        1,  # strategic, not a quick win
    },
}

_DEFAULT_IMPACT = 2  # fallback for unrecognised opportunity types


# ── Dimension scorers (all return int 1–5) ─────────────────────────────────────

def _frequency_score(theme: dict) -> int:
    """
    Score 1–5 based on the number of customer reviews in the theme.
    More reviews = stronger evidence that this is a widespread issue.
    """
    n = theme.get("review_count", 0)
    if n >= 11:
        return 5
    if n >= 7:
        return 4
    if n >= 4:
        return 3
    if n >= 2:
        return 2
    return 1


def _severity_score(theme: dict) -> int:
    """
    Score 1–5 based on the average severity of reviews in the theme.
    Uses avg_severity (float 1–5) pre-computed by cluster_themes().
    Falls back to dominant_severity string if avg_severity is not present.
    """
    # Prefer the numeric average computed by the pipeline
    if "avg_severity" in theme:
        return max(1, min(5, round(float(theme["avg_severity"]))))

    # Fallback: map dominant severity string to a score
    string_map = {"critical": 5, "high": 4, "medium": 3, "low": 2}
    return string_map.get(theme.get("dominant_severity", "medium"), 3)


def _nps_risk_score(theme: dict) -> int:
    """
    Score 1–5 based on how low the theme's average NPS score is.
    NPS is on a 0–10 scale; a lower NPS = higher churn risk = higher score.

    Thresholds (avg NPS → risk score):
      < 2  → 5  (promoters absent, almost all detractors)
      < 4  → 4
      < 6  → 3
      < 8  → 2
      ≥ 8  → 1  (mostly promoters, low risk)
    """
    avg_nps = float(theme.get("avg_nps_score", 5.0))
    if avg_nps < 2:
        return 5
    if avg_nps < 4:
        return 4
    if avg_nps < 6:
        return 3
    if avg_nps < 8:
        return 2
    return 1


def _business_impact_score(theme: dict, product_goal: str) -> int:
    """
    Score 1–5 based on how directly this opportunity type addresses the
    product manager's selected goal.

    Looks up (product_goal, opportunity_type) in the business impact table.
    Returns the default impact score (2) for unrecognised combinations.
    """
    goal_table = _BUSINESS_IMPACT_TABLE.get(
        product_goal,
        _BUSINESS_IMPACT_TABLE[_DEFAULT_GOAL],
    )
    opp_type = theme.get("opportunity_type", "")
    return goal_table.get(opp_type, _DEFAULT_IMPACT)


def _confidence_score(theme: dict) -> int:
    """
    Score 1–5 based on the volume and consistency of supporting evidence.

    A theme with many reviews that all report similar severity is scored higher
    than a theme with one or two reviews, or reviews with wildly different
    severity levels (low consistency).

    Combines:
      - confidence_level from cluster_themes() ("high" / "medium" / "low")
      - review_count as a volume proxy
    """
    n    = theme.get("review_count", 0)
    conf = theme.get("confidence_level", "low")

    if conf == "high" and n >= 5:
        return 5
    if conf == "high" and n >= 3:
        return 4
    if conf == "high":
        return 3
    if conf == "medium" and n >= 3:
        return 3
    if conf == "medium":
        return 2
    return 1  # low confidence or single review


# ── Main scoring function ──────────────────────────────────────────────────────

def calculate_priority_score(theme: dict, product_goal: str) -> dict:
    """
    Calculate a priority score for one product opportunity theme.

    Each of the five dimension scores (1–5) is multiplied by its
    goal-specific weight. The weighted sum (also 1–5) is then normalised
    to a 0–100 scale using:

        priority_score = (weighted_sum - 1) / 4 * 100

    This ensures a theme that scores 1 on every dimension gets 0/100,
    and a theme that scores 5 on every dimension gets 100/100.

    Parameters
    ----------
    theme : dict
        A theme dict from cluster_themes(), containing at minimum:
        review_count, avg_nps_score, avg_severity, confidence_level,
        and opportunity_type.
    product_goal : str
        One of the five product goals defined in GOAL_WEIGHTS.

    Returns
    -------
    dict
        priority_score   : float, 0–100
        score_breakdown  : dict with each dimension score (1–5) and
                           the weights used
        priority_level   : "High", "Medium", or "Low"
    """
    weights = GOAL_WEIGHTS.get(product_goal, GOAL_WEIGHTS[_DEFAULT_GOAL])

    # Calculate each dimension
    f = _frequency_score(theme)
    s = _severity_score(theme)
    n = _nps_risk_score(theme)
    b = _business_impact_score(theme, product_goal)
    c = _confidence_score(theme)

    # Weighted sum → normalise to 0–100
    weighted_sum = (
        f * weights["frequency"]       +
        s * weights["severity"]        +
        n * weights["nps_risk"]        +
        b * weights["business_impact"] +
        c * weights["confidence"]
    )
    raw_score    = (weighted_sum - 1) / 4 * 100
    priority_score = round(max(0.0, min(100.0, raw_score)), 1)

    # Priority tier
    if priority_score >= 67:
        priority_level = "High"
    elif priority_score >= 34:
        priority_level = "Medium"
    else:
        priority_level = "Low"

    return {
        "priority_score": priority_score,
        "priority_level": priority_level,
        "score_breakdown": {
            "frequency_score":       f,
            "severity_score":        s,
            "nps_risk_score":        n,
            "business_impact_score": b,
            "confidence_score":      c,
            "weights_used":          weights,
            "weighted_sum":          round(weighted_sum, 3),
        },
    }


# ── Plain-English explanation ──────────────────────────────────────────────────

def explain_score(theme: dict, product_goal: str) -> str:
    """
    Generate a plain-English explanation of why a theme received its score.

    Designed for display in the app so product managers can quickly understand
    the reasoning without reading raw numbers.

    Parameters
    ----------
    theme : dict
        A scored theme dict (output of rank_themes or calculate_priority_score).
    product_goal : str
        The selected product goal.

    Returns
    -------
    str
        A 2–4 sentence explanation of the score.
    """
    result     = calculate_priority_score(theme, product_goal)
    score      = result["priority_score"]
    level      = result["priority_level"]
    bd         = result["score_breakdown"]
    name       = theme.get("name", "This theme")
    n_reviews  = theme.get("review_count", 0)
    avg_nps    = theme.get("avg_nps_score", 5.0)
    opp_type   = theme.get("opportunity_type", "")

    # Frequency sentence
    freq_s = bd["frequency_score"]
    if freq_s >= 4:
        freq_line = f"{n_reviews} customers reported this, indicating a widespread issue."
    elif freq_s == 3:
        freq_line = f"{n_reviews} customers reported this — a moderate volume."
    else:
        freq_line = f"Only {n_reviews} customer(s) reported this, so evidence is limited."

    # Severity sentence
    sev_s = bd["severity_score"]
    sev_labels = {5: "critical", 4: "high", 3: "medium", 2: "low", 1: "very low"}
    sev_line = f"Impact severity is rated {sev_labels.get(sev_s, 'medium')} ({sev_s}/5)."

    # NPS risk sentence
    nps_s = bd["nps_risk_score"]
    if nps_s >= 4:
        nps_line = (
            f"The average NPS of {avg_nps:.1f}/10 in this group signals serious churn risk."
        )
    elif nps_s == 3:
        nps_line = f"The average NPS of {avg_nps:.1f}/10 indicates moderate dissatisfaction."
    else:
        nps_line = f"The average NPS of {avg_nps:.1f}/10 suggests customers are not at risk of churning."

    # Business impact sentence
    biz_s = bd["business_impact_score"]
    if biz_s >= 4:
        biz_line = f"Resolving '{opp_type}' is a direct lever for '{product_goal}'."
    elif biz_s == 3:
        biz_line = f"This opportunity is moderately relevant to '{product_goal}'."
    else:
        biz_line = f"This opportunity has limited direct impact on '{product_goal}'."

    return (
        f"**{name}** scored **{level} ({score}/100)** under '{product_goal}'. "
        f"{freq_line} {sev_line} {nps_line} {biz_line}"
    )


# ── rank_themes wrapper ────────────────────────────────────────────────────────

def rank_themes(themes: list[dict], product_goal: str = _DEFAULT_GOAL) -> list[dict]:
    """
    Score and rank a list of themes by priority score.

    Calls calculate_priority_score() for every theme, adds the scoring fields
    to each theme dict in-place, sorts the list highest → lowest, and assigns
    priority tier labels (P1 / P2 / P3 and High / Medium / Low).

    Also sets opportunity_score = priority_score so that any code that reads
    the old field name still works correctly.

    Parameters
    ----------
    themes : list[dict]
        Themes from cluster_themes().
    product_goal : str
        The selected product goal. Defaults to "Improve customer satisfaction".

    Returns
    -------
    list[dict]
        The same list, mutated and sorted.
    """
    if not themes:
        return themes

    for theme in themes:
        result = calculate_priority_score(theme, product_goal)
        bd     = result["score_breakdown"]

        # Flatten scoring results onto the theme dict
        theme["priority_score"]        = result["priority_score"]
        theme["priority_level"]        = result["priority_level"]
        theme["frequency_score"]       = bd["frequency_score"]
        theme["severity_score"]        = bd["severity_score"]
        theme["nps_risk_score"]        = bd["nps_risk_score"]
        theme["business_impact_score"] = bd["business_impact_score"]
        theme["confidence_score"]      = bd["confidence_score"]
        theme["score_breakdown"]       = bd
        theme["score_explanation"]     = explain_score(theme, product_goal)

        # Backward-compatible alias (old code reads opportunity_score)
        theme["opportunity_score"] = result["priority_score"]

    themes.sort(key=lambda t: t["priority_score"], reverse=True)

    # Assign P1 / P2 / P3 labels based on priority_level
    for theme in themes:
        level = theme.get("priority_level", "Low")
        theme["priority"] = {"High": "P1", "Medium": "P2", "Low": "P3"}.get(level, "P3")

    return themes


# ── Self-contained tests ───────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Quick smoke-tests. Run with:  python scoring.py
    No external dependencies required.
    """
    print("=" * 60)
    print("scoring.py — self-test")
    print("=" * 60)

    # --- Sample themes ---
    sample_themes = [
        {
            "name":             "Outage Communication",
            "opportunity_type": "outage communication",
            "review_count":     8,
            "avg_nps_score":    2.5,
            "avg_severity":     4.3,
            "dominant_severity":"high",
            "confidence_level": "high",
        },
        {
            "name":             "Billing Transparency",
            "opportunity_type": "billing transparency",
            "review_count":     5,
            "avg_nps_score":    4.0,
            "avg_severity":     3.2,
            "dominant_severity":"medium",
            "confidence_level": "medium",
        },
        {
            "name":             "Green Energy Messaging",
            "opportunity_type": "green energy messaging",
            "review_count":     2,
            "avg_nps_score":    7.5,
            "avg_severity":     1.5,
            "dominant_severity":"low",
            "confidence_level": "low",
        },
        {
            "name":             "Retention Risk",
            "opportunity_type": "retention risk",
            "review_count":     3,
            "avg_nps_score":    1.3,
            "avg_severity":     4.7,
            "dominant_severity":"critical",
            "confidence_level": "medium",
        },
    ]

    goals_to_test = [
        "Reduce churn",
        "Identify urgent bugs or service failures",
        "Find quick wins for the product roadmap",
    ]

    for goal in goals_to_test:
        print(f"\n{'─' * 60}")
        print(f"Goal: {goal}")
        print(f"{'─' * 60}")

        ranked = rank_themes([t.copy() for t in sample_themes], product_goal=goal)

        for i, t in enumerate(ranked, 1):
            print(
                f"  {i}. [{t['priority']}] {t['name']:30s} "
                f"score={t['priority_score']:5.1f}  "
                f"f={t['frequency_score']} s={t['severity_score']} "
                f"n={t['nps_risk_score']} b={t['business_impact_score']} "
                f"c={t['confidence_score']}"
            )

    # --- Detailed breakdown for one theme ---
    print(f"\n{'─' * 60}")
    print("Detailed score for 'Outage Communication' under 'Reduce churn'")
    print(f"{'─' * 60}")
    test_theme = sample_themes[0].copy()
    result = calculate_priority_score(test_theme, "Reduce churn")
    print(f"  priority_score  : {result['priority_score']}")
    print(f"  priority_level  : {result['priority_level']}")
    print(f"  score_breakdown :")
    for k, v in result["score_breakdown"].items():
        if k != "weights_used":
            print(f"    {k}: {v}")
    print(f"\n  Explanation:\n  {explain_score(test_theme, 'Reduce churn')}")

    # --- Verify weights sum to 1.0 for every goal ---
    print(f"\n{'─' * 60}")
    print("Weight validation")
    print(f"{'─' * 60}")
    all_ok = True
    for goal, w in GOAL_WEIGHTS.items():
        total = round(sum(w.values()), 10)
        status = "✅" if total == 1.0 else "❌"
        print(f"  {status} {goal}: weights sum = {total}")
        if total != 1.0:
            all_ok = False
    print(f"\n  All weight sets valid: {'YES' if all_ok else 'NO — CHECK ABOVE'}")
    print("=" * 60)
