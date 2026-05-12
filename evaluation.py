"""
evaluation.py
Lightweight evaluation framework for Review2Roadmap.

Design principle: keep scoring transparent and explainable.
Every score is derived from rule-based checks against the output text or
structured data — no model-as-judge, no embeddings, no external calls.
A human reviewer can verify any score by reading the check that produced it.

Public API
----------
  RUBRIC                          — 6-dimension rubric dict
  TEST_CASES                      — 5 built-in test cases
  evaluate_output(output, case)   — score one output dict against one test case
  compare_baseline_vs_agentic(baseline_output, agentic_output)
                                  — side-by-side comparison dict
  export_evaluation_results(results, path)
                                  — write results to CSV
"""

import csv
import os
from datetime import datetime

# ── Evaluation rubric ─────────────────────────────────────────────────────────
# Six dimensions, each scored 1–5 by a human or by evaluate_output().
# Descriptions explain what each score level means in plain language.

RUBRIC: dict[str, dict] = {
    "theme_accuracy": {
        "dimension":   "Theme Accuracy",
        "question":    "Does the output correctly identify the dominant issue in the reviews?",
        "score_1":     "Wrong theme identified, or no theme at all",
        "score_2":     "Partially correct — right area but wrong framing",
        "score_3":     "Correct theme identified but missing key sub-issues",
        "score_4":     "Correct theme with most sub-issues captured",
        "score_5":     "Correct theme, all major sub-issues named, well-labelled",
        "baseline_typical": 2,
        "agentic_typical":  4,
    },
    "completeness": {
        "dimension":   "Completeness",
        "question":    "Does the output cover all major issues in the input, not just the loudest ones?",
        "score_1":     "Only the most obvious issue mentioned",
        "score_2":     "2 of 5 expected issues surfaced",
        "score_3":     "3 of 5 expected issues surfaced",
        "score_4":     "4 of 5 expected issues surfaced",
        "score_5":     "All expected issues surfaced with supporting detail",
        "baseline_typical": 2,
        "agentic_typical":  4,
    },
    "evidence_grounding": {
        "dimension":   "Evidence Grounding",
        "question":    "Are claims supported by direct quotes or review counts? Is anything invented?",
        "score_1":     "No evidence — all claims are assertions",
        "score_2":     "One paraphrased example, no verbatim quotes",
        "score_3":     "Some verbatim quotes present but not consistently tied to claims",
        "score_4":     "Most claims backed by quotes or counts",
        "score_5":     "Every claim tied to a verbatim quote or review count; nothing invented",
        "baseline_typical": 1,
        "agentic_typical":  5,
    },
    "actionability": {
        "dimension":   "Actionability",
        "question":    "Could a PM hand this to a developer or ops team and start work this sprint?",
        "score_1":     "Vague observations only — no recommended actions",
        "score_2":     "General direction given but no concrete steps",
        "score_3":     "Clear recommendations but not structured as backlog items",
        "score_4":     "Structured action items present; missing acceptance criteria",
        "score_5":     "Sprint-ready backlog cards with user story, acceptance criteria, and owner",
        "baseline_typical": 2,
        "agentic_typical":  5,
    },
    "prioritization_quality": {
        "dimension":   "Prioritization Quality",
        "question":    "Are issues ranked clearly? Is the ranking method transparent?",
        "score_1":     "No ranking — all issues treated equally",
        "score_2":     "Implicit ranking (order of mention) with no explanation",
        "score_3":     "Explicit ranking present but method not explained",
        "score_4":     "Ranked with P1/P2/P3 labels; scoring method mentioned",
        "score_5":     "Quantified scores with transparent weights; goal-aligned ranking",
        "baseline_typical": 1,
        "agentic_typical":  5,
    },
    "governance_and_uncertainty": {
        "dimension":   "Governance and Uncertainty Handling",
        "question":    "Does the output flag uncertainty and require human review before action?",
        "score_1":     "Fully automated output, no caveats, no human review step",
        "score_2":     "Generic disclaimer present but no specific uncertainty flags",
        "score_3":     "Low-confidence items flagged; no human approval gate",
        "score_4":     "Human review notes present on each output item",
        "score_5":     "Explicit human approval required before any external action; "
                       "uncertainty and data quality caveats called out per item",
        "baseline_typical": 1,
        "agentic_typical":  5,
    },
}

# ── Built-in test cases ───────────────────────────────────────────────────────
# 5 cases, each a small cluster of synthetic reviews with a known dominant theme.
# evaluate_output() checks whether the pipeline's output correctly surfaces that theme.
# Reviews use the same schema as the uploaded CSV so they can feed into the pipeline.

TEST_CASES: list[dict] = [
    {
        "case_id":        "TC-01",
        "label":          "Billing confusion",
        "dominant_theme": "billing transparency",
        "expected_keywords": ["bill", "billing", "breakdown", "charge", "statement"],
        "reviews": [
            {
                "customer_first_name": "Alice", "customer_last_name": "Moore",
                "customer_id": "TC01-A", "survey_completed_at": "2025-03-01",
                "recommendation_score_0_to_10": 2,
                "written_explanation": "My bill doubled this quarter and the portal shows no breakdown of the charges.",
                "electricity_plan": "Standard", "customer_tenure": "2 years",
            },
            {
                "customer_first_name": "Brian", "customer_last_name": "Carter",
                "customer_id": "TC01-B", "survey_completed_at": "2025-03-02",
                "recommendation_score_0_to_10": 3,
                "written_explanation": "I cannot tell from the bill why I was charged a reconnection fee. The itemisation is completely missing.",
                "electricity_plan": "Basic", "customer_tenure": "6 months",
            },
            {
                "customer_first_name": "Carol", "customer_last_name": "Evans",
                "customer_id": "TC01-C", "survey_completed_at": "2025-03-03",
                "recommendation_score_0_to_10": 2,
                "written_explanation": "The billing statement has 3 line items with no explanation. I have no idea what I am paying for.",
                "electricity_plan": "Standard", "customer_tenure": "3 years",
            },
        ],
    },
    {
        "case_id":        "TC-02",
        "label":          "High bill complaints",
        "dominant_theme": "high bill investigation",
        "expected_keywords": ["bill jumped", "bill doubled", "way too high", "unexpected bill", "massive bill", "bill went up", "huge bill"],
        "reviews": [
            {
                "customer_first_name": "David", "customer_last_name": "Harris",
                "customer_id": "TC02-A", "survey_completed_at": "2025-03-04",
                "recommendation_score_0_to_10": 1,
                "written_explanation": "My bill jumped 45% this month with no explanation. My usage has not changed at all.",
                "electricity_plan": "Standard", "customer_tenure": "1 year",
            },
            {
                "customer_first_name": "Emma", "customer_last_name": "Lewis",
                "customer_id": "TC02-B", "survey_completed_at": "2025-03-05",
                "recommendation_score_0_to_10": 2,
                "written_explanation": "Estimated bill was way too high — $150 above actual usage. Why estimate for 6 months without a meter read?",
                "electricity_plan": "Basic", "customer_tenure": "4 years",
            },
            {
                "customer_first_name": "Frank", "customer_last_name": "Robinson",
                "customer_id": "TC02-C", "survey_completed_at": "2025-03-06",
                "recommendation_score_0_to_10": 1,
                "written_explanation": "Bill went up 60% year on year. Nobody has contacted me to explain why.",
                "electricity_plan": "Premium", "customer_tenure": "2 years",
            },
        ],
    },
    {
        "case_id":        "TC-03",
        "label":          "Mobile app payment issues",
        "dominant_theme": "app improvement",
        "expected_keywords": ["app", "portal", "crash", "payment", "mobile", "website", "login"],
        "reviews": [
            {
                "customer_first_name": "Grace", "customer_last_name": "Walker",
                "customer_id": "TC03-A", "survey_completed_at": "2025-03-07",
                "recommendation_score_0_to_10": 1,
                "written_explanation": "The mobile app crashes every time I try to pay my bill. Been happening for 3 weeks.",
                "electricity_plan": "Standard", "customer_tenure": "1 year",
            },
            {
                "customer_first_name": "Henry", "customer_last_name": "Young",
                "customer_id": "TC03-B", "survey_completed_at": "2025-03-08",
                "recommendation_score_0_to_10": 2,
                "written_explanation": "Why can I not pay my bill in the app? I have to go to the website every single time. Completely useless on iOS.",
                "electricity_plan": "Basic", "customer_tenure": "8 months",
            },
            {
                "customer_first_name": "Iris", "customer_last_name": "Allen",
                "customer_id": "TC03-C", "survey_completed_at": "2025-03-09",
                "recommendation_score_0_to_10": 1,
                "written_explanation": "Usage graphs in the portal show zero consumption even when my bill is $300. The app is clearly broken.",
                "electricity_plan": "Premium", "customer_tenure": "3 years",
            },
        ],
    },
    {
        "case_id":        "TC-04",
        "label":          "Outage communication issues",
        "dominant_theme": "outage communication",
        "expected_keywords": ["outage", "power out", "no power", "restoration", "blackout", "electricity out"],
        "reviews": [
            {
                "customer_first_name": "James", "customer_last_name": "King",
                "customer_id": "TC04-A", "survey_completed_at": "2025-03-10",
                "recommendation_score_0_to_10": 1,
                "written_explanation": "8-hour outage last week. Not a single text, email, or app notification the whole time.",
                "electricity_plan": "Basic", "customer_tenure": "2 years",
            },
            {
                "customer_first_name": "Karen", "customer_last_name": "Wright",
                "customer_id": "TC04-B", "survey_completed_at": "2025-03-11",
                "recommendation_score_0_to_10": 2,
                "written_explanation": "4-hour power outage. I only found out electricity was restored when the lights came on. Zero communication.",
                "electricity_plan": "Standard", "customer_tenure": "6 years",
            },
            {
                "customer_first_name": "Liam", "customer_last_name": "Scott",
                "customer_id": "TC04-C", "survey_completed_at": "2025-03-12",
                "recommendation_score_0_to_10": 1,
                "written_explanation": "Four outages in two months. Every estimated restoration time has been wildly inaccurate.",
                "electricity_plan": "Green", "customer_tenure": "1 year",
            },
        ],
    },
    {
        "case_id":        "TC-05",
        "label":          "Renewal and variable rate complaints",
        "dominant_theme": "renewal education",
        "expected_keywords": ["renew", "renewal", "contract", "expiry", "variable rate", "rate increase", "rate went up"],
        "reviews": [
            {
                "customer_first_name": "Maria", "customer_last_name": "Green",
                "customer_id": "TC05-A", "survey_completed_at": "2025-03-13",
                "recommendation_score_0_to_10": 2,
                "written_explanation": "My contract ended and I was rolled onto a variable rate without any warning. My bill is now much higher.",
                "electricity_plan": "Standard", "customer_tenure": "2 years",
            },
            {
                "customer_first_name": "Nathan", "customer_last_name": "Adams",
                "customer_id": "TC05-B", "survey_completed_at": "2025-03-14",
                "recommendation_score_0_to_10": 2,
                "written_explanation": "Rate went up 20% at contract renewal with barely any notice. I did not know I needed to act to lock in the old rate.",
                "electricity_plan": "Basic", "customer_tenure": "3 years",
            },
            {
                "customer_first_name": "Olivia", "customer_last_name": "Baker",
                "customer_id": "TC05-C", "survey_completed_at": "2025-03-15",
                "recommendation_score_0_to_10": 3,
                "written_explanation": "I had no idea my contract was expiring. Why is there no renewal reminder? I ended up on a more expensive plan by default.",
                "electricity_plan": "Standard", "customer_tenure": "1 year",
            },
        ],
    },
]

# ── Evaluation functions ───────────────────────────────────────────────────────

def evaluate_output(output: dict, case: dict) -> dict:
    """
    Score one pipeline output dict against one test case.

    Scoring is rule-based and transparent — each dimension is checked with
    simple keyword or field presence tests. The scoring logic is documented
    inline so a human reviewer can verify any score without running code.

    Parameters
    ----------
    output : dict
        The result dict from run_agent_pipeline(), or any dict containing
        'classified_reviews', 'scored_opportunities', 'backlog_cards',
        'insights_brief', and 'duplicates' keys.
    case : dict
        One entry from TEST_CASES.

    Returns
    -------
    dict
        dimension → score (int 1–5) and rationale (str), plus metadata.
    """
    scores: dict[str, dict] = {}
    dominant = case["dominant_theme"]
    keywords = case["expected_keywords"]

    # ── 1. Theme accuracy ─────────────────────────────────────────────────────
    # Check whether the correct dominant theme appears in the scored opportunities.
    scored = output.get("scored_opportunities") or output.get("themes") or []
    theme_names = [
        (t.get("opportunity_type", "") + " " + t.get("name", "")).lower()
        for t in scored
    ]
    top_theme = theme_names[0] if theme_names else ""
    dominant_in_top  = dominant.lower() in top_theme
    dominant_present = any(dominant.lower() in n for n in theme_names)

    if dominant_in_top:
        ta_score, ta_rationale = 5, f"'{dominant}' is the top-ranked theme."
    elif dominant_present:
        ta_score, ta_rationale = 3, f"'{dominant}' detected but not ranked #1."
    elif theme_names:
        ta_score, ta_rationale = 2, f"'{dominant}' not found. Top theme: '{theme_names[0]}'."
    else:
        ta_score, ta_rationale = 1, "No themes produced."
    scores["theme_accuracy"] = {"score": ta_score, "rationale": ta_rationale}

    # ── 2. Completeness ───────────────────────────────────────────────────────
    # Count how many expected keywords appear in the full output text.
    brief = str(output.get("insights_brief", ""))
    cards_text = " ".join(
        str(c.get("title", "")) + " " + str(c.get("problem", ""))
        for c in (output.get("backlog_cards") or [])
    )
    full_text = (brief + " " + cards_text + " " + " ".join(theme_names)).lower()
    hits = sum(1 for kw in keywords if kw.lower() in full_text)
    ratio = hits / len(keywords) if keywords else 0

    if ratio >= 0.8:
        comp_score, comp_rationale = 5, f"{hits}/{len(keywords)} expected keywords found."
    elif ratio >= 0.6:
        comp_score, comp_rationale = 4, f"{hits}/{len(keywords)} expected keywords found."
    elif ratio >= 0.4:
        comp_score, comp_rationale = 3, f"{hits}/{len(keywords)} expected keywords found."
    elif ratio >= 0.2:
        comp_score, comp_rationale = 2, f"{hits}/{len(keywords)} expected keywords found."
    else:
        comp_score, comp_rationale = 1, f"{hits}/{len(keywords)} expected keywords found."
    scores["completeness"] = {"score": comp_score, "rationale": comp_rationale}

    # ── 3. Evidence grounding ─────────────────────────────────────────────────
    # Check whether verbatim evidence quotes are present and non-empty.
    all_quotes = []
    for t in scored:
        all_quotes.extend(t.get("sample_evidence_quotes", []))
    for c in (output.get("backlog_cards") or []):
        all_quotes.extend(c.get("evidence_quotes", []))
    non_empty_quotes = [q for q in all_quotes if q and len(str(q).strip()) > 10]

    if len(non_empty_quotes) >= 3:
        eg_score, eg_rationale = 5, f"{len(non_empty_quotes)} verbatim evidence quotes present."
    elif len(non_empty_quotes) == 2:
        eg_score, eg_rationale = 4, "2 verbatim evidence quotes present."
    elif len(non_empty_quotes) == 1:
        eg_score, eg_rationale = 3, "1 verbatim evidence quote present."
    else:
        eg_score, eg_rationale = 1, "No verbatim evidence quotes found."
    scores["evidence_grounding"] = {"score": eg_score, "rationale": eg_rationale}

    # ── 4. Actionability ──────────────────────────────────────────────────────
    # Check whether backlog cards were generated with the expected fields.
    cards = output.get("backlog_cards") or []
    has_cards       = len(cards) > 0
    has_user_story  = any(c.get("user_story") for c in cards)
    has_criteria    = any(c.get("acceptance_criteria") for c in cards)
    has_action      = any(c.get("recommended_action") for c in cards)

    action_score = 1 + has_cards + has_user_story + has_criteria + has_action
    action_rationale = (
        f"Cards: {'✓' if has_cards else '✗'}  "
        f"User story: {'✓' if has_user_story else '✗'}  "
        f"Acceptance criteria: {'✓' if has_criteria else '✗'}  "
        f"Recommended action: {'✓' if has_action else '✗'}"
    )
    scores["actionability"] = {"score": action_score, "rationale": action_rationale}

    # ── 5. Prioritization quality ─────────────────────────────────────────────
    # Check whether themes are scored and have P1/P2/P3 labels.
    has_scores  = any(t.get("opportunity_score") for t in scored)
    has_p_labels = any(t.get("priority") in {"P1", "P2", "P3"} for t in scored)
    has_breakdown = any(t.get("score_breakdown") for t in scored)
    top_is_correct = dominant_in_top

    pq_score = 1 + has_scores + has_p_labels + has_breakdown + top_is_correct
    pq_rationale = (
        f"Scores present: {'✓' if has_scores else '✗'}  "
        f"P1/P2/P3 labels: {'✓' if has_p_labels else '✗'}  "
        f"Breakdown: {'✓' if has_breakdown else '✗'}  "
        f"Correct top theme: {'✓' if top_is_correct else '✗'}"
    )
    scores["prioritization_quality"] = {"score": pq_score, "rationale": pq_rationale}

    # ── 6. Governance and uncertainty handling ────────────────────────────────
    # Check for human review notes, confidence flags, and duplicate detection.
    has_human_notes  = any(c.get("human_review_notes") for c in cards)
    has_confidence   = any(
        t.get("confidence_level") in {"low", "medium", "high"} for t in scored
    )
    has_dup_check    = bool(output.get("duplicates"))
    has_low_conf_flag = any(
        "Needs Human Validation" in c.get("labels", []) for c in cards
    )

    gov_score = 1 + has_human_notes + has_confidence + has_dup_check + has_low_conf_flag
    gov_rationale = (
        f"Human review notes: {'✓' if has_human_notes else '✗'}  "
        f"Confidence labels: {'✓' if has_confidence else '✗'}  "
        f"Duplicate detection: {'✓' if has_dup_check else '✗'}  "
        f"Low-confidence flagged: {'✓' if has_low_conf_flag else '✗'}"
    )
    scores["governance_and_uncertainty"] = {"score": gov_score, "rationale": gov_rationale}

    # ── Summary ───────────────────────────────────────────────────────────────
    total = sum(s["score"] for s in scores.values())
    max_possible = 5 * len(scores)

    return {
        "case_id":        case["case_id"],
        "case_label":     case["label"],
        "dominant_theme": dominant,
        "scores":         scores,
        "total_score":    total,
        "max_score":      max_possible,
        "pct_score":      round(100 * total / max_possible, 1),
        "evaluated_at":   datetime.now().strftime("%Y-%m-%d %H:%M"),
    }


def compare_baseline_vs_agentic(
    baseline_output: str,
    agentic_output: dict,
) -> list[dict]:
    """
    Rule-based side-by-side comparison of baseline text vs. agentic pipeline output.

    The baseline output is a plain string (no structure); the agentic output is the
    full result dict from run_agent_pipeline(). Each dimension is scored 1–5
    using the same transparent checks as evaluate_output().

    Returns a list of row dicts ready for display as a comparison table.
    """
    baseline_text = baseline_output.lower() if baseline_output else ""
    scored   = agentic_output.get("scored_opportunities") or []
    cards    = agentic_output.get("backlog_cards") or []
    brief    = str(agentic_output.get("insights_brief", "")).lower()

    rows = []
    for key, rubric_row in RUBRIC.items():
        dim = rubric_row["dimension"]

        # Baseline score: apply heuristics to the raw text string
        if key == "theme_accuracy":
            # Baseline scores 2 if it mentions any theme keywords, else 1
            theme_words = ["billing", "outage", "app", "payment", "renewal", "support"]
            b_score = 2 if any(w in baseline_text for w in theme_words) else 1
            b_note  = "Issue area mentioned but no structured theme extraction."

        elif key == "completeness":
            # Count how many theme words appear in baseline text
            theme_words = ["billing", "outage", "app", "payment", "renewal",
                           "support", "smart meter", "pricing", "green"]
            hits = sum(1 for w in theme_words if w in baseline_text)
            b_score = max(1, min(3, hits // 2))
            b_note  = f"{hits} topic areas mentioned in free text."

        elif key == "evidence_grounding":
            # Baseline almost never includes verbatim quotes
            has_quote = '"' in baseline_output or "'" in baseline_output
            b_score = 2 if has_quote else 1
            b_note  = "Summary only — no verbatim review quotes." if not has_quote else "Some quoted text present."

        elif key == "actionability":
            # Baseline may have bullet recommendations but no backlog structure
            has_action = any(w in baseline_text for w in ["recommend", "action", "fix", "improve", "add"])
            b_score = 2 if has_action else 1
            b_note  = "General recommendations only — no backlog cards or acceptance criteria."

        elif key == "prioritization_quality":
            # Baseline may list top issues but ranking is implicit (order of mention)
            has_rank = any(w in baseline_text for w in ["top", "first", "most", "critical", "urgent"])
            b_score = 2 if has_rank else 1
            b_note  = "Implicit ranking by mention order — no quantified scores."

        elif key == "governance_and_uncertainty":
            # Baseline output has no human review gate or confidence flags
            has_caveat = any(w in baseline_text for w in ["uncertain", "verify", "validate", "caveat", "note"])
            b_score = 2 if has_caveat else 1
            b_note  = "No human approval gate, no confidence flags, no uncertainty handling."

        else:
            b_score = rubric_row.get("baseline_typical", 2)
            b_note  = "—"

        # Agentic score: use the structured output
        if key == "theme_accuracy":
            a_score = 4 if scored else 1
            a_note  = f"{len(scored)} named themes; top-ranked by priority score."

        elif key == "completeness":
            a_score = 4 if len(scored) >= 3 else (2 if scored else 1)
            a_note  = f"{len(scored)} distinct opportunity themes identified."

        elif key == "evidence_grounding":
            quotes = sum(len(t.get("sample_evidence_quotes", [])) for t in scored)
            a_score = 5 if quotes >= 3 else (3 if quotes >= 1 else 1)
            a_note  = f"{quotes} verbatim evidence quotes across all themes."

        elif key == "actionability":
            a_score = 5 if cards and any(c.get("acceptance_criteria") for c in cards) else (3 if cards else 1)
            a_note  = f"{len(cards)} backlog cards with user stories and acceptance criteria."

        elif key == "prioritization_quality":
            has_scores = any(t.get("opportunity_score") for t in scored)
            a_score = 5 if has_scores else (3 if scored else 1)
            a_note  = "0–100 priority score with goal-specific weights." if has_scores else "Themes listed without scores."

        elif key == "governance_and_uncertainty":
            has_notes = any(c.get("human_review_notes") for c in cards)
            a_score = 5 if has_notes else (3 if cards else 1)
            a_note  = "Human review notes on every card; approval gate before Trello write."

        else:
            a_score = rubric_row.get("agentic_typical", 4)
            a_note  = "—"

        rows.append({
            "Dimension":          dim,
            "Baseline Score":     b_score,
            "Baseline Note":      b_note,
            "Agentic Score":      a_score,
            "Agentic Note":       a_note,
            "Score Difference":   a_score - b_score,
        })

    return rows


def export_evaluation_results(results: list[dict], path: str) -> str:
    """
    Write evaluation results to a CSV file.

    Each row in 'results' is the output of evaluate_output() for one test case.
    The file is written to 'path'. Returns the absolute path written.

    Parameters
    ----------
    results : list[dict]
        List of evaluate_output() return dicts.
    path : str
        File path to write. Parent directory must exist.

    Returns
    -------
    str
        Absolute path of the written file.
    """
    if not results:
        raise ValueError("No results to export.")

    # Flatten nested scores dict into one row per case
    flat_rows = []
    for r in results:
        row: dict = {
            "case_id":        r.get("case_id", ""),
            "case_label":     r.get("case_label", ""),
            "dominant_theme": r.get("dominant_theme", ""),
            "total_score":    r.get("total_score", ""),
            "max_score":      r.get("max_score", ""),
            "pct_score":      r.get("pct_score", ""),
            "evaluated_at":   r.get("evaluated_at", ""),
        }
        for dim_key, dim_data in r.get("scores", {}).items():
            row[f"{dim_key}_score"]     = dim_data.get("score", "")
            row[f"{dim_key}_rationale"] = dim_data.get("rationale", "")
        flat_rows.append(row)

    abs_path = os.path.abspath(path)
    fieldnames = list(flat_rows[0].keys()) if flat_rows else []
    with open(abs_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_rows)

    return abs_path


# ── Backward-compatible helpers used by the old Evaluation Preview tab ────────

def classification_accuracy(predictions: list[dict]) -> dict:
    """Compare predicted labels against TEST_SET ground truth (legacy helper)."""
    # Build a ground-truth map from the old TEST_SET format for backward compat
    _LEGACY_TEST_SET = [
        {"review_id": "T01", "ground_truth_category": "billing",    "ground_truth_sentiment": "negative", "ground_truth_severity": "high",     "ground_truth_opportunity_type": "pain_point"},
        {"review_id": "T02", "ground_truth_category": "app_portal", "ground_truth_sentiment": "negative", "ground_truth_severity": "high",     "ground_truth_opportunity_type": "pain_point"},
        {"review_id": "T03", "ground_truth_category": "app_portal", "ground_truth_sentiment": "neutral",  "ground_truth_severity": "low",      "ground_truth_opportunity_type": "feature_request"},
        {"review_id": "T04", "ground_truth_category": "pricing",    "ground_truth_sentiment": "positive", "ground_truth_severity": "low",      "ground_truth_opportunity_type": "compliment"},
        {"review_id": "T05", "ground_truth_category": "smart_meter","ground_truth_sentiment": "negative", "ground_truth_severity": "high",     "ground_truth_opportunity_type": "pain_point"},
        {"review_id": "T06", "ground_truth_category": "service",    "ground_truth_sentiment": "negative", "ground_truth_severity": "medium",   "ground_truth_opportunity_type": "pain_point"},
        {"review_id": "T07", "ground_truth_category": "outage",     "ground_truth_sentiment": "negative", "ground_truth_severity": "critical",  "ground_truth_opportunity_type": "pain_point"},
        {"review_id": "T08", "ground_truth_category": "billing",    "ground_truth_sentiment": "positive", "ground_truth_severity": "low",      "ground_truth_opportunity_type": "compliment"},
        {"review_id": "T09", "ground_truth_category": "app_portal", "ground_truth_sentiment": "neutral",  "ground_truth_severity": "low",      "ground_truth_opportunity_type": "question"},
        {"review_id": "T10", "ground_truth_category": "pricing",    "ground_truth_sentiment": "negative", "ground_truth_severity": "medium",   "ground_truth_opportunity_type": "pain_point"},
    ]
    gt_map  = {r["review_id"]: r for r in _LEGACY_TEST_SET}
    fields  = ["category", "sentiment", "severity", "opportunity_type"]
    correct = {f: 0 for f in fields}
    total   = 0
    for pred in predictions:
        rid = pred.get("review_id")
        if rid not in gt_map:
            continue
        gt = gt_map[rid]
        total += 1
        for f in fields:
            if pred.get(f) == gt.get(f"ground_truth_{f}"):
                correct[f] += 1
    if total == 0:
        return {f: None for f in fields}
    return {f: round(100 * correct[f] / total, 1) for f in fields}


def blank_rubric_scorecard() -> list[dict]:
    """Return an empty scorecard for manual human scoring."""
    return [
        {
            "Dimension":            v["dimension"],
            "Baseline Score (1–5)": "",
            "Agentic Score (1–5)":  "",
            "Notes":                "",
        }
        for v in RUBRIC.values()
    ]
