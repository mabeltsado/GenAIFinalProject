"""
evaluation.py
Evaluation framework for Review2Roadmap.

Contains:
  TEST_SET   — 10 held-out synthetic reviews with ground-truth labels
  RUBRIC     — scoring dimensions for comparing baseline vs. agentic output
  Functions  — classification accuracy, rubric scoring template, comparison display
"""

# ── Test set ──────────────────────────────────────────────────────────────────
# 10 synthetic reviews with human-assigned ground-truth labels.
# Used to measure Step 4 (classification) accuracy in live mode.

TEST_SET = [
    {
        "review_id": "T01",
        "review_text": "My bill doubled this quarter and the portal shows no breakdown.",
        "rating": 1,
        "customer_segment": "residential",
        "ground_truth_category": "billing",
        "ground_truth_sentiment": "negative",
        "ground_truth_severity": "high",
        "ground_truth_opportunity_type": "pain_point",
    },
    {
        "review_id": "T02",
        "review_text": "The outage map showed my area as operational during a 6-hour blackout.",
        "rating": 1,
        "customer_segment": "residential",
        "ground_truth_category": "app_portal",
        "ground_truth_sentiment": "negative",
        "ground_truth_severity": "high",
        "ground_truth_opportunity_type": "pain_point",
    },
    {
        "review_id": "T03",
        "review_text": "I wish the app let me set budget alerts so I know before I overspend.",
        "rating": 3,
        "customer_segment": "residential",
        "ground_truth_category": "app_portal",
        "ground_truth_sentiment": "neutral",
        "ground_truth_severity": "low",
        "ground_truth_opportunity_type": "feature_request",
    },
    {
        "review_id": "T04",
        "review_text": "The time-of-use plan has cut my monthly bill by about $30. Love it.",
        "rating": 5,
        "customer_segment": "residential",
        "ground_truth_category": "pricing",
        "ground_truth_sentiment": "positive",
        "ground_truth_severity": "low",
        "ground_truth_opportunity_type": "compliment",
    },
    {
        "review_id": "T05",
        "review_text": "Smart meter was installed without notice and now my readings look wrong.",
        "rating": 1,
        "customer_segment": "residential",
        "ground_truth_category": "smart_meter",
        "ground_truth_sentiment": "negative",
        "ground_truth_severity": "high",
        "ground_truth_opportunity_type": "pain_point",
    },
    {
        "review_id": "T06",
        "review_text": "I was on hold for 50 minutes. The agent could not resolve my billing issue.",
        "rating": 1,
        "customer_segment": "residential",
        "ground_truth_category": "service",
        "ground_truth_sentiment": "negative",
        "ground_truth_severity": "medium",
        "ground_truth_opportunity_type": "pain_point",
    },
    {
        "review_id": "T07",
        "review_text": "Power out for 10 hours with no estimated restoration time communicated.",
        "rating": 1,
        "customer_segment": "residential",
        "ground_truth_category": "outage",
        "ground_truth_sentiment": "negative",
        "ground_truth_severity": "critical",
        "ground_truth_opportunity_type": "pain_point",
    },
    {
        "review_id": "T08",
        "review_text": "The new bill layout is clear and easy to read. Good improvement.",
        "rating": 4,
        "customer_segment": "residential",
        "ground_truth_category": "billing",
        "ground_truth_sentiment": "positive",
        "ground_truth_severity": "low",
        "ground_truth_opportunity_type": "compliment",
    },
    {
        "review_id": "T09",
        "review_text": "How do I switch from paper bills to e-billing? I can't find it in the app.",
        "rating": 3,
        "customer_segment": "residential",
        "ground_truth_category": "app_portal",
        "ground_truth_sentiment": "neutral",
        "ground_truth_severity": "low",
        "ground_truth_opportunity_type": "question",
    },
    {
        "review_id": "T10",
        "review_text": "Rate went up 18% with two weeks' notice. Feels unfair and opaque.",
        "rating": 2,
        "customer_segment": "commercial",
        "ground_truth_category": "pricing",
        "ground_truth_sentiment": "negative",
        "ground_truth_severity": "medium",
        "ground_truth_opportunity_type": "pain_point",
    },
]

# ── Evaluation rubric ─────────────────────────────────────────────────────────
# Evaluators score each output 1–5 on every dimension.
# Used to compare the baseline prompt against the full agentic pipeline.

RUBRIC = [
    {
        "Dimension": "Issue Coverage",
        "Description": "Does the output identify all major problem areas in the reviews?",
        "Baseline Expectation": "Covers top issues but may miss edge cases",
        "Agentic Expectation": "Systematic classification ensures all categories are surfaced",
        "Score 1": "Misses 3+ major issues",
        "Score 3": "Covers most issues",
        "Score 5": "All major issues identified with supporting evidence",
    },
    {
        "Dimension": "Actionability",
        "Description": "Can a PM immediately use this output to plan work?",
        "Baseline Expectation": "General recommendations, not sprint-ready",
        "Agentic Expectation": "Backlog cards with user stories and acceptance criteria",
        "Score 1": "Vague observations only",
        "Score 3": "Clear suggestions but not structured",
        "Score 5": "Ready-to-use backlog cards with full context",
    },
    {
        "Dimension": "Prioritization",
        "Description": "Does the output clearly rank issues by importance?",
        "Baseline Expectation": "Lists issues but ranking is implicit or subjective",
        "Agentic Expectation": "Scored opportunities with explicit P1/P2/P3 labels",
        "Score 1": "No prioritization",
        "Score 3": "Rough ranking present",
        "Score 5": "Quantified scores with transparent weighting",
    },
    {
        "Dimension": "Evidence Grounding",
        "Description": "Are claims tied to specific review evidence?",
        "Baseline Expectation": "General summary without specific examples",
        "Agentic Expectation": "Themes backed by review counts and examples",
        "Score 1": "No evidence cited",
        "Score 3": "Some examples given",
        "Score 5": "Every claim tied to review volume and examples",
    },
    {
        "Dimension": "Governance / Human Control",
        "Description": "Does the workflow keep a human in the loop before action is taken?",
        "Baseline Expectation": "None — output is final with no review step",
        "Agentic Expectation": "Card approval gate before any Trello write",
        "Score 1": "Fully automated, no review gate",
        "Score 3": "Output is reviewable but no explicit approval step",
        "Score 5": "Explicit human approval required before any external action",
    },
]

# ── Accuracy helpers ──────────────────────────────────────────────────────────

def classification_accuracy(predictions: list[dict]) -> dict:
    """
    Compare predicted labels against TEST_SET ground truth.
    predictions: list of dicts with review_id + predicted fields.
    Returns per-field accuracy as percentages.
    """
    gt_map = {r["review_id"]: r for r in TEST_SET}
    fields = ["category", "sentiment", "severity", "opportunity_type"]
    correct = {f: 0 for f in fields}
    total = 0

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
    """Return an empty scorecard table ready for a human evaluator to fill in."""
    return [
        {
            "Dimension": row["Dimension"],
            "Baseline Score (1–5)": "",
            "Agentic Score (1–5)": "",
            "Notes": "",
        }
        for row in RUBRIC
    ]
