"""
agent_pipeline.py
Eight-step agentic pipeline: validate → clean → deduplicate → classify
→ cluster → score → brief → cards.

Each function is self-contained and returns structured data. In mock mode
(no API key) every LLM step falls back to a deterministic, keyword-based
or template-based implementation so the full UI can be demonstrated.
"""

import json
import re
import pandas as pd
from difflib import SequenceMatcher

from prompts import (
    SYSTEM_PROMPT,
    CLASSIFY_PROMPT,
    CLUSTER_PROMPT,
    BRIEF_PROMPT,
    CARD_PROMPT,
    BASELINE_PROMPT,
)
from scoring import rank_themes

# ── Client helper ─────────────────────────────────────────────────────────────

def get_client(api_key: str | None):
    """Return an Anthropic client, or None if no key is provided."""
    if not api_key:
        return None
    from anthropic import Anthropic
    return Anthropic(api_key=api_key)


def _llm(client, prompt: str, max_tokens: int = 2048) -> str:
    """Single-turn LLM call. Returns the text content of the first block."""
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def _parse_json(text: str) -> list | dict:
    """Extract and parse the first JSON array or object from a text blob."""
    # Strip markdown fences if present
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    return json.loads(text)


# ── Sample data ───────────────────────────────────────────────────────────────

def generate_sample_data() -> pd.DataFrame:
    """Return 30 synthetic electricity customer survey rows."""
    rows = [
        # Billing (7)
        ("R01", "My bill jumped 45% this month with no explanation. Usage hasn't changed.", 1, "residential"),
        ("R02", "Charged a $200 reconnection fee even though I paid on time. Rep put me on hold.", 1, "residential"),
        ("R03", "The billing portal never loads my historical statements.", 2, "residential"),
        ("R04", "Received a bill for a property I moved out of 3 months ago — second time this happened.", 1, "residential"),
        ("R05", "Estimated bill was $150 higher than actual usage. Why estimate for 6 months?", 2, "residential"),
        ("R06", "On autopay but still got a late fee. Bank info 'expired' — nobody told me.", 2, "residential"),
        ("R07", "New itemized bill format is really clear. I finally understand what I'm paying for.", 4, "residential"),
        # App / portal (6)
        ("R08", "Mobile app crashes every time I view usage history. Useless on iOS 17.", 1, "residential"),
        ("R09", "Why can't I pay my bill through the app? Have to go to the website every time.", 2, "residential"),
        ("R10", "Usage graphs show zero consumption even when my bill is $300. Clearly broken.", 1, "residential"),
        ("R11", "Held for 45 min to ask what my balance is — that should be answerable in the app.", 2, "residential"),
        ("R12", "Outage map said my area was 'operational' while we had no power. Completely wrong.", 1, "residential"),
        ("R13", "New payment flow in the app is much smoother after the last update.", 4, "residential"),
        # Outages (6)
        ("R14", "8-hour outage. Not a single text, email, or app update the whole time.", 1, "residential"),
        ("R15", "4-hour outage; only knew power was back when lights came on. Zero communication.", 2, "residential"),
        ("R16", "Lost power 3 days in the ice storm. I get outages happen — just keep us informed.", 2, "residential"),
        ("R17", "Reported outage online, got confirmation, power restored in 2 hours. Good experience.", 4, "residential"),
        ("R18", "Four outages in 2 months. Estimated restoration times are always wildly wrong.", 1, "residential"),
        ("R19", "Planned maintenance — only found out the day before. Two weeks' notice would help.", 2, "residential"),
        # Smart meter (4)
        ("R20", "Meter crew missed our appointment twice with no warning. Rescheduled 3 times.", 1, "residential"),
        ("R21", "Smart meter shows usage 40% above my old analog meter. Readings must be wrong.", 1, "residential"),
        ("R22", "Smart meter installed smoothly. Hourly usage view is really helpful for planning.", 5, "residential"),
        ("R23", "Ever since the smart meter went in, my bill is up significantly. No explanation.", 2, "residential"),
        # Customer service (4)
        ("R24", "45-min hold just for a simple billing question. Please add self-service options.", 1, "residential"),
        ("R25", "Chatbot can't answer anything and there's no easy path to a real person.", 1, "residential"),
        ("R26", "Called about a billing error; agent resolved it in under 10 minutes. Great service.", 5, "residential"),
        ("R27", "Incorrect meter reading complaint open for 3 weeks with zero updates.", 1, "residential"),
        # Pricing (3)
        ("R28", "Time-of-use rates saved me $40 last month by shifting laundry to evenings.", 5, "residential"),
        ("R29", "20% year-over-year rate increase with no explanation or notice. Very frustrating.", 2, "commercial"),
        ("R30", "The seasonal rate transparency report finally explains why my bill changes. Helpful.", 4, "residential"),
    ]
    return pd.DataFrame(rows, columns=["review_id", "review_text", "rating", "customer_segment"])


# ── Step 1: Validate ──────────────────────────────────────────────────────────

REQUIRED_COLUMNS = {"review_id", "review_text", "rating"}

def validate_csv(df: pd.DataFrame) -> dict:
    """
    Check that the dataframe has the required columns and plausible data.
    Returns {"valid": bool, "errors": [...], "warnings": [...], "row_count": int}.
    """
    errors, warnings = [], []

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        errors.append(f"Missing required columns: {', '.join(sorted(missing))}")

    if len(df) == 0:
        errors.append("CSV has no data rows.")
    elif len(df) < 5:
        warnings.append(f"Only {len(df)} rows — themes may not be meaningful.")

    if "review_text" in df.columns:
        blank = df["review_text"].isna().sum() + (df["review_text"].astype(str).str.strip() == "").sum()
        if blank:
            warnings.append(f"{blank} blank review_text values will be dropped.")

    if "rating" in df.columns:
        try:
            ratings = pd.to_numeric(df["rating"], errors="coerce")
            out_of_range = (~ratings.between(1, 5)).sum()
            if out_of_range:
                warnings.append(f"{out_of_range} rating(s) are outside the 1–5 range.")
        except Exception:
            warnings.append("Could not validate rating column values.")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "row_count": len(df),
        "columns": list(df.columns),
    }


# ── Step 2: Clean ─────────────────────────────────────────────────────────────

def clean_reviews(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Drop blanks, strip whitespace, standardize dtypes.
    Returns (cleaned_df, stats_dict).
    """
    original_count = len(df)
    df = df.copy()

    # Drop rows with no review text
    df["review_text"] = df["review_text"].astype(str).str.strip()
    df = df[df["review_text"].str.len() > 0].copy()

    # Normalize rating to int where possible
    if "rating" in df.columns:
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(3).astype(int)

    # Strip whitespace from string columns
    for col in df.select_dtypes("object").columns:
        df[col] = df[col].str.strip()

    df = df.reset_index(drop=True)

    stats = {
        "original_count": original_count,
        "after_clean": len(df),
        "dropped": original_count - len(df),
    }
    return df, stats


# ── Step 3: Detect duplicates ─────────────────────────────────────────────────

def detect_duplicates(df: pd.DataFrame, threshold: float = 0.72) -> tuple[pd.DataFrame, list]:
    """
    Flag near-duplicate reviews using Jaccard similarity on word sets.
    Returns (df_with_flag, list_of_duplicate_pairs).
    Keeps one review per duplicate pair (the first occurrence).
    """
    df = df.copy()
    texts = df["review_text"].str.lower().tolist()
    n = len(texts)
    flagged = set()
    pairs = []

    for i in range(n):
        if i in flagged:
            continue
        set_i = set(texts[i].split())
        for j in range(i + 1, n):
            if j in flagged:
                continue
            set_j = set(texts[j].split())
            union = set_i | set_j
            if not union:
                continue
            sim = len(set_i & set_j) / len(union)
            if sim >= threshold:
                flagged.add(j)
                pairs.append({"index_a": i, "index_b": j, "similarity": round(sim, 2)})

    df["is_duplicate"] = df.index.isin(flagged)
    df_deduped = df[~df["is_duplicate"]].reset_index(drop=True)
    return df_deduped, pairs


# ── Step 4: Classify ──────────────────────────────────────────────────────────

def classify_reviews(df: pd.DataFrame, client=None, mock: bool = True) -> pd.DataFrame:
    """
    Add classification columns to every review row.
    Live mode: one batched LLM call with structured JSON output.
    Mock mode: keyword + rating heuristics (fast, deterministic).
    """
    if not mock and client is not None:
        return _classify_live(df, client)
    return _classify_mock(df)


def _classify_live(df: pd.DataFrame, client) -> pd.DataFrame:
    reviews_input = df[["review_id", "review_text"]].to_dict(orient="records")
    prompt = CLASSIFY_PROMPT.format(reviews_json=json.dumps(reviews_input, indent=2))
    text = _llm(client, prompt, max_tokens=4096)
    try:
        results = _parse_json(text)
        for field in ["category", "sentiment", "severity", "opportunity_type", "customer_impact"]:
            df[field] = [r.get(field, "other") for r in results]
    except Exception:
        # Graceful fallback if LLM returns malformed JSON
        df = _classify_mock(df)
    return df


_CATEGORY_KEYWORDS = {
    "billing":      ["bill", "charge", "payment", "invoice", "fee", "autopay", "credit", "cost"],
    "outage":       ["outage", "power out", "blackout", "no power", "electricity out", "restoration"],
    "app_portal":   ["app", "portal", "website", "login", "crash", "map", "graph", "mobile", "online"],
    "smart_meter":  ["smart meter", "meter", "reading", "installation", "install", "technician"],
    "service":      ["hold", "agent", "representative", "customer service", "wait", "chatbot", "complaint"],
    "pricing":      ["rate", "price", "pricing", "time-of-use", "tou", "plan", "tariff"],
}


def _classify_mock(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    categories, sentiments, severities, opp_types, impacts = [], [], [], [], []

    for _, row in df.iterrows():
        text = str(row.get("review_text", "")).lower()
        rating = int(row.get("rating", 3))

        # Category: first keyword match wins, else "other"
        cat = "other"
        for c, keywords in _CATEGORY_KEYWORDS.items():
            if any(kw in text for kw in keywords):
                cat = c
                break
        categories.append(cat)

        # Sentiment from rating
        if rating >= 4:
            sentiments.append("positive")
        elif rating == 3:
            sentiments.append("neutral")
        else:
            sentiments.append("negative")

        # Severity from rating
        if rating == 1:
            severities.append("high")
        elif rating == 2:
            severities.append("medium")
        elif rating == 5:
            severities.append("low")
        else:
            severities.append("low")

        # Opportunity type
        if rating >= 4:
            opp_types.append("compliment")
        elif any(w in text for w in ["wish", "should", "would be nice", "feature", "add", "allow"]):
            opp_types.append("feature_request")
        elif any(w in text for w in ["how", "where", "can i", "what is"]):
            opp_types.append("question")
        else:
            opp_types.append("pain_point")

        # Customer impact — simple heuristic
        if any(w in text for w in ["neighborhood", "area", "everyone", "whole street", "widespread"]):
            impacts.append("neighborhood")
        else:
            impacts.append("individual")

    df["category"] = categories
    df["sentiment"] = sentiments
    df["severity"] = severities
    df["opportunity_type"] = opp_types
    df["customer_impact"] = impacts
    return df


# ── Step 5: Cluster themes ────────────────────────────────────────────────────

def cluster_themes(df: pd.DataFrame, client=None, mock: bool = True) -> list[dict]:
    """
    Group classified reviews into themes.
    Live mode: LLM clusters by semantic similarity.
    Mock mode: group by category, merge small groups into "other".
    """
    if not mock and client is not None:
        return _cluster_live(df, client)
    return _cluster_mock(df)


def _cluster_live(df: pd.DataFrame, client) -> list[dict]:
    classified = df.to_dict(orient="records")
    prompt = CLUSTER_PROMPT.format(classified_json=json.dumps(classified, indent=2))
    text = _llm(client, prompt, max_tokens=2048)
    try:
        raw_themes = _parse_json(text)
    except Exception:
        return _cluster_mock(df)

    themes = []
    for t in raw_themes:
        indices = t.get("review_indices", [])
        reviews = [classified[i] for i in indices if i < len(classified)]
        themes.append({
            **t,
            "review_count": len(reviews),
            "reviews": reviews,
        })
    return themes


_CATEGORY_LABELS = {
    "billing":     "Billing & Payment Issues",
    "outage":      "Outage Communication Failures",
    "app_portal":  "App & Portal Usability",
    "smart_meter": "Smart Meter Problems",
    "service":     "Customer Service & Wait Times",
    "pricing":     "Rate & Pricing Transparency",
    "other":       "General Feedback",
}

_CATEGORY_PAIN_POINTS = {
    "billing":     "Bills change unexpectedly with no explanation or breakdown",
    "outage":      "Customers receive no updates during or after power outages",
    "app_portal":  "The app and portal fail at the most common tasks",
    "smart_meter": "Meter installation is unreliable and readings are questioned",
    "service":     "Long hold times with no self-service alternative",
    "pricing":     "Rate changes arrive without notice or explanation",
    "other":       "Miscellaneous customer concerns",
}


def _cluster_mock(df: pd.DataFrame) -> list[dict]:
    themes = []
    grouped = df.groupby("category")

    for cat, group in grouped:
        reviews = group.to_dict(orient="records")
        severity_order = ["critical", "high", "medium", "low"]
        dominant = max(
            severity_order,
            key=lambda s: sum(1 for r in reviews if r.get("severity") == s),
        )
        themes.append({
            "name": _CATEGORY_LABELS.get(cat, cat.replace("_", " ").title()),
            "description": f"Reviews related to {cat.replace('_', ' ')} experiences",
            "category": cat,
            "review_indices": group.index.tolist(),
            "dominant_severity": dominant,
            "key_pain_point": _CATEGORY_PAIN_POINTS.get(cat, ""),
            "review_count": len(reviews),
            "reviews": reviews,
        })

    return themes


# ── Step 6: Score opportunities ───────────────────────────────────────────────

def score_opportunities(themes: list[dict]) -> list[dict]:
    """Delegate to scoring.rank_themes — pure math, no LLM."""
    return rank_themes(themes)


# ── Step 7: Generate insights brief ──────────────────────────────────────────

def generate_brief(themes: list[dict], product_goal: str, client=None, mock: bool = True) -> str:
    """
    Produce a markdown product insights brief.
    Live mode: LLM narrative.
    Mock mode: template built from scored theme data.
    """
    if not mock and client is not None:
        return _brief_live(themes, product_goal, client)
    return _brief_mock(themes, product_goal)


def _brief_live(themes, product_goal, client) -> str:
    slim = [
        {k: v for k, v in t.items() if k not in ("reviews", "review_indices")}
        for t in themes
    ]
    prompt = BRIEF_PROMPT.format(
        product_goal=product_goal,
        themes_json=json.dumps(slim, indent=2),
    )
    return _llm(client, prompt, max_tokens=800)


def _brief_mock(themes: list[dict], product_goal: str) -> str:
    top3 = themes[:3]
    lines = [
        f"## Product Insights Brief",
        f"*Goal: {product_goal}*",
        "",
        "### Executive Summary",
        f"Analysis of {sum(t['review_count'] for t in themes)} customer reviews surfaced "
        f"{len(themes)} distinct opportunity themes. "
        f"The most critical area is **{top3[0]['name']}** "
        f"(score {top3[0]['opportunity_score']}/10), driven by {top3[0]['key_pain_point'].lower()}. "
        "Immediate action on the top two themes could meaningfully reduce inbound contacts "
        "and protect NPS.",
        "",
        "### Top 3 Opportunities",
    ]
    for i, t in enumerate(top3, 1):
        lines += [
            f"**{i}. {t['name']}** — {t['priority']} | Score: {t['opportunity_score']}/10 | "
            f"{t['review_count']} reviews",
            f"> {t['key_pain_point']}",
            "",
        ]
    lines += [
        "### Risk",
        f"If **{top3[0]['name']}** is not addressed, customers experiencing "
        f"{top3[0]['dominant_severity']}-severity issues are likely to escalate calls, "
        "churn, or leave negative public reviews — directly harming NPS and support costs.",
        "",
        "### Recommended Next Step",
        f"Prioritize a discovery sprint for **{top3[0]['name']}** this cycle. "
        "Map the end-to-end customer journey for the affected flow, identify the root cause, "
        "and size a minimum lovable fix that can ship within two sprints.",
    ]
    return "\n".join(lines)


# ── Step 8: Generate backlog cards ────────────────────────────────────────────

def generate_backlog_cards(themes: list[dict], product_goal: str, client=None, mock: bool = True) -> list[dict]:
    """
    Convert scored themes into Trello-ready backlog card dicts.
    Live mode: LLM generates rich, context-aware cards.
    Mock mode: template-based cards derived from theme metadata.
    """
    if not mock and client is not None:
        return _cards_live(themes, product_goal, client)
    return _cards_mock(themes, product_goal)


def _cards_live(themes, product_goal, client) -> list[dict]:
    slim = [
        {k: v for k, v in t.items() if k not in ("reviews", "review_indices")}
        for t in themes
    ]
    prompt = CARD_PROMPT.format(
        product_goal=product_goal,
        themes_json=json.dumps(slim, indent=2),
    )
    text = _llm(client, prompt, max_tokens=3000)
    try:
        return _parse_json(text)
    except Exception:
        return _cards_mock(themes, product_goal)


_EFFORT_MAP = {"P1": "L", "P2": "M", "P3": "S"}
_CATEGORY_LABELS_SHORT = {
    "billing":     "billing",
    "outage":      "outage",
    "app_portal":  "app",
    "smart_meter": "meter",
    "service":     "service",
    "pricing":     "pricing",
}


def _cards_mock(themes: list[dict], product_goal: str) -> list[dict]:
    cards = []
    for t in themes:
        cat_label = _CATEGORY_LABELS_SHORT.get(t["category"], t["category"])
        priority = t.get("priority", "P3")
        cards.append({
            "title": f"Resolve {t['name']}",
            "description": (
                f"{t['review_count']} customers reported issues related to {t['name'].lower()}. "
                f"Core pain point: {t['key_pain_point']}."
            ),
            "user_story": (
                f"As a customer, I want {t['key_pain_point'].rstrip('.').lower()} "
                f"so that I can {product_goal.lower()} without friction."
            ),
            "acceptance_criteria": [
                f"The {t['category'].replace('_', ' ')} flow completes without error in 95% of test cases.",
                f"Customer-reported {cat_label} complaints decrease by at least 20% within 60 days.",
                f"User satisfaction score for the {cat_label} experience reaches ≥ 4.0/5.",
            ],
            "labels": [cat_label, "ux"] if t["category"] in ("app_portal",) else [cat_label],
            "estimated_effort": _EFFORT_MAP.get(priority, "M"),
            "priority": priority,
            "opportunity_score": t.get("opportunity_score", 0),
        })
    return cards


# ── Baseline (for evaluation comparison) ─────────────────────────────────────

def run_baseline(df: pd.DataFrame, product_goal: str, client=None, mock: bool = True) -> str:
    """
    Single-prompt baseline: summarize reviews and list top issues.
    This is the 'before' state the agentic pipeline is compared against.
    """
    if not mock and client is not None:
        reviews_text = "\n".join(
            f"- [{row.rating}/5] {row.review_text}"
            for _, row in df.iterrows()
        )
        prompt = BASELINE_PROMPT.format(reviews_text=reviews_text)
        return _llm(client, prompt, max_tokens=600)

    # Mock baseline — representative of what a simple prompt would return
    return """\
**Summary:** Customers are frustrated primarily with unexpected billing changes, \
lack of communication during power outages, and a mobile app that frequently fails. \
Positive feedback exists around time-of-use pricing and recent service interactions.

**Top 5 Issues:**
- Unexplained bill increases or billing errors
- No proactive communication during outages
- Mobile app crashes and inaccurate usage graphs
- Long customer service hold times
- Smart meter installation delays and inaccurate readings

**Recommended Actions:**
1. Add real-time outage status notifications via SMS and the app.
2. Investigate and resolve mobile app stability issues on iOS.
3. Introduce a billing explainer that breaks down each charge clearly.

*Note: This is the baseline output — a single LLM prompt with no classification, \
clustering, or scoring. Compare against the agentic pipeline in the Evaluation tab.*"""
