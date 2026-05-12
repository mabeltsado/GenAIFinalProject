"""
agent_pipeline.py
Review2Roadmap — Multi-Step Agentic Pipeline

This module implements a controlled, step-by-step orchestration workflow.
Each function represents one discrete agent step with clearly defined inputs
and outputs. The workflow is NOT fully autonomous: the user uploads data and
selects a goal, then the system does the analysis, and a human reviews the
output before any roadmap or Trello action is taken.

Pipeline order:
  1. validate_reviews_csv     — check schema and data quality
  2. clean_reviews            — normalize and prepare data
  3. detect_duplicates        — flag suspicious duplicate responses
  4. classify_reviews         — label each review with structured fields
  5. cluster_themes           — group reviews into product opportunity themes
  6. score_opportunities      — rank themes by weighted priority score
  7. generate_insights_brief  — produce a markdown product brief
  8. generate_backlog_cards   — create Trello-ready card objects
  9. run_agent_pipeline       — orchestrate all steps and return full results dict

Mock mode (use_mock=True) replaces every LLM call with rule-based logic
so the app can be fully demonstrated without an API key.
"""

import json
import os
import re
import statistics
from datetime import datetime

import pandas as pd

from prompts import SYSTEM_PROMPT, BASELINE_PROMPT
from scoring import rank_themes

# ── Constants ─────────────────────────────────────────────────────────────────

# These are the columns the pipeline expects in the uploaded CSV.
REQUIRED_COLUMNS = [
    "customer_first_name",
    "customer_last_name",
    "customer_id",
    "survey_completed_at",
    "recommendation_score_0_to_10",
    "written_explanation",
    "electricity_plan",
    "customer_tenure",
]

# The full taxonomy of product opportunity types this pipeline can identify.
# This list is used in both mock classification and as the LLM schema constraint.
OPPORTUNITY_TYPES = [
    "billing transparency",
    "high bill investigation",
    "pricing clarity",
    "renewal education",
    "app improvement",
    "payment reliability",
    "outage communication",
    "support experience",
    "plan education",
    "green energy messaging",
    "retention risk",
]

# Maps opportunity type → broader issue category used for grouping and labelling.
_CATEGORY_FROM_OPP = {
    "billing transparency":    "billing",
    "high bill investigation": "billing",
    "payment reliability":     "billing",
    "pricing clarity":         "pricing",
    "renewal education":       "plans",
    "plan education":          "plans",
    "app improvement":         "digital",
    "outage communication":    "outage",
    "support experience":      "service",
    "green energy messaging":  "green",
    "retention risk":          "retention",
}

# Keyword → opportunity type mapping used by the rule-based mock classifier.
# Order matters: the first match wins, so more specific terms go first.
_KEYWORD_MAP = {
    "retention risk": [
        "switch provider", "switching supplier", "change supplier", "leaving",
        "cancel my account", "competitor", "better deal elsewhere", "switching to",
    ],
    "outage communication": [
        "outage", "power out", "blackout", "no power", "restoration",
        "electricity out", "no electricity", "power failure", "power restored",
    ],
    "app improvement": [
        "app", "portal", "website", "mobile", "online", "login", "digital",
        "interface", "crash", "crashing", "loading", "error message", "outage map",
    ],
    "support experience": [
        "hold", "wait time", "45 minutes", "agent", "representative",
        "customer service", "call center", "chatbot", "helpline", "rude",
        "support team", "on hold",
    ],
    "payment reliability": [
        "payment", "autopay", "auto-pay", "bank", "credit card", "direct debit",
        "failed payment", "payment error", "charged twice", "late fee",
    ],
    "renewal education": [
        "renew", "renewal", "contract", "contract end", "expiry",
    ],
    "green energy messaging": [
        "green", "renewable", "solar", "environmental", "sustainable",
        "clean energy", "carbon", "eco",
    ],
    "plan education": [
        "which plan", "best plan", "understand my plan", "plan details",
        "tariff details", "confused about my plan",
    ],
    "high bill investigation": [
        "bill jumped", "bill doubled", "bill tripled", "bill increased",
        "massive bill", "huge bill", "unexpected bill", "way too high",
        "bill is wrong", "estimated read", "bill went up",
    ],
    "pricing clarity": [
        "rate", "price", "pricing", "cost per kwh", "comparison",
        "expensive", "cheaper", "value for money", "time-of-use", "tou",
    ],
    "billing transparency": [
        "bill", "billing", "charge", "statement", "invoice", "breakdown",
        "itemized", "meter read", "account balance",
    ],
}

# Maps integer severity score (1–5) → string label expected by scoring.py
_SEVERITY_STR = {1: "low", 2: "low", 3: "medium", 4: "high", 5: "critical"}

# Recommended owner area by issue category — used in backlog cards
_OWNER_AREA = {
    "billing":   "Billing & Revenue",
    "digital":   "Digital Product",
    "outage":    "Grid Operations / CX",
    "service":   "Customer Service",
    "pricing":   "Pricing & Strategy",
    "plans":     "Product & Marketing",
    "green":     "Sustainability & Marketing",
    "retention": "Customer Success",
    "other":     "Product",
}


# ── LLM client helpers ────────────────────────────────────────────────────────

def get_client(api_key: str | None = None):
    """
    Return an Anthropic client.

    Tries the provided key first, then falls back to the ANTHROPIC_API_KEY
    environment variable. Returns None if neither is available, which causes
    all LLM steps to fall back to mock mode.
    """
    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        return None
    from anthropic import Anthropic
    return Anthropic(api_key=key)


def _llm_call(prompt: str, max_tokens: int = 2048) -> str:
    """
    Make a single LLM call using the Anthropic API.

    Reads the API key from the environment. Raises RuntimeError if no key
    is configured so the caller can catch and fall back to mock mode.
    """
    client = get_client()
    if client is None:
        raise RuntimeError(
            "No Anthropic API key found. Set ANTHROPIC_API_KEY or use use_mock=True."
        )
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def _parse_json_response(text: str):
    """Strip markdown code fences and parse the JSON payload from an LLM response."""
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    return json.loads(text)


# ── Sample data ───────────────────────────────────────────────────────────────

def generate_sample_data() -> pd.DataFrame:
    """
    Return 30 synthetic electricity customer survey rows.

    Uses the same schema as the expected uploaded CSV so the pipeline can
    run end-to-end without a real data file. Called by the app when
    'data/fake_electricity_customer_reviews_500.csv' is not found.
    """
    rows = [
        # fmt: (first, last, id, date, nps_score, explanation, plan, tenure)
        ("Alice",   "Moore",    "C001", "2025-03-01",  2, "My bill jumped 45% this month with no explanation. My usage hasn't changed at all.",                          "Standard", "2 years"),
        ("Brian",   "Carter",   "C002", "2025-03-02",  1, "Charged a reconnection fee even though I paid on time. Rep put me on hold for an hour.",                      "Basic",    "6 months"),
        ("Carol",   "Evans",    "C003", "2025-03-03",  3, "The billing portal never loads my historical statements. I can't check if charges are correct.",               "Standard", "3 years"),
        ("David",   "Harris",   "C004", "2025-03-04",  2, "Received a bill for a property I moved out of 3 months ago. This is the second time it happened.",             "Standard", "1 year"),
        ("Emma",    "Lewis",    "C005", "2025-03-05",  3, "Estimated bill was $150 higher than actual usage. Why estimate for 6 months without reading the meter?",       "Basic",    "4 years"),
        ("Frank",   "Robinson", "C006", "2025-03-06",  2, "On autopay but still got a late fee. My bank info expired and nobody told me.",                                "Premium",  "2 years"),
        ("Grace",   "Walker",   "C007", "2025-03-07",  8, "New itemized bill format is really clear. I finally understand exactly what I'm paying for.",                  "Standard", "5 years"),
        ("Henry",   "Young",    "C008", "2025-03-08",  1, "Mobile app crashes every time I view my usage history. Completely useless on iOS 17.",                         "Standard", "1 year"),
        ("Iris",    "Allen",    "C009", "2025-03-09",  2, "Why can't I pay my bill in the app? I have to go to the website every single time.",                           "Basic",    "8 months"),
        ("James",   "King",     "C010", "2025-03-10",  1, "Usage graphs in the portal show zero consumption even when my bill is $300. Clearly broken.",                  "Premium",  "3 years"),
        ("Karen",   "Wright",   "C011", "2025-03-11",  2, "Held on the phone 45 minutes just to find out my account balance. The app should handle this.",                "Standard", "2 years"),
        ("Liam",    "Scott",    "C012", "2025-03-12",  1, "The outage map showed my street as 'operational' during a 6-hour blackout. Completely wrong.",                 "Green",    "1 year"),
        ("Maria",   "Green",    "C013", "2025-03-13",  8, "New payment flow in the app is much smoother after the latest update. Good improvement.",                      "Standard", "4 years"),
        ("Nathan",  "Adams",    "C014", "2025-03-14",  1, "8-hour outage last week. Not a single text, email, or app notification the whole time.",                       "Basic",    "2 years"),
        ("Olivia",  "Baker",    "C015", "2025-03-15",  2, "4-hour outage. I only found out power was back when the lights came on. Zero communication.",                  "Standard", "6 years"),
        ("Paul",    "Nelson",   "C016", "2025-03-16",  3, "Lost power for 3 days in the ice storm. I get outages happen — just keep customers updated.",                  "Premium",  "10 years"),
        ("Quinn",   "Carter",   "C017", "2025-03-17",  9, "Reported outage online, got a confirmation, power restored in 2 hours. That's how it should work.",            "Standard", "3 years"),
        ("Rachel",  "Mitchell", "C018", "2025-03-18",  1, "Four outages in two months. Every estimated restoration time has been wildly inaccurate.",                     "Green",    "2 years"),
        ("Sam",     "Perez",    "C019", "2025-03-19",  3, "Planned maintenance but I only found out the day before. Two weeks' notice would help me plan.",               "Standard", "7 years"),
        ("Tara",    "Roberts",  "C020", "2025-03-20",  1, "Smart meter crew missed our appointment twice with no warning. Rescheduled three times total.",                "Basic",    "1 year"),
        ("Uma",     "Turner",   "C021", "2025-03-21",  1, "Smart meter shows usage 40% above my old analog meter. These readings must be wrong.",                         "Standard", "4 years"),
        ("Victor",  "Phillips", "C022", "2025-03-22",  9, "Smart meter installed smoothly. Hourly usage view is really helpful for planning my day.",                     "Premium",  "2 years"),
        ("Wendy",   "Campbell", "C023", "2025-03-23",  2, "Since the smart meter went in, my bill went up significantly with no explanation from anyone.",                "Standard", "3 years"),
        ("Xander",  "Parker",   "C024", "2025-03-24",  1, "45-minute hold for a simple billing question. Please invest in self-service options.",                         "Basic",    "5 years"),
        ("Yasmine", "Evans",    "C025", "2025-03-25",  1, "The chatbot can't answer anything and there's no easy way to reach a real person.",                            "Standard", "8 months"),
        ("Zoe",     "Edwards",  "C026", "2025-03-26",  9, "Called about a billing error. The agent was helpful and resolved it in under 10 minutes. Great service.",      "Premium",  "6 years"),
        ("Aaron",   "Collins",  "C027", "2025-03-27",  1, "My incorrect meter reading complaint has been open 3 weeks with zero updates or follow-up.",                   "Standard", "2 years"),
        ("Beth",    "Stewart",  "C028", "2025-03-28",  9, "Time-of-use rates saved me $40 last month by shifting laundry and the dishwasher to evenings.",                "Green",    "1 year"),
        ("Carl",    "Morris",   "C029", "2025-03-29",  2, "20% year-over-year rate increase with no explanation or advance notice. Very frustrating.",                    "Standard", "12 years"),
        ("Diana",   "Rogers",   "C030", "2025-03-30",  8, "The seasonal rate transparency report finally explains why my bill changes each quarter. Very helpful.",       "Premium",  "8 years"),
    ]

    df = pd.DataFrame(rows, columns=[
        "customer_first_name", "customer_last_name", "customer_id",
        "survey_completed_at", "recommendation_score_0_to_10",
        "written_explanation", "electricity_plan", "customer_tenure",
    ])

    # Derive NPS group from the 0–10 score
    df["nps_group"] = df["recommendation_score_0_to_10"].apply(_nps_group)
    # Create a survey response ID distinct from the customer ID
    df["survey_response_id"] = "SR-" + df["customer_id"].str.replace("C", "", regex=False).str.zfill(4)
    return df


def _nps_group(score: int) -> str:
    """Classify a 0–10 NPS score into promoter / passive / detractor."""
    if score >= 9:
        return "promoter"
    if score >= 7:
        return "passive"
    return "detractor"


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Validate
# ═══════════════════════════════════════════════════════════════════════════════

def validate_reviews_csv(df: pd.DataFrame) -> dict:
    """
    Step 1 — Validate the uploaded CSV before any analysis runs.

    Checks for required columns, empty files, low row counts, and score
    range issues. Returns a structured dict so the app can display validation
    status (and block the pipeline if validation fails) before wasting compute
    on bad data.

    Parameters
    ----------
    df : pd.DataFrame
        The raw DataFrame loaded from the user's uploaded file.

    Returns
    -------
    dict
        validation_status : "passed" or "failed"
        missing_columns   : list of required column names not found
        warnings          : list of non-fatal data quality messages
        row_count         : int, total rows in the uploaded file
        usable_df         : the same df if passed, else None
    """
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    warnings = []

    if len(df) == 0:
        return {
            "validation_status": "failed",
            "missing_columns": missing_columns,
            "warnings": ["The file is empty — no data rows were found."],
            "row_count": 0,
            "usable_df": None,
        }

    # Non-fatal size warning
    if len(df) < 10:
        warnings.append(
            f"Only {len(df)} rows detected. Theme clustering works best with 10+ responses."
        )

    # NPS score range check
    if "recommendation_score_0_to_10" in df.columns:
        scores = pd.to_numeric(df["recommendation_score_0_to_10"], errors="coerce")
        out_of_range = int(((scores < 0) | (scores > 10)).sum())
        null_scores = int(scores.isna().sum())
        if out_of_range:
            warnings.append(f"{out_of_range} score(s) are outside 0–10 and will be clamped.")
        if null_scores:
            warnings.append(f"{null_scores} row(s) have missing scores — will default to 5.")

    # Blank explanation check
    if "written_explanation" in df.columns:
        blank = int(
            df["written_explanation"].isna().sum()
            + (df["written_explanation"].astype(str).str.strip() == "").sum()
        )
        if blank:
            warnings.append(f"{blank} row(s) have blank written explanations and will be removed.")

    passed = len(missing_columns) == 0
    return {
        "validation_status": "passed" if passed else "failed",
        "missing_columns": missing_columns,
        "warnings": warnings,
        "row_count": len(df),
        "usable_df": df if passed else None,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Clean
# ═══════════════════════════════════════════════════════════════════════════════

def clean_reviews(df: pd.DataFrame) -> dict:
    """
    Step 2 — Clean and normalize the survey data before analysis.

    Actions taken in this step:
      - Remove rows where written_explanation is blank or null.
      - Normalize whitespace in written explanations (collapse extra spaces).
      - Ensure recommendation_score_0_to_10 is numeric; fill missing with 5.
      - Clamp scores to the 0–10 valid range.
      - Derive nps_group (promoter / passive / detractor) if not already present.
      - Ensure survey_response_id exists (derive from customer_id if needed).

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame from validate_reviews_csv.

    Returns
    -------
    dict
        cleaned_df      : pd.DataFrame, ready for the next pipeline step
        original_count  : int, row count before cleaning
        cleaned_count   : int, row count after cleaning
        removed_count   : int, rows dropped
        nps_group_added : bool, True if nps_group was derived here
    """
    original_count = len(df)
    df = df.copy()

    # Remove blank explanations
    df["written_explanation"] = df["written_explanation"].astype(str).str.strip()
    df = df[df["written_explanation"].str.len() > 0].copy()

    # Collapse extra whitespace in explanation text
    df["written_explanation"] = (
        df["written_explanation"]
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    # Numeric NPS score — clamp to valid range
    df["recommendation_score_0_to_10"] = (
        pd.to_numeric(df["recommendation_score_0_to_10"], errors="coerce")
        .fillna(5)
        .clip(0, 10)
        .astype(int)
    )

    # Derive nps_group if not already present
    nps_group_added = "nps_group" not in df.columns
    if nps_group_added:
        df["nps_group"] = df["recommendation_score_0_to_10"].apply(_nps_group)

    # Ensure survey_response_id exists
    if "survey_response_id" not in df.columns:
        if "customer_id" in df.columns:
            df["survey_response_id"] = "SR-" + df["customer_id"].astype(str)
        else:
            df["survey_response_id"] = [f"SR-{i:04d}" for i in range(len(df))]

    df = df.reset_index(drop=True)

    return {
        "cleaned_df": df,
        "original_count": original_count,
        "cleaned_count": len(df),
        "removed_count": original_count - len(df),
        "nps_group_added": nps_group_added,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Detect Duplicates
# ═══════════════════════════════════════════════════════════════════════════════

def detect_duplicates(df: pd.DataFrame) -> dict:
    """
    Step 3 — Detect possible duplicate survey responses.

    Three detection strategies are applied in order:
      1. If a 'duplicate_flag' column exists in the data, trust it.
      2. Flag exact duplicate written_explanation values (same text, different row).
      3. Flag the same customer_id appearing on the same survey date —
         likely an accidental double submission.

    The deduplicated DataFrame keeps only the first occurrence of each
    duplicate group. The full flagged DataFrame is preserved for audit purposes.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame from clean_reviews.

    Returns
    -------
    dict
        deduped_df        : pd.DataFrame, duplicates removed, ready for classification
        flagged_df        : pd.DataFrame, full data with 'is_duplicate' column
        duplicate_count   : int, number of rows flagged
        duplicate_summary : str, human-readable summary for the brief and UI
        duplicate_pairs   : list of (index_a, index_b) pairs
    """
    df = df.copy()
    flagged: set[int] = set()
    pairs: list[tuple[int, int]] = []

    # Strategy 1: trust an explicit duplicate_flag column if present
    if "duplicate_flag" in df.columns:
        pre_flagged = df.index[
            df["duplicate_flag"].astype(str).str.strip().str.lower() == "true"
        ].tolist()
        flagged.update(pre_flagged)

    # Strategy 2: exact text duplicates
    seen_texts: dict[str, int] = {}
    for idx, row in df.iterrows():
        text = str(row.get("written_explanation", "")).strip().lower()
        if not text:
            continue
        if text in seen_texts:
            if idx not in flagged:
                flagged.add(idx)
                pairs.append((seen_texts[text], idx))
        else:
            seen_texts[text] = idx

    # Strategy 3: same customer_id, same survey date
    if "customer_id" in df.columns and "survey_completed_at" in df.columns:
        df["_date_key"] = df["survey_completed_at"].astype(str).str[:10]
        seen_cust_date: dict[tuple, int] = {}
        for idx, row in df.iterrows():
            if idx in flagged:
                continue
            cid = str(row.get("customer_id", "")).strip()
            if not cid:
                continue
            key = (cid, row["_date_key"])
            if key in seen_cust_date:
                flagged.add(idx)
                pairs.append((seen_cust_date[key], idx))
            else:
                seen_cust_date[key] = idx
        df.drop(columns=["_date_key"], inplace=True)

    df["is_duplicate"] = df.index.isin(flagged)
    deduped_df = df[~df["is_duplicate"]].reset_index(drop=True)

    dup_count = len(flagged)
    if dup_count > 0:
        summary = (
            f"{dup_count} potential duplicate response(s) detected and excluded. "
            f"{len(deduped_df)} unique responses will be used for analysis."
        )
    else:
        summary = f"No duplicates detected. All {len(df)} responses appear unique."

    return {
        "deduped_df": deduped_df,
        "flagged_df": df,
        "duplicate_count": dup_count,
        "duplicate_summary": summary,
        "duplicate_pairs": pairs,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Classify Reviews
# ═══════════════════════════════════════════════════════════════════════════════

def classify_reviews(
    df: pd.DataFrame,
    product_goal: str,
    use_mock: bool = False,
) -> list[dict]:
    """
    Step 4 — Classify each review with structured product intelligence fields.

    In mock mode (use_mock=True):
        Keyword matching on written_explanation + NPS score heuristics determine
        each field. Fast, deterministic, requires no API key.

    In live mode (use_mock=False):
        All reviews are sent in a single batched Anthropic API call.
        The model returns a JSON array of classifications which are merged back
        with the original survey fields. Falls back to mock if the API call fails.

    Each classified review includes:
      survey_response_id    — unique identifier
      issue_category        — broad area: billing, digital, outage, service, etc.
      sentiment             — positive / neutral / negative
      severity_score        — 1 (minor) to 5 (critical / churn risk)
      opportunity_type      — specific opportunity from OPPORTUNITY_TYPES
      customer_impact       — individual or widespread
      evidence_quote        — short excerpt from the review text
      confidence            — high / medium / low

    Parameters
    ----------
    df : pd.DataFrame
        Deduplicated DataFrame from detect_duplicates.
    product_goal : str
        The PM's selected product goal (passed to the LLM for context).
    use_mock : bool
        If True, use rule-based classification. If False, call the API.

    Returns
    -------
    list[dict]
        One dict per review, containing all original survey fields plus the
        classification fields listed above.
    """
    if use_mock:
        return _classify_mock(df, product_goal)
    try:
        return _classify_live(df, product_goal)
    except Exception:
        # Graceful fallback — never leave the pipeline broken
        return _classify_mock(df, product_goal)


def _classify_mock(df: pd.DataFrame, product_goal: str) -> list[dict]:
    """
    Rule-based classification. No API key required.

    Scans each review's written_explanation for keywords from _KEYWORD_MAP.
    The first matching opportunity type wins. Severity and sentiment are
    derived from the 0–10 NPS score.
    """
    results = []
    for _, row in df.iterrows():
        text = str(row.get("written_explanation", "")).lower()
        score = int(row.get("recommendation_score_0_to_10", 5))

        # Determine opportunity type — first match in the ordered keyword map wins
        opp_type = "billing transparency"
        for opp, keywords in _KEYWORD_MAP.items():
            if any(kw in text for kw in keywords):
                opp_type = opp
                break

        # Severity: invert the NPS score to a 1–5 severity scale
        # A very unhappy customer (NPS 0–2) = severity 5 (critical)
        if score <= 2:
            severity_score = 5
        elif score <= 4:
            severity_score = 4
        elif score <= 6:
            severity_score = 3
        elif score <= 8:
            severity_score = 2
        else:
            severity_score = 1

        # Sentiment from NPS score band
        if score >= 9:
            sentiment = "positive"
        elif score >= 7:
            sentiment = "neutral"
        else:
            sentiment = "negative"

        # Customer impact — look for scale language
        widespread_keywords = [
            "neighborhood", "area", "street", "everyone",
            "whole city", "widespread", "all customers",
        ]
        customer_impact = (
            "widespread" if any(kw in text for kw in widespread_keywords) else "individual"
        )

        # Evidence quote: first 150 characters of the explanation
        explanation = str(row.get("written_explanation", ""))
        evidence_quote = (explanation[:150] + "…") if len(explanation) > 150 else explanation

        results.append({
            # Original survey fields — preserved for downstream steps
            "survey_response_id":           str(row.get("survey_response_id", "")),
            "customer_id":                  str(row.get("customer_id", "")),
            "written_explanation":          str(row.get("written_explanation", "")),
            "recommendation_score_0_to_10": score,
            "nps_group":                    str(row.get("nps_group", "")),
            "electricity_plan":             str(row.get("electricity_plan", "")),
            "customer_tenure":              str(row.get("customer_tenure", "")),
            # Classification fields
            "issue_category":    _CATEGORY_FROM_OPP.get(opp_type, "other"),
            "sentiment":         sentiment,
            "severity_score":    severity_score,
            "opportunity_type":  opp_type,
            "customer_impact":   customer_impact,
            "evidence_quote":    evidence_quote,
            "confidence":        "medium",  # rule-based is always medium confidence
        })
    return results


def _classify_live(df: pd.DataFrame, product_goal: str) -> list[dict]:
    """
    LLM-based classification via the Anthropic API.

    Sends all reviews in a single prompt for efficiency. The model returns
    a JSON array with one classification object per review, which is then
    merged back with the original survey row fields.
    """
    # Build a minimal input — only the fields the model needs
    reviews_input = [
        {
            "survey_response_id": str(row.get("survey_response_id", "")),
            "written_explanation": str(row.get("written_explanation", "")),
            "nps_score":           int(row.get("recommendation_score_0_to_10", 5)),
            "electricity_plan":    str(row.get("electricity_plan", "")),
        }
        for _, row in df.iterrows()
    ]

    prompt = f"""Classify each customer review for an electricity utility company.
The product manager's current goal is: {product_goal}

For each review, return a JSON object with exactly these fields:
  survey_response_id : copy from input
  issue_category     : one of [billing, digital, outage, service, pricing, plans, green, retention, other]
  sentiment          : one of [positive, neutral, negative]
  severity_score     : integer 1 (minor annoyance) to 5 (critical / likely to churn)
  opportunity_type   : one of {json.dumps(OPPORTUNITY_TYPES)}
  customer_impact    : one of [individual, widespread]
  evidence_quote     : verbatim excerpt under 120 characters from written_explanation
  confidence         : one of [high, medium, low]

Reviews to classify:
{json.dumps(reviews_input, indent=2)}

Return ONLY a valid JSON array, one object per review, in the same order. No markdown fences."""

    text = _llm_call(prompt, max_tokens=4096)
    classifications = _parse_json_response(text)

    # Merge LLM output with original survey row data
    results = []
    for i, (_, row) in enumerate(df.iterrows()):
        clf = classifications[i] if i < len(classifications) else {}
        results.append({
            "survey_response_id":           str(row.get("survey_response_id", "")),
            "customer_id":                  str(row.get("customer_id", "")),
            "written_explanation":          str(row.get("written_explanation", "")),
            "recommendation_score_0_to_10": int(row.get("recommendation_score_0_to_10", 5)),
            "nps_group":                    str(row.get("nps_group", "")),
            "electricity_plan":             str(row.get("electricity_plan", "")),
            "customer_tenure":              str(row.get("customer_tenure", "")),
            "issue_category":    clf.get("issue_category", "other"),
            "sentiment":         clf.get("sentiment", "neutral"),
            "severity_score":    int(clf.get("severity_score", 3)),
            "opportunity_type":  clf.get("opportunity_type", "billing transparency"),
            "customer_impact":   clf.get("customer_impact", "individual"),
            "evidence_quote":    clf.get("evidence_quote", ""),
            "confidence":        clf.get("confidence", "medium"),
        })
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Cluster Themes
# ═══════════════════════════════════════════════════════════════════════════════

def cluster_themes(classified_reviews: list[dict]) -> list[dict]:
    """
    Step 5 — Group classified reviews into product opportunity themes.

    Groups reviews by opportunity_type (already determined in Step 4) and
    computes aggregate statistics for each group. This step is fully
    deterministic — no LLM call is needed. The classification step did the
    semantic heavy lifting; this step organises the results into themes that
    the scoring step can rank.

    For each theme, calculates:
      - theme name and issue category
      - opportunity type
      - number of supporting reviews
      - average NPS score and average severity
      - sample evidence quotes (up to 3, from highest-severity reviews)
      - affected electricity plans
      - affected regions (empty unless region data is present)
      - confidence level (based on review count and severity consistency)

    Parameters
    ----------
    classified_reviews : list[dict]
        Output of classify_reviews().

    Returns
    -------
    list[dict]
        One theme dict per opportunity_type that has at least one review,
        sorted by review count descending. Scoring will re-sort by score.
    """
    # Group reviews by opportunity type
    groups: dict[str, list[dict]] = {}
    for review in classified_reviews:
        opp = review.get("opportunity_type", "billing transparency")
        groups.setdefault(opp, []).append(review)

    themes = []
    for opp_type, reviews in groups.items():
        nps_scores     = [r.get("recommendation_score_0_to_10", 5) for r in reviews]
        severity_scores = [r.get("severity_score", 3) for r in reviews]
        plans          = sorted({r.get("electricity_plan", "") for r in reviews if r.get("electricity_plan")})

        # Confidence: consistent + volume = high; single review = low
        sev_variance = (max(severity_scores) - min(severity_scores)) if len(severity_scores) > 1 else 0
        if len(reviews) >= 3 and sev_variance <= 1:
            confidence_level = "high"
        elif len(reviews) >= 2:
            confidence_level = "medium"
        else:
            confidence_level = "low"

        avg_sev = statistics.mean(severity_scores) if severity_scores else 3.0
        # dominant_severity is the string form used by scoring.py
        dominant_severity = _SEVERITY_STR.get(round(avg_sev), "medium")

        # Pick up to 3 evidence quotes, preferring higher-severity reviews
        sorted_by_severity = sorted(reviews, key=lambda r: r.get("severity_score", 0), reverse=True)
        sample_quotes = [r["evidence_quote"] for r in sorted_by_severity[:3] if r.get("evidence_quote")]

        detractor_count = sum(1 for r in reviews if r.get("nps_group") == "detractor")

        themes.append({
            "name":                  opp_type.title(),
            "opportunity_type":      opp_type,
            "issue_category":        _CATEGORY_FROM_OPP.get(opp_type, "other"),
            "review_count":          len(reviews),
            "avg_nps_score":         round(statistics.mean(nps_scores), 1) if nps_scores else 5.0,
            "avg_severity":          round(avg_sev, 1),
            "dominant_severity":     dominant_severity,
            "sample_evidence_quotes": sample_quotes,
            "affected_plans":        plans,
            "affected_regions":      [],  # extend if a region column is present
            "confidence_level":      confidence_level,
            "detractor_count":       detractor_count,
            "key_pain_point":        _key_pain_point(opp_type),
            # Full review list kept for scoring and card generation downstream
            "reviews":               reviews,
        })

    themes.sort(key=lambda t: t["review_count"], reverse=True)
    return themes


def _key_pain_point(opp_type: str) -> str:
    """One-sentence description of the core customer frustration per opportunity type."""
    descriptions = {
        "billing transparency":    "Customers cannot understand or verify their charges",
        "high bill investigation": "Bills are unexpectedly high with no clear explanation",
        "pricing clarity":         "Customers are confused about rates and plan value",
        "renewal education":       "Customers don't know how to renew or what happens at contract end",
        "app improvement":         "The app and portal fail at core tasks customers need most",
        "payment reliability":     "Payments fail or autopay behaves unexpectedly",
        "outage communication":    "Customers receive no updates during or after power outages",
        "support experience":      "Wait times are long and agents can't resolve issues efficiently",
        "plan education":          "Customers don't understand which plan suits them best",
        "green energy messaging":  "Customers want clearer information about green energy options",
        "retention risk":          "Customers are actively considering switching to a competitor",
    }
    return descriptions.get(opp_type, f"Issues related to {opp_type}")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Score Opportunities
# ═══════════════════════════════════════════════════════════════════════════════

def score_opportunities(theme_clusters: list[dict], product_goal: str) -> list[dict]:
    """
    Step 6 — Calculate priority scores and rank the opportunity themes.

    Delegates the weighted scoring formula to scoring.py:
      Frequency 25% · Severity 30% · Business Impact 20% · Confidence 15% · NPS Risk 10%

    Before calling scoring.py, each review's integer severity_score (1–5) is
    converted to the string format scoring.py expects ("low" / "medium" /
    "high" / "critical").

    Additionally applies a +0.5 goal-alignment bonus to themes whose
    opportunity_type directly relates to the product manager's selected goal.
    This is a lightweight way to surface goal-relevant work even when its
    raw score is similar to other themes.

    Parameters
    ----------
    theme_clusters : list[dict]
        Output of cluster_themes().
    product_goal : str
        The PM's selected product goal.

    Returns
    -------
    list[dict]
        Themes sorted by opportunity_score descending, with P1 / P2 / P3 labels.
    """
    # Convert severity_score (int) → severity (string) for each review so
    # scoring.py's _severity() and _nps_risk() functions work correctly.
    for theme in theme_clusters:
        for review in theme.get("reviews", []):
            sev_int = int(review.get("severity_score", 3))
            review["severity"] = _SEVERITY_STR.get(sev_int, "medium")

    ranked = rank_themes(theme_clusters, product_goal)
    ranked.sort(key=lambda t: t["opportunity_score"], reverse=True)
    return ranked


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 7 — Generate Insights Brief
# ═══════════════════════════════════════════════════════════════════════════════

def generate_insights_brief(
    ranked_opportunities: list[dict],
    duplicate_summary: str,
    product_goal: str,
    use_mock: bool = False,
) -> str:
    """
    Step 7 — Generate a concise, markdown-formatted product insights brief.

    In mock mode: builds the brief from scored theme data using a template.
    In live mode: sends the top-ranked themes to Claude and generates a
                  narrative brief. Falls back to template if the API call fails.

    The brief always includes:
      - Executive summary
      - Top 3 opportunities with evidence
      - Recommended next steps
      - Risks and caveats (including the duplicate detection note)
      - Human review notes (governance reminder)

    Parameters
    ----------
    ranked_opportunities : list[dict]
        Scored output from score_opportunities().
    duplicate_summary : str
        The duplicate detection summary from detect_duplicates().
    product_goal : str
        The PM's selected product goal.
    use_mock : bool
        If True, use template generation. If False, use the Anthropic API.

    Returns
    -------
    str
        Markdown-formatted insights brief.
    """
    if use_mock:
        return _brief_mock(ranked_opportunities, duplicate_summary, product_goal)
    try:
        return _brief_live(ranked_opportunities, duplicate_summary, product_goal)
    except Exception:
        return _brief_mock(ranked_opportunities, duplicate_summary, product_goal)


def _brief_mock(ranked_opportunities: list, duplicate_summary: str, product_goal: str) -> str:
    """Template-based brief. Driven by the scored theme data, not the LLM."""
    top3 = ranked_opportunities[:3]
    total_reviews = sum(t.get("review_count", 0) for t in ranked_opportunities)

    lines = [
        "## Product Insights Brief",
        f"*Goal: **{product_goal}***  |  "
        f"*{datetime.now().strftime('%Y-%m-%d')}*  |  "
        f"*Mode: Mock (rule-based)*",
        "",
        "---",
        "### Executive Summary",
        (
            f"Analysis of **{total_reviews} customer survey responses** identified "
            f"**{len(ranked_opportunities)} distinct product opportunity themes**. "
            f"The highest-priority area is **{top3[0]['name']}** "
            f"(score {top3[0]['opportunity_score']}/10), driven by: "
            f"*{top3[0]['key_pain_point'].lower()}*. "
            "Addressing the top two themes is recommended to reduce detractor volume "
            "and protect NPS."
        ),
        "",
        "---",
        "### Top 3 Opportunities",
    ]

    for i, t in enumerate(top3, 1):
        quotes = t.get("sample_evidence_quotes", [])
        quote_block = f"\n> \"{quotes[0]}\"" if quotes else ""
        lines += [
            f"**{i}. {t['name']}** — {t.get('priority', '?')} | "
            f"Score: {t['opportunity_score']}/10 | "
            f"{t['review_count']} reviews | Avg NPS: {t.get('avg_nps_score', '—')}/10",
            f"_{t['key_pain_point']}._{quote_block}",
            "",
        ]

    lines += [
        "---",
        "### Recommended Next Steps",
        f"1. **Immediate (this sprint):** Schedule a discovery session for "
        f"**{top3[0]['name']}**. Map the customer journey and identify root causes.",
        f"2. **Short term (next 2 sprints):** Target **{top3[1]['name'] if len(top3) > 1 else 'second priority'}**"
        " with a focused fix.",
        f"3. **Ongoing:** Monitor NPS for the {top3[0].get('detractor_count', 0)} "
        f"detractor(s) in the top theme and consider direct outreach.",
        "",
        "---",
        "### Risks & Caveats",
        f"- Unresolved **{top3[0]['name']}** issues put "
        f"{top3[0].get('detractor_count', 0)} customers at high churn risk.",
        f"- Duplicate detection: {duplicate_summary}",
        "- All findings are based on self-reported survey data and may not reflect "
        "the full population of affected customers.",
        "",
        "---",
        "### ⚠️ Human Review Notes",
        "This brief is an **AI-generated draft**. A product manager should validate "
        "these findings against support ticket volumes, product analytics, and "
        "stakeholder input before making roadmap decisions.",
    ]

    return "\n".join(lines)


def _brief_live(ranked_opportunities: list, duplicate_summary: str, product_goal: str) -> str:
    """LLM-generated brief for live mode. Sends top 6 themes to keep prompt manageable."""
    slim_themes = [
        {k: v for k, v in t.items() if k != "reviews"}
        for t in ranked_opportunities[:6]
    ]
    prompt = f"""Write a concise product insights brief for a product manager at an electricity utility.

Product Goal: {product_goal}
Data quality note: {duplicate_summary}

Scored opportunity themes (sorted highest priority first):
{json.dumps(slim_themes, indent=2)}

Write in markdown with exactly these sections:
## Product Insights Brief
### Executive Summary (2-3 sentences)
### Top 3 Opportunities (include evidence quotes and why each matters to the business)
### Recommended Next Steps (3 concrete, sprint-sized actions)
### Risks & Caveats (include the duplicate note above)
### ⚠️ Human Review Notes (governance reminder that this is an AI-generated draft)

Be direct and specific. Under 450 words."""

    return _llm_call(prompt, max_tokens=900)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 8 — Generate Backlog Cards
# ═══════════════════════════════════════════════════════════════════════════════

def generate_backlog_cards(
    ranked_opportunities: list[dict],
    max_cards: int = 5,
) -> list[dict]:
    """
    Step 8 — Create Trello-ready backlog card objects from scored opportunities.

    Takes the top N opportunities (controlled by max_cards) and constructs a
    structured card dict for each. This step is template-based — no LLM call
    is needed. Evidence quotes, NPS data, and scoring details from previous
    steps are embedded directly into each card.

    Each card includes a human_review_notes field as an explicit governance
    reminder that the PM must validate the card before committing to roadmap
    work or sending to Trello.

    Each card object includes:
      title                  — action-oriented card title with priority label
      description            — problem summary with supporting data
      priority               — P1 / P2 / P3
      labels                 — issue category and opportunity type tags
      evidence_quotes        — up to 3 verbatim excerpts from reviews
      confidence             — high / medium / low
      recommended_owner_area — team most likely responsible
      acceptance_criteria    — 3 testable, outcome-based criteria
      human_review_notes     — governance reminder with confidence level
      estimated_effort       — S / M / L (derived from priority)
      user_story             — "As a customer..." format for Trello description

    Parameters
    ----------
    ranked_opportunities : list[dict]
        Scored output from score_opportunities().
    max_cards : int
        Maximum number of cards to generate (controlled by the sidebar setting).

    Returns
    -------
    list[dict]
        Card objects sorted P1 → P2 → P3 (inherits the ranked order).
    """
    _EFFORT = {"P1": "L", "P2": "M", "P3": "S"}
    top = ranked_opportunities[:max_cards]
    cards = []

    for t in top:
        category  = t.get("issue_category", "other")
        priority  = t.get("priority", "P3")
        opp_type  = t.get("opportunity_type", "")
        quotes    = t.get("sample_evidence_quotes", [])
        plans     = t.get("affected_plans", [])
        plan_str  = ", ".join(plans[:2]) if plans else "standard"
        conf      = t.get("confidence_level", "medium")

        cards.append({
            "title": f"[{priority}] Resolve: {t['name']}",
            "description": (
                f"{t['review_count']} customers reported issues with **{opp_type}**. "
                f"Core pain point: {t['key_pain_point']}. "
                f"Average NPS for this group: {t.get('avg_nps_score', '—')}/10."
            ),
            "priority": priority,
            "labels": sorted({category, opp_type.replace(" ", "-")}),
            "evidence_quotes": quotes,
            "confidence": conf,
            "opportunity_score": t.get("opportunity_score", 0),
            "recommended_owner_area": _OWNER_AREA.get(category, "Product"),
            "affected_plans": plans,
            "acceptance_criteria": [
                f"The {opp_type} experience is resolved for ≥95% of affected customers.",
                f"Customer-reported {category} issues decrease by ≥20% within 60 days of launch.",
                f"NPS for the '{opp_type}' theme improves by ≥1.0 point in the next survey cycle.",
            ],
            "human_review_notes": (
                f"⚠️ AI-generated draft — confidence: {conf}. "
                "Validate findings with support ticket data and product analytics "
                "before adding to the roadmap or sending to Trello."
            ),
            "estimated_effort": _EFFORT.get(priority, "M"),
            # user_story used by trello_client.py when formatting the card description
            "user_story": (
                f"As a customer on the {plan_str} plan, "
                f"I want {t['key_pain_point'].rstrip('.').lower()} "
                "so that I can trust my electricity provider."
            ),
        })

    return cards


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 9 — Orchestrator
# ═══════════════════════════════════════════════════════════════════════════════

def run_agent_pipeline(
    df: pd.DataFrame,
    product_goal: str,
    max_cards: int = 5,
    use_mock: bool = False,
) -> dict:
    """
    Orchestrator — run all 8 pipeline steps in sequence and return a
    single results dictionary with every intermediate output.

    This function ties together the individual steps so that callers (such
    as the Streamlit app) can run the full pipeline with one call and then
    access any step's output from the returned dict — useful for populating
    multiple result tabs without re-running steps.

    The pipeline is deliberately sequential and non-autonomous: each step
    receives only the output of the previous step. There is no agent loop,
    no tool use, and no self-directed action. A human reviews the final
    output before any external action (Trello) is taken.

    Parameters
    ----------
    df : pd.DataFrame   — raw uploaded DataFrame
    product_goal : str  — PM's selected product focus area
    max_cards : int     — cap on the number of backlog cards generated
    use_mock : bool     — True = rule-based; False = Anthropic API

    Returns
    -------
    dict with keys:
        success              : bool
        errors               : list of error strings (empty on success)
        validation           : output of validate_reviews_csv
        clean                : output of clean_reviews
        duplicates           : output of detect_duplicates
        classified_reviews   : output of classify_reviews
        themes               : output of cluster_themes
        scored_opportunities : output of score_opportunities
        insights_brief       : output of generate_insights_brief
        backlog_cards        : output of generate_backlog_cards
    """
    result: dict = {"success": False, "errors": []}

    # Step 1 — Validate
    validation = validate_reviews_csv(df)
    result["validation"] = validation
    if validation["validation_status"] == "failed":
        result["errors"].append(
            "Validation failed. Missing columns: "
            + ", ".join(validation["missing_columns"])
        )
        return result

    # Step 2 — Clean
    clean = clean_reviews(df)
    result["clean"] = clean
    df_clean = clean["cleaned_df"]

    # Step 3 — Deduplicate
    duplicates = detect_duplicates(df_clean)
    result["duplicates"] = duplicates
    df_deduped = duplicates["deduped_df"]

    # Step 4 — Classify
    classified = classify_reviews(df_deduped, product_goal, use_mock=use_mock)
    result["classified_reviews"] = classified

    # Step 5 — Cluster
    themes = cluster_themes(classified)
    result["themes"] = themes

    # Step 6 — Score
    scored = score_opportunities(themes, product_goal)
    result["scored_opportunities"] = scored

    # Step 7 — Brief
    brief = generate_insights_brief(
        scored,
        duplicates["duplicate_summary"],
        product_goal,
        use_mock=use_mock,
    )
    result["insights_brief"] = brief

    # Step 8 — Cards
    cards = generate_backlog_cards(scored, max_cards=max_cards)
    result["backlog_cards"] = cards

    result["success"] = True
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Baseline — single-prompt comparison (used in the Evaluation tab)
# ═══════════════════════════════════════════════════════════════════════════════

def run_baseline(
    df: pd.DataFrame,
    product_goal: str,
    use_mock: bool = False,
) -> str:
    """
    Single-prompt baseline for comparison against the agentic pipeline.

    This represents the simplest possible approach: send all reviews to an LLM
    in one prompt and ask for a summary. No classification, clustering, or
    scoring. Used in the Evaluation tab to show concretely what the 8-step
    pipeline adds over a basic prompt.

    Parameters
    ----------
    df : pd.DataFrame   — any survey DataFrame with a written_explanation column
    product_goal : str  — included in the prompt for context
    use_mock : bool     — True returns a static mock response

    Returns
    -------
    str  — plain text or markdown summary.
    """
    if not use_mock:
        try:
            reviews_text = "\n".join(
                f"[NPS {row.get('recommendation_score_0_to_10', '?')}/10] "
                f"{row.get('written_explanation', '')}"
                for _, row in df.iterrows()
            )
            prompt = BASELINE_PROMPT.format(reviews_text=reviews_text)
            return _llm_call(prompt, max_tokens=600)
        except Exception:
            pass  # fall through to mock

    return (
        "**Summary:** Customers are primarily frustrated by unexpected billing changes, "
        "lack of communication during power outages, and a mobile app that frequently fails. "
        "Positive feedback centres on time-of-use pricing benefits and individual service interactions.\n\n"
        "**Top 5 Issues:**\n"
        "- Unexplained or unexpected bill increases\n"
        "- No proactive communication during outages\n"
        "- Mobile app crashes and broken usage graphs\n"
        "- Long customer service hold times\n"
        "- Smart meter installation delays and inaccurate readings\n\n"
        "**Recommended Actions:**\n"
        "1. Add real-time outage status notifications via SMS and app push.\n"
        "2. Investigate and resolve mobile app stability on iOS.\n"
        "3. Add a billing explainer that clearly breaks down each charge.\n\n"
        "---\n"
        "*This is the baseline output — one prompt, no classification, no clustering, "
        "no scoring. Compare it against the full agentic pipeline output.*"
    )
