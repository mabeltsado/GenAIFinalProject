"""
prompts.py
All LLM prompts used by the Review2Roadmap pipeline.

Keeping prompts in a single file makes them easy to inspect, version, and swap —
which matters when comparing prompt strategies against the baseline.
"""

# ── System context ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a senior product manager analyzing customer feedback for an \
electricity utility company. You extract structured insights from reviews and generate \
actionable product backlog items. Always respond with valid JSON when asked to do so."""

# ── Step 4: Classify ──────────────────────────────────────────────────────────

CLASSIFY_PROMPT = """\
Classify each customer review below. Return a JSON array — one object per review, \
in the same order — with exactly these fields:

  category       : one of [billing, outage, app_portal, service, pricing, smart_meter, other]
  sentiment      : one of [positive, neutral, negative]
  severity       : one of [low, medium, high, critical]
  opportunity_type: one of [pain_point, feature_request, compliment, question]
  customer_impact: one of [individual, neighborhood, widespread]

Reviews (JSON array of objects with review_id and review_text):
{reviews_json}

Return ONLY the JSON array. No explanation, no markdown fences."""

# ── Step 5: Cluster ───────────────────────────────────────────────────────────

CLUSTER_PROMPT = """\
You have classified customer reviews by category and severity. Now group them into \
meaningful themes — clusters of related issues that point to the same underlying \
product problem or opportunity.

Classified reviews:
{classified_json}

Return a JSON array of theme objects. Each object must have:
  name            : short descriptive label, 5 words max
  description     : one sentence explaining what customers are experiencing
  category        : primary category for this theme
  review_indices  : list of 0-based indices from the input that belong here
  dominant_severity: most common severity level in this theme
  key_pain_point  : the core frustration or unmet need in plain language

Aim for 4–7 themes. Every review index must appear in exactly one theme. \
Return ONLY the JSON array."""

# ── Step 7: Insights brief ────────────────────────────────────────────────────

BRIEF_PROMPT = """\
Write a concise product insights brief for a product manager.

Product Goal: {product_goal}

Scored opportunity themes (sorted highest priority first):
{themes_json}

Structure the brief in markdown with these sections:
1. **Executive Summary** — 2–3 sentences on the overall picture
2. **Top 3 Opportunities** — why each matters and what customers are saying
3. **Risk** — what happens if the top issue is ignored
4. **Recommended Next Step** — one concrete action the team can take this sprint

Keep the total under 350 words. Be direct and specific."""

# ── Step 8: Backlog cards ─────────────────────────────────────────────────────

CARD_PROMPT = """\
Convert these scored product opportunities into Trello-ready backlog cards.

Product Goal: {product_goal}

Opportunities (sorted by priority):
{themes_json}

Return a JSON array of card objects, sorted P1 → P2 → P3. Each card must have:
  title              : action-oriented title, under 10 words
  description        : 1–2 sentences describing the problem or opportunity
  user_story         : "As a [user], I want [action] so that [benefit]"
  acceptance_criteria: list of exactly 3 testable criteria
  labels             : list of relevant tags from [billing, outage, app, ux, communication, meter, pricing, data]
  estimated_effort   : one of [S, M, L, XL]
  priority           : one of [P1, P2, P3]

Return ONLY the JSON array."""

# ── Baseline (comparison benchmark) ──────────────────────────────────────────

BASELINE_PROMPT = """\
You are a product manager assistant. Read the customer reviews below and respond with:

1. A 3-sentence summary of the main issues customers are experiencing.
2. A bulleted list of the top 5 issues.
3. Three recommended product actions.

Be concise and practical.

Reviews:
{reviews_text}"""
