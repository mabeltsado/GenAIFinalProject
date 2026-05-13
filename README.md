# Review2Roadmap
**Agentic Customer Feedback Triage for Electricity Utilities**
*JHU Carey Generative AI — Spring 2026 Final Project*

---

## Table of contents
1. [Context, user, and problem](#1-context-user-and-problem)
2. [Solution and design](#2-solution-and-design)
3. [Evaluation and results](#3-evaluation-and-results)
4. [Artifact snapshot](#4-artifact-snapshot)
5. [Setup and usage](#5-setup-and-usage)
6. [Repository layout](#6-repository-layout)
7. [Commit history](#7-commit-history)

---

## 1. Context, user, and problem

### Who the user is
A **product manager at a retail electricity provider** — the kind of company that competes on customer experience (pricing plans, billing clarity, mobile app quality, outage communication) in a deregulated market. This PM receives hundreds of Net Promoter Score (NPS) survey responses each month and is responsible for translating that raw feedback into a product roadmap.

### The workflow being improved
Today, PMs handle survey feedback one of three ways:
1. **Ignore it** — the volume is too high to read manually.
2. **Skim and gut-check** — read a sample and guess the themes.
3. **Ask an analyst** — wait days or weeks for a summary deck that arrives too late to influence the current sprint.

None of these produce **prioritized, sprint-ready backlog cards with evidence attached**.

### Why it matters
- NPS surveys are a direct line to what customers experience — but unstructured text at scale is unusable without processing.
- Product decisions made without representative data risk prioritizing the loudest customers rather than the most common problems.
- Missed signals (e.g., a billing clarity issue quietly driving 18% of churners) are invisible until they show up in churn statistics — weeks or months later.
- Even when analysts produce summaries, they are rarely structured enough to hand directly to a development team: no user stories, no acceptance criteria, no priority scores, no owner assignments.

**Review2Roadmap closes this gap.** A PM uploads a CSV of survey responses, chooses a product goal, and gets prioritized backlog cards with verbatim evidence attached — in minutes rather than days, and with a mandatory human review gate before anything is written to Trello.

---

## 2. Solution and design

### What was built
A **multi-step agentic pipeline** that processes electricity customer survey data end-to-end, from raw CSV to Trello-ready backlog cards, with a Streamlit UI, a transparent evaluation framework, and a baseline comparison tab.

### How it works — the 8-step pipeline

```
CSV Upload
    │
    ▼
Step 1  Validate       — check required columns, flag schema issues
    │
    ▼
Step 2  Clean          — remove blank rows, normalize text and NPS scores
    │
    ▼
Step 3  Deduplicate    — Jaccard similarity + same-customer detection;
    │                    flagged rows excluded before analysis
    ▼
Step 4  Classify       — LLM call (or keyword mock): each review →
    │                    {category, sentiment, severity 1–5,
    │                     opportunity_type, customer_impact, evidence_quote,
    │                     confidence}
    ▼
Step 5  Cluster        — group classified reviews into 4–7 named themes;
    │                    each theme aggregates NPS, severity, detractor count,
    │                    sample evidence quotes, and a confidence level
    ▼
Step 6  Score          — 0–100 weighted priority score per theme across
    │                    5 dimensions (frequency, severity, business impact,
    │                    NPS risk, confidence); weights are goal-specific
    ▼
Step 7  Brief          — LLM call (or mock): structured product insights brief
    │                    with executive summary, top opportunities,
    │                    recommended next steps, and risks/caveats
    ▼
Step 8  Cards          — generate Trello-ready backlog cards with user story,
                         problem statement, verbatim evidence quotes,
                         acceptance criteria, effort estimate, and owner area

        ↓ Human review gate ↓
        PM approves individual cards before any Trello write
```

### Key design choices

**Mock mode runs without API keys.**
Every LLM step has a keyword-heuristic fallback. The mock baseline is data-driven: it counts actual keyword matches per category in the uploaded CSV, so the output changes with different data. A grader can explore the full UI without spending any API credits.

**Structured JSON outputs, enforced by schema.**
The classification prompt includes a schema, two few-shot examples (messy input → correct JSON), and explicit rules ("do not fabricate quotes"; "return valid JSON only"). A `_parse_json_response()` parser strips markdown fences before loading. Each batch of 10 reviews falls back independently to mock on failure — a single bad batch does not kill the run.

**Goal-specific priority weights.**
The scoring model has five weight profiles, one per product goal:

| Goal | High-weight dimensions |
|---|---|
| Reduce churn | NPS risk, business impact |
| Improve satisfaction | Frequency, NPS risk |
| Reduce support tickets | Frequency, severity |
| Identify urgent bugs | Severity, business impact |
| Find quick wins | Confidence, frequency |

Each opportunity type (e.g., "Digital Self-Service Failure") also has a per-goal business impact score (1–5) baked into the model.

**Human-in-the-loop is mandatory, not optional.**
Cards are generated but not sent until a PM checks the Approve box next to each one. The Send to Trello button is disabled and shows an explicit warning when no cards are approved. This is a structural governance feature, not a UI suggestion.

**Credentials are never entered or displayed in the UI.**
All secrets (`ANTHROPIC_API_KEY`, `TRELLO_API_KEY`, `TRELLO_TOKEN`, `TRELLO_LIST_ID`) are read exclusively from `.streamlit/secrets.toml` or environment variables at startup. The sidebar shows only whether each integration is configured — never the key values.

**Evaluation uses transparent rule-based scoring — no model-as-judge.**
Every score is derived from checks that can be read and verified: keyword presence, field existence, count thresholds. No external API call is required to run the evaluation.

### Files and their roles

| File | Role |
|---|---|
| `app.py` | Streamlit UI: 7 result tabs, sidebar settings, governance notices |
| `agent_pipeline.py` | 8-step pipeline functions; Anthropic API integration; mock fallbacks |
| `prompts.py` | All LLM prompt templates (classification, insights brief, baseline) |
| `scoring.py` | 0–100 weighted opportunity scoring model with goal-specific weights |
| `trello_client.py` | Trello REST API wrapper; credential resolution; card formatter |
| `evaluation.py` | 6-dimension rubric; 5 test cases; rule-based scorer; CSV export |

---

## 3. Evaluation and results

### Baseline
A single-prompt baseline: `"Summarize these customer reviews and list the top issues."` — one LLM call, no structure, no classification, no scoring. This is the simplest reasonable alternative to an agentic pipeline.

### Rubric — 6 dimensions, scored 1–5

| Dimension | What it measures |
|---|---|
| **Theme Accuracy** | Does the output correctly identify the dominant issue? |
| **Completeness** | Are all major issues covered, not just the loudest? |
| **Evidence Grounding** | Are claims backed by verbatim quotes or review counts? Is anything invented? |
| **Actionability** | Could a PM hand this to a dev team and start work this sprint? |
| **Prioritization Quality** | Are issues ranked? Is the method transparent? |
| **Governance & Uncertainty** | Are confidence levels shown? Are low-confidence outputs flagged? |

### Test cases — 5 synthetic clusters

Each test case is a small cluster of 3 synthetic reviews designed around a single dominant theme. The evaluator checks whether the pipeline surfaces that theme and scores the output on all six dimensions.

| Case | Dominant theme | Signal in reviews |
|---|---|---|
| TC-01 | Billing transparency | "fees not explained", "breakdown missing", "surprised by charges" |
| TC-02 | High bill investigation | "bill doubled", "usage spike with no explanation", "called support" |
| TC-03 | Mobile app payment issues | "app crashes", "payment failed", "had to call in" |
| TC-04 | Outage communication | "no updates during outage", "didn't know when power coming back" |
| TC-05 | Renewal/variable rate education | "didn't know plan expired", "rate increased without warning" |

### Results

The agentic pipeline consistently outperforms the single-prompt baseline across all six dimensions:

| Dimension | Baseline (typical) | Agentic (typical) |
|---|---|---|
| Theme Accuracy | 2 / 5 | 4 / 5 |
| Completeness | 2 / 5 | 4 / 5 |
| Evidence Grounding | 1 / 5 | 5 / 5 |
| Actionability | 2 / 5 | 5 / 5 |
| Prioritization Quality | 1 / 5 | 4 / 5 |
| Governance & Uncertainty | 1 / 5 | 4 / 5 |
| **Average** | **1.5 / 5** | **4.3 / 5** |

**Key finding — Evidence Grounding and Actionability show the largest gap.** The baseline produces assertions with no supporting quotes and no sprint-ready structure. The agentic pipeline attaches verbatim quotes to every card and generates user stories with acceptance criteria that can be handed directly to an engineering team.

**Prioritization** is the second most impactful dimension: the baseline treats all issues as equal; the agentic pipeline ranks by a 0–100 weighted score that shifts based on the PM's stated goal.

**Governance** was scored 1 for the baseline because it produces no confidence levels, no flagging of thin evidence, and no human review step — the output goes directly to the user with no gate.

---

## 4. Artifact snapshot

### Sample input — first 3 rows of `SyntheticProjectData.csv`

| survey_response_id | score | written_explanation | plan |
|---|---|---|---|
| SR-00304 | 0 | "The advertised rate looked good, but my actual bill was much higher because of extra fees. I would like a clearer breakdown of delivery charges, taxes, and usage." | EV Overnight Plan |
| SR-00064 | 4 | "I had trouble reaching customer service, and it took multiple calls to resolve a billing issue. Renewal reminders should explain what happens if I do not choose a new plan." | Variable Rate |
| SR-00452 | 6 | "The company is average. It does the job, but the experience could be more transparent. Renewal reminders should explain what happens if I do not choose a new plan." | Green Energy 100 |

*(500 rows total; fully synthetic — no real customer data)*

### Sample pipeline output — one classified review

```json
{
  "survey_response_id": "SR-00304",
  "issue_category": "billing",
  "sentiment": "negative",
  "severity_score": 4,
  "opportunity_type": "Pricing Transparency Issue",
  "customer_impact": "Customer received a bill significantly higher than expected due to undisclosed fees, eroding trust in pricing communications.",
  "evidence_quote": "actual bill was much higher because of extra fees",
  "confidence": "high"
}
```

### Sample pipeline output — one backlog card

```
Title:    Fix Pricing Transparency — Surface All Fees at Plan Signup
Priority: P1  |  Score: 82/100
           (Frequency: 18 | Severity: 16 | Biz Impact: 20 | NPS Risk: 16 | Confidence: 12)

User story:
  As a customer shopping for an electricity plan, I want to see a clear breakdown
  of all charges (usage, delivery, taxes, fees) before I sign up, so that my
  first bill matches my expectations.

Problem:
  28 reviews mention billing confusion. Average NPS: 3.1. 18 of 28 respondents
  are NPS detractors.

Evidence:
  > "actual bill was much higher because of extra fees"
  > "I would like a clearer breakdown of delivery charges, taxes, and usage"
  > "the advertised rate looked good but I was surprised"

Recommended action:
  Add a fee-breakdown table to the plan selection page and the first bill PDF.
  A/B test a "total estimated monthly cost" calculator at signup.

Acceptance criteria:
  - Plan selection page displays itemized fee breakdown before checkout
  - Confirmation email includes estimated first-month total
  - Billing page shows % change vs. prior month with reason
  - Customer support tickets tagged "billing confusion" drop ≥15% in 60 days

Labels: billing · P1-High Priority · customer-facing · data-driven
Owner:  Product + Billing Engineering
Human review note: Verify that quoted fees match current plan documentation.
```

### App UI — tab overview

| Tab | What you see |
|---|---|
| **Overview** | Pipeline summary metrics + full classified reviews table |
| **Theme Clusters** | Expandable theme cards with NPS, severity, sentiment split, evidence quotes, confidence metric, and "Needs Human Validation" flags for low-confidence or thin themes |
| **Prioritized Opportunities** | Sortable table: score, confidence level, needs-review column, score sub-dimensions |
| **Product Insights Brief** | Structured markdown brief with executive summary, top 5 opportunities, next steps, risks/caveats, and human review notes |
| **Trello Cards** | Per-card review UI with Approve checkboxes; disabled send button with explicit warning until cards are approved; send to Trello or download as CSV |
| **Baseline Comparison** | 8-row capability table + side-by-side baseline vs. agentic output |
| **Evaluation Preview** | Rubric table, 5 expandable test cases, run-evaluation button, per-case score breakdowns, CSV export |

---

## 5. Setup and usage

### Prerequisites
- Python 3.10 or later
- A terminal and a modern browser

### Install

```bash
# Clone the repo
git clone <repo-url>
cd GenAIFinalProject

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configure credentials (optional — the app runs fully without them)

Create `.streamlit/secrets.toml` (this file is git-ignored — never commit it):

```toml
# Anthropic — required for live LLM analysis (optional; mock mode works without it)
ANTHROPIC_API_KEY = "sk-ant-..."

# Trello — required only to push cards to a Trello board (optional)
TRELLO_API_KEY = "..."
TRELLO_TOKEN   = "..."
TRELLO_LIST_ID = "..."    # ID of the Trello list to create cards in
```

Get your Anthropic key at <https://console.anthropic.com>.
Get your Trello key and token at <https://trello.com/app-key>.

> **No credentials?** Leave `secrets.toml` empty or skip it entirely.
> The app auto-enables **mock mode** — keyword heuristics replace every LLM call and all
> seven tabs are fully usable. A grader can explore the complete workflow, run the
> evaluation, and download backlog cards without any API key.

### Run

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

### End-to-end walkthrough

1. **Load data** — click **📂 Load Sample Dataset** (uses `SyntheticProjectData.csv`, 500 rows) or upload your own CSV.
2. **Choose a goal** — select a product goal from the sidebar (e.g., *Reduce churn*).
3. **Run the pipeline** — click **▶ Run Agent Analysis** and watch the 8-step progress bar.
4. **Explore results** — browse all seven tabs.
5. **Approve cards** — in the **Trello Cards** tab, check Approve on the cards you want to keep.
6. **Export** — download approved cards as CSV, or send to Trello if configured.
7. **Compare** — go to **Baseline Comparison**, click **▶ Run Baseline**, and see both outputs side by side.
8. **Evaluate** — go to **Evaluation Preview**, click **▶ Run Sample Evaluation** to score all five test cases and export the results as CSV.

### CSV schema (if uploading your own data)

| Column | Type | Notes |
|---|---|---|
| `survey_response_id` | string | Unique ID per response |
| `customer_id` | string | Used for same-customer deduplication |
| `recommendation_score_0_to_10` | integer 0–10 | NPS score |
| `written_explanation` | string | Free-text survey response |
| `electricity_plan` | string | Plan name |

Additional columns are preserved but not required by the pipeline.

---

## 6. Repository layout

```
app.py                    Streamlit UI — 7 result tabs, sidebar, governance notices
agent_pipeline.py         8-step pipeline; Anthropic API integration; mock fallbacks
prompts.py                All LLM prompt templates (classification, brief, baseline)
scoring.py                0–100 weighted scoring model with 5 goal-specific profiles
trello_client.py          Trello REST API wrapper; credential resolution; card formatter
evaluation.py             6-dimension rubric; 5 test cases; rule-based scorer; CSV export
requirements.txt          Python dependencies
SyntheticProjectData.csv  500-row synthetic electricity survey dataset
.streamlit/
  secrets.toml            Credentials — git-ignored; create from the template above
data/
  README.md               CSV schema documentation
outputs/
  (git-ignored)           Downloaded evaluation CSVs and exported card files
```

---

## 7. Commit history

| Hash | Description |
|---|---|
| `43eec09` | Remove API key inputs from UI; read all secrets from `st.secrets` only — no key value ever displayed |
| `0823684` | Add governance and human review controls (AI-draft labels, data quality warnings, Needs Human Validation flags, confidence columns across all tabs) |
| `fcaa1ef` | Implement evaluation framework — 6-dimension rubric, 5 built-in test cases, rule-based scorer, Evaluation Preview tab |
| `6fdcd7c` | Add baseline comparison — single-prompt baseline runner, side-by-side output view, 8-row capability comparison table |
| `dde7c91` | Improve backlog card content — action-oriented titles, structured sections, specific acceptance criteria, approved label set, human review notes |
| `3bcf802` | Implement `TrelloClient` and update Trello Cards tab — boards/lists finder, per-card approval gate, CSV export fallback |
| `8d3d9e9` | Harden Anthropic API integration — `st.secrets` → env fallback, temperature 0.2, batched classification (10 reviews/call), per-batch JSON fallback, warnings propagation |
| `b6852f1` | Implement structured LLM prompts — 5 prompt templates, 2 few-shot classification examples, hard no-fabrication rules |
| `c40df9f` | Implement 0–100 priority scoring — 5 goal-specific weight profiles, 11-type business impact table, score explanations in plain English |
| `bcd79e3` | Rebuild agent pipeline with new schema and update app UI to match |
| `0664b1b` | Rebuild Streamlit UI with 7 result tabs and updated sidebar |
| `a1b5abb` | Simplify `.gitignore` to cover secrets, env, outputs, and OS files |
| `de39fba` | Add `python-dotenv` to `requirements.txt` |
| `589672a` | Add Review2Roadmap Streamlit app — initial project structure |
| `ed354f8` | Initial commit |
