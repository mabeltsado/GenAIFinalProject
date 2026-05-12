# Review2Roadmap
**An Agentic Product Feedback Triage Tool**
*JHU Carey Generative AI — Spring 2026 Final Project*

---

## What it does

Review2Roadmap turns raw electricity customer survey reviews into prioritized,
Trello-ready product backlog cards using a multi-step agentic pipeline powered by Claude.

A product manager uploads a CSV (or loads the built-in sample dataset), selects a product
goal, and the pipeline does the rest — classifying, clustering, scoring, and drafting cards.
The PM reviews and approves cards before anything is sent to Trello.

---

## Agentic pipeline (8 steps)

| Step | What happens |
|------|-------------|
| 1 Validate | Check CSV schema and data quality |
| 2 Clean | Remove blanks, normalize whitespace and ratings |
| 3 Deduplicate | Flag near-identical survey responses (Jaccard similarity) |
| 4 Classify | Label each review: category, sentiment, severity, opportunity type |
| 5 Cluster | Group reviews into 4–7 actionable themes |
| 6 Score | Rank themes by frequency, severity, business impact, confidence, NPS risk |
| 7 Brief | Generate a markdown product insights brief |
| 8 Cards | Generate Trello-ready backlog cards with user stories & acceptance criteria |

A **human review gate** (Step 9) requires the PM to approve cards before any Trello write.

---

## Course concepts demonstrated

| Concept | Where |
|---------|-------|
| Multi-step orchestration | `agent_pipeline.py` — 8 sequential steps |
| Function-style tool steps | Each step is a pure, composable function |
| Structured outputs | JSON classification schema; typed card objects |
| Human-in-the-loop | Card approval UI before Trello export |
| Evaluation design | Rubric + 10-review test set in `evaluation.py` |
| Governance | Validation, deduplication, and review gate |
| Baseline comparison | Single-prompt vs. agentic output side-by-side |

---

## Quick start

### 1 — Install dependencies
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2 — Add credentials (optional)
```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edit secrets.toml with your Anthropic API key and (optionally) Trello credentials
```
> **No API key?** Leave the key blank — the app runs in mock mode using keyword heuristics
> and templates so you can explore the full UI without spending any credits.

### 3 — Run
```bash
streamlit run app.py
```

---

## Repository layout

```
app.py                        Streamlit UI — all three tabs
agent_pipeline.py             8-step pipeline (validate → cards)
prompts.py                    All LLM prompts in one place
scoring.py                    Weighted opportunity scoring formula
trello_client.py              Trello REST API wrapper + mock mode
evaluation.py                 Test set (10 reviews) + rubric
requirements.txt              Python dependencies
.streamlit/
  secrets.toml.example        Credentials template (copy → secrets.toml)
data/
  README.md                   CSV schema + sample data instructions
outputs/
  .gitkeep                    Placeholder — downloaded outputs go here
```

---

## Data & privacy

All data is **synthetic**. No real customer PII is committed to this repo.
API keys and Trello credentials are stored in `.streamlit/secrets.toml`, which is git-ignored.

---

## Baseline vs. agentic

The **Evaluation** tab runs a single-prompt baseline (one call, summary + bullet list) alongside
the 8-step agentic output. A rubric with five dimensions — issue coverage, actionability,
prioritization, evidence grounding, and governance — guides side-by-side scoring.

---

## Trello setup (optional)

1. Get your API key and token at <https://trello.com/app-key>
2. Create a board with a **Backlog** list
3. Find your Board ID and List ID (visible in the board URL or Trello REST API)
4. Add all four values to `.streamlit/secrets.toml`
