"""
app.py
Review2Roadmap — Agentic Product Feedback Triage Tool
JHU Carey Generative AI Final Project (Spring 2026)

Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd

import agent_pipeline as pipeline
import evaluation as ev
from trello_client import TrelloClient, is_configured as trello_ready, mock_create_card

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Review2Roadmap",
    page_icon="🗺️",
    layout="wide",
)

# ── Sidebar: configuration ────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Configuration")

    st.subheader("Anthropic API")
    anthropic_key = st.text_input(
        "API Key",
        type="password",
        value=st.secrets.get("ANTHROPIC_API_KEY", ""),
        help="Leave blank to run in mock mode — no key needed for demo.",
    )
    mock_mode = not bool(anthropic_key)

    if mock_mode:
        st.info("🔄 Mock mode — keyword heuristics replace LLM calls.")
    else:
        st.success("✅ Live mode — using Claude via Anthropic API.")

    st.divider()
    st.subheader("Product Goal")
    product_goal = st.selectbox(
        "Select your focus area",
        [
            "Improve digital self-service",
            "Reduce billing disputes",
            "Improve outage communication",
            "Increase mobile app adoption",
            "Reduce call center volume",
        ],
    )

    st.divider()
    st.subheader("Trello (optional)")
    trello_key   = st.text_input("API Key",  type="password", key="tk", value=st.secrets.get("TRELLO_API_KEY", ""))
    trello_token = st.text_input("Token",    type="password", key="tt", value=st.secrets.get("TRELLO_TOKEN", ""))
    trello_board = st.text_input("Board ID", key="tb",                  value=st.secrets.get("TRELLO_BOARD_ID", ""))
    trello_list  = st.text_input("List ID (Backlog)", key="tl",         value=st.secrets.get("TRELLO_LIST_ID", ""))
    trello_configured = trello_ready(trello_key, trello_token, trello_board, trello_list)

    if trello_configured:
        st.success("✅ Trello connected.")
    else:
        st.caption("Fill all four fields to enable Trello export.")

# ── Session state defaults ────────────────────────────────────────────────────

_STATE_KEYS = ["df_raw", "df_clean", "df_classified", "themes",
               "scored_themes", "brief", "cards", "pipeline_done",
               "baseline_out", "dup_pairs", "validation"]

for k in _STATE_KEYS:
    if k not in st.session_state:
        st.session_state[k] = None

if "approved" not in st.session_state:
    st.session_state.approved = set()

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_workflow, tab_eval, tab_about = st.tabs(["🔄 Workflow", "📊 Evaluation", "ℹ️ About"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — WORKFLOW
# ══════════════════════════════════════════════════════════════════════════════

with tab_workflow:
    st.title("🗺️ Review2Roadmap")
    st.caption(
        "Upload electricity customer survey data → get a prioritized product backlog. "
        "Every step is shown so you can see how the agentic pipeline works."
    )

    # ── Step 1: Load data ──────────────────────────────────────────────────────

    st.header("Step 1 · Load Data")
    col_upload, col_sample = st.columns([3, 1])

    with col_upload:
        uploaded = st.file_uploader("Upload a survey CSV", type=["csv"])

    with col_sample:
        st.write("")  # vertical spacing
        st.write("")
        if st.button("📂 Load Sample Dataset"):
            st.session_state.df_raw = pipeline.generate_sample_data()
            # Reset downstream state when new data is loaded
            for k in ["df_clean", "df_classified", "themes", "scored_themes",
                      "brief", "cards", "pipeline_done", "approved"]:
                st.session_state[k] = None
            st.session_state.approved = set()
            st.success("30 synthetic electricity reviews loaded.")

    if uploaded is not None:
        try:
            st.session_state.df_raw = pd.read_csv(uploaded)
            for k in ["df_clean", "df_classified", "themes", "scored_themes",
                      "brief", "cards", "pipeline_done"]:
                st.session_state[k] = None
            st.session_state.approved = set()
            st.success(f"Uploaded {len(st.session_state.df_raw):,} rows.")
        except Exception as exc:
            st.error(f"Could not read CSV: {exc}")

    if st.session_state.df_raw is not None:
        with st.expander("Preview raw data", expanded=False):
            st.dataframe(st.session_state.df_raw.head(10), use_container_width=True)

    st.divider()

    # ── Run pipeline ───────────────────────────────────────────────────────────

    run_disabled = st.session_state.df_raw is None
    if st.button("🚀 Run Pipeline", type="primary", disabled=run_disabled):
        client = pipeline.get_client(None if mock_mode else anthropic_key)
        prog = st.progress(0, text="Starting…")
        status = st.empty()

        # 1 — Validate
        status.markdown("**Step 1/8 — Validating CSV…**")
        val = pipeline.validate_csv(st.session_state.df_raw)
        st.session_state.validation = val
        prog.progress(10)

        if not val["valid"]:
            status.empty()
            st.error("Validation failed: " + " | ".join(val["errors"]))
            st.stop()

        if val["warnings"]:
            for w in val["warnings"]:
                st.warning(w)

        # 2 — Clean
        status.markdown("**Step 2/8 — Cleaning reviews…**")
        df_clean, clean_stats = pipeline.clean_reviews(st.session_state.df_raw)
        st.session_state.df_clean = df_clean
        prog.progress(22)

        # 3 — Deduplicate
        status.markdown("**Step 3/8 — Detecting duplicate responses…**")
        df_deduped, dup_pairs = pipeline.detect_duplicates(df_clean)
        st.session_state.dup_pairs = dup_pairs
        prog.progress(34)

        # 4 — Classify
        status.markdown("**Step 4/8 — Classifying reviews…**")
        df_classified = pipeline.classify_reviews(df_deduped, client=client, mock=mock_mode)
        st.session_state.df_classified = df_classified
        prog.progress(48)

        # 5 — Cluster
        status.markdown("**Step 5/8 — Clustering themes…**")
        themes = pipeline.cluster_themes(df_classified, client=client, mock=mock_mode)
        st.session_state.themes = themes
        prog.progress(62)

        # 6 — Score
        status.markdown("**Step 6/8 — Scoring opportunities…**")
        scored = pipeline.score_opportunities(themes)
        st.session_state.scored_themes = scored
        prog.progress(74)

        # 7 — Brief
        status.markdown("**Step 7/8 — Generating insights brief…**")
        brief = pipeline.generate_brief(scored, product_goal, client=client, mock=mock_mode)
        st.session_state.brief = brief
        prog.progress(88)

        # 8 — Cards
        status.markdown("**Step 8/8 — Generating backlog cards…**")
        cards = pipeline.generate_backlog_cards(scored, product_goal, client=client, mock=mock_mode)
        st.session_state.cards = cards
        st.session_state.approved = set()
        prog.progress(100)
        status.markdown("✅ **Pipeline complete!**")
        st.session_state.pipeline_done = True

    # ── Results ────────────────────────────────────────────────────────────────

    if st.session_state.pipeline_done:

        # Pipeline summary metrics
        val = st.session_state.validation or {}
        dup_count = len(st.session_state.dup_pairs or [])
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Reviews loaded",     val.get("row_count", "—"))
        col2.metric("After cleaning",     len(st.session_state.df_clean))
        col3.metric("Duplicates flagged", dup_count)
        col4.metric("Themes found",       len(st.session_state.themes))

        st.divider()

        # Step 4 — Classified table
        st.subheader("Step 4 · Classified Reviews")
        show_cols = ["review_id", "review_text", "category", "sentiment", "severity", "opportunity_type"]
        available = [c for c in show_cols if c in st.session_state.df_classified.columns]
        st.dataframe(
            st.session_state.df_classified[available],
            use_container_width=True,
            height=250,
        )

        st.divider()

        # Step 6 — Scored opportunities
        st.subheader("Step 6 · Scored Opportunities")
        st.caption("Weighted score (1–10): frequency 25% · severity 30% · business impact 20% · confidence 15% · NPS risk 10%")
        score_rows = [
            {
                "Priority": t.get("priority", "?"),
                "Theme": t["name"],
                "Reviews": t["review_count"],
                "Severity": t.get("dominant_severity", ""),
                "Score": t.get("opportunity_score", 0),
                "Freq": t.get("frequency_score", 0),
                "Sev": t.get("severity_score", 0),
                "BizImpact": t.get("business_impact_score", 0),
            }
            for t in st.session_state.scored_themes
        ]
        st.dataframe(
            pd.DataFrame(score_rows).sort_values("Score", ascending=False),
            use_container_width=True,
        )

        st.divider()

        # Step 7 — Insights brief
        st.subheader("Step 7 · Product Insights Brief")
        st.markdown(st.session_state.brief)

        st.divider()

        # Step 8–9 — Card approval
        st.subheader("Step 8–9 · Review & Approve Backlog Cards")
        st.caption(
            "Check the cards you want to send to Trello, then click **Send Approved Cards**. "
            "This human review gate is a governance feature of the agentic workflow."
        )

        cards = st.session_state.cards or []
        for i, card in enumerate(cards):
            approved_icon = "✅" if i in st.session_state.approved else "⬜"
            priority = card.get("priority", "?")
            with st.expander(
                f"{approved_icon} [{priority}] {card['title']}",
                expanded=(i < 2),
            ):
                left, right = st.columns([5, 1])
                with left:
                    st.markdown(f"**User Story:** {card.get('user_story', '')}")
                    st.markdown(f"**Description:** {card.get('description', '')}")
                    criteria = card.get("acceptance_criteria", [])
                    if criteria:
                        st.markdown("**Acceptance Criteria:**")
                        for c in criteria:
                            st.markdown(f"- {c}")
                    labels = card.get("labels", [])
                    if labels:
                        st.markdown("**Labels:** " + "  ·  ".join(f"`{l}`" for l in labels))
                    st.markdown(
                        f"**Effort:** `{card.get('estimated_effort', '?')}`  "
                        f"**Score:** {card.get('opportunity_score', '—')}"
                    )
                with right:
                    checked = st.checkbox(
                        "Approve",
                        key=f"approve_{i}",
                        value=(i in st.session_state.approved),
                    )
                    if checked:
                        st.session_state.approved.add(i)
                    else:
                        st.session_state.approved.discard(i)

        st.divider()

        # Step 10 — Trello send
        st.subheader("Step 10 · Send to Trello")
        n_approved = len(st.session_state.approved)

        if not trello_configured:
            st.info(
                "Trello credentials are not configured. "
                "Fill in all four Trello fields in the sidebar to enable this step. "
                "Cards will be sent in mock mode for demonstration."
            )

        send_label = f"📤 Send {n_approved} Approved Card(s) to Trello"
        if n_approved == 0:
            st.caption("Approve at least one card above to enable sending.")

        if st.button(send_label, disabled=(n_approved == 0)):
            approved_cards = [cards[i] for i in sorted(st.session_state.approved)]
            results = []

            if trello_configured:
                trello = TrelloClient(trello_key, trello_token, trello_board)
                for card in approved_cards:
                    results.append(trello.create_card(card, list_id=trello_list))
            else:
                # Mock send
                for card in approved_cards:
                    results.append(mock_create_card(card))

            successes = [r for r in results if r.get("success")]
            failures  = [r for r in results if not r.get("success")]

            if successes:
                mock_note = " (mock)" if not trello_configured else ""
                st.success(f"✅ {len(successes)} card(s) created in Trello{mock_note}!")
                for r in successes:
                    if r.get("url") and "mock" not in r.get("url", ""):
                        st.markdown(f"- [{r['url']}]({r['url']})")
            if failures:
                st.error(f"❌ {len(failures)} card(s) failed.")
                for r in failures:
                    st.caption(r.get("error", "Unknown error"))

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

with tab_eval:
    st.title("📊 Evaluation: Agentic Pipeline vs. Baseline")
    st.markdown(
        "The **baseline** is a single LLM prompt that summarizes reviews and lists top issues — "
        "similar to what a product manager might do manually in 10 minutes. "
        "The **agentic pipeline** runs 8 structured steps with scoring, clustering, and human review. "
        "Use the rubric below to score both outputs."
    )

    col_base, col_agent = st.columns(2)

    with col_base:
        st.subheader("Baseline Output")
        st.caption("Single prompt → summary + issue list")
        if st.button("Run Baseline"):
            df = st.session_state.df_raw if st.session_state.df_raw is not None else pipeline.generate_sample_data()
            client = pipeline.get_client(None if mock_mode else anthropic_key)
            st.session_state.baseline_out = pipeline.run_baseline(df, product_goal, client=client, mock=mock_mode)
        if st.session_state.baseline_out:
            st.markdown(st.session_state.baseline_out)
        else:
            st.info("Click **Run Baseline** to generate the comparison output.")

    with col_agent:
        st.subheader("Agentic Pipeline Output")
        st.caption("8-step orchestrated pipeline → scored themes + backlog cards")
        if st.session_state.brief:
            st.markdown(st.session_state.brief)
            n_cards = len(st.session_state.cards or [])
            st.caption(f"Plus {n_cards} backlog card(s) — see the Workflow tab.")
        else:
            st.info("Run the pipeline first on the **Workflow** tab.")

    st.divider()

    # Rubric
    st.subheader("Evaluation Rubric")
    st.caption(
        "Score each output 1–5 on every dimension. "
        "A score of 5 = fully meets the criterion; 1 = does not meet it at all."
    )
    st.dataframe(pd.DataFrame(ev.RUBRIC), use_container_width=True, height=280)

    st.divider()

    # Test set
    st.subheader("Classification Test Set")
    st.caption(
        "10 held-out reviews with human-assigned ground-truth labels. "
        "In live mode, run the classifier on these reviews and compare against ground truth."
    )
    test_df = pd.DataFrame(ev.TEST_SET)
    st.dataframe(test_df, use_container_width=True, height=300)

    st.divider()

    # Blank scorecard for human evaluators
    st.subheader("Your Scorecard")
    st.caption("Fill this in after reviewing both outputs above.")
    scorecard_df = pd.DataFrame(ev.blank_rubric_scorecard())
    st.dataframe(scorecard_df, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════

with tab_about:
    st.title("ℹ️ About Review2Roadmap")
    st.markdown("""
**Review2Roadmap** is a final project for the JHU Carey Generative AI course (Spring 2026).

---

### What it does
Transforms raw electricity customer survey reviews into prioritized, Trello-ready product
backlog cards using a multi-step agentic pipeline powered by Claude.

---

### Design choices (one user, one workflow, one deliverable)

| Dimension | Choice |
|-----------|--------|
| **User** | Product manager / product experience analyst |
| **Workflow** | Raw feedback CSV → prioritized backlog cards |
| **Deliverable** | Insights brief + approved backlog cards |
| **Baseline** | Single LLM prompt: summarize + list issues |

---

### Course concepts demonstrated

| Concept | Where it appears |
|---------|-----------------|
| Multi-step orchestration | 8-step pipeline in `agent_pipeline.py` |
| Tool / function-style steps | Each step is a pure function with defined inputs/outputs |
| Structured outputs | JSON classification schema, card schema |
| Human-in-the-loop | Card approval gate before any Trello write (Step 9) |
| Evaluation design | Rubric + 10-review test set in `evaluation.py` |
| Governance | Validation, deduplication, and review gate before external action |

---

### Repository layout

```
app.py              — Streamlit UI
agent_pipeline.py   — 8-step pipeline (validate → cards)
prompts.py          — All LLM prompts
scoring.py          — Opportunity scoring formula
trello_client.py    — Trello REST API wrapper
evaluation.py       — Test set + evaluation rubric
data/               — Place your CSV here (no PII)
outputs/            — Downloaded reports land here
.streamlit/         — secrets.toml.example for credentials
```

---

### Data & privacy
All data in this project is **synthetic**. No real customer PII is used or committed.
API keys are loaded from `.streamlit/secrets.toml` (git-ignored).
    """)
