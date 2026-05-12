"""
app.py
Review2Roadmap: Agentic Customer Feedback Triage
JHU Carey Generative AI Final Project — Spring 2026

Run with:  streamlit run app.py
"""

import os
import pandas as pd
import streamlit as st

import agent_pipeline as pipeline
import evaluation as ev
from trello_client import TrelloClient, is_configured as trello_ready, mock_create_card

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Review2Roadmap",
    page_icon="🗺️",
    layout="wide",
)

SAMPLE_DATA_PATH = os.path.join("data", "fake_electricity_customer_reviews_500.csv")

# Pull required columns directly from the pipeline so there's one source of truth.
REQUIRED_COLUMNS = pipeline.REQUIRED_COLUMNS

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Settings")

    product_goal = st.selectbox(
        "Product goal",
        [
            "Reduce churn",
            "Improve customer satisfaction",
            "Reduce support tickets",
            "Identify urgent bugs or service failures",
            "Find quick wins for the product roadmap",
        ],
    )

    max_cards = st.number_input(
        "Maximum backlog cards to generate",
        min_value=1,
        max_value=10,
        value=5,
        step=1,
    )

    mock_mode = st.checkbox("Use mock mode instead of live LLM", value=True)

    if not mock_mode:
        anthropic_key = st.text_input(
            "Anthropic API key",
            type="password",
            value=st.secrets.get("ANTHROPIC_API_KEY", ""),
        )
        # Make the key available to pipeline._llm_call() via the environment
        if anthropic_key:
            os.environ["ANTHROPIC_API_KEY"] = anthropic_key
    else:
        anthropic_key = ""

    st.divider()
    st.caption(
        "⚠️ **Governance notice:** Outputs are AI-generated drafts and require "
        "human review before roadmap decisions or Trello card creation."
    )

# ── Session state defaults ────────────────────────────────────────────────────

_STATE_KEYS = [
    "df_raw", "df_clean", "classified_reviews", "themes",
    "scored_themes", "brief", "cards", "pipeline_done",
    "baseline_out", "dup_result", "validation",
]
for k in _STATE_KEYS:
    if k not in st.session_state:
        st.session_state[k] = None

if "approved" not in st.session_state:
    st.session_state.approved = set()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _reset_pipeline_state():
    for k in ["df_clean", "classified_reviews", "themes", "scored_themes",
              "brief", "cards", "pipeline_done", "dup_result", "validation"]:
        st.session_state[k] = None
    st.session_state.approved = set()


def _quick_duplicate_count(df: pd.DataFrame) -> int:
    """Count exact duplicate written_explanation values as a fast preview heuristic."""
    col = "written_explanation" if "written_explanation" in df.columns else None
    if col is None:
        return 0
    return int(df[col].duplicated().sum())


def _column_status(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "Column": col,
            "Required": "✅ Yes",
            "Present": "✅ Found" if col in df.columns else "❌ Missing",
        }
        for col in REQUIRED_COLUMNS
    ])


# ── Header ────────────────────────────────────────────────────────────────────

st.title("🗺️ Review2Roadmap: Agentic Customer Feedback Triage")
st.markdown(
    "This app turns raw electricity customer survey reviews into prioritized product "
    "backlog cards using a multi-step GenAI workflow."
)
st.divider()

# ── Data loading ──────────────────────────────────────────────────────────────

col_upload, col_sample = st.columns([3, 1])

with col_upload:
    uploaded = st.file_uploader("Upload a survey CSV", type=["csv"])

with col_sample:
    st.write("")
    st.write("")
    if st.button("📂 Load Sample Dataset"):
        if os.path.exists(SAMPLE_DATA_PATH):
            try:
                st.session_state.df_raw = pd.read_csv(SAMPLE_DATA_PATH)
                _reset_pipeline_state()
                st.success(f"Loaded {len(st.session_state.df_raw):,} rows from sample dataset.")
            except Exception as exc:
                st.error(f"Could not read sample file: {exc}")
        else:
            st.session_state.df_raw = pipeline.generate_sample_data()
            _reset_pipeline_state()
            st.info(
                "Sample file not found at `data/fake_electricity_customer_reviews_500.csv`. "
                "Loaded 30-row built-in dataset instead."
            )

if uploaded is not None:
    try:
        st.session_state.df_raw = pd.read_csv(uploaded)
        _reset_pipeline_state()
        st.success(f"Uploaded {len(st.session_state.df_raw):,} rows.")
    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")

# ── Data preview ──────────────────────────────────────────────────────────────

if st.session_state.df_raw is not None:
    df = st.session_state.df_raw

    st.subheader("Data Preview")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total rows", f"{len(df):,}")
    m2.metric("Columns", len(df.columns))
    m3.metric(
        "Required columns present",
        f"{sum(c in df.columns for c in REQUIRED_COLUMNS)}/{len(REQUIRED_COLUMNS)}",
    )
    m4.metric("Exact duplicates", _quick_duplicate_count(df))

    with st.expander("First 10 rows", expanded=True):
        st.dataframe(df.head(10), use_container_width=True)

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("**Column names**")
        st.code(", ".join(df.columns.tolist()))
    with col_right:
        st.markdown("**Required column status**")
        st.dataframe(_column_status(df), use_container_width=True, hide_index=True)

    st.divider()

    # ── Run Agent Analysis ────────────────────────────────────────────────────

    if st.button("▶ Run Agent Analysis", type="primary"):
        prog = st.progress(0)
        status = st.empty()

        steps = [
            "Validating CSV",
            "Cleaning reviews",
            "Detecting duplicates",
            "Classifying reviews",
            "Clustering themes",
            "Scoring product opportunities",
            "Generating product insights brief",
            "Creating backlog cards",
        ]
        total = len(steps)

        def advance(i: int):
            prog.progress((i + 1) / total, text=f"Step {i + 1}/{total} — {steps[i]}")

        # Step 1 — Validate
        advance(0)
        val = pipeline.validate_reviews_csv(df)
        st.session_state.validation = val
        if val["validation_status"] == "failed":
            prog.empty()
            status.error(
                "Validation failed. Missing columns: "
                + ", ".join(val["missing_columns"])
            )
            st.stop()
        for w in val.get("warnings", []):
            st.warning(w)

        # Step 2 — Clean
        advance(1)
        clean_result = pipeline.clean_reviews(df)
        df_clean = clean_result["cleaned_df"]
        st.session_state.df_clean = df_clean

        # Step 3 — Deduplicate
        advance(2)
        dup_result = pipeline.detect_duplicates(df_clean)
        st.session_state.dup_result = dup_result
        df_deduped = dup_result["deduped_df"]
        dup_summary = dup_result["duplicate_summary"]

        # Step 4 — Classify (returns list[dict], not a DataFrame)
        advance(3)
        classified = pipeline.classify_reviews(df_deduped, product_goal, use_mock=mock_mode)
        st.session_state.classified_reviews = classified

        # Step 5 — Cluster (takes list[dict])
        advance(4)
        themes = pipeline.cluster_themes(classified)
        st.session_state.themes = themes

        # Step 6 — Score (now requires product_goal for alignment bonus)
        advance(5)
        scored = pipeline.score_opportunities(themes, product_goal)
        st.session_state.scored_themes = scored

        # Step 7 — Brief (now requires dup_summary)
        advance(6)
        brief = pipeline.generate_insights_brief(
            scored, dup_summary, product_goal, use_mock=mock_mode
        )
        st.session_state.brief = brief

        # Step 8 — Cards (max_cards replaces the manual slice)
        advance(7)
        cards = pipeline.generate_backlog_cards(scored, max_cards=int(max_cards))
        st.session_state.cards = cards
        st.session_state.approved = set()

        prog.progress(1.0, text="✅ Analysis complete!")
        st.session_state.pipeline_done = True

# ── Results tabs ──────────────────────────────────────────────────────────────

if st.session_state.pipeline_done:

    (
        tab_overview,
        tab_clusters,
        tab_opps,
        tab_brief,
        tab_trello,
        tab_baseline,
        tab_eval,
    ) = st.tabs([
        "Overview",
        "Theme Clusters",
        "Prioritized Opportunities",
        "Product Insights Brief",
        "Trello Cards",
        "Baseline Comparison",
        "Evaluation Preview",
    ])

    # ── Overview ──────────────────────────────────────────────────────────────

    with tab_overview:
        st.subheader("Pipeline Summary")
        val        = st.session_state.validation or {}
        dup_result = st.session_state.dup_result or {}

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Reviews loaded",     f"{val.get('row_count', 0):,}")
        c2.metric("After cleaning",     len(st.session_state.df_clean))
        c3.metric("Duplicates flagged", dup_result.get("duplicate_count", 0))
        c4.metric("Themes found",       len(st.session_state.themes or []))
        c5.metric("Cards generated",    len(st.session_state.cards or []))

        st.divider()
        st.subheader("Classified Reviews")
        st.caption("One row per survey response after deduplication and classification.")

        # classified_reviews is a list[dict] — convert to DataFrame for display
        clf_df = pd.DataFrame(st.session_state.classified_reviews or [])
        show_cols = [
            "survey_response_id", "written_explanation",
            "issue_category", "sentiment", "severity_score", "opportunity_type",
        ]
        available = [c for c in show_cols if c in clf_df.columns]
        st.dataframe(clf_df[available], use_container_width=True, height=300)

    # ── Theme Clusters ────────────────────────────────────────────────────────

    with tab_clusters:
        st.subheader("Theme Clusters")
        st.caption("Each theme groups reviews with a common underlying issue or opportunity.")

        for i, t in enumerate(st.session_state.themes or [], 1):
            # Count sentiment breakdown from the theme's reviews
            sentiment_counts: dict[str, int] = {}
            for r in t.get("reviews", []):
                s = r.get("sentiment", "unknown")
                sentiment_counts[s] = sentiment_counts.get(s, 0) + 1

            label = f"{i}. {t['name']} — {t['review_count']} reviews"
            with st.expander(label, expanded=(i <= 3)):
                st.markdown(f"**Key pain point:** {t.get('key_pain_point', '—')}")

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Issue Category",    t.get("issue_category", "—").replace("_", " ").title())
                c2.metric("Dominant Severity", t.get("dominant_severity", "—").title())
                c3.metric("Avg NPS Score",     t.get("avg_nps_score", "—"))
                c4.metric("Detractors",        t.get("detractor_count", 0))

                if sentiment_counts:
                    st.markdown(
                        "**Sentiment split:** "
                        + "  ·  ".join(f"{k}: {v}" for k, v in sentiment_counts.items())
                    )

                quotes = t.get("sample_evidence_quotes", [])
                if quotes:
                    st.markdown("**Sample quotes:**")
                    for q in quotes:
                        st.markdown(f"> {q}")

                plans = t.get("affected_plans", [])
                if plans:
                    st.markdown("**Affected plans:** " + ", ".join(f"`{p}`" for p in plans))

                st.markdown(
                    f"**Confidence:** `{t.get('confidence_level', '—')}`"
                )

    # ── Prioritized Opportunities ─────────────────────────────────────────────

    with tab_opps:
        st.subheader("Prioritized Opportunities")
        st.caption(
            "Weighted opportunity score (1–10): "
            "Frequency 25% · Severity 30% · Business Impact 20% · Confidence 15% · NPS Risk 10%"
        )

        opp_rows = [
            {
                "Priority":     t.get("priority", "?"),
                "Theme":        t["name"],
                "Reviews":      t["review_count"],
                "Avg NPS":      t.get("avg_nps_score", "—"),
                "Severity":     t.get("dominant_severity", "").title(),
                "Score":        t.get("opportunity_score", 0),
                "Frequency":    t.get("frequency_score", 0),
                "Severity Sc.": t.get("severity_score", 0),
                "Biz Impact":   t.get("business_impact_score", 0),
                "Confidence":   t.get("confidence_score", 0),
                "NPS Risk":     t.get("nps_risk_score", 0),
                "Goal Aligned": "✅" if t.get("goal_aligned") else "—",
            }
            for t in (st.session_state.scored_themes or [])
        ]
        st.dataframe(
            pd.DataFrame(opp_rows).sort_values("Score", ascending=False),
            use_container_width=True,
        )

    # ── Product Insights Brief ────────────────────────────────────────────────

    with tab_brief:
        st.subheader("Product Insights Brief")
        st.caption(f"Goal: {product_goal}")
        st.markdown(st.session_state.brief)

    # ── Trello Cards ──────────────────────────────────────────────────────────

    with tab_trello:
        st.subheader("Trello-Ready Backlog Cards")
        st.caption(
            "Review each card. Check **Approve** on the cards you want to send to Trello. "
            "This human review gate is a governance feature of the agentic workflow."
        )

        cards = st.session_state.cards or []
        for i, card in enumerate(cards):
            icon     = "✅" if i in st.session_state.approved else "⬜"
            priority = card.get("priority", "?")

            with st.expander(f"{icon} [{priority}] {card['title']}", expanded=(i == 0)):
                left, right = st.columns([5, 1])

                with left:
                    st.markdown(f"**User Story:** {card.get('user_story', '—')}")
                    st.markdown(f"**Description:** {card.get('description', '—')}")

                    criteria = card.get("acceptance_criteria", [])
                    if criteria:
                        st.markdown("**Acceptance Criteria:**")
                        for c in criteria:
                            st.markdown(f"- {c}")

                    quotes = card.get("evidence_quotes", [])
                    if quotes:
                        st.markdown("**Evidence Quotes:**")
                        for q in quotes:
                            st.markdown(f"> {q}")

                    labels = card.get("labels", [])
                    if labels:
                        st.markdown("**Labels:** " + "  ·  ".join(f"`{l}`" for l in labels))

                    st.markdown(
                        f"**Effort:** `{card.get('estimated_effort', '?')}`  "
                        f"**Score:** {card.get('opportunity_score', '—')}  "
                        f"**Owner:** {card.get('recommended_owner_area', '—')}"
                    )

                    # Governance reminder embedded in each card
                    st.info(card.get("human_review_notes", ""))

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
        st.subheader("Send to Trello")

        with st.expander("Trello credentials (optional)", expanded=False):
            trello_key   = st.text_input("API Key",  type="password", key="tk",
                                         value=st.secrets.get("TRELLO_API_KEY", ""))
            trello_token = st.text_input("Token",    type="password", key="tt",
                                         value=st.secrets.get("TRELLO_TOKEN", ""))
            trello_board = st.text_input("Board ID", key="tb",
                                         value=st.secrets.get("TRELLO_BOARD_ID", ""))
            trello_list  = st.text_input("List ID (Backlog)", key="tl",
                                         value=st.secrets.get("TRELLO_LIST_ID", ""))

        # Widget keys are mirrored in session_state — read from there so the
        # values are available outside the expander after it collapses.
        trello_configured = trello_ready(
            st.session_state.get("tk", ""),
            st.session_state.get("tt", ""),
            st.session_state.get("tb", ""),
            st.session_state.get("tl", ""),
        )

        n_approved = len(st.session_state.approved)
        if n_approved == 0:
            st.caption("Approve at least one card above to enable sending.")

        if st.button(f"📤 Send {n_approved} Approved Card(s) to Trello", disabled=(n_approved == 0)):
            approved_cards = [cards[i] for i in sorted(st.session_state.approved)]
            results = []

            if trello_configured:
                trello = TrelloClient(
                    st.session_state.get("tk", ""),
                    st.session_state.get("tt", ""),
                    st.session_state.get("tb", ""),
                )
                for card in approved_cards:
                    results.append(trello.create_card(card, list_id=st.session_state.get("tl", "")))
            else:
                for card in approved_cards:
                    results.append(mock_create_card(card))

            successes = [r for r in results if r.get("success")]
            failures  = [r for r in results if not r.get("success")]

            if successes:
                label = "" if trello_configured else " (mock — Trello not configured)"
                st.success(f"✅ {len(successes)} card(s) sent{label}!")
                for r in successes:
                    if r.get("url") and "mock" not in r["url"]:
                        st.markdown(f"- {r['url']}")
            if failures:
                st.error(f"❌ {len(failures)} card(s) failed.")
                for r in failures:
                    st.caption(r.get("error", "Unknown error"))

    # ── Baseline Comparison ───────────────────────────────────────────────────

    with tab_baseline:
        st.subheader("Baseline vs. Agentic Pipeline")
        st.markdown(
            "The **baseline** is a single LLM prompt that summarizes reviews and lists top issues — "
            "the starting point the agentic pipeline improves on."
        )

        col_b, col_a = st.columns(2)

        with col_b:
            st.markdown("#### Baseline")
            st.caption("One prompt → summary + bullet list")
            if st.button("Run Baseline"):
                df_for_baseline = (
                    st.session_state.df_raw
                    if st.session_state.df_raw is not None
                    else pipeline.generate_sample_data()
                )
                st.session_state.baseline_out = pipeline.run_baseline(
                    df_for_baseline, product_goal, use_mock=mock_mode
                )
            if st.session_state.baseline_out:
                st.markdown(st.session_state.baseline_out)
            else:
                st.info("Click **Run Baseline** to generate the comparison output.")

        with col_a:
            st.markdown("#### Agentic Pipeline")
            st.caption("8 steps → classified themes, scores, brief, cards")
            if st.session_state.brief:
                st.markdown(st.session_state.brief)
                st.caption(
                    f"{len(st.session_state.cards or [])} backlog cards generated — see Trello Cards tab."
                )
            else:
                st.info("Run the agent analysis first.")

    # ── Evaluation Preview ────────────────────────────────────────────────────

    with tab_eval:
        st.subheader("Evaluation Preview")

        st.markdown("#### Evaluation Rubric")
        st.caption("Score each output 1–5 on every dimension.")
        st.dataframe(pd.DataFrame(ev.RUBRIC), use_container_width=True, height=270)

        st.divider()

        st.markdown("#### Classification Test Set")
        st.caption(
            "10 held-out synthetic reviews with human-assigned ground-truth labels. "
            "In live mode, run the classifier on these to measure accuracy."
        )
        st.dataframe(pd.DataFrame(ev.TEST_SET), use_container_width=True, height=300)

        st.divider()

        st.markdown("#### Your Scorecard")
        st.caption("Fill this in after reviewing both outputs in the Baseline Comparison tab.")
        st.dataframe(pd.DataFrame(ev.blank_rubric_scorecard()), use_container_width=True)
