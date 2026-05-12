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
from trello_client import (
    TrelloClient,
    trello_config_available,
    get_default_list_id,
    format_card_description,
)

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Review2Roadmap",
    page_icon="🗺️",
    layout="wide",
)

SAMPLE_DATA_PATH = os.path.join("data", "fake_electricity_customer_reviews_500.csv")

# Pull required columns directly from the pipeline so there's one source of truth.
REQUIRED_COLUMNS = pipeline.REQUIRED_COLUMNS


# ── Credential resolution (startup, never displayed) ─────────────────────────

def _secret_configured(key: str) -> bool:
    """Return True if a secret is set in st.secrets or the environment. Never returns the value."""
    try:
        if st.secrets.get(key, ""):
            return True
    except Exception:
        pass
    return bool(os.environ.get(key, ""))


_ANTHROPIC_CONFIGURED = _secret_configured("ANTHROPIC_API_KEY")
_TRELLO_CONFIGURED    = trello_config_available()


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

    mock_mode = st.checkbox(
        "Use mock mode instead of live LLM",
        value=not _ANTHROPIC_CONFIGURED,
        disabled=not _ANTHROPIC_CONFIGURED,
        help=(
            "Mock mode uses keyword heuristics — no API key required."
            if not _ANTHROPIC_CONFIGURED
            else "Uncheck to call the live Anthropic API."
        ),
    )

    st.divider()

    # ── Integration status (configured / not configured only — no key values) ─
    st.markdown("**Integration status**")

    if _ANTHROPIC_CONFIGURED:
        st.success("Anthropic API: ✅ Configured")
    else:
        st.warning(
            "Anthropic API not configured — mock mode will be used. "
            "Add `ANTHROPIC_API_KEY` to `.streamlit/secrets.toml` to enable live analysis."
        )

    if _TRELLO_CONFIGURED:
        st.success("Trello: ✅ Configured")
    else:
        st.info(
            "Trello is not configured. Add `TRELLO_API_KEY`, `TRELLO_TOKEN`, and "
            "`TRELLO_LIST_ID` to `.streamlit/secrets.toml` to enable card creation."
        )

    st.divider()

    # ── Governance notice (persistent, always visible) ────────────────────────
    st.warning(
        "**AI-generated drafts only.**\n\n"
        "All outputs require human review before being used as the basis "
        "for roadmap decisions, stakeholder communications, or Trello card creation."
    )

    st.divider()

    # ── Data privacy notice ───────────────────────────────────────────────────
    st.info(
        "**Data privacy reminder**\n\n"
        "Use only **synthetic or publicly available** data with this tool. "
        "Do not upload private customer data, personally identifiable information (PII), "
        "or confidential company data."
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

st.warning(
    "⚠️ **All outputs are AI-generated draft insights.** "
    "They should not be the sole basis for roadmap decisions, stakeholder commitments, "
    "or product actions. A product manager must review and validate every output "
    "before acting on it."
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

    dup_count_preview = _quick_duplicate_count(df)
    cols_present      = sum(c in df.columns for c in REQUIRED_COLUMNS)
    missing_cols      = [c for c in REQUIRED_COLUMNS if c not in df.columns]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total rows", f"{len(df):,}")
    m2.metric("Columns", len(df.columns))
    m3.metric(
        "Required columns present",
        f"{cols_present}/{len(REQUIRED_COLUMNS)}",
        delta=None if cols_present == len(REQUIRED_COLUMNS) else f"−{len(missing_cols)} missing",
        delta_color="off" if cols_present == len(REQUIRED_COLUMNS) else "inverse",
    )
    m4.metric("Exact duplicates", dup_count_preview)

    # ── Pre-analysis data quality warnings ────────────────────────────────────

    if missing_cols:
        st.error(
            f"**Missing required columns:** {', '.join(f'`{c}`' for c in missing_cols)}. "
            "The pipeline cannot run until these columns are present. "
            "See `data/README.md` for the expected schema."
        )

    if len(df) < 10:
        st.warning(
            f"⚠️ Only **{len(df)} rows** detected. "
            "Theme clustering works best with 10 or more unique responses. "
            "Results from small datasets may not be representative."
        )

    if "written_explanation" in df.columns:
        blank_pct = (
            df["written_explanation"]
            .astype(str)
            .str.strip()
            .eq("")
            .mean()
        )
        if blank_pct > 0.3:
            st.warning(
                f"⚠️ **{blank_pct:.0%} of written explanations are blank.** "
                "Rows with no text will be removed before analysis. "
                "A high blank rate may result in themes with limited evidence."
            )

    if dup_count_preview > 0:
        dup_pct = dup_count_preview / len(df)
        level   = "error" if dup_pct > 0.3 else "warning"
        msg     = (
            f"⚠️ **{dup_count_preview} exact duplicate responses detected** "
            f"({dup_pct:.0%} of rows). "
            "Duplicates will be excluded before analysis. "
        )
        if dup_pct > 0.3:
            msg += (
                "A duplication rate above 30% may indicate a data collection issue — "
                "review your source data before running the pipeline."
            )
        getattr(st, level)(msg)

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

        # Warn when a large proportion of responses were duplicates
        dup_pct = dup_result["duplicate_count"] / max(len(df_clean), 1)
        if dup_pct > 0.2:
            st.warning(
                f"⚠️ **{dup_result['duplicate_count']} duplicate responses removed** "
                f"({dup_pct:.0%} of cleaned data). "
                "Insights are based on the remaining unique responses only."
            )

        # Warn when usable dataset is very small after deduplication
        if len(df_deduped) < 10:
            st.warning(
                f"⚠️ Only **{len(df_deduped)} usable responses** remain after "
                "cleaning and deduplication. Results may not be representative. "
                "Consider uploading a larger dataset before making decisions."
            )

        # Collect pipeline-level warnings (auto-mock fallbacks, batch failures)
        pipeline_warnings: list[str] = []

        # Step 4 — Classify (returns list[dict], not a DataFrame)
        advance(3)
        classified = pipeline.classify_reviews(
            df_deduped, product_goal, use_mock=mock_mode, warnings=pipeline_warnings
        )
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
            scored, dup_summary, product_goal,
            use_mock=mock_mode, warnings=pipeline_warnings,
        )
        st.session_state.brief = brief

        # Step 8 — Cards (max_cards replaces the manual slice)
        advance(7)
        cards = pipeline.generate_backlog_cards(scored, max_cards=int(max_cards))
        st.session_state.cards = cards
        st.session_state.approved = set()

        prog.progress(1.0, text="✅ Analysis complete!")
        st.session_state.pipeline_done = True

        # Surface any non-fatal pipeline warnings (e.g. auto-mock fallback)
        for pw in pipeline_warnings:
            st.warning(f"⚠️ {pw}")

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
        st.caption("🤖 *AI-generated draft insights — review all outputs before acting on them.*")
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
        st.caption(
            "One row per survey response after deduplication and classification. "
            "Classifications are AI-generated — verify individual rows before using as evidence."
        )

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
        st.caption(
            "Each theme groups reviews with a common underlying issue or opportunity. "
            "🤖 *Theme names and groupings are AI-generated — verify before presenting externally.*"
        )

        for i, t in enumerate(st.session_state.themes or [], 1):
            # Count sentiment breakdown from the theme's reviews
            sentiment_counts: dict[str, int] = {}
            for r in t.get("reviews", []):
                s = r.get("sentiment", "unknown")
                sentiment_counts[s] = sentiment_counts.get(s, 0) + 1

            conf        = t.get("confidence_level", "medium")
            review_cnt  = t.get("review_count", 0)
            needs_review = conf == "low" or review_cnt < 3

            label = f"{i}. {t['name']} — {review_cnt} reviews"
            with st.expander(label, expanded=(i <= 3)):

                if needs_review:
                    reasons = []
                    if conf == "low":
                        reasons.append("low model confidence")
                    if review_cnt < 3:
                        reasons.append(f"only {review_cnt} supporting review(s)")
                    st.warning(
                        f"⚠️ **Needs Human Validation** — {', '.join(reasons)}. "
                        "Treat this theme as a weak signal until more evidence is available."
                    )

                st.markdown(f"**Key pain point:** {t.get('key_pain_point', '—')}")

                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Issue Category",    t.get("issue_category", "—").replace("_", " ").title())
                c2.metric("Dominant Severity", t.get("dominant_severity", "—").title())
                c3.metric("Avg NPS Score",     t.get("avg_nps_score", "—"))
                c4.metric("Detractors",        t.get("detractor_count", 0))
                c5.metric("Confidence",        conf.title())

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

    # ── Prioritized Opportunities ─────────────────────────────────────────────

    with tab_opps:
        st.subheader("Prioritized Opportunities")
        st.caption(
            "Weighted opportunity score (0–100): weights are goal-specific "
            "(Frequency · Severity · NPS Risk · Business Impact · Confidence)"
        )

        opp_rows = [
            {
                "Priority":        t.get("priority", "?"),
                "Theme":           t["name"],
                "Reviews":         t["review_count"],
                "Avg NPS":         t.get("avg_nps_score", "—"),
                "Severity":        t.get("dominant_severity", "").title(),
                "Score":           t.get("opportunity_score", 0),
                "Confidence":      t.get("confidence_level", "—").title(),
                "Frequency":       t.get("frequency_score", 0),
                "Severity Sc.":    t.get("severity_score", 0),
                "Biz Impact":      t.get("business_impact_score", 0),
                "Conf. Score":     t.get("confidence_score", 0),
                "NPS Risk":        t.get("nps_risk_score", 0),
                "Goal Aligned":    "✅" if t.get("goal_aligned") else "—",
                "Needs Review":    "⚠️ Yes" if t.get("confidence_level") == "low" or t.get("review_count", 0) < 3 else "—",
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
        st.warning(
            "🤖 **AI-generated draft.** This brief was produced by an automated pipeline. "
            "Verify key claims against the classified reviews before sharing with stakeholders."
        )
        st.markdown(st.session_state.brief)

    # ── Trello Cards ──────────────────────────────────────────────────────────

    with tab_trello:
        st.subheader("Trello-Ready Backlog Cards")
        st.caption(
            "Review each card. Check **Approve** on the cards you want to send to Trello. "
            "This human review gate is a governance feature of the agentic workflow."
        )

        cards = st.session_state.cards or []

        # ── Card review with individual approval checkboxes ───────────────────
        for i, card in enumerate(cards):
            icon     = "✅" if i in st.session_state.approved else "⬜"
            priority = card.get("priority", "?")

            with st.expander(f"{icon} [{priority}] {card['title']}", expanded=(i == 0)):
                left, right = st.columns([5, 1])

                with left:
                    st.markdown(f"*{card.get('user_story', '—')}*")
                    st.divider()

                    st.markdown("**Problem**")
                    st.markdown(card.get("problem", card.get("description", "—")))

                    quotes = card.get("evidence_quotes", [])
                    if quotes:
                        st.markdown("**Evidence**")
                        for q in quotes:
                            st.markdown(f"> {q}")

                    why = card.get("why_it_matters", "")
                    if why:
                        st.markdown("**Why it matters**")
                        st.markdown(why)

                    action = card.get("recommended_action", "")
                    if action:
                        st.markdown("**Recommended action**")
                        st.markdown(action)

                    sc1, sc2, sc3 = st.columns(3)
                    sc1.metric("Priority score", f"{card.get('opportunity_score', '—')}/100")
                    sc2.metric("Confidence", card.get("confidence", "—").title())
                    sc3.metric("Effort", card.get("estimated_effort", "—"))

                    bd = card.get("score_breakdown_str", "")
                    if bd:
                        st.caption(f"Score breakdown: {bd}")

                    criteria = card.get("acceptance_criteria", [])
                    if criteria:
                        st.markdown("**Acceptance criteria**")
                        for c in criteria:
                            st.markdown(f"- {c}")

                    labels = card.get("labels", [])
                    if labels:
                        st.markdown("**Labels:** " + "  ·  ".join(f"`{lbl}`" for lbl in labels))

                    st.markdown(
                        f"**Owner:** {card.get('recommended_owner_area', '—')}"
                    )

                    st.info(f"⚠️ **Human review notes:** {card.get('human_review_notes', '')}")

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

        n_approved = len(st.session_state.approved)

        st.divider()

        # ── Path A: Trello not configured → setup instructions + CSV export ───
        if not trello_config_available():
            st.subheader("Trello Not Configured")
            st.info(
                "To send cards directly to Trello, add these credentials to "
                "`.streamlit/secrets.toml`:\n\n"
                "```toml\n"
                'TRELLO_API_KEY = "your-key"\n'
                'TRELLO_TOKEN   = "your-token"\n'
                'TRELLO_LIST_ID = "your-list-id"\n'
                "```\n\n"
                "Get your key and token at **https://trello.com/app-key**. "
                "Use the list ID helper below to find the correct list ID."
            )

            st.markdown("**In the meantime, export your approved cards as CSV:**")

            if n_approved == 0:
                st.warning(
                    "⚠️ No cards approved. "
                    "Check **Approve** on at least one card above to enable CSV export."
                )
            else:
                approved_cards = [cards[i] for i in sorted(st.session_state.approved)]
                export_rows = [
                    {
                        "title":               c.get("title", ""),
                        "priority":            c.get("priority", ""),
                        "user_story":          c.get("user_story", ""),
                        "description":         c.get("description", ""),
                        "acceptance_criteria": " | ".join(c.get("acceptance_criteria", [])),
                        "evidence_quotes":     " | ".join(c.get("evidence_quotes", [])),
                        "labels":              ", ".join(c.get("labels", [])),
                        "estimated_effort":    c.get("estimated_effort", ""),
                        "opportunity_score":   c.get("opportunity_score", ""),
                        "recommended_owner":   c.get("recommended_owner_area", ""),
                    }
                    for c in approved_cards
                ]
                st.download_button(
                    label=f"⬇ Download {n_approved} Approved Card(s) as CSV",
                    data=pd.DataFrame(export_rows).to_csv(index=False),
                    file_name="backlog_cards.csv",
                    mime="text/csv",
                )

        # ── Path B: Trello configured → helper + send ─────────────────────────
        else:
            # Helper: fetch boards and lists to find the correct list ID
            with st.expander(
                "🔍 Find your Trello list ID",
                expanded=(not get_default_list_id()),
            ):
                st.caption(
                    "Use this to look up the list ID you want cards sent to, "
                    "then add it as `TRELLO_LIST_ID` in your secrets."
                )

                if st.button("Fetch my Trello boards"):
                    client = TrelloClient()
                    boards = client.get_boards()
                    if boards and "error" not in boards[0]:
                        st.session_state["_trello_boards"] = {
                            b["name"]: b["id"] for b in boards
                        }
                    else:
                        st.error(boards[0].get("error", "Could not fetch boards."))

                if "_trello_boards" in st.session_state:
                    board_names = list(st.session_state["_trello_boards"].keys())
                    selected_board = st.selectbox(
                        "Select a board", board_names, key="_trello_board_sel"
                    )
                    selected_board_id = st.session_state["_trello_boards"][selected_board]

                    if st.button("Fetch lists for this board"):
                        client = TrelloClient()
                        lists = client.get_lists(selected_board_id)
                        if lists and "error" not in lists[0]:
                            st.session_state["_trello_lists"] = lists
                        else:
                            st.error(lists[0].get("error", "Could not fetch lists."))

                    if "_trello_lists" in st.session_state:
                        st.markdown("**Lists on this board** — copy the ID of the list you want:")
                        for lst in st.session_state["_trello_lists"]:
                            col_name, col_id = st.columns([2, 3])
                            col_name.markdown(f"**{lst['name']}**")
                            col_id.code(lst["id"])
                        st.caption(
                            "Add the chosen ID as `TRELLO_LIST_ID` in `.streamlit/secrets.toml`, "
                            "then reload the app."
                        )

            # Send approved cards to Trello
            st.subheader("Send to Trello")

            list_id = get_default_list_id()
            if not list_id:
                st.warning(
                    "`TRELLO_LIST_ID` is not set. "
                    "Use the finder above to locate your list ID, then add it to secrets."
                )

            if n_approved == 0:
                st.warning(
                    "⚠️ No cards approved. "
                    "Check **Approve** on at least one card above to enable sending to Trello."
                )
            elif not list_id:
                pass  # already warned above via the list_id check

            send_disabled = n_approved == 0 or not list_id
            if st.button(
                f"📤 Send {n_approved} Approved Card(s) to Trello",
                disabled=send_disabled,
                type="primary",
            ):
                approved_cards = [cards[i] for i in sorted(st.session_state.approved)]
                client = TrelloClient()
                successes: list[dict] = []
                failures:  list[tuple[str, str]] = []

                for card in approved_cards:
                    result = client.create_card(
                        list_id=list_id,
                        name=card.get("title", "Untitled"),
                        description=format_card_description(card),
                        labels=card.get("labels"),
                    )
                    if result.get("success"):
                        successes.append(result)
                    else:
                        failures.append(
                            (card.get("title", "?"), result.get("error", "Unknown error"))
                        )

                if successes:
                    st.success(f"✅ {len(successes)} card(s) created in Trello!")
                    for r in successes:
                        if r.get("url"):
                            st.markdown(f"- [{r['url']}]({r['url']})")
                if failures:
                    st.error(f"❌ {len(failures)} card(s) failed.")
                    for title, err in failures:
                        st.caption(f"**{title}**: {err}")

    # ── Baseline Comparison ───────────────────────────────────────────────────

    with tab_baseline:
        st.subheader("Baseline vs. Agentic Pipeline")
        st.markdown(
            "The **baseline** uses a single prompt — "
            "*'Summarize these customer reviews and list the top issues'* — "
            "with no structure, no classification, and no scoring. "
            "The **agentic pipeline** performs a fuller product triage process by cleaning data, "
            "detecting duplicates, clustering themes, scoring opportunities, generating backlog "
            "cards, and requiring human approval before any Trello action is taken. "
            "The comparison below makes the difference concrete."
        )

        # ── Comparison table ──────────────────────────────────────────────────
        comparison_rows = [
            {
                "Capability":        "Structured output",
                "Baseline":          "❌  Free-text summary only",
                "Agentic Pipeline":  "✅  JSON classification with 8 fields per review",
            },
            {
                "Capability":        "Duplicate handling",
                "Baseline":          "❌  Duplicates counted as separate signals",
                "Agentic Pipeline":  "✅  Exact-match and same-customer detection; excluded before analysis",
            },
            {
                "Capability":        "Theme clustering",
                "Baseline":          "❌  Issues listed as a flat bullet list",
                "Agentic Pipeline":  "✅  Reviews grouped into named opportunity themes with NPS and severity stats",
            },
            {
                "Capability":        "Evidence quotes",
                "Baseline":          "❌  No quotes — summary only",
                "Agentic Pipeline":  "✅  Verbatim quotes from reviews attached to each theme and card",
            },
            {
                "Capability":        "Priority scoring",
                "Baseline":          "❌  No ranking — all issues treated equally",
                "Agentic Pipeline":  "✅  0–100 weighted score across 5 dimensions; goal-specific weights",
            },
            {
                "Capability":        "Product insights brief",
                "Baseline":          "❌  Generic summary; not aligned to a product goal",
                "Agentic Pipeline":  "✅  Structured brief with executive summary, top opportunities, next steps, risks",
            },
            {
                "Capability":        "Trello-ready cards",
                "Baseline":          "❌  None generated",
                "Agentic Pipeline":  "✅  Action-oriented cards with acceptance criteria, effort, and owner area",
            },
            {
                "Capability":        "Human approval before action",
                "Baseline":          "❌  Output goes directly to user with no review gate",
                "Agentic Pipeline":  "✅  PM approves individual cards before any Trello creation",
            },
        ]

        st.dataframe(
            pd.DataFrame(comparison_rows).set_index("Capability"),
            use_container_width=True,
            height=330,
        )

        st.divider()

        # ── Side-by-side outputs ──────────────────────────────────────────────
        col_b, col_a = st.columns(2)

        with col_b:
            st.markdown("#### Baseline Output")
            st.caption("Prompt: *'Summarize these customer reviews and list the top issues.'*")
            if st.button("▶ Run Baseline", key="run_baseline_btn"):
                df_for_baseline = (
                    st.session_state.df_raw
                    if st.session_state.df_raw is not None
                    else pipeline.generate_sample_data()
                )
                st.session_state.baseline_out = pipeline.run_baseline_summary(
                    df_for_baseline, product_goal, use_mock=mock_mode
                )
            if st.session_state.baseline_out:
                st.markdown(st.session_state.baseline_out)
            else:
                st.info("Click **▶ Run Baseline** to generate the baseline output.")

        with col_a:
            st.markdown("#### Agentic Pipeline Output")
            st.caption("8 steps: validate → clean → deduplicate → classify → cluster → score → brief → cards")
            if st.session_state.brief:
                st.markdown(st.session_state.brief)
                n_cards = len(st.session_state.cards or [])
                n_themes = len(st.session_state.themes or [])
                st.caption(
                    f"{n_themes} themes identified · "
                    f"{n_cards} backlog cards generated · "
                    "See **Trello Cards** tab to review and approve."
                )
            else:
                st.info("Run the **▶ Run Agent Analysis** button above to generate the pipeline output.")

    # ── Evaluation Preview ────────────────────────────────────────────────────

    with tab_eval:
        st.subheader("Evaluation Framework")
        st.markdown(
            "This evaluation uses transparent rule-based scoring — no external model. "
            "Every score is derived from checks you can read and verify yourself."
        )

        # ── Section 1: Rubric ─────────────────────────────────────────────────
        st.markdown("### Evaluation Rubric")
        st.caption(
            "Six dimensions, each scored 1–5. "
            "The table shows what each score level means and the typical expected "
            "score for the baseline vs. the agentic pipeline."
        )

        rubric_rows = [
            {
                "Dimension":         v["dimension"],
                "Question":          v["question"],
                "Score 1":           v["score_1"],
                "Score 3":           v["score_3"],
                "Score 5":           v["score_5"],
                "Baseline Typical":  v["baseline_typical"],
                "Agentic Typical":   v["agentic_typical"],
            }
            for v in ev.RUBRIC.values()
        ]
        st.dataframe(
            pd.DataFrame(rubric_rows).set_index("Dimension"),
            use_container_width=True,
            height=280,
        )

        st.divider()

        # ── Section 2: Built-in test cases ────────────────────────────────────
        st.markdown("### Built-in Test Cases")
        st.caption(
            "5 small clusters of synthetic reviews, each with a known dominant theme. "
            "Run the evaluation below to see how the pipeline scores on each case."
        )

        for case in ev.TEST_CASES:
            with st.expander(f"**{case['case_id']}** — {case['label']}", expanded=False):
                st.markdown(f"**Expected dominant theme:** `{case['dominant_theme']}`")
                st.markdown(f"**Expected keywords:** {', '.join(f'`{k}`' for k in case['expected_keywords'])}")
                st.dataframe(
                    pd.DataFrame(case["reviews"])[[
                        "customer_id", "written_explanation",
                        "recommendation_score_0_to_10", "electricity_plan",
                    ]],
                    use_container_width=True,
                    hide_index=True,
                )

        st.divider()

        # ── Section 3: Run sample evaluation ─────────────────────────────────
        st.markdown("### Run Sample Evaluation")
        st.caption(
            "Runs the full agentic pipeline on each test case in mock mode, "
            "scores the output on all 6 dimensions, and shows the results."
        )

        if "eval_results" not in st.session_state:
            st.session_state.eval_results = None

        if st.button("▶ Run Sample Evaluation", key="run_eval_btn"):
            eval_results = []
            prog = st.progress(0)
            for i, case in enumerate(ev.TEST_CASES):
                prog.progress((i + 1) / len(ev.TEST_CASES), text=f"Evaluating {case['case_id']}…")
                case_df = pd.DataFrame(case["reviews"])
                try:
                    output = pipeline.run_agent_pipeline(
                        case_df, product_goal, max_cards=3, use_mock=True
                    )
                    result = ev.evaluate_output(output, case)
                except Exception as exc:
                    result = {
                        "case_id":    case["case_id"],
                        "case_label": case["label"],
                        "error":      str(exc),
                        "pct_score":  0,
                        "scores":     {},
                    }
                eval_results.append(result)
            prog.empty()
            st.session_state.eval_results = eval_results

        if st.session_state.eval_results:
            results = st.session_state.eval_results

            # Summary table — one row per test case
            summary_rows = []
            for r in results:
                row = {
                    "Case":            r.get("case_label", r.get("case_id", "?")),
                    "Expected Theme":  r.get("dominant_theme", "—"),
                    "Total":           f"{r.get('total_score', '?')}/{r.get('max_score', '?')}",
                    "Score %":         f"{r.get('pct_score', 0)}%",
                }
                for dim_key, dim_meta in ev.RUBRIC.items():
                    dim_scores = r.get("scores", {})
                    row[dim_meta["dimension"]] = dim_scores.get(dim_key, {}).get("score", "—")
                summary_rows.append(row)

            st.dataframe(
                pd.DataFrame(summary_rows).set_index("Case"),
                use_container_width=True,
            )

            # Per-case detail expandable
            st.markdown("**Per-case score rationales:**")
            for r in results:
                with st.expander(f"{r.get('case_id')} — {r.get('case_label', '')}  ({r.get('pct_score', 0)}%)", expanded=False):
                    if "error" in r:
                        st.error(r["error"])
                        continue
                    detail_rows = [
                        {
                            "Dimension": ev.RUBRIC[k]["dimension"],
                            "Score":     v["score"],
                            "Rationale": v["rationale"],
                        }
                        for k, v in r.get("scores", {}).items()
                    ]
                    st.dataframe(
                        pd.DataFrame(detail_rows).set_index("Dimension"),
                        use_container_width=True,
                        hide_index=False,
                    )

            # CSV export
            st.divider()
            try:
                import io, csv as csv_mod
                flat_rows = []
                for r in results:
                    row = {
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

                buf = io.StringIO()
                writer = csv_mod.DictWriter(buf, fieldnames=list(flat_rows[0].keys()))
                writer.writeheader()
                writer.writerows(flat_rows)
                st.download_button(
                    label="⬇ Export Evaluation Results as CSV",
                    data=buf.getvalue(),
                    file_name="evaluation_results.csv",
                    mime="text/csv",
                )
            except Exception as exc:
                st.caption(f"CSV export unavailable: {exc}")

        st.divider()

        # ── Section 4: Baseline vs. Agentic comparison ────────────────────────
        st.markdown("### Baseline vs. Agentic Comparison")
        st.caption(
            "Scores the current session's baseline and agentic outputs side by side. "
            "Run **▶ Run Baseline** in the Baseline Comparison tab and "
            "**▶ Run Agent Analysis** first."
        )

        baseline_out  = st.session_state.baseline_out
        agentic_ready = st.session_state.pipeline_done

        if not baseline_out or not agentic_ready:
            missing = []
            if not baseline_out:
                missing.append("baseline output (go to **Baseline Comparison** tab → Run Baseline)")
            if not agentic_ready:
                missing.append("agentic pipeline output (click **▶ Run Agent Analysis** above)")
            st.info("Generate both outputs first:\n- " + "\n- ".join(missing))
        else:
            agentic_result = {
                "scored_opportunities": st.session_state.scored_themes,
                "backlog_cards":        st.session_state.cards,
                "insights_brief":       st.session_state.brief,
                "duplicates":           st.session_state.dup_result,
                "themes":               st.session_state.themes,
            }
            comparison = ev.compare_baseline_vs_agentic(baseline_out, agentic_result)
            comp_df = pd.DataFrame(comparison).set_index("Dimension")
            st.dataframe(comp_df, use_container_width=True)

            avg_baseline = sum(r["Baseline Score"] for r in comparison) / len(comparison)
            avg_agentic  = sum(r["Agentic Score"]  for r in comparison) / len(comparison)
            c1, c2, c3 = st.columns(3)
            c1.metric("Avg Baseline Score",  f"{avg_baseline:.1f} / 5")
            c2.metric("Avg Agentic Score",   f"{avg_agentic:.1f} / 5")
            c3.metric("Avg Improvement",     f"+{avg_agentic - avg_baseline:.1f} pts")
