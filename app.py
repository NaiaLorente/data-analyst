"""Streamlit web app for the AI Data Analyst Agent — bring your own LLM, your own key."""

import base64
import pickle
import urllib.request

import pandas as pd
import streamlit as st

from agent.analyst import narrate_drift, run_query
from agent.cleaning import (
    cap_outliers,
    convert_to_datetime,
    convert_to_numeric,
    count_open_issues,
    detect_all_issues,
    drop_column,
    drop_missing_rows,
    fill_missing,
    looks_datetime,
    remove_duplicate_rows,
    remove_outlier_rows,
    trim_whitespace,
)
from agent.cohort import compute_retention, plot_retention_heatmap
from agent.drift import (
    common_categorical_columns,
    common_numeric_columns,
    generate_drift_report,
    drift_report_to_html,
    drift_report_to_markdown,
    split_by_column,
    suggest_metric_column,
    suggest_segment_column,
    suggest_split_column,
)
from agent.insights import escape_markdown_math, generate_insights, insights_to_html, insights_to_markdown
from agent.join import JOIN_TYPE_DESCRIPTIONS, JOIN_TYPES, join_dataframes, join_stats, suggest_join_keys
from agent.loaders import SUPPORTED_EXTENSIONS, load_dataframe
from agent.notebook import build_notebook, notebook_to_json
from agent.pii import detect_pii_columns
from agent.predict import plot_feature_importance, predict_importance
from agent.profiling import plot_column_distribution, profile_column
from agent.providers import DEFAULT_PROVIDER, PROVIDERS
from agent.report import build_html_report, build_markdown_report
from agent.segment import find_segments, plot_segments
from agent.session_io import SessionLoadError, build_session, parse_session, restore_dataframe, session_to_json
from agent.stats import compare_conversion_rates
from agent.timeseries import analyze_trend, plot_trend_forecast


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Subtle depth on the bordered section cards (Auto-Insights, What Changed, etc.) —
# Streamlit's own st.container(border=True) renders a flat 1px outline with no
# elevation, which reads as a plain wireframe rather than a set of distinct panels.
st.markdown(
    """
    <style>
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 10px;
        box-shadow: 0 1px 3px rgba(11, 11, 11, 0.06);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📊 AI Data Analyst")
    st.caption("Chat with any dataset. Bring your own AI — free, private, your choice.")
    st.divider()

    st.markdown("**1. Choose your AI provider**")
    provider_ids = list(PROVIDERS.keys())
    provider_id = st.selectbox(
        "Provider",
        options=provider_ids,
        format_func=lambda pid: PROVIDERS[pid].label,
        index=provider_ids.index(DEFAULT_PROVIDER),
        label_visibility="collapsed",
    )
    provider = PROVIDERS[provider_id]

    model = st.text_input(
        "Model",
        value=provider.default_model,
        key=f"model_{provider_id}",
        help="Change to any model your provider supports.",
    )

    base_url = provider.base_url
    api_key = ""
    if provider.requires_key:
        api_key = st.text_input(
            f"{provider.label} API key",
            type="password",
            key=f"api_key_{provider_id}",
            help=f"{provider.key_help} Get one at {provider.get_key_url}",
        )
    else:
        st.success("No API key needed — runs locally via Ollama.")
        with st.expander("Advanced: Ollama host"):
            base_url = st.text_input("Base URL", value=provider.base_url, key="ollama_base_url")

    st.caption(
        "🔒 Your API key stays in this browser session only — it's never stored, "
        f"logged, or sent anywhere except directly to {provider.label}'s API."
    )

    st.divider()
    st.markdown("**2. Upload your data**")
    uploaded = st.file_uploader(
        "Upload a file",
        type=list(SUPPORTED_EXTENSIONS),
        label_visibility="collapsed",
    )
    if uploaded:
        st.success(f"Loaded: {uploaded.name}")

    st.divider()
    st.markdown("**3. Compare with another snapshot (optional)**")
    compare_mode = st.radio(
        "Comparison mode",
        options=["Upload a second file", "Split this file by a column"],
        label_visibility="collapsed",
        key="compare_mode",
    )

    compare_uploaded = None
    if compare_mode == "Upload a second file":
        st.caption("Upload last week's/last month's version to see what changed and why.")
        compare_uploaded = st.file_uploader(
            "Comparison file",
            type=list(SUPPORTED_EXTENSIONS),
            label_visibility="collapsed",
            key="compare_uploader",
        )
        if compare_uploaded:
            st.success(f"Comparing against: {compare_uploaded.name}")
    else:
        st.caption(
            "No second file needed — pick a column (e.g. month, quarter, cohort) and two "
            "values to compare, right below Auto-Insights."
        )

    st.divider()
    st.markdown("**Example questions**")
    examples = [
        "Give me a full summary of this dataset",
        "Which columns have the most missing values?",
        "Show the distribution of [column]",
        "What are the top correlations?",
        "Filter rows where [column] > 100",
        "Plot [x] vs [y] coloured by [category]",
    ]
    for ex in examples:
        st.markdown(f"- *{ex}*")

# ── Main area ──────────────────────────────────────────────────────────────────
st.header("Chat with your data")

# Widget state tied to the *previous* dataset's columns (split/metric/segment
# pickers) must not carry over when the primary dataset changes — otherwise a
# stale selection is either an invalid option (Streamlit raises) or, worse,
# visually "sticks" to the old label while the computation silently uses the
# new default underneath. Scoping their keys to a dataset_version counter
# forces a full widget remount on every dataset change, avoiding both.
def _reset_dataset_dependent_state():
    st.session_state["messages"] = []
    st.session_state["history"] = []
    st.session_state["drift_narrative"] = None
    st.session_state.pop("compare_df", None)
    st.session_state.pop("compare_filename", None)
    for key in [k for k in st.session_state if k.startswith("cleaned_df_")]:
        del st.session_state[key]
    st.session_state["dataset_version"] = st.session_state.get("dataset_version", 0) + 1


SAMPLE_DATASET_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"


def _load_sample_titanic() -> pd.DataFrame:
    with urllib.request.urlopen(SAMPLE_DATASET_URL, timeout=5) as r:
        return pd.read_csv(r)


if not uploaded:
    # Auto-load a sample the instant the app opens — no click, no upload — so a
    # first-time visitor (especially on a hosted demo) sees real results before
    # they've picked an AI provider or entered a key. Falls back to the plain
    # upload prompt (with a manual retry button) if fetching it fails, e.g. no
    # internet access, as with a fully local Ollama setup.
    if "df" not in st.session_state and not st.session_state.get("sample_load_failed"):
        try:
            df_sample = _load_sample_titanic()
        except Exception:
            st.session_state["sample_load_failed"] = True
        else:
            st.session_state["df"] = df_sample
            st.session_state["filename"] = "titanic.csv"
            st.session_state["using_sample"] = True
            _reset_dataset_dependent_state()

    if "df" not in st.session_state:
        st.info("Upload a file in the sidebar to get started — CSV, Excel, JSON, or Parquet.")
        if st.button("Use sample dataset (Titanic)"):
            try:
                df_sample = _load_sample_titanic()
            except Exception as exc:
                st.error(f"Couldn't load the sample dataset: {exc}")
            else:
                st.session_state["df"] = df_sample
                st.session_state["filename"] = "titanic.csv"
                st.session_state["using_sample"] = True
                _reset_dataset_dependent_state()
                st.rerun()
        st.stop()

    if st.session_state.get("using_sample"):
        st.info(
            "👋 Showing the sample Titanic dataset so you can try everything instantly. "
            "Upload your own file in the sidebar to analyze your data instead."
        )
else:
    if st.session_state.get("filename") != uploaded.name:
        try:
            new_df = load_dataframe(uploaded, uploaded.name)
        except Exception as exc:
            st.error(f"Couldn't read {uploaded.name}: {exc}")
            if "df" not in st.session_state:
                st.stop()
        else:
            st.session_state["df"] = new_df
            st.session_state["filename"] = uploaded.name
            st.session_state["using_sample"] = False
            _reset_dataset_dependent_state()

df: pd.DataFrame = st.session_state["df"]
filename: str = st.session_state.get("filename", "dataset.csv")
# Read fresh (not before the reset above) so a same-run dataset switch is
# reflected immediately in the widget keys below, not one rerun later.
dataset_version = st.session_state.get("dataset_version", 0)

# Filled in by the predict/segment/trend sections below (only when a result is
# applicable) and read by "Export notebook" further down the page.
notebook_predict_result = None
notebook_segment_result = None
notebook_trend_result = None

# Dataset preview
with st.expander(f"Preview: {filename}  ({df.shape[0]:,} rows × {df.shape[1]} cols)", expanded=False):
    st.dataframe(df.head(50), use_container_width=True)

tab_overview, tab_predict, tab_cohorts, tab_clean, tab_compare, tab_export = st.tabs(
    ["Overview", "Predict & Forecast", "Cohorts", "Clean & Join", "Compare", "Export"]
)

with tab_overview:
    # "PII scan" — flags columns that look like personal data *before* anything can
    # reach a third-party AI provider. Local-only sections below (Auto-Insights,
    # cleaning, What Changed's own numbers, etc.) always use the full `df` — only
    # `df_for_ai`, built here and used wherever the app calls out to a chosen AI
    # provider, has excluded columns actually removed.
    pii_findings = detect_pii_columns(df)
    excluded_pii_columns: list[str] = []
    if pii_findings:
        with st.container(border=True):
            st.markdown("### 🔒 Possible personal data detected")
            pii_cols = [f["column"] for f in pii_findings]
            st.warning(
                "These columns look like they might contain personal data: "
                + ", ".join(f"**{f['column']}** ({f['label']})" for f in pii_findings)
                + ". Excluded columns below are never sent to the AI provider you chose — not in chat, not in narration."
            )
            # A restored session (see "Save / load session" in the sidebar) may carry
            # its own prior exclusion choice — honor it if this is the same restore,
            # intersected with what's actually flagged now, so a stale/foreign value
            # can never be passed to a widget that doesn't have it as a valid option.
            restore_hint = st.session_state.get("pii_exclusions_to_restore")
            if restore_hint and restore_hint[0] == dataset_version:
                default_pii_exclusions = [c for c in pii_cols if c in restore_hint[1]]
            else:
                default_pii_exclusions = pii_cols
            excluded_pii_columns = st.multiselect(
                "Exclude these columns from anything sent to the AI",
                options=pii_cols,
                default=default_pii_exclusions,
                key=f"pii_exclude_{dataset_version}",
            )
    df_for_ai = df.drop(columns=excluded_pii_columns) if excluded_pii_columns else df

    # Auto-insights — computed instantly with pandas, no API key or cost required
    insights = generate_insights(df)
    with st.container(border=True):
        st.markdown("### 🔎 Auto-Insights")
        st.caption("Computed instantly — no AI call, no API key needed.")
        st.markdown(escape_markdown_math(insights_to_markdown(insights)))

    # "Explore a column" — full stats and a distribution chart for any single column,
    # on demand, computed instantly with pandas.
    with st.container(border=True):
        st.markdown("### 🔬 Explore a column")
        st.caption("Full stats and a distribution chart for any column, computed instantly.")
        profile_col = st.selectbox(
            "Column",
            options=list(df.columns),
            key=f"profile_column_{dataset_version}",
            label_visibility="collapsed",
        )
        profile = profile_column(df, profile_col)
        kind = profile["kind"]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Type", profile["dtype"])
        c2.metric("Non-null", f"{profile['count'] - profile['missing']:,}")
        c3.metric("Missing", f"{profile['missing']:,} ({profile['missing_pct']}%)")

        if kind == "numeric" and "mean" in profile:
            c4.metric("Mean", f"{profile['mean']:,.2f}")
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Median", f"{profile['median']:,.2f}")
            d2.metric("Std dev", f"{profile['std']:,.2f}")
            d3.metric("Min", f"{profile['min']:,.2f}")
            d4.metric("Max", f"{profile['max']:,.2f}")
        elif kind == "categorical":
            c4.metric("Unique values", f"{profile['unique']:,}" if profile["unique"] is not None else "—")
            if profile["top_values"]:
                st.dataframe(pd.DataFrame(profile["top_values"]), use_container_width=True, hide_index=True)
        elif kind == "datetime" and "min" in profile:
            c4.metric("Range", f"{profile['min'][:10]} → {profile['max'][:10]}")
        elif kind == "boolean":
            c4.metric("Values", ", ".join(f"{k}: {v:,}" for k, v in profile["value_counts"].items()))

        dist_chart = plot_column_distribution(df, profile_col)
        if dist_chart:
            st.image(base64.b64decode(dist_chart), use_container_width=True)

with tab_predict:
    # "What predicts this?" — a quick baseline Random Forest, not just correlation.
    # Gated behind a button (training a model on every widget interaction elsewhere
    # on the page would be wasteful) and cached until the target or dataset changes.
    with st.container(border=True):
        st.markdown("### 🤖 What predicts this?")
        st.caption("Train a quick baseline model to see which columns actually drive a target — not just correlate with it.")
        predict_target = st.selectbox(
            "Target column", options=list(df.columns), key=f"predict_target_{dataset_version}"
        )
        predict_scope = f"{dataset_version}_{predict_target}"
        if st.button("Run", key=f"predict_run_{dataset_version}"):
            with st.spinner("Training a baseline model…"):
                st.session_state["predict_result"] = predict_importance(df, predict_target)
                st.session_state["predict_result_scope"] = predict_scope

        if st.session_state.get("predict_result_scope") == predict_scope:
            predict_result = st.session_state["predict_result"]
            if not predict_result["applicable"]:
                st.caption(predict_result["reason"])
            else:
                score_label = "Accuracy" if predict_result["task"] == "classification" else "R²"
                score_value = predict_result["score"]
                p1, p2, p3 = st.columns(3)
                p1.metric("Task", predict_result["task"].capitalize())
                p2.metric(score_label, f"{score_value:.1%}" if predict_result["task"] == "classification" else f"{score_value:.3f}")
                p3.metric("Rows used", f"{predict_result['n_rows_used']:,}")
                st.caption(f"Evaluated on a held-out {predict_result['n_test']:,}-row test split (trained on {predict_result['n_train']:,}).")
                st.image(
                    base64.b64decode(plot_feature_importance(predict_result["features"])), use_container_width=True
                )
                notebook_predict_result = predict_result
                st.download_button(
                    "⬇️ Download trained model (.pkl)",
                    data=pickle.dumps(predict_result["model_bundle"]),
                    file_name=f"{filename.rsplit('.', 1)[0]}_{predict_result['target']}_model.pkl",
                    mime="application/octet-stream",
                    key=f"model_download_{dataset_version}_{predict_target}",
                    help="A scikit-learn bundle (model + feature columns + encoders) — load with pickle.load() in Python.",
                )

    # "Segments" — unsupervised complement to "What predicts this?": KMeans on
    # standardized numeric columns finds natural groups with no target needed,
    # profiled by which columns deviate most from the overall mean per cluster.
    with st.container(border=True):
        st.markdown("### 🧩 Segments")
        st.caption("Find natural groups in your data (e.g. customer segments) without picking a target column.")
        segment_numeric_cols = list(df.select_dtypes(include="number").columns)
        if len(segment_numeric_cols) < 2:
            st.caption("Needs at least 2 numeric columns.")
        else:
            s1, s2 = st.columns([3, 1])
            with s1:
                segment_cols = st.multiselect(
                    "Columns to segment on",
                    options=segment_numeric_cols,
                    default=segment_numeric_cols,
                    key=f"segment_cols_{dataset_version}",
                )
            with s2:
                segment_n = st.number_input(
                    "Segments", min_value=2, max_value=8, value=4, key=f"segment_n_{dataset_version}"
                )
            segment_scope = f"{dataset_version}_{segment_cols}_{segment_n}"
            if st.button("Find segments", key=f"segment_run_{dataset_version}"):
                with st.spinner("Clustering…"):
                    st.session_state["segment_result"] = find_segments(df, columns=segment_cols, n_clusters=segment_n)
                    st.session_state["segment_result_scope"] = segment_scope

            if st.session_state.get("segment_result_scope") == segment_scope:
                segment_result = st.session_state["segment_result"]
                if not segment_result["applicable"]:
                    st.caption(segment_result["reason"])
                else:
                    st.image(base64.b64decode(plot_segments(segment_result)), use_container_width=True)
                    notebook_segment_result = segment_result
                    for cluster in segment_result["clusters"]:
                        top = cluster["top_features"][0]
                        direction = "higher" if top["z_diff"] > 0 else "lower"
                        st.caption(
                            f"**Cluster {cluster['cluster']}** — {cluster['size']:,} rows ({cluster['pct_of_rows']:.1f}%): "
                            f"notably {direction} **{top['column']}** ({top['cluster_mean']:.2f} vs. overall {top['overall_mean']:.2f})"
                        )

    # "Trend & forecast" — linear trend, weekly seasonality, and a short forecast for
    # a date column + metric, pure pandas/numpy (zero AI cost).
    with st.container(border=True):
        st.markdown("### 📈 Trend & forecast")
        st.caption("Detects the trend and weekly pattern in a metric over time, then forecasts a couple of weeks ahead.")
        date_candidates = [
            c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c]) or looks_datetime(df[c].dropna())
        ]
        trend_numeric_candidates = list(df.select_dtypes(include="number").columns)
        if not date_candidates or not trend_numeric_candidates:
            st.caption("Needs at least one date-like column and one numeric column.")
        else:
            t1, t2, t3 = st.columns(3)
            with t1:
                trend_date_col = st.selectbox("Date column", options=date_candidates, key=f"trend_date_{dataset_version}")
            with t2:
                trend_metric_col = st.selectbox(
                    "Metric", options=trend_numeric_candidates, key=f"trend_metric_{dataset_version}"
                )
            with t3:
                trend_agg = st.radio("Aggregation", options=["sum", "mean"], horizontal=True, key=f"trend_agg_{dataset_version}")

            trend_result = analyze_trend(df, trend_date_col, trend_metric_col, agg=trend_agg)
            if not trend_result["applicable"]:
                st.caption(trend_result["reason"])
            else:
                r1, r2, r3 = st.columns(3)
                r1.metric("Trend", trend_result["trend_direction"].capitalize())
                change = trend_result["trend_pct_over_period"]
                r2.metric("Change over period", f"{change:+.1f}%" if change is not None else "—")
                r3.metric("Weekly pattern", "Yes" if trend_result["has_weekly_seasonality"] else "No")
                st.image(base64.b64decode(plot_trend_forecast(trend_result)), use_container_width=True)
                notebook_trend_result = trend_result

with tab_cohorts:
    # "Cohort & retention" — group users by first activity, track what fraction
    # stick around in later periods. Needs an event log (one row per user per
    # activity, e.g. a transactions or logins table), not an aggregated snapshot.
    with st.container(border=True):
        st.markdown("### 👥 Cohort & retention")
        st.caption(
            "Group users by their first activity and track what fraction stick around later — "
            "needs one row per user per event, not an aggregated snapshot."
        )
        cohort_date_candidates = [
            c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c]) or looks_datetime(df[c].dropna())
        ]
        if not cohort_date_candidates:
            st.caption("Needs at least one date-like column.")
        else:
            cohort_user_candidates = list(df.columns)
            default_user_col = next(
                (c for c in cohort_user_candidates if "user" in str(c).lower() or "id" in str(c).lower()),
                cohort_user_candidates[0],
            )
            cc1, cc2, cc3 = st.columns(3)
            with cc1:
                cohort_user_col = st.selectbox(
                    "User/ID column",
                    options=cohort_user_candidates,
                    index=cohort_user_candidates.index(default_user_col),
                    key=f"cohort_user_{dataset_version}",
                )
            with cc2:
                cohort_date_col = st.selectbox(
                    "Event date column", options=cohort_date_candidates, key=f"cohort_date_{dataset_version}"
                )
            with cc3:
                cohort_period = st.radio(
                    "Period",
                    options=["M", "W", "D"],
                    format_func=lambda p: {"M": "Monthly", "W": "Weekly", "D": "Daily"}[p],
                    horizontal=True,
                    key=f"cohort_period_{dataset_version}",
                )

            cohort_result = compute_retention(df, cohort_user_col, cohort_date_col, period=cohort_period)
            if not cohort_result["applicable"]:
                st.caption(cohort_result["reason"])
            else:
                st.image(base64.b64decode(plot_retention_heatmap(cohort_result)), use_container_width=True)

with tab_clean:
    # "Clean your data" — one-click fixes for the same issues Auto-Insights just
    # flagged. Fixes apply instantly to a working copy scoped to this dataset, so
    # the original upload is never touched until you download the result.
    cleaned_key = f"cleaned_df_{dataset_version}"
    if cleaned_key not in st.session_state:
        st.session_state[cleaned_key] = df.copy()
    cleaned_df = st.session_state[cleaned_key]
    issues = detect_all_issues(cleaned_df)
    open_issues = count_open_issues(issues)
    has_changes = not cleaned_df.equals(df)

    with st.container(border=True):
        st.markdown("### 🧹 Clean your data")
        if open_issues == 0 and not has_changes:
            st.caption("No common data-quality issues detected — nothing to clean.")
        else:
            if open_issues:
                st.caption(f"{open_issues} issue(s) left — each fix applies instantly to a working copy.")
            else:
                st.success("All detected issues fixed — download your cleaned file below.")

            if issues["duplicates"]:
                d = issues["duplicates"]
                c1, c2 = st.columns([4, 1])
                with c1:
                    st.write(f"🔁 {d['count']:,} duplicate rows ({d['pct']}%)")
                with c2:
                    if st.button("Remove", key=f"fix_dup_{dataset_version}"):
                        st.session_state[cleaned_key] = remove_duplicate_rows(cleaned_df)
                        st.rerun()

            for m in issues["missing"]:
                c1, c2, c3 = st.columns([4, 1.4, 1])
                with c1:
                    st.write(f"🕳 '{m['column']}' is {m['pct']}% missing ({m['count']:,} rows)")
                with c2:
                    if m["numeric"]:
                        strategy = st.selectbox(
                            "Fill with",
                            options=["median", "mean", "zero"],
                            key=f"fill_strategy_{dataset_version}_{m['column']}",
                            label_visibility="collapsed",
                        )
                    else:
                        strategy = "mode"
                        st.caption("Fill with most common value")
                    if st.button("Fill", key=f"fix_fill_{dataset_version}_{m['column']}"):
                        st.session_state[cleaned_key] = fill_missing(cleaned_df, m["column"], strategy)
                        st.rerun()
                with c3:
                    if st.button("Drop rows", key=f"fix_dropna_{dataset_version}_{m['column']}"):
                        st.session_state[cleaned_key] = drop_missing_rows(cleaned_df, m["column"])
                        st.rerun()

            for o in issues["outliers"]:
                c1, c2, c3 = st.columns([4, 1, 1])
                with c1:
                    st.write(f"📈 '{o['column']}' has {o['count']:,} likely outliers ({o['pct']}%)")
                with c2:
                    if st.button("Cap", key=f"fix_cap_{dataset_version}_{o['column']}"):
                        st.session_state[cleaned_key] = cap_outliers(cleaned_df, o["column"])
                        st.rerun()
                with c3:
                    if st.button("Remove rows", key=f"fix_rmout_{dataset_version}_{o['column']}"):
                        st.session_state[cleaned_key] = remove_outlier_rows(cleaned_df, o["column"])
                        st.rerun()

            for w in issues["whitespace"]:
                c1, c2 = st.columns([4, 1])
                with c1:
                    st.write(f"✂️ '{w['column']}' has {w['count']:,} values with stray whitespace")
                with c2:
                    if st.button("Trim", key=f"fix_trim_{dataset_version}_{w['column']}"):
                        st.session_state[cleaned_key] = trim_whitespace(cleaned_df, w["column"])
                        st.rerun()

            for t in issues["types"]:
                is_date = t["kind"] == "type_datetime"
                c1, c2 = st.columns([4, 1])
                with c1:
                    st.write(f"🔢 '{t['column']}' looks like a {'date' if is_date else 'number'} but is stored as text")
                with c2:
                    if st.button("Convert", key=f"fix_type_{dataset_version}_{t['column']}"):
                        converter = convert_to_datetime if is_date else convert_to_numeric
                        st.session_state[cleaned_key] = converter(cleaned_df, t["column"])
                        st.rerun()

            for cc in issues["constant_columns"]:
                c1, c2 = st.columns([4, 1])
                with c1:
                    if cc["kind"] == "empty_column":
                        st.write(f"🚫 '{cc['column']}' is completely empty")
                    else:
                        st.write(f"🚫 '{cc['column']}' is the same value in every row ({cc['value']!r})")
                with c2:
                    if st.button("Drop column", key=f"fix_const_{dataset_version}_{cc['column']}"):
                        st.session_state[cleaned_key] = drop_column(cleaned_df, cc["column"])
                        st.rerun()

        if has_changes:
            dl_col, reset_col = st.columns([3, 1])
            with dl_col:
                st.download_button(
                    "⬇️ Download cleaned CSV",
                    data=cleaned_df.to_csv(index=False),
                    file_name=f"{filename.rsplit('.', 1)[0]}_cleaned.csv",
                    mime="text/csv",
                    key=f"cleaned_download_{dataset_version}",
                )
            with reset_col:
                if st.button("Reset", key=f"cleaned_reset_{dataset_version}"):
                    st.session_state[cleaned_key] = df.copy()
                    st.rerun()

    # "Join files" — the app otherwise assumes one flat file; this lets you enrich
    # the working dataset with a second file on a shared key (orders + customers).
    with st.container(border=True):
        st.markdown("### 🔗 Join files")
        st.caption("Combine this dataset with a second file on a shared key — orders + customers, events + users.")
        join_upload = st.file_uploader(
            "File to join in", type=list(SUPPORTED_EXTENSIONS), key=f"join_uploader_{dataset_version}"
        )
        if join_upload:
            try:
                join_df_b = load_dataframe(join_upload, join_upload.name)
            except Exception as exc:
                st.error(f"Couldn't read {join_upload.name}: {exc}")
            else:
                suggested_keys = suggest_join_keys(cleaned_df, join_df_b)
                key_options_a = list(cleaned_df.columns)
                key_options_b = list(join_df_b.columns)
                default_key = suggested_keys[0] if suggested_keys else None

                j1, j2, j3 = st.columns(3)
                with j1:
                    join_left_key = st.selectbox(
                        "Key in this dataset",
                        options=key_options_a,
                        index=key_options_a.index(default_key) if default_key in key_options_a else 0,
                        key=f"join_left_{dataset_version}_{join_upload.name}",
                    )
                with j2:
                    join_right_key = st.selectbox(
                        "Key in the joined file",
                        options=key_options_b,
                        index=key_options_b.index(default_key) if default_key in key_options_b else 0,
                        key=f"join_right_{dataset_version}_{join_upload.name}",
                    )
                with j3:
                    join_how = st.radio(
                        "Join type", options=JOIN_TYPES, horizontal=True, key=f"join_how_{dataset_version}_{join_upload.name}"
                    )
                st.caption(JOIN_TYPE_DESCRIPTIONS[join_how])

                try:
                    join_result = join_dataframes(cleaned_df, join_df_b, join_left_key, join_right_key, how=join_how)
                except Exception as exc:
                    st.error(f"Couldn't join on these keys: {exc}")
                else:
                    join_result_stats = join_stats(cleaned_df, join_df_b, join_result, join_left_key, join_right_key)
                    jc1, jc2, jc3 = st.columns(3)
                    jc1.metric("Rows before", f"{join_result_stats['rows_a']:,} + {join_result_stats['rows_b']:,}")
                    jc2.metric("Rows after", f"{join_result_stats['rows_result']:,}")
                    match_rate = join_result_stats["match_rate"]
                    jc3.metric("Key match rate", f"{match_rate:.1%}" if match_rate is not None else "—")

                    st.dataframe(join_result.head(20), use_container_width=True)

                    jd1, jd2 = st.columns(2)
                    with jd1:
                        if st.button("Use joined result as working dataset", key=f"join_use_{dataset_version}"):
                            st.session_state["df"] = join_result
                            st.session_state["filename"] = f"{filename.rsplit('.', 1)[0]}_joined.csv"
                            st.session_state["using_sample"] = False
                            _reset_dataset_dependent_state()
                            st.rerun()
                    with jd2:
                        st.download_button(
                            "⬇️ Download joined CSV",
                            data=join_result.to_csv(index=False),
                            file_name=f"{filename.rsplit('.', 1)[0]}_joined.csv",
                            mime="text/csv",
                            key=f"join_download_{dataset_version}",
                        )

with tab_compare:
    # "What Changed" — drift/root-cause analysis, either against a second snapshot
    # or against another slice of this same file, if requested
    drift_report = None
    drift_narrative = None
    drift_df_a = drift_df_b = None
    exclude_columns: list[str] = []
    compare_label = ""
    # Scopes the metric/segment widget keys below so they fully remount (rather
    # than visually sticking to a stale label) whenever *what's being compared*
    # changes — new dataset, new comparison file, or a different split column.
    compare_scope = f"{dataset_version}"

    if compare_mode == "Upload a second file" and compare_uploaded:
        if st.session_state.get("compare_filename") != compare_uploaded.name:
            try:
                st.session_state["compare_df"] = load_dataframe(compare_uploaded, compare_uploaded.name)
            except Exception as exc:
                st.error(f"Couldn't read {compare_uploaded.name}: {exc}")
                st.session_state.pop("compare_df", None)
            else:
                st.session_state["compare_filename"] = compare_uploaded.name
        if "compare_df" in st.session_state:
            drift_df_a, drift_df_b = df, st.session_state["compare_df"]
            compare_label = f"Comparing {filename} → {compare_uploaded.name}."
            compare_scope = f"{dataset_version}_file_{compare_uploaded.name}"

    elif compare_mode == "Split this file by a column":
        df_columns = list(df.columns)
        default_split_column = suggest_split_column(df)
        split_column = st.selectbox(
            "Column to split by",
            options=df_columns,
            index=df_columns.index(default_split_column) if default_split_column in df_columns else 0,
            key=f"split_column_{dataset_version}",
        )
        try:
            unique_vals = sorted(df[split_column].dropna().unique().tolist(), key=str)[:200]
        except TypeError:
            unique_vals = None
            st.caption(f"'{split_column}' contains values that can't be compared (e.g. nested objects) — pick another column.")

        if unique_vals is not None and len(unique_vals) < 2:
            st.caption(f"'{split_column}' needs at least two distinct values to compare.")
        elif unique_vals is not None:
            col_a, col_b = st.columns(2)
            with col_a:
                value_a = st.selectbox(
                    "Group A (before)", options=unique_vals, index=0, key=f"split_value_a_{dataset_version}_{split_column}"
                )
            with col_b:
                value_b = st.selectbox(
                    "Group B (after)",
                    options=unique_vals,
                    index=min(1, len(unique_vals) - 1),
                    key=f"split_value_b_{dataset_version}_{split_column}",
                )
            if value_a == value_b:
                st.caption("Pick two different values to compare.")
            else:
                group_a, group_b = split_by_column(df, split_column, value_a, value_b)
                if group_a.empty or group_b.empty:
                    st.caption("One of the selected groups has no rows.")
                else:
                    drift_df_a, drift_df_b = group_a, group_b
                    exclude_columns = [split_column]
                    compare_label = f"Comparing {split_column} = {value_a!r} → {split_column} = {value_b!r}."
                    compare_scope = f"{dataset_version}_split_{split_column}_{value_a!r}_{value_b!r}"

    if drift_df_a is not None and drift_df_b is not None:
        numeric_cols = [c for c in common_numeric_columns(drift_df_a, drift_df_b) if c not in exclude_columns]
        categorical_cols = [c for c in common_categorical_columns(drift_df_a, drift_df_b) if c not in exclude_columns]

        with st.container(border=True):
            st.markdown("### 🔀 What Changed")
            st.caption(f"{compare_label} All numbers computed directly with pandas — never guessed by the AI.")

            metric = segment = None
            if numeric_cols and categorical_cols:
                default_metric = suggest_metric_column(drift_df_a, drift_df_b, exclude=tuple(exclude_columns))
                default_segment = suggest_segment_column(drift_df_a, drift_df_b, exclude=tuple(exclude_columns))
                col1, col2 = st.columns(2)
                with col1:
                    metric = st.selectbox(
                        "Metric to explain",
                        options=numeric_cols,
                        index=numeric_cols.index(default_metric) if default_metric in numeric_cols else 0,
                        key=f"drift_metric_{compare_scope}",
                    )
                with col2:
                    segment = st.selectbox(
                        "Break down by",
                        options=categorical_cols,
                        index=categorical_cols.index(default_segment) if default_segment in categorical_cols else 0,
                        key=f"drift_segment_{compare_scope}",
                    )
            else:
                st.caption("Add a shared numeric and categorical column to see a driver breakdown.")

            drift_report = generate_drift_report(
                drift_df_a, drift_df_b, metric=metric, segment=segment, exclude_columns=exclude_columns
            )
            st.markdown(escape_markdown_math(drift_report_to_markdown(drift_report)))

            if "driver" in drift_report:
                with st.expander("Full segment breakdown"):
                    st.dataframe(pd.DataFrame(drift_report["driver"]["by_segment"]), use_container_width=True)

            # A previously generated narrative is only valid for the exact metric/segment/
            # comparison it was generated from — checking that at read time (rather than
            # trying to catch every place the selection could change) means it can never
            # go stale and get shown next to numbers it doesn't actually describe.
            narrative_scope = f"{compare_scope}_{metric}_{segment}"
            if st.session_state.get("drift_narrative_scope") == narrative_scope:
                drift_narrative = st.session_state.get("drift_narrative")

            narrative_blocked_by_pii = bool(excluded_pii_columns) and (
                metric in excluded_pii_columns or segment in excluded_pii_columns
            )
            if narrative_blocked_by_pii:
                st.caption(
                    "🔒 AI narration is disabled — the selected metric/segment column was excluded "
                    "as possible personal data above."
                )
            elif st.button("🧠 Explain in plain English"):
                if provider.requires_key and not api_key:
                    st.error(f"Add your {provider.label} API key in the sidebar to generate a narrative.")
                else:
                    with st.spinner(f"Narrating with {provider.label}…"):
                        try:
                            drift_narrative = narrate_drift(
                                drift_report, provider_id=provider_id, model=model, api_key=api_key, base_url=base_url
                            )
                        except Exception as exc:
                            drift_narrative = f"⚠️ Error calling {provider.label}: {exc}"
                    st.session_state["drift_narrative"] = drift_narrative
                    st.session_state["drift_narrative_scope"] = narrative_scope

            if drift_narrative:
                st.info(escape_markdown_math(drift_narrative))

    # "A/B test calculator" — a standalone two-proportion z-test. Doesn't touch the
    # uploaded dataset at all, so it works for numbers pasted in from any A/B testing
    # tool, not just columns already in this file.
    with st.container(border=True):
        st.markdown("### 🧪 A/B test calculator")
        st.caption(
            "Two-proportion z-test for a binary outcome (conversion, click-through, churn) — "
            "paste in counts from any A/B test tool."
        )
        ab_col_a, ab_col_b = st.columns(2)
        with ab_col_a:
            st.markdown("**Group A**")
            ab_count_a = st.number_input("Conversions (A)", min_value=0, value=0, step=1, key="ab_count_a")
            ab_total_a = st.number_input("Total (A)", min_value=0, value=0, step=1, key="ab_total_a")
        with ab_col_b:
            st.markdown("**Group B**")
            ab_count_b = st.number_input("Conversions (B)", min_value=0, value=0, step=1, key="ab_count_b")
            ab_total_b = st.number_input("Total (B)", min_value=0, value=0, step=1, key="ab_total_b")

        if ab_total_a > 0 and ab_total_b > 0:
            ab_result = compare_conversion_rates(int(ab_count_a), int(ab_total_a), int(ab_count_b), int(ab_total_b))
            if not ab_result["applicable"]:
                st.caption(ab_result["reason"])
            else:
                a1, a2, a3 = st.columns(3)
                a1.metric("Rate A", f"{ab_result['rate_a']:.2%}")
                a2.metric("Rate B", f"{ab_result['rate_b']:.2%}")
                a3.metric("Difference", f"{ab_result['diff']:+.2%}")
                verdict = "Statistically significant" if ab_result["significant"] else "Not statistically significant"
                st.markdown(
                    f"**{verdict}** (p={ab_result['p_value']:.4f}, z={ab_result['z']:.2f}) — "
                    f"95% CI on the difference: {ab_result['ci_low']:+.2%} to {ab_result['ci_high']:+.2%}"
                )
        else:
            st.caption("Enter totals greater than 0 for both groups to run the test.")

with tab_export:
    # "Export notebook" — packages the dataset plus whichever analyses above were
    # actually run into a self-contained, standalone .ipynb (pure pandas/sklearn,
    # no dependency on this app) so the work continues outside the browser.
    with st.container(border=True):
        st.markdown("### 📓 Export as notebook")
        st.caption("Download a self-contained Jupyter notebook with your data and the analyses you've run above.")
        notebook = build_notebook(
            cleaned_df,
            filename,
            predict_result=notebook_predict_result,
            segment_result=notebook_segment_result,
            trend_result=notebook_trend_result,
        )
        st.download_button(
            "⬇️ Download notebook (.ipynb)",
            data=notebook_to_json(notebook),
            file_name=f"{filename.rsplit('.', 1)[0]}_analysis.ipynb",
            mime="application/x-ipynb+json",
            key=f"notebook_download_{dataset_version}",
        )

st.divider()

# Chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "history" not in st.session_state:
    st.session_state["history"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(escape_markdown_math(msg["content"]))
        for chart_b64 in msg.get("charts", []):
            st.image(base64.b64decode(chart_b64), use_container_width=True)
        if msg.get("code"):
            with st.expander("🐍 View the code"):
                st.code("\n".join(msg["code"]), language="python")

if excluded_pii_columns:
    st.caption(f"🔒 Excluded from chat: {', '.join(excluded_pii_columns)}")

# Input
question = st.chat_input("Ask anything about your data…")

if question:
    if provider.requires_key and not api_key:
        st.error(
            f"Add your {provider.label} API key in the sidebar (or switch to Ollama "
            "for free local analysis) to ask questions."
        )
    else:
        st.session_state["messages"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(escape_markdown_math(question))

        with st.chat_message("assistant"):
            with st.spinner(f"Analysing with {provider.label}…"):
                try:
                    answer, charts, code = run_query(
                        df_for_ai,
                        question,
                        st.session_state["history"],
                        provider_id=provider_id,
                        model=model,
                        api_key=api_key,
                        base_url=base_url,
                    )
                except Exception as exc:
                    answer, charts, code = f"⚠️ Error calling {provider.label}: {exc}", [], []

            st.markdown(escape_markdown_math(answer))
            for chart_b64 in charts:
                st.image(base64.b64decode(chart_b64), use_container_width=True)
            if code:
                with st.expander("🐍 View the code"):
                    st.code("\n".join(code), language="python")

        st.session_state["messages"].append(
            {"role": "assistant", "content": answer, "charts": charts, "code": code}
        )
        # Keep a rolling window of 20 turns for the API history
        st.session_state["history"].append({"role": "user", "content": question})
        st.session_state["history"].append({"role": "assistant", "content": answer})
        st.session_state["history"] = st.session_state["history"][-20:]

# Shareable report export
if st.session_state["messages"]:
    st.divider()
    st.markdown("**Share this analysis**")
    drift_md = drift_html = None
    if drift_report is not None:
        drift_md = drift_report_to_markdown(drift_report)
        drift_html = drift_report_to_html(drift_report)

    col1, col2 = st.columns(2)
    with col1:
        html_report = build_html_report(
            filename, insights_to_html(insights), st.session_state["messages"], drift_html, drift_narrative
        )
        st.download_button(
            "📄 Download report (HTML)",
            data=html_report,
            file_name=f"{filename}_report.html",
            mime="text/html",
            use_container_width=True,
        )
    with col2:
        md_report = build_markdown_report(
            filename, insights_to_markdown(insights), st.session_state["messages"], drift_md, drift_narrative
        )
        st.download_button(
            "📝 Download report (Markdown)",
            data=md_report,
            file_name=f"{filename}_report.md",
            mime="text/markdown",
            use_container_width=True,
        )

# "Save / load session" — everything above lives only in this browser tab's
# memory; closing it throws away the dataset, the conversation, and any
# cleaning progress. This bundles that state into one downloadable file and
# restores it later, with no server and no cost — the local alternative to
# server-side persistence. Predict/segment/trend results are cheap to
# regenerate and deliberately left out to keep the file small.
with st.sidebar:
    st.divider()
    st.markdown("**Save / load your session**")
    st.caption(
        "Nothing here survives closing this tab. Save your dataset, chat, and cleaning "
        "progress to a file, then load it back anytime — no account, no cost."
    )

    session_bundle = build_session(
        df, filename, st.session_state["messages"], st.session_state["history"], excluded_pii_columns, cleaned_df
    )
    st.download_button(
        "💾 Save session",
        data=session_to_json(session_bundle),
        file_name=f"{filename.rsplit('.', 1)[0]}_session.json",
        mime="application/json",
        use_container_width=True,
        key=f"session_save_{dataset_version}",
    )

    session_uploaded = st.file_uploader("Load a saved session", type=["json"], key="session_uploader")
    if session_uploaded and st.session_state.get("loaded_session_filename") != session_uploaded.name:
        st.session_state["loaded_session_filename"] = session_uploaded.name
        try:
            loaded_session = parse_session(session_uploaded.getvalue().decode("utf-8"))
            restored_df = restore_dataframe(loaded_session)
        except SessionLoadError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Couldn't load that session file: {exc}")
        else:
            st.session_state["df"] = restored_df
            st.session_state["filename"] = loaded_session.get("filename", "restored_dataset.csv")
            st.session_state["using_sample"] = False
            _reset_dataset_dependent_state()
            new_version = st.session_state["dataset_version"]
            st.session_state["messages"] = loaded_session.get("messages", [])
            st.session_state["history"] = loaded_session.get("history", [])
            if "excluded_pii_columns" in loaded_session:
                st.session_state["pii_exclusions_to_restore"] = (new_version, loaded_session["excluded_pii_columns"])
            if "cleaned_dataset_csv" in loaded_session:
                st.session_state[f"cleaned_df_{new_version}"] = restore_dataframe(
                    loaded_session, "cleaned_dataset_csv"
                )
            st.success("Session restored.")
            st.rerun()
