"""REACT decision engine view."""

from dashboard_core import *
from views.participant_detail import monitoring_participant_roster, summarize_monitor_days


def render_live_decision_view(include_demo_devices: bool) -> None:
    """Live decisions in a recent bounded window, from monitoring timeline."""
    st.header("Decision Engine")
    st.caption("Actual recent decision events from the monitoring API, not bundled analysis files.")
    try:
        roster = monitoring_participant_roster(include_demo_devices)
    except Exception as exc:
        st.error(f"Monitoring roster unavailable: {exc}")
        return
    today = pd.Timestamp.now(tz=LOCAL_TIMEZONE).date()
    rows = []
    errors = []
    for participant in roster:
        try:
            data = summarize_monitor_days(load_monitor_timeline(int(participant["user_id"]), local_date=today.isoformat(), days=7))
        except Exception:
            errors.append(str(participant["participant_id"]))
            continue
        for d in data["decisions"]:
            failure = str(d.get("delivery_error") or "").strip()
            rows.append({"Participant": participant["participant_id"],
                         "Time (ET)": parse_monitor_time(d.get("at")),
                         "Reason": decision_activity_reason(d),
                         "Prompt selected": bool(d.get("send_prompt")),
                         "Distress active": d.get("distress_flag_active") is True,
                         "Delivery error": failure})
    if errors:
        st.warning(f"Timeline unavailable for {len(errors)} participant(s); totals are partial.")
    if not rows:
        st.info("No recent decision events returned.")
        return
    df = pd.DataFrame(rows)
    stats = st.columns(3)
    stats[0].metric("Decision events", len(df))
    stats[1].metric("Prompts selected", int(df["Prompt selected"].sum()))
    stats[2].metric("Recorded delivery errors", int(df["Delivery error"].ne("").sum()))
    st.subheader("Decision reasons")
    st.dataframe(df["Reason"].value_counts().rename_axis("Reason").reset_index(name="Count"), use_container_width=True, hide_index=True)
    df = df.sort_values("Time (ET)", ascending=False)
    df["Time (ET)"] = df["Time (ET)"].apply(format_pipeline_time)
    st.subheader("Recent decisions")
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.caption("Covers up to the last 7 Eastern calendar days. A distress-active decision is not a separate distress flag. Missing device receipts do not indicate failed delivery.")


def render_decision_view(log_df: pd.DataFrame, summary_df: pd.DataFrame, user_table: pd.DataFrame) -> None:
    # Header
    st.header("Decision Engine")
    st.caption("Prompt totals and decision explanations from the decision engine.")


    # Top-level metrics
    total_users = summary_df["user_id"].nunique()
    total_prompts = int(log_df["send_prompt"].sum())
    total_records = len(log_df)
    count_mismatches = int((~user_table["count_matches"]).sum())

    metric_columns = st.columns(4)
    metric_columns[0].metric("Users", total_users)
    metric_columns[1].metric("Prompts sent", total_prompts)
    metric_columns[2].metric("Decision records", total_records)
    metric_columns[3].metric("Count mismatches", count_mismatches)

    if count_mismatches == 0:
        st.success("Prompt totals in the summary match the detailed decision log.")
    else:
        st.warning("Some prompt totals in the summary do not match the log.")


    # Cohort-level views
    st.divider()
    left_column, right_column = st.columns(2)

    with left_column:
        st.subheader("Prompt count by user")

        prompt_distribution = (
        user_table["prompts_sent"]
        .value_counts()
        .sort_index()
        .rename_axis("prompts_sent")
        .reset_index(name="number_of_users")
        )

        st.bar_chart(
            prompt_distribution.set_index("prompts_sent"),
            height=320,
        )

        st.caption(
            "This chart shows how many users received each number of prompts."
        )

        st.dataframe(
            user_table.sort_values(
                ["prompts_sent", "user_id"],
                ascending=[False, True],
            ),
            use_container_width=True,
            hide_index=True,
        )

    with right_column:
        st.subheader("Why decisions were made")

        reason_counts = (
            log_df["decision_reason"]
            .fillna("missing reason")
            .value_counts()
            .rename_axis("decision_reason")
            .reset_index(name="count")
        )

        st.bar_chart(
            reason_counts.set_index("decision_reason"),
            horizontal=True,
            height=420,
        )

        st.dataframe(
            reason_counts,
            use_container_width=True,
            hide_index=True,
        )


    # Per-user detail view
    st.divider()
    st.subheader("Per-user details")

    available_users = sorted(summary_df["user_id"].dropna().unique().tolist())

    if not available_users:
        st.warning("No users were found in decision_summary.csv.")
        st.stop()

    selected_user = st.selectbox("Choose a user", available_users)

    user_log = log_df[log_df["user_id"] == selected_user].copy()
    user_summary_rows = summary_df[summary_df["user_id"] == selected_user]

    if user_summary_rows.empty:
        st.error("The selected user is missing from decision_summary.csv.")
        st.stop()

    user_summary = user_summary_rows.iloc[0]

    if user_log["decision_reason"].dropna().empty:
        most_common_reason = "No reason recorded"
    else:
        most_common_reason = user_log["decision_reason"].mode().iloc[0]

    user_metric_columns = st.columns(3)
    user_metric_columns[0].metric(
        "Prompts sent",
        int(user_summary["prompts_sent"]),
    )
    user_metric_columns[1].metric("Decision records", len(user_log))
    user_metric_columns[2].metric("Most common reason", most_common_reason)


    detail_left, detail_right = st.columns(2)

    with detail_left:
        st.markdown("### Reason breakdown")

        user_reason_counts = (
            user_log["decision_reason"]
            .fillna("missing reason")
            .value_counts()
            .rename_axis("decision_reason")
            .reset_index(name="count")
        )

        st.bar_chart(
            user_reason_counts.set_index("decision_reason"),
            horizontal=True,
        )

        st.dataframe(
            user_reason_counts,
            use_container_width=True,
            hide_index=True,
        )

    with detail_right:
        st.markdown("### MSSD versus threshold")

        timeline_df = (
            user_log[["timestamp", "observed_mssd", "user_threshold"]]
            .dropna(subset=["timestamp"])
            .set_index("timestamp")
        )

        if timeline_df.empty:
            st.info("No valid timestamps are available for this user.")
        else:
            st.line_chart(timeline_df, height=320)

        st.caption(
            "Observed MSSD represents measured volatility. The threshold is the "
            "personalized comparison value used by the decision engine."
        )


    # Detailed decision history
    st.markdown("### Decision history")

    preferred_history_columns = [
        "timestamp",
        "ema",
        "observed_mssd",
        "user_threshold",
        "send_prompt",
        "decision_reason",
    ]

    history_columns = [
        column for column in preferred_history_columns if column in user_log.columns
    ]

    st.dataframe(
        user_log[history_columns],
        use_container_width=True,
        hide_index=True,
    )


    # Optional raw data
    with st.expander("Show raw input data"):
        st.markdown("#### decision_log")
        st.dataframe(log_df, use_container_width=True, hide_index=True)

        st.markdown("#### decision_summary")
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

