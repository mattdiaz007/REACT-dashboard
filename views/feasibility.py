"""REACT feasibility view."""

from dashboard_core import *
from views.participant_detail import monitoring_participant_roster, summarize_monitor_days


def render_feasibility_view(log_df: pd.DataFrame, data_mode: str, include_demo_devices: bool) -> None:
    """Render paper-ready feasibility results at cohort and participant levels."""
    st.header("Feasibility Results")
    st.caption(
        "EMA completion, prompt-to-response latency, wear-time gaps, and "
        "missingness at cohort and participant levels."
    )

    if data_mode == "Live":
        st.success("LIVE DATA MODE — only live backend fields are displayed. Seed values are never substituted.")
    else:
        st.info("SEED DATA MODE — synthetic/mock data for testing and demonstration only.")

    participant_df, daily_df, metadata = build_feasibility_data(log_df)

    try:
        pipeline_source_df, pipeline_source_name = load_pipeline_source(data_mode)
        pipeline_df, pipeline_metadata = build_pipeline_data(pipeline_source_df)
        if data_mode == "Live":
            pipeline_df = filter_demo_participants(
                pipeline_df,
                include_demo_devices=include_demo_devices,
            )
        pipeline_load_error = None
    except Exception as exc:
        if data_mode == "Live":
            st.error("Live backend data could not be loaded. Seed data was not substituted.")
            st.code(str(exc))
            return
        pipeline_source_name = "seed decision-log fallback"
        pipeline_df, pipeline_metadata = build_pipeline_data(log_df)
        pipeline_df["data_source"] = pipeline_source_name
        pipeline_df["is_seed_data"] = True
        pipeline_load_error = str(exc)

    if participant_df.empty:
        st.warning("No valid participant timestamps are available for feasibility analysis.")
        return

    is_seed_source = bool(
        "is_seed_data" in pipeline_df.columns
        and pipeline_df["is_seed_data"].fillna(False).astype(bool).any()
    )

    st.subheader("Pipeline health")
    st.caption(f"Pipeline source: {pipeline_source_name}")
    if is_seed_source:
        st.info(
            "Seed data is displayed for layout and testing only. Operational stale "
            "flags are disabled until the live backend is selected."
        )

    if pipeline_load_error:
        st.warning(
            "The backend health source could not be loaded, so pipeline monitoring "
            "is temporarily using the decision log."
        )
        st.code(pipeline_load_error)

    if not pipeline_metadata["timestamp_fields_present"]:
        st.info(
            "Pipeline timestamp fields are not present in the current analysis source. "
            "This section will populate automatically when the necessary timestamp "
            "fields are included."
        )
    else:
        last_sync = pipeline_df["last_sync_at"].max()
        last_decision = pipeline_df["decision_made_at"].max()
        last_push = pipeline_df["push_sent_at"].max()
        last_device_receipt = pipeline_df["device_received_at"].max()
        last_backend_receipt = pipeline_df["receipt_reported_at"].max()

        pushed_rows = pipeline_df[
            pipeline_df["push_sent_at"].notna()
        ]

        waiting_count = int(
            (
                pushed_rows["push_sent_at"].notna()
                & pushed_rows["device_received_at"].isna()
            ).sum()
        )

        error_count = int(
            (pipeline_df["pipeline_state"] == "Error").sum()
        )

        if error_count > 0:
            overall_pipeline_status = "Error"
        elif waiting_count > 0:
            overall_pipeline_status = "Waiting"
        elif pd.notna(last_device_receipt):
            overall_pipeline_status = "Healthy"
        elif pd.notna(last_push):
            overall_pipeline_status = "Waiting for receipt"
        elif pd.notna(last_decision):
            overall_pipeline_status = "Waiting for push"
        else:
            overall_pipeline_status = "No activity"

        health_columns = st.columns(5)

        health_columns[0].metric(
            "Last sync",
            format_pipeline_time(last_sync),
        )

        health_columns[1].metric(
            "Last decision",
            format_pipeline_time(last_decision),
        )

        health_columns[2].metric(
            "Last push",
            format_pipeline_time(last_push),
        )

        health_columns[3].metric(
            "Last receipt",
            format_pipeline_time(last_device_receipt),
            help=(
                "The latest device_received_at timestamp reported by "
                "a participant device."
            ),
        )

        health_columns[4].metric(
            "Pipeline status",
            overall_pipeline_status,
            help=(
                f"{waiting_count} delivery or deliveries waiting for a device "
                f"receipt. {error_count} delivery error or errors."
            ),
        )

        st.caption("All pipeline timestamps shown in US Eastern time (America/New_York).")


        if pd.notna(last_backend_receipt):
            st.caption(
                "Latest backend acknowledgment: "
                f"{format_pipeline_time(last_backend_receipt)}"
            )

        # One health row per participant.
        participant_column = (
            "participant_id"
            if "participant_id" in pipeline_df.columns
            else "user_id"
        )

        participant_health = (
            pipeline_df.groupby(participant_column, as_index=False)
            .agg(
                last_sync_at=("last_sync_at", "max"),
                last_push_at=("push_sent_at", "max"),
                last_receipt_at=("device_received_at", "max"),
                last_backend_receipt_at=("receipt_reported_at", "max"),
            )
        )

        # A participant is stale when no sync has been recorded or the most
        # recent sync is more than 24 hours old. Calculate this after grouping
        # so it is based on each participant's latest sync timestamp.
        stale_cutoff = pd.Timestamp.now(tz=LOCAL_TIMEZONE) - pd.Timedelta(hours=24)
        if is_seed_source:
            participant_health["stale"] = pd.NA
        else:
            participant_health["stale"] = (
                participant_health["last_sync_at"].isna()
                | participant_health["last_sync_at"].lt(stale_cutoff)
            )

        def participant_status(row):
            if is_seed_source:
                return "Seed data"
            if row["stale"]:
                return "Stale"
            if pd.notna(row["last_receipt_at"]):
                return "Healthy"
            if pd.notna(row["last_push_at"]):
                return "Waiting for receipt"
            return "No activity"

        participant_health["status"] = participant_health.apply(
            participant_status,
            axis=1,
        )

        participant_health = participant_health.rename(
    columns={
        participant_column: "Participant",
        "last_sync_at": "Last sync",
        "last_push_at": "Last push",
        "last_receipt_at": "Last receipt",
        "last_backend_receipt_at": "Receipt reported",
        "stale": "Stale",
        "status": "Status",
    }
)

        st.markdown("#### Participant health")
        participant_health_display = participant_health.copy()
        if "Participant" in participant_health_display.columns:
            participant_health_display["Participant"] = (
                participant_health_display["Participant"].astype(str).map(
                    lambda participant_id: (
                        f"{participant_id} — {DEMO_PARTICIPANTS[participant_id]}"
                        if participant_id in DEMO_PARTICIPANTS
                        else participant_id
                    )
                )
            )

        timestamp_columns = [
            "Last sync",
            "Last push",
            "Last receipt",
            "Receipt reported",
        ]

        for column in timestamp_columns:
            if column in participant_health_display.columns:
                participant_health_display[column] = participant_health_display[column].apply(
                    lambda value: (
                        pd.Timestamp(value).strftime("%b %d, %I:%M:%S %p ET")
                        if pd.notna(value)
                        else "Not recorded"
                    )
                )

        st.dataframe(
            participant_health_display,
            use_container_width=True,
            hide_index=True,
        )

    st.divider()
    st.subheader("Delivery latency")

    completed_deliveries = pipeline_df.dropna(
        subset=["decision_made_at", "device_received_at"]
    ).copy()

    if completed_deliveries.empty:
        st.info(
            "No completed decision-to-device deliveries have been recorded yet. "
            "The first pipeline test will appear here once both "
            "`decision_made_at` and `device_received_at` are available."
        )
    else:
        end_to_end_values = completed_deliveries[
            "end_to_end_seconds"
        ].dropna()

        backend_queue_values = completed_deliveries[
            "backend_queue_seconds"
        ].dropna()

        push_delivery_values = completed_deliveries[
            "push_delivery_seconds"
        ].dropna()

        latest_delivery = completed_deliveries.sort_values(
            "device_received_at"
        ).iloc[-1]

        latency_columns = st.columns(5)

        latency_columns[0].metric(
            "Latest end-to-end",
            format_latency(latest_delivery["end_to_end_seconds"]),
            help="Time from decision_made_at to device_received_at.",
        )

        latency_columns[1].metric(
            "Median end-to-end",
            format_latency(end_to_end_values.median()),
        )

        latency_columns[2].metric(
            "95th percentile",
            format_latency(end_to_end_values.quantile(0.95)),
        )

        latency_columns[3].metric(
            "Median backend queue",
            format_latency(backend_queue_values.median()),
            help="Time from decision_made_at to push_sent_at.",
        )

        latency_columns[4].metric(
            "Median push delivery",
            format_latency(push_delivery_values.median()),
            help="Time from push_sent_at to device_received_at.",
        )

        st.caption(
            f"{len(completed_deliveries):,} completed pipeline delivery "
            f"{'test' if len(completed_deliveries) == 1 else 'tests'} recorded."
        )

        latency_display = completed_deliveries.copy()

        preferred_columns = [
            participant_column,
            "decision_made_at",
            "push_sent_at",
            "device_received_at",
            "receipt_reported_at",
            "backend_queue_seconds",
            "push_delivery_seconds",
            "end_to_end_seconds",
            "pipeline_state",
        ]

        latency_display = latency_display[
            [
                column
                for column in preferred_columns
                if column in latency_display.columns
            ]
        ].sort_values(
            "device_received_at",
            ascending=False,
        )

        latency_display = latency_display.rename(
            columns={
                participant_column: "Participant",
                "decision_made_at": "Decision made",
                "push_sent_at": "Push sent",
                "device_received_at": "Device received",
                "receipt_reported_at": "Receipt reported",
                "backend_queue_seconds": "Backend queue (sec)",
                "push_delivery_seconds": "Push delivery (sec)",
                "end_to_end_seconds": "End-to-end (sec)",
                "pipeline_state": "Status",
            }
        )

        if "Participant" in latency_display.columns:
            latency_display["Participant"] = latency_display["Participant"].astype(str).map(
                lambda participant_id: (
                    f"{participant_id} — {DEMO_PARTICIPANTS[participant_id]}"
                    if participant_id in DEMO_PARTICIPANTS
                    else participant_id
                )
            )

        st.dataframe(
            latency_display,
            use_container_width=True,
            hide_index=True,
        )

    st.divider()

    filter_columns = st.columns([2, 2, 3])
    available_users = sorted(participant_df["user_id"].tolist())
    selected_participant = filter_columns[0].selectbox(
        "Participant detail", ["All participants"] + available_users
    )

    min_date = pd.to_datetime(log_df["timestamp"], errors="coerce").min()
    max_date = pd.to_datetime(log_df["timestamp"], errors="coerce").max()
    filter_columns[1].text_input(
        "Study period",
        value=(
            f"{min_date:%b %d, %Y} – {max_date:%b %d, %Y}"
            if pd.notna(min_date) and pd.notna(max_date)
            else "Unavailable"
        ),
        disabled=True,
    )
    filter_columns[2].info(
        "Metrics update automatically when uploaded analysis files use the supported columns."
    )

    total_scheduled = int(participant_df["scheduled_emas"].sum())
    total_completed = int(participant_df["completed_emas"].sum())
    cohort_response_rate = total_completed / total_scheduled if total_scheduled else 0.0
    cohort_missingness = 1.0 - cohort_response_rate
    participant_latencies = pd.to_numeric(
        participant_df["median_latency_minutes"], errors="coerce"
    ).dropna()
    cohort_latency = participant_latencies.median() if not participant_latencies.empty else pd.NA
    total_wear_gaps = int(participant_df["wear_gaps"].sum())

    metric_columns = st.columns(4)
    metric_columns[0].metric(
        "EMA response rate",
        f"{cohort_response_rate:.1%}",
        help=f"{total_completed:,} completed of {total_scheduled:,} scheduled EMA assessments.",
    )
    metric_columns[1].metric(
        "Median response latency",
        f"{cohort_latency:.0f} min" if pd.notna(cohort_latency) else "Not captured",
        help="Requires a latency field or a response timestamp in the uploaded decision log.",
    )
    metric_columns[2].metric(
        "Wear-time gaps",
        f"{total_wear_gaps:,}",
        help=f"Current basis: {metadata['wear_gap_basis']}.",
    )
    metric_columns[3].metric(
        "Overall missingness",
        f"{cohort_missingness:.1%}",
        help="Share of scheduled EMA rows with no EMA response.",
    )

    if not metadata["latency_available"]:
        st.warning(
            "Prompt-to-response latency is not present in the current synthetic data. "
            "Add `response_latency_minutes` or `response_timestamp` to calculate it."
        )
    if metadata["wear_gap_basis"] != "wearable telemetry":
        st.info(
            "Wearable timestamps are not present, so the current wear-gap value is an "
            "EMA observation-gap proxy. Add `wearable_timestamp`, `sensor_timestamp`, "
            "or `device_timestamp` for true wear-time gaps."
        )

    st.subheader("Feasibility over time")
    trend_df = daily_df.copy()
    trend_df["date"] = pd.to_datetime(trend_df["date"])
    trend_df = trend_df.set_index("date")[["response_rate", "missingness"]]
    trend_df = trend_df.rename(
        columns={"response_rate": "EMA response rate", "missingness": "Missingness"}
    )
    st.line_chart(trend_df, height=300)
    st.caption("Daily proportions range from 0 to 1; hover for exact values.")

    st.subheader("Participant-level results")
    display_df = participant_df.copy()
    display_df["EMA response rate"] = display_df["response_rate"].map(lambda value: f"{value:.1%}")
    display_df["Median latency"] = display_df["median_latency_minutes"].map(
        lambda value: f"{value:.0f} min" if pd.notna(value) else "Not captured"
    )
    display_df["Missingness"] = display_df["missingness"].map(lambda value: f"{value:.1%}")
    display_df = display_df.rename(
        columns={
            "user_id": "Participant",
            "scheduled_emas": "Scheduled EMAs",
            "completed_emas": "Completed EMAs",
            "wear_gaps": "Wear gaps",
        }
    )[
        [
            "Participant",
            "Scheduled EMAs",
            "Completed EMAs",
            "EMA response rate",
            "Median latency",
            "Wear gaps",
            "Missingness",
        ]
    ]
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.download_button(
        label="Download feasibility table as CSV",
        data=display_df.to_csv(index=False).encode("utf-8"),
        file_name="react_feasibility_table.csv",
        mime="text/csv",
        key="download_feasibility_csv",
    )

    detail_user = available_users[0] if selected_participant == "All participants" else selected_participant
    st.subheader(f"Participant detail: {detail_user}")
    user_data = log_df[log_df["user_id"] == detail_user].copy().sort_values("timestamp")
    user_data["date"] = user_data["timestamp"].dt.date
    user_data["EMA completed"] = user_data["ema"].notna().astype(int)
    user_daily = user_data.groupby("date").agg(
        scheduled=("EMA completed", "size"), completed=("EMA completed", "sum")
    )
    user_daily["Response rate"] = user_daily["completed"] / user_daily["scheduled"]
    user_daily["Missingness"] = 1.0 - user_daily["Response rate"]

    detail_left, detail_right = st.columns(2)
    with detail_left:
        st.markdown("#### Response and missingness")
        st.line_chart(user_daily[["Response rate", "Missingness"]], height=280)
    with detail_right:
        st.markdown("#### Observation timeline")
        observation_timeline = user_data.set_index("timestamp")[["EMA completed"]]
        st.bar_chart(observation_timeline, height=280)
        st.caption("1 = completed EMA; 0 = missing EMA.")

    with st.expander("Metric definitions and supported columns"):
        st.markdown(
            """
- **EMA response rate:** completed EMA values divided by scheduled EMA rows.
- **Response latency:** median minutes from prompt to response; supports `response_latency_minutes`, `latency_minutes`, `prompt_to_response_minutes`, or a response timestamp.
- **Wear-time gaps:** gaps above two hours when wearable timestamps exist. The current fallback is explicitly labeled as an EMA observation-gap proxy.
- **Missingness:** scheduled EMA rows without an EMA value, summarized overall and by day.
            """
        )


def render_live_feasibility_view(include_demo_devices: bool) -> None:
    """Use only measurable live scheduled-check-in completion from existing APIs."""
    st.header("Feasibility Results")
    st.caption("Live scheduled-check-in completion. Wear-time and HRV are not available from this monitoring feed.")
    try:
        roster = monitoring_participant_roster(include_demo_devices)
    except Exception as exc:
        st.error(f"Live roster unavailable: {exc}")
        return
    yesterday = pd.Timestamp.now(tz=LOCAL_TIMEZONE).date() - pd.Timedelta(days=1)
    records = []
    for p in sorted(roster, key=lambda r: str(r.get("participant_id"))):
        try:
            data = summarize_monitor_days(load_monitor_timeline(int(p["user_id"]), local_date=yesterday.isoformat(), days=7))
            vals = list(data["by_date"].values())
            if len(vals) != 7 or any(v["completed"] is None for v in vals):
                raise ValueError("Incomplete 7-day response")
            total = sum(v["completed"] for v in vals)
            scheduled = sum(v["scheduled"] for v in vals)
            records.append({"Participant": p["participant_id"], "Completed": total, "Scheduled": scheduled,
                            "Completion %": f"{total / scheduled:.1%}" if scheduled else "N/A"})
        except Exception:
            records.append({"Participant": p["participant_id"], "Completed": None, "Scheduled": None, "Completion %": "Unavailable"})
    frame = pd.DataFrame(records)
    if frame.empty:
        st.info("No RS participants returned.")
        return
    valid = frame.dropna(subset=["Completed", "Scheduled"])
    total, scheduled = int(valid["Completed"].sum()), int(valid["Scheduled"].sum())
    if len(valid) == len(frame) and scheduled:
        st.metric("7-day EMA completion", f"{total / scheduled:.1%}", help=f"{total} of {scheduled} scheduled slots")
    else:
        st.metric("7-day EMA completion", "Incomplete data")
        st.caption(f"{len(valid)} of {len(frame)} participants returned complete 7-day data.")
    st.dataframe(frame, use_container_width=True, hide_index=True)
    st.caption("This measures completed scheduled slots, not notification delivery. Wearable sync, wear-time gaps and HRV are intentionally omitted.")

