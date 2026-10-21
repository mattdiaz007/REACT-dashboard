"""REACT participant detail view."""

from dashboard_core import *


def monitoring_participant_roster(include_demo_devices: bool) -> list[dict]:
    """Use the monitoring grid's study-ID roster, not the legacy wearable feed."""
    payload = load_monitor_grid(metric="ema_scheduled_n", phase="all")
    rows = filter_monitor_participants(payload.get("rows", []), include_demo_devices)
    return [r for r in rows if study_id_group(r.get("participant_id")) == "RS" and r.get("user_id") is not None]


@st.cache_data(ttl=60)
def monitoring_token_map() -> dict[str, str]:
    """Return backend token registration, never the token value itself.

    Missing response / backend errors are Unknown, not Missing.
    """
    result = {}
    try:
        health, _ = load_backend_health(use_mock=False)
        if not {"user_id", "has_push_token"}.issubset(health.columns):
            return result
        for _, row in health.iterrows():
            uid, registered = row.get("user_id"), row.get("has_push_token")
            if pd.isna(uid) or pd.isna(registered):
                continue
            normalized = str(registered).strip().lower()
            if normalized in ("true", "1"):
                result[str(uid)] = "Present"
            elif normalized in ("false", "0"):
                result[str(uid)] = "Missing"
    except Exception:
        # Never claim a token is missing when the status API was unavailable.
        return {}
    return result


def summarize_monitor_days(payload: dict, end_date=None) -> dict:
    """Count unique completed scheduled slots and direct decision events."""
    days = timeline_days(payload)
    by_date = {}
    last_completed = pd.NaT
    decisions = []
    for day in days:
        date = pd.to_datetime(day.get("local_date"), errors="coerce")
        if pd.isna(date):
            continue
        local_date = date.date()
        slots = day.get("slots")
        completed = set()
        for event in day.get("events") or []:
            if event.get("kind") == "ema" and event.get("ema_type") == "scheduled_check_in" and str(event.get("status")).lower() == "completed":
                at = parse_monitor_time(event.get("at"))
                if pd.notna(at) and (pd.isna(last_completed) or at > last_completed):
                    last_completed = at
                if event.get("slot") is not None:
                    completed.add(str(event["slot"]))
            elif event.get("kind") == "decision":
                item = dict(event)
                item["local_date"] = local_date
                decisions.append(item)
        by_date[local_date] = {
            "completed": len(completed) if isinstance(slots, list) else None,
            "scheduled": len(slots) if isinstance(slots, list) else None,
        }
    return {"by_date": by_date, "last_completed": last_completed, "decisions": decisions}


def render_participant_detail_view(log_df, summary_df, data_mode, include_demo_devices):
    if data_mode != "Live":
        render_participant_detail_seed_view(log_df, summary_df, data_mode, include_demo_devices)
        return

    st.header("Participant Detail")
    st.caption("Live check-ins, decisions, prompt handoffs and token registration. Times are Eastern.")
    st.success("LIVE DATA MODE — monitoring API only; no seed substitution.")
    try:
        roster = monitoring_participant_roster(include_demo_devices)
    except Exception as exc:
        st.error(f"Participant roster unavailable: {exc}")
        return
    if not roster:
        st.info("No RS participants returned by the monitoring API.")
        return
    roster = sorted(roster, key=lambda r: str(r.get("participant_id")))
    labels = {f"{r['participant_id']} (user {r['user_id']})": r for r in roster}
    label = st.selectbox("Choose participant", list(labels), key="live_participant_detail")
    item = labels[label]
    user_id = int(item["user_id"])
    today = pd.Timestamp.now(tz=LOCAL_TIMEZONE).date()
    yesterday = today - pd.Timedelta(days=1)
    tokens = monitoring_token_map()
    token = tokens.get(str(user_id), "Unknown")
    try:
        payload = load_monitor_timeline(user_id=user_id, local_date=today.isoformat(), days=14)
    except Exception as exc:
        st.error(f"Participant timeline unavailable: {exc}")
        st.metric("Push token registered", token)
        return
    detail = summarize_monitor_days(payload)
    last = detail["last_completed"]
    cutoff = pd.Timestamp.now(tz=LOCAL_TIMEZONE) - pd.Timedelta(hours=24)
    stale = pd.isna(last) or last < cutoff
    yesterday_data = detail["by_date"].get(yesterday)
    today_data = detail["by_date"].get(today)
    yesterday_label = (f"{yesterday_data['completed']}/{yesterday_data['scheduled']}" 
                       if yesterday_data and yesterday_data["completed"] is not None else "Unavailable")
    recent_decisions = detail["decisions"]
    yesterday_decisions = [d for d in recent_decisions if d["local_date"] == yesterday]
    distress = sum(d.get("distress_flag_active") is True for d in yesterday_decisions)
    distress_display = distress if yesterday_data is not None else "Unavailable"
    if token == "Missing":
        st.error("NEEDS ATTENTION — no push token registered. Check notification registration.")
    elif stale:
        st.warning("NEEDS ATTENTION — no completed scheduled check-in in the past 24 hours.")
    else:
        st.success("Recent scheduled check-in recorded. Review any delivery errors below.")
    cols = st.columns(4)
    cols[0].metric("Last check-in", format_pipeline_time(last))
    cols[1].metric("Yesterday EMA", yesterday_label)
    cols[2].metric("Push token", token)
    cols[3].metric("Distress-active decisions yesterday", distress_display)
    st.caption("Distress-active decisions are not unique flags raised. A registered token does not prove notification delivery.")
    if today_data and today_data["scheduled"] is not None:
        st.caption(f"Today: {today_data['completed']} / {today_data['scheduled']} scheduled check-ins completed.")
    st.subheader("Recent decision and prompt activity")
    if recent_decisions:
        rows = []
        for d in sorted(recent_decisions, key=lambda e: str(e.get("at") or ""), reverse=True)[:30]:
            push = parse_monitor_time(d.get("push_sent_at"))
            receipt = parse_monitor_time(d.get("device_received_at"))
            failure = str(d.get("delivery_error") or "").strip()
            delivery = ("Failed" if failure else "Sent (receipt not required)" if pd.notna(push)
                        else "Not sent")
            rows.append({"Decision": format_pipeline_time(parse_monitor_time(d.get("at"))),
                         "Reason": decision_activity_reason(d), "Push": format_pipeline_time(push),
                         "Opened/receipt": format_pipeline_time(receipt), "Delivery": delivery,
                         "Error": failure})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        n_fail = sum(bool(str(d.get("delivery_error") or "").strip()) for d in recent_decisions)
        if n_fail:
            st.warning(f"{n_fail} decision event(s) contain a recorded delivery error in the available 14-day timeline.")
        st.caption("Device receipt generally indicates the prompt was opened or the app was already open. No receipt alone is not an error.")
    else:
        st.info("No decision activity returned in the available 14-day timeline.")
    st.subheader("Recent scheduled check-ins")
    checkin_rows = [{"Date (ET)": day.isoformat(), "Completed": val["completed"], "Scheduled slots": val["scheduled"]}
                    for day, val in sorted(detail["by_date"].items(), reverse=True)]
    st.dataframe(pd.DataFrame(checkin_rows), use_container_width=True, hide_index=True)
    st.caption("History is limited to up to 14 days returned by the monitoring API; this is not the participant's entire study history.")


def render_participant_detail_seed_view(log_df: pd.DataFrame, summary_df: pd.DataFrame, data_mode: str, include_demo_devices: bool) -> None:
    """Render a fast pre-visit status screen for one participant."""
    st.header("Participant Detail")
    st.caption(
        "Five-second pre-visit check: device syncing, check-ins, and prompt delivery. "
        "Times are Eastern."
    )

    if data_mode == "Live":
        st.success(
            "LIVE DATA MODE — only live backend fields are displayed. "
            "Seed values are never substituted."
        )
    else:
        st.info("SEED DATA MODE — synthetic/mock data for testing and demonstration only.")

    try:
        source_df, source_name = load_pipeline_source(data_mode)
        pipeline_df, _ = build_pipeline_data(source_df)
        if data_mode == "Live":
            pipeline_df = filter_demo_participants(
                pipeline_df,
                include_demo_devices=include_demo_devices,
            )
        load_error = None
    except Exception as exc:
        if data_mode == "Live":
            st.error("Live backend data could not be loaded. Seed data was not substituted.")
            st.code(str(exc))
            return
        source_name = "seed decision-log fallback"
        pipeline_df, _ = build_pipeline_data(log_df)
        pipeline_df["data_source"] = source_name
        pipeline_df["is_seed_data"] = True
        load_error = str(exc)

    is_seed_source = bool(
        "is_seed_data" in pipeline_df.columns
        and pipeline_df["is_seed_data"].fillna(False).astype(bool).any()
    )

    if load_error:
        st.warning(
            "The live backend could not be loaded, so operational fields are using "
            "seed/fallback data."
        )
        st.code(load_error)

    # Build participant options from backend identifiers first. Keep user_id
    # attached when available so seed EMA/decision rows can be linked correctly.
    participant_options = []
    seen_option_keys = set()

    if not pipeline_df.empty:
        if "participant_id" in pipeline_df.columns:
            pair_columns = ["participant_id"]
            if "user_id" in pipeline_df.columns:
                pair_columns.append("user_id")

            pairs = pipeline_df[pair_columns].drop_duplicates()
            for _, row in pairs.iterrows():
                participant_id = row.get("participant_id")
                user_id = row.get("user_id") if "user_id" in pair_columns else pd.NA

                if pd.isna(participant_id):
                    continue

                key = ("participant", str(participant_id))
                if key in seen_option_keys:
                    continue
                seen_option_keys.add(key)

                participant_id_text = str(participant_id)
                label = (
                    f"{participant_id_text} — {DEMO_PARTICIPANTS[participant_id_text]}"
                    if participant_id_text in DEMO_PARTICIPANTS
                    else participant_id_text
                )
                if pd.notna(user_id):
                    label += f" (user {user_id})"

                participant_options.append(
                    {
                        "label": label,
                        "participant_id": participant_id,
                        "user_id": user_id,
                    }
                )
        elif "user_id" in pipeline_df.columns:
            for user_id in pipeline_df["user_id"].dropna().drop_duplicates().tolist():
                key = ("user", str(user_id))
                if key in seen_option_keys:
                    continue
                seen_option_keys.add(key)
                participant_options.append(
                    {
                        "label": f"User {user_id}",
                        "participant_id": pd.NA,
                        "user_id": user_id,
                    }
                )

    # Decision-log-only participants are seed/analysis records and must never
    # appear as live participants.
    if data_mode == "Seed":
        represented_users = {
            str(option["user_id"])
            for option in participant_options
            if pd.notna(option["user_id"])
        }
        for user_id in log_df["user_id"].dropna().drop_duplicates().tolist():
            if str(user_id) in represented_users:
                continue
            participant_options.append(
                {
                    "label": f"User {user_id} (analysis only)",
                    "participant_id": pd.NA,
                    "user_id": user_id,
                }
            )

        try:
            hrv_df = load_default_hrv_data()
        except ValueError as exc:
            hrv_df = pd.DataFrame()
            st.error(str(exc))

        for user_id in hrv_df.get("user_id", pd.Series(dtype=object)).dropna().drop_duplicates():
            if str(user_id) not in represented_users:
                participant_options.append(
                    {
                        "label": f"User {user_id} (HRV data only)",
                        "participant_id": pd.NA,
                        "user_id": user_id,
                    }
                )

    if not participant_options:
        st.warning("No participants are available in the current data sources.")
        return

    participant_options = sorted(
        participant_options,
        key=lambda option: option["label"].lower(),
    )

    selected_label = st.selectbox(
        "Choose participant",
        [option["label"] for option in participant_options],
    )
    selected = next(
        option for option in participant_options if option["label"] == selected_label
    )
    selected_participant_id = selected["participant_id"]
    selected_user_id = selected["user_id"]

    # Filter backend/pipeline rows for the selected participant.
    participant_pipeline = pipeline_df.iloc[0:0].copy()
    if pd.notna(selected_participant_id) and "participant_id" in pipeline_df.columns:
        participant_pipeline = pipeline_df[
            pipeline_df["participant_id"].astype(str) == str(selected_participant_id)
        ].copy()
    elif pd.notna(selected_user_id) and "user_id" in pipeline_df.columns:
        participant_pipeline = pipeline_df[
            pipeline_df["user_id"].astype(str) == str(selected_user_id)
        ].copy()

    # Recover user_id when the selected backend row carries one.
    if (
        pd.isna(selected_user_id)
        and not participant_pipeline.empty
        and "user_id" in participant_pipeline.columns
    ):
        linked_users = participant_pipeline["user_id"].dropna()
        if not linked_users.empty:
            selected_user_id = linked_users.iloc[0]

    participant_log = log_df.iloc[0:0].copy()
    if data_mode == "Seed" and pd.notna(selected_user_id):
        participant_log = log_df[
            log_df["user_id"].astype(str) == str(selected_user_id)
        ].copy().sort_values("timestamp")

    now_et = pd.Timestamp.now(tz=LOCAL_TIMEZONE)
    today = now_et.date()
    yesterday = (now_et - pd.Timedelta(days=1)).date()
    stale_cutoff = now_et - pd.Timedelta(hours=24)

    last_sync = (
        participant_pipeline["last_sync_at"].max()
        if "last_sync_at" in participant_pipeline
        else pd.NaT
    )
    last_push = (
        participant_pipeline["push_sent_at"].max()
        if "push_sent_at" in participant_pipeline
        else pd.NaT
    )
    last_receipt = (
        participant_pipeline["device_received_at"].max()
        if "device_received_at" in participant_pipeline
        else pd.NaT
    )
    last_backend_receipt = (
        participant_pipeline["receipt_reported_at"].max()
        if "receipt_reported_at" in participant_pipeline
        else pd.NaT
    )

    stale = False if is_seed_source else (pd.isna(last_sync) or last_sync < stale_cutoff)

    # Evaluate delivery state using the participant's latest delivery event.
    latest_delivery = None
    latest_event_waiting = False
    latest_event_error = False

    if not participant_pipeline.empty:
        event_sort_time = participant_pipeline["push_sent_at"].copy()
        event_sort_time = event_sort_time.fillna(participant_pipeline["decision_made_at"])
        event_rows = participant_pipeline.assign(
            _event_sort_time=event_sort_time
        ).sort_values(
            "_event_sort_time",
            ascending=False,
            na_position="last",
        )
        latest_delivery = event_rows.iloc[0]
        latest_event_waiting = (
            pd.notna(latest_delivery.get("push_sent_at"))
            and pd.isna(latest_delivery.get("device_received_at"))
        )
        latest_event_error = latest_delivery.get("pipeline_state") == "Error"

    # Calculate participant EMA/check-in metrics from seed analysis data only.
    today_rate = None
    yesterday_rate = None
    overall_response_rate = None
    prompts_sent = 0
    today_completed = today_scheduled = 0
    yesterday_completed = yesterday_scheduled = 0

    if not participant_log.empty:
        participant_log["local_date"] = participant_log["timestamp"].dt.date
        participant_log["ema_completed"] = (
            participant_log["ema"].notna()
            if "ema" in participant_log.columns
            else False
        )

        def completion_rate_for(day):
            rows = participant_log[participant_log["local_date"] == day]
            if rows.empty:
                return None, 0, 0
            scheduled = len(rows)
            completed = int(rows["ema_completed"].sum())
            return completed / scheduled, completed, scheduled

        today_rate, today_completed, today_scheduled = completion_rate_for(today)
        yesterday_rate, yesterday_completed, yesterday_scheduled = completion_rate_for(yesterday)
        overall_response_rate = float(participant_log["ema_completed"].mean())
        prompts_sent = int(participant_log["send_prompt"].sum())

    # Five-second quick-look statuses.
    if latest_event_error:
        overall_status = "NEEDS ATTENTION"
        overall_message = "A delivery error was recorded for this participant."
    elif stale:
        overall_status = "NEEDS ATTENTION"
        overall_message = "The participant has not synced in the past 24 hours."
    elif latest_event_waiting:
        overall_status = "CHECK NEEDED"
        overall_message = "The latest push is still waiting for a device receipt."
    elif pd.notna(last_sync) or pd.notna(last_receipt):
        overall_status = "ALL SYSTEMS OK"
        overall_message = "No current operational issues were detected."
    elif is_seed_source:
        overall_status = "SEED DATA"
        overall_message = "Synthetic operational data is displayed for testing."
    else:
        overall_status = "NO ACTIVITY"
        overall_message = "No operational activity has been recorded."

    if is_seed_source:
        sync_status = "Demo"
    elif stale:
        sync_status = "Stale"
    elif pd.notna(last_sync):
        sync_status = "OK"
    else:
        sync_status = "Missing"

    if latest_event_error:
        prompt_status = "Error"
    elif latest_event_waiting:
        prompt_status = "Waiting"
    elif pd.notna(last_receipt):
        prompt_status = "OK"
    elif pd.notna(last_push):
        prompt_status = "No receipt"
    else:
        prompt_status = "No recent prompt"

    if data_mode == "Live":
        checkin_status = "Not live yet"
        checkin_detail = "Live check-in feed is not connected yet"
    elif today_rate is None:
        checkin_status = "No rows"
        checkin_detail = "No scheduled check-ins today"
    elif today_rate >= 0.5:
        checkin_status = "OK"
        checkin_detail = f"{today_completed}/{today_scheduled} completed today"
    else:
        checkin_status = "Behind"
        checkin_detail = f"{today_completed}/{today_scheduled} completed today"

    # Overall status banner first: this is the primary five-second read.
    if overall_status == "ALL SYSTEMS OK":
        st.success(f"### 🟢 {overall_status}\n{overall_message}")
    elif overall_status == "CHECK NEEDED":
        st.warning(f"### 🟡 {overall_status}\n{overall_message}")
    elif overall_status == "NEEDS ATTENTION":
        st.error(f"### 🔴 {overall_status}\n{overall_message}")
    else:
        st.info(f"### {overall_status}\n{overall_message}")

    st.markdown("### Current status")
    quick_columns = st.columns(3)

    quick_columns[0].metric(
        "Device syncing",
        sync_status,
        help="A problem means no live sync has been recorded in the past 24 hours.",
    )
    quick_columns[0].caption(f"Last sync: {format_pipeline_time(last_sync)}")

    quick_columns[1].metric(
        "Check-ins",
        checkin_status,
        help=checkin_detail,
    )
    quick_columns[1].caption(checkin_detail)

    quick_columns[2].metric(
        "Prompts arriving",
        prompt_status,
        help="Based on whether the latest push reached the participant device.",
    )
    if pd.notna(last_receipt):
        quick_columns[2].caption(f"Last receipt: {format_pipeline_time(last_receipt)}")
    elif pd.notna(last_push):
        quick_columns[2].caption(f"Last push: {format_pipeline_time(last_push)}")
    else:
        quick_columns[2].caption("No prompt has been sent recently")

    actions = []
    if not is_seed_source and stale:
        actions.append(
            "Check the participant's device connection and confirm that syncing "
            "is restored before the visit ends."
        )
    if latest_event_waiting:
        actions.append(
            "Confirm the participant's device has network access and verify whether "
            "the pending prompt arrives."
        )
    if latest_event_error:
        actions.append(
            "Review the latest delivery error and notify the technical team if the "
            "problem persists."
        )
    if yesterday_rate is not None and yesterday_rate < 0.5:
        actions.append(
            f"Yesterday's check-in completion was {yesterday_rate:.0%}. "
            "Confirm with the participant whether check-ins are being received."
        )

    if actions:
        st.markdown("#### Staff action")
        for action in actions:
            st.warning(action)

    st.markdown("#### Latest activity")
    activity_columns = st.columns(3)
    activity_columns[0].metric("Last sync", format_pipeline_time(last_sync))
    activity_columns[1].metric("Last push", format_pipeline_time(last_push))
    activity_columns[2].metric("Last receipt", format_pipeline_time(last_receipt))
    st.caption("Operational timestamps are shown in US Eastern time (America/New_York).")

    with st.expander("Recent operational activity"):
        if participant_pipeline.empty:
            st.info("No backend operational records are available for this participant.")
        else:
            recent = participant_pipeline.copy()
            recent["_event_sort_time"] = recent["decision_made_at"].fillna(
                recent["push_sent_at"]
            )
            recent = recent.sort_values(
                "_event_sort_time",
                ascending=False,
                na_position="last",
            ).head(20)

            recent_columns = [
                column
                for column in [
                    "decision_made_at",
                    "push_sent_at",
                    "device_received_at",
                    "receipt_reported_at",
                    "end_to_end_seconds",
                    "pipeline_state",
                ]
                if column in recent.columns
            ]
            recent_display = recent[recent_columns].copy()

            for column in [
                "decision_made_at",
                "push_sent_at",
                "device_received_at",
                "receipt_reported_at",
            ]:
                if column in recent_display.columns:
                    recent_display[column] = recent_display[column].apply(
                        format_pipeline_time
                    )

            if "end_to_end_seconds" in recent_display.columns:
                recent_display["end_to_end_seconds"] = recent_display[
                    "end_to_end_seconds"
                ].apply(format_latency)

            recent_display = recent_display.rename(
                columns={
                    "decision_made_at": "Decision made",
                    "push_sent_at": "Push sent",
                    "device_received_at": "Device received",
                    "receipt_reported_at": "Receipt reported",
                    "end_to_end_seconds": "End-to-end",
                    "pipeline_state": "Status",
                }
            )
            st.dataframe(recent_display, use_container_width=True, hide_index=True)

            if pd.notna(last_backend_receipt):
                st.caption(
                    "Latest backend acknowledgment: "
                    f"{format_pipeline_time(last_backend_receipt)}"
                )

    with st.expander("Participant study history"):
        if participant_log.empty:
            if data_mode == "Live":
                st.info(
                    "Live decision/EMA history is not connected yet. Seed history is "
                    "intentionally hidden while Live mode is selected."
                )
            else:
                st.info(
                    "No decision/EMA rows are linked to this participant in the current "
                    "seed analysis source. Operational status above is still valid."
                )
        else:
            if (
                "decision_reason" in participant_log.columns
                and not participant_log["decision_reason"].dropna().empty
            ):
                most_common_reason = participant_log["decision_reason"].mode().iloc[0]
            else:
                most_common_reason = "No reason recorded"

            history_metrics = st.columns(4)
            history_metrics[0].metric("Decision records", len(participant_log))
            history_metrics[1].metric("Prompts in log", prompts_sent)
            history_metrics[2].metric(
                "Overall EMA response",
                f"{overall_response_rate:.0%}"
                if overall_response_rate is not None
                else "Not available",
            )
            history_metrics[3].metric("Most common reason", most_common_reason)

            preferred_history_columns = [
                "timestamp",
                "ema",
                "observed_mssd",
                "user_threshold",
                "send_prompt",
                "decision_reason",
            ]
            history_columns = [
                column
                for column in preferred_history_columns
                if column in participant_log.columns
            ]
            history_display = participant_log[history_columns].sort_values(
                "timestamp",
                ascending=False,
            ).head(50).copy()

            if "timestamp" in history_display.columns:
                history_display["timestamp"] = history_display["timestamp"].apply(
                    format_pipeline_time
                )

            st.dataframe(history_display, use_container_width=True, hide_index=True)

    st.markdown("#### Heart-rate variability")
    try:
        hrv_df = load_default_hrv_data()
    except ValueError as exc:
        hrv_df = pd.DataFrame()
        st.error(str(exc))

    if hrv_df.empty:
        st.info("No HRV series is available. Add data/hrv_5min_series.csv to display it.")
    elif pd.isna(selected_user_id):
        st.info("HRV cannot be linked because this participant has no user ID.")
    else:
        user_hrv = hrv_df[hrv_df["user_id"].astype(str) == str(selected_user_id)].copy()
        user_hrv = user_hrv.dropna(subset=["interval_start", "rmssd"])
        if user_hrv.empty:
            st.info(f"No HRV observations are available for {selected_user_id}.")
        else:
            hrv_chart = user_hrv.set_index("interval_start")[["rmssd"]].rename(
                columns={"rmssd": "RMSSD (ms)"}
            )
            if "baseline" in user_hrv.columns and user_hrv["baseline"].notna().any():
                hrv_chart["Baseline (ms)"] = user_hrv.set_index("interval_start")["baseline"]
            st.line_chart(hrv_chart, height=300)

            latest_hrv = user_hrv.iloc[-1]
            hrv_metrics = st.columns(3)
            hrv_metrics[0].metric("Latest RMSSD", f"{latest_hrv['rmssd']:.1f} ms")
            hrv_metrics[1].metric(
                "Latest HRV class",
                str(latest_hrv.get("hrv_class") or "Not classified"),
            )
            hrv_metrics[2].metric("Intervals shown", f"{len(user_hrv):,}")
            st.caption(
                "HRV is RMSSD from five-minute intervals. The baseline and class come "
                "from the supplied synthetic HRV export; this section is descriptive "
                "and does not change decision-engine results."
            )

