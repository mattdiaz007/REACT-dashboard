"""REACT weekly summary view."""

from dashboard_core import *
from views.participant_detail import monitoring_participant_roster, monitoring_token_map, summarize_monitor_days


def render_weekly_summary_view(log_df, data_mode, include_demo_devices):
    if data_mode != "Live":
        render_weekly_summary_seed_view(log_df, data_mode, include_demo_devices)
        return
    st.header("Weekly Summary")
    st.caption("Live 7-day scheduled check-in activity from the monitoring timeline API. Times are Eastern.")
    today = pd.Timestamp.now(tz=LOCAL_TIMEZONE).date()
    end_date = today - pd.Timedelta(days=1)
    start_date = end_date - pd.Timedelta(days=6)
    st.caption(f"Reporting window: {start_date:%b %d}–{end_date:%b %d, %Y}")
    try:
        roster = monitoring_participant_roster(include_demo_devices)
    except Exception as exc:
        st.error(f"Participant roster unavailable: {exc}")
        return
    tokens = monitoring_token_map()
    rows = []
    for r in sorted(roster, key=lambda x: x.get("participant_id", "")):
        pid, uid = str(r["participant_id"]), int(r["user_id"])
        try:
            payload = load_monitor_timeline(user_id=uid, local_date=end_date.isoformat(), days=7)
            summary = summarize_monitor_days(payload)
            values = [v for date, v in summary["by_date"].items() if start_date <= date <= end_date]
            valid = [v for v in values if v["completed"] is not None]
            completed = sum(v["completed"] for v in valid)
            scheduled = sum(v["scheduled"] for v in valid)
            ema = f"{completed}/{scheduled}" if valid and len(valid) == 7 else "Partial / unavailable"
            decisions = [d for d in summary["decisions"] if start_date <= d["local_date"] <= end_date]
            prompts = sum(bool(d.get("send_prompt")) for d in decisions)
            failed = sum(bool(str(d.get("delivery_error") or "").strip()) for d in decisions)
            distress = sum(d.get("distress_flag_active") is True for d in decisions)
            rows.append({"Participant": pid, "Scheduled check-ins": ema, "Prompt decisions sent": prompts,
                         "Delivery errors": failed, "Distress-active decisions": distress,
                         "Push token": tokens.get(str(uid), "Unknown")})
        except Exception:
            rows.append({"Participant": pid, "Scheduled check-ins": "Unavailable", "Prompt decisions sent": None,
                         "Delivery errors": None, "Distress-active decisions": None, "Push token": tokens.get(str(uid), "Unknown")})
    if not rows:
        st.info("No RS participants available.")
        return
    report = pd.DataFrame(rows)
    cols = st.columns(3)
    cols[0].metric("RS participants", len(report))
    cols[1].metric("Missing push tokens", int((report["Push token"] == "Missing").sum()))
    cols[2].metric("Recorded delivery errors", int(pd.to_numeric(report["Delivery errors"], errors="coerce").sum()))
    st.dataframe(report, use_container_width=True, hide_index=True)
    st.download_button("Download live 7-day summary CSV", report.to_csv(index=False).encode("utf-8"),
                       file_name=f"REACT_weekly_summary_{end_date}.csv", mime="text/csv")
    # Keep the faculty-ready PDF export available in Live mode.
    pdf_rows = pd.DataFrame({
        "Participant": report["Participant"],
        "Completion": report["Scheduled check-ins"],
        "Delivery": report["Prompt decisions sent"].apply(lambda x: f"{int(x)} selected" if pd.notna(x) else "Unavailable"),
        "Health": report["Push token"],
    })
    pdf_summary = {
        "overall_status": "Live monitoring",
        "week_label": f"{start_date:%B %d} - {end_date:%B %d, %Y}",
        "source_label": "LIVE monitoring timeline | Eastern time",
        "cards": [
            ("Participants", str(len(report)), "RS study IDs"),
            ("Tokens missing", str(int((report["Push token"] == "Missing").sum())), "Registration only"),
            ("Push errors", str(int(pd.to_numeric(report["Delivery errors"], errors="coerce").sum())), "Recorded errors only"),
        ],
        "notes": [
            "Completion denotes completed / scheduled check-in slots. Incomplete participant histories are labeled unavailable.",
            "Push token presence does not establish working notification delivery; missing device receipts are not considered failures.",
            "Distress-active decisions are decision events with an active flag, not a count of distinct distress flags.",
            "PDF participant snapshot shows at most six IDs; the CSV includes every returned RS participant.",
        ],
        "footer": f"Generated {pd.Timestamp.now(tz=LOCAL_TIMEZONE):%Y-%m-%d %H:%M %Z} | Study IDs only",
    }
    try:
        pdf = build_weekly_summary_pdf(pdf_summary, pdf_rows)
        st.download_button("Download weekly summary PDF", pdf,
                           file_name=f"REACT_weekly_summary_{end_date}.pdf", mime="application/pdf")
    except Exception as exc:
        st.warning(f"Could not render PDF: {exc}")

    st.caption("This live table covers 7 calendar days; the older PDF format is retained in Seed mode only. Missing receipts are not classified as delivery failures. Distress-active decisions are not unique flags.")


def render_weekly_summary_seed_view(log_df, data_mode, include_demo_devices):
    """Render and export a faculty-ready one-page weekly summary."""
    st.header("Weekly Summary")
    st.caption("A one-page faculty update covering participants, completion, and delivery health.")
    try:
        source_df, source_name = load_pipeline_source(data_mode)
        pipeline_df, _ = build_pipeline_data(source_df)
    except Exception as exc:
        source_name = "seed decision-log fallback"
        pipeline_df, _ = build_pipeline_data(log_df)

    today = pd.Timestamp.now(tz=LOCAL_TIMEZONE)
    default_monday = (today - pd.DateOffset(days=today.weekday() + 7)).date()
    selected_monday = st.date_input("Week beginning", value=default_monday,
                                    help="Reports run Monday through Sunday in US Eastern time.")
    selected_monday = selected_monday - pd.Timedelta(days=selected_monday.weekday())
    summary, participant_rows = build_weekly_summary_data(log_df, pipeline_df, selected_monday, data_mode, source_name)

    st.info("SEED DATA MODE — the export is clearly labeled as seed data.")
    st.subheader(summary["week_label"])
    for column, (title, value, detail) in zip(st.columns(3), summary["cards"]):
        column.metric(title, value)
        column.caption(detail)
    for note in summary["notes"]:
        st.caption(note)
    st.markdown("#### Participant snapshot")
    st.caption("Delivery = confirmed / matched sends. PDF shows the first six IDs; totals cover all returned IDs.")
    st.dataframe(participant_rows, use_container_width=True, hide_index=True)
    pdf = build_weekly_summary_pdf(summary, participant_rows)
    file_week = pd.Timestamp(selected_monday).strftime("%Y-%m-%d")
    st.download_button("Download weekly summary PDF", data=pdf,
                       file_name=f"REACT_weekly_summary_{file_week}.pdf",
                       mime="application/pdf", type="primary")
    st.caption("The one-page PDF includes coverage limits and missing-data explanations.")

