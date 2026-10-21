"""REACT cohort view."""

from dashboard_core import *


def render_cohort_view(data_mode: str) -> None:
    """Render the Stage 1 cohort monitoring board."""

    st.header("Cohort Analytics")
    st.caption(
    "Weekly study-level feasibility and protocol-integrity monitoring. "
    "Cohort metrics are calculated by the REACT backend."
    )

    if data_mode != "Live":
        st.info(
            "Cohort Analytics currently uses the live monitoring API. "
            "Switch to Live mode to view it."
        )
        return

    phase_options = {
        "All participants": "all",
        "Phase 1": "phase1",
        "Phase 2": "phase2",
    }

    selected_phase_label = st.selectbox(
        "Cohort",
        list(phase_options.keys()),
    )
    selected_phase = phase_options[selected_phase_label]

    try:
        snapshot = load_monitor_cohort(selected_phase)
        alerts_payload = load_monitor_alerts()
    except Exception as exc:
        st.error(f"Could not load cohort monitoring data: {exc}")
        return

    if not snapshot.get("available", True):
        st.warning(
            snapshot.get(
                "detail",
                "No monitoring snapshot is available yet.",
            )
        )
        return

    as_of = snapshot.get("as_of")

    if as_of:
        as_of_time = pd.to_datetime(as_of, utc=True).tz_convert(LOCAL_TIMEZONE)

        st.caption(
            f"Monitoring snapshot: "
            f"{as_of_time.strftime('%b %d, %Y at %I:%M %p ET')}"
        )

    participant_count = snapshot.get("n_participants", 0)
    active_count = snapshot.get("n_active", 0)

    metrics = st.columns(3)

    metrics[0].metric(
        "Participants",
        participant_count,
    )

    metrics[1].metric(
        "Active",
        active_count,
    )

    metrics[2].metric(
        "Open alerts",
        alerts_payload.get("open", 0),
    )

    st.divider()

    st.subheader("Study benchmarks")
    st.caption("Wearable coverage is monitored outside this dashboard and is not interpreted here.")

    benchmarks = snapshot.get("benchmarks", {})

    benchmark_labels = {
        "slot_coverage": "Slot coverage",
        "prompt_response": "Prompt response",
        "retention": "Retention",
    }

    columns = st.columns(4)

    for column, (key, label) in zip(
        columns,
        benchmark_labels.items(),
    ):
        entry = benchmarks.get(key, {})

        with column:
            st.markdown(f"**{label}**")

            if not entry.get("measurable", True):
                st.metric(label, "Unavailable")
                st.caption("No source currently available.")
                continue

            numerator = entry.get("numerator")
            denominator = entry.get("denominator")

            if entry.get("suppressed"):
                st.metric(
                    label,
                    f"{numerator}/{denominator}",
                )
                st.caption("Rate suppressed due to limited sample size.")
                continue

            value = entry.get("value")

            if value is None:
                st.metric(label, "—")
            else:
                st.metric(
                    label,
                    f"{value:.0%}",
                )

            low = entry.get("wilson_low")
            high = entry.get("wilson_high")

            if low is not None and high is not None:
                st.caption(
                    f"95% CI: {low:.0%}–{high:.0%}"
                )

            if numerator is not None and denominator is not None:
                st.caption(
                    f"{numerator}/{denominator}"
                )

    st.divider()

    st.subheader("MRT integrity")
    st.caption(
        "Protocol checks for the intervention period. "
        "Unavailable values mean there is not yet enough applicable decision data."
    )

    integrity = snapshot.get("integrity", {})

    integrity_rows = []

    for name, entry in integrity.items():
        if not isinstance(entry, dict):
            continue

        value = entry.get("value")
        numerator = entry.get("numerator")
        denominator = entry.get("denominator")

        if value is not None:
            display_value = f"{value:.0%}"
        elif numerator is not None and denominator is not None:
            display_value = f"{numerator}/{denominator}"
        elif name == "cooldown_violations":
            display_value = str(entry.get("violations", 0))
        elif name == "randomization_audit":
            draws = entry.get("draws", 0)
            display_value = f"{draws} draws"
        else:
            display_value = "Not available"

        integrity_rows.append(
            {
                "Metric": name.replace("_", " ").title(),
                "Result": display_value,
                "Status": (
                    "Attention"
                    if entry.get("alarm")
                    else "OK"
                    if value is not None
                    else "Pending"
                ),
            }
        )

    if integrity_rows:
        st.dataframe(
            pd.DataFrame(integrity_rows),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No MRT integrity results are available yet.")

    st.divider()

    st.subheader("Open alerts")

    alerts = alerts_payload.get("alerts", [])

    if not alerts:
        st.success("No open monitoring alerts.")
    else:
        alert_rows = []

        for alert in alerts:
            alert_rows.append(
                {
                    "Severity": alert.get("severity"),
                    "Rule": alert.get("rule_id"),
                    "Participant": (
                        alert.get("participant_id")
                        or "Cohort"
                    ),
                    "Fired": alert.get("fired_at"),
                }
            )

        st.dataframe(
            pd.DataFrame(alert_rows),
            use_container_width=True,
            hide_index=True,
        )

