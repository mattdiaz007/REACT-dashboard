"""REACT participant grid view."""

from dashboard_core import *


def build_participant_grid_frame(payload: dict) -> pd.DataFrame:
    """Convert the monitoring grid response into one row per participant-day."""

    records = []
    study_days = payload.get("study_days", [])

    for row in payload.get("rows", []):
        risk_components = row.get("risk_components") or {}

        if risk_components:
            top_risk = max(
                risk_components.items(),
                key=lambda item: item[1],
            )

            risk_reason = (
                top_risk[0].replace("_", " ")
                if top_risk[1] > 0
                else ""
            )
        else:
            risk_reason = ""

        values = row.get("values", [])
        local_dates = row.get("local_dates", [])
        run_in = row.get("run_in", [])
        covered = row.get("covered", [])
        reminded = row.get("reminded", [])
        silent = row.get("silent", [])
        alerts = row.get("alert", [])

        for index, day in enumerate(study_days):

            records.append(
                {
                    "user_id": row.get("user_id"),
                    "participant_id": row.get("participant_id"),
                    "phase": row.get("phase"),
                    "study_day_now": row.get("study_day_now"),
                    "risk_score": row.get("risk_score"),
                    "risk_reason": risk_reason,
                    "study_day": day,

                    "value": (
                        values[index]
                        if index < len(values)
                        else None
                    ),

                    "local_date": (
                        local_dates[index]
                        if index < len(local_dates)
                        else None
                    ),

                    "run_in": (
                        run_in[index]
                        if index < len(run_in)
                        else None
                    ),

                    "covered": (
                        covered[index]
                        if index < len(covered)
                        else None
                    ),

                    "reminded": (
                        reminded[index]
                        if index < len(reminded)
                        else None
                    ),

                    "silent": (
                        silent[index]
                        if index < len(silent)
                        else None
                    ),

                    "alert": (
                        alerts[index]
                        if index < len(alerts)
                        else None
                    ),
                }
            )

    frame = pd.DataFrame(records)

    if not frame.empty:
        frame["local_date"] = pd.to_datetime(
            frame["local_date"],
            errors="coerce",
        )

    return frame


def build_slot_split_frame(
    grid_df: pd.DataFrame,
    participant_order: list[str],
) -> pd.DataFrame:
    """Summarize covered, reminded-uncovered, and silent slots over 7 days."""

    lived = grid_df.dropna(subset=["local_date"]).copy()

    if lived.empty:
        return pd.DataFrame()

    latest_date = lived["local_date"].max()
    cutoff = latest_date - pd.Timedelta(days=6)

    recent = lived[
        lived["local_date"] >= cutoff
    ].copy()

    split = recent.melt(
        id_vars=[
            "participant_id",
        ],
        value_vars=[
            "covered",
            "reminded",
            "silent",
        ],
        var_name="Slot state",
        value_name="Slots",
    )

    split["Slots"] = pd.to_numeric(
        split["Slots"],
        errors="coerce",
    ).fillna(0)

    split = (
        split.groupby(
            ["participant_id", "Slot state"],
            as_index=False,
        )["Slots"]
        .sum()
    )

    split["participant_id"] = (
        split["participant_id"].astype(str)
    )

    return split


def format_monitor_age(timestamp) -> str:
    """Show how old a monitoring timestamp is and its Eastern-time value."""

    if not timestamp:
        return "Never"

    moment = pd.to_datetime(
        timestamp,
        errors="coerce",
        utc=True,
    )

    if pd.isna(moment):
        return "Never"

    now = pd.Timestamp.now(tz="UTC")

    hours = (
        now - moment
    ).total_seconds() / 3600

    local = moment.tz_convert(LOCAL_TIMEZONE)

    if hours >= 1:
        age = f"{hours:.0f} h ago"
    else:
        age = f"{hours * 60:.0f} min ago"

    return (
        f"{age} · "
        f"{local.strftime('%b %d, %I:%M %p ET')}"
    )


def format_monitor_percent(value) -> str:
    if value is None or pd.isna(value):
        return "—"

    return f"{float(value):.0%}"


def format_monitor_counts(
    rollup: dict,
    prefix: str,
) -> str:

    numerator = rollup.get(f"{prefix}_num")
    denominator = rollup.get(f"{prefix}_den")

    if numerator is None or denominator is None:
        return "—"

    return f"{numerator} / {denominator}"


def render_participant_grid_rail(
    user_id: int,
    study_days: int,
) -> None:
    """Render the Stage 2 detail rail for one selected participant."""

    try:
        detail = load_monitor_participant(user_id)
    except Exception as exc:
        st.error(
            f"Could not load participant details: {exc}"
        )
        return

    rollup = detail.get("participant") or {}

    participant_id = (
        detail.get("participant_id")
        or str(user_id)
    )

    st.subheader(participant_id)

    phase = rollup.get("phase") or "—"
    day_now = rollup.get("study_day_now")

    if day_now is None:
        st.caption(phase)
    else:
        remaining = max(
            0,
            study_days - 1 - int(day_now),
        )

        st.caption(
            f"{phase.replace('_', ' ')} · "
            f"study day {day_now} of {study_days} · "
            f"{remaining} remaining"
        )

    risk_score = rollup.get("risk_score")

    st.metric(
        "Risk score",
        risk_score
        if risk_score is not None
        else "—",
    )

    risk_components = (
        rollup.get("risk_components")
        or {}
    )

    if risk_components:
        st.markdown("**Risk breakdown**")

        risk_rows = [
            {
                "Factor": factor.replace("_", " ").title(),
                "Points": points,
            }
            for factor, points in sorted(
                risk_components.items(),
                key=lambda item: -item[1],
            )
        ]

        st.dataframe(
            pd.DataFrame(risk_rows),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("**Cumulative metrics**")

    cumulative = pd.DataFrame(
        [
            {
                "Metric": "Slot coverage",
                "Value": format_monitor_percent(
                    rollup.get("slot_coverage_rate")
                ),
                "Counts": format_monitor_counts(
                    rollup,
                    "slot_coverage",
                ),
            },
            {
                "Metric": "Prompt response",
                "Value": format_monitor_percent(
                    rollup.get("prompt_response_rate")
                ),
                "Counts": format_monitor_counts(
                    rollup,
                    "prompt_response",
                ),
            },
            {
                "Metric": "Wear",
                "Value": format_monitor_percent(
                    rollup.get("wear_rate")
                ),
                "Counts": format_monitor_counts(
                    rollup,
                    "wear",
                ),
            },
        ]
    )

    st.dataframe(
        cumulative,
        use_container_width=True,
        hide_index=True,
    )

    daily = detail.get("daily") or []

    prompts_delivered = sum(
        day.get("delivered_n") or 0
        for day in daily
    )

    st.markdown("**To date**")

    st.write(
        f"Prompts delivered: **{prompts_delivered}**"
    )

    st.write(
        "Last check-in: "
        f"**{format_monitor_age(rollup.get('last_ema_at'))}**"
    )

    st.write(
        "Last EMA: "
        f"**{format_monitor_age(rollup.get('last_ema_at'))}**"
    )

    alerts = detail.get("alerts") or []

    st.markdown(
        f"**Open alerts ({len(alerts)})**"
    )

    if not alerts:
        st.success("No open participant alerts.")
    else:
        for alert in alerts:
            severity = alert.get(
                "severity",
                "unknown",
            )

            rule = alert.get(
                "rule_id",
                "unknown",
            )

            st.write(
                f"`{severity}` "
                f"{rule.replace('_', ' ')}"
            )


def render_participant_grid_view( data_mode: str, include_demo_devices: bool) -> None:
    """Render the Stage 2 participant monitoring grid."""

    st.header("Participant Grid")
    st.caption(
        "Daily operational view for identifying participants who may need attention."
    )

    if data_mode != "Live":
        st.info(
            "Participant Grid currently uses the live monitoring API. "
            "Switch to Live mode to view it."
        )
        return

    metric_options = {
        "Slot coverage": "slots_covered",
        "Prompts delivered": "delivered_n",
        "Item completeness": "completeness_mean",
    }

    phase_options = {
        "All participants": "all",
        "Phase 1": "phase1",
        "Phase 2": "phase2",
    }

    control_columns = st.columns(2)

    with control_columns[0]:
        selected_metric_label = st.selectbox(
            "Grid metric",
            list(metric_options.keys()),
            key="participant_grid_metric",
        )

    with control_columns[1]:
        selected_phase_label = st.selectbox(
            "Cohort",
            list(phase_options.keys()),
            key="participant_grid_phase",
        )

    selected_metric = metric_options[selected_metric_label]
    selected_phase = phase_options[selected_phase_label]

    try:
        payload = load_monitor_grid(
            selected_metric,
            selected_phase,
        )

        payload = dict(payload)

        payload["rows"] = filter_monitor_participants(
        payload.get("rows", []),
        include_demo_devices,
)

        payload["n_participants"] = len(payload["rows"])
    except Exception as exc:
        st.error(
            f"Could not load participant grid data: {exc}"
        )
        return

    if not payload.get("rows"):
        st.warning("No participants are available for this cohort.")
        return

    grid_df = build_participant_grid_frame(payload)

    if grid_df.empty:
        st.warning("The monitoring endpoint returned no participant-day rows.")
        return

    # -----------------------------
    # Risk-ordered participant list
    # -----------------------------
    participant_rows = (
        grid_df[
            [
                "user_id",
                "participant_id",
                "phase",
                "study_day_now",
                "risk_score",
                "risk_reason",
            ]
        ]
        .drop_duplicates("user_id")
        .copy()
    )

    participant_rows["_unscored"] = (
        participant_rows["risk_score"].isna()
    )

    participant_rows = participant_rows.sort_values(
        [
            "_unscored",
            "risk_score",
            "participant_id",
        ],
        ascending=[
            True,
            False,
            True,
        ],
    ).drop(columns="_unscored")

    left_column, right_column = st.columns(
    [4, 1],
    gap="medium",
    )
    with left_column:

        # st.subheader("Call list")
        st.caption(
            "Participants are ordered by monitoring risk score. "
            "A blank risk score means the participant is outside the active study period."
        )

        call_list = participant_rows.rename(
            columns={
                "participant_id": "Participant",
                "study_day_now": "Study day",
                "phase": "Phase",
                "risk_score": "Risk",
                "risk_reason": "Primary risk factor",
            }
        )[
            [
                "Participant",
                "Study day",
                "Phase",
                "Risk",
                "Primary risk factor",
            ]
        ]

        selection_event = st.dataframe(
        call_list,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="participant_grid_call_list",
    )

        st.divider()

        # -----------------------------
        # Participant-day heatmap
        # -----------------------------
        st.subheader(selected_metric_label)

        st.caption(
            "Blank cells are outside the participant's active study window. "
            "Shaded cells are run-in baseline days. "
            "A dot marks an open alert, and a slash means at least one slot "
            "was silent with no reminder sent."
        )

        plot_df = grid_df.dropna(
            subset=["local_date"]
        ).copy()

        if selected_metric == "slots_covered":
            denominator = (
                pd.to_numeric(plot_df["covered"], errors="coerce").fillna(0)
                + pd.to_numeric(plot_df["reminded"], errors="coerce").fillna(0)
                + pd.to_numeric(plot_df["silent"], errors="coerce").fillna(0)
            )

            plot_df["display_value"] = (
                pd.to_numeric(
                    plot_df["value"],
                    errors="coerce",
                )
                / denominator.replace(0, pd.NA)
            )

            value_title = "Slot coverage"
            scale_domain = [0, 1]

        else:
            plot_df["display_value"] = pd.to_numeric(
                plot_df["value"],
                errors="coerce",
            )

            value_title = selected_metric_label

            if selected_metric in {
                "wear_valid_pct",
                "completeness_mean",
            }:
                scale_domain = [0, 1]
            else:
                scale_domain = None

        order = participant_rows[
            "participant_id"
        ].astype(str).tolist()

        plot_df["participant_id"] = (
            plot_df["participant_id"].astype(str)
        )

        base = alt.Chart(plot_df).encode(
            x=alt.X(
                "study_day:O",
                title="Study day",
                sort="ascending",
            ),
            y=alt.Y(
                "participant_id:N",
                title="Participant",
                sort=order,
            ),
        )

        # Main heatmap cells
        cells = base.mark_rect(
            stroke="white",
            strokeWidth=0.5,
        ).encode(
            color=alt.Color(
                "display_value:Q",
                title=value_title,
                scale=(
                    alt.Scale(domain=scale_domain)
                    if scale_domain
                    else alt.Scale()
                ),
            ),
            tooltip=[
                alt.Tooltip(
                    "participant_id:N",
                    title="Participant",
                ),
                alt.Tooltip(
                    "study_day:O",
                    title="Study day",
                ),
                alt.Tooltip(
                    "local_date:T",
                    title="Date",
                ),
                alt.Tooltip(
                    "display_value:Q",
                    title=value_title,
                    format=".2f",
                ),
                alt.Tooltip(
                    "covered:Q",
                    title="Covered",
                ),
                alt.Tooltip(
                    "reminded:Q",
                    title="Reminded",
                ),
                alt.Tooltip(
                    "silent:Q",
                    title="Silent",
                ),
            ],
        )

        # Run-in baseline overlay
        run_in_overlay = (
            base
            .transform_filter(
                alt.datum.run_in == True
            )
            .mark_rect(
                color="black",
                opacity=0.12,
            )
        )

        # Alert indicator
        alert_overlay = (
            base
            .transform_filter(
                alt.datum.alert == True
            )
            .mark_point(
                shape="circle",
                size=35,
                filled=True,
                color="black",
            )
        )

        # Silent-slot indicator
        silent_overlay = (
            base
            .transform_filter(
                alt.datum.silent > 0
            )
            .mark_text(
                text="/",
                fontSize=16,
                fontWeight="bold",
                color="#4a4a8a",
            )
        )

        chart = (
            alt.layer(
                cells,
                run_in_overlay,
                alert_overlay,
                silent_overlay,
            )
            .properties(
                height=max(
                    220,
                    len(order) * 34,
                )
            )
        )

        st.altair_chart(
            chart,
            use_container_width=True,
        )
        st.subheader("Where the empty slots went")

    st.caption(
        "Trailing 7 days. Covered means the participant completed the slot. "
        "Reminded means they were asked but did not complete it. "
        "Silent means no reminder was sent, which points to a scheduling "
        "or delivery issue rather than participant non-response."
    )

    split_df = build_slot_split_frame(
        grid_df,
        order,
    )

    if split_df.empty:
        st.info("No slot data is available.")
    else:
        slot_colors = {
            "covered": "#1a9850",
            "reminded": "#fdae61",
            "silent": "#4a4a8a",
        }

        split_chart = (
            alt.Chart(split_df)
            .mark_bar()
            .encode(
                x=alt.X(
                    "Slots:Q",
                    title="Slots over trailing 7 days",
                    stack="zero",
                ),
                y=alt.Y(
                    "participant_id:N",
                    title="Participant",
                    sort=order,
                ),
                color=alt.Color(
                    "Slot state:N",
                    title=None,
                    scale=alt.Scale(
                        domain=list(slot_colors.keys()),
                        range=list(slot_colors.values()),
                    ),
                ),
                tooltip=[
                    alt.Tooltip(
                        "participant_id:N",
                        title="Participant",
                    ),
                    alt.Tooltip(
                        "Slot state:N",
                        title="State",
                    ),
                    alt.Tooltip(
                        "Slots:Q",
                        title="Slots",
                    ),
                ],
            )
            .properties(
                height=max(
                    180,
                    len(order) * 32,
                )
            )
        )

        st.altair_chart(
            split_chart,
            use_container_width=True,
        )

    with right_column:

        selected_rows = (
            selection_event.selection.rows
            if selection_event
            else []
        )

        if not selected_rows:
            st.info(
                "Select a participant from the call list "
                "to inspect their monitoring details."
            )

        else:
            selected_position = selected_rows[0]

            selected_user_id = participant_rows.iloc[
                selected_position
            ]["user_id"]

            render_participant_grid_rail(
                int(selected_user_id),
                len(payload.get("study_days", [])),
            )

