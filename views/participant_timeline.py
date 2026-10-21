"""REACT participant timeline view."""

from dashboard_core import *


def timeline_minutes(value, day_start):
    """Convert a timestamp into minutes after local midnight."""

    if value is None or day_start is None:
        return None

    return (
        pd.Timestamp(value) - pd.Timestamp(day_start)
    ).total_seconds() / 60


def render_participant_timeline_view(data_mode: str, include_demo_devices: bool) -> None:
    """Render the Stage 3 participant diagnostic timeline."""

    st.header("Participant Timeline")
    st.info("The Wear and Sync lanes are legacy diagnostics. Use check-ins and decision/delivery events for current pilot monitoring.")
    st.caption(
        "Participant-level diagnostic view. Events share a 24-hour clock "
        "so check-in, decision, and delivery activity can be compared. Wearable and sync lanes are legacy diagnostics."
    )

    if data_mode != "Live":
        st.info(
            "Participant Timeline currently uses the live monitoring API. "
            "Switch to Live mode to view it."
        )
        return

    # Get participant roster from the monitoring grid.
    try:
        roster = load_monitor_grid(
            metric="slots_covered",
            phase="all",
        )
    except Exception as exc:
        st.error(f"Could not load participant roster: {exc}")
        return

    rows = filter_monitor_participants(
    roster.get("rows", []),
    include_demo_devices)

    if not rows:
        st.warning("No participants are available.")
        return

    participant_options = {
        str(row.get("participant_id")): row.get("user_id")
        for row in rows
    }

    controls = st.columns([2, 1, 1])

    with controls[0]:
        selected_participant = st.selectbox(
            "Participant",
            list(participant_options.keys()),
            key="timeline_participant",
        )

    with controls[1]:
        days = st.slider(
            "Days",
            min_value=1,
            max_value=14,
            value=1,
        )

    with controls[2]:
        end_date = st.date_input(
            "Ending on",
            value=pd.Timestamp.now(
                tz=LOCAL_TIMEZONE
            ).date(),
        )

    user_id = participant_options[selected_participant]

    try:
        timeline_payload = load_monitor_timeline(
            int(user_id),
            local_date=str(end_date),
            days=days,
        )

        funnel_payload = load_monitor_funnel(
            int(user_id)
        )

    except Exception as exc:
        st.error(
            f"Could not load participant timeline: {exc}"
        )
        return

    # ----------------------------
    # Delivery funnel
    # ----------------------------

    st.subheader("Delivery funnel")
    st.caption(
        "Whole-study prompt delivery for this participant."
    )

    stages = funnel_payload.get("stages") or []

    if stages:
        funnel_df = pd.DataFrame(stages)

        if "stage" in funnel_df.columns:
            funnel_df["Stage"] = (
                funnel_df["stage"]
                .astype(str)
                .str.replace("_", " ", regex=False)
                .str.title()
            )

        if "n" in funnel_df.columns:
            funnel_chart = (
                alt.Chart(funnel_df)
                .mark_bar()
                .encode(
                    x=alt.X(
                        "n:Q",
                        title="Prompts",
                    ),
                    y=alt.Y(
                        "Stage:N",
                        title=None,
                        sort=None,
                    ),
                    tooltip=[
                        alt.Tooltip(
                            "Stage:N",
                            title="Stage",
                        ),
                        alt.Tooltip(
                            "n:Q",
                            title="Count",
                        ),
                    ],
                )
            )

            st.altair_chart(
                funnel_chart,
                use_container_width=True,
            )
    else:
        st.info(
            "No prompt-delivery funnel is available for this participant."
        )

    st.divider()

    # Timeline endpoint returns either one day directly
    # or {"days": [...]} for multi-day requests.
    if days > 1:
        timeline_days = timeline_payload.get(
            "days",
            [],
        )
    else:
        timeline_days = [timeline_payload]

    if not timeline_days:
        st.info(
            "No timeline data is available for the selected period."
        )
        return

    for day in timeline_days:

        st.markdown(
            f"### {day.get('local_date', 'Unknown date')} "
            f"· Study day {day.get('study_day', '—')}"
        )

        day_start = day.get("day_start")

        # =====================================
        # 1. WEAR
        # =====================================

        st.markdown("#### Wear (legacy diagnostic)")

        wear = day.get("wear") or {}

        if not wear.get("has_data"):
            st.caption(
                "No heart-rate samples recorded for this day."
            )
        else:
            wear_rows = []

            for span in wear.get("covered", []):
                wear_rows.append(
                    {
                        "start": span.get("start"),
                        "end": span.get("end"),
                        "state": "Covered",
                    }
                )

            for gap in wear.get("gaps", []):
                wear_rows.append(
                    {
                        "start": gap.get("start"),
                        "end": gap.get("end"),
                        "state": (
                            "Gap > threshold"
                            if gap.get("over_threshold")
                            else "Gap"
                        ),
                    }
                )

            wear_df = pd.DataFrame(wear_rows)

            if not wear_df.empty:

                wear_chart = (
                    alt.Chart(wear_df)
                    .mark_bar(size=18)
                    .encode(
                        x=alt.X(
                            "start:Q",
                            title="Local time",
                            scale=alt.Scale(
                                domain=[0, 1440]
                            ),
                            axis=alt.Axis(
                                values=[
                                    0,
                                    180,
                                    360,
                                    540,
                                    720,
                                    900,
                                    1080,
                                    1260,
                                    1440,
                                ],
                                labelExpr=(
                                    "floor(datum.value / 60) + ':00'"
                                ),
                            ),
                        ),
                        x2="end:Q",
                        color=alt.Color(
                            "state:N",
                            title=None,
                        ),
                        tooltip=[
                            "state:N",
                            "start:Q",
                            "end:Q",
                        ],
                    )
                    .properties(height=55)
                )

                st.altair_chart(
                    wear_chart,
                    use_container_width=True,
                )

        # =====================================
        # 2. SYNC
        # =====================================

        st.markdown("#### Sync (legacy diagnostic)")

        sync = day.get("sync") or {}

        if not sync.get("observed"):
            st.caption(
                "No sync-history records are available for this participant-day."
            )
        else:
            sync_rows = []

            carried = sync.get("carried_in")

            if carried:
                last_sync = carried.get(
                    "last_synced_at"
                )

                stale_minutes = (
                    None
                    if last_sync is None
                    else max(
                        0,
                        -timeline_minutes(
                            last_sync,
                            day_start,
                        ),
                    )
                )

                sync_rows.append(
                    {
                        "at": 0,
                        "stale_minutes": stale_minutes,
                        "source": "Carried in",
                    }
                )

            for event in sync.get(
                "advances",
                [],
            ):
                at = timeline_minutes(
                    event.get("observed_at"),
                    day_start,
                )

                last_sync = event.get(
                    "last_synced_at"
                )

                if (
                    at is not None
                    and last_sync is not None
                ):
                    last_sync_minutes = (
                        timeline_minutes(
                            last_sync,
                            day_start,
                        )
                    )

                    stale = max(
                        0,
                        at - last_sync_minutes,
                    )
                else:
                    stale = None

                sync_rows.append(
                    {
                        "at": at,
                        "stale_minutes": stale,
                        "source": event.get(
                            "source",
                            "sync",
                        ),
                    }
                )

            sync_df = pd.DataFrame(sync_rows)

            if not sync_df.empty:

                sync_chart = (
                    alt.Chart(sync_df)
                    .mark_line(
                        point=True,
                        interpolate="step-after",
                    )
                    .encode(
                        x=alt.X(
                            "at:Q",
                            title="Local time",
                            scale=alt.Scale(
                                domain=[0, 1440]
                            ),
                        ),
                        y=alt.Y(
                            "stale_minutes:Q",
                            title="Minutes behind",
                        ),
                        tooltip=[
                            "source:N",
                            "stale_minutes:Q",
                        ],
                    )
                    .properties(height=80)
                )

                st.altair_chart(
                    sync_chart,
                    use_container_width=True,
                )

        # =====================================
        # 3. CHECK-INS
        # =====================================

        st.markdown("#### Check-ins")

        ema_rows = []

        for event in day.get(
            "events",
            [],
        ):

            if event.get("kind") != "ema":
                continue

            ema_rows.append(
                {
                    "at": timeline_minutes(
                        event.get("at"),
                        day_start,
                    ),
                    "type": event.get(
                        "ema_type",
                        "EMA",
                    ),
                    "status": event.get(
                        "status",
                    ),
                    "slot": event.get(
                        "slot",
                    ),
                }
            )

        ema_df = pd.DataFrame(ema_rows)

        if ema_df.empty:
            st.caption(
                "No check-ins recorded for this day."
            )
        else:

            ema_chart = (
                alt.Chart(ema_df)
                .mark_point(
                    size=100,
                    filled=True,
                )
                .encode(
                    x=alt.X(
                        "at:Q",
                        title="Local time",
                        scale=alt.Scale(
                            domain=[0, 1440]
                        ),
                    ),
                    y=alt.Y(
                        "type:N",
                        title=None,
                    ),
                    color=alt.Color(
                        "status:N",
                        title="Status",
                    ),
                    tooltip=[
                        "type:N",
                        "status:N",
                        "slot:Q",
                    ],
                )
                .properties(height=80)
            )

            st.altair_chart(
                ema_chart,
                use_container_width=True,
            )

        # =====================================
        # 4. DECISION POINTS
        # =====================================

        st.markdown("#### Decision points")

        decision_rows = []

        for event in day.get(
            "events",
            [],
        ):

            if event.get("kind") != "decision":
                continue

            decision_rows.append(
                {
                    "at": timeline_minutes(
                        event.get("at"),
                        day_start,
                    ),
                    "reason": event.get(
                        "trigger_reason",
                    ),
                    "sent": event.get(
                        "send_prompt",
                    ),
                    "mssd": event.get(
                        "observed_mssd",
                    ),
                    "threshold": event.get(
                        "threshold_at_decision",
                    ),
                }
            )

        decision_df = pd.DataFrame(
            decision_rows
        )

        if decision_df.empty:
            st.caption(
                "No decision points recorded for this day."
            )
        else:

            decision_chart = (
                alt.Chart(decision_df)
                .mark_point(
                    size=100,
                    filled=True,
                )
                .encode(
                    x=alt.X(
                        "at:Q",
                        title="Local time",
                        scale=alt.Scale(
                            domain=[0, 1440]
                        ),
                    ),
                    y=alt.Y(
                        "reason:N",
                        title=None,
                    ),
                    color=alt.Color(
                        "sent:N",
                        title="Prompt sent",
                    ),
                    tooltip=[
                        "reason:N",
                        "sent:N",
                        "mssd:Q",
                        "threshold:Q",
                    ],
                )
                .properties(height=100)
            )

            st.altair_chart(
                decision_chart,
                use_container_width=True,
            )

        # =====================================
        # 5. DELIVERY
        # =====================================

        st.markdown("#### Delivery")

        delivery_rows = []

        for event in day.get(
            "events",
            [],
        ):

            if event.get("kind") != "decision":
                continue

            push_sent = event.get(
                "push_sent_at"
            )

            if not push_sent:
                continue

            sent_at = timeline_minutes(
                push_sent,
                day_start,
            )

            received_at = timeline_minutes(
                event.get(
                    "device_received_at"
                ),
                day_start,
            )

            delivery_rows.append(
                {
                    "sent": sent_at,
                    "received": received_at,
                    "status": event.get(
                        "delivery_status",
                    ),
                    "error": event.get(
                        "delivery_error",
                    ),
                }
            )

        delivery_df = pd.DataFrame(
            delivery_rows
        )

        if delivery_df.empty:
            st.caption(
                "No prompts were sent this day."
            )
        else:

            received = delivery_df[
                delivery_df[
                    "received"
                ].notna()
            ]

            layers = []

            if not received.empty:

                layers.append(
                    alt.Chart(received)
                    .mark_bar(size=10)
                    .encode(
                        x=alt.X(
                            "sent:Q",
                            title="Local time",
                            scale=alt.Scale(
                                domain=[0, 1440]
                            ),
                        ),
                        x2="received:Q",
                        tooltip=[
                            "status:N",
                            "error:N",
                        ],
                    )
                )

            layers.append(
                alt.Chart(delivery_df)
                .mark_point(
                    size=100,
                    filled=True,
                )
                .encode(
                    x="sent:Q",
                    color=alt.Color(
                        "status:N",
                        title="Delivery status",
                    ),
                    tooltip=[
                        "status:N",
                        "error:N",
                    ],
                )
            )

            delivery_chart = (
                alt.layer(*layers)
                .properties(height=65)
            )

            st.altair_chart(
                delivery_chart,
                use_container_width=True,
            )

        # =====================================
        # 6. MSSD / VOLATILITY
        # =====================================

        st.markdown("#### MSSD / volatility")

        mssd_rows = []

        for point in day.get(
            "mssd",
            [],
        ):

            mssd_rows.append(
                {
                    "at": timeline_minutes(
                        point.get("at"),
                        day_start,
                    ),
                    "observed_mssd": point.get(
                        "observed_mssd"
                    ),
                    "threshold": point.get(
                        "threshold"
                    ),
                    "reason": point.get(
                        "trigger_reason"
                    ),
                }
            )

        mssd_df = pd.DataFrame(
            mssd_rows
        )

        if mssd_df.empty:
            st.caption(
                "No volatility decision points are available for this day."
            )
        else:

            observed = (
                alt.Chart(mssd_df)
                .mark_line(
                    point=True,
                    interpolate="step-after",
                )
                .encode(
                    x=alt.X(
                        "at:Q",
                        title="Local time",
                        scale=alt.Scale(
                            domain=[0, 1440]
                        ),
                    ),
                    y=alt.Y(
                        "observed_mssd:Q",
                        title="MSSD",
                    ),
                    tooltip=[
                        "observed_mssd:Q",
                        "threshold:Q",
                        "reason:N",
                    ],
                )
            )

            thresholds = (
                alt.Chart(
                    mssd_df[
                        mssd_df[
                            "threshold"
                        ].notna()
                    ]
                )
                .mark_line(
                    strokeDash=[4, 3],
                )
                .encode(
                    x="at:Q",
                    y="threshold:Q",
                )
            )

            st.altair_chart(
                alt.layer(
                    observed,
                    thresholds,
                ).properties(
                    height=130,
                ),
                use_container_width=True,
            )

        st.divider()

