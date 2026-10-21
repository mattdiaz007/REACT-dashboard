"""REACT daily monitoring view."""

from dashboard_core import *
from views.participant_detail import monitoring_token_map


def render_daily_monitoring_view(
    log_df: pd.DataFrame,
    data_mode: str,
    include_demo_devices: bool,
) -> None:
    """Render the live morning monitoring screen for pilot operations."""

    st.header("Daily Monitoring")

    st.caption(
        "Morning operational check: completed check-ins, prompt decisions, "
        "delivery health, and participants needing attention. Times are Eastern."
    )

    # Daily Monitoring should use real monitoring data in Live.

    if data_mode != "Live":
        st.info(
            "SEED DATA MODE — Daily Monitoring operational status is intended "
            "for the live monitoring API. Switch to Live mode for pilot status."
        )
        return

    st.success(
        "LIVE DATA MODE — monitoring API data only. "
        "Seed values are never substituted."
    )

    now_et = pd.Timestamp.now(tz=LOCAL_TIMEZONE)

    yesterday = (
        now_et - pd.Timedelta(days=1)
    ).date()

    stale_cutoff = (
        now_et - pd.Timedelta(hours=24)
    )

    # Load participant roster from monitoring API

    try:
        roster_payload = load_monitor_grid(
            metric="ema_scheduled_n",
            phase="all",
        )

        roster_rows = filter_monitor_participants(
            roster_payload.get("rows", []),
            include_demo_devices,
        )

    except Exception as exc:
        st.error(
            "Live monitoring data could not be loaded."
        )
        st.code(str(exc))
        return

    # Reuse the same token-registration lookup as Participant Detail and Weekly Summary.
    status_by_user = monitoring_token_map()

    # Build one participant record per user

    roster = []
    seen_users = set()

    for row in roster_rows:

        user_id = row.get("user_id")

        participant_id = str(
            row.get("participant_id") or ""
        ).strip().upper()

        if user_id is None:
            continue

        if user_id in seen_users:
            continue

        seen_users.add(user_id)

        roster.append(
            {
                "user_id": user_id,
                "participant_id": participant_id,
                "group": study_id_group(
                    participant_id
                ),
            }
        )

    # Study groups
    #
    # RS = actual pilot participants
    # ST = staff/test accounts shown separately
    # OTHER = numeric/non-study accounts excluded

    rs_roster = [
        row
        for row in roster
        if row["group"] == "RS"
    ]

    st_roster = [
        row
        for row in roster
        if row["group"] == "ST"
    ]

    other_roster = [
        row
        for row in roster
        if row["group"] == "OTHER"
    ]

    # Pull two-day timeline for each RS participant.
    #
    # Two days gives us:
    # - last 24h completed-check-in status
    # - yesterday EMA completion
    # - recent decision activity

    participant_rows = []
    activity_rows = []
    yesterday_participant_rows = []

    yesterday_completed = 0
    yesterday_scheduled = 0

    with st.spinner(
        "Loading participant check-ins and decision activity..."
    ):

        for participant in rs_roster:

            user_id = participant["user_id"]

            participant_id = participant[
                "participant_id"
            ]

            token_status = status_by_user.get(str(user_id), "Unknown")

            distress_active_decisions = 0

            try:
                timeline = load_monitor_timeline(
                    user_id=user_id,
                    local_date=(
                        now_et.date().isoformat()
                    ),
                    days=2,
                )

            except Exception as exc:

                participant_rows.append(
                    {
                        "Participant": participant_id,
                        "Last completed check-in": pd.NaT,
                        "Stale": False,
                        "Last push": pd.NaT,
                        "Last receipt": pd.NaT,
                        "Load error": str(exc),
                        "Push token": token_status,
                    }
                )

                yesterday_participant_rows.append(
                    {
                        "Participant": participant_id,
                        "Completed": None,
                        "Delivered": None,
                        "Last completed yesterday": pd.NaT,
                        "Distress-active decisions": None,
                        "Push token": token_status,
                    }
                )

                continue

            days = timeline_days(timeline)

            completed_checkins = []

            last_push = pd.NaT
            last_receipt = pd.NaT

            participant_yesterday_delivered = 0
            participant_yesterday_completed = 0
            participant_yesterday_last = pd.NaT
            yesterday_returned = False

            # Process yesterday + today

            for day in days:

                raw_day_date = pd.to_datetime(
                    day.get("local_date"),
                    errors="coerce",
                )

                if pd.isna(raw_day_date):
                    day_date = None
                else:
                    day_date = (
                        raw_day_date.date()
                    )

                # Number of scheduled slots yesterday
                if day_date == yesterday:
                    yesterday_returned = True
                    delivered_slots = len(
                        day.get("slots") or []
                    )

                    yesterday_scheduled += delivered_slots
                    participant_yesterday_delivered = delivered_slots

                completed_slots = set()

                # Events

                for event in day.get(
                    "events",
                    [],
                ):

                    event_kind = event.get(
                        "kind"
                    )

                    # CHECK-INS

                    if event_kind == "ema":

                        event_time = (
                            parse_monitor_time(
                                event.get("at")
                            )
                        )

                        completed = (
                            str(
                                event.get(
                                    "status"
                                )
                                or ""
                            ).lower()
                            == "completed"
                        )

                        if (
                            completed
                            and event.get("ema_type") == "scheduled_check_in"
                            and pd.notna(event_time)
                        ):
                            completed_checkins.append(
                                event_time
                            )

                        # Yesterday EMA only counts completed
                        # scheduled check-ins.
                        if (
                            day_date == yesterday
                            and completed
                            and event.get("ema_type") == "scheduled_check_in"
                            and event.get("slot") is not None
                        ):
                            completed_slots.add(
                                event.get("slot")
                            )

                            if pd.notna(event_time):
                                if (
                                    pd.isna(participant_yesterday_last)
                                    or event_time > participant_yesterday_last
                                ):
                                    participant_yesterday_last = event_time

                    # DECISIONS

                    if event_kind != "decision":
                        continue

                    # Count decisions with an active distress flag yesterday.
                    # This is not a count of newly raised flags.
                    if day_date == yesterday:
                        if event.get("distress_flag_active") is True:
                            distress_active_decisions += 1

                    decision_time = (
                        parse_monitor_time(
                            event.get("at")
                        )
                    )

                    push_time = (
                        parse_monitor_time(
                            event.get(
                                "push_sent_at"
                            )
                        )
                    )

                    receipt_time = (
                        parse_monitor_time(
                            event.get(
                                "device_received_at"
                            )
                        )
                    )

                    send_prompt = bool(
                        event.get(
                            "send_prompt"
                        )
                    )

                    delivery_error = str(
                        event.get(
                            "delivery_error"
                        )
                        or ""
                    ).strip()

                    # Track latest actual push
                    if pd.notna(push_time):

                        if (
                            pd.isna(last_push)
                            or push_time
                            > last_push
                        ):
                            last_push = (
                                push_time
                            )

                    # Track latest actual receipt
                    if pd.notna(
                        receipt_time
                    ):

                        if (
                            pd.isna(
                                last_receipt
                            )
                            or receipt_time
                            > last_receipt
                        ):
                            last_receipt = (
                                receipt_time
                            )

                    # Delivery state

                    if delivery_error:

                        delivery = "Error"

                    elif not send_prompt:

                        delivery = "Not sent"

                    elif pd.notna(
                        receipt_time
                    ):

                        delivery = "Received"

                    elif pd.notna(push_time):
                        delivery = "Push sent (not opened)"

                    else:
                        delivery = "Push not recorded"

                    activity_rows.append(
                        {
                            "Participant": (
                                participant_id
                            ),
                            "Decision made": (
                                decision_time
                            ),
                            "Reason": (
                                decision_activity_reason(
                                    event
                                )
                            ),
                            "Push sent": (
                                push_time
                            ),
                            "Device received": (
                                receipt_time
                            ),
                            "Delivery": (
                                delivery
                            ),
                            "Error": (
                                delivery_error
                            ),
                        }
                    )

                # Yesterday completed scheduled slots

                if day_date == yesterday:
                    completed_count = len(
                        completed_slots
                    )
                    yesterday_completed += completed_count

                    participant_yesterday_completed = (
                        completed_count
                    )

            yesterday_participant_rows.append(
                {
                    "Participant": participant_id,
                    "Completed": participant_yesterday_completed,
                    "Delivered": participant_yesterday_delivered,
                    "Last completed yesterday": participant_yesterday_last,
                    "Distress-active decisions": (
                        distress_active_decisions if yesterday_returned else None
                    ),
                    "Push token": token_status,
                }
            ) 

            # Participant stale state
            #
            # IMPORTANT:
            # This is now based on COMPLETED check-ins,
            # not wearable sync.

            last_completed = (
                max(completed_checkins)
                if completed_checkins
                else pd.NaT
            )

            is_stale = (
                pd.isna(last_completed)
                or last_completed
                < stale_cutoff
            )

            participant_rows.append(
                {
                    "Participant": (
                        participant_id
                    ),
                    "Last completed check-in": (
                        last_completed
                    ),
                    "Stale": (
                        is_stale
                    ),
                    "Last push": (
                        last_push
                    ),
                    "Last receipt": (
                        last_receipt
                    ),
                    "Load error": "",
                    "Push token": token_status,
                }
            )

    # Convert monitoring results to DataFrames

    health = pd.DataFrame(
        participant_rows
    )

    activity = pd.DataFrame(
        activity_rows
    )

    yesterday_by_participant = pd.DataFrame(
        yesterday_participant_rows
    )

    # Summary metrics

    active_participants = len(
        rs_roster
    )

    stale_count = (
        int(
            health["Stale"].sum()
        )
        if not health.empty
        else 0
    )

    missing_token_count = (
        int(health["Push token"].eq("Missing").sum())
        if not health.empty
        else 0
    )

    if not activity.empty:

        error_count = int(
            activity["Error"]
            .fillna("")
            .astype(str)
            .str.strip()
            .ne("")
            .sum()
        )

    else:

        error_count = 0

    yesterday_rate = (
        yesterday_completed
        / yesterday_scheduled
        if yesterday_scheduled
        else None
    )

    # Top metric cards

    metrics = st.columns(5)

    metrics[0].metric(
        "RS participants",
        active_participants,
    )

    metrics[1].metric(
        "No check-in >24h",
        stale_count,
    )

    metrics[2].metric(
        "Missing push tokens",
        missing_token_count,
    )

    metrics[3].metric(
        "Pipeline errors",
        error_count,
    )

    metrics[4].metric(
        "Yesterday EMA",
        (
            f"{yesterday_rate:.0%}"
            if yesterday_rate
            is not None
            else "No scheduled rows"
        ),
        help=(
            f"{yesterday_completed} completed "
            f"of {yesterday_scheduled} scheduled "
            f"check-ins on "
            f"{yesterday:%b %d}."
        ),
    )

    st.markdown(
        "#### Yesterday check-ins by participant"
    )

    if yesterday_by_participant.empty:

        st.info(
            "No participant-level check-in data "
            "was returned for yesterday."
        )

    else:

        yesterday_display = (
            yesterday_by_participant.copy()
        )

        yesterday_display["Check-ins"] = yesterday_display.apply(
            lambda row: (
                f"{int(row['Completed'])} / {int(row['Delivered'])}"
                if pd.notna(row["Completed"]) and pd.notna(row["Delivered"])
                else "Unavailable"
            ),
            axis=1,
        )

        yesterday_display[
            "Last completed yesterday"
        ] = yesterday_display[
            "Last completed yesterday"
        ].apply(
            lambda value: (
                pd.Timestamp(value).strftime(
                    "%I:%M:%S %p ET"
                )
                if pd.notna(value)
                else "None"
            )
        )

        st.dataframe(
            yesterday_display[
                [
                    "Participant",
                    "Check-ins",
                    "Last completed yesterday",
                    "Distress-active decisions",
                    "Push token",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "Distress-active decisions = decisions made while a distress flag "
            "was active yesterday, not the number of new flags raised. "
            "Push token indicates registration, not confirmed delivery. "
            "Unavailable indicates missing timeline information."
        )

    # NEEDS ATTENTION

    st.markdown(
        "#### Needs attention"
    )

    attention_rows = []

    if not health.empty:

        for _, row in health.iterrows():

            reasons = []

            if bool(row["Stale"]):

                reasons.append(
                    "No completed check-in in 24h"
                )

            # A missing device receipt does not establish a failed push.
            if row.get("Push token") == "Missing":
                reasons.append("No registered push token")

            if row.get("Load error"):

                reasons.append(
                    "Monitoring data could not be loaded"
                )

            if reasons:

                attention_rows.append(
                    {
                        "Participant": (
                            row["Participant"]
                        ),
                        "Reason": (
                            "; ".join(
                                reasons
                            )
                        ),
                        "Last completed check-in": (
                            row[
                                "Last completed check-in"
                            ]
                        ),
                        "Last push": (
                            row["Last push"]
                        ),
                        "Last receipt": (
                            row[
                                "Last receipt"
                            ]
                        ),
                    }
                )

    attention_df = pd.DataFrame(
        attention_rows
    )

    if attention_df.empty:

        st.success(
            "No RS participants currently meet "
            "the dashboard's attention rules."
        )

    else:

        attention_display = (
            attention_df.copy()
        )

        for column in [
            "Last completed check-in",
            "Last push",
            "Last receipt",
        ]:

            attention_display[
                column
            ] = attention_display[
                column
            ].apply(
                lambda value: (
                    pd.Timestamp(
                        value
                    ).strftime(
                        "%b %d, %I:%M:%S %p ET"
                    )
                    if pd.notna(value)
                    else "Not recorded"
                )
            )

        st.dataframe(
            attention_display,
            use_container_width=True,
            hide_index=True,
        )

    # DECISION ACTIVITY

    st.markdown(
        "#### Latest decision activity"
    )

    if activity.empty:

        st.info(
            "No recent decision activity "
            "was returned for RS participants."
        )

    else:

        recent = (
            activity.sort_values(
                "Decision made",
                ascending=False,
                na_position="last",
            )
            .head(30)
            .copy()
        )

        for column in [
            "Decision made",
            "Push sent",
            "Device received",
        ]:

            recent[column] = recent[
                column
            ].apply(
                lambda value: (
                    pd.Timestamp(
                        value
                    ).strftime(
                        "%b %d, %I:%M:%S %p ET"
                    )
                    if pd.notna(value)
                    else "—"
                )
            )

        st.dataframe(
            recent[
                [
                    "Participant",
                    "Decision made",
                    "Reason",
                    "Push sent",
                    "Device received",
                    "Delivery",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

    # ST ACCOUNTS

    st.markdown(
        "#### ST accounts"
    )

    if st_roster:

        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Participant": (
                            row[
                                "participant_id"
                            ]
                        )
                    }
                    for row
                    in st_roster
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.caption(
            "No ST study IDs were returned "
            "by the monitoring API."
        )

    # Explain filtered accounts without displaying their IDs.

    if other_roster:

        st.caption(
            f"{len(other_roster)} non-study "
            "account(s) were excluded because "
            "their participant ID does not "
            "begin with RS."
        )

