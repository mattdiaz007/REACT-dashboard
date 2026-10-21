"""Shared constants, data preparation, formatting and backend clients for REACT."""

from pathlib import Path
from typing import Optional
from backend_client import (
    load_backend_health,
    load_monitor_cohort,
    load_monitor_alerts,
    load_monitor_grid,
    load_monitor_participant,
    load_monitor_funnel,
    load_monitor_timeline,
)

import pandas as pd
import streamlit as st
import altair as alt


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_LOG_JSON_PATH = DATA_DIR / "decision_log.json"
DEFAULT_SUMMARY_JSON_PATH = DATA_DIR / "decision_summary.json"
DEFAULT_HRV_CSV_PATH = DATA_DIR / "hrv_5min_series.csv"

DEFAULT_LOG_CSV_PATH = DATA_DIR / "decision_log"
DEFAULT_SUMMARY_CSV_PATH = DATA_DIR / "decision_summary"
LOCAL_TIMEZONE = "America/New_York"

REQUIRED_LOG_COLUMNS = {
    "user_id",
    "timestamp",
    "observed_mssd",
    "user_threshold",
    "send_prompt",
    "decision_reason",
}

REQUIRED_SUMMARY_COLUMNS = {
    "user_id",
    "prompts_sent",
}

# Optional manual overrides for unusual team/test devices that do not carry
# recognizable test metadata. Normally this can stay empty.
DEMO_PARTICIPANT_OVERRIDES = {
    "496",
    "497",
    "530",
    "532",
    "fullcircle-35",
    "fullcircle-36",
    "fullcircle-37",
    "888888",
    "34"
}

# Runtime registry used only for display labels after automatic detection.
DEMO_PARTICIPANTS = {}



def add_demo_flags(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Use backend demo detection, with optional manual overrides as a fallback."""
    flagged = dataframe.copy()

    participant_column = first_existing_column(
        flagged,
        ["participant_id", "user_id"],
    )

    if participant_column is None:
        flagged["is_demo"] = False
        flagged["participant_label"] = ""
        return flagged

    participant_ids = flagged[participant_column].astype(str)

    backend_demo = (
        flagged["is_demo"].fillna(False).astype(bool)
        if "is_demo" in flagged.columns
        else pd.Series(False, index=flagged.index, dtype=bool)
    )
    manual_demo = participant_ids.isin(DEMO_PARTICIPANT_OVERRIDES)

    flagged["is_demo"] = backend_demo | manual_demo

    # Populate display registry from whatever the backend identified this run.
    for participant_id in participant_ids[flagged["is_demo"]].unique():
        DEMO_PARTICIPANTS[str(participant_id)] = "Demo device"

    flagged["participant_label"] = participant_ids.map(
        lambda participant_id: (
            f"{participant_id} — Demo device"
            if participant_id in DEMO_PARTICIPANTS
            else participant_id
        )
    )
    return flagged


def filter_demo_participants(
    dataframe: pd.DataFrame,
    include_demo_devices: bool,
) -> pd.DataFrame:
    """Hide known demo/test devices from operational views when requested."""
    flagged = add_demo_flags(dataframe)
    if include_demo_devices:
        return flagged
    return flagged.loc[~flagged["is_demo"]].copy()


def filter_monitor_participants(
    rows: list[dict],
    include_demo_devices: bool,
) -> list[dict]:
    """Filter known demo/test participants from monitoring API rows."""

    if include_demo_devices:
        return rows

    return [
        row
        for row in rows
        if str(row.get("participant_id", "")).strip()
        not in DEMO_PARTICIPANT_OVERRIDES
    ]


def study_id_group(value) -> str:
    """Classify dashboard participant IDs."""
    label = str(value or "").strip().upper()

    if label.startswith("RS"):
        return "RS"

    if label.startswith("ST"):
        return "ST"

    return "OTHER"


def parse_monitor_time(value):
    """Convert backend timestamp to Eastern time."""
    parsed = pd.to_datetime(value, errors="coerce", utc=True)

    if pd.isna(parsed):
        return pd.NaT

    return parsed.tz_convert(LOCAL_TIMEZONE)


def timeline_days(payload: dict) -> list[dict]:
    """Normalize one-day and multi-day timeline responses."""
    if isinstance(payload.get("days"), list):
        return payload["days"]

    if payload.get("local_date"):
        return [payload]

    return []


def decision_activity_reason(event: dict) -> str:
    """Convert backend decision data into staff-facing reasons."""
    suppression = str(
        event.get("suppression_reason") or ""
    ).strip().lower()

    trigger = str(
        event.get("trigger_reason") or ""
    ).strip().lower()

    if suppression == "run_in" or "run-in" in trigger:
        return "Run in"

    if suppression == "cooldown" or "cooldown" in trigger:
        return "Cooldown"

    if suppression == "daily_cap" or "daily cap" in trigger:
        return "Cap"

    if suppression.startswith("distress_") or "distress" in trigger:
        return "Distress"

    if bool(event.get("send_prompt")):
        return "Sent"

    if event.get("randomization_draw") is not None:
        return "Lost the draw"

    return "Not eligible"


def to_eastern(series: pd.Series) -> pd.Series:
    """Parse timestamps as UTC and convert them to US Eastern time."""
    return pd.to_datetime(series, errors="coerce", utc=True).dt.tz_convert(LOCAL_TIMEZONE)


def validate_columns(
    dataframe: pd.DataFrame,
    required_columns: set[str],
    file_label: str,
) -> None:
    """Stop the app with a clear error when an analysis file is missing columns."""
    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        st.error(f"{file_label} is missing required columns: {missing_text}")
        st.stop()


def convert_to_boolean(series: pd.Series) -> pd.Series:
    """Convert common CSV true/false values into actual Boolean values."""
    normalized = series.astype(str).str.strip().str.lower()

    return (
        normalized.map(
            {
                "true": True,
                "false": False,
                "1": True,
                "0": False,
                "yes": True,
                "no": False,
            }
        )
        .fillna(False)
        .astype(bool)
    )


@st.cache_data
def load_default_data() -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Load bundled seed files, preferring JSON over CSV."""

    if DEFAULT_LOG_JSON_PATH.exists() and DEFAULT_SUMMARY_JSON_PATH.exists():
        return (
            pd.read_json(DEFAULT_LOG_JSON_PATH),
            pd.read_json(DEFAULT_SUMMARY_JSON_PATH),
            "bundled seed JSON files",
        )

    if DEFAULT_LOG_CSV_PATH.exists() and DEFAULT_SUMMARY_CSV_PATH.exists():
        return (
            pd.read_csv(DEFAULT_LOG_CSV_PATH),
            pd.read_csv(DEFAULT_SUMMARY_CSV_PATH),
            "bundled seed CSV files",
        )

    raise FileNotFoundError(
        "No bundled decision_log/decision_summary JSON or CSV pair was found."
    )


@st.cache_data
def load_default_hrv_data() -> pd.DataFrame:
    """Load the bundled five-minute HRV series when it is available."""
    if not DEFAULT_HRV_CSV_PATH.exists():
        return pd.DataFrame()

    hrv_df = pd.read_csv(DEFAULT_HRV_CSV_PATH)
    required_columns = {"user_id", "interval_start", "rmssd"}
    missing_columns = required_columns - set(hrv_df.columns)
    if missing_columns:
        raise ValueError(
            "hrv_5min_series.csv is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    hrv_df = hrv_df.copy()
    hrv_df["interval_start"] = to_eastern(hrv_df["interval_start"])
    for column in ["rmssd", "baseline", "ratio", "count"]:
        if column in hrv_df.columns:
            hrv_df[column] = pd.to_numeric(hrv_df[column], errors="coerce")

    return hrv_df.sort_values(["user_id", "interval_start"])


def clean_data(
    log_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate columns, normalize types, and sort the data."""
    validate_columns(log_df, REQUIRED_LOG_COLUMNS, "decision_log")
    validate_columns(summary_df, REQUIRED_SUMMARY_COLUMNS, "decision_summary")

    log_df = log_df.copy()
    summary_df = summary_df.copy()

    log_df["timestamp"] = to_eastern(log_df["timestamp"])
    log_df["send_prompt"] = convert_to_boolean(log_df["send_prompt"])

    summary_df["prompts_sent"] = pd.to_numeric(
        summary_df["prompts_sent"],
        errors="coerce",
    ).fillna(0).astype(int)

    log_df = log_df.sort_values(
        ["user_id", "timestamp"],
        na_position="last",
    )

    return log_df, summary_df


def build_consistency_table(
    log_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compare prompt totals in the summary against totals from the log."""
    log_prompt_counts = (
        log_df.groupby("user_id")["send_prompt"]
        .sum()
        .astype(int)
        .rename("prompts_from_log")
        .reset_index()
    )

    user_table = summary_df.merge(
        log_prompt_counts,
        on="user_id",
        how="left",
    )

    user_table["prompts_from_log"] = (
        user_table["prompts_from_log"].fillna(0).astype(int)
    )

    user_table["count_matches"] = (
        user_table["prompts_sent"] == user_table["prompts_from_log"]
    )

    return user_table


def first_existing_column(
    dataframe: pd.DataFrame,
    candidates: list[str],
) -> Optional[str]:
    """Return the first candidate column present in the dataframe."""
    return next((column for column in candidates if column in dataframe.columns), None)


def build_pipeline_data(log_df: pd.DataFrame):
    """Prepare notification-pipeline timestamps, latency metrics, and health data."""

    pipeline = log_df.copy()


    column_aliases = {
    "decision_made_at": [
        "decision_made_at",
        "decision_timestamp",
    ],
    "push_sent_at": [
        "push_sent_at",
        "push_timestamp",
        "push_sent_timestamp",
        "last_push_timestamp",
    ],
    "device_received_at": [
        "device_received_at",
        "receipt_timestamp",
        "last_receipt_timestamp",
    ],
    "receipt_reported_at": [
        "receipt_reported_at",
        "reciept_reported_at",
    ],
    "last_sync_at": [
        "last_sync_at",
        "last_synced_at",
        "last_sync_timestamp",
        "sync_timestamp",
        "synced_at",
        "last_sync",
    ],
}

    detected_columns = {}

    for standard_name, candidates in column_aliases.items():
        detected_column = first_existing_column(pipeline, candidates)
        detected_columns[standard_name] = detected_column

        if detected_column:
            pipeline[standard_name] = to_eastern(pipeline[detected_column])
        else:
            pipeline[standard_name] = pd.Series(
                pd.NaT,
                index=pipeline.index,
                dtype=f"datetime64[ns, {LOCAL_TIMEZONE}]",
            )

    # the timestamp design defines these three primary latency stages.
    pipeline["backend_queue_seconds"] = (
        pipeline["push_sent_at"] - pipeline["decision_made_at"]
    ).dt.total_seconds()

    pipeline["push_delivery_seconds"] = (
        pipeline["device_received_at"] - pipeline["push_sent_at"]
    ).dt.total_seconds()

    pipeline["end_to_end_seconds"] = (
        pipeline["device_received_at"] - pipeline["decision_made_at"]
    ).dt.total_seconds()

    # Server-observed fallback when the device and backend clocks disagree.
    pipeline["server_observed_seconds"] = (
        pipeline["receipt_reported_at"] - pipeline["push_sent_at"]
    ).dt.total_seconds()

    latency_columns = [
        "backend_queue_seconds",
        "push_delivery_seconds",
        "end_to_end_seconds",
        "server_observed_seconds",
    ]

    # Negative durations indicate invalid ordering or unsynchronized clocks.
    for column in latency_columns:
        pipeline[column] = pipeline[column].where(pipeline[column] >= 0)


    def determine_delivery_state(row):
        delivery_error = row.get("delivery_error")

        if pd.notna(delivery_error) and str(delivery_error).strip():
            return "Error"

        if pd.notna(row["device_received_at"]):
            return "Healthy"

        if pd.notna(row["push_sent_at"]):
            return "Waiting for device"

        if pd.notna(row["decision_made_at"]):
            return "Waiting for push"

        return "No pipeline activity"

    pipeline["pipeline_state"] = pipeline.apply(
        determine_delivery_state,
        axis=1,
    )

    timestamp_fields_present = any(
        detected_columns[field] is not None
        for field in [
            "decision_made_at",
            "push_sent_at",
            "device_received_at",
            "receipt_reported_at",
        ]
    )

    completed_deliveries = pipeline.dropna(
        subset=["decision_made_at", "device_received_at"]
    ).copy()

    metadata = {
        "timestamp_fields_present": timestamp_fields_present,
        "detected_columns": detected_columns,
        "completed_deliveries": len(completed_deliveries),
    }

    return pipeline, metadata


def format_pipeline_time(value) -> str:
    """Format timestamps compactly for dashboard metric cards."""

    if pd.isna(value):
        return "Not recorded"

    return pd.Timestamp(value).strftime("%b %d, %I:%M:%S %p ET")


def format_latency(value) -> str:
    """Format a latency value in seconds."""

    if pd.isna(value):
        return "Not captured"

    value = float(value)

    if value < 1:
        return f"{value * 1000:.0f} ms"

    return f"{value:.1f} sec"


def build_weekly_summary_pdf(summary: dict, participant_rows: pd.DataFrame) -> bytes:
    """One Letter page with embedded fonts and measured text wrapping."""
    from io import BytesIO
    import reportlab
    from reportlab.pdfgen import canvas
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    fonts = Path(reportlab.__file__).parent / "fonts"
    for name, file_name in [("ReactRegular", "Vera.ttf"), ("ReactBold", "VeraBd.ttf")]:
        if name not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont(name, str(fonts / file_name)))
    output = BytesIO()
    page = canvas.Canvas(output, pagesize=(612, 792))
    page.setTitle("REACT weekly summary")
    page.setAuthor("REACT")

    def rect(x, y, width, height, color):
        page.setFillColorRGB(*color)
        page.rect(x, y, width, height, fill=1, stroke=0)

    def label(x, y, value, size=9, bold=False, color=(.12, .18, .25), width=528):
        font = "ReactBold" if bold else "ReactRegular"
        value = str(value)
        if pdfmetrics.stringWidth(value, font, size) > width:
            while value and pdfmetrics.stringWidth(value + "...", font, size) > width:
                value = value[:-1]
            value += "..."
        page.setFillColorRGB(*color)
        page.setFont(font, size)
        page.drawString(x, y, value)

    rect(0, 684, 612, 108, (.12, .18, .25))
    label(42, 746, "REACT WEEKLY SUMMARY", 18, True, (1, 1, 1), 385)
    label(451, 746, summary["overall_status"].upper(), 9, True, (1, .78, .35), 125)
    label(42, 721, summary["week_label"], 11, color=(.82, .88, .93))
    label(42, 702, summary["source_label"], 8, color=(.82, .88, .93))
    for i, (title, value, detail) in enumerate(summary["cards"]):
        x = 42 + i * 179
        rect(x, 594, 164, 70, (.96, .97, .98))
        label(x + 10, 644, title.upper(), 7, True, width=144)
        label(x + 10, 620, value, 18, True, width=144)
        label(x + 10, 604, detail, 6.5, width=144)
    label(42, 558, "PARTICIPANT SNAPSHOT", 10, True)
    for x, title in [(42, "Participant"), (257, "Completion"), (359, "Delivery"), (478, "Status")]:
        label(x, 525, title.upper(), 7, True, width=92)
    y = 499
    visible = participant_rows.head(6)
    for i, (_, row) in enumerate(visible.iterrows()):
        if i % 2 == 0:
            rect(42, y - 9, 528, 24, (.96, .97, .98))
        for x, key, width in [(48, "Participant", 198), (257, "Completion", 92),
                              (359, "Delivery", 110), (478, "Health", 88)]:
            label(x, y, row[key], 7.5, width=width)
        y -= 27
    if len(participant_rows) > len(visible):
        label(48, y, f"{len(participant_rows) - len(visible)} additional IDs not shown; totals include all IDs.", 8)
        y -= 24
    y = min(290, y - 18)
    label(42, y, "INTERPRETATION AND COVERAGE", 9, True)
    y -= 18
    for note in summary["notes"]:
        line = ""
        for word in note.split():
            candidate = (line + " " + word).strip()
            if line and pdfmetrics.stringWidth(candidate, "ReactRegular", 8) > 528:
                label(42, y, line, 8)
                y -= 11
                line = word
            else:
                line = candidate
        if line:
            label(42, y, line, 8)
            y -= 11
        y -= 5
    label(42, 42, summary["footer"], 6.5)
    page.showPage()
    page.save()
    return output.getvalue()


def build_weekly_summary_data(log_df, pipeline_df, week_start, data_mode, source_name):
    """Observed weekly data only; snapshots never count as notification events."""
    start = pd.Timestamp(week_start, tz=LOCAL_TIMEZONE)
    end = start + pd.DateOffset(days=7)
    now = pd.Timestamp.now(tz=LOCAL_TIMEZONE)
    cutoff = min(end, now)

    def identity(row):
        for key in ["participant_id", "user_id"]:
            value = row.get(key)
            if pd.notna(value) and str(value).strip():
                return str(value).strip()
        return None

    def stamp(row, names):
        for name in names:
            value = pd.to_datetime(row.get(name), errors="coerce", utc=True)
            if pd.notna(value):
                return value.tz_convert(LOCAL_TIMEZONE)
        return pd.NaT

    participants, active, events = set(), set(), []
    unknown = 0
    provenance = data_mode == "Seed" or "weekly_event" in pipeline_df.columns
    for row in pipeline_df.to_dict("records"):
        person = identity(row)
        if person:
            participants.add(person)
        sync = stamp(row, ["last_sync_at", "last_sync_timestamp"])
        if person and pd.notna(sync) and start <= sync < cutoff:
            active.add(person)
        raw = row.get("weekly_event") if data_mode == "Live" else row
        if not isinstance(raw, dict):
            continue
        sent = stamp(raw, ["push_sent_at", "push_timestamp", "push_sent_timestamp"])
        receipt = stamp(raw, ["device_received_at", "receipt_timestamp"])
        decision = stamp(raw, ["decision_made_at", "decision_timestamp"])
        in_week = any(pd.notna(t) and start <= t < cutoff for t in [sent, receipt, decision])
        if person and in_week:
            active.add(person)
        message = raw.get("message_id")
        # Only an explicit notification ID supports deduplication and matching.
        if not person or pd.isna(message) or not str(message).strip():
            unknown += int(in_week)
            continue
        events.append({"Participant": person, "message": str(message).strip(),
                       "sent": sent, "receipt": receipt})

    matched = []
    unmatched = invalid = 0
    if events:
        frame = pd.DataFrame(events)
        for column in ["sent", "receipt"]:
            frame[column] = pd.to_datetime(frame[column], errors="coerce", utc=True)
        for (person, message), group in frame.groupby(["Participant", "message"]):
            sends = group["sent"].dropna().unique()
            receipts = group["receipt"].dropna()
            if len(sends) != 1:
                if any(pd.notna(t) and start <= t < cutoff for t in list(sends) + receipts.tolist()):
                    unmatched += 1
                continue
            sent = pd.Timestamp(sends[0])
            if not start <= sent < cutoff:
                continue
            bad = bool((receipts < sent).any())
            confirmed = bool(((receipts >= sent) & (receipts < cutoff)).any())
            invalid += int(bad)
            matched.append({"Participant": person, "confirmed": confirmed})
    sent_total = len(matched)
    delivered_total = sum(row["confirmed"] for row in matched)
    rows = []
    for person in sorted(participants):
        notifications = [row for row in matched if row["Participant"] == person]
        sent = len(notifications)
        delivered = sum(row["confirmed"] for row in notifications)
        rows.append({"Participant": person, "Completion": "Unavailable",
                     "Delivery": f"{delivered}/{sent}" if sent else "No matched sends",
                     "Health": ("Confirmed" if delivered == sent else "Unconfirmed") if sent else "Not assessed"})
    participant_rows = pd.DataFrame(rows, columns=["Participant", "Completion", "Delivery", "Health"])
    # No live schedule/response endpoint exists in this client. Decision-log rows
    # alone do not establish scheduled assessments, including in seed mode.
    notes = [
        "EMA completion is unavailable: scheduled assessments and responses are not connected.",
        f"Delivery: {delivered_total} of {sent_total} matched notifications sent this week have device confirmation by the reporting cutoff. Missing confirmation does not establish delivery failure.",
        "Coverage: live results use up to 500 returned events; complete weekly history is not verified. Participant counts reflect returned IDs, not verified enrollment." if data_mode == "Live" else "SEED DATA: demonstration only; not pilot results.",
        f"Data checks: {unknown} weekly rows lack a participant or message ID; {unmatched} notification IDs have missing or conflicting sends; {invalid} have receipts before send time. Unmatchable records are excluded.",
        "Active means an observed sync, decision, send or receipt this week. Snapshot syncs do not establish complete activity history."
    ]
    if not provenance:
        notes.append("Event provenance unavailable: replace backend_client.py with the accompanying version.")
    demo_count = int(pipeline_df.loc[pipeline_df.get("is_demo", pd.Series(False, index=pipeline_df.index)).fillna(False).astype(bool)].apply(identity, axis=1).nunique()) if not pipeline_df.empty else 0
    notes.append(f"Demo/test filter: {demo_count} flagged participant IDs included; unflagged test devices may remain.")
    summary = {
        "week_label": f"{start:%B %d} - {(end - pd.DateOffset(days=1)):%B %d, %Y}",
        "source_label": f"{data_mode.upper()} DATA | {'Partial week' if now < end else 'Completed reporting week'} | Eastern time",
        "overall_status": "Limited data" if data_mode == "Live" else "Seed demo",
        "status_color": "1 0.78 0.35",
        "cards": [("Participants observed", str(len(participants)), f"{len(active)} active in available data"),
                  ("EMA completion", "Unavailable", "Schedule / responses not connected"),
                  ("Confirmed delivery", f"{delivered_total / sent_total:.0%}" if sent_total else "Unavailable", f"{delivered_total} / {sent_total} matched sends")],
        "notes": notes,
        "footer": f"Generated {now:%Y-%m-%d %H:%M %Z} | Receipts counted before {cutoff:%Y-%m-%d %H:%M %Z} | IDs only",
    }
    return summary, participant_rows


def build_feasibility_data(log_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Calculate participant- and cohort-level feasibility measures.

    The current synthetic dataset contains scheduled EMA rows and EMA values, so
    response rate and missingness are directly measurable. Response latency is
    calculated when a response timestamp or latency column is supplied. Wear
    gaps are calculated from wearable timestamps when available; otherwise an
    EMA-observation-gap proxy is shown and labeled as such.
    """
    feasibility = log_df.copy()
    feasibility = feasibility.dropna(subset=["user_id", "timestamp"])
    feasibility["date"] = feasibility["timestamp"].dt.date
    feasibility["ema_responded"] = feasibility["ema"].notna() if "ema" in feasibility else False

    latency_column = first_existing_column(
        feasibility,
        ["response_latency_minutes", "latency_minutes", "prompt_to_response_minutes"],
    )
    response_timestamp_column = first_existing_column(
        feasibility,
        ["response_timestamp", "ema_response_timestamp", "completed_at"],
    )

    if latency_column:
        feasibility["latency_minutes"] = pd.to_numeric(
            feasibility[latency_column], errors="coerce"
        )
    elif response_timestamp_column:
        response_times = to_eastern(feasibility[response_timestamp_column])
        feasibility["latency_minutes"] = (
            response_times - feasibility["timestamp"]
        ).dt.total_seconds() / 60
    else:
        feasibility["latency_minutes"] = pd.NA

    # Negative latencies are invalid and should not enter summaries.
    feasibility["latency_minutes"] = pd.to_numeric(
        feasibility["latency_minutes"], errors="coerce"
    ).where(lambda values: values >= 0)

    wearable_timestamp_column = first_existing_column(
        feasibility,
        ["wearable_timestamp", "sensor_timestamp", "device_timestamp"],
    )
    wear_gap_threshold_hours = 2.0

    participant_rows = []
    for user_id, user_data in feasibility.groupby("user_id"):
        user_data = user_data.sort_values("timestamp")
        scheduled = len(user_data)
        completed = int(user_data["ema_responded"].sum())
        response_rate = completed / scheduled if scheduled else 0.0
        missingness = 1.0 - response_rate

        valid_latencies = user_data.loc[
            user_data["ema_responded"], "latency_minutes"
        ].dropna()
        median_latency = valid_latencies.median() if not valid_latencies.empty else pd.NA

        if wearable_timestamp_column:
            wear_times = to_eastern(
                user_data[wearable_timestamp_column]
            ).dropna().sort_values()
            gap_hours = wear_times.diff().dt.total_seconds().div(3600)
            wear_gaps = int((gap_hours > wear_gap_threshold_hours).sum())
            gap_basis = "wearable telemetry"
        else:
            # Current-data proxy: gaps between completed EMA observations.
            observed_times = user_data.loc[user_data["ema_responded"], "timestamp"]
            expected_interval_hours = (
                user_data["timestamp"].sort_values().diff().dt.total_seconds().div(3600).median()
            )
            if pd.isna(expected_interval_hours) or expected_interval_hours <= 0:
                expected_interval_hours = 3.0
            proxy_threshold = max(wear_gap_threshold_hours, expected_interval_hours * 1.5)
            gap_hours = observed_times.diff().dt.total_seconds().div(3600)
            wear_gaps = int((gap_hours > proxy_threshold).sum())
            gap_basis = "EMA observation-gap proxy"

        participant_rows.append(
            {
                "user_id": user_id,
                "scheduled_emas": scheduled,
                "completed_emas": completed,
                "response_rate": response_rate,
                "median_latency_minutes": median_latency,
                "wear_gaps": wear_gaps,
                "missingness": missingness,
            }
        )

    participant_df = pd.DataFrame(participant_rows)

    daily_df = (
        feasibility.groupby("date")
        .agg(
            scheduled_emas=("ema_responded", "size"),
            completed_emas=("ema_responded", "sum"),
        )
        .reset_index()
    )
    daily_df["response_rate"] = (
        daily_df["completed_emas"] / daily_df["scheduled_emas"]
    )
    daily_df["missingness"] = 1.0 - daily_df["response_rate"]

    metadata = {
        "latency_available": feasibility["latency_minutes"].notna().any(),
        "wear_gap_basis": gap_basis if participant_rows else "unavailable",
        "wear_gap_threshold_hours": wear_gap_threshold_hours,
    }
    return participant_df, daily_df, metadata


@st.cache_data(ttl=30)
def load_pipeline_source(data_mode: str) -> tuple[pd.DataFrame, str]:
    """Load the operational source selected by the global dashboard mode."""
    use_mock = data_mode == "Seed"
    return load_backend_health(use_mock=use_mock)


