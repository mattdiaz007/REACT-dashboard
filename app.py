"""
Installation
------------
From Terminal, inside the REACT-dashboard folder:

    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install streamlit pandas reportlab
    streamlit run app.py
"""

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

st.markdown(
    """
    <style>
    div[data-testid="stMetricValue"] {
        font-size: 1.65rem;
        line-height: 1.2;
    }

    div[data-testid="stMetricLabel"] {
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.set_page_config(
    page_title="REACT Decision Dashboard",
    page_icon="📊",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_LOG_JSON_PATH = DATA_DIR / "decision_log.json"
DEFAULT_SUMMARY_JSON_PATH = DATA_DIR / "decision_summary.json"
DEFAULT_HRV_CSV_PATH = DATA_DIR / "hrv_5min_series.csv"

DEFAULT_LOG_CSV_PATH = DATA_DIR / "decision_log"
DEFAULT_SUMMARY_CSV_PATH = DATA_DIR / "decision_summary"
LOCAL_TIMEZONE = "America/New_York"

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


def to_eastern(series: pd.Series) -> pd.Series:
    """Parse timestamps as UTC and convert them to US Eastern time."""
    return pd.to_datetime(series, errors="coerce", utc=True).dt.tz_convert(LOCAL_TIMEZONE)

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
    all_latencies = pd.to_numeric(
        log_df.get("response_latency_minutes", pd.Series(dtype=float)), errors="coerce"
    ).dropna()

    # Use computed latency values so timestamp-derived latency is also supported.
    feasibility_copy = log_df.copy()
    _, _, _ = participant_df, daily_df, metadata
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


def render_daily_monitoring_view(log_df: pd.DataFrame, data_mode: str, include_demo_devices: bool) -> None:
    """Render the single morning screen used to identify pilot issues quickly."""
    st.header("Daily Monitoring")
    st.caption(
        "Morning operational check: data freshness, delivery health, yesterday's "
        "EMA completion, and participants needing attention. Times are Eastern."
    )

    if data_mode == "Live":
        st.success("LIVE DATA MODE — only live backend fields are displayed. Seed values are never substituted.")
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
    if is_seed_source:
        st.info(
            "SEED DATA MODE — use this screen to test layout only. Stale counts and "
            "operational alerts are intentionally disabled."
        )
    else:
        st.success(f"LIVE DATA MODE — source: {source_name}")

    if load_error:
        st.warning("The live monitoring source could not be loaded.")
        st.code(load_error)

    participant_column = "participant_id" if "participant_id" in pipeline_df.columns else "user_id"
    now_et = pd.Timestamp.now(tz=LOCAL_TIMEZONE)
    yesterday = (now_et - pd.Timedelta(days=1)).date()
    stale_cutoff = now_et - pd.Timedelta(hours=24)

    health = (
        pipeline_df.groupby(participant_column, as_index=False)
        .agg(
            last_sync=("last_sync_at", "max"),
            last_push=("push_sent_at", "max"),
            last_receipt=("device_received_at", "max"),
            last_backend_receipt=("receipt_reported_at", "max"),
        )
    ) if participant_column in pipeline_df.columns else pd.DataFrame()

    if not health.empty:
        health["stale"] = False if is_seed_source else (
            health["last_sync"].isna() | health["last_sync"].lt(stale_cutoff)
        )

        # Determine waiting status from each participant's latest delivery event.
        # This prevents an older receipt from masking a newer unreceived push.
        latest_events = pipeline_df.copy()
        latest_events["_event_sort_time"] = latest_events["push_sent_at"].fillna(
            latest_events["decision_made_at"]
        )
        latest_events = (
            latest_events.sort_values(
                [participant_column, "_event_sort_time"],
                ascending=[True, True],
                na_position="first",
            )
            .groupby(participant_column, as_index=False)
            .tail(1)
        )
        latest_events = latest_events[[
            participant_column, "push_sent_at", "device_received_at"
        ]].copy()
        latest_events["waiting"] = (
            latest_events["push_sent_at"].notna()
            & latest_events["device_received_at"].isna()
        )
        health = health.merge(
            latest_events[[participant_column, "waiting"]],
            on=participant_column,
            how="left",
        )
        health["waiting"] = health["waiting"].fillna(False).astype(bool)
    active_participants = int(health[participant_column].nunique()) if not health.empty else 0
    stale_count = 0 if is_seed_source or health.empty else int(health["stale"].sum())
    waiting_count = int(health["waiting"].sum()) if not health.empty else 0
    error_count = int((pipeline_df["pipeline_state"] == "Error").sum())

    if data_mode == "Seed":
        feasibility = log_df.dropna(subset=["user_id", "timestamp"]).copy()
        feasibility["local_date"] = feasibility["timestamp"].dt.date
        feasibility["ema_completed"] = feasibility["ema"].notna() if "ema" in feasibility else False
        yesterday_rows = feasibility[feasibility["local_date"] == yesterday]
        yesterday_scheduled = len(yesterday_rows)
        yesterday_completed = int(yesterday_rows["ema_completed"].sum()) if yesterday_scheduled else 0
        yesterday_rate = yesterday_completed / yesterday_scheduled if yesterday_scheduled else None
    else:
        yesterday_rows = pd.DataFrame()
        yesterday_scheduled = 0
        yesterday_completed = 0
        yesterday_rate = None

    metrics = st.columns(5)
    metrics[0].metric("Participants", active_participants)
    metrics[1].metric("Stale >24h", "Disabled" if is_seed_source else stale_count)
    metrics[2].metric("Waiting receipts", waiting_count)
    metrics[3].metric("Pipeline errors", error_count)
    metrics[4].metric(
        "Yesterday EMA",
        (f"{yesterday_rate:.0%}" if yesterday_rate is not None else "No rows")
        if data_mode == "Seed" else "Not live yet",
        help=(
            f"{yesterday_completed} completed of {yesterday_scheduled} scheduled on {yesterday:%b %d}."
            if data_mode == "Seed"
            else "EMA completion is not yet supplied by the live backend, so seed EMA data is not shown in Live mode."
        )
    )

    st.markdown("#### Needs attention")
    attention_rows = []
    if not health.empty:
        for _, row in health.iterrows():
            reasons = []
            if not is_seed_source and row["stale"]:
                reasons.append("No sync in 24h")
            if row["waiting"]:
                reasons.append("Push waiting for receipt")
            if reasons:
                participant_value = str(row[participant_column])
                attention_rows.append({
                    "Participant": (
                        f"{participant_value} — {DEMO_PARTICIPANTS[participant_value]}"
                        if participant_value in DEMO_PARTICIPANTS
                        else participant_value
                    ),
                    "Reason": "; ".join(reasons),
                    "Last sync": row["last_sync"],
                    "Last push": row["last_push"],
                    "Last receipt": row["last_receipt"],
                })

    participant_yesterday = (
        yesterday_rows.groupby("user_id")
        .agg(Scheduled=("ema_completed", "size"), Completed=("ema_completed", "sum"))
        .reset_index()
    ) if data_mode == "Seed" and not yesterday_rows.empty else pd.DataFrame()
    if not participant_yesterday.empty:
        participant_yesterday["Rate"] = participant_yesterday["Completed"] / participant_yesterday["Scheduled"]
        for _, row in participant_yesterday[participant_yesterday["Rate"] < 0.5].iterrows():
            attention_rows.append({
                "Participant": row["user_id"],
                "Reason": f"Yesterday EMA completion {row['Rate']:.0%}",
                "Last sync": pd.NaT,
                "Last push": pd.NaT,
                "Last receipt": pd.NaT,
            })

    attention_df = pd.DataFrame(attention_rows)

    if attention_df.empty:
        st.success("No participants currently meet the dashboard's attention rules.")
    else:
        attention_display = attention_df.copy()

        for column in ["Last sync", "Last push", "Last receipt"]:
            if column in attention_display.columns:
                attention_display[column] = attention_display[column].apply(
                    lambda value: (
                        pd.Timestamp(value).strftime("%b %d, %I:%M:%S %p ET")
                        if pd.notna(value)
                        else "Not recorded"
                    )
                )

        st.dataframe(
            attention_display,
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("#### Latest operational activity")

    recent = pipeline_df.sort_values(
        "decision_made_at",
        ascending=False,
        na_position="last",
    ).head(20)

    recent_columns = [
        c
        for c in [
            participant_column,
            "decision_made_at",
            "push_sent_at",
            "device_received_at",
            "pipeline_state",
        ]
        if c in recent.columns
    ]

    recent_display = recent[recent_columns].copy()

    if participant_column in recent_display.columns:
        recent_display[participant_column] = recent_display[participant_column].astype(str).map(
            lambda participant_id: (
                f"{participant_id} — {DEMO_PARTICIPANTS[participant_id]}"
                if participant_id in DEMO_PARTICIPANTS
                else participant_id
            )
        )

    for column in [
        "decision_made_at",
        "push_sent_at",
        "device_received_at",
    ]:
        if column in recent_display.columns:
            recent_display[column] = recent_display[column].apply(
                lambda value: (
                    pd.Timestamp(value).strftime("%b %d, %I:%M:%S %p ET")
                    if pd.notna(value)
                    else "Not recorded"
                )
            )

    recent_display = recent_display.rename(
        columns={
            participant_column: "Participant",
            "decision_made_at": "Decision made",
            "push_sent_at": "Push sent",
            "device_received_at": "Device received",
            "pipeline_state": "Status",
        }
    )

    st.dataframe(
        recent_display,
        use_container_width=True,
        hide_index=True,
    )


def render_weekly_summary_view(log_df, data_mode, include_demo_devices):
    """Render and export a faculty-ready one-page weekly summary."""
    st.header("Weekly Summary")
    st.caption("A one-page faculty update covering participants, completion, and delivery health.")
    try:
        source_df, source_name = load_pipeline_source(data_mode)
        pipeline_df, _ = build_pipeline_data(source_df)
        if data_mode == "Live":
            pipeline_df = filter_demo_participants(pipeline_df, include_demo_devices)
    except Exception as exc:
        if data_mode == "Live":
            st.error("Live backend data could not be loaded. Seed data was not substituted.")
            st.code(str(exc))
            return
        source_name = "seed decision-log fallback"
        pipeline_df, _ = build_pipeline_data(log_df)

    today = pd.Timestamp.now(tz=LOCAL_TIMEZONE)
    default_monday = (today - pd.DateOffset(days=today.weekday() + 7)).date()
    selected_monday = st.date_input("Week beginning", value=default_monday,
                                    help="Reports run Monday through Sunday in US Eastern time.")
    selected_monday = selected_monday - pd.Timedelta(days=selected_monday.weekday())
    summary, participant_rows = build_weekly_summary_data(log_df, pipeline_df, selected_monday, data_mode, source_name)

    if data_mode == "Seed":
        st.info("SEED DATA MODE — the export is clearly labeled as seed data.")
    else:
        st.success(
            "LIVE DATA MODE — demo/test devices follow the dashboard's current filter."
        )
    st.subheader(summary["week_label"])
    for column, (title, value, detail) in zip(st.columns(3), summary["cards"]):
        column.metric(title, value)
        column.caption(detail)
    if data_mode == "Live":
        st.warning("Live EMA completion is not connected yet; the PDF states this explicitly instead of substituting seed values.")
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



def render_participant_detail_view(log_df: pd.DataFrame, summary_df: pd.DataFrame, data_mode: str, include_demo_devices: bool) -> None:
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


def render_hrv_summary_view(data_mode: str) -> None:
    """Render the bundled HRV export as a selectable dashboard view."""
    st.header("HRV Summary")
    st.caption(
        "Five-minute RMSSD trends from the supplied synthetic HRV export. "
        "Times are Eastern."
    )

    if data_mode == "Live":
        st.info(
            "The current HRV export is bundled seed data. Switch the global data mode "
            "to Seed to view this summary."
        )
        return

    try:
        hrv_df = load_default_hrv_data()
    except ValueError as exc:
        st.error(str(exc))
        return

    valid_hrv = hrv_df.dropna(subset=["user_id", "interval_start", "rmssd"])
    if valid_hrv.empty:
        st.info("No HRV observations are available in data/hrv_5min_series.csv.")
        return

    metric_columns = st.columns(4)
    metric_columns[0].metric("Participants", valid_hrv["user_id"].nunique())
    metric_columns[1].metric("Valid intervals", len(valid_hrv))
    metric_columns[2].metric("Average RMSSD", f"{valid_hrv['rmssd'].mean():.1f} ms")
    metric_columns[3].metric("Peak RMSSD", f"{valid_hrv['rmssd'].max():.1f} ms")

    class_counts = (
        valid_hrv["hrv_class"].fillna("Unclassified").value_counts()
        if "hrv_class" in valid_hrv.columns
        else pd.Series(dtype=int)
    )
    left_column, right_column = st.columns([2, 1])
    with left_column:
        st.subheader("HRV class distribution")
        if class_counts.empty:
            st.info("No HRV classifications are present.")
        else:
            st.bar_chart(class_counts.rename("Intervals"), height=260)
    with right_column:
        selected_user = st.selectbox(
            "Participant",
            sorted(valid_hrv["user_id"].astype(str).unique()),
        )

    selected_hrv = valid_hrv[valid_hrv["user_id"].astype(str) == selected_user]
    chart_data = selected_hrv.set_index("interval_start")[["rmssd"]].rename(
        columns={"rmssd": "RMSSD (ms)"}
    )
    if "baseline" in selected_hrv.columns and selected_hrv["baseline"].notna().any():
        chart_data["Baseline (ms)"] = selected_hrv.set_index("interval_start")["baseline"]

    st.subheader(f"RMSSD over time: {selected_user}")
    st.line_chart(chart_data, height=340)
    st.caption(
        "RMSSD is the primary HRV measure. Baseline and HRV class are carried through "
        "from the source export and do not alter decision-engine results."
    )

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

    benchmarks = snapshot.get("benchmarks", {})

    benchmark_labels = {
        "slot_coverage": "Slot coverage",
        "prompt_response": "Prompt response",
        "wear": "Wear coverage",
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
        "Last sync: "
        f"**{format_monitor_age(rollup.get('last_sync_at'))}**"
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
        "Wear coverage": "wear_valid_pct",
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
    st.caption(
        "Participant-level diagnostic view. Events share a 24-hour clock "
        "so wearable, check-in, decision, and delivery activity can be compared."
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

        st.markdown("#### Wear")

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

        st.markdown("#### Sync")

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

# Sidebar data selection
st.sidebar.header("Data mode")

data_mode = st.sidebar.radio(
    "Choose dashboard source",
    ["Live", "Seed"],
    index=0,
    help=(
        "Live uses only the backend. Seed uses bundled/mock data for testing. "
        "The dashboard never substitutes seed values while Live mode is selected."
    ),
)

if data_mode == "Live":
    st.sidebar.success("LIVE DATA")
    include_demo_devices = st.sidebar.toggle(
        "Include demo/test devices",
        value=False,
        help=(
            "Live test/demo devices are detected automatically from backend metadata "
            "and hidden from participant counts, alerts, and participant lists unless enabled."
        ),
    )
    if not include_demo_devices:
        st.sidebar.caption("Automatically detected demo/test devices are excluded from live monitoring.")
else:
    st.sidebar.info("SEED DATA — DEMO ONLY")
    include_demo_devices = True

use_uploaded_files = False
if data_mode == "Seed":
    use_uploaded_files = st.sidebar.toggle(
        "Upload different seed analysis files",
        value=False,
        help="Supports JSON or CSV decision log and summary files.",
    )

if use_uploaded_files:
    uploaded_log = st.sidebar.file_uploader(
        "Upload decision_log",
        type=["json", "csv"],
    )
    uploaded_summary = st.sidebar.file_uploader(
        "Upload decision_summary",
        type=["json", "csv"],
    )

    if uploaded_log is None or uploaded_summary is None:
        st.info("Upload both analysis files to continue.")
        st.stop()

    if uploaded_log.name.lower().endswith(".json"):
        raw_log_df = pd.read_json(uploaded_log)
    else:
        raw_log_df = pd.read_csv(uploaded_log)

    if uploaded_summary.name.lower().endswith(".json"):
        raw_summary_df = pd.read_json(uploaded_summary)
    else:
        raw_summary_df = pd.read_csv(uploaded_summary)

    decision_data_source = "uploaded seed analysis files"
else:
    try:
        raw_log_df, raw_summary_df, decision_data_source = load_default_data()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

# Seed analysis files remain loaded so Seed mode can use them, but Live mode
# renderers must never surface them as live values.
log_df, summary_df = clean_data(raw_log_df, raw_summary_df)
user_table = build_consistency_table(log_df, summary_df)



# Page navigation
st.sidebar.divider()
selected_view = st.sidebar.radio(
    "Dashboard view",
    [
        "Daily monitoring",
        "Cohort analytics",
        "Participant Grid",
        "Participant timeline",
        "Weekly summary",
        "Participant detail",
        "HRV summary",
        "Decision engine",
        "Feasibility",
    ],
    index=0,
)

st.title("REACT Decision Dashboard")
if data_mode == "Live":
    st.caption("GLOBAL MODE: LIVE — backend data only. Seed values are never substituted.")
else:
    st.caption(f"GLOBAL MODE: SEED — {decision_data_source}; mock operational data. (4 demo participants)")

if selected_view == "Daily monitoring":
    render_daily_monitoring_view(log_df, data_mode, include_demo_devices)
elif selected_view == "Weekly summary":
    render_weekly_summary_view(log_df, data_mode, include_demo_devices)
elif selected_view == "Participant detail":
    render_participant_detail_view(log_df, summary_df, data_mode, include_demo_devices)
elif selected_view == "HRV summary":
    render_hrv_summary_view(data_mode)
elif selected_view == "Cohort analytics":
    render_cohort_view(data_mode)
elif selected_view == "Participant Grid":
    render_participant_grid_view(
        data_mode,
        include_demo_devices
    )
elif selected_view == "Participant timeline":
    render_participant_timeline_view(
        data_mode,
        include_demo_devices
    )
elif selected_view == "Feasibility":
    if data_mode == "Live":
        st.header("Feasibility Results")
        st.info(
            "Live feasibility/EMA data is not connected yet. Switch to Seed mode to view "
            "the synthetic feasibility analysis. No seed values are shown in Live mode."
        )
    else:
        render_feasibility_view(log_df, data_mode, include_demo_devices)
else:
    if data_mode == "Live":
        st.header("Decision Engine")
        st.info(
            "Live decision-engine output is not connected yet. Switch to Seed mode to view "
            "the synthetic decision log and summary. No seed values are shown in Live mode."
        )
    else:
        render_decision_view(log_df, summary_df, user_table)
