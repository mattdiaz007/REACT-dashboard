from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd
import streamlit as st


LOCAL_TIMEZONE = "America/New_York"


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MOCK_PATH = BASE_DIR / "data" / "mock_backend_health.json"

BACKEND_BASE_URL = "https://react-backend-prod-8db300645555.herokuapp.com"
PARTICIPANTS_URL = f"{BACKEND_BASE_URL}/dashboard/participants/"
LATENCY_EVENTS_URL = f"{BACKEND_BASE_URL}/dashboard/latency-events/?limit=500"


def _read_json(url: str, api_key: str) -> Any:
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "X-Dashboard-API-Key": api_key,
        },
        method="GET",
    )

    try:
        with urlopen(request, timeout=15) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise RuntimeError(f"Backend returned HTTP {exc.code} for {url}") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not reach backend: {exc.reason}") from exc


def _load_mock() -> list[dict[str, Any]]:
    with DEFAULT_MOCK_PATH.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, list):
        raise ValueError("Mock backend file must contain a JSON list.")

    return payload


def _load_live(api_key: str) -> list[dict[str, Any]]:
    participants = _read_json(PARTICIPANTS_URL, api_key)
    latency_events = _read_json(LATENCY_EVENTS_URL, api_key)

    if not isinstance(participants, list):
        raise ValueError("Participants endpoint did not return a JSON list.")
    if not isinstance(latency_events, list):
        raise ValueError("Latency endpoint did not return a JSON list.")

    participant_map: dict[Any, dict[str, Any]] = {}
    for row in participants:
        key = row.get("participant_id") or row.get("user_id")
        participant_map[key] = dict(row)

    combined_rows: list[dict[str, Any]] = []
    event_keys = set()

    for event in latency_events:
        key = event.get("participant_id") or event.get("user_id")
        event_keys.add(key)
        combined = participant_map.get(key, {}).copy()
        combined.update(event)
        # Preserve raw event provenance before participant snapshots are merged.
        combined["weekly_event"] = dict(event)
        combined_rows.append(combined)

    for key, row in participant_map.items():
        if key not in event_keys:
            combined_rows.append({**row, "weekly_event": None})

    return combined_rows



def _classify_demo_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Identify obvious live test/demo records from backend metadata.

    This is an interim safeguard until the backend supplies an explicit
    participant-type field. Strong test markers currently include:
    - reserved @example.invalid email addresses
    - email local-parts beginning with test/demo
    - simulated prompt message IDs beginning with PROMPT-SIM-
    """
    classified = frame.copy()

    existing_demo = (
        classified["is_demo"].fillna(False).astype(bool)
        if "is_demo" in classified.columns
        else pd.Series(False, index=classified.index, dtype=bool)
    )

    email = classified.get(
        "email",
        pd.Series("", index=classified.index, dtype="object"),
    ).fillna("").astype(str).str.strip().str.lower()

    message_id = classified.get(
        "message_id",
        pd.Series("", index=classified.index, dtype="object"),
    ).fillna("").astype(str).str.strip().str.upper()

    reserved_test_email = email.str.endswith("@example.invalid")
    named_test_email = email.str.match(r"^(test|demo)([-+_.]|@)")
    simulated_prompt = message_id.str.startswith("PROMPT-SIM-")

    classified["is_demo"] = (
        existing_demo
        | reserved_test_email
        | named_test_email
        | simulated_prompt
    )

    reasons = []
    for idx in classified.index:
        row_reasons = []
        if bool(existing_demo.loc[idx]):
            row_reasons.append("backend flag")
        if bool(reserved_test_email.loc[idx]):
            row_reasons.append("reserved test email")
        elif bool(named_test_email.loc[idx]):
            row_reasons.append("test/demo email")
        if bool(simulated_prompt.loc[idx]):
            row_reasons.append("simulated prompt id")
        reasons.append("; ".join(row_reasons))

    classified["demo_reason"] = reasons
    return classified

def load_backend_health(use_mock: Optional[bool] = None) -> tuple[pd.DataFrame, str]:
    """Load either the live backend or seed/mock backend data explicitly.

    When ``use_mock`` is None, preserve the environment-variable behavior for
    backward compatibility. Passing True/False lets the dashboard's global
    Data mode switch control the source without silently mixing modes.
    """
    if use_mock is None:
        use_mock = os.getenv("REACT_USE_MOCK_DATA", "false").strip().lower() in {
            "1", "true", "yes", "on"
        }

    if use_mock:
        records = _load_mock()
        source = "seed data"
        is_seed_data = True
    else:
        api_key = os.getenv("REACT_DASHBOARD_API_KEY", "").strip()
        if not api_key:
            try:
                api_key = str(st.secrets["REACT_DASHBOARD_API_KEY"]).strip()
            except Exception:
                api_key = ""

        if not api_key:
            raise RuntimeError(
                "REACT_DASHBOARD_API_KEY is not configured."
            )
        records = _load_live(api_key)
        source = "live backend"
        is_seed_data = False

    frame = pd.DataFrame(records)
    frame["data_source"] = source
    frame["is_seed_data"] = is_seed_data

    if is_seed_data:
        frame["is_demo"] = False
        frame["demo_reason"] = ""
    else:
        frame = _classify_demo_rows(frame)

    timestamp_columns = [
        "last_sync_timestamp",
        "last_push_timestamp",
        "last_receipt_timestamp",
        "decision_made_at",
        "push_sent_timestamp",
        "receipt_timestamp",
        "receipt_reported_at",
    ]

    for column in timestamp_columns:
        if column in frame.columns:
            frame[column] = (
                pd.to_datetime(frame[column], errors="coerce", utc=True)
                .dt.tz_convert(LOCAL_TIMEZONE)
            )

    return frame, source
