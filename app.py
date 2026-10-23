"""REACT dashboard Streamlit entry point. Run: streamlit run app.py."""

import pandas as pd
import streamlit as st
import os
import hmac

st.set_page_config(
    page_title="REACT Decision Dashboard",
    page_icon="📊",
    layout="wide",
)

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

def require_login():
    """Keep dashboard views and live API calls behind a shared team password."""
    password_config = os.getenv("DASHBOARD_PASSWORD", "")
    if not password_config:
        try:
            password_config = str(st.secrets.get("DASHBOARD_PASSWORD", ""))
        except (FileNotFoundError, KeyError):
            password_config = ""

    if not password_config:
        st.error("Dashboard access is not configured. Set DASHBOARD_PASSWORD in Heroku Config Vars.")
        st.stop()

    if st.session_state.get("react_authenticated", False):
        return

    st.title("REACT Research Dashboard")
    st.subheader("Team Login")
    with st.form("react_login_form"):
        entered_password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log In")

    if submitted:
        if hmac.compare_digest(entered_password, password_config):
            st.session_state["react_authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()


require_login()



from dashboard_core import load_default_data, clean_data, build_consistency_table
from views.daily_monitoring import render_daily_monitoring_view
from views.participant_detail import render_participant_detail_view, render_participant_detail_seed_view
from views.weekly_summary import render_weekly_summary_view, render_weekly_summary_seed_view
from views.feasibility import render_feasibility_view, render_live_feasibility_view
from views.decision_engine import render_live_decision_view, render_decision_view
from views.hrv import render_hrv_summary_view
from views.cohort import render_cohort_view
from views.participant_grid import render_participant_grid_rail, render_participant_grid_view
from views.participant_timeline import render_participant_timeline_view

# Sidebar data selection
st.sidebar.header("Data mode")

if st.sidebar.button("Log Out"):
    st.session_state["react_authenticated"] = False
    st.rerun()

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

# Keep seed analysis optional. Live pilot views read only their live endpoints.
log_df = pd.DataFrame()
summary_df = pd.DataFrame()
user_table = pd.DataFrame()
decision_data_source = ""

if data_mode == "Seed":
    use_uploaded_files = st.sidebar.toggle(
        "Upload different seed analysis files",
        value=False,
        help="Supports JSON or CSV decision log and summary files.",
    )
    if use_uploaded_files:
        uploaded_log = st.sidebar.file_uploader("Upload decision_log", type=["json", "csv"])
        uploaded_summary = st.sidebar.file_uploader("Upload decision_summary", type=["json", "csv"])
        if uploaded_log is None or uploaded_summary is None:
            st.info("Upload both analysis files to continue.")
            st.stop()
        raw_log_df = (pd.read_json(uploaded_log) if uploaded_log.name.lower().endswith(".json")
                      else pd.read_csv(uploaded_log))
        raw_summary_df = (pd.read_json(uploaded_summary) if uploaded_summary.name.lower().endswith(".json")
                          else pd.read_csv(uploaded_summary))
        decision_data_source = "uploaded seed analysis files"
    else:
        try:
            raw_log_df, raw_summary_df, decision_data_source = load_default_data()
        except FileNotFoundError as exc:
            st.error(str(exc))
            st.stop()
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
        render_live_feasibility_view(include_demo_devices)
    else:
        render_feasibility_view(log_df, data_mode, include_demo_devices)
else:
    if data_mode == "Live":
        render_live_decision_view(include_demo_devices)
    else:
        render_decision_view(log_df, summary_df, user_table)