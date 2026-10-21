"""REACT hrv view."""

from dashboard_core import *


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

