How to run

Open Terminal, go into this folder, then run:

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py

The **Weekly summary** view creates a one-page PDF for faculty. Select the
reporting week, review participants, completion, and delivery health, then
download the forward-ready PDF. Live exports never substitute seed completion
data when the live EMA feed is unavailable.

## Live backend setup

The feasibility dashboard reads from two live, read-only backend endpoints:

- `/dashboard/participants/`
- `/dashboard/latency-events/`

The API key must be provided through an environment variable. Do not place the key directly in the source code.

macOS/Linux:
export REACT_DASHBOARD_API_KEY="<actual key>"
export REACT_USE_MOCK_DATA="false"

To run with live data:

export REACT_DASHBOARD_API_KEY="<actual key>"
streamlit cache clear
streamlit run app.py

## Project structure

REACT-dashboard/
├── app.py
├── backend_client.py
├── requirements.txt
├── README.md
└── data/
    ├── decision_log.csv
    ├── decision_summary.csv
    └── mock_backend_health.json
