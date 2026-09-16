How to run

Open Terminal, go into this folder, then run:

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py

## Host on Heroku

The root `Procfile` starts Streamlit on Heroku's assigned port. Heroku installs
the Python packages listed in `requirements.txt`.

1. Sign in to the Heroku CLI with `heroku login`.
2. From this directory, commit the deployment file:

   ```sh
   git add Procfile README.md
   git commit -m "Configure Streamlit for Heroku"
   ```

3. Attach this repository to the existing `react-dashboard-prod` app with a
   remote named for that app, then verify its destination:

   ```sh
   heroku git:remote -a react-dashboard-prod -r react-dashboard-prod
   git remote -v
   ```

   The push URL should be `https://git.heroku.com/react-dashboard-prod.git`.
4. For live backend data, set the API key as a Heroku config var:

   ```sh
   heroku config:set REACT_DASHBOARD_API_KEY="YOUR_API_KEY" -a react-dashboard-prod
   ```

5. Deploy and inspect the app:

   ```sh
   git push react-dashboard-prod main
   heroku open -a react-dashboard-prod
   heroku logs --tail -a react-dashboard-prod
   ```

Keep the API key out of Git. Without it, the live-data view cannot load;
the dashboard can still use its bundled seed data. A Heroku app has a public
URL, so arrange access control before exposing participant data.

The **Weekly summary** view creates a one-page PDF for faculty. Select the
reporting week, review participants, completion, and delivery health, then
download the forward-ready PDF. Live exports never substitute seed completion
data when the live EMA feed is unavailable.

## Live backend setup

The feasibility dashboard reads from two live, read-only backend endpoints:

Backend: `https://react-backend-prod-8db300645555.herokuapp.com`

- `/dashboard/participants/`
- `/dashboard/latency-events/`

The API key must be provided through an environment variable. Do not place the key directly in the source code.

macOS/Linux:
export REACT_DASHBOARD_API_KEY="<actual key>"
export REACT_USE_MOCK_DATA="false"

Windows PowerShell:
$env:REACT_DASHBOARD_API_KEY = "<actual key>"
$env:REACT_USE_MOCK_DATA = "false"

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
    ├── hrv_5min_series.csv
    └── mock_backend_health.json
