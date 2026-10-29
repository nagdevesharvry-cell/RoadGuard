# RoadGuard Maharashtra — Intelligent Traffic Safety Prototype

A Streamlit prototype upgraded from the supplied `accident-risk-platform` project. It reuses the supplied accident CSV and the existing `src/data.py`, `src/hotspots.py`, and `src/models.py` logic where compatible.

## What is included

- Login/signup screen with **demo mode** for immediate evaluation.
- Optional Firebase email/password sign-in and account creation.
- Optional Google OpenID Connect sign-in through Streamlit's authentication support.
- English, Hindi, and Marathi UI language options.
- Dark modern dashboard, overview metrics, interactive historical hotspot analysis, downloadable segment CSV.
- Maharashtra reference map with city markers; optional clearly labelled illustrative demo overlays.
- Route geometry comparison via the public OSRM routing endpoint (requires internet).
- Severity-class what-if prediction using the existing Random Forest model.
- Session-only hazard report form.
- Setup/data-limitations page.

## Important accuracy and safety limitation

The bundled CSV has 8,116 raw records and chainage kilometre values, but no latitude/longitude and no verified named-road/corridor-to-Maharashtra mapping. Therefore:
- The existing hotspot analysis ranks segments on the source corridor only. It must **not** be described as a verified Maharashtra-wide map.
- City markers are reference locations, not accident locations.
- Any optional map circles and route risk scores are illustrative UI demo data, not actual accident predictions.
- The existing classifier estimates severity class **among recorded accidents**. It does not predict whether an accident will occur at a GPS location.
- The labels in `data/labels.csv` explicitly mark severity descriptions as assumptions. Verify the original dataset codebook before making claims about fatality/severity classes.
- Route geometry from OSRM is not itself a safety rating. A defensible route risk score needs verified geospatial accident points/road segments, exposure data, and validation.

Do not use this prototype as an emergency alert or as the sole basis for real-world route decisions.

## Run locally on Windows

Open PowerShell in the `accident-risk-platform` folder:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks activation, run this for the current terminal only and activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Open the local URL Streamlit prints (usually `http://localhost:8501`). You can use **Explore interactive demo** without configuring authentication.

## Optional: Firebase email/password authentication

1. Create a Firebase project.
2. In Firebase Authentication, enable the **Email/Password** provider.
3. Copy the Firebase Web API key.
4. Create `.streamlit/secrets.toml` locally (do not commit it) using:

```toml
[firebase]
api_key = "YOUR_FIREBASE_WEB_API_KEY"
```

The app sends email/password sign-in and account-creation requests to Firebase Authentication. Passwords are not saved in this app. For a production app, add token refresh, account recovery, logout/token invalidation, rate limiting, and server-side authorization as appropriate.

## Optional: Google sign-in (Streamlit OIDC)

Google sign-in uses Streamlit's OIDC support. Configure OAuth consent and a Web OAuth client in Google Cloud. Set the authorized redirect URI to your deployed app's `/_stcore/authorize` endpoint (and the matching localhost redirect during local development). Then add the following to `.streamlit/secrets.toml`:

```toml
[auth]
redirect_uri = "http://localhost:8501/oauth2callback"
cookie_secret = "REPLACE_WITH_A_LONG_RANDOM_SECRET"
client_id = "YOUR_GOOGLE_OAUTH_CLIENT_ID"
client_secret = "YOUR_GOOGLE_OAUTH_CLIENT_SECRET"
server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"
```

Follow the current Streamlit OIDC documentation for the exact redirect URI and configuration required by your Streamlit version/deployment. Never upload `secrets.toml` or publish OAuth secrets. If Google OIDC is not configured, the Google button is disabled and clearly says setup is required.

For Streamlit Community Cloud, enter credentials under the app's **Secrets** settings instead of committing a secrets file. Use the deployed app's required redirect URI and authorized domain in Google Cloud.

## Deployment with Antigravity

1. Extract this ZIP locally.
2. Open the extracted project folder in Antigravity.
3. Ask Antigravity to review and run the app; do not ask it to fabricate a GPS mapping for the dataset.
4. Install packages from `requirements.txt`.
5. Run `streamlit run app.py` and test the demo flow.
6. Configure secrets only in the deployment platform's secret manager.
7. Deploy using a platform that supports Streamlit. If Antigravity is used to edit code, it does not itself guarantee a public deployment URL.

## Suggested next dataset improvement

Obtain a verified source with at least:
- accident ID/date/time and severity codebook,
- latitude/longitude or official road-segment IDs,
- road-segment geometry for Maharashtra,
- exposure/traffic volume (if available),
- consistent weather/road conditions,
- licence/terms that allow redistribution.

Then spatially join accidents to road segments, calibrate a segment-level risk method, backtest on held-out time periods, and only then show data-backed route-risk recommendations.


### RoadGuard Assistant chatbot
The sidebar now includes RoadGuard chatbot in English, Hindi and Marathi. This prototype assistant uses transparent intent-based answers and current app/data context; it requires no LLM API key. It explains map and route-data limitations rather than inventing real-time risk predictions.
