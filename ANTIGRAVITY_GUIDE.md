# Antigravity build/deploy guide — RoadGuard Maharashtra

This repository is a Streamlit Python application. Start by opening this folder in Antigravity.

## First prompt to Antigravity

> Review the existing RoadGuard Maharashtra Streamlit project. Preserve the current data-processing and hotspot modules. Install `requirements.txt`, run `streamlit run app.py`, inspect errors, and fix only concrete compatibility/runtime issues. Do not invent accident coordinates, road names, model accuracy, or real-time risk. The supplied dataset contains chainage kilometres but no GPS coordinates, so keep the app's map/demo-risk labels explicit. Never commit `.streamlit/secrets.toml`, passwords, API keys, or OAuth credentials. After local testing, explain the exact deployment steps for a Streamlit-compatible host and the required secrets.

## Before publishing

- Test demo login and each page.
- Test English, Hindi and Marathi switching.
- Test route geometry with internet access; the OSRM public demo endpoint may have usage limits.
- Verify that map demo layers are clearly labelled illustrative.
- Add Firebase API key and Google OAuth credentials through the deployment platform's secrets manager, not in source control.
- Do not present the app as an operational road-safety authority or a verified Maharashtra-wide prediction system until geospatial accident data has been linked and validated.

## What cannot be enabled by the ZIP alone

Real Google sign-in requires Google Cloud OAuth configuration. Email/password login requires Firebase Authentication configuration. Public deployment requires a hosting account and environment-specific secrets. The ZIP contains configuration templates, not credentials or an already-deployed URL.


## Chatbot feature

The app includes a built-in multilingual RoadGuard Assistant page (English, Hindi, Marathi). It answers common questions about route planning, map layers, historical hotspots, severity prediction, hazard reporting, and dataset limitations. This first version is a local rule-based assistant and does not need an API key or send chat messages to a third party. It is intentionally transparent and must not be described as a live emergency service or a verified road-safety oracle. To add generative AI later, connect a provider through deployment secrets and retain the same safety/data caveats.
