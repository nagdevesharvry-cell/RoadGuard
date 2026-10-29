"""RoadGuard Maharashtra — interactive traffic-risk prototype.

Run: streamlit run app.py
Data honesty: bundled accident records have chainage but no latitude/longitude or road-name
mapping. Corridor hotspot analysis is data-derived; statewide map hotspots and route-risk
overlays are explicitly illustrative until a verified geospatial dataset is supplied.
"""
from __future__ import annotations

import io
import os
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
import folium
from streamlit_folium import st_folium

from src.data import clean
from src import hotspots as hs, models
from src.config import DEFAULT_CSV, DEFAULT_SEVERITY_WEIGHTS, SEVERE_CLASSES, label, load_labels

st.set_page_config(page_title="RoadGuard Maharashtra", page_icon="🚦", layout="wide", initial_sidebar_state="expanded")

# ------------------------------- translations --------------------------------
T = {
 "English": {
  "tagline":"Intelligent Traffic Safety & Accident Risk Intelligence",
  "login_title":"Your safer journey starts here","login_sub":"Explore road-risk insights, compare routes, and understand accident patterns.",
  "email":"Email address","password":"Password","login":"Sign in","signup":"Create account","google":"Continue with Google",
  "demo":"Explore interactive demo","auth_note":"Authentication can be enabled with Firebase / Google OAuth. Demo mode is for prototype evaluation only.",
  "dashboard":"Overview","map":"Maharashtra risk map","routes":"Safer route planner","analyzer":"AI risk analyzer","hotspots":"Historical hotspots","reports":"Road hazard reports","chatbot":"RoadGuard chatbot","settings":"Settings",
  "welcome":"Welcome to RoadGuard","riskmap":"Explore Maharashtra roads","mapnote":"Map locations below are reference city locations. The bundled dataset has no GPS coordinates or road-name mapping, so its chainage hotspots cannot honestly be plotted on these roads yet.",
  "demo_overlay":"Show illustrative demo risk zones","demo_warning":"Illustrative demo layer — not measured risk and not a live safety warning.",
  "from":"Starting point","to":"Destination","compare":"Compare routes","route_note":"Route geometry comes from the public OSRM routing service. Risk values shown in demo mode are illustrative only; this dataset is not geospatially linked to the routes.",
  "risk":"Risk level","distance":"Distance","duration":"Estimated time","low":"Lower relative risk","medium":"Moderate relative risk","high":"Higher relative risk",
  "historical":"Historical corridor analysis","data_caveat":"The uploaded data contains chainage kilometre values but no GPS coordinates, named road/corridor, or Maharashtra location identifier. These historical hotspot rankings describe the source corridor only, not all Maharashtra roads.",
  "records":"Accident records","years":"Years covered","corridor":"Max chainage (km)","hotspot_table":"Highest-ranked source-corridor segments",
  "severity":"Severity prediction (what-if)","predict_note":"This model estimates the severity class among recorded accidents using historical accident attributes. It does not predict whether a crash will occur at a given GPS location.",
  "day":"Day of week","location_type":"Location type code","side":"Road-side code","road_feature":"Road feature code","road_condition":"Road condition code","weather":"Weather code","cause":"Cause code","vehicle1":"Vehicle type 1 code","vehicle2":"Vehicle type 2 code","hour":"Hour of day","month":"Month",
  "predict":"Estimate severity probabilities","prediction":"Model-estimated severity probabilities","risk_factors":"Interpretation","setup":"Setup & data limitations","language":"Language","theme":"Appearance","dark":"Dark","light":"Light","logout":"Sign out",
  "report_title":"Report a road hazard","hazard_type":"Hazard type","description":"Description","submit":"Save report locally","report_success":"Report added to this session. Connect a database to persist reports across users.",
  "pothole":"Pothole / damaged road","water":"Waterlogging","lighting":"Poor street lighting","signage":"Missing / damaged signage","visibility":"Low visibility / fog","other":"Other",
  "no_geo":"No verified GPS mapping in bundled dataset","source":"Data source","model_status":"Model status","demo_mode":"Prototype demo mode",
 },
 "Hindi": {
  "tagline":"स्मार्ट ट्रैफिक सुरक्षा और दुर्घटना जोखिम विश्लेषण","login_title":"आपकी सुरक्षित यात्रा यहीं से शुरू होती है","login_sub":"सड़क जोखिम देखें, रास्तों की तुलना करें और दुर्घटनाओं के पैटर्न समझें।",
  "email":"ईमेल पता","password":"पासवर्ड","login":"लॉग इन","signup":"अकाउंट बनाएं","google":"Google से जारी रखें","demo":"इंटरैक्टिव डेमो खोलें","auth_note":"Firebase / Google OAuth से असली लॉगिन सक्षम किया जा सकता है। डेमो मोड केवल प्रोटोटाइप के लिए है।",
  "dashboard":"ओवरव्यू","map":"महाराष्ट्र जोखिम मैप","routes":"सुरक्षित रास्ता प्लानर","analyzer":"AI जोखिम विश्लेषक","hotspots":"ऐतिहासिक हॉटस्पॉट","reports":"सड़क समस्या रिपोर्ट","chatbot":"RoadGuard चैटबॉट","settings":"सेटिंग्स","welcome":"RoadGuard में आपका स्वागत है","riskmap":"महाराष्ट्र की सड़कें देखें",
  "mapnote":"नीचे दिए स्थान शहरों के संदर्भ बिंदु हैं। उपलब्ध डेटा में GPS निर्देशांक या सड़क-नाम मैपिंग नहीं है, इसलिए उसके chainage हॉटस्पॉट को अभी इन सड़कों पर सही तरह से नहीं दिखाया जा सकता।","demo_overlay":"उदाहरण के लिए डेमो जोखिम क्षेत्र दिखाएं","demo_warning":"यह केवल उदाहरण वाला डेमो है—मापा हुआ जोखिम या लाइव चेतावनी नहीं।",
  "from":"शुरुआती स्थान","to":"गंतव्य","compare":"रास्तों की तुलना करें","route_note":"रास्ते का आकार OSRM सेवा से आता है। डेमो जोखिम मान केवल उदाहरण हैं; उपलब्ध डेटा इन रास्तों से भौगोलिक रूप से जुड़ा नहीं है।",
  "risk":"जोखिम स्तर","distance":"दूरी","duration":"अनुमानित समय","low":"तुलनात्मक रूप से कम जोखिम","medium":"मध्यम जोखिम","high":"तुलनात्मक रूप से अधिक जोखिम",
  "historical":"ऐतिहासिक कॉरिडोर विश्लेषण","data_caveat":"डेटा में chainage किलोमीटर हैं, लेकिन GPS, सड़क/कॉरिडोर का नाम या महाराष्ट्र लोकेशन पहचान नहीं है। हॉटस्पॉट रैंकिंग केवल मूल कॉरिडोर के लिए है, पूरे महाराष्ट्र के लिए नहीं।",
  "records":"दुर्घटना रिकॉर्ड","years":"कवर किए गए वर्ष","corridor":"अधिकतम chainage (किमी)","hotspot_table":"मूल कॉरिडोर के सबसे ऊंचे जोखिम वाले सेगमेंट",
  "severity":"गंभीरता का अनुमान (what-if)","predict_note":"यह मॉडल दर्ज दुर्घटनाओं के गुणों के आधार पर उनकी severity class का अनुमान लगाता है। यह GPS स्थान पर दुर्घटना होने की संभावना नहीं बताता।",
  "day":"सप्ताह का दिन","location_type":"लोकेशन कोड","side":"सड़क की तरफ का कोड","road_feature":"सड़क फीचर कोड","road_condition":"सड़क स्थिति कोड","weather":"मौसम कोड","cause":"कारण कोड","vehicle1":"वाहन प्रकार 1 कोड","vehicle2":"वाहन प्रकार 2 कोड","hour":"दिन का घंटा","month":"महीना",
  "predict":"गंभीरता की संभावना निकालें","prediction":"मॉडल द्वारा अनुमानित गंभीरता","risk_factors":"व्याख्या","setup":"सेटअप और डेटा सीमाएं","language":"भाषा","theme":"दिखावट","dark":"डार्क","light":"लाइट","logout":"लॉग आउट",
  "report_title":"सड़क समस्या रिपोर्ट करें","hazard_type":"समस्या का प्रकार","description":"विवरण","submit":"इस सत्र में रिपोर्ट सेव करें","report_success":"रिपोर्ट इस सत्र में जोड़ दी गई। सभी यूज़र के लिए सेव करने हेतु डेटाबेस जोड़ें।",
  "pothole":"गड्ढा / खराब सड़क","water":"जलभराव","lighting":"कम स्ट्रीट लाइट","signage":"गायब / खराब संकेतक","visibility":"कम दृश्यता / कोहरा","other":"अन्य","no_geo":"डेटासेट में सत्यापित GPS मैपिंग नहीं है","source":"डेटा स्रोत","model_status":"मॉडल स्थिति","demo_mode":"प्रोटोटाइप डेमो मोड",
 },
 "Marathi": {
  "tagline":"स्मार्ट वाहतूक सुरक्षा आणि अपघात जोखीम विश्लेषण","login_title":"तुमचा सुरक्षित प्रवास इथून सुरू होतो","login_sub":"रस्त्यांची जोखीम पाहा, मार्गांची तुलना करा आणि अपघातांचे नमुने समजून घ्या.",
  "email":"ईमेल पत्ता","password":"पासवर्ड","login":"लॉग इन","signup":"खाते तयार करा","google":"Google ने पुढे जा","demo":"इंटरॅक्टिव्ह डेमो उघडा","auth_note":"Firebase / Google OAuth वापरून खरे लॉगिन सक्षम करता येईल. डेमो मोड फक्त प्रोटोटाइपसाठी आहे.",
  "dashboard":"आढावा","map":"महाराष्ट्र जोखीम नकाशा","routes":"सुरक्षित मार्ग नियोजक","analyzer":"AI जोखीम विश्लेषक","hotspots":"ऐतिहासिक हॉटस्पॉट","reports":"रस्त्यावरील समस्या नोंदवा","chatbot":"RoadGuard चॅटबॉट","settings":"सेटिंग्ज","welcome":"RoadGuard मध्ये स्वागत आहे","riskmap":"महाराष्ट्रातील रस्ते पाहा",
  "mapnote":"खालील ठिकाणे शहरांचे संदर्भ बिंदू आहेत. दिलेल्या डेटामध्ये GPS निर्देशांक किंवा रस्त्यांच्या नावांचे मॅपिंग नाही, त्यामुळे chainage हॉटस्पॉट सध्या या रस्त्यांवर अचूक दाखवता येत नाहीत.","demo_overlay":"उदाहरणार्थ डेमो जोखीम क्षेत्र दाखवा","demo_warning":"हे फक्त उदाहरणार्थ डेमो आहे—मोजलेली जोखीम किंवा थेट सुरक्षा सूचना नाही.",
  "from":"सुरुवातीचे ठिकाण","to":"गंतव्य","compare":"मार्गांची तुलना करा","route_note":"मार्ग OSRM सेवेतून मिळतो. डेमोमधील जोखीम मूल्ये उदाहरणार्थ आहेत; उपलब्ध डेटा या मार्गांशी भौगोलिकरीत्या जोडलेला नाही.",
  "risk":"जोखीम पातळी","distance":"अंतर","duration":"अंदाजे वेळ","low":"तुलनेने कमी जोखीम","medium":"मध्यम जोखीम","high":"तुलनेने जास्त जोखीम",
  "historical":"ऐतिहासिक कॉरिडॉर विश्लेषण","data_caveat":"डेटामध्ये chainage किलोमीटर आहेत, परंतु GPS, रस्त्याचे नाव किंवा महाराष्ट्रातील स्थान ओळख नाही. हॉटस्पॉट क्रमवारी फक्त मूळ कॉरिडॉरसाठी आहे, संपूर्ण महाराष्ट्रासाठी नाही.",
  "records":"अपघात नोंदी","years":"समाविष्ट वर्षे","corridor":"कमाल chainage (किमी)","hotspot_table":"मूळ कॉरिडॉरमधील सर्वाधिक क्रमांकाचे सेगमेंट",
  "severity":"गंभीरतेचा अंदाज (what-if)","predict_note":"हे मॉडेल नोंदवलेल्या अपघातांच्या वैशिष्ट्यांवरून त्यांची severity class अंदाजते. एखाद्या GPS ठिकाणी अपघात होईल का हे ते सांगत नाही.",
  "day":"आठवड्याचा दिवस","location_type":"स्थान कोड","side":"रस्त्याच्या बाजूचा कोड","road_feature":"रस्ता वैशिष्ट्य कोड","road_condition":"रस्त्याची स्थिती कोड","weather":"हवामान कोड","cause":"कारण कोड","vehicle1":"वाहन प्रकार 1 कोड","vehicle2":"वाहन प्रकार 2 कोड","hour":"दिवसातील तास","month":"महिना",
  "predict":"गंभीरतेची शक्यता मोजा","prediction":"मॉडेलचा गंभीरता अंदाज","risk_factors":"स्पष्टीकरण","setup":"सेटअप आणि डेटाच्या मर्यादा","language":"भाषा","theme":"दिसणे","dark":"डार्क","light":"लाईट","logout":"लॉग आउट",
  "report_title":"रस्त्यावरील समस्या नोंदवा","hazard_type":"समस्येचा प्रकार","description":"वर्णन","submit":"या सत्रात नोंदवा","report_success":"नोंद या सत्रात जोडली. सर्व वापरकर्त्यांसाठी जतन करण्यासाठी डेटाबेस जोडावा लागेल.",
  "pothole":"खड्डा / खराब रस्ता","water":"पाणी साचणे","lighting":"रस्त्यावरील कमी प्रकाश","signage":"गहाळ / खराब चिन्हे","visibility":"कमी दृश्यमानता / धुके","other":"इतर","no_geo":"डेटासेटमध्ये पडताळलेले GPS मॅपिंग नाही","source":"डेटा स्रोत","model_status":"मॉडेल स्थिती","demo_mode":"प्रोटोटाइप डेमो मोड",
 }
}

# City reference coordinates are for navigation/demo context only, not accident labels.
CITIES = {
 "Mumbai": (19.0760, 72.8777), "Pune": (18.5204, 73.8567), "Nagpur": (21.1458, 79.0882),
 "Nashik": (19.9975, 73.7898), "Chhatrapati Sambhajinagar": (19.8762, 75.3433),
 "Kolhapur": (16.7050, 74.2433), "Solapur": (17.6599, 75.9064), "Amravati": (20.9374, 77.7796),
 "Thane": (19.2183, 72.9781), "Ratnagiri": (16.9902, 73.3120), "Jalgaon": (21.0077, 75.5626),
}
CITY_RISK_DEMO = {"Mumbai": 0.64, "Pune": 0.73, "Nagpur": 0.48, "Nashik": 0.57,
                  "Chhatrapati Sambhajinagar": 0.69, "Kolhapur": 0.43, "Solapur": 0.61,
                  "Amravati": 0.46, "Thane": 0.71, "Ratnagiri": 0.38, "Jalgaon": 0.53}

def tr(key: str) -> str:
    return T.get(st.session_state.get("language", "English"), T["English"]).get(key, T["English"].get(key, key))

def get_auth_config():
    try:
        return st.secrets.get("firebase", {})
    except Exception:
        return {}

def firebase_request(endpoint: str, payload: dict):
    cfg = get_auth_config()
    api_key = cfg.get("api_key", "")
    if not api_key:
        return None, "Firebase API key is not configured in .streamlit/secrets.toml."
    try:
        resp = requests.post(f"https://identitytoolkit.googleapis.com/v1/{endpoint}?key={api_key}",
                             json=payload, timeout=15)
        data = resp.json()
        if not resp.ok:
            return None, data.get("error", {}).get("message", "Authentication failed.")
        return data, None
    except Exception as e:
        return None, str(e)

def init_auth():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "demo_user" not in st.session_state:
        st.session_state.demo_user = False
    try:
        if st.user.is_logged_in:
            st.session_state.authenticated = True
            st.session_state.user_email = getattr(st.user, "email", "Google user")
            st.session_state.demo_user = False
    except Exception:
        pass

def auth_screen():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@500;600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    
    .stApp {
        background: radial-gradient(circle at 10% 12%, rgba(14, 165, 233, 0.12), transparent 32%),
                    radial-gradient(circle at 90% 80%, rgba(99, 102, 241, 0.08), transparent 35%),
                    #070e1b !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }
    header[data-testid="stHeader"] {
        background: rgba(7, 14, 27, 0.85) !important;
        backdrop-filter: blur(14px) !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06) !important;
    }
    .block-container {
        max-width: 1140px;
        padding-top: 2rem;
    }
    .auth-hero {
        padding: 2.2rem 2.4rem;
        border: 1px solid rgba(56, 189, 248, 0.22);
        border-radius: 22px;
        background: linear-gradient(135deg, rgba(15, 30, 54, 0.92), rgba(9, 18, 33, 0.94));
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
        margin-bottom: 1.2rem;
    }
    .auth-hero h1 {
        font-family: 'Outfit', sans-serif;
        font-size: 2.6rem;
        font-weight: 800;
        color: #ffffff;
        margin: 0.3rem 0;
        letter-spacing: -0.02em;
    }
    .auth-eyebrow {
        letter-spacing: 0.18em;
        text-transform: uppercase;
        font-size: 0.74rem;
        color: #38bdf8;
        font-weight: 700;
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        background: rgba(14, 165, 233, 0.12);
        border: 1px solid rgba(14, 165, 233, 0.28);
    }
    .auth-features {
        display: flex;
        flex-direction: column;
        gap: 10px;
        margin-top: 1.2rem;
    }
    .auth-feat-item {
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 0.9rem;
        color: #cbd5e1;
    }
    .auth-feat-icon {
        font-size: 1.1rem;
        color: #38bdf8;
    }
    .demo-card-highlight {
        background: linear-gradient(135deg, rgba(14, 165, 233, 0.18), rgba(99, 102, 241, 0.14));
        border: 1px solid rgba(56, 189, 248, 0.45);
        border-radius: 18px;
        padding: 22px;
        box-shadow: 0 8px 30px rgba(14, 165, 233, 0.15);
        margin-bottom: 1rem;
    }
    .login-container-card {
        background: rgba(12, 22, 38, 0.85);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 22px;
        backdrop-filter: blur(14px);
    }
    </style>""", unsafe_allow_html=True)
    
    left, right = st.columns([1.15, 0.85], gap="large")
    with left:
        st.markdown(f"""<div class="auth-hero">
            <span class="auth-eyebrow">ROADGUARD • MAHARASHTRA</span>
            <h1>🚦 RoadGuard</h1>
            <p style="color:#94a3b8; font-size:1.05rem; margin-bottom:1rem;">{tr('tagline')}</p>
            <hr style="border-color:rgba(255,255,255,0.08); margin:1.2rem 0;">
            <h3 style="color:#f8fafc; font-family:'Outfit', sans-serif; font-size:1.35rem; margin-bottom:0.4rem;">{tr('login_title')}</h3>
            <p style="color:#94a3b8; font-size:0.94rem; margin-bottom:1.2rem;">{tr('login_sub')}</p>
            <div class="auth-features">
                <div class="auth-feat-item"><span class="auth-feat-icon">🗺️</span> <b>Interactive Map</b> · Reference locations & illustrative risk overlays</div>
                <div class="auth-feat-item"><span class="auth-feat-icon">🧭</span> <b>Safer Route Planner</b> · Multi-route geometry comparison via OSRM</div>
                <div class="auth-feat-item"><span class="auth-feat-icon">📊</span> <b>Corridor Hotspot Engine</b> · Empirical-Bayes loss scoring along chainage</div>
                <div class="auth-feat-item"><span class="auth-feat-icon">🧠</span> <b>AI Risk Analyzer</b> · Machine learning accident severity prediction</div>
                <div class="auth-feat-item"><span class="auth-feat-icon">💬</span> <b>RoadGuard Chatbot</b> · Multilingual assistant in English, Hindi & Marathi</div>
            </div>
        </div>""", unsafe_allow_html=True)
        st.caption("ℹ️ " + tr("auth_note"))
        
    with right:
        # High-priority Instant Demo Card (Solves user confusion about Firebase API key)
        st.markdown(f"""
        <div class="demo-card-highlight">
            <div style="font-family:'Outfit', sans-serif; font-size:1.15rem; font-weight:700; color:#ffffff; margin-bottom:4px;">
                ⚡ {tr('demo')}
            </div>
            <div style="font-size:0.85rem; color:#cbd5e1; margin-bottom:14px; line-height:1.4;">
                Instant access to all 8 platform pages. No Firebase API key, Google login, or password required.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 " + tr("demo") + " (Instant Access)", type="primary", use_container_width=True, key="hero_demo_button"):
            st.session_state.authenticated = True
            st.session_state.demo_user = True
            st.session_state.user_email = "Demo visitor"
            st.rerun()
            
        st.markdown("<div style='text-align:center; color:#64748b; font-size:0.8rem; margin:14px 0;'>— OR SIGN IN WITH FIREBASE / GOOGLE —</div>", unsafe_allow_html=True)
        
        with st.expander("🔐 Optional Firebase / Google Sign In", expanded=False):
            st.caption("Requires Firebase Web API Key or Google OAuth in `.streamlit/secrets.toml`.")
            auth_tab, signup_tab = st.tabs([tr("login"), tr("signup")])
            with auth_tab:
                with st.form("firebase_login"):
                    email = st.text_input(tr("email"), placeholder="you@example.com")
                    password = st.text_input(tr("password"), type="password")
                    submitted = st.form_submit_button(tr("login"), use_container_width=True)
                if submitted:
                    data, err = firebase_request("accounts:signInWithPassword", {"email": email, "password": password, "returnSecureToken": True})
                    if data:
                        st.session_state.authenticated = True
                        st.session_state.demo_user = False
                        st.session_state.user_email = data.get("email", email)
                        st.session_state.firebase_id_token = data.get("idToken", "")
                        st.rerun()
                    else:
                        st.error(err or "Login unavailable. Use the instant demo button above.")
            with signup_tab:
                with st.form("firebase_signup"):
                    email = st.text_input(tr("email"), key="signup_email", placeholder="you@example.com")
                    password = st.text_input(tr("password"), key="signup_password", type="password", help="Use at least 6 characters.")
                    submitted = st.form_submit_button(tr("signup"), use_container_width=True)
                if submitted:
                    data, err = firebase_request("accounts:signUp", {"email": email, "password": password, "returnSecureToken": True})
                    if data:
                        st.session_state.authenticated = True
                        st.session_state.demo_user = False
                        st.session_state.user_email = data.get("email", email)
                        st.session_state.firebase_id_token = data.get("idToken", "")
                        st.success("Account created.")
                        st.rerun()
                    else:
                        st.error(err or "Signup unavailable. Use the instant demo button above.")
            st.divider()
            try:
                oidc_configured = bool(st.secrets.get("auth", {}).get("client_id"))
            except Exception:
                oidc_configured = False
            if oidc_configured:
                if st.button(tr("google"), use_container_width=True):
                    st.login()
            else:
                st.button(tr("google") + " · setup required", disabled=True, use_container_width=True)
                st.caption("Google OIDC requires client_id in `.streamlit/secrets.toml`.")

@st.cache_data(show_spinner="Loading and cleaning the bundled accident dataset…")
def load_data():
    raw = pd.read_csv(DEFAULT_CSV)
    return clean(raw)

@st.cache_data(show_spinner="Calculating historical corridor hotspots…")
def segment_data(df, seg_km, years, weight_items):
    cov = hs.coverage_matrix(df, seg_km)
    df_sel = df[df["year"].isin(years)].copy()
    seg = hs.segment_table(df_sel, cov, seg_km, dict(weight_items), list(years))
    return seg, cov, df_sel

@st.cache_resource(show_spinner="Training the historical severity model (first run only)…")
def fitted_model(df):
    return models.fit_final(df)

def risk_tier(v):
    if v >= .67: return "High"
    if v >= .48: return "Moderate"
    return "Lower"

def fmt_duration(minutes):
    h, m = divmod(int(minutes), 60)
    return f"{h}h {m}m" if h else f"{m} min"

def get_route(start: str, end: str):
    a, b = CITIES[start], CITIES[end]
    url = f"https://router.project-osrm.org/route/v1/driving/{a[1]},{a[0]};{b[1]},{b[0]}"
    try:
        res = requests.get(url, params={"overview":"full","geometries":"geojson","alternatives":"true","steps":"false"}, timeout=18)
        res.raise_for_status()
        data = res.json()
        if data.get("code") == "Ok" and data.get("routes"):
            return data["routes"]
    except Exception:
        return None
    return None

def page_map():
    st.title("🗺️ " + tr("riskmap"))
    st.caption(tr("mapnote"))
    st.markdown(f"""
    <div class="notice-banner">
        <div class="notice-icon">⚠️</div>
        <div class="notice-text">
            <span class="notice-tag">SPATIAL CAVEAT</span>
            {tr("no_geo")}. {tr("data_caveat")}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    m_col, opt_col = st.columns([3, 1])
    with opt_col:
        st.markdown("#### Map Options")
        map_style = st.selectbox("Base Map Style", ["Dark Matter (Recommended)", "Positron (Light)"], index=0)
        demo = st.toggle(tr("demo_overlay"), value=False)
        if demo:
            st.caption("🔴 Illustrative demo zones — not measured risk and not live alerts.")
            
    tile_name = "CartoDB dark_matter" if "Dark" in map_style else "CartoDB positron"
    m = folium.Map(location=[19.4, 75.4], zoom_start=6, tiles=tile_name, control_scale=True)
    
    for city, (lat, lon) in CITIES.items():
        folium.CircleMarker(
            [lat, lon], radius=6, color="#38bdf8", fill=True, fill_color="#0284c7", fill_opacity=0.9,
            tooltip=city, popup=f"<b>{city}</b><br>Reference city location — not an accident hotspot"
        ).add_to(m)
        if demo:
            v = CITY_RISK_DEMO[city]
            color = "#f43f5e" if v >= .67 else "#f59e0b" if v >= .48 else "#10b981"
            folium.Circle(
                [lat, lon], radius=13000 + v*8000, color=color, fill=True, fill_color=color,
                fill_opacity=0.22, tooltip=f"Illustrative demo score: {v:.0%}"
            ).add_to(m)
            
    with m_col:
        st_folium(m, use_container_width=True, height=560, returned_objects=[])
        
    st.markdown("#### Map Legend")
    a, b, c = st.columns(3)
    a.markdown("""<div style="background:rgba(16,185,129,0.12);border:1px solid rgba(16,185,129,0.3);border-radius:10px;padding:8px 14px;color:#6ee7b7;">🟢 Lower Relative Demo Score (&lt;48%)</div>""", unsafe_allow_html=True)
    b.markdown("""<div style="background:rgba(245,158,11,0.12);border:1px solid rgba(245,158,11,0.3);border-radius:10px;padding:8px 14px;color:#fde68a;">🟠 Moderate Relative Demo Score (48–67%)</div>""", unsafe_allow_html=True)
    c.markdown("""<div style="background:rgba(244,63,94,0.12);border:1px solid rgba(244,63,94,0.3);border-radius:10px;padding:8px 14px;color:#fda4af;">🔴 Higher Relative Demo Score (&gt;67%)</div>""", unsafe_allow_html=True)
    st.caption("For a real statewide risk map, add a verified dataset containing accident latitude/longitude or a road-segment-to-GPS lookup. The current CSV only has a chainage kilometre field.")

def page_routes():
    st.title("🧭 " + tr("routes"))
    st.caption(tr("route_note"))
    st.markdown(f"""
    <div class="notice-banner">
        <div class="notice-icon">⚠️</div>
        <div class="notice-text">
            <span class="notice-tag">DEMO WARNING</span>
            {tr("demo_warning")}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    cities = list(CITIES)
    start = c1.selectbox(tr("from"), cities, index=cities.index("Nagpur"), key="route_start")
    dest_options = [x for x in cities if x != start]
    end = c2.selectbox(tr("to"), dest_options, index=dest_options.index("Pune") if "Pune" in dest_options else 0, key="route_end")
    
    if st.button(tr("compare"), type="primary", use_container_width=True):
        with st.spinner("Requesting real route geometry from OSRM…"):
            routes = get_route(start, end)
        if not routes:
            st.error("Could not retrieve route geometry. Check internet access or try again later.")
            return
        m = folium.Map(location=[(CITIES[start][0]+CITIES[end][0])/2, (CITIES[start][1]+CITIES[end][1])/2], zoom_start=6, tiles="CartoDB dark_matter")
        colors = ["#10b981", "#f43f5e", "#0ea5e9"]
        cards = []
        for i, r in enumerate(routes[:3]):
            coords = r["geometry"]["coordinates"]
            latlon = [[pt[1], pt[0]] for pt in coords]
            folium.PolyLine(latlon, color=colors[i%len(colors)], weight=5, opacity=0.85, tooltip=f"Route {i+1}").add_to(m)
            demo_score = float(np.clip(0.34 + (r["distance"]/1000 % 37)/100 + i*0.07, 0.25, 0.88))
            cards.append({
                "Route": f"Route {i+1}",
                "Distance": f"{r['distance']/1000:.1f} km",
                "Estimated time": fmt_duration(r["duration"]/60),
                "Illustrative demo score": f"{demo_score:.0%}",
                "Risk label": risk_tier(demo_score)
            })
        folium.Marker(CITIES[start], tooltip=start, icon=folium.Icon(color="green")).add_to(m)
        folium.Marker(CITIES[end], tooltip=end, icon=folium.Icon(color="red")).add_to(m)
        st_folium(m, use_container_width=True, height=470, returned_objects=[])
        st.dataframe(pd.DataFrame(cards), hide_index=True, use_container_width=True)
        st.caption("Route scores above are deliberately marked illustrative and are NOT model predictions. Do not use them to choose a real-world route until the accident dataset is geospatially joined and the risk method validated.")
    st.markdown("**Next step for a data-backed route score:** map each route polyline to verified road segments, aggregate historical accident exposure and apply a calibrated segment-risk model.")

def page_overview(df, seg_df, df_sel, years, seg_km):
    st.title("🚦 " + tr("welcome"))
    st.caption(tr("tagline"))
    
    st.markdown(f"""
    <div class="notice-banner">
        <div class="notice-icon">⚠️</div>
        <div class="notice-text">
            <span class="notice-tag">DATA CAVEAT</span>
            {tr("data_caveat")}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    hotspots_count = int(seg_df["tier"].isin(["Critical","High"]).sum())
    st.markdown(f"""
    <div class="metrics-grid">
        <div class="metric-card cyan">
            <div class="metric-label"><span>{tr('records')}</span> 📊</div>
            <div class="metric-val">{len(df_sel):,}</div>
            <div class="metric-sub">Cleaned records in corridor</div>
        </div>
        <div class="metric-card indigo">
            <div class="metric-label"><span>{tr('years')}</span> 📅</div>
            <div class="metric-val">{len(years)}</div>
            <div class="metric-sub">Surveyed periods analyzed</div>
        </div>
        <div class="metric-card emerald">
            <div class="metric-label"><span>{tr('corridor')}</span> 🛣️</div>
            <div class="metric-val">{df['km'].max():,.1f} <span style="font-size:1.1rem;color:#94a3b8">km</span></div>
            <div class="metric-sub">Source corridor chainage</div>
        </div>
        <div class="metric-card rose">
            <div class="metric-label"><span>Hotspot Segments</span> ⚠️</div>
            <div class="metric-val">{hotspots_count}</div>
            <div class="metric-sub">Critical & High risk stretches</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    l, r = st.columns([1.2, 1], gap="medium")
    with l:
        st.markdown("""<div class="glass-card-header"><span class="glass-card-title">📈 Accidents by Year</span><span class="glass-card-badge">Corridor Trend</span></div>""", unsafe_allow_html=True)
        yearly = df_sel.groupby("year").size().reset_index(name="Accidents")
        fig_y = px.bar(yearly, x="year", y="Accidents", color="Accidents",
                       color_continuous_scale=["#0284c7", "#f43f5e"])
        fig_y.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=15, b=10),
            font=dict(color="#cbd5e1", family="'Plus Jakarta Sans', sans-serif"),
            coloraxis_showscale=False,
            height=320
        )
        fig_y.update_xaxes(showgrid=False, tickfont=dict(color="#94a3b8"))
        fig_y.update_yaxes(gridcolor="rgba(255,255,255,0.06)", tickfont=dict(color="#94a3b8"))
        st.plotly_chart(fig_y, use_container_width=True)
    with r:
        st.markdown("""<div class="glass-card-header"><span class="glass-card-title">🎯 Severity Distribution</span><span class="glass-card-badge">Class Breakdown</span></div>""", unsafe_allow_html=True)
        sev = df_sel["severity"].value_counts().sort_index().rename_axis("Severity code").reset_index(name="Records")
        fig_p = px.pie(sev, names="Severity code", values="Records", hole=0.62,
                       color_discrete_sequence=["#f43f5e", "#f59e0b", "#06b6d4", "#10b981"])
        fig_p.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=15, b=10),
            font=dict(color="#cbd5e1", family="'Plus Jakarta Sans', sans-serif"),
            legend=dict(orientation="h", yanchor="bottom", y=-0.22, xanchor="center", x=0.5),
            height=320
        )
        st.plotly_chart(fig_p, use_container_width=True)
        
    st.markdown(f"""<div class="glass-card-header" style="margin-top:1.2rem;"><span class="glass-card-title">🚨 {tr('hotspot_table')}</span><span class="glass-card-badge">Top 10 Loss Rank</span></div>""", unsafe_allow_html=True)
    obs = seg_df[seg_df["observed"]].sort_values("loss", ascending=False).head(10)
    if len(obs):
        display = obs[["label","tier","n","years_obs","eb_rate","loss"]].rename(columns={"label":"Segment","tier":"Relative tier","n":"Accident records","years_obs":"Years observed","eb_rate":"Estimated rate/year","loss":"Severity-weighted score"})
        st.dataframe(display, hide_index=True, use_container_width=True)
    st.caption("Risk tiers are relative rankings within the source corridor and selected years, not a calibrated probability that a future crash will occur.")

def page_hotspots(df, seg_df, df_sel, seg_km):
    st.title("🔥 " + tr("historical"))
    st.markdown(f"""
    <div class="notice-banner">
        <div class="notice-icon">⚠️</div>
        <div class="notice-text">
            <span class="notice-tag">CORRIDOR DATA ONLY</span>
            {tr("data_caveat")}
        </div>
    </div>
    """, unsafe_allow_html=True)
    obs = seg_df[seg_df["observed"]].sort_values("loss", ascending=False)
    if obs.empty:
        st.info("No segments with exposure in the selected year range.")
        return
    fig = px.bar(obs.head(25).sort_values("loss"), x="loss", y="label", orientation="h", color="tier",
                 title="Top 25 source-corridor segments", labels={"loss": "Relative severity-weighted score", "label": "Chainage segment"},
                 template="plotly_dark", color_discrete_map={"Critical": "#f43f5e", "High": "#f59e0b", "Medium": "#eab308", "Low": "#10b981"})
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=40, b=10),
        font=dict(color="#cbd5e1", family="'Plus Jakarta Sans', sans-serif")
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)", tickfont=dict(color="#94a3b8"))
    fig.update_yaxes(showgrid=False, tickfont=dict(color="#cbd5e1"))
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(obs[["label","tier","n","years_obs","raw_rate","eb_rate","mean_weight","loss","p_high"]].rename(
        columns={"label":"Segment","tier":"Tier","n":"Accidents","years_obs":"Years observed","raw_rate":"Raw rate/year","eb_rate":"Smoothed rate/year","mean_weight":"Severity weight","loss":"Relative score","p_high":"P(rate > 2x corridor avg)"}),
        hide_index=True, use_container_width=True)
    st.download_button("📥 Download segment analysis CSV", seg_df.to_csv(index=False).encode("utf-8"), "roadguard_corridor_hotspots.csv", "text/csv")

def page_analyzer(df):
    st.title("🧠 " + tr("severity"))
    st.info(tr("predict_note"))
    lab = load_labels()
    row = {}
    a,b,c = st.columns(3)
    row["dow"] = a.selectbox(tr("day"), [1,2,3,4,5,6,7], index=4)
    row["location_type"] = b.selectbox(tr("location_type"), sorted(df.location_type.unique().tolist()))
    row["side"] = c.selectbox(tr("side"), sorted(df.side.unique().tolist()))
    a,b,c = st.columns(3)
    row["road_feature"] = a.selectbox(tr("road_feature"), sorted(df.road_feature.unique().tolist()))
    row["road_condition"] = b.selectbox(tr("road_condition"), sorted(df.road_condition.unique().tolist()))
    row["weather"] = c.selectbox(tr("weather"), sorted(df.weather.unique().tolist()))
    a,b,c = st.columns(3)
    row["cause"] = a.selectbox(tr("cause"), sorted(df.cause.unique().tolist()))
    row["veh1"] = b.selectbox(tr("vehicle1"), sorted(df.veh1.unique().tolist()))
    row["veh2"] = c.selectbox(tr("vehicle2"), sorted(df.veh2.unique().tolist()))
    a,b = st.columns(2)
    row["hour"] = a.slider(tr("hour"), 0, 23, 17)
    row["month"] = b.select_slider(tr("month"), options=list(range(1,13)), value=7)
    if st.button(tr("predict"), type="primary"):
        with st.spinner("Training/loading model and estimating class probabilities…"):
            model = fitted_model(df)
            probs = models.predict_proba(model, row).sort_values(ascending=False)
        names = {1: "Class 1 (label unverified)", 2: "Class 2 (label unverified)", 3: "Class 3 (label unverified)", 4: "Class 4 (label unverified)"}
        out = pd.DataFrame({"Severity class": [names.get(int(k), f"Class {k}") for k in probs.index], "Model probability": probs.values})
        st.subheader(tr("prediction"))
        fig = px.bar(out, x="Severity class", y="Model probability", range_y=[0,1],
                     template="plotly_dark", color="Model probability",
                     color_continuous_scale=["#10b981", "#f59e0b", "#f43f5e"])
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=25, b=10),
            font=dict(color="#cbd5e1", family="'Plus Jakarta Sans', sans-serif"),
            coloraxis_showscale=False
        )
        fig.update_xaxes(showgrid=False, tickfont=dict(color="#cbd5e1"))
        fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)", tickformat=".0%", tickfont=dict(color="#94a3b8"))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(out.assign(**{"Model probability": out["Model probability"].map(lambda x: f"{x:.1%}")}), hide_index=True)
        st.caption("Severity code meanings in labels.csv are marked as assumptions. Verify the official data dictionary before presenting class 1 as fatal or interpreting model outputs operationally.")

def page_reports():
    st.title("📣 " + tr("report_title"))
    st.caption("Prototype-only report form. Reports stay in the current session and are not sent to authorities.")
    with st.form("hazard_report"):
        kind = st.selectbox(tr("hazard_type"), [tr("pothole"), tr("water"), tr("lighting"), tr("signage"), tr("visibility"), tr("other")])
        city = st.selectbox("Nearest city", list(CITIES))
        description = st.text_area(tr("description"), placeholder="Describe the issue and nearest landmark…")
        consent = st.checkbox("I understand this report is stored only in this demo session.")
        submit = st.form_submit_button(tr("submit"), type="primary")
    if submit:
        if not consent: 
            st.error("Please confirm the demo storage limitation.")
        else:
            st.session_state.setdefault("hazard_reports", [])
            st.session_state.hazard_reports.append({"type": kind, "city": city, "description": description})
            st.success(tr("report_success"))
    reports = st.session_state.get("hazard_reports", [])
    if reports:
        st.subheader("Reports in this session")
        st.dataframe(pd.DataFrame(reports), hide_index=True, use_container_width=True)




def chatbot_reply(message: str, language: str, df: pd.DataFrame, years: list[int]) -> str:
    """Lightweight, data-aware assistant for the prototype; no external AI key required."""
    q = message.casefold()
    if language == "Hindi":
        if any(k in q for k in ["route", "रास्त", "मार्ग", "destination", "दूरी"]):
            return ("Safer Route Planner में starting point और destination चुनें। अभी route geometry OSRM से आती है, "
                    "लेकिन dataset में GPS-linked accident records नहीं हैं; इसलिए route risk को वास्तविक/सत्यापित जोखिम न मानें। "
                    "यात्रा से पहले आधिकारिक ट्रैफिक और मौसम अपडेट भी देखें।")
        if any(k in q for k in ["map", "मैप", "नक्श", "red", "लाल", "hotspot", "हॉटस्पॉट"]):
            return ("Maharashtra Risk Map शहरों के reference points दिखाता है। लाल/पीले demo zones illustrative हैं, "
                    "इस dataset से निकले वास्तविक road-level hotspots नहीं। Dataset में GPS coordinates और named-road mapping नहीं है।")
        if any(k in q for k in ["severity", "गंभीर", "prediction", "predict", "मॉडल"]):
            return ("AI Risk Analyzer दर्ज दुर्घटनाओं के attributes से severity class की probability का what-if अनुमान देता है। "
                    "यह किसी सड़क पर दुर्घटना होने की probability नहीं बताता।")
        if any(k in q for k in ["data", "dataset", "डेटा", "record", "रिकॉर्ड"]):
            return (f"वर्तमान चयनित वर्षों: {', '.join(map(str, years)) or 'कोई नहीं'}। साफ किए गए dataset में {len(df):,} records हैं। "
                    "इसमें chainage kilometre है, लेकिन verified GPS coordinates/road names नहीं हैं; इसलिए पूरे महाराष्ट्र के लिए road-level prediction उपलब्ध नहीं है।")
        if any(k in q for k in ["report", "pothole", "गड्ढ", "जलभराव", "hazard", "समस्या"]):
            return ("Road Hazard Reports पेज पर pothole, waterlogging, खराब lighting या signage दर्ज कर सकते हैं। "
                    "वर्तमान prototype में reports केवल इसी session में रहती हैं और emergency services को नहीं भेजी जातीं।")
        return ("मैं route planning, map legend, historical hotspots, severity prediction और data limitations में मदद कर सकता/सकती हूँ। "
                "पूछें: 'क्या route safe है?', 'severity model क्या करता है?' या 'dataset में क्या है?'")
    if language == "Marathi":
        if any(k in q for k in ["route", "मार्ग", "रस्ता", "destination", "अंतर"]):
            return ("Safer Route Planner मध्ये सुरुवातीचे ठिकाण आणि गंतव्य निवडा. मार्ग OSRM कडून मिळतो; परंतु डेटासेटमध्ये GPS-संलग्न अपघात नोंदी नाहीत. "
                    "म्हणून route risk हा सत्यापित वास्तविक धोका समजू नका. अधिकृत वाहतूक व हवामान सूचना तपासा.")
        if any(k in q for k in ["map", "नकाश", "hotspot", "हॉटस्पॉट", "लाल"]):
            return ("Maharashtra Risk Map मधील शहरांचे बिंदू संदर्भासाठी आहेत. रंगीत demo zones उदाहरणार्थ आहेत; "
                    "डेटासेटमध्ये GPS coordinates किंवा रस्त्यांची नावे नसल्याने ते वास्तविक road-level hotspots नाहीत.")
        if any(k in q for k in ["severity", "गंभीर", "prediction", "मॉडेल"]):
            return ("AI Risk Analyzer नोंदवलेल्या अपघातांच्या वैशिष्ट्यांवरून severity class ची शक्यता मोजतो. "
                    "एखाद्या रस्त्यावर अपघात होण्याची शक्यता तो सांगत नाही.")
        if any(k in q for k in ["data", "डेटा", "record", "नोंद"]):
            return (f"निवडलेली वर्षे: {', '.join(map(str, years)) or 'निवडलेली नाहीत'}. स्वच्छ डेटासेटमध्ये {len(df):,} नोंदी आहेत. "
                    "डेटामध्ये chainage आहे, पण पडताळलेले GPS coordinates/रस्त्यांची नावे नाहीत.")
        if any(k in q for k in ["report", "खड्ड", "पाणी", "hazard", "समस्या"]):
            return ("Road Hazard Reports पेजवर खड्डे, पाणी साचणे, कमी प्रकाश किंवा खराब चिन्हे नोंदवा. "
                    "सध्या नोंदी फक्त चालू session मध्ये राहतात; आपत्कालीन सेवांना पाठवल्या जात नाहीत.")
        return ("मी route planning, map legend, historical hotspots, severity prediction आणि डेटाच्या मर्यादा याबद्दल मदत करू शकतो/शकते. "
                "'मार्ग सुरक्षित आहे का?' किंवा 'severity model काय करतो?' असे विचारा.")
    # English default
    if any(k in q for k in ["route", "destination", "directions", "distance", "safe road", "safer"]):
        return ("Open **Safer route planner**, choose a start and destination, then compare route options. Route geometry is retrieved from OSRM. "
                "The bundled accident data is not GPS-linked to those routes, so any demo route-risk values are illustrative—not verified safety estimates. "
                "Check official traffic and weather advisories before travelling.")
    if any(k in q for k in ["map", "hotspot", "red zone", "risk zone", "green", "red road"]):
        return ("Open **Maharashtra risk map** to explore reference city locations. The coloured demo zones are illustrative only. "
                "The bundled dataset has chainage values but no verified GPS coordinates or named-road mapping, so it cannot currently support real statewide road-level hotspot overlays.")
    if any(k in q for k in ["severity", "predict", "prediction", "model", "random forest"]):
        return ("The **AI risk analyzer** estimates severity-class probabilities for a hypothetical recorded-accident profile. "
                "It does not estimate the probability that a crash will happen at a particular GPS location.")
    if any(k in q for k in ["data", "dataset", "records", "years", "source"]):
        return (f"The current year selection is {', '.join(map(str, years)) or 'none'}. The cleaned dataset has {len(df):,} records. "
                "It includes chainage kilometres but lacks verified GPS coordinates and named-road mapping, so it should not be treated as a complete Maharashtra road-risk dataset.")
    if any(k in q for k in ["report", "pothole", "waterlogging", "hazard", "streetlight"]):
        return ("Use **Road hazard reports** to record potholes, waterlogging, poor lighting or damaged signage. Reports are session-only in this prototype and are not sent to emergency services.")
    return ("I can help with route planning, map colours, historical hotspots, severity predictions and data limitations. Try asking: "
            "**“Is this route safe?”**, **“What does the severity model predict?”**, or **“What data does this app use?”**")


def page_chatbot(df: pd.DataFrame, years: list[int], language: str) -> None:
    st.title("💬 RoadGuard Assistant")
    subtitles = {
        "English": "Ask how to use the app, understand risk indicators, or learn what the current dataset can and cannot tell you.",
        "Hindi": "ऐप इस्तेमाल करने, जोखिम संकेत समझने या उपलब्ध डेटा की सीमाओं के बारे में पूछें।",
        "Marathi": "अॅप वापरणे, जोखीम संकेत समजून घेणे किंवा उपलब्ध डेटाच्या मर्यादांबद्दल विचारा."
    }
    st.caption(subtitles.get(language, subtitles["English"]))
    st.info({
        "English": "Prototype assistant: answers are based on app documentation and bundled data. It does not provide emergency dispatch or verified live safety advice.",
        "Hindi": "प्रोटोटाइप सहायक: जवाब ऐप और उपलब्ध डेटा पर आधारित हैं। यह आपातकालीन सेवा या सत्यापित लाइव सुरक्षा सलाह नहीं देता।",
        "Marathi": "प्रोटोटाइप सहाय्यक: उत्तरे अॅप आणि उपलब्ध डेटावर आधारित आहेत. ही आपत्कालीन सेवा किंवा पडताळलेला थेट सुरक्षा सल्ला नाही."
    }.get(language, "Prototype assistant."))
    
    # Quick question suggestions
    prompts_by_lang = {
        "English": [
            "Is this route safe?",
            "What does the severity model predict?",
            "What data does this app use?",
            "How do I report a pothole?"
        ],
        "Hindi": [
            "क्या यह रास्ता सुरक्षित है?",
            "गंभीरता मॉडल क्या करता है?",
            "डेटासेट में क्या है?",
            "सड़क पर गड्ढे की रिपोर्ट कैसे करें?"
        ],
        "Marathi": [
            "हा मार्ग सुरक्षित आहे का?",
            "गंभीरता मॉडेल काय सांगते?",
            "डेटासेटमध्ये काय आहे?",
            "रस्त्यावरील खड्ड्यांची तक्रार कशी करावी?"
        ]
    }
    st.markdown("<p style='font-size:0.85rem;color:#94a3b8;margin-bottom:6px;'>⚡ Quick Questions / त्वरित प्रश्न:</p>", unsafe_allow_html=True)
    q_list = prompts_by_lang.get(language, prompts_by_lang["English"])
    q_cols = st.columns(len(q_list))
    for i, q_text in enumerate(q_list):
        if q_cols[i].button(q_text, key=f"quick_q_{i}", use_container_width=True):
            st.session_state.setdefault("chat_messages", [])
            st.session_state.chat_messages.append({"role": "user", "content": q_text})
            ans = chatbot_reply(q_text, language, df, years)
            st.session_state.chat_messages.append({"role": "assistant", "content": ans})
            st.rerun()

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if st.button("🗑️ Clear chat / चैट साफ करें / चॅट पुसा", key="clear_chat"):
        st.session_state.chat_messages = []
        st.rerun()
        
    for item in st.session_state.chat_messages:
        with st.chat_message(item["role"]):
            st.markdown(item["content"])
            
    prompt = st.chat_input({
        "English": "Ask about routes, hotspots, severity or the data…",
        "Hindi": "रास्ते, हॉटस्पॉट, गंभीरता या डेटा के बारे में पूछें…",
        "Marathi": "मार्ग, हॉटस्पॉट, गंभीरता किंवा डेटाबद्दल विचारा…"
    }.get(language, "Ask RoadGuard…"))
    if prompt:
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        answer = chatbot_reply(prompt, language, df, years)
        st.session_state.chat_messages.append({"role": "assistant", "content": answer})
        st.rerun()


def main():
    init_auth()
    if not st.session_state.authenticated:
        lang = st.selectbox("Language / भाषा / भाषा", ["English", "Hindi", "Marathi"], index=0, key="prelogin_language")
        st.session_state.language = lang
        auth_screen()
        return
    if "language" not in st.session_state: 
        st.session_state.language = "English"
        
    # High-End Modern Design System CSS
    st.markdown("""<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    /* Global Reset & Typography */
    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background: radial-gradient(circle at 12% 14%, rgba(14, 165, 233, 0.08), transparent 28%),
                    radial-gradient(circle at 88% 75%, rgba(99, 102, 241, 0.06), transparent 32%),
                    #070e1b !important;
        color: #f1f5f9 !important;
    }
    
    /* Fix Stark White Streamlit Header */
    header[data-testid="stHeader"] {
        background: rgba(7, 14, 27, 0.85) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08) !important;
    }
    header[data-testid="stHeader"] * {
        color: #cbd5e1 !important;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: #091222 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: rgba(255, 255, 255, 0.08) !important;
    }
    
    /* Sidebar Navigation Pills */
    div[data-testid="stRadio"] > div {
        gap: 6px !important;
    }
    div[data-testid="stRadio"] label {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 10px !important;
        padding: 9px 13px !important;
        margin-bottom: 2px !important;
        transition: all 0.2s ease !important;
        cursor: pointer !important;
    }
    div[data-testid="stRadio"] label:hover {
        background: rgba(14, 165, 233, 0.12) !important;
        border-color: rgba(14, 165, 233, 0.35) !important;
        transform: translateX(3px) !important;
    }
    div[data-testid="stRadio"] label[data-checked="true"] {
        background: linear-gradient(135deg, rgba(14, 165, 233, 0.22), rgba(99, 102, 241, 0.18)) !important;
        border-color: #0ea5e9 !important;
        box-shadow: 0 0 12px rgba(14, 165, 233, 0.2) !important;
    }
    div[data-testid="stRadio"] label p {
        color: #e2e8f0 !important;
        font-weight: 500 !important;
        font-size: 0.92rem !important;
    }
    
    /* Layout Container */
    .block-container {
        padding-top: 1.25rem !important;
        max-width: 1480px !important;
    }
    
    /* Metric Cards Grid */
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin: 1rem 0 1.5rem 0;
    }
    @media (max-width: 900px) {
        .metrics-grid { grid-template-columns: repeat(2, 1fr); }
    }
    .metric-card {
        background: linear-gradient(135deg, rgba(14, 25, 45, 0.85), rgba(10, 18, 32, 0.92));
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 18px 20px;
        backdrop-filter: blur(12px);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        position: relative;
        overflow: hidden;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(14, 165, 233, 0.4);
    }
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
    }
    .metric-card.cyan::before { background: linear-gradient(90deg, #06b6d4, #0ea5e9); }
    .metric-card.indigo::before { background: linear-gradient(90deg, #818cf8, #6366f1); }
    .metric-card.emerald::before { background: linear-gradient(90deg, #10b981, #059669); }
    .metric-card.rose::before { background: linear-gradient(90deg, #f43f5e, #e11d48); }
    
    .metric-label {
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94a3b8;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .metric-val {
        font-size: 2.1rem;
        font-weight: 800;
        font-family: 'Outfit', sans-serif;
        color: #ffffff;
        line-height: 1.1;
        margin-bottom: 4px;
    }
    .metric-sub {
        font-size: 0.78rem;
        color: #64748b;
    }
    
    /* Glassmorphism Section Headers */
    .glass-card-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 12px;
        padding-bottom: 8px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    }
    .glass-card-title {
        font-size: 1.12rem;
        font-weight: 700;
        color: #f8fafc;
        font-family: 'Outfit', sans-serif;
    }
    .glass-card-badge {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        padding: 3px 9px;
        border-radius: 20px;
        background: rgba(14, 165, 233, 0.15);
        color: #38bdf8;
        border: 1px solid rgba(14, 165, 233, 0.3);
    }
    
    /* Custom Notice / Disclaimer Banner */
    .notice-banner {
        display: flex;
        align-items: flex-start;
        gap: 12px;
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.08), rgba(217, 119, 6, 0.04));
        border: 1px solid rgba(245, 158, 11, 0.28);
        border-radius: 14px;
        padding: 14px 18px;
        margin: 12px 0 20px 0;
        backdrop-filter: blur(10px);
    }
    .notice-icon {
        font-size: 1.25rem;
        line-height: 1;
        margin-top: 2px;
    }
    .notice-text {
        font-size: 0.88rem;
        line-height: 1.5;
        color: #fde68a;
    }
    .notice-tag {
        font-weight: 700;
        text-transform: uppercase;
        font-size: 0.7rem;
        letter-spacing: 0.08em;
        padding: 2px 7px;
        border-radius: 6px;
        background: rgba(245, 158, 11, 0.2);
        color: #fbbf24;
        margin-right: 6px;
    }
    
    /* Streamlit Alert Overrides */
    div[data-testid="stAlert"] {
        background: rgba(15, 23, 42, 0.85) !important;
        border: 1px solid rgba(245, 158, 11, 0.35) !important;
        border-radius: 14px !important;
        color: #fde68a !important;
        backdrop-filter: blur(10px) !important;
    }
    
    /* Plotly Chart Container Fix */
    div.stPlotlyChart {
        background: rgba(14, 25, 45, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 16px !important;
        padding: 10px !important;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.25) !important;
    }
    
    /* Dataframes & Tables */
    div[data-testid="stDataFrame"] {
        background: rgba(11, 20, 36, 0.75) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 14px !important;
        overflow: hidden !important;
    }
    
    /* Buttons */
    .stButton button[kind="primary"] {
        background: linear-gradient(135deg, #0ea5e9, #0284c7) !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        padding: 9px 18px !important;
        box-shadow: 0 4px 14px rgba(14, 165, 233, 0.35) !important;
        transition: all 0.2s ease !important;
    }
    .stButton button[kind="primary"]:hover {
        box-shadow: 0 6px 20px rgba(14, 165, 233, 0.55) !important;
        transform: translateY(-1px) !important;
    }
    .stButton button[kind="secondary"] {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        color: #e2e8f0 !important;
        border-radius: 12px !important;
        transition: all 0.2s ease !important;
    }
    .stButton button[kind="secondary"]:hover {
        background: rgba(255, 255, 255, 0.1) !important;
        border-color: rgba(255, 255, 255, 0.25) !important;
    }
    </style>""", unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("## 🚦 RoadGuard")
        st.caption(tr("tagline"))
        status_badge = "🟢 Demo Mode" if st.session_state.demo_user else "🔵 Signed In"
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);border-radius:10px;padding:8px 12px;margin-bottom:12px;font-size:0.82rem;">
            <div style="font-weight:700;color:#38bdf8;margin-bottom:2px;">{status_badge}</div>
            <div style="color:#94a3b8;">{st.session_state.get('user_email','')}</div>
        </div>
        """, unsafe_allow_html=True)
        
        page = st.radio("Navigation", [tr("dashboard"), tr("map"), tr("routes"), tr("analyzer"), tr("hotspots"), tr("reports"), tr("chatbot"), tr("settings")], label_visibility="collapsed")
        st.divider()
        lang = st.selectbox(tr("language"), ["English", "Hindi", "Marathi"], index=["English", "Hindi", "Marathi"].index(st.session_state.language))
        st.session_state.language = lang
        if st.button(tr("logout"), use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.demo_user = False
            st.session_state.pop("firebase_id_token", None)
            try:
                if st.user.is_logged_in: 
                    st.logout()
            except Exception: 
                pass
            st.rerun()
            
    try:
        df, report = load_data()
    except Exception as e:
        st.error(f"Could not load bundled accident dataset: {e}")
        return
        
    years_all = sorted(df.year.unique().tolist())
    with st.sidebar:
        years = st.multiselect("Analysis years", years_all, default=years_all)
        seg_km = st.select_slider("Segment length (km)", options=[2, 5, 10], value=5)
        
    if not years: 
        st.warning("Select at least one year.")
        return
        
    seg_df, cov, df_sel = segment_data(df, seg_km, years, tuple(DEFAULT_SEVERITY_WEIGHTS.items()))
    
    if page == tr("dashboard"): 
        page_overview(df, seg_df, df_sel, years, seg_km)
    elif page == tr("map"): 
        page_map()
    elif page == tr("routes"): 
        page_routes()
    elif page == tr("analyzer"): 
        page_analyzer(df)
    elif page == tr("hotspots"): 
        page_hotspots(df, seg_df, df_sel, seg_km)
    elif page == tr("reports"): 
        page_reports()
    elif page == tr("chatbot"): 
        page_chatbot(df, years, st.session_state.language)
    else:
        st.title("⚙️ " + tr("settings"))
        st.selectbox(tr("language"), ["English", "Hindi", "Marathi"], key="settings_language", index=["English", "Hindi", "Marathi"].index(st.session_state.language))
        if st.session_state.settings_language != st.session_state.language:
            st.session_state.language = st.session_state.settings_language
            st.rerun()
        st.markdown("### " + tr("setup"))
        st.markdown("- Bundled dataset has 8,116 raw rows and chainage-based location values, but no latitude/longitude or named-road lookup.\n- Severity label meanings are assumptions in `data/labels.csv`; verify official codebook.\n- Route risk scores and optional city circles are illustrative demo data, not model-derived.\n- Firebase email/password requires `firebase.api_key`; Google sign-in requires Streamlit OIDC settings.\n- Hazard reports are session-only and do not contact emergency services.")
        st.markdown("### Model/data validation")
        st.write({"Cleaned rows": len(df), "Years": years_all, "Chainage range km": [float(df.km.min()), float(df.km.max())], "Raw columns": list(pd.read_csv(DEFAULT_CSV, nrows=0).columns)})
        st.download_button("📥 Download data cleaning report", pd.DataFrame([report]).to_csv(index=False).encode(), "data_cleaning_report.csv", "text/csv")

if __name__ == "__main__":
    main()
