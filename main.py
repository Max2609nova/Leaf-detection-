"""
Leaf Disease Detection - Streamlit Frontend
============================================

Interactive web UI for the Leaf Disease Detection API. Upload (or try a
sample) leaf image, get an AI diagnosis with severity, symptoms, causes,
and treatment advice, and keep a running history of everything you've
scanned this session.
"""

import json
import os
from datetime import datetime
from pathlib import Path

import requests
import streamlit as st

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

DEFAULT_API_URL = "https://leaf-diseases-detect.vercel.app"
REQUEST_TIMEOUT_SECONDS = 60
MAX_CLIENT_UPLOAD_MB = 10
SAMPLE_IMAGE_PATH = Path(__file__).parent / "Media" / "brown-spot-4 (1).jpg"

SEVERITY_COLORS = {
    "mild": "#66bb6a",
    "moderate": "#ffa726",
    "severe": "#e53935",
    "none": "#4caf50",
}

st.set_page_config(
    page_title="Leaf Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "history" not in st.session_state:
    st.session_state.history = []  # list of {timestamp, filename, result}
if "pending_image" not in st.session_state:
    st.session_state.pending_image = None  # (bytes, filename, mime)


def get_api_url() -> str:
    """
    Resolve the backend API URL, preferring (in order): a value the user
    typed into the sidebar this session, Streamlit secrets, the
    LEAF_API_URL environment variable, then a hardcoded default.
    Using http:// here (instead of https://) is a common source of silent
    failures once deployed, since most hosts (including Vercel) reject or
    redirect plain HTTP -- so the default is https.
    """
    if st.session_state.get("api_url_override"):
        return st.session_state.api_url_override.rstrip("/")
    secret_url = st.secrets.get("API_URL") if hasattr(st, "secrets") else None
    return (secret_url or os.environ.get("LEAF_API_URL") or DEFAULT_API_URL).rstrip("/")


# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #e3f2fd 0%, #f7f9fa 100%);
    }
    .result-card {
        background: rgba(255,255,255,0.95);
        border-radius: 18px;
        box-shadow: 0 4px 24px rgba(44,62,80,0.10);
        padding: 2.2em 2em;
        margin-top: 1em;
        margin-bottom: 1.5em;
        transition: box-shadow 0.3s;
    }
    .result-card:hover {
        box-shadow: 0 8px 32px rgba(44,62,80,0.18);
    }
    .disease-title {
        color: #1b5e20;
        font-size: 2em;
        font-weight: 700;
        margin-bottom: 0.4em;
        letter-spacing: 0.5px;
    }
    .section-title {
        color: #1976d2;
        font-size: 1.15em;
        margin-top: 1.1em;
        margin-bottom: 0.4em;
        font-weight: 600;
    }
    .timestamp {
        color: #757575;
        font-size: 0.9em;
        margin-top: 1em;
        text-align: right;
    }
    .info-badge {
        display: inline-block;
        background: #e3f2fd;
        color: #1976d2;
        border-radius: 8px;
        padding: 0.3em 0.9em;
        font-size: 0.95em;
        font-weight: 600;
        margin-right: 0.5em;
        margin-bottom: 0.4em;
    }
    .severity-badge {
        display: inline-block;
        color: white;
        border-radius: 8px;
        padding: 0.3em 0.9em;
        font-size: 0.95em;
        font-weight: 600;
        margin-right: 0.5em;
        margin-bottom: 0.4em;
    }
    .symptom-list, .cause-list, .treatment-list {
        margin-left: 1em;
        margin-bottom: 0.5em;
    }
    .history-item {
        background: rgba(255,255,255,0.7);
        border-radius: 10px;
        padding: 0.6em 0.9em;
        margin-bottom: 0.5em;
        font-size: 0.9em;
    }
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div style='text-align: center; margin-top: 0.5em;'>
        <span style='font-size:2.5em;'>🌿</span>
        <h1 style='color: #1565c0; margin-bottom:0;'>Leaf Disease Detection</h1>
        <p style='color: #616161; font-size:1.1em;'>
            Upload a leaf image to detect diseases and get AI-powered treatment recommendations.
        </p>
    </div>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Sidebar: settings, about, history
# --------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Settings")
    st.text_input(
        "API URL",
        value=get_api_url(),
        key="api_url_override",
        help="Point this at your own deployment if you're not using the default.",
    )

    st.divider()
    st.header("ℹ️ About")
    st.markdown(
        "This app sends your leaf photo to a Llama Vision model (via Groq) "
        "which identifies plant diseases, rates severity, and suggests "
        "treatments. Works best with a single, well-lit, in-focus leaf."
    )

    st.divider()
    st.header(f"📜 History ({len(st.session_state.history)})")
    if not st.session_state.history:
        st.caption("Your scanned leaves will show up here.")
    else:
        if st.button("Clear history", use_container_width=True):
            st.session_state.history = []
            st.rerun()
        for entry in reversed(st.session_state.history[-10:]):
            r = entry["result"]
            label = r.get("disease_name") or (
                "Healthy" if not r.get("disease_detected") else r.get("disease_type", "Unknown")
            )
            st.markdown(
                f"<div class='history-item'>🕒 {entry['timestamp']}<br>"
                f"<b>{entry['filename']}</b> — {label}</div>",
                unsafe_allow_html=True,
            )

# --------------------------------------------------------------------------
# Main layout
# --------------------------------------------------------------------------

col1, col2 = st.columns([1, 2])

with col1:
    uploaded_file = st.file_uploader(
        "Upload Leaf Image", type=["jpg", "jpeg", "png", "webp"])

    st.caption("or")
    if st.button("🌱 Try a sample image", use_container_width=True):
        if SAMPLE_IMAGE_PATH.exists():
            st.session_state.pending_image = (
                SAMPLE_IMAGE_PATH.read_bytes(),
                SAMPLE_IMAGE_PATH.name,
                "image/jpeg",
            )
        else:
            st.warning("Sample image not found in this deployment.")

    if uploaded_file is not None:
        image_bytes = uploaded_file.getvalue()
        st.session_state.pending_image = (image_bytes, uploaded_file.name, uploaded_file.type)

    active_image = st.session_state.pending_image
    if active_image:
        image_bytes, filename, mime = active_image
        size_mb = len(image_bytes) / (1024 * 1024)
        st.image(image_bytes, caption=f"{filename} ({size_mb:.1f} MB)")
        if size_mb > MAX_CLIENT_UPLOAD_MB:
            st.error(
                f"This image is {size_mb:.1f} MB, which is over the "
                f"{MAX_CLIENT_UPLOAD_MB} MB limit. Please choose a smaller image."
            )

with col2:
    active_image = st.session_state.pending_image
    if active_image is None:
        st.info("Upload a leaf photo (or try the sample image) to get started.")
    else:
        image_bytes, filename, mime = active_image
        size_mb = len(image_bytes) / (1024 * 1024)
        detect_clicked = st.button(
            "🔍 Detect Disease",
            use_container_width=True,
            disabled=size_mb > MAX_CLIENT_UPLOAD_MB,
        )

        if detect_clicked:
            api_url = get_api_url()
            with st.spinner("Analyzing image and contacting API..."):
                try:
                    files = {"file": (filename, image_bytes, mime or "image/jpeg")}
                    response = requests.post(
                        f"{api_url}/disease-detection-file",
                        files=files,
                        timeout=REQUEST_TIMEOUT_SECONDS,
                    )
                except requests.exceptions.Timeout:
                    st.error(
                        "⏱️ The API took too long to respond. It may be cold-starting "
                        "or overloaded — please try again in a moment."
                    )
                    response = None
                except requests.exceptions.ConnectionError:
                    st.error(
                        f"🔌 Couldn't reach the API at `{api_url}`. Check the API URL "
                        "in the sidebar and make sure the backend is running."
                    )
                    response = None
                except requests.exceptions.RequestException as e:
                    st.error(f"Request failed: {e}")
                    response = None

                if response is not None:
                    if response.status_code == 200:
                        result = response.json()
                        st.session_state.history.append({
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "filename": filename,
                            "result": result,
                        })

                        disease_type = result.get("disease_type")
                        confidence = result.get("confidence")

                        if disease_type == "invalid_image":
                            st.markdown("<div class='result-card'>", unsafe_allow_html=True)
                            st.markdown(
                                "<div class='disease-title'>⚠️ Invalid Image</div>",
                                unsafe_allow_html=True)
                            st.markdown(
                                "<div style='color:#ff5722; font-size:1.05em; "
                                "margin-bottom:1em;'>Please upload a clear image of a "
                                "plant leaf for accurate disease detection.</div>",
                                unsafe_allow_html=True)
                            if result.get("treatment"):
                                st.markdown("<div class='section-title'>What to do</div>",
                                            unsafe_allow_html=True)
                                for treat in result.get("treatment", []):
                                    st.markdown(f"- {treat}")
                            st.markdown("</div>", unsafe_allow_html=True)

                        else:
                            healthy = not result.get("disease_detected")
                            st.markdown("<div class='result-card'>", unsafe_allow_html=True)

                            if healthy:
                                st.markdown(
                                    "<div class='disease-title'>✅ Healthy Leaf</div>",
                                    unsafe_allow_html=True)
                                st.markdown(
                                    "<div style='color:#4caf50; font-size:1.05em; "
                                    "margin-bottom:0.8em;'>No disease detected — "
                                    "this plant appears to be healthy!</div>",
                                    unsafe_allow_html=True)
                            else:
                                st.markdown(
                                    f"<div class='disease-title'>🦠 {result.get('disease_name', 'N/A')}</div>",
                                    unsafe_allow_html=True)

                            severity = (result.get("severity") or "none").lower()
                            sev_color = SEVERITY_COLORS.get(severity, "#757575")
                            st.markdown(
                                f"<span class='info-badge'>Type: {result.get('disease_type', 'N/A')}</span>"
                                f"<span class='severity-badge' style='background:{sev_color};'>"
                                f"Severity: {severity.title()}</span>",
                                unsafe_allow_html=True)

                            if isinstance(confidence, (int, float)):
                                st.markdown("<div class='section-title'>Confidence</div>",
                                            unsafe_allow_html=True)
                                st.progress(min(max(int(confidence), 0), 100) / 100)
                                st.caption(f"{confidence:.0f}%")

                            if not healthy:
                                if result.get("symptoms"):
                                    st.markdown("<div class='section-title'>Symptoms</div>",
                                                unsafe_allow_html=True)
                                    for symptom in result.get("symptoms", []):
                                        st.markdown(f"- {symptom}")
                                if result.get("possible_causes"):
                                    st.markdown("<div class='section-title'>Possible Causes</div>",
                                                unsafe_allow_html=True)
                                    for cause in result.get("possible_causes", []):
                                        st.markdown(f"- {cause}")
                                if result.get("treatment"):
                                    st.markdown("<div class='section-title'>Treatment</div>",
                                                unsafe_allow_html=True)
                                    for treat in result.get("treatment", []):
                                        st.markdown(f"- {treat}")

                            st.markdown(
                                f"<div class='timestamp'>🕒 {result.get('analysis_timestamp', 'N/A')}</div>",
                                unsafe_allow_html=True)
                            st.markdown("</div>", unsafe_allow_html=True)

                        st.download_button(
                            "⬇️ Download report (JSON)",
                            data=json.dumps(result, indent=2),
                            file_name=f"leaf_report_{Path(filename).stem}.json",
                            mime="application/json",
                            use_container_width=True,
                        )

                    else:
                        try:
                            detail = response.json().get("detail", response.text)
                        except ValueError:
                            detail = response.text
                        st.error(f"API Error ({response.status_code}): {detail}")

st.markdown(
    "<div style='text-align:center; color:#9e9e9e; margin-top:2em; font-size:0.85em;'>"
    "Built with Streamlit + FastAPI, powered by Llama Vision via Groq."
    "</div>",
    unsafe_allow_html=True,
)
