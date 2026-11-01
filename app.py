import hmac
import io
import json
import os
import re
import tempfile
import uuid
from urllib.parse import quote

import librosa
import numpy as np
import requests
import soundfile as sf
import streamlit as st
from pocket_tts import TTSModel


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Lenchos Studio",
    page_icon="🎙️",
    layout="wide",
)


# =========================================================
# PRIVATE APP ACCESS
# =========================================================

try:
    APP_PASSWORD = str(st.secrets.get("APP_PASSWORD", ""))
except Exception:
    APP_PASSWORD = ""

if not APP_PASSWORD:
    st.error("App password is not configured. Contact the app owner.")
    st.stop()

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🎙️ Lenchos Studio")
    st.caption("Your private AI voice library and conversation studio.")

    password = st.text_input(
        "Enter your private app password",
        type="password",
    )

    if st.button("Unlock Studio", type="primary"):
        if hmac.compare_digest(password, APP_PASSWORD):
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password.")

    st.stop()


# =========================================================
# CONFIGURATION
# =========================================================

BUCKET = "lenchos-voices"
LIBRARY_FILE = "library.json"

BUILTIN_VOICES = {
    "alba": ("Alba", "F", "Feminine voice"),
    "jean": ("Jean", "M", "Masculine voice"),
    "javert": ("Javert", "M", "Masculine voice"),
    "marius": ("Marius", "M", "Masculine voice"),
    "jane": ("Jane", "F", "Feminine voice"),
    "mary": ("Mary", "F", "Feminine voice"),
    "michael": ("Michael", "M", "Masculine voice"),
    "paul": ("Paul", "M", "Masculine voice"),
    "peter_yearsley": ("Peter Yearsley", "M", "Masculine voice"),
    "stuart_bell": ("Stuart Bell", "M", "Masculine voice"),
    "vera": ("Vera", "F", "Feminine voice"),
}


# =========================================================
# SUPABASE CONFIGURATION
# =========================================================

def get_supabase_config():
    try:
        base_url = str(st.secrets["SUPABASE_URL"]).rstrip("/")
        key = str(st.secrets["SUPABASE_SERVICE_ROLE_KEY"])

        if not base_url or not key:
