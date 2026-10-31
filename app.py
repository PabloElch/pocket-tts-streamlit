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

# Private app access
APP_PASSWORD = st.secrets.get("APP_PASSWORD", "")

if not APP_PASSWORD:
    st.error("App password is not configured. Contact the app owner.")
    st.stop()

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("Lenchos Studio")
    password = st.text_input("Enter your private app password", type="password")

    if st.button("Unlock Studio"):
        if hmac.compare_digest(password, APP_PASSWORD):
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password.")

    st.stop()
# =========================================================
# LENCHOS STUDIO
# =========================================================

st.set_page_config(
    page_title="Lenchos Studio",
    page_icon="🎙️",
    layout="wide",
)

st.title("🎙️ Lenchos Studio")
st.caption("Your permanent AI voice library and conversation studio.")


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


def get_supabase_config():
    try:
        return (
            st.secrets["SUPABASE_URL"].rstrip("/"),
            st.secrets["SUPABASE_SERVICE_ROLE_KEY"],
        )
    except Exception:
        return None, None


def storage_headers(key, content_type="application/octet-stream"):
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": content_type,
    }


def object_url(base, bucket, path, authenticated=False):
    endpoint = "object/authenticated" if authenticated else "object"
    return (
        f"{base}/storage/v1/{endpoint}/"
        f"{quote(bucket, safe='')}/{quote(path, safe='/')}"
    )


def download_object(path):
    base, key = get_supabase_config()

    if not base or not key:
        raise RuntimeError("Supabase storage is not configured.")

    response = requests.get(
        object_url(base, BUCKET, path, authenticated=True),
        headers=storage_headers(key),
        timeout=60,
    )

   if not response.ok:
    raise RuntimeError(
        f"Supabase returned HTTP {response.status_code}: {response.text}"
    )
    return response.content


def upload_object(path, data, content_type="application/octet-stream"):
    base, key = get_supabase_config()

    if not base or not key:
        raise RuntimeError("Supabase storage is not configured.")

    headers = storage_headers(key, content_type)
    headers["x-upsert"] = "true"

    response = requests.post(
        object_url(base, BUCKET, path),
        headers=headers,
        data=data,
        timeout=120,
    )

    response.raise_for_status()


def delete_object(path):
    base, key = get_supabase_config()

    if not base or not key:
        raise RuntimeError("Supabase storage is not configured.")

    response = requests.delete(
        f"{base}/storage/v1/object/{BUCKET}",
        headers={
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        json={"prefixes": [path]},
        timeout=60,
    )

    response.raise_for_status()


# =========================================================
# PERMANENT VOICE LIBRARY
# =========================================================

def load_library():
    try:
        data = download_object(LIBRARY_FILE)
        return json.loads(data.decode("utf-8"))
    except requests.HTTPError as error:
        if error.response is not None and error.response.status_code == 404:
            return []
        raise
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise RuntimeError("The voice library index is invalid.")


def save_library(voices):
    upload_object(
        LIBRARY_FILE,
        json.dumps(voices, indent=2).encode("utf-8"),
        "application/json",
    )


# =========================================================
# MODEL
# =========================================================

@st.cache_resource
def load_model():
    return TTSModel.load_model()


@st.cache_resource
def load_builtin_state(voice_id):
    return load_model().get_state_for_audio_prompt(voice_id)


@st.cache_resource
def load_custom_state(audio_bytes):
    model = load_model()
    path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False,
        ) as file:
            file.write(audio_bytes)
            path = file.name

        return model.get_state_for_audio_prompt(path)

    finally:
        if path and os.path.exists(path):
            os.remove(path)


def get_voice_state(voice_key, library):
    if voice_key.startswith("builtin:"):
        voice_id = voice_key.split(":", 1)[1]
        return load_builtin_state(voice_id)

    if voice_key.startswith("custom:"):
        voice_id = voice_key.split(":", 1)[1]

        voice = next(
            (v for v in library if v["id"] == voice_id),
            None,
        )

        if voice is None:
            raise ValueError("This saved voice no longer exists.")

        audio_bytes = download_object(voice["file"])
        return load_custom_state(audio_bytes)

    raise ValueError("Invalid voice selection.")


# =========================================================
# AUDIO UTILITIES
# =========================================================

def generate_audio(state, text):
    model = load_model()

    audio = model.generate_audio(
        state,
        text,
        copy_state=True,
    )

    return audio.detach().cpu().numpy().astype(np.float32)


def change_speed(audio, speed):
    if speed == 1.0:
        return audio

    return librosa.effects.time_stretch(
        audio.astype(np.float32),
        rate=speed,
    )


def make_wav(audio, sample_rate):
    buffer = io.BytesIO()

    sf.write(
        buffer,
        audio,
        sample_rate,
        format="WAV",
    )

    return buffer.getvalue()


# =========================================================
# SCRIPT PARSER
# =========================================================

def parse_script(script):
    turns = []
    current_speaker = None

    for line in script.splitlines():
        line = line.strip()

        if not line:
            continue

        match = re.match(
            r"^([AB])\s*:\s*(.*)$",
            line,
            re.IGNORECASE,
        )

        if match:
            current_speaker = match.group(1).upper()
            content = match.group(2).strip()

            if content:
                turns.append({
                    "speaker": current_speaker,
                    "text": content,
                })

        elif current_speaker and turns:
            turns[-1]["text"] += " " + line

        else:
            raise ValueError(
                "Start each conversation turn with A: or B:."
            )

    if not turns:
        raise ValueError("Your script is empty.")

    if {t["speaker"] for t in turns} != {"A", "B"}:
        raise ValueError("Include at least one A: and one B: turn.")

    return turns


# =========================================================
# LOAD LIBRARY
# =========================================================

base_url, api_key = get_supabase_config()
storage_ready = bool(base_url and api_key)

if storage_ready:
    try:
        library = load_library()
    except Exception as error:
        library = []
        st.error("Could not load your saved voice library.")
        st.exception(error)
else:
    library = []


# =========================================================
# NAVIGATION
# =========================================================

page = st.radio(
    "Navigation",
    ["💬 Conversation Studio", "🎭 Voice Library"],
    horizontal=True,
    label_visibility="collapsed",
)

st.divider()


# =========================================================
# VOICE OPTIONS
# =========================================================

def voice_options():
    result = ["builtin:" + v for v in BUILTIN_VOICES]

    for voice in library:
        result.append("custom:" + voice["id"])

    return result


def voice_label(key):
    if key.startswith("builtin:"):
        voice_id = key.split(":", 1)[1]
        name, gender, description = BUILTIN_VOICES[voice_id]
        return f"{name} ({gender}) — {description}"

    voice_id = key.split(":", 1)[1]

    voice = next(
        v for v in library if v["id"] == voice_id
    )

    return (
        f"⭐ {voice['name']} ({voice['gender']}) "
        f"— {voice['description']} [Saved]"
    )


# =========================================================
# VOICE LIBRARY PAGE
# =========================================================

if page == "🎭 Voice Library":

    st.header("🎭 Your Permanent Voice Library")

    if not storage_ready:
        st.warning(
            "Persistent storage is not connected yet. "
            "Complete the Supabase setup before saving voices."
        )

    st.write(
        "Clone a voice once, save it, and reuse it whenever "
        "you create a conversation."
    )

    st.subheader("Add a new voice")

    uploaded = st.file_uploader(
        "Upload a clear voice recording",
        type=["wav"],
        help="WAV only. Use a clean recording of up to 30 seconds.",
    )

    if uploaded:
        st.audio(uploaded.getvalue(), format="audio/wav")

    name = st.text_input(
        "Voice name",
        placeholder="Example: My Podcast Voice",
    )

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox(
            "Gender label",
            ["F", "M", "Other / Not specified"],
        )

    with col2:
        description = st.text_input(
            "Voice description",
            placeholder="Deep, calm, conversational...",
        )

    consent = st.checkbox(
        "I own this voice or have explicit permission to clone it."
    )

    if st.button(
        "✨ Clone and Save Voice Permanently",
        type="primary",
        use_container_width=True,
        disabled=not storage_ready,
    ):

        if not uploaded:
            st.warning("Upload a WAV recording first.")

        elif not name.strip():
            st.warning("Enter a name for this voice.")

        elif not consent:
            st.warning("Please confirm permission to clone this voice.")

        elif any(
            v["name"].lower() == name.strip().lower()
            for v in library
        ):
            st.warning("A voice with this name already exists.")

        else:
            try:
                with st.spinner("Processing and saving your voice..."):

                    raw_audio = uploaded.getvalue()

                    audio, sr = sf.read(
                        io.BytesIO(raw_audio),
                        dtype="float32",
                    )

                    if audio.ndim == 2:
                        audio = np.mean(audio, axis=1)

                    if len(audio) == 0:
                        raise ValueError("The recording is empty.")

                    if sr != 24000:
                        audio = librosa.resample(
                            audio,
                            orig_sr=sr,
                            target_sr=24000,
                        )
                        sr = 24000

                    # Use at most the first 30 seconds.
                    audio = audio[: int(sr * 30)]

                    normalized_wav = make_wav(audio, sr)

                    # Verify Pocket TTS can process the recording.
                    with tempfile.NamedTemporaryFile(
                        suffix=".wav",
                        delete=False,
                    ) as temp:
                        temp.write(normalized_wav)
                        temp_path = temp.name

                    try:
                        model = load_model()
                        model.get_state_for_audio_prompt(temp_path)
                    finally:
                        os.remove(temp_path)

                    voice_id = str(uuid.uuid4())
                    file_path = f"voices/{voice_id}.wav"

                    upload_object(
                        file_path,
                        normalized_wav,
                        "audio/wav",
                    )

                    new_voice = {
                        "id": voice_id,
                        "name": name.strip(),
                        "gender": gender,
                        "description": (
                            description.strip()
                            or "Custom cloned voice"
                        ),
                        "file": file_path,
                    }

                    library.append(new_voice)
                    save_library(library)

                st.success(
                    f"{name.strip()} has been saved to your permanent library!"
                )

                st.rerun()

            except Exception as error:
                st.error("Voice cloning or saving failed.")
                st.exception(error)

    # -----------------------------------------------------
    # SAVED VOICES
    # -----------------------------------------------------

    st.divider()
    st.subheader(f"Saved voices ({len(library)})")

    if not library:
        st.info("Your permanent voice library is currently empty.")

    for voice in library:

        with st.expander(
            f"⭐ {voice['name']} ({voice['gender']})"
        ):

            st.write(voice["description"])

            if st.button(
                f"▶ Preview {voice['name']}",
                key=f"preview_{voice['id']}",
            ):

                try:
                    with st.spinner("Generating preview..."):

                        state = get_voice_state(
                            "custom:" + voice["id"],
                            library,
                        )

                        audio = generate_audio(
                            state,
                            "Hello! Welcome to Lenchos Studio. "
                            "This is a preview of my saved voice.",
                        )

                        st.session_state[
                            "preview_" + voice["id"]
                        ] = make_wav(
                            audio,
                            load_model().sample_rate,
                        )

                except Exception as error:
                    st.error("Preview failed.")
                    st.exception(error)

            preview_key = "preview_" + voice["id"]

            if preview_key in st.session_state:
                st.audio(
                    st.session_state[preview_key],
                    format="audio/wav",
                )

            if st.button(
                f"Delete {voice['name']}",
                key=f"delete_{voice['id']}",
            ):

                try:
                    delete_object(voice["file"])

                    library = [
                        v for v in library
                        if v["id"] != voice["id"]
                    ]

                    save_library(library)

                    st.success("Voice deleted from your library.")
                    st.rerun()

                except Exception as error:
                    st.error("Could not delete this voice.")
                    st.exception(error)


# =========================================================
# CONVERSATION STUDIO
# =========================================================

elif page == "💬 Conversation Studio":

    st.header("💬 Two-Person Conversation Studio")

    if not storage_ready:
        st.info(
            "Built-in voices are available now. "
            "Connect Supabase to use permanently saved cloned voices."
        )

    options = voice_options()

    col_a, col_b = st.columns(2)

    with col_a:

        st.subheader("🅰️ Speaker A")

        voice_a = st.selectbox(
            "Choose voice A",
            options,
            format_func=voice_label,
            key="speaker_a",
        )

        speed_a = st.slider(
            "Speaking speed A",
            0.75,
            1.35,
            1.0,
            0.05,
        )

        if st.button("▶ Preview A", use_container_width=True):

            try:
                state = get_voice_state(voice_a, library)

                audio = generate_audio(
                    state,
                    "Hello! Welcome to Lenchos Studio.",
                )

                audio = change_speed(audio, speed_a)

                st.session_state["preview_a"] = make_wav(
                    audio,
                    load_model().sample_rate,
                )

            except Exception as error:
                st.error("Preview failed.")
                st.exception(error)

        if "preview_a" in st.session_state:
            st.audio(st.session_state["preview_a"])

    with col_b:

        st.subheader("🅱️ Speaker B")

        voice_b = st.selectbox(
            "Choose voice B",
            options,
            index=min(1, len(options) - 1),
            format_func=voice_label,
            key="speaker_b",
        )

        speed_b = st.slider(
            "Speaking speed B",
            0.75,
            1.35,
            1.0,
            0.05,
        )

        if st.button("▶ Preview B", use_container_width=True):

            try:
                state = get_voice_state(voice_b, library)

                audio = generate_audio(
                    state,
                    "Hello! It's great to be here.",
                )

                audio = change_speed(audio, speed_b)

                st.session_state["preview_b"] = make_wav(
                    audio,
                    load_model().sample_rate,
                )

            except Exception as error:
                st.error("Preview failed.")
                st.exception(error)

        if "preview_b" in st.session_state:
            st.audio(st.session_state["preview_b"])

    st.divider()
    st.subheader("Write your script")

    script = st.text_area(
        "A: and B: conversation",
        value="""A: Hey! How are you doing today?
B: I'm doing great! What about you?
A: I'm good too. Have you heard about Lenchos Studio?
B: Yes! We can save cloned voices and reuse them.
A: That's right. Let's make our own podcast.
B: Sounds amazing!""",
        height=260,
    )

    if st.button(
        "🎙️ Generate Full Conversation",
        type="primary",
        use_container_width=True,
    ):

        try:
            turns = parse_script(script)

            model = load_model()

            states = {
                "A": get_voice_state(voice_a, library),
                "B": get_voice_state(voice_b, library),
            }

            speeds = {
                "A": speed_a,
                "B": speed_b,
            }

            parts = []

            sr = model.sample_rate
            silence = np.zeros(int(sr * 0.25), dtype=np.float32)

            progress = st.progress(0)
            status = st.empty()

            for i, turn in enumerate(turns):

                speaker = turn["speaker"]

                status.write(
                    f"Generating turn {i + 1}/{len(turns)} "
                    f"— Speaker {speaker}"
                )

                audio = generate_audio(
                    states[speaker],
                    turn["text"],
                )

                audio = change_speed(
                    audio,
                    speeds[speaker],
                )

                parts.append(audio)

                if i < len(turns) - 1:
                    parts.append(silence)

                progress.progress((i + 1) / len(turns))

            status.write("Combining conversation...")

            final_audio = np.concatenate(parts)
            wav = make_wav(final_audio, sr)

            st.session_state["final_audio"] = wav
            st.session_state["final_duration"] = len(final_audio) / sr

            progress.empty()
            status.empty()

            st.success("Your conversation is ready!")

        except Exception as error:
            st.error("Conversation generation failed.")
            st.exception(error)

    if st.session_state.get("final_audio"):

        st.divider()
        st.subheader("Your finished conversation")

        st.metric(
            "Duration",
            f"{st.session_state['final_duration']:.1f} seconds",
        )

        st.audio(
            st.session_state["final_audio"],
            format="audio/wav",
        )

        st.download_button(
            "⬇️ Download final conversation",
            data=st.session_state["final_audio"],
            file_name="lenchos_studio_conversation.wav",
            mime="audio/wav",
            use_container_width=True,
        )


# =========================================================
# FOOTER
# =========================================================

st.divider()
st.caption("Lenchos Studio | Powered by Kyutai Pocket TTS")
