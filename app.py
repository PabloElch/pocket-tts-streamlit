
import io
import os
import re
import tempfile

import librosa
import numpy as np
import soundfile as sf
import streamlit as st
from pocket_tts import TTSModel


# ============================================================
# LENCHOS STUDIO
# ============================================================

st.set_page_config(
    page_title="Lenchos Studio",
    page_icon="🎙️",
    layout="wide",
)

st.title("🎙️ Lenchos Studio")
st.caption("Your personal AI voice and conversation studio.")

# ============================================================
# SESSION STORAGE
# ============================================================

if "custom_voices" not in st.session_state:
    st.session_state["custom_voices"] = {}

if "voice_metadata" not in st.session_state:
    st.session_state["voice_metadata"] = {}

if "final_audio" not in st.session_state:
    st.session_state["final_audio"] = None


# ============================================================
# BUILT-IN VOICE LIBRARY
# ============================================================

BUILTIN_VOICES = {
    "alba": {
        "name": "Alba",
        "gender": "F",
        "description": "Natural feminine voice",
    },
    "javert": {
        "name": "Javert",
        "gender": "M",
        "description": "Masculine voice",
    },
    "marius": {
        "name": "Marius",
        "gender": "M",
        "description": "Natural masculine voice",
    },
    "mary": {
        "name": "Mary",
        "gender": "F",
        "description": "Feminine voice",
    },
    "michael": {
        "name": "Michael",
        "gender": "M",
        "description": "Masculine voice",
    },
    "paul": {
        "name": "Paul",
        "gender": "M",
        "description": "Masculine voice",
    },
    "peter_yearsley": {
        "name": "Peter Yearsley",
        "gender": "M",
        "description": "Masculine voice",
    },
    "stuart_bell": {
        "name": "Stuart Bell",
        "gender": "M",
        "description": "Masculine voice",
    },
    "vera": {
        "name": "Vera",
        "gender": "F",
        "description": "Feminine voice",
    },
}


# ============================================================
# MODEL
# ============================================================

@st.cache_resource
def load_model():
    return TTSModel.load_model()


@st.cache_resource
def load_builtin_voice(voice_id):
    model = load_model()
    return model.get_state_for_audio_prompt(voice_id)


def get_voice_state(voice_key):
    if voice_key.startswith("builtin:"):
        voice_id = voice_key.split(":", 1)[1]
        return load_builtin_voice(voice_id)

    if voice_key.startswith("custom:"):
        voice_name = voice_key.split(":", 1)[1]
        return st.session_state["custom_voices"][voice_name]

    raise ValueError("Unknown voice selection.")


# ============================================================
# VOICE SELECTOR HELPERS
# ============================================================

def get_voice_options():
    options = []

    for voice_id, info in BUILTIN_VOICES.items():
        options.append("builtin:" + voice_id)

    for voice_name in st.session_state["custom_voices"]:
        options.append("custom:" + voice_name)

    return options


def voice_label(key):
    if key.startswith("builtin:"):
        voice_id = key.split(":", 1)[1]
        info = BUILTIN_VOICES[voice_id]

        return (
            f"{info['name']} ({info['gender']}) "
            f"— {info['description']}"
        )

    voice_name = key.split(":", 1)[1]
    info = st.session_state["voice_metadata"][voice_name]

    return (
        f"⭐ {voice_name} ({info['gender']}) "
        f"— {info['description']} [Cloned]"
    )


# ============================================================
# AUDIO HELPERS
# ============================================================

def generate_audio(voice_key, text):
    model = load_model()
    voice_state = get_voice_state(voice_key)

    audio_tensor = model.generate_audio(
        voice_state,
        text,
        copy_state=True,
    )

    audio = (
        audio_tensor
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    return audio, model.sample_rate


def change_speed(audio, speed):
    if abs(speed - 1.0) < 0.001:
        return audio

    return librosa.effects.time_stretch(
        y=audio.astype(np.float32),
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


# ============================================================
# SCRIPT PARSER
# ============================================================

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
                "Every new conversation turn must start with A: or B:."
            )

    if not turns:
        raise ValueError("The conversation script is empty.")

    speakers = {turn["speaker"] for turn in turns}

    if speakers != {"A", "B"}:
        raise ValueError(
            "Please include at least one A: turn and one B: turn."
        )

    return turns


# ============================================================
# NAVIGATION
# ============================================================

st.divider()

page = st.radio(
    "Studio navigation",
    ["💬 Conversation Studio", "🎭 Voice Cloning"],
    horizontal=True,
    label_visibility="collapsed",
)

st.divider()


# ============================================================
# PAGE 1: VOICE CLONING
# ============================================================

if page == "🎭 Voice Cloning":

    st.header("🎭 Voice Cloning Studio")

    st.write(
        "Upload a voice recording, give your cloned voice a name, "
        "and use it later in your conversations."
    )

    st.info(
        "Use only your own voice or recordings for which you have "
        "explicit permission to clone the speaker."
    )

    st.subheader("1. Upload a voice recording")

    uploaded_file = st.file_uploader(
        "Choose an audio recording",
        type=["wav", "mp3", "flac", "ogg", "m4a"],
        help="Use a clear recording with minimal background noise.",
    )

    if uploaded_file:

        st.audio(
            uploaded_file.getvalue(),
            format=uploaded_file.type,
        )

        st.caption(
            f"File: {uploaded_file.name} | "
            f"Size: {uploaded_file.size / (1024 * 1024):.2f} MB"
        )

    st.subheader("2. Give your voice a name")

    voice_name = st.text_input(
        "Voice name",
        placeholder="Example: My Podcast Voice",
    )

    col_gender, col_description = st.columns(2)

    with col_gender:
        clone_gender = st.selectbox(
            "Voice label",
            ["F", "M", "Other / Not specified"],
        )

    with col_description:
        clone_description = st.text_input(
            "Description",
            placeholder="Example: Deep, calm, conversational",
        )

    consent = st.checkbox(
        "I own this voice or have permission from the speaker to clone it."
    )

    st.caption(
        "Pocket TTS processes up to the first 30 seconds of the "
        "recording for voice conditioning."
    )

    if st.button(
        "✨ Clone This Voice",
        type="primary",
        use_container_width=True,
    ):

        if not uploaded_file:
            st.warning("Please upload an audio recording.")

        elif not voice_name.strip():
            st.warning("Please give your cloned voice a name.")

        elif not consent:
            st.warning(
                "Please confirm that you have permission to clone this voice."
            )

        elif voice_name.strip() in st.session_state["custom_voices"]:
            st.warning(
                "A cloned voice with that name already exists. "
                "Choose another name."
            )

        else:
            temporary_path = None

            try:
                with st.spinner(
                    "Loading Pocket TTS and processing your voice..."
                ):

                    suffix = os.path.splitext(
                        uploaded_file.name
                    )[1].lower()

                    with tempfile.NamedTemporaryFile(
                        suffix=suffix,
                        delete=False,
                    ) as temporary_file:

                        temporary_file.write(
                            uploaded_file.getvalue()
                        )

                        temporary_path = temporary_file.name

                    model = load_model()

                    cloned_state = model.get_state_for_audio_prompt(
                        temporary_path
                    )

                    clean_name = voice_name.strip()

                    st.session_state["custom_voices"][clean_name] = (
                        cloned_state
                    )

                    st.session_state["voice_metadata"][clean_name] = {
                        "gender": clone_gender,
                        "description": (
                            clone_description.strip()
                            or "Custom cloned voice"
                        ),
                    }

                st.success(
                    f"Successfully created the voice: {clean_name}"
                )

                st.info(
                    "Your new voice is now available in Conversation Studio."
                )

            except Exception as error:
                st.error("Voice cloning failed.")
                st.exception(error)

            finally:
                if temporary_path and os.path.exists(temporary_path):
                    os.remove(temporary_path)

    # --------------------------------------------------------
    # CLONED VOICE LIBRARY
    # --------------------------------------------------------

    st.divider()
    st.subheader("3. Your cloned voice library")

    custom_voices = st.session_state["custom_voices"]

    if not custom_voices:
        st.info(
            "No cloned voices yet. Upload a recording above to create one."
        )

    else:
        for name in list(custom_voices.keys()):

            info = st.session_state["voice_metadata"][name]

            with st.expander(
                f"⭐ {name} ({info['gender']})"
            ):

                st.write(info["description"])

                if st.button(
                    f"▶ Preview {name}",
                    key=f"preview_clone_{name}",
                ):

                    try:
                        with st.spinner("Generating voice preview..."):

                            audio, sr = generate_audio(
                                "custom:" + name,
                                "Hello! Welcome to Lenchos Studio. "
                                "This is a preview of my cloned voice.",
                            )

                            preview = make_wav(audio, sr)

                            st.session_state[
                                "clone_preview_" + name
                            ] = preview

                    except Exception as error:
                        st.error("Preview failed.")
                        st.exception(error)

                preview_key = "clone_preview_" + name

                if preview_key in st.session_state:
                    st.audio(
                        st.session_state[preview_key],
                        format="audio/wav",
                    )

                if st.button(
                    f"Delete {name}",
                    key=f"delete_clone_{name}",
                ):

                    del st.session_state["custom_voices"][name]
                    del st.session_state["voice_metadata"][name]

                    if preview_key in st.session_state:
                        del st.session_state[preview_key]

                    st.rerun()


# ============================================================
# PAGE 2: CONVERSATION STUDIO
# ============================================================

elif page == "💬 Conversation Studio":

    st.header("💬 Two-Person Conversation Studio")

    st.write(
        "Choose two voices, preview them, adjust their individual "
        "speaking speeds, and turn your script into one audio file."
    )

    voice_options = get_voice_options()

    # --------------------------------------------------------
    # SPEAKER CONFIGURATION
    # --------------------------------------------------------

    st.subheader("1. Configure your speakers")

    col_a, col_b = st.columns(2)

    with col_a:

        st.markdown("### 🅰️ Speaker A")

        voice_a = st.selectbox(
            "Choose Speaker A",
            voice_options,
            format_func=voice_label,
            key="conversation_voice_a",
        )

        speed_a = st.slider(
            "Speaking speed — A",
            min_value=0.75,
            max_value=1.35,
            value=1.0,
            step=0.05,
            key="conversation_speed_a",
        )

        if st.button(
            "▶ Preview Speaker A",
            use_container_width=True,
            key="conversation_preview_a",
        ):

            try:
                with st.spinner("Generating Speaker A preview..."):

                    audio, sr = generate_audio(
                        voice_a,
                        "Hello! Welcome to Lenchos Studio. "
                        "How are you doing today?",
                    )

                    audio = change_speed(audio, speed_a)

                    st.session_state["conversation_preview_a_audio"] = (
                        make_wav(audio, sr)
                    )

            except Exception as error:
                st.error("Speaker A preview failed.")
                st.exception(error)

        if "conversation_preview_a_audio" in st.session_state:
            st.audio(
                st.session_state["conversation_preview_a_audio"],
                format="audio/wav",
            )

    with col_b:

        st.markdown("### 🅱️ Speaker B")

        voice_b = st.selectbox(
            "Choose Speaker B",
            voice_options,
            index=min(1, len(voice_options) - 1),
            format_func=voice_label,
            key="conversation_voice_b",
        )

        speed_b = st.slider(
            "Speaking speed — B",
            min_value=0.75,
            max_value=1.35,
            value=1.0,
            step=0.05,
            key="conversation_speed_b",
        )

        if st.button(
            "▶ Preview Speaker B",
            use_container_width=True,
            key="conversation_preview_b",
        ):

            try:
                with st.spinner("Generating Speaker B preview..."):

                    audio, sr = generate_audio(
                        voice_b,
                        "Hello! It's great to be here. "
                        "What are we talking about today?",
                    )

                    audio = change_speed(audio, speed_b)

                    st.session_state["conversation_preview_b_audio"] = (
                        make_wav(audio, sr)
                    )

            except Exception as error:
                st.error("Speaker B preview failed.")
                st.exception(error)

        if "conversation_preview_b_audio" in st.session_state:
            st.audio(
                st.session_state["conversation_preview_b_audio"],
                format="audio/wav",
            )

    # --------------------------------------------------------
    # SCRIPT
    # --------------------------------------------------------

    st.divider()
    st.subheader("2. Write your conversation")

    default_script = """A: Hey! How are you doing today?
B: I'm doing great! What about you?
A: I'm good too. Have you heard about Lenchos Studio?
B: Yes! We can use different voices to create a conversation.
A: Exactly. We can even clone our own voices.
B: That's impressive! Let's make something amazing."""

    script = st.text_area(
        "Conversation script",
        value=default_script,
        height=280,
        help="Use A: and B: to identify the speakers.",
    )

    st.caption(
        "Each A: line uses Speaker A. Each B: line uses Speaker B. "
        "Continuation lines remain part of the previous turn."
    )

    # --------------------------------------------------------
    # GENERATE
    # --------------------------------------------------------

    st.divider()
    st.subheader("3. Generate full conversation")

    if st.button(
        "🎙️ Generate Full Conversation",
        type="primary",
        use_container_width=True,
    ):

        try:
            turns = parse_script(script)

            model = load_model()
            sample_rate = model.sample_rate

            states = {
                "A": get_voice_state(voice_a),
                "B": get_voice_state(voice_b),
            }

            speeds = {
                "A": speed_a,
                "B": speed_b,
            }

            audio_parts = []

            pause = np.zeros(
                int(sample_rate * 0.25),
                dtype=np.float32,
            )

            progress = st.progress(0)
            status = st.empty()

            for index, turn in enumerate(turns):

                speaker = turn["speaker"]
                text = turn["text"]

                status.write(
                    f"Generating turn {index + 1}/{len(turns)} "
                    f"— Speaker {speaker}"
                )

                audio_tensor = model.generate_audio(
                    states[speaker],
                    text,
                    copy_state=True,
                )

                audio = (
                    audio_tensor
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )

                audio = change_speed(
                    audio,
                    speeds[speaker],
                )

                audio_parts.append(audio)

                if index < len(turns) - 1:
                    audio_parts.append(pause)

                progress.progress(
                    (index + 1) / len(turns)
                )

            status.write("Combining all conversation turns...")

            final_audio = np.concatenate(audio_parts)

            wav_bytes = make_wav(
                final_audio,
                sample_rate,
            )

            st.session_state["final_audio"] = wav_bytes

            st.session_state["final_duration"] = (
                len(final_audio) / sample_rate
            )

            progress.empty()
            status.empty()

            st.success("Your conversation is ready!")

        except Exception as error:
            st.error("Conversation generation failed.")
            st.exception(error)

    # --------------------------------------------------------
    # FINAL AUDIO
    # --------------------------------------------------------

    if st.session_state["final_audio"] is not None:

        st.divider()
        st.subheader("4. Your finished conversation")

        st.metric(
            "Audio duration",
            f"{st.session_state['final_duration']:.1f} seconds",
        )

        st.audio(
            st.session_state["final_audio"],
            format="audio/wav",
        )

        st.download_button(
            "⬇️ Download Final Conversation (WAV)",
            data=st.session_state["final_audio"],
            file_name="lenchos_studio_conversation.wav",
            mime="audio/wav",
            use_container_width=True,
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Lenchos Studio | Powered by Kyutai Pocket TTS"
)
