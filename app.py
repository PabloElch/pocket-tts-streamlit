
import io
import re

import librosa
import numpy as np
import soundfile as sf
import streamlit as st
from pocket_tts import TTSModel


# ==================================================
# LENCHOS STUDIO
# ==================================================

st.set_page_config(
    page_title="Lenchos Studio",
    page_icon="🎙️",
    layout="wide",
)

st.title("🎙️ Lenchos Studio")
st.caption("Two voices. One conversation. Your own audio studio.")

# ==================================================
# VOICE LIBRARY
# ==================================================

VOICES = {
    "alba": {
        "name": "Alba",
        "gender": "F",
        "description": "Clear, natural female voice",
    },
    "marius": {
        "name": "Marius",
        "gender": "M",
        "description": "Warm, masculine voice",
    },
    "javert": {
        "name": "Javert",
        "gender": "M",
        "description": "Deep, authoritative male voice",
    },
    "jean": {
        "name": "Jean",
        "gender": "M",
        "description": "Gentle male voice",
    },
    "fantine": {
        "name": "Fantine",
        "gender": "F",
        "description": "Soft, feminine voice",
    },
    "cosette": {
        "name": "Cosette",
        "gender": "F",
        "description": "Young-sounding female voice",
    },
    "eponine": {
        "name": "Eponine",
        "gender": "F",
        "description": "Expressive female voice",
    },
    "azelma": {
        "name": "Azelma",
        "gender": "F",
        "description": "Bright female voice",
    },
}


def voice_label(voice_id):
    voice = VOICES[voice_id]

    return (
        f"{voice['name']} ({voice['gender']}) "
        f"— {voice['description']}"
    )


# ==================================================
# MODEL LOADING
# ==================================================

@st.cache_resource
def load_model():
    return TTSModel.load_model()


@st.cache_resource
def load_voice_state(voice_id):
    model = load_model()
    return model.get_state_for_audio_prompt(voice_id)


# ==================================================
# AUDIO GENERATION
# ==================================================

def generate_audio(voice_id, text):
    model = load_model()
    voice_state = load_voice_state(voice_id)

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
        y=audio,
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


# ==================================================
# SCRIPT PARSER
# ==================================================

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
                "Every new conversation turn must begin with A: or B:."
            )

    if not turns:
        raise ValueError("Your script is empty.")

    speakers = {turn["speaker"] for turn in turns}

    if speakers != {"A", "B"}:
        raise ValueError(
            "Please include at least one A: turn and one B: turn."
        )

    return turns


# ==================================================
# SPEAKER CONFIGURATION
# ==================================================

st.divider()
st.header("1. Configure your voices")

col_a, col_b = st.columns(2)

with col_a:
    st.subheader("🅰️ Speaker A")

    voice_a = st.selectbox(
        "Choose Speaker A's voice",
        options=list(VOICES.keys()),
        index=0,
        format_func=voice_label,
        key="voice_a",
    )

    speed_a = st.slider(
        "Speaking speed — A",
        min_value=0.75,
        max_value=1.35,
        value=1.0,
        step=0.05,
        key="speed_a",
    )

    if st.button(
        "▶ Preview Speaker A",
        use_container_width=True,
        key="preview_a",
    ):
        try:
            with st.spinner("Generating Speaker A preview..."):
                audio, sr = generate_audio(
                    voice_a,
                    "Hello! Welcome to Lenchos Studio. "
                    "This is a preview of my voice.",
                )

                audio = change_speed(audio, speed_a)

                st.session_state["preview_a_audio"] = make_wav(
                    audio, sr
                )

        except Exception as error:
            st.error(f"Preview failed: {error}")

    if "preview_a_audio" in st.session_state:
        st.audio(
            st.session_state["preview_a_audio"],
            format="audio/wav",
        )

with col_b:
    st.subheader("🅱️ Speaker B")

    voice_b = st.selectbox(
        "Choose Speaker B's voice",
        options=list(VOICES.keys()),
        index=1,
        format_func=voice_label,
        key="voice_b",
    )

    speed_b = st.slider(
        "Speaking speed — B",
        min_value=0.75,
        max_value=1.35,
        value=1.0,
        step=0.05,
        key="speed_b",
    )

    if st.button(
        "▶ Preview Speaker B",
        use_container_width=True,
        key="preview_b",
    ):
        try:
            with st.spinner("Generating Speaker B preview..."):
                audio, sr = generate_audio(
                    voice_b,
                    "Hello! It's great to meet you. "
                    "This is my voice preview.",
                )

                audio = change_speed(audio, speed_b)

                st.session_state["preview_b_audio"] = make_wav(
                    audio, sr
                )

        except Exception as error:
            st.error(f"Preview failed: {error}")

    if "preview_b_audio" in st.session_state:
        st.audio(
            st.session_state["preview_b_audio"],
            format="audio/wav",
        )


# ==================================================
# CONVERSATION SCRIPT
# ==================================================

st.divider()
st.header("2. Write your conversation")

default_script = """A: Hey! How are you doing today?
B: I'm doing great! What about you?
A: I'm good too. Have you heard about Lenchos Studio?
B: Yes! It can turn a written conversation into two-person audio.
A: Exactly. We can choose different voices for each speaker.
B: That's impressive! Let's create something amazing."""

script = st.text_area(
    "Your conversation",
    value=default_script,
    height=300,
    help="Use A: and B: to identify the speakers.",
)

st.markdown("**Script format example:**")

st.code(
    """A: Hello! How are you?
B: I'm doing well, thank you.
A: What are you working on?
B: I'm creating a podcast.""",
    language="text",
)


# ==================================================
# GENERATE FULL CONVERSATION
# ==================================================

st.divider()
st.header("3. Generate your conversation")

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
            "A": load_voice_state(voice_a),
            "B": load_voice_state(voice_b),
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
                f"Generating turn {index + 1} of {len(turns)} "
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

        status.write("Combining the conversation...")

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


# ==================================================
# FINAL AUDIO
# ==================================================

if "final_audio" in st.session_state:

    st.divider()
    st.header("4. Your finished conversation")

    duration = st.session_state["final_duration"]

    st.metric(
        "Audio duration",
        f"{duration:.1f} seconds",
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


# ==================================================
# FOOTER
# ==================================================

st.divider()

st.caption(
    "Lenchos Studio | Powered by Kyutai Pocket TTS"
)
