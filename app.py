
import io
import re

import librosa
import numpy as np
import soundfile as sf
import streamlit as st
from pocket_tts import TTSModel


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Pocket TTS Conversation Studio",
    page_icon="🎙️",
    layout="centered",
)

st.title("🎙️ Pocket TTS Conversation Studio")
st.caption("Create natural two-person conversations from text.")


# --------------------------------------------------
# MODEL AND VOICE LOADING
# --------------------------------------------------

@st.cache_resource
def load_model():
    return TTSModel.load_model()


@st.cache_resource
def load_voice_state(voice_name):
    model = load_model()
    return model.get_state_for_audio_prompt(voice_name)


# --------------------------------------------------
# AVAILABLE VOICES
# --------------------------------------------------

VOICES = [
    "alba",
    "marius",
    "javert",
    "jean",
    "fantine",
    "cosette",
    "eponine",
    "azelma",
]


# --------------------------------------------------
# SCRIPT PARSER
# --------------------------------------------------

def parse_script(script):
    turns = []
    current_speaker = None

    for line in script.splitlines():
        line = line.strip()

        if not line:
            continue

        match = re.match(r"^([AB])\s*:\s*(.*)$", line, re.IGNORECASE)

        if match:
            speaker = match.group(1).upper()
            text = match.group(2).strip()

            if text:
                turns.append({
                    "speaker": speaker,
                    "text": text,
                })

            current_speaker = speaker

        elif current_speaker and turns:
            turns[-1]["text"] += " " + line

        else:
            raise ValueError(
                "Every line must start with A: or B:. "
                "Continuation lines are allowed after a speaker label."
            )

    if not turns:
        raise ValueError("Your script is empty.")

    speakers_used = {turn["speaker"] for turn in turns}

    if speakers_used != {"A", "B"}:
        raise ValueError(
            "Your script must contain at least one A: turn "
            "and one B: turn."
        )

    return turns


# --------------------------------------------------
# AUDIO SPEED ADJUSTMENT
# --------------------------------------------------

def adjust_speed(audio, speed):
    if speed == 1.0:
        return audio

    return librosa.effects.time_stretch(
        y=audio.astype(np.float32),
        rate=speed,
    )


# --------------------------------------------------
# INTERFACE: SPEAKER SETTINGS
# --------------------------------------------------

st.subheader("1. Configure your speakers")

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("### 🅰️ Speaker A")

    voice_a = st.selectbox(
        "Choose A's voice",
        VOICES,
        index=0,
        key="voice_a",
    )

    speed_a = st.slider(
        "A's speaking speed",
        min_value=0.75,
        max_value=1.35,
        value=1.0,
        step=0.05,
        key="speed_a",
    )

with col_b:
    st.markdown("### 🅱️ Speaker B")

    voice_b = st.selectbox(
        "Choose B's voice",
        VOICES,
        index=1,
        key="voice_b",
    )

    speed_b = st.slider(
        "B's speaking speed",
        min_value=0.75,
        max_value=1.35,
        value=1.0,
        step=0.05,
        key="speed_b",
    )


# --------------------------------------------------
# INTERFACE: SCRIPT
# --------------------------------------------------

st.divider()
st.subheader("2. Write your conversation")

default_script = """A: Hey! How are you doing today?
B: I'm doing great! What about you?
A: I'm good too. Have you heard about Pocket TTS?
B: Yes! It can generate speech using different voices.
A: Exactly. We can use it to create our own conversations.
B: That's actually pretty impressive!"""

script = st.text_area(
    "Conversation script",
    value=default_script,
    height=280,
    help="Start each new turn with A: or B:. "
         "Continuation lines belong to the previous speaker.",
)

st.caption(
    "Example: A: Hello!  |  B: Hi! "
    "The app generates each turn in the order you write it."
)


# --------------------------------------------------
# GENERATE CONVERSATION
# --------------------------------------------------

st.divider()
st.subheader("3. Generate your conversation")

if st.button(
    "🎙️ Generate full conversation",
    type="primary",
    use_container_width=True,
):

    try:
        turns = parse_script(script)

        model = load_model()

        st.info(f"Generating {len(turns)} conversation turns...")

        progress = st.progress(0)
        status = st.empty()

        # Load both voice states once.
        status.write("Loading Speaker A and Speaker B voices...")

        state_a = load_voice_state(voice_a)
        state_b = load_voice_state(voice_b)

        voice_states = {
            "A": state_a,
            "B": state_b,
        }

        voice_speeds = {
            "A": speed_a,
            "B": speed_b,
        }

        audio_segments = []

        sample_rate = model.sample_rate

        # Small silence between conversation turns.
        pause_duration = 0.25
        silence = np.zeros(
            int(sample_rate * pause_duration),
            dtype=np.float32,
        )

        # Generate every speaker turn in order.
        for index, turn in enumerate(turns):

            speaker = turn["speaker"]
            text = turn["text"]

            status.write(
                f"Generating turn {index + 1}/{len(turns)} "
                f"— Speaker {speaker}..."
            )

            audio_tensor = model.generate_audio(
                voice_states[speaker],
                text,
            )

            audio = (
                audio_tensor
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            # Apply this speaker's individual speed.
            audio = adjust_speed(
                audio,
                voice_speeds[speaker],
            )

            audio_segments.append(audio)
            audio_segments.append(silence)

            progress.progress(
                (index + 1) / len(turns)
            )

        # Merge all turns into one audio file.
        status.write("Combining all conversation turns...")

        final_audio = np.concatenate(audio_segments)

        # Convert generated audio into WAV.
        output_buffer = io.BytesIO()

        sf.write(
            output_buffer,
            final_audio,
            sample_rate,
            format="WAV",
        )

        wav_bytes = output_buffer.getvalue()

        # Save results in session state.
        st.session_state["conversation_audio"] = wav_bytes
        st.session_state["conversation_duration"] = (
            len(final_audio) / sample_rate
        )

        progress.empty()
        status.empty()

        st.success("Your conversation is ready!")

    except Exception as error:
        st.error("Conversation generation failed.")
        st.exception(error)


# --------------------------------------------------
# FINAL AUDIO PLAYER AND DOWNLOAD
# --------------------------------------------------

if "conversation_audio" in st.session_state:

    st.divider()
    st.subheader("4. Your completed conversation")

    duration = st.session_state["conversation_duration"]

    st.caption(f"Duration: {duration:.1f} seconds")

    st.audio(
        st.session_state["conversation_audio"],
        format="audio/wav",
    )

    st.download_button(
        label="⬇️ Download complete conversation",
        data=st.session_state["conversation_audio"],
        file_name="pocket_tts_conversation.wav",
        mime="audio/wav",
        use_container_width=True,
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Powered by Kyutai Pocket TTS. "
    "Generation runs on the hosting server's CPU."
)
