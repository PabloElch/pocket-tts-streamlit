
import io
import os
import tempfile

import streamlit as st
import torch
import soundfile as sf
from pocket_tts import TTSModel

st.set_page_config(
    page_title="Pocket TTS Studio",
    page_icon="🎙️",
    layout="centered",
)

st.title("🎙️ Pocket TTS Studio")
st.caption("Generate natural speech with Pocket TTS.")

@st.cache_resource(show_spinner="Loading Pocket TTS model...")
def load_model():
    return TTSModel.load_model()

@st.cache_resource
def load_voice_state(_model, voice_name):
    return _model.get_state_for_audio_prompt(voice_name)

try:
    model = load_model()
except Exception as e:
    st.error("Could not load the Pocket TTS model.")
    st.exception(e)
    st.stop()

st.subheader("1. Choose a voice")

voices = [
    "alba",
    "marius",
    "javert",
    "jean",
    "fantine",
    "cosette",
    "eponine",
    "azelma",
]

voice_choice = st.selectbox("Built-in voice", voices)

st.subheader("2. Enter your script")

script = st.text_area(
    "Text to speak",
    height=180,
    placeholder="Enter the text you want to turn into speech...",
)

st.subheader("3. Generate audio")

if st.button("Generate speech", type="primary", use_container_width=True):
    if not script.strip():
        st.warning("Please enter some text first.")
    else:
        try:
            with st.spinner("Generating speech..."):
                voice_state = load_voice_state(model, voice_choice)
                audio = model.generate_audio(
                    voice_state,
                    script.strip(),
                )

                                audio_np = audio.detach().cpu().numpy()
                sample_rate = model.sample_rate

                buffer = io.BytesIO()
                sf.write(buffer, audio_np, sample_rate, format="WAV")
                wav_bytes = buffer.getvalue()

            st.success("Audio generated!")
            st.audio(wav_bytes, format="audio/wav")

            st.download_button(
                "Download WAV",
                data=wav_bytes,
                file_name="pocket_tts_output.wav",
                mime="audio/wav",
                use_container_width=True,
            )

        except Exception as e:
            st.error("Audio generation failed.")
            st.exception(e)

st.divider()
st.caption("Powered by Pocket TTS. Model loading and generation may take time on free hosting.")
