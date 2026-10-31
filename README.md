
# Pocket TTS Studio

A simple web interface for generating speech using [Pocket TTS](https://github.com/kyutai-labs/pocket-tts).

## Features

- Built-in voice selection
- Text-to-speech generation
- Audio playback
- WAV download
- Runs through Streamlit

## Run locally

Install the dependencies:

```bash
pip install -r requirements.txt
```

Start the app:

```bash
streamlit run app.py
```

## Deployment

Deploy this repository using Streamlit Community Cloud.

Select `app.py` as the main file.

The first launch may take some time because the model needs to download and load.
