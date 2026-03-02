# researchmix/config.py
import os
import json
import streamlit as st
from dotenv import load_dotenv


def init_app_config():
    load_dotenv()

    st.set_page_config(
        page_title="ResearchMix",
        page_icon="🎙️",
        layout="wide",
    )


def get_env():
    """
    Central place to read env vars.
    """
    mistral_api_key = os.getenv("MISTRAL_API_KEY")
    eleven_api_key = os.getenv("ELEVENLABS_API_KEY")
    eleven_voice_id = os.getenv("ELEVENLABS_VOICE_ID", "EXAVITQu4vr4xnSDxMaL")

    default_llm_model = os.getenv("MISTRAL_LLM_MODEL", "mistral-tiny-latest")
    default_stt_model = os.getenv("MISTRAL_STT_MODEL", "voxtral-mini-transcribe-2507")

    demo_users_json = os.getenv("DEMO_USERS_JSON", "")
    default_demo_users = {"demo": "demo", "judge": "demo", "fairgame": "researchmix"}

    try:
        demo_users = json.loads(demo_users_json) if demo_users_json.strip() else default_demo_users
    except Exception:
        demo_users = default_demo_users

    return {
        "MISTRAL_API_KEY": mistral_api_key,
        "ELEVENLABS_API_KEY": eleven_api_key,
        "ELEVENLABS_VOICE_ID": eleven_voice_id,
        "DEFAULT_LLM_MODEL": default_llm_model,
        "DEFAULT_STT_MODEL": default_stt_model,
        "DEMO_USERS": demo_users,
    }


def require_env():
    env = get_env()
    missing = []
    if not env["MISTRAL_API_KEY"]:
        missing.append("MISTRAL_API_KEY")
    if not env["ELEVENLABS_API_KEY"]:
        missing.append("ELEVENLABS_API_KEY")

    if missing:
        st.error(f"Missing env vars: {', '.join(missing)}. Add them to your .env and restart.")
        st.stop()