# app.py
import streamlit as st

from researchmix.config import init_app_config, require_env
from researchmix.state import (
    ensure_global_state,
    apply_pending_nav,
    NAV_PAGES,
    goto,
    u_bucket,
)
from researchmix.auth import render_login_gate
from researchmix.ui.components import render_top_bar, render_now_playing_compact
from researchmix.ui.pages import render_home, render_paper, render_chat, render_library


# ----------------------------
# Init / Config
# ----------------------------
init_app_config()
require_env()

# ----------------------------
# Global state + auth gate
# ----------------------------
ensure_global_state()

if not render_login_gate():
    st.stop()

# IMPORTANT: apply pending nav BEFORE nav widget is created
apply_pending_nav()

# ----------------------------
# Top bar + now playing
# ----------------------------
render_top_bar()
render_now_playing_compact()

# ----------------------------
# Sidebar: navigation + settings
# ----------------------------
with st.sidebar:
    st.header("🧭 Navigate")
    nav = st.radio("Go to", NAV_PAGES, key="nav", label_visibility="collapsed")

    st.divider()
    st.header("⚙️ Settings")

    bucket = u_bucket()

    st.subheader("Models")
    bucket["settings"]["stt_model"] = st.text_input(
        "Voxtral STT model",
        value=bucket["settings"]["stt_model"],
    )
    bucket["settings"]["llm_model"] = st.text_input(
        "Mistral LLM model",
        value=bucket["settings"]["llm_model"],
    )

    st.subheader("ElevenLabs")
    bucket["settings"]["voice_id"] = st.text_input(
        "ElevenLabs Voice ID",
        value=bucket["settings"]["voice_id"],
    )

    st.subheader("Cost / Safety")
    bucket["settings"]["max_chars_to_speak"] = st.slider(
        "Max characters to speak",
        200,
        4000,
        bucket["settings"]["max_chars_to_speak"],
        100,
    )

    st.divider()
    st.subheader("Personalization (optional)")
    interests_text = st.text_input(
        "Interests (comma-separated)",
        value=", ".join(bucket["user_profile"].get("interests") or []),
        placeholder="e.g., agents, recsys, VLM, RAG, evaluation",
    )
    about_me = st.text_area(
        "About",
        value=bucket["user_profile"].get("about_me") or "",
        placeholder="e.g., I like implementable papers; benchmarks; system design",
        height=80,
    )
    if st.button("Save profile", use_container_width=True):
        interests = [x.strip() for x in interests_text.split(",") if x.strip()]
        bucket["user_profile"]["interests"] = interests
        bucket["user_profile"]["about_me"] = about_me.strip()
        st.success("Saved ✅")

    if st.button("🔄 Refresh playlists", use_container_width=True):
        bucket["playlists"]["trending"] = []
        bucket["playlists"]["recommended"] = []
        st.success("Will refresh on Home.")


# ----------------------------
# Route pages
# ----------------------------
if nav == "Home":
    render_home()
elif nav == "Paper":
    render_paper()
elif nav == "Chat":
    render_chat()
else:
    render_library()