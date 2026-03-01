import os
import json
import re
import requests
import streamlit as st
from dotenv import load_dotenv
from audiorecorder import audiorecorder

# ----------------------------
# Setup
# ----------------------------
load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "EXAVITQu4vr4xnSDxMaL")  # Sarah (premade)

DEFAULT_LLM_MODEL = os.getenv("MISTRAL_LLM_MODEL", "mistral-tiny-latest")
DEFAULT_STT_MODEL = os.getenv("MISTRAL_STT_MODEL", "voxtral-mini-transcribe-2507")

st.set_page_config(page_title="VoxScholar", page_icon="🎙️", layout="centered")
st.title("🎙️ VoxScholar — Talk to a Research Agent")
st.caption("Record → Voxtral transcribes → Mistral answers → ElevenLabs speaks")

missing = []
if not MISTRAL_API_KEY:
    missing.append("MISTRAL_API_KEY")
if not ELEVENLABS_API_KEY:
    missing.append("ELEVENLABS_API_KEY")

if missing:
    st.error(f"Missing env vars: {', '.join(missing)}. Add them to your .env and restart.")
    st.stop()


# ----------------------------
# API Helpers
# ----------------------------
def voxtral_transcribe(audio_bytes: bytes, filename: str, model: str) -> str:
    url = "https://api.mistral.ai/v1/audio/transcriptions"
    files = {"file": (filename, audio_bytes)}
    data = {"model": model}

    r = requests.post(
        url,
        headers={"Authorization": f"Bearer {MISTRAL_API_KEY}"},
        files=files,
        data=data,
        timeout=180,
    )
    if r.status_code != 200:
        raise RuntimeError(f"Voxtral STT error {r.status_code}: {r.text}")

    js = r.json()
    text = js.get("text")
    if not text:
        raise RuntimeError(f"Unexpected Voxtral response: {js}")
    return text


def mistral_chat(user_text: str, model: str, system_prompt: str = None, temperature: float = 0.2) -> str:
    """
    Same endpoint you already use, but now customizable so we can:
    - ask for JSON (intent/topic extraction)
    - ask for snazzy playlist generation
    """
    url = "https://api.mistral.ai/v1/chat/completions"

    if system_prompt is None:
        system_prompt = (
            "You are VoxScholar, a concise, helpful voice research assistant. "
            "Answer clearly with short bullets when helpful. "
            "If the user asks for a plan, give steps."
        )

    payload = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
    }

    r = requests.post(
        url,
        headers={"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"},
        json=payload,
        timeout=120,
    )
    if r.status_code != 200:
        raise RuntimeError(f"Mistral chat error {r.status_code}: {r.text}")

    js = r.json()
    try:
        return js["choices"][0]["message"]["content"]
    except Exception:
        raise RuntimeError(f"Unexpected Mistral response: {js}")


def elevenlabs_tts(text: str, voice_id: str) -> bytes:
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

    r = requests.post(
        url,
        headers={
            "xi-api-key": ELEVENLABS_API_KEY,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        },
        json={"text": text, "model_id": "eleven_multilingual_v2"},
        timeout=120,
    )
    if r.status_code != 200:
        raise RuntimeError(f"ElevenLabs TTS error {r.status_code}: {r.text}")
    return r.content


# ----------------------------
# Playlist + Personalization helpers (A)
# ----------------------------
def init_state():
    if "interest_profile" not in st.session_state:
        st.session_state.interest_profile = {
            "topics": {},         # str -> weight
            "entities": {},       # str -> weight
            "keywords": {},       # str -> weight
            "recent_queries": [],
        }
    if "last_playlist" not in st.session_state:
        st.session_state.last_playlist = None


def _bump(d, key, inc=1.0, cap=10.0):
    key = (key or "").strip()
    if not key:
        return
    d[key] = min(cap, d.get(key, 0.0) + inc)


def _safe_json_parse(s: str) -> dict:
    s = (s or "").strip()
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{.*\}", s, flags=re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return {}
        return {}


def classify_intent(user_text: str, llm_model: str) -> dict:
    system = "You are an intent classifier for a voice-first research assistant. Return ONLY valid JSON."
    prompt = f"""
Classify the user's intent into one of:
- "playlist": wants recommendations / what to read next / spotify for papers / list of papers
- "explain": wants summary/explanation of a paper or concept
- "compare": wants comparison between approaches/papers
- "followup": refining/continuing previous result ("more", "add", "shorter", "next")
- "other"

User text:
{user_text}

Return JSON with keys: intent, confidence, reason.
"""
    raw = mistral_chat(prompt, llm_model, system_prompt=system, temperature=0.0)
    out = _safe_json_parse(raw)
    if "intent" not in out:
        out = {"intent": "other", "confidence": 0.0, "reason": "parse_failed"}
    return out


def extract_topics(user_text: str, llm_model: str) -> dict:
    system = "Extract personalization signals from text. Return ONLY valid JSON."
    prompt = f"""
Extract:
- topics: broad areas (e.g., transformers, VLMs, RAG)
- entities: specific named things (models/datasets/papers/authors/tools/venues)
- keywords: fine-grained concepts (e.g., contrastive learning, retrieval, attention)

Text:
{user_text}

Return JSON with keys: topics (list), entities (list), keywords (list).
"""
    raw = mistral_chat(prompt, llm_model, system_prompt=system, temperature=0.2)
    out = _safe_json_parse(raw)
    out.setdefault("topics", [])
    out.setdefault("entities", [])
    out.setdefault("keywords", [])
    # de-dupe lightly
    out["topics"] = list(dict.fromkeys([t.strip() for t in out["topics"] if str(t).strip()]))
    out["entities"] = list(dict.fromkeys([e.strip() for e in out["entities"] if str(e).strip()]))
    out["keywords"] = list(dict.fromkeys([k.strip() for k in out["keywords"] if str(k).strip()]))
    return out


def update_interest_profile(user_text: str, topics_obj: dict):
    prof = st.session_state.interest_profile
    prof["recent_queries"].append(user_text)
    prof["recent_queries"] = prof["recent_queries"][-20:]

    for t in topics_obj.get("topics", []):
        _bump(prof["topics"], t, inc=1.5)
    for e in topics_obj.get("entities", []):
        _bump(prof["entities"], e, inc=1.2)
    for k in topics_obj.get("keywords", []):
        _bump(prof["keywords"], k, inc=1.0)


def generate_playlist(seed_text: str, llm_model: str, max_items: int = 6) -> dict:
    prof = st.session_state.interest_profile
    top_topics = [k for k, _ in sorted(prof["topics"].items(), key=lambda x: -x[1])[:8]]
    top_entities = [k for k, _ in sorted(prof["entities"].items(), key=lambda x: -x[1])[:6]]
    top_keywords = [k for k, _ in sorted(prof["keywords"].items(), key=lambda x: -x[1])[:10]]

    system = "You are VoxScholar, a voice-first research discovery agent. Return ONLY valid JSON (no markdown)."
    prompt = f"""
Create a Spotify-style research paper playlist.

Seed:
{seed_text}

Personalization context:
- Top topics: {top_topics}
- Top entities: {top_entities}
- Top keywords: {top_keywords}

Constraints:
- Exactly {max_items} items.
- Use plausible paper titles. Do NOT claim they are real links.
- Each item must include:
  paper_title, one_liner, why_this, difficulty (beginner|intermediate|advanced),
  tags (2-4), estimated_minutes (10-35)
- Add playlist title + 1-sentence description.
- Add listening_order (list of 3 section strings).

Return JSON with keys: title, description, listening_order, items.
"""
    raw = mistral_chat(prompt, llm_model, system_prompt=system, temperature=0.6)
    playlist = _safe_json_parse(raw)
    if not isinstance(playlist.get("items"), list):
        playlist["items"] = []
    playlist["items"] = playlist["items"][:max_items]
    st.session_state.last_playlist = playlist
    return playlist


def render_interest_profile():
    prof = st.session_state.interest_profile
    st.subheader("🧠 Interest profile (this session)")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.caption("Top topics")
        for k, v in sorted(prof["topics"].items(), key=lambda x: -x[1])[:8]:
            st.write(f"- {k} ({v:.1f})")
    with c2:
        st.caption("Top keywords")
        for k, v in sorted(prof["keywords"].items(), key=lambda x: -x[1])[:10]:
            st.write(f"- {k} ({v:.1f})")
    with c3:
        st.caption("Top entities")
        for k, v in sorted(prof["entities"].items(), key=lambda x: -x[1])[:8]:
            st.write(f"- {k} ({v:.1f})")

    if st.button("Reset profile"):
        st.session_state.interest_profile = {"topics": {}, "entities": {}, "keywords": {}, "recent_queries": []}
        st.session_state.last_playlist = None
        st.success("Reset.")


def render_playlist(playlist: dict):
    st.subheader("🎧 Paper Playlist")
    st.markdown(f"### {playlist.get('title','')}")
    if playlist.get("description"):
        st.caption(playlist["description"])

    lo = playlist.get("listening_order", [])
    if isinstance(lo, list) and lo:
        st.markdown("**Listening order:** " + " → ".join([str(x) for x in lo]))

    st.caption("Note: These are generated suggestions (not verified links). arXiv integration can come next.")

    for i, it in enumerate(playlist.get("items", []), start=1):
        with st.container(border=True):
            st.markdown(f"**{i}. {it.get('paper_title','')}**")
            st.write(it.get("one_liner", ""))
            st.caption(f"Why: {it.get('why_this','')}")
            meta = []
            if it.get("difficulty"):
                meta.append(f"Difficulty: {it.get('difficulty')}")
            if it.get("estimated_minutes"):
                meta.append(f"~{it.get('estimated_minutes')} min")
            if meta:
                st.caption(" • ".join(meta))
            tags = it.get("tags", [])
            if isinstance(tags, list) and tags:
                st.caption("Tags: " + ", ".join([str(t) for t in tags]))


def playlist_to_speak_text(playlist: dict) -> str:
    title = playlist.get("title", "Your Research Playlist")
    desc = playlist.get("description", "")
    items = playlist.get("items", [])

    parts = [f"{title}.", desc, "Here’s your lineup:"]
    for idx, it in enumerate(items, start=1):
        parts.append(
            f"{idx}. {it.get('paper_title','Untitled')}. "
            f"{it.get('why_this','')}. "
            f"Difficulty: {it.get('difficulty','')}. "
        )
    return " ".join([p for p in parts if p and str(p).strip()])


# init session state
init_state()


# ----------------------------
# UI Controls
# ----------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    st.subheader("Models")
    stt_model = st.text_input("Voxtral STT model", value=DEFAULT_STT_MODEL)
    llm_model = st.text_input("Mistral LLM model", value=DEFAULT_LLM_MODEL)

    st.subheader("ElevenLabs")
    voice_id = st.text_input("ElevenLabs Voice ID", value=ELEVENLABS_VOICE_ID)
    st.caption("Example premade: EXAVITQu4vr4xnSDxMaL (Sarah)")

    st.subheader("Cost / Safety")
    max_chars_to_speak = st.slider("Max characters to speak", 200, 4000, 1200, 100)
    st.caption("Limits TTS cost + prevents super long audio.")

    st.subheader("Mode")
    force_playlist = st.toggle("🎧 Force playlist mode", value=False)
    show_profile = st.toggle("Show interest profile", value=True)


# ----------------------------
# Main: Record or Type
# ----------------------------
st.write("## 1) Speak (recommended)")
st.write("Click to record, click again to stop. Then we transcribe + answer + speak back.")

audio = audiorecorder("🎤 Record", "⏺️ Recording...")

audio_bytes = None
if len(audio) > 0:
    wav_io = audio.export(format="wav")
    audio_bytes = wav_io.read()
    st.audio(audio_bytes, format="audio/wav")
    st.success("Recording captured ✅")

st.write("---")
st.write("## 2) Or type (backup)")
typed = st.text_area("Type your question", placeholder="e.g., Build me a playlist of papers on RAG for production...")

# Choose input
final_input = None
mode = None
if audio_bytes:
    mode = "voice"
    final_input = audio_bytes
elif typed.strip():
    mode = "text"
    final_input = typed.strip()

if final_input is None:
    st.info("Record audio or type a message to begin.")
    st.stop()

# ----------------------------
# Run pipeline
# ----------------------------
if mode == "voice":
    st.write("## 2) Transcribe (Voxtral)")
    try:
        with st.spinner("Transcribing..."):
            transcript = voxtral_transcribe(audio_bytes, "recording.wav", stt_model)
        st.text_area("Transcript", transcript, height=140)
    except Exception as e:
        st.error(f"Transcription failed: {e}")
        st.stop()

    user_text = transcript
else:
    user_text = typed.strip()

# Update personalization every turn (cheap-ish)
try:
    with st.spinner("Updating interest profile..."):
        topics_obj = extract_topics(user_text, llm_model)
    update_interest_profile(user_text, topics_obj)
except Exception:
    # If JSON parsing fails, don't break the demo
    pass

if show_profile:
    with st.expander("🧠 Personalization", expanded=False):
        render_interest_profile()

# Route intent (or force playlist)
intent = "playlist" if force_playlist else "other"
if not force_playlist:
    try:
        with st.spinner("Routing..."):
            intent_obj = classify_intent(user_text, llm_model)
        intent = intent_obj.get("intent", "other")
    except Exception:
        intent = "other"

# ----------------------------
# Branch: Playlist vs Normal answer
# ----------------------------
if intent == "playlist":
    st.write("## 3) Curate (Playlist Mode)")
    try:
        with st.spinner("Building your playlist..."):
            playlist = generate_playlist(user_text, llm_model, max_items=6)
        render_playlist(playlist)
    except Exception as e:
        st.error(f"Playlist generation failed: {e}")
        st.stop()

    st.write("## 4) Speak (ElevenLabs)")
    speak_text = playlist_to_speak_text(playlist)[:max_chars_to_speak]
    try:
        with st.spinner("Generating speech..."):
            audio_out = elevenlabs_tts(speak_text, voice_id)
        st.audio(audio_out, format="audio/mpeg")
    except Exception as e:
        st.error(f"ElevenLabs TTS failed: {e}")
        st.info("You can still use the playlist UI; TTS is optional for the demo.")
else:
    st.write("## 3) Think (Mistral)")
    try:
        with st.spinner("Thinking..."):
            answer = mistral_chat(user_text, llm_model)
        st.text_area("Answer", answer, height=200)
    except Exception as e:
        st.error(f"Mistral call failed: {e}")
        st.stop()

    st.write("## 4) Speak (ElevenLabs)")
    speak_text = answer[:max_chars_to_speak]
    try:
        with st.spinner("Generating speech..."):
            audio_out = elevenlabs_tts(speak_text, voice_id)
        st.audio(audio_out, format="audio/mpeg")
    except Exception as e:
        st.error(f"ElevenLabs TTS failed: {e}")
        st.info("You can still use the transcript + answer; TTS is optional for the demo.")