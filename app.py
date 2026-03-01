# app.py
import os
import json
import uuid
import requests
import streamlit as st
from dotenv import load_dotenv
from audiorecorder import audiorecorder
from typing import Optional, Dict, Any, List

# ----------------------------
# Setup
# ----------------------------
load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "EXAVITQu4vr4xnSDxMaL")  # Sarah (premade)

DEFAULT_LLM_MODEL = os.getenv("MISTRAL_LLM_MODEL", "mistral-tiny-latest")
DEFAULT_STT_MODEL = os.getenv("MISTRAL_STT_MODEL", "voxtral-mini-transcribe-2507")

st.set_page_config(page_title="ResearchMix", page_icon="🎙️", layout="wide")

missing = []
if not MISTRAL_API_KEY:
    missing.append("MISTRAL_API_KEY")
if not ELEVENLABS_API_KEY:
    missing.append("ELEVENLABS_API_KEY")

if missing:
    st.error(f"Missing env vars: {', '.join(missing)}. Add them to your .env and restart.")
    st.stop()

# ----------------------------
# Key helper (fixes StreamlitDuplicateElementKey)
# ----------------------------
def ui_key(*parts: Any) -> str:
    """
    Create stable, globally-unique Streamlit keys.
    Use this for ALL buttons/widgets that can appear more than once.
    """
    safe = []
    for p in parts:
        if p is None:
            p = "none"
        safe.append(str(p).replace(" ", "_"))
    return "k:" + ":".join(safe)

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


def mistral_chat_raw(messages, model: str, temperature: float = 0.2, max_tokens: int = 900) -> str:
    url = "https://api.mistral.ai/v1/chat/completions"
    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": messages,
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


def mistral_chat(user_text: str, model: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are VoxScholar, a concise, helpful voice research assistant. "
                "Answer clearly with short bullets when helpful. "
                "If the user asks for a plan, give steps."
            ),
        },
        {"role": "user", "content": user_text},
    ]
    return mistral_chat_raw(messages, model=model, temperature=0.2, max_tokens=900)


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
# State
# ----------------------------
def _init_state():
    if "paper_cache" not in st.session_state:
        st.session_state.paper_cache = {}  # paper_id -> paper dict
    if "paper_views" not in st.session_state:
        st.session_state.paper_views = {}  # paper_id -> {quick_summary, deep_dive, suggested_qs}
    if "history" not in st.session_state:
        st.session_state.history = []  # list of paper_ids (most recent first)
    if "last_played" not in st.session_state:
        st.session_state.last_played = None
    if "selected_paper" not in st.session_state:
        st.session_state.selected_paper = None
    if "saved" not in st.session_state:
        st.session_state.saved = set()
    if "playlists" not in st.session_state:
        st.session_state.playlists = {"recommended": [], "trending": []}
    if "user_profile" not in st.session_state:
        st.session_state.user_profile = {"interests": [], "about_me": ""}
    if "now_playing_mode" not in st.session_state:
        st.session_state.now_playing_mode = "summary"  # summary|deepdive|qa
    if "tts_last_audio" not in st.session_state:
        st.session_state.tts_last_audio = None
    if "last_error" not in st.session_state:
        st.session_state.last_error = None

_init_state()


# ----------------------------
# Utilities
# ----------------------------
def upsert_paper(paper: Dict[str, Any]) -> str:
    paper_id = paper.get("paper_id") or f"llm:{uuid.uuid4().hex[:10]}"
    paper["paper_id"] = paper_id
    st.session_state.paper_cache[paper_id] = paper
    return paper_id


def bump_history(paper_id: str):
    hist = st.session_state.history
    if paper_id in hist:
        hist.remove(paper_id)
    hist.insert(0, paper_id)
    st.session_state.history = hist[:50]


def set_selected_paper(paper_id: str):
    if paper_id not in st.session_state.paper_cache:
        return
    st.session_state.selected_paper = paper_id
    bump_history(paper_id)
    st.session_state.last_played = paper_id


def get_paper(paper_id: str) -> Optional[Dict[str, Any]]:
    return st.session_state.paper_cache.get(paper_id)


def get_view(paper_id: str) -> Dict[str, Any]:
    if paper_id not in st.session_state.paper_views:
        st.session_state.paper_views[paper_id] = {
            "quick_summary": None,
            "deep_dive": None,
            "suggested_qs": None,
        }
    return st.session_state.paper_views[paper_id]


def safe_json_extract(text: str):
    t = text.strip()
    if t.startswith("```"):
        parts = t.split("```")
        if len(parts) >= 3:
            t = parts[1].strip()
            if "\n" in t and t.split("\n", 1)[0].strip().lower() in ("json", "javascript"):
                t = t.split("\n", 1)[1].strip()

    try:
        return json.loads(t)
    except Exception:
        pass

    for open_ch, close_ch in [("[", "]"), ("{", "}")]:
        start = t.find(open_ch)
        end = t.rfind(close_ch)
        if start != -1 and end != -1 and end > start:
            snippet = t[start : end + 1]
            try:
                return json.loads(snippet)
            except Exception:
                continue
    return None


def compute_reco_seed_text() -> str:
    interests = st.session_state.user_profile.get("interests") or []
    about = (st.session_state.user_profile.get("about_me") or "").strip()

    hist_titles = []
    for pid in st.session_state.history[:8]:
        p = get_paper(pid)
        if p:
            hist_titles.append(p.get("title", ""))

    parts = []
    if interests:
        parts.append("Interests: " + ", ".join(interests))
    if about:
        parts.append("About: " + about)
    if hist_titles:
        parts.append("Recent papers listened to: " + " | ".join([t for t in hist_titles if t]))

    if not parts:
        return "No user history. Default to broadly popular ML/AI research topics and practical engineering relevance."
    return "\n".join(parts)


def paper_aware_system_prompt(paper: Optional[Dict[str, Any]]) -> str:
    base = (
        "You are VoxScholar, a reliable research assistant. "
        "Be concise, structured, and avoid hallucinating. "
        "If you are unsure, say what is missing and offer the best next step. "
        "Prefer bullet points. "
    )
    if not paper:
        return base + "The user may ask about any research topic."

    title = paper.get("title", "Unknown title")
    abstract = paper.get("abstract", "")
    topics = paper.get("topics", [])
    topics_str = ", ".join(topics) if topics else "unknown"

    return (
        base
        + "The user is currently focused on the following paper.\n"
        + f"TITLE: {title}\n"
        + f"TOPICS: {topics_str}\n"
        + (f"ABSTRACT: {abstract}\n" if abstract else "ABSTRACT: (not available; rely on title + user context)\n")
        + "Answer questions primarily about this paper; if the user asks generally, respond normally.\n"
    )


# ----------------------------
# LLM generators
# ----------------------------
def llm_generate_papers(kind: str, n: int, model: str) -> List[Dict[str, Any]]:
    seed = compute_reco_seed_text() if kind == "recommended" else "Trending + hot papers across ML/AI, practical and exciting."

    sys = (
        "You generate plausible research paper metadata for a hackathon demo.\n"
        "IMPORTANT:\n"
        "- Output MUST be valid JSON only (no markdown, no commentary).\n"
        "- Generate realistic titles and abstracts, not too long.\n"
        "- Do NOT use real DOIs unless you are sure; it's okay to be synthetic.\n"
        "- Keep each abstract to 3-6 sentences.\n"
        "- Include a short 'topics' list (3-6 items).\n"
        "- Include 'year' as an integer.\n"
    )
    user = (
        f"Generate {n} {kind} papers as a JSON array. "
        "Each item: {"
        "\"title\": str, "
        "\"authors\": [str, ...] (2-5 names), "
        "\"year\": int (2019-2026), "
        "\"abstract\": str, "
        "\"topics\": [str, ...]"
        "}\n\n"
        f"Personalization seed:\n{seed}\n\n"
        "Ensure diversity (different subtopics)."
    )

    out = mistral_chat_raw(
        messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
        model=model,
        temperature=0.7 if kind == "trending" else 0.5,
        max_tokens=1400,
    )
    js = safe_json_extract(out)
    if not isinstance(js, list):
        return []

    papers = []
    for item in js:
        if not isinstance(item, dict):
            continue
        title = (item.get("title") or "").strip()
        abstract = (item.get("abstract") or "").strip()
        if not title:
            continue
        papers.append(
            {
                "paper_id": f"llm:{uuid.uuid4().hex[:10]}",
                "title": title,
                "authors": item.get("authors") or [],
                "year": item.get("year") or 2025,
                "abstract": abstract,
                "topics": item.get("topics") or [],
                "source": f"llm_{kind}",
            }
        )
    return papers


def llm_quick_summary(paper: Dict[str, Any], model: str) -> str:
    sys = (
        "You are a research paper summarizer.\n"
        "Write a quick summary that is highly readable and useful.\n"
        "Output format:\n"
        "- 1 line: TL;DR\n"
        "- 5 bullets: key contributions/results\n"
        "- 1 line: Best use-case / when to use\n"
        "Do not invent citations or numbers unless present in the abstract.\n"
    )
    user = (
        f"TITLE: {paper.get('title','')}\n"
        f"AUTHORS: {', '.join(paper.get('authors') or [])}\n"
        f"YEAR: {paper.get('year','')}\n"
        f"ABSTRACT:\n{paper.get('abstract','')}\n"
    )
    return mistral_chat_raw(
        messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
        model=model,
        temperature=0.2,
        max_tokens=700,
    )


def llm_deep_dive(paper: Dict[str, Any], model: str) -> str:
    sys = (
        "You are a research explainer.\n"
        "Give a structured deep dive suitable for a spoken narration and Q&A.\n"
        "Use headings and concise bullets.\n"
        "Include:\n"
        "1) Problem + motivation\n"
        "2) Core idea (plain English)\n"
        "3) Method outline\n"
        "4) What the experiments likely look like (high level)\n"
        "5) Limitations + open questions\n"
        "Avoid making up dataset names or exact metrics.\n"
    )
    user = f"TITLE: {paper.get('title','')}\nABSTRACT:\n{paper.get('abstract','')}\n"
    return mistral_chat_raw(
        messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
        model=model,
        temperature=0.25,
        max_tokens=1100,
    )


def llm_suggested_questions(paper: Dict[str, Any], model: str) -> List[str]:
    sys = (
        "Generate suggested questions a user can ask about a research paper.\n"
        "Output MUST be JSON only: an array of 6 short questions.\n"
        "No markdown.\n"
    )
    user = f"TITLE: {paper.get('title','')}\nABSTRACT:\n{paper.get('abstract','')}\nGenerate 6 questions."
    out = mistral_chat_raw(
        messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
        model=model,
        temperature=0.5,
        max_tokens=350,
    )
    js = safe_json_extract(out)
    if isinstance(js, list):
        qs = [q.strip() for q in js if isinstance(q, str) and q.strip()]
        return qs[:6]
    return []


# ----------------------------
# Sidebar controls
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

    st.subheader("Personalization")
    interests_text = st.text_input(
        "Interests (comma-separated)",
        value=", ".join(st.session_state.user_profile.get("interests") or []),
        placeholder="e.g., recsys, agents, vision-language, efficient LLMs",
    )
    about_me = st.text_area(
        "About (optional)",
        value=st.session_state.user_profile.get("about_me") or "",
        placeholder="e.g., I like practical papers I can implement; edge deployment; eval + benchmarks",
        height=90,
    )
    if st.button("Save profile", key=ui_key("sidebar", "save_profile"), use_container_width=True):
        interests = [x.strip() for x in interests_text.split(",") if x.strip()]
        st.session_state.user_profile["interests"] = interests
        st.session_state.user_profile["about_me"] = about_me.strip()
        st.success("Saved ✅")

    st.divider()
    if st.button("🔄 Regenerate playlists", key=ui_key("sidebar", "regen_playlists"), use_container_width=True):
        st.session_state.playlists["recommended"] = []
        st.session_state.playlists["trending"] = []
        st.success("Will regenerate on Home.")


# ----------------------------
# UI components
# ----------------------------
def render_now_playing():
    pid = st.session_state.last_played
    if not pid:
        return
    p = get_paper(pid)
    if not p:
        return

    mode = st.session_state.now_playing_mode
    title = p.get("title", "Unknown")
    year = p.get("year", "")
    topics = p.get("topics") or []
    topics_str = ", ".join(topics[:4]) if topics else ""

    with st.container(border=True):
        cols = st.columns([6, 2, 2, 2])
        cols[0].markdown(
            f"**Now Playing:** {title} ({year})  \n"
            f"<span style='opacity:0.8'>Mode:</span> `{mode}`"
            + (f"  \n<span style='opacity:0.8'>Topics:</span> {topics_str}" if topics_str else ""),
            unsafe_allow_html=True,
        )

        if cols[1].button("📄 Open", key=ui_key("now", "open", pid), use_container_width=True):
            st.session_state.selected_paper = pid
            st.rerun()

        if cols[2].button(
            "⭐ Save" if pid not in st.session_state.saved else "✅ Saved",
            key=ui_key("now", "save", pid),
            use_container_width=True,
        ):
            if pid in st.session_state.saved:
                st.session_state.saved.remove(pid)
            else:
                st.session_state.saved.add(pid)
            st.rerun()

        if cols[3].button(
            "🗣️ Speak last",
            key=ui_key("now", "speak_last", pid),
            use_container_width=True,
            disabled=st.session_state.tts_last_audio is None,
        ):
            if st.session_state.tts_last_audio:
                st.audio(st.session_state.tts_last_audio, format="audio/mpeg")


def paper_card(paper_id: str, show_actions: bool = True, key_prefix: str = "card"):
    """
    key_prefix MUST be different per section (home_reco, home_trending, history, saved, etc.)
    so the same paper can appear multiple times without duplicate widget keys.
    """
    p = get_paper(paper_id)
    if not p:
        return

    title = p.get("title", "Unknown")
    authors = p.get("authors") or []
    year = p.get("year", "")
    abstract = p.get("abstract", "")
    topics = p.get("topics") or []

    with st.container(border=True):
        st.markdown(f"**{title}**")
        meta = []
        if authors:
            meta.append(", ".join(authors[:3]) + (" et al." if len(authors) > 3 else ""))
        if year:
            meta.append(str(year))
        if topics:
            meta.append(" • ".join(topics[:3]))
        if meta:
            st.caption(" | ".join(meta))

        if abstract:
            st.write(abstract[:220] + ("…" if len(abstract) > 220 else ""))

        if show_actions:
            c1, c2, c3, c4 = st.columns([2, 2, 2, 2])

            if c1.button("▶️ Play", key=ui_key(key_prefix, "play", paper_id), use_container_width=True):
                set_selected_paper(paper_id)
                st.session_state.now_playing_mode = "summary"
                st.rerun()

            if c2.button("📄 Open", key=ui_key(key_prefix, "open", paper_id), use_container_width=True):
                set_selected_paper(paper_id)
                st.rerun()

            saved_label = "⭐ Save" if paper_id not in st.session_state.saved else "✅ Saved"
            if c3.button(saved_label, key=ui_key(key_prefix, "save", paper_id), use_container_width=True):
                if paper_id in st.session_state.saved:
                    st.session_state.saved.remove(paper_id)
                else:
                    st.session_state.saved.add(paper_id)
                st.rerun()

            if c4.button("🧠 Ask", key=ui_key(key_prefix, "ask", paper_id), use_container_width=True):
                set_selected_paper(paper_id)
                st.session_state.now_playing_mode = "qa"
                st.session_state._jump_to_tab = "Chat"
                st.rerun()


# ----------------------------
# Tabs: Home / Paper / Chat / Library
# ----------------------------
st.title("🎙️ ResearchMix — Spotify for Research Papers")
st.caption("LLM-generated paper playlists now • arXiv/alphaXiv later • Voice Q&A + narrations")

render_now_playing()

tabs = st.tabs(["Home", "Paper", "Chat", "Library"])

if "_jump_to_tab" in st.session_state:
    st.info("Tip: Open the **Chat** tab to ask questions by voice or text about the selected paper.")
    del st.session_state._jump_to_tab


# ----------------------------
# HOME TAB
# ----------------------------
with tabs[0]:
    left, right = st.columns([1.15, 0.85], gap="large")

    with left:
        st.subheader("🏠 Home")

        need_trending = len(st.session_state.playlists.get("trending", [])) == 0
        need_reco = len(st.session_state.playlists.get("recommended", [])) == 0

        if need_trending:
            with st.spinner("Generating trending playlist…"):
                papers = llm_generate_papers(kind="trending", n=10, model=llm_model)
            if not papers:
                st.warning("Could not generate trending papers (LLM JSON parse). Try Regenerate playlists.")
            else:
                st.session_state.playlists["trending"] = [upsert_paper(p) for p in papers]

        if need_reco:
            with st.spinner("Generating recommended playlist…"):
                papers = llm_generate_papers(kind="recommended", n=10, model=llm_model)
            if not papers:
                st.warning("Could not generate recommended papers (LLM JSON parse). Try Regenerate playlists.")
            else:
                st.session_state.playlists["recommended"] = [upsert_paper(p) for p in papers]

        if st.session_state.last_played:
            st.markdown("### ⏯️ Continue")
            paper_card(st.session_state.last_played, key_prefix="home_continue")

        st.markdown("### 🎧 Recommended for you")
        reco_ids = st.session_state.playlists.get("recommended", [])[:8]
        if reco_ids:
            for pid in reco_ids:
                paper_card(pid, key_prefix="home_reco")
        else:
            st.info("No recommendations yet. Add interests in the sidebar and regenerate playlists.")

        st.markdown("### 🔥 Trending now")
        trend_ids = st.session_state.playlists.get("trending", [])[:8]
        if trend_ids:
            for pid in trend_ids:
                paper_card(pid, key_prefix="home_trending")
        else:
            st.info("No trending papers yet. Click Regenerate playlists in the sidebar.")

    with right:
        st.subheader("🔎 Search (LLM synth for now)")
        st.caption(
            "For the hackathon MVP, this search generates plausible papers (not arXiv). "
            "Later, swap this block with arXiv/alphaXiv."
        )

        query = st.text_input("Search query", placeholder="e.g., efficient agents for RAG on edge devices", key=ui_key("home", "search_query"))
        n_results = st.slider("Results", 3, 12, 6, 1, key=ui_key("home", "search_n"))

        if st.button(
            "Generate results",
            key=ui_key("home", "search_generate"),
            type="primary",
            use_container_width=True,
            disabled=not query.strip(),
        ):
            sys = (
                "You generate plausible research paper metadata for a hackathon demo.\n"
                "Output MUST be valid JSON only: an array of papers.\n"
                "Do not add markdown.\n"
            )
            user = (
                f"Generate {n_results} papers for the query: {query}\n"
                "Each item: {\"title\": str, \"authors\": [str], \"year\": int, \"abstract\": str, \"topics\": [str]}.\n"
                "Make the papers diverse but relevant.\n"
            )
            try:
                with st.spinner("Searching…"):
                    out = mistral_chat_raw(
                        messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
                        model=llm_model,
                        temperature=0.6,
                        max_tokens=1400,
                    )
                js = safe_json_extract(out)
                if not isinstance(js, list):
                    st.error("Model did not return valid JSON. Try again.")
                else:
                    ids = []
                    for item in js:
                        if not isinstance(item, dict):
                            continue
                        title = (item.get("title") or "").strip()
                        if not title:
                            continue
                        pid = upsert_paper(
                            {
                                "paper_id": f"llm:{uuid.uuid4().hex[:10]}",
                                "title": title,
                                "authors": item.get("authors") or [],
                                "year": item.get("year") or 2025,
                                "abstract": (item.get("abstract") or "").strip(),
                                "topics": item.get("topics") or [],
                                "source": "llm_search",
                            }
                        )
                        ids.append(pid)

                    if ids:
                        st.success(f"Generated {len(ids)} results ✅")
                        st.session_state.playlists["trending"] = ids + st.session_state.playlists.get("trending", [])
                        for pid in ids:
                            paper_card(pid, key_prefix="home_search_results")
                    else:
                        st.warning("No usable results returned. Try another query.")
            except Exception as e:
                st.error(f"Search failed: {e}")

        st.divider()
        st.subheader("📌 New user experience")
        st.write("- If you have no history, you’ll mostly see Trending.")
        st.write("- Add interests in sidebar to steer recommendations.")
        st.write("- Clicking any paper sets it as the selected paper for Paper/Chat tabs.")


# ----------------------------
# PAPER TAB
# ----------------------------
with tabs[1]:
    st.subheader("📄 Paper")

    pid = st.session_state.selected_paper or st.session_state.last_played
    if not pid:
        st.info("Select a paper from **Home** first.")
    else:
        p = get_paper(pid)
        v = get_view(pid)

        with st.container(border=True):
            st.markdown(f"## {p.get('title','Unknown')}")
            meta = []
            authors = p.get("authors") or []
            if authors:
                meta.append(", ".join(authors))
            if p.get("year"):
                meta.append(str(p.get("year")))
            topics = p.get("topics") or []
            if topics:
                meta.append(" • ".join(topics))
            if meta:
                st.caption(" | ".join(meta))

            if p.get("abstract"):
                st.write(p["abstract"])

            a1, a2, a3, a4 = st.columns([1, 1, 1, 1])
            if a1.button("▶️ Play summary", key=ui_key("paper", "play_summary", pid), use_container_width=True):
                st.session_state.last_played = pid
                st.session_state.now_playing_mode = "summary"
                st.rerun()
            if a2.button("🧠 Ask about this", key=ui_key("paper", "ask_about", pid), use_container_width=True):
                st.session_state.last_played = pid
                st.session_state.now_playing_mode = "qa"
                st.info("Open the **Chat** tab to ask questions.")
            if a3.button(
                "⭐ Save" if pid not in st.session_state.saved else "✅ Saved",
                key=ui_key("paper", "save", pid),
                use_container_width=True,
            ):
                if pid in st.session_state.saved:
                    st.session_state.saved.remove(pid)
                else:
                    st.session_state.saved.add(pid)
                st.rerun()
            if a4.button("🔄 Refresh Qs", key=ui_key("paper", "refresh_qs", pid), use_container_width=True):
                v["suggested_qs"] = None
                st.rerun()

        st.divider()
        colA, colB = st.columns([1.05, 0.95], gap="large")

        with colA:
            st.markdown("### ⚡ Quick Summary")
            if v["quick_summary"] is None:
                if st.button("Generate quick summary", key=ui_key("paper", "gen_quick", pid), type="primary", use_container_width=True):
                    try:
                        with st.spinner("Summarizing…"):
                            v["quick_summary"] = llm_quick_summary(p, llm_model)
                        st.success("Done ✅")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Summary failed: {e}")
            else:
                st.text_area("Quick summary", v["quick_summary"], height=220, key=ui_key("paper", "quick_text", pid))
                s1, s2 = st.columns([1, 1])
                if s1.button("🗣️ Narrate summary", key=ui_key("paper", "tts_quick", pid), use_container_width=True):
                    try:
                        speak_text = (v["quick_summary"] or "")[:max_chars_to_speak]
                        with st.spinner("Generating speech…"):
                            audio_out = elevenlabs_tts(speak_text, voice_id)
                        st.session_state.tts_last_audio = audio_out
                        st.audio(audio_out, format="audio/mpeg")
                        st.session_state.last_played = pid
                        st.session_state.now_playing_mode = "summary"
                    except Exception as e:
                        st.error(f"TTS failed: {e}")
                if s2.button("✂️ Regenerate", key=ui_key("paper", "regen_quick", pid), use_container_width=True):
                    v["quick_summary"] = None
                    st.rerun()

        with colB:
            st.markdown("### 🔎 Deep Dive")
            if v["deep_dive"] is None:
                if st.button("Generate deep dive", key=ui_key("paper", "gen_deep", pid), use_container_width=True):
                    try:
                        with st.spinner("Explaining…"):
                            v["deep_dive"] = llm_deep_dive(p, llm_model)
                        st.success("Done ✅")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Deep dive failed: {e}")
            else:
                st.text_area("Deep dive", v["deep_dive"], height=320, key=ui_key("paper", "deep_text", pid))
                d1, d2 = st.columns([1, 1])
                if d1.button("🗣️ Narrate deep dive", key=ui_key("paper", "tts_deep", pid), use_container_width=True):
                    try:
                        speak_text = (v["deep_dive"] or "")[:max_chars_to_speak]
                        with st.spinner("Generating speech…"):
                            audio_out = elevenlabs_tts(speak_text, voice_id)
                        st.session_state.tts_last_audio = audio_out
                        st.audio(audio_out, format="audio/mpeg")
                        st.session_state.last_played = pid
                        st.session_state.now_playing_mode = "deepdive"
                    except Exception as e:
                        st.error(f"TTS failed: {e}")
                if d2.button("✂️ Regenerate deep dive", key=ui_key("paper", "regen_deep", pid), use_container_width=True):
                    v["deep_dive"] = None
                    st.rerun()

        st.divider()
        st.markdown("### 💡 Suggested questions")
        if v["suggested_qs"] is None:
            if st.button("Generate suggested questions", key=ui_key("paper", "gen_qs", pid), use_container_width=True):
                try:
                    with st.spinner("Generating…"):
                        v["suggested_qs"] = llm_suggested_questions(p, llm_model)
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed: {e}")
        else:
            qs = v["suggested_qs"] or []
            if not qs:
                st.info("No suggested questions available.")
            else:
                for i, q in enumerate(qs, 1):
                    if st.button(f"{i}. {q}", key=ui_key("paper", "qs", pid, i), use_container_width=True):
                        st.session_state.chat_prefill = q
                        st.session_state.now_playing_mode = "qa"
                        st.info("Question queued. Open the **Chat** tab to run it.")


# ----------------------------
# CHAT TAB
# ----------------------------
with tabs[2]:
    st.subheader("💬 Chat (Voice + Text)")

    pid = st.session_state.selected_paper or st.session_state.last_played
    paper = get_paper(pid) if pid else None

    if paper:
        st.caption(f"Scoped to: **{paper.get('title','Unknown')}**")
    else:
        st.caption("No paper selected — general chat mode.")

    st.write("## 1) Speak (recommended)")
    st.write("Click to record, click again to stop. Then we transcribe → answer → (optional) speak back.")

    audio = audiorecorder("🎤 Record", "⏺️ Recording...")

    audio_bytes = None
    if len(audio) > 0:
        wav_io = audio.export(format="wav")
        audio_bytes = wav_io.read()
        st.audio(audio_bytes, format="audio/wav")
        st.success("Recording captured ✅")

    st.write("---")
    st.write("## 2) Or type (backup)")
    prefill = st.session_state.pop("chat_prefill", "") if "chat_prefill" in st.session_state else ""
    typed = st.text_area(
        "Type your question",
        value=prefill,
        placeholder="e.g., What is the main idea and why does it matter?",
        key=ui_key("chat", "typed"),
    )

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
    else:
        if mode == "voice":
            st.write("## 2) Transcribe (Voxtral)")
            try:
                with st.spinner("Transcribing..."):
                    transcript = voxtral_transcribe(audio_bytes, "recording.wav", stt_model)
                st.text_area("Transcript", transcript, height=120, key=ui_key("chat", "transcript"))
            except Exception as e:
                st.error(f"Transcription failed: {e}")
                st.stop()
            user_text = transcript
        else:
            user_text = typed.strip()

        st.write("## 3) Think (Mistral)")
        try:
            with st.spinner("Thinking..."):
                sys_prompt = paper_aware_system_prompt(paper)
                messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_text}]
                answer = mistral_chat_raw(messages, model=llm_model, temperature=0.2, max_tokens=900)
            st.text_area("Answer", answer, height=220, key=ui_key("chat", "answer"))
            if pid:
                st.session_state.last_played = pid
                st.session_state.now_playing_mode = "qa"
        except Exception as e:
            st.error(f"Mistral call failed: {e}")
            st.stop()

        st.write("## 4) Speak (ElevenLabs)")
        speak_text = answer[:max_chars_to_speak]
        try:
            with st.spinner("Generating speech..."):
                audio_out = elevenlabs_tts(speak_text, voice_id)
            st.session_state.tts_last_audio = audio_out
            st.audio(audio_out, format="audio/mpeg")
        except Exception as e:
            st.error(f"ElevenLabs TTS failed: {e}")
            st.info("You can still use transcript + answer; TTS is optional for the demo.")


# ----------------------------
# LIBRARY TAB
# ----------------------------
with tabs[3]:
    st.subheader("📚 Library")

    c1, c2 = st.columns([1, 1], gap="large")

    with c1:
        st.markdown("### 🕘 History")
        if not st.session_state.history:
            st.info("No history yet. Play a paper from Home.")
        else:
            for pid in st.session_state.history[:15]:
                paper_card(pid, key_prefix="lib_history")

    with c2:
        st.markdown("### ⭐ Saved")
        if not st.session_state.saved:
            st.info("No saved papers yet.")
        else:
            for pid in list(st.session_state.saved)[:15]:
                paper_card(pid, key_prefix="lib_saved")

    st.divider()
    st.markdown("### 🎶 Playlists")
    pl_cols = st.columns(2)

    with pl_cols[0]:
        st.markdown("**Recommended**")
        for pid in st.session_state.playlists.get("recommended", [])[:8]:
            p = get_paper(pid)
            if p:
                if st.button(p.get("title", "Untitled"), key=ui_key("lib", "pl_reco", pid), use_container_width=True):
                    set_selected_paper(pid)
                    st.rerun()

    with pl_cols[1]:
        st.markdown("**Trending**")
        for pid in st.session_state.playlists.get("trending", [])[:8]:
            p = get_paper(pid)
            if p:
                if st.button(p.get("title", "Untitled"), key=ui_key("lib", "pl_trend", pid), use_container_width=True):
                    set_selected_paper(pid)
                    st.rerun()

    st.caption("Hackathon MVP note: playlists/search are LLM-synth. Replace with arXiv/alphaXiv after the demo.")