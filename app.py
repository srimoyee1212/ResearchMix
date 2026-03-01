# app.py
import os
import re
import json
import uuid
import time
import requests
import streamlit as st
import xml.etree.ElementTree as ET
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

# Dummy auth (hackathon/demo)
# You can override via .env:
# DEMO_USERS_JSON='{"srimoyee":"pass123","judge":"demo"}'
DEMO_USERS_JSON = os.getenv("DEMO_USERS_JSON", "")
DEFAULT_DEMO_USERS = {"demo": "demo", "judge": "demo", "fairgame": "researchmix"}
try:
    DEMO_USERS = json.loads(DEMO_USERS_JSON) if DEMO_USERS_JSON.strip() else DEFAULT_DEMO_USERS
except Exception:
    DEMO_USERS = DEFAULT_DEMO_USERS

st.set_page_config(page_title="ResearchMix", page_icon="🎙️", layout="wide")

# ----------------------------
# Hard requirements
# ----------------------------
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
# Small text utils
# ----------------------------
def _strip(s: str) -> str:
    return (s or "").strip()

def _clean_ws(s: str) -> str:
    return re.sub(r"\s+", " ", _strip(s))

def first_sentence(text: str, max_len: int = 180) -> str:
    """
    Returns the first sentence-ish chunk for card previews.
    """
    t = _clean_ws(text or "")
    if not t:
        return ""
    # sentence boundary: . ! ? (fallback to max_len)
    m = re.search(r"^(.{20,}?[.!?])\s", t)
    if m:
        s = m.group(1).strip()
    else:
        s = t[:max_len].strip()
    if len(s) > max_len:
        s = s[:max_len].rstrip() + "…"
    return s


# ----------------------------
# arXiv API (Atom feed parsing) — cached
# ----------------------------
ARXIV_API = "http://export.arxiv.org/api/query"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

@st.cache_data(ttl=3600, show_spinner=False)
def arxiv_query(search_query: str, start: int = 0, max_results: int = 10,
               sortBy: str = "submittedDate", sortOrder: str = "descending") -> List[Dict[str, Any]]:
    """
    Cached arXiv query.
    search_query examples:
      - all:transformer
      - ti:"vision language" AND cat:cs.CV
      - cat:cs.AI OR cat:cs.LG
    """
    params = {
        "search_query": search_query,
        "start": start,
        "max_results": max_results,
        "sortBy": sortBy,
        "sortOrder": sortOrder,
    }
    r = requests.get(ARXIV_API, params=params, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"arXiv API error {r.status_code}: {r.text[:400]}")

    root = ET.fromstring(r.text)
    entries: List[Dict[str, Any]] = []

    for entry in root.findall("atom:entry", ATOM_NS):
        entry_id = _strip(entry.findtext("atom:id", default="", namespaces=ATOM_NS))
        title = _clean_ws(entry.findtext("atom:title", default="", namespaces=ATOM_NS))
        summary = _clean_ws(entry.findtext("atom:summary", default="", namespaces=ATOM_NS))
        published = _strip(entry.findtext("atom:published", default="", namespaces=ATOM_NS))
        updated = _strip(entry.findtext("atom:updated", default="", namespaces=ATOM_NS))

        # authors
        authors: List[str] = []
        for a in entry.findall("atom:author", ATOM_NS):
            name = _strip(a.findtext("atom:name", default="", namespaces=ATOM_NS))
            if name:
                authors.append(name)

        # categories (topics)
        cats: List[str] = []
        for c in entry.findall("atom:category", ATOM_NS):
            term = c.attrib.get("term", "")
            if term:
                cats.append(term)

        # derive arXiv id
        paper_id = ""
        if entry_id:
            m = re.search(r"arxiv\.org/abs/([^/]+)$", entry_id)
            paper_id = f"arxiv:{m.group(1)}" if m else f"arxiv:{uuid.uuid4().hex[:10]}"

        year = None
        if published and len(published) >= 4 and published[:4].isdigit():
            year = int(published[:4])

        # PDF link
        pdf_url = ""
        for link in entry.findall("atom:link", ATOM_NS):
            if link.attrib.get("title") == "pdf":
                pdf_url = link.attrib.get("href", "")
                break

        entries.append({
            "paper_id": paper_id,
            "title": title or "Untitled (arXiv)",
            "authors": authors,
            "year": year or 2025,
            "abstract": summary,
            "topics": cats,
            "source": "arxiv",
            "arxiv_id": paper_id.replace("arxiv:", ""),
            "url": entry_id,
            "pdf_url": pdf_url,
            "published": published,
            "updated": updated,
        })

    return entries


def arxiv_search_free_text(query: str, max_results: int = 10) -> List[Dict[str, Any]]:
    q = _clean_ws(query)
    return arxiv_query(search_query=f"all:{q}", start=0, max_results=max_results)


def arxiv_trending_default(max_results: int = 12) -> List[Dict[str, Any]]:
    cat_query = "cat:cs.AI OR cat:cs.LG OR cat:cs.CL OR cat:cs.CV"
    return arxiv_query(search_query=cat_query, start=0, max_results=max_results,
                       sortBy="submittedDate", sortOrder="descending")


# ----------------------------
# State: global + per-user buckets
# ----------------------------
def _ensure_global_state():
    if "auth" not in st.session_state:
        st.session_state.auth = {"logged_in": False, "username": None}
    if "users" not in st.session_state:
        st.session_state.users = {}
    if "nav" not in st.session_state:
        st.session_state.nav = "Home"  # Home | Paper | Chat | Library
    if "nav_target" not in st.session_state:
        st.session_state.nav_target = None
    if "chat_prefill" not in st.session_state:
        st.session_state.chat_prefill = ""

_ensure_global_state()


def _default_user_bucket() -> Dict[str, Any]:
    return {
        "paper_cache": {},
        "paper_views": {},
        "history": [],
        "last_played": None,
        "selected_paper": None,
        "saved": set(),
        "playlists": {"recommended": [], "trending": []},
        "user_profile": {"interests": [], "about_me": ""},
        "now_playing_mode": "summary",
        "tts_last_audio": None,
    }


def _u() -> Dict[str, Any]:
    username = st.session_state.auth.get("username") or "__anon__"
    if username not in st.session_state.users:
        st.session_state.users[username] = _default_user_bucket()
    return st.session_state.users[username]


# ----------------------------
# Dummy login gate
# ----------------------------
def render_login():
    st.title("🔐 ResearchMix Login")
    st.caption("Hackathon demo login (dummy auth).")

    with st.container(border=True):
        username = st.text_input("Username", placeholder="demo / judge / fairgame")
        password = st.text_input("Password", type="password", placeholder="demo / demo / researchmix")
        c1, c2 = st.columns([1, 1])
        with c1:
            if st.button("Log in", type="primary", use_container_width=True):
                if username in DEMO_USERS and DEMO_USERS[username] == password:
                    st.session_state.auth["logged_in"] = True
                    st.session_state.auth["username"] = username
                    _u()
                    st.success("Logged in ✅")
                    st.rerun()
                else:
                    st.error("Invalid username/password.")
        with c2:
            if st.button("Use demo account", use_container_width=True):
                st.session_state.auth["logged_in"] = True
                st.session_state.auth["username"] = "demo"
                _u()
                st.success("Logged in as demo ✅")
                st.rerun()

    st.info("Tip for judges: use **judge / demo** or click **Use demo account**.")


if not st.session_state.auth.get("logged_in"):
    render_login()
    st.stop()


# ----------------------------
# Navigation helpers (replaces tabs so we can jump pages)
# ----------------------------
NAV_PAGES = ["Home", "Paper", "Chat", "Library"]

def goto(page: str):
    if page in NAV_PAGES:
        st.session_state.nav_target = page
        st.rerun()

# Apply pending nav BEFORE the widget is created (prevents session_state key errors)
if st.session_state.nav_target in NAV_PAGES:
    st.session_state.nav = st.session_state.nav_target
    st.session_state.nav_target = None


# ----------------------------
# User-level helpers
# ----------------------------
def upsert_paper(paper: Dict[str, Any]) -> str:
    bucket = _u()
    paper_id = paper.get("paper_id") or f"llm:{uuid.uuid4().hex[:10]}"
    paper["paper_id"] = paper_id
    bucket["paper_cache"][paper_id] = paper
    return paper_id


def get_paper(paper_id: str) -> Optional[Dict[str, Any]]:
    return _u()["paper_cache"].get(paper_id)


def get_view(paper_id: str) -> Dict[str, Any]:
    bucket = _u()
    if paper_id not in bucket["paper_views"]:
        bucket["paper_views"][paper_id] = {"quick_summary": None, "deep_dive": None, "suggested_qs": None}
    return bucket["paper_views"][paper_id]


def bump_history(paper_id: str):
    bucket = _u()
    hist = bucket["history"]
    if paper_id in hist:
        hist.remove(paper_id)
    hist.insert(0, paper_id)
    bucket["history"] = hist[:50]


def set_selected_paper(paper_id: str):
    bucket = _u()
    if paper_id not in bucket["paper_cache"]:
        return
    bucket["selected_paper"] = paper_id
    bucket["last_played"] = paper_id
    bump_history(paper_id)


# ----------------------------
# Prompts (paper-aware)
# ----------------------------
def paper_aware_system_prompt(paper: Optional[Dict[str, Any]]) -> str:
    base = (
        "You are ResearchMix, a reliable research assistant. "
        "Be concise, structured, and avoid hallucinating. "
        "If unsure, say what's missing and suggest the best next step. "
        "Prefer bullet points. "
    )
    if not paper:
        return base + "The user may ask about any research topic."

    title = paper.get("title", "Unknown title")
    abstract = paper.get("abstract", "")
    topics = paper.get("topics", [])
    topics_str = ", ".join(topics[:8]) if topics else "unknown"

    return (
        base
        + "The user is currently focused on the following paper.\n"
        + f"TITLE: {title}\n"
        + f"TOPICS: {topics_str}\n"
        + (f"ABSTRACT: {abstract}\n" if abstract else "ABSTRACT: (not available; rely on title + user context)\n")
        + "Answer primarily about this paper; if asked generally, respond normally.\n"
    )


# ----------------------------
# LLM summarizers / explainers
# ----------------------------
def llm_quick_summary(paper: dict, model: str) -> str:
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


def llm_deep_dive(paper: dict, model: str) -> str:
    sys = (
        "You are a research explainer.\n"
        "Give a structured deep dive suitable for spoken narration and Q&A.\n"
        "Use headings and concise bullets.\n"
        "Include:\n"
        "1) Problem + motivation\n"
        "2) Core idea (plain English)\n"
        "3) Method outline\n"
        "4) What the experiments likely look like (high level)\n"
        "5) Limitations + open questions\n"
        "Avoid making up dataset names or exact metrics.\n"
    )
    user = (
        f"TITLE: {paper.get('title','')}\n"
        f"ABSTRACT:\n{paper.get('abstract','')}\n"
    )
    return mistral_chat_raw(
        messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
        model=model,
        temperature=0.25,
        max_tokens=1100,
    )


# ----------------------------
# UI components
# ----------------------------
def render_top_bar():
    username = st.session_state.auth.get("username")
    c1, c2, c3 = st.columns([6, 2, 2])
    c1.title("🎙️ ResearchMix")
    c1.caption("arXiv-powered discovery • One-click summaries • Voice Q&A")
    if c2.button("🚪 Log out", use_container_width=True):
        st.session_state.auth = {"logged_in": False, "username": None}
        st.rerun()
    c3.write(f"👤 **{username}**")


def render_now_playing_compact():
    bucket = _u()
    pid = bucket.get("last_played")
    if not pid:
        return
    p = get_paper(pid)
    if not p:
        return

    title = p.get("title", "Unknown")
    year = p.get("year", "")
    source = p.get("source", "")

    with st.container(border=True):
        c1, c2, c3 = st.columns([7, 1.5, 1.5])
        c1.markdown(f"**Last played:** {title} ({year})" + (f" · `{source}`" if source else ""))
        if c2.button("Open", key="np_open", use_container_width=True):
            bucket["selected_paper"] = pid
            goto("Paper")
        if c3.button("Speak last", key="np_speak", use_container_width=True, disabled=bucket.get("tts_last_audio") is None):
            if bucket.get("tts_last_audio"):
                st.audio(bucket["tts_last_audio"], format="audio/mpeg")


def paper_card_home(paper_id: str, section: str):
    """
    Super simple Home card:
    - title + meta
    - 1 sentence abstract
    - Open (+ PDF if available)
    """
    bucket = _u()
    p = get_paper(paper_id)
    if not p:
        return

    title = p.get("title", "Untitled")
    authors = p.get("authors") or []
    year = p.get("year", "")
    topics = p.get("topics") or []
    abstract = p.get("abstract", "")
    pdf_url = p.get("pdf_url", "")
    url = p.get("url", "")

    meta_parts = []
    if authors:
        meta_parts.append(", ".join(authors[:2]) + (" et al." if len(authors) > 2 else ""))
    if year:
        meta_parts.append(str(year))
    if topics:
        meta_parts.append(" • ".join(topics[:2]))

    with st.container(border=True):
        st.markdown(f"**{title}**")
        if meta_parts:
            st.caption(" | ".join(meta_parts))

        preview = first_sentence(abstract, max_len=180)
        if preview:
            st.write(preview)

        b1, b2 = st.columns([1, 1])
        if b1.button("Open", key=f"{section}:open:{paper_id}", use_container_width=True):
            set_selected_paper(paper_id)
            st.toast("Opened in Paper", icon="📄")
            goto("Paper")

        # Optional link button (Streamlit has st.link_button in newer versions)
        if pdf_url:
            # safest cross-version approach: markdown link
            b2.markdown(f"[PDF]({pdf_url})")
        elif url:
            b2.markdown(f"[arXiv]({url})")


# ----------------------------
# Sidebar controls + NAV (this replaces tabs)
# ----------------------------
with st.sidebar:
    st.header("🧭 Navigate")
    nav = st.radio("Go to", NAV_PAGES, key="nav", label_visibility="collapsed")

    st.divider()
    st.header("⚙️ Settings")

    st.subheader("Models")
    stt_model = st.text_input("Voxtral STT model", value=DEFAULT_STT_MODEL)
    llm_model = st.text_input("Mistral LLM model", value=DEFAULT_LLM_MODEL)

    st.subheader("ElevenLabs")
    voice_id = st.text_input("ElevenLabs Voice ID", value=ELEVENLABS_VOICE_ID)

    st.subheader("Cost / Safety")
    max_chars_to_speak = st.slider("Max characters to speak", 200, 4000, 1200, 100)

    st.divider()
    st.subheader("Personalization (optional)")
    bucket = _u()
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
# Recommend query builder (simple)
# ----------------------------
def build_recommended_arxiv_query() -> str:
    bucket = _u()
    interests = bucket["user_profile"].get("interests") or []
    hist_titles = []
    for pid in bucket["history"][:3]:
        p = get_paper(pid)
        if p and p.get("title"):
            hist_titles.append(p["title"])

    tokens: List[str] = []
    tokens.extend(interests[:5])
    for t in hist_titles:
        for w in re.findall(r"[A-Za-z][A-Za-z\-]{2,}", t.lower()):
            if w not in ("the", "and", "for", "with", "from", "via", "using", "toward", "towards"):
                tokens.append(w)

    tokens = [t for t in tokens if t][:8]
    if not tokens:
        return "cat:cs.AI OR cat:cs.LG OR cat:cs.CL OR cat:cs.CV"

    clauses = [f"all:{tok}" for tok in tokens[:4]]
    cat = "(cat:cs.AI OR cat:cs.LG OR cat:cs.CL OR cat:cs.CV)"
    return f"({ ' AND '.join(clauses) }) AND {cat}"


# ----------------------------
# Main
# ----------------------------
render_top_bar()
render_now_playing_compact()

bucket = _u()

# ----------------------------
# HOME
# ----------------------------
if nav == "Home":
    st.subheader("🏠 Home")

    # Load playlists if empty (cached arXiv makes this fast)
    if len(bucket["playlists"].get("trending", [])) == 0:
        with st.spinner("Fetching trending from arXiv…"):
            papers = arxiv_trending_default(max_results=10)
            time.sleep(0.15)  # light etiquette
        bucket["playlists"]["trending"] = [upsert_paper(p) for p in papers]

    if len(bucket["playlists"].get("recommended", [])) == 0:
        with st.spinner("Building recommended playlist…"):
            q = build_recommended_arxiv_query()
            papers = arxiv_query(q, start=0, max_results=10)
            time.sleep(0.15)
        bucket["playlists"]["recommended"] = [upsert_paper(p) for p in papers]

    left, right = st.columns([1.15, 0.85], gap="large")

    with left:
        if bucket.get("last_played"):
            st.markdown("### ⏯️ Continue")
            paper_card_home(bucket["last_played"], section="continue")

        # If they have history, recommended feels personal
        if bucket["history"]:
            last_title = (get_paper(bucket["history"][0]) or {}).get("title", "")
            if last_title:
                st.caption(f"Because you listened to: **{last_title}**")

        st.markdown("### 🎧 Recommended for you")
        for pid in bucket["playlists"].get("recommended", [])[:8]:
            paper_card_home(pid, section="reco")

        st.markdown("### 🔥 Trending now")
        for pid in bucket["playlists"].get("trending", [])[:8]:
            paper_card_home(pid, section="trend")

    with right:
        st.subheader("🔎 Search arXiv")
        query = st.text_input("Search query", placeholder='e.g., "agentic RAG evaluation"')
        n_results = st.slider("Results", 3, 25, 8, 1)

        if st.button("Search", type="primary", use_container_width=True, disabled=not query.strip()):
            try:
                with st.spinner("Searching arXiv…"):
                    papers = arxiv_search_free_text(query.strip(), max_results=n_results)
                    time.sleep(0.15)
                if not papers:
                    st.warning("No results found.")
                else:
                    ids = [upsert_paper(p) for p in papers]
                    st.success(f"Found {len(ids)} papers ✅")
                    for pid in ids:
                        paper_card_home(pid, section="search")
            except Exception as e:
                st.error(f"Search failed: {e}")

        st.divider()
        st.caption("Tip for judges: click **Open** on any paper → Paper page.")


# ----------------------------
# PAPER
# ----------------------------
elif nav == "Paper":
    st.subheader("📄 Paper")

    pid = bucket.get("selected_paper") or bucket.get("last_played")
    if not pid:
        st.info("Open a paper from **Home** first.")
    else:
        p = get_paper(pid)
        v = get_view(pid)

        with st.container(border=True):
            st.markdown(f"## {p.get('title','Untitled')}")
            meta = []
            if p.get("authors"):
                meta.append(", ".join(p["authors"]))
            if p.get("year"):
                meta.append(str(p.get("year")))
            if p.get("topics"):
                meta.append(" • ".join((p.get("topics") or [])[:6]))
            if p.get("source"):
                meta.append(p.get("source"))
            if meta:
                st.caption(" | ".join(meta))

            if p.get("url") or p.get("pdf_url"):
                links = []
                if p.get("url"):
                    links.append(f"[arXiv]({p['url']})")
                if p.get("pdf_url"):
                    links.append(f"[PDF]({p['pdf_url']})")
                st.caption(" • ".join(links))

            if p.get("abstract"):
                st.write(p["abstract"])

        st.markdown("### ▶️ One-click demo")
        c1, c2, c3 = st.columns([1.4, 1, 1])
        if c1.button("▶ Generate + Narrate Summary", type="primary", use_container_width=True):
            try:
                with st.spinner("Generating summary…"):
                    if v["quick_summary"] is None:
                        v["quick_summary"] = llm_quick_summary(p, llm_model)

                speak_text = (v["quick_summary"] or "")[:max_chars_to_speak]
                with st.spinner("Generating speech…"):
                    audio_out = elevenlabs_tts(speak_text, voice_id)

                bucket["tts_last_audio"] = audio_out
                bucket["last_played"] = pid
                bucket["now_playing_mode"] = "summary"
                bump_history(pid)

                st.success("Ready ✅")
                st.audio(audio_out, format="audio/mpeg")
            except Exception as e:
                st.error(f"Failed: {e}")

        if c2.button("💬 Ask about this paper", use_container_width=True):
            bucket["last_played"] = pid
            bucket["selected_paper"] = pid
            bump_history(pid)
            st.session_state.chat_prefill = "Give me the TL;DR and the main contributions."
            goto("Chat")

        if c3.button("🧠 Generate Deep Dive", use_container_width=True):
            try:
                with st.spinner("Generating deep dive…"):
                    v["deep_dive"] = llm_deep_dive(p, llm_model)
                st.success("Done ✅")
            except Exception as e:
                st.error(f"Deep dive failed: {e}")

        st.divider()

        # Show summary/deep dive if present
        colA, colB = st.columns([1, 1], gap="large")
        with colA:
            st.markdown("### ⚡ Quick Summary")
            if v["quick_summary"]:
                st.text_area("Summary", v["quick_summary"], height=240)
        with colB:
            st.markdown("### 🔎 Deep Dive")
            if v["deep_dive"]:
                st.text_area("Deep dive", v["deep_dive"], height=240)


# ----------------------------
# CHAT
# ----------------------------
elif nav == "Chat":
    st.subheader("💬 Chat (Voice + Text)")

    pid = bucket.get("selected_paper") or bucket.get("last_played")
    paper = get_paper(pid) if pid else None

    if paper:
        st.caption(f"Scoped to: **{paper.get('title','Untitled')}**")
    else:
        st.caption("No paper selected — general chat mode.")

    st.write("## 1) Speak")
    audio = audiorecorder("🎤 Record", "⏺️ Recording...")
    audio_bytes = None
    if len(audio) > 0:
        wav_io = audio.export(format="wav")
        audio_bytes = wav_io.read()
        st.audio(audio_bytes, format="audio/wav")
        st.success("Recording captured ✅")

    st.write("---")
    st.write("## 2) Or type")

    prefill = st.session_state.chat_prefill or ""
    typed = st.text_area(
        "Type your question",
        value=prefill,
        placeholder="e.g., What is the core idea, and what are the limitations?",
    )

    # Clear prefill after displaying once
    st.session_state.chat_prefill = ""

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
            st.write("## 3) Transcribe (Voxtral)")
            try:
                with st.spinner("Transcribing…"):
                    transcript = voxtral_transcribe(audio_bytes, "recording.wav", stt_model)
                st.text_area("Transcript", transcript, height=110)
            except Exception as e:
                st.error(f"Transcription failed: {e}")
                st.stop()
            user_text = transcript
        else:
            user_text = typed.strip()

        st.write("## 4) Think (Mistral)")
        try:
            with st.spinner("Thinking…"):
                sys_prompt = paper_aware_system_prompt(paper)
                messages = [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_text},
                ]
                answer = mistral_chat_raw(messages, model=llm_model, temperature=0.2, max_tokens=900)

            st.text_area("Answer", answer, height=220)

            if pid:
                bucket["last_played"] = pid
                bucket["now_playing_mode"] = "qa"
                bump_history(pid)

        except Exception as e:
            st.error(f"Mistral call failed: {e}")
            st.stop()

        st.write("## 5) Speak (ElevenLabs)")
        speak_text = answer[:max_chars_to_speak]
        try:
            with st.spinner("Generating speech…"):
                audio_out = elevenlabs_tts(speak_text, voice_id)
            bucket["tts_last_audio"] = audio_out
            st.audio(audio_out, format="audio/mpeg")
        except Exception as e:
            st.error(f"ElevenLabs TTS failed: {e}")
            st.info("You can still use transcript + answer; TTS is optional for the demo.")


# ----------------------------
# LIBRARY
# ----------------------------
else:
    st.subheader("📚 Library")

    c1, c2 = st.columns([1, 1], gap="large")

    with c1:
        st.markdown("### 🕘 History")
        if not bucket["history"]:
            st.info("No history yet. Open a paper from Home.")
        else:
            for pid in bucket["history"][:15]:
                p = get_paper(pid)
                if not p:
                    continue
                with st.container(border=True):
                    st.markdown(f"**{p.get('title','Untitled')}**")
                    if st.button("Open", key=f"lib_hist_open:{pid}", use_container_width=True):
                        set_selected_paper(pid)
                        goto("Paper")

    with c2:
        st.markdown("### ⭐ Saved")
        if not bucket["saved"]:
            st.info("No saved papers yet.")
        else:
            for pid in list(bucket["saved"])[:15]:
                p = get_paper(pid)
                if not p:
                    continue
                with st.container(border=True):
                    st.markdown(f"**{p.get('title','Untitled')}**")
                    if st.button("Open", key=f"lib_saved_open:{pid}", use_container_width=True):
                        set_selected_paper(pid)
                        goto("Paper")