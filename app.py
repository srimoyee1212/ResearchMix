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

# Dummy auth (for hackathon/demo)
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
# arXiv API (Atom feed parsing)
# ----------------------------
ARXIV_API = "http://export.arxiv.org/api/query"


def _strip(s: str) -> str:
    return (s or "").strip()


def _clean_ws(s: str) -> str:
    return re.sub(r"\s+", " ", _strip(s))


def arxiv_query(
    search_query: str,
    start: int = 0,
    max_results: int = 10,
    sortBy: str = "submittedDate",
    sortOrder: str = "descending",
) -> List[Dict[str, Any]]:
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
    ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

    entries = []
    for entry in root.findall("atom:entry", ns):
        entry_id = _strip(entry.findtext("atom:id", default="", namespaces=ns))
        title = _clean_ws(entry.findtext("atom:title", default="", namespaces=ns))
        summary = _clean_ws(entry.findtext("atom:summary", default="", namespaces=ns))
        published = _strip(entry.findtext("atom:published", default="", namespaces=ns))
        updated = _strip(entry.findtext("atom:updated", default="", namespaces=ns))

        authors = []
        for a in entry.findall("atom:author", ns):
            name = _strip(a.findtext("atom:name", default="", namespaces=ns))
            if name:
                authors.append(name)

        cats = []
        for c in entry.findall("atom:category", ns):
            term = c.attrib.get("term", "")
            if term:
                cats.append(term)

        paper_id = ""
        if entry_id:
            m = re.search(r"arxiv\.org/abs/([^/]+)$", entry_id)
            paper_id = f"arxiv:{m.group(1)}" if m else f"arxiv:{uuid.uuid4().hex[:10]}"

        year = None
        if published and len(published) >= 4 and published[:4].isdigit():
            year = int(published[:4])

        pdf_url = ""
        for link in entry.findall("atom:link", ns):
            if link.attrib.get("title") == "pdf":
                pdf_url = link.attrib.get("href", "")
                break

        entries.append(
            {
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
            }
        )

    return entries


def arxiv_search_free_text(query: str, max_results: int = 10) -> List[Dict[str, Any]]:
    q = _clean_ws(query)
    return arxiv_query(search_query=f"all:{q}", start=0, max_results=max_results)


def arxiv_trending_default(max_results: int = 12) -> List[Dict[str, Any]]:
    cat_query = "cat:cs.AI OR cat:cs.LG OR cat:cs.CL"
    return arxiv_query(search_query=cat_query, start=0, max_results=max_results)


# ----------------------------
# State: per-user buckets
# ----------------------------
def _ensure_global_state():
    if "auth" not in st.session_state:
        st.session_state.auth = {"logged_in": False, "username": None}
    if "users" not in st.session_state:
        st.session_state.users = {}  # username -> bucket


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


def _username() -> str:
    return st.session_state.auth.get("username") or "__anon__"


def uk(*parts: str) -> str:
    safe = [str(p) for p in parts if p is not None]
    return f"{_username()}::" + "::".join(safe)


# ----------------------------
# Dummy login gate
# ----------------------------
def render_login():
    st.title("🔐 ResearchMix Login")
    st.caption("Hackathon demo login (dummy auth).")

    with st.container(border=True):
        username = st.text_input("Username", placeholder="demo / judge / fairgame", key="login_user")
        password = st.text_input("Password", type="password", placeholder="demo / demo / researchmix", key="login_pass")
        c1, c2 = st.columns([1, 1])
        with c1:
            if st.button("Log in", type="primary", use_container_width=True, key="login_btn"):
                if username in DEMO_USERS and DEMO_USERS[username] == password:
                    st.session_state.auth["logged_in"] = True
                    st.session_state.auth["username"] = username
                    _u()
                    st.success("Logged in ✅")
                    st.rerun()
                else:
                    st.error("Invalid username/password.")
        with c2:
            if st.button("Use demo account", use_container_width=True, key="login_demo"):
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
# Personalization + prompts
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
    topics_str = ", ".join(topics) if topics else "unknown"

    return (
        base
        + "The user is currently focused on the following paper.\n"
        + f"TITLE: {title}\n"
        + f"TOPICS: {topics_str}\n"
        + (f"ABSTRACT: {abstract}\n" if abstract else "ABSTRACT: (not available; rely on title + user context)\n")
        + "Answer primarily about this paper; if asked generally, respond normally.\n"
    )


def build_recommended_arxiv_query() -> str:
    bucket = _u()
    interests = bucket["user_profile"].get("interests") or []

    hist_titles = []
    for pid in bucket["history"][:3]:
        p = get_paper(pid)
        if p and p.get("title"):
            hist_titles.append(p["title"])

    tokens = []
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
# LLM: summaries + deep dive + suggested Qs
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


def llm_suggested_questions(paper: Dict[str, Any], model: str) -> List[str]:
    sys = (
        "Generate suggested questions a user can ask about a research paper.\n"
        "Output MUST be JSON only: an array of 6 short questions.\n"
        "No markdown.\n"
    )
    user = (
        f"TITLE: {paper.get('title','')}\n"
        f"ABSTRACT:\n{paper.get('abstract','')}\n"
        f"Generate 6 questions."
    )
    out = mistral_chat_raw(
        messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
        model=model,
        temperature=0.5,
        max_tokens=350,
    )
    try:
        js = json.loads(out)
    except Exception:
        # simple fallback parse
        t = out.strip()
        if t.startswith("```"):
            parts = t.split("```")
            if len(parts) >= 2:
                t = parts[1].strip()
                if "\n" in t and t.split("\n", 1)[0].strip().lower() in ("json", "javascript"):
                    t = t.split("\n", 1)[1].strip()
        try:
            js = json.loads(t)
        except Exception:
            js = None

    if isinstance(js, list):
        qs = [q.strip() for q in js if isinstance(q, str) and q.strip()]
        return qs[:6]
    return []


# ----------------------------
# Navigation (CRITICAL FIX)
# ----------------------------
NAV_KEY = uk("nav")
NAV_NEXT_KEY = uk("nav_next")

# apply pending navigation BEFORE creating the radio widget
if NAV_NEXT_KEY in st.session_state:
    st.session_state[NAV_KEY] = st.session_state[NAV_NEXT_KEY]
    del st.session_state[NAV_NEXT_KEY]

if NAV_KEY not in st.session_state:
    st.session_state[NAV_KEY] = "Home"


def goto(page_name: str):
    """
    Schedule navigation safely (cannot set NAV_KEY after radio is instantiated).
    """
    st.session_state[NAV_NEXT_KEY] = page_name
    st.rerun()


# ----------------------------
# UI components
# ----------------------------
def render_top_bar():
    c1, c2, c3 = st.columns([6, 2, 2])
    c1.title("🎙️ ResearchMix")
    c1.caption("arXiv-powered discovery • clean judge-friendly flow • Voice Q&A + narrations")
    if c2.button("🚪 Log out", use_container_width=True, key=uk("auth", "logout")):
        st.session_state.auth = {"logged_in": False, "username": None}
        st.rerun()
    c3.write(f"👤 **{_username()}**")


def render_last_played_bar():
    bucket = _u()
    pid = bucket.get("last_played")
    if not pid:
        return
    p = get_paper(pid)
    if not p:
        return

    with st.container(border=True):
        st.markdown(f"**Last played:** {p.get('title','Unknown')} ({p.get('year','')})")
        links = []
        if p.get("url"):
            links.append(f"[arXiv]({p['url']})")
        if p.get("pdf_url"):
            links.append(f"[PDF]({p['pdf_url']})")
        if links:
            st.caption(" • ".join(links))

        if st.button("📄 Open", use_container_width=True, key=uk("last", "open", pid)):
            _u()["selected_paper"] = pid
            goto("Paper")


def paper_card_home(paper_id: str, section: str):
    """
    Home cards: minimal + ONE Open button that jumps to Paper.
    """
    bucket = _u()
    p = get_paper(paper_id)
    if not p:
        return

    with st.container(border=True):
        st.markdown(f"**{p.get('title','Unknown')}**")

        meta = []
        authors = p.get("authors") or []
        if authors:
            meta.append(", ".join(authors[:3]) + (" et al." if len(authors) > 3 else ""))
        if p.get("year"):
            meta.append(str(p["year"]))
        topics = p.get("topics") or []
        if topics:
            meta.append(" • ".join(topics[:3]))
        if p.get("source"):
            meta.append(p["source"])
        if meta:
            st.caption(" | ".join(meta))

        abstract = p.get("abstract") or ""
        if abstract:
            st.write(abstract[:220] + ("…" if len(abstract) > 220 else ""))

        links = []
        if p.get("url"):
            links.append(f"[arXiv]({p['url']})")
        if p.get("pdf_url"):
            links.append(f"[PDF]({p['pdf_url']})")
        if links:
            st.caption(" • ".join(links))

        if st.button("📄 Open", use_container_width=True, key=uk("home", section, "open", paper_id)):
            set_selected_paper(paper_id)
            goto("Paper")


# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    st.subheader("Models")
    stt_model = st.text_input("Voxtral STT model", value=DEFAULT_STT_MODEL, key=uk("cfg", "stt"))
    llm_model = st.text_input("Mistral LLM model", value=DEFAULT_LLM_MODEL, key=uk("cfg", "llm"))

    st.subheader("ElevenLabs")
    voice_id = st.text_input("ElevenLabs Voice ID", value=ELEVENLABS_VOICE_ID, key=uk("cfg", "voice"))
    st.caption("Premade example: EXAVITQu4vr4xnSDxMaL (Sarah)")

    st.subheader("Cost / Safety")
    max_chars_to_speak = st.slider("Max characters to speak", 200, 4000, 1200, 100, key=uk("cfg", "maxchars"))

    st.subheader("Personalization")
    bucket = _u()
    interests_text = st.text_input(
        "Interests (comma-separated)",
        value=", ".join(bucket["user_profile"].get("interests") or []),
        placeholder="e.g., agents, recsys, vision-language, eval",
        key=uk("profile", "interests"),
    )
    about_me = st.text_area(
        "About (optional)",
        value=bucket["user_profile"].get("about_me") or "",
        placeholder="e.g., I prefer practical papers I can implement quickly.",
        height=90,
        key=uk("profile", "about"),
    )
    if st.button("Save profile", use_container_width=True, key=uk("profile", "save")):
        interests = [x.strip() for x in interests_text.split(",") if x.strip()]
        bucket["user_profile"]["interests"] = interests
        bucket["user_profile"]["about_me"] = about_me.strip()
        st.success("Saved ✅")

    st.divider()
    if st.button("🔄 Refresh playlists", use_container_width=True, key=uk("pl", "refresh")):
        bucket["playlists"]["recommended"] = []
        bucket["playlists"]["trending"] = []
        st.success("Will refresh on Home.")


# ----------------------------
# Main UI
# ----------------------------
render_top_bar()

nav = st.radio(
    "Navigation",
    ["Home", "Paper", "Chat", "Library"],
    horizontal=True,
    key=NAV_KEY,
    label_visibility="collapsed",
)

render_last_played_bar()

# ----------------------------
# HOME
# ----------------------------
if nav == "Home":
    bucket = _u()
    left, right = st.columns([1.15, 0.85], gap="large")

    with left:
        st.subheader("🏠 Home")

        need_trending = len(bucket["playlists"].get("trending", [])) == 0
        need_reco = len(bucket["playlists"].get("recommended", [])) == 0

        if need_trending:
            with st.spinner("Fetching trending papers from arXiv…"):
                try:
                    papers = arxiv_trending_default(max_results=12)
                except Exception as e:
                    st.warning(f"arXiv trending failed: {e}")
                    papers = []
            bucket["playlists"]["trending"] = [upsert_paper(p) for p in papers]

        if need_reco:
            with st.spinner("Building recommended playlist…"):
                try:
                    q = build_recommended_arxiv_query()
                    papers = arxiv_query(q, start=0, max_results=12)
                except Exception as e:
                    st.warning(f"arXiv recommended failed: {e}")
                    papers = []
            bucket["playlists"]["recommended"] = [upsert_paper(p) for p in papers]

        st.markdown("### 🎧 Recommended for you")
        reco_ids = bucket["playlists"].get("recommended", [])[:8]
        if reco_ids:
            for pid in reco_ids:
                paper_card_home(pid, section="reco")
        else:
            st.info("No recommendations yet. Add interests in the sidebar and refresh playlists.")

        st.markdown("### 🔥 Trending now (latest arXiv)")
        trend_ids = bucket["playlists"].get("trending", [])[:8]
        if trend_ids:
            for pid in trend_ids:
                paper_card_home(pid, section="trend")
        else:
            st.info("No trending papers yet. Refresh playlists in the sidebar.")

    with right:
        st.subheader("🔎 Search (arXiv)")
        query = st.text_input("Search query", placeholder='e.g., "agentic rag evaluation"', key=uk("search", "q"))
        n_results = st.slider("Results", 3, 25, 8, 1, key=uk("search", "n"))

        if st.button("Search arXiv", type="primary", use_container_width=True, disabled=not query.strip(), key=uk("search", "go")):
            try:
                with st.spinner("Searching arXiv…"):
                    papers = arxiv_search_free_text(query.strip(), max_results=n_results)
                    time.sleep(0.2)
                if not papers:
                    st.warning("No results found.")
                else:
                    ids = [upsert_paper(p) for p in papers]
                    st.success(f"Found {len(ids)} papers ✅")
                    st.markdown("### Results")
                    for pid in ids:
                        paper_card_home(pid, section="search")
            except Exception as e:
                st.error(f"arXiv search failed: {e}")

# ----------------------------
# PAPER
# ----------------------------
elif nav == "Paper":
    bucket = _u()
    st.subheader("📄 Paper")

    pid = bucket.get("selected_paper") or bucket.get("last_played")
    if not pid:
        st.info("Open a paper from **Home** first.")
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
            if p.get("source"):
                meta.append(p.get("source"))
            if meta:
                st.caption(" | ".join(meta))

            if p.get("abstract"):
                st.write(p["abstract"])

            links = []
            if p.get("url"):
                links.append(f"[arXiv]({p['url']})")
            if p.get("pdf_url"):
                links.append(f"[PDF]({p['pdf_url']})")
            if links:
                st.caption(" • ".join(links))

            a1, a2, a3 = st.columns([1, 1, 1])
            if a1.button("⚡ Summary", use_container_width=True, key=uk("paper", pid, "mode_summary")):
                bucket["last_played"] = pid
                bucket["now_playing_mode"] = "summary"
                bump_history(pid)

            if a2.button("💬 Ask (Chat)", use_container_width=True, key=uk("paper", pid, "goto_chat")):
                bucket["last_played"] = pid
                bucket["now_playing_mode"] = "qa"
                bump_history(pid)
                goto("Chat")

            saved_label = "⭐ Save" if pid not in bucket["saved"] else "✅ Saved"
            if a3.button(saved_label, use_container_width=True, key=uk("paper", pid, "save_toggle")):
                if pid in bucket["saved"]:
                    bucket["saved"].remove(pid)
                else:
                    bucket["saved"].add(pid)
                st.rerun()

        st.divider()
        colA, colB = st.columns([1.05, 0.95], gap="large")

        with colA:
            st.markdown("### ⚡ Quick Summary")
            if v["quick_summary"] is None:
                if st.button("Generate summary", type="primary", use_container_width=True, key=uk("paper", pid, "gen_quick")):
                    try:
                        with st.spinner("Summarizing…"):
                            v["quick_summary"] = llm_quick_summary(p, llm_model)
                        st.success("Done ✅")
                    except Exception as e:
                        st.error(f"Summary failed: {e}")
            else:
                st.text_area("Summary", v["quick_summary"], height=240, key=uk("paper", pid, "quick_text"))
                if st.button("🗣️ Narrate summary", use_container_width=True, key=uk("paper", pid, "tts_quick")):
                    try:
                        speak_text = (v["quick_summary"] or "")[:max_chars_to_speak]
                        with st.spinner("Generating speech…"):
                            audio_out = elevenlabs_tts(speak_text, voice_id)
                        bucket["tts_last_audio"] = audio_out
                        st.audio(audio_out, format="audio/mpeg")
                    except Exception as e:
                        st.error(f"TTS failed: {e}")

        with colB:
            st.markdown("### 🔎 Deep Dive")
            if v["deep_dive"] is None:
                if st.button("Generate deep dive", use_container_width=True, key=uk("paper", pid, "gen_deep")):
                    try:
                        with st.spinner("Explaining…"):
                            v["deep_dive"] = llm_deep_dive(p, llm_model)
                        st.success("Done ✅")
                    except Exception as e:
                        st.error(f"Deep dive failed: {e}")
            else:
                st.text_area("Deep dive", v["deep_dive"], height=320, key=uk("paper", pid, "deep_text"))

        st.divider()
        st.markdown("### 💡 Suggested questions")
        if v["suggested_qs"] is None:
            if st.button("Generate suggested questions", use_container_width=True, key=uk("paper", pid, "gen_qs")):
                try:
                    with st.spinner("Generating…"):
                        v["suggested_qs"] = llm_suggested_questions(p, llm_model)
                except Exception as e:
                    st.error(f"Failed: {e}")
        else:
            qs = v["suggested_qs"] or []
            if not qs:
                st.info("No suggested questions available.")
            else:
                for i, q in enumerate(qs, 1):
                    if st.button(f"{i}. {q}", key=uk("paper", pid, "qs", str(i)), use_container_width=True):
                        st.session_state[uk("chat_prefill")] = q
                        goto("Chat")

# ----------------------------
# CHAT
# ----------------------------
elif nav == "Chat":
    bucket = _u()
    st.subheader("💬 Chat (Voice + Text)")

    pid = bucket.get("selected_paper") or bucket.get("last_played")
    paper = get_paper(pid) if pid else None

    if paper:
        st.caption(f"Scoped to: **{paper.get('title','Unknown')}**")
    else:
        st.caption("No paper selected — general chat mode.")

    st.write("## 1) Speak (recommended)")
    audio = audiorecorder("🎤 Record", "⏺️ Recording...", key=uk("chat", "recorder"))

    audio_bytes = None
    if len(audio) > 0:
        wav_io = audio.export(format="wav")
        audio_bytes = wav_io.read()
        st.audio(audio_bytes, format="audio/wav")
        st.success("Recording captured ✅")

    st.write("---")
    prefill_key = uk("chat_prefill")
    prefill = st.session_state.pop(prefill_key, "") if prefill_key in st.session_state else ""
    typed = st.text_area(
        "Type your question",
        value=prefill,
        placeholder="e.g., What is the core idea and why does it matter?",
        key=uk("chat", "typed"),
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
            try:
                with st.spinner("Transcribing (Voxtral)…"):
                    transcript = voxtral_transcribe(audio_bytes, "recording.wav", stt_model)
                st.text_area("Transcript", transcript, height=120, key=uk("chat", "transcript"))
                user_text = transcript
            except Exception as e:
                st.error(f"Transcription failed: {e}")
                st.stop()
        else:
            user_text = typed.strip()

        try:
            with st.spinner("Thinking (Mistral)…"):
                sys_prompt = paper_aware_system_prompt(paper)
                messages = [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_text},
                ]
                answer = mistral_chat_raw(messages, model=llm_model, temperature=0.2, max_tokens=900)
            st.text_area("Answer", answer, height=240, key=uk("chat", "answer"))
            if pid:
                bucket["last_played"] = pid
                bucket["now_playing_mode"] = "qa"
                bump_history(pid)
        except Exception as e:
            st.error(f"Mistral call failed: {e}")
            st.stop()

        speak_text = answer[:max_chars_to_speak]
        try:
            with st.spinner("Speaking (ElevenLabs)…"):
                audio_out = elevenlabs_tts(speak_text, voice_id)
            bucket["tts_last_audio"] = audio_out
            st.audio(audio_out, format="audio/mpeg")
        except Exception as e:
            st.error(f"ElevenLabs TTS failed: {e}")
            st.info("You can still use transcript + answer; TTS is optional for the demo.")

# ----------------------------
# LIBRARY
# ----------------------------
else:  # Library
    bucket = _u()
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
                    if st.button("📄 Open", use_container_width=True, key=uk("lib", "hist", "open", pid)):
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
                    if st.button("📄 Open", use_container_width=True, key=uk("lib", "saved", "open", pid)):
                        set_selected_paper(pid)
                        goto("Paper")