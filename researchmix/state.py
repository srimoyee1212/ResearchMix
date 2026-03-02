# researchmix/state.py
import re
import uuid
import streamlit as st
from typing import Dict, Any, Optional, List

from researchmix.config import get_env

NAV_PAGES = ["Home", "Paper", "Chat", "Library"]


def ensure_global_state():
    if "auth" not in st.session_state:
        st.session_state.auth = {"logged_in": False, "username": None}
    if "users" not in st.session_state:
        st.session_state.users = {}
    if "nav" not in st.session_state:
        st.session_state.nav = "Home"
    if "nav_target" not in st.session_state:
        st.session_state.nav_target = None
    if "chat_prefill" not in st.session_state:
        st.session_state.chat_prefill = ""


def _default_user_bucket() -> Dict[str, Any]:
    env = get_env()
    return {
        "paper_cache": {},
        "paper_views": {},
        "history": [],
        "last_played": None,
        "selected_paper": None,
        "saved": set(),
        "playlists": {"recommended": [], "trending": []},

        # ✅ NEW: persist search results across reruns so Open works
        "search_results": [],

        "user_profile": {"interests": [], "about_me": ""},
        "now_playing_mode": "summary",
        "tts_last_audio": None,

        # keep settings per-user (matches your behavior; easy for demo)
        "settings": {
            "llm_model": env["DEFAULT_LLM_MODEL"],
            "stt_model": env["DEFAULT_STT_MODEL"],
            "voice_id": env["ELEVENLABS_VOICE_ID"],
            "max_chars_to_speak": 1200,
        },
    }


def u_bucket() -> Dict[str, Any]:
    username = st.session_state.auth.get("username") or "__anon__"
    if username not in st.session_state.users:
        st.session_state.users[username] = _default_user_bucket()
    return st.session_state.users[username]


def goto(page: str):
    if page in NAV_PAGES:
        st.session_state.nav_target = page
        st.rerun()


def apply_pending_nav():
    """
    Apply pending nav BEFORE the nav radio widget is created in app.py.
    This prevents: StreamlitAPIException cannot modify key after widget instantiation.
    """
    if st.session_state.nav_target in NAV_PAGES:
        st.session_state.nav = st.session_state.nav_target
        st.session_state.nav_target = None


# ----------------------------
# Paper store helpers
# ----------------------------
def upsert_paper(paper: Dict[str, Any]) -> str:
    bucket = u_bucket()
    paper_id = paper.get("paper_id") or f"llm:{uuid.uuid4().hex[:10]}"
    paper["paper_id"] = paper_id
    bucket["paper_cache"][paper_id] = paper
    return paper_id


def get_paper(paper_id: str) -> Optional[Dict[str, Any]]:
    return u_bucket()["paper_cache"].get(paper_id)


def get_view(paper_id: str) -> Dict[str, Any]:
    bucket = u_bucket()
    if paper_id not in bucket["paper_views"]:
        bucket["paper_views"][paper_id] = {"quick_summary": None, "deep_dive": None}
    return bucket["paper_views"][paper_id]


def bump_history(paper_id: str):
    bucket = u_bucket()
    hist = bucket["history"]
    if paper_id in hist:
        hist.remove(paper_id)
    hist.insert(0, paper_id)
    bucket["history"] = hist[:50]


def set_selected_paper(paper_id: str):
    bucket = u_bucket()
    if paper_id not in bucket["paper_cache"]:
        return
    bucket["selected_paper"] = paper_id
    bucket["last_played"] = paper_id
    bump_history(paper_id)


# ----------------------------
# Recommended query builder (same logic)
# ----------------------------
def build_recommended_arxiv_query() -> str:
    bucket = u_bucket()
    interests = bucket["user_profile"].get("interests") or []
    hist_titles: List[str] = []
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