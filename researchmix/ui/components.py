# researchmix/ui/components.py
import streamlit as st

from researchmix.state import u_bucket, get_paper, set_selected_paper, goto
from researchmix.text_utils import first_sentence


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
    bucket = u_bucket()
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
        if c3.button(
            "Speak last",
            key="np_speak",
            use_container_width=True,
            disabled=bucket.get("tts_last_audio") is None,
        ):
            if bucket.get("tts_last_audio"):
                st.audio(bucket["tts_last_audio"], format="audio/mpeg")


def paper_card_home(paper_id: str, section: str):
    """
    Super simple Home card:
    - title + meta
    - 1 sentence abstract
    - Open (+ PDF if available)
    """
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

        if pdf_url:
            b2.markdown(f"[PDF]({pdf_url})")
        elif url:
            b2.markdown(f"[arXiv]({url})")