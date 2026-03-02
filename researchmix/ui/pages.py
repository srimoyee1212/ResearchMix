# researchmix/ui/pages.py
import time
import streamlit as st
from audiorecorder import audiorecorder
from researchmix.ui.graph import render_related_paper_graph


from researchmix.state import (
    u_bucket,
    upsert_paper,
    get_paper,
    get_view,
    bump_history,
    set_selected_paper,
    goto,
    build_recommended_arxiv_query,
)
from researchmix.arxiv_client import arxiv_trending_default, arxiv_query, arxiv_search_free_text
from researchmix.llm_client import (
    mistral_chat_raw,
    paper_aware_system_prompt,
    llm_quick_summary,
    llm_deep_dive,
)
from researchmix.stt_client import voxtral_transcribe
from researchmix.tts_client import elevenlabs_tts
from researchmix.ui.components import paper_card_home


def render_home():
    bucket = u_bucket()
    st.subheader("🏠 Home")

    # Load playlists if empty (cached arXiv makes this fast)
    if len(bucket["playlists"].get("trending", [])) == 0:
        with st.spinner("Fetching trending from arXiv…"):
            papers = arxiv_trending_default(max_results=10)
            time.sleep(0.15)
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

        # ✅ Perform search and persist ids
        if st.button("Search", type="primary", use_container_width=True, disabled=not query.strip()):
            try:
                with st.spinner("Searching arXiv…"):
                    papers = arxiv_search_free_text(query.strip(), max_results=n_results)
                    time.sleep(0.15)

                if not papers:
                    bucket["search_results"] = []
                    st.warning("No results found.")
                else:
                    ids = [upsert_paper(p) for p in papers]
                    bucket["search_results"] = ids  # ✅ persist across reruns
                    st.success(f"Found {len(ids)} papers ✅")

            except Exception as e:
                bucket["search_results"] = []
                st.error(f"Search failed: {e}")

        # ✅ Always render persisted results (so Open works reliably)
        if bucket.get("search_results"):
            st.markdown("### Results")
            for pid in bucket["search_results"]:
                paper_card_home(pid, section="search")

            if st.button("Clear results", use_container_width=True):
                bucket["search_results"] = []
                st.rerun()

        st.divider()
        st.caption("Tip for judges: click **Open** on any paper → Paper page.")


def render_paper():
    bucket = u_bucket()
    settings = bucket["settings"]
    st.subheader("📄 Paper")

    pid = bucket.get("selected_paper") or bucket.get("last_played")
    if not pid:
        st.info("Open a paper from **Home** first.")
        return

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
                if v.get("quick_summary") is None:
                    v["quick_summary"] = llm_quick_summary(p, settings["llm_model"])

            speak_text = (v["quick_summary"] or "")[: settings["max_chars_to_speak"]]
            with st.spinner("Generating speech…"):
                audio_out = elevenlabs_tts(speak_text, settings["voice_id"])

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
                v["deep_dive"] = llm_deep_dive(p, settings["llm_model"])
            st.success("Done ✅")
        except Exception as e:
            st.error(f"Deep dive failed: {e}")

    st.divider()

    colA, colB = st.columns([1, 1], gap="large")
    with colA:
        st.markdown("### ⚡ Quick Summary")
        if v.get("quick_summary"):
            st.text_area("Summary", v["quick_summary"], height=240)
    with colB:
        st.markdown("### 🔎 Deep Dive")
        if v.get("deep_dive"):
            st.text_area("Deep dive", v["deep_dive"], height=240)
    st.divider()
    st.markdown("### 🕸️ Connected Papers (Topic Graph)")

    show_graph = st.toggle("Show related graph", value=True, key=f"show_graph_{pid}")
    if show_graph:
        render_related_paper_graph(
            pid,
            max_neighbors=10,
            min_overlap=1,
            height=520,
        )


def render_chat():
    bucket = u_bucket()
    settings = bucket["settings"]
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
        return

    if mode == "voice":
        st.write("## 3) Transcribe (Voxtral)")
        try:
            with st.spinner("Transcribing…"):
                transcript = voxtral_transcribe(audio_bytes, "recording.wav", settings["stt_model"])
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
            messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_text}]
            answer = mistral_chat_raw(messages, model=settings["llm_model"], temperature=0.2, max_tokens=900)

        st.text_area("Answer", answer, height=220)

        if pid:
            bucket["last_played"] = pid
            bucket["now_playing_mode"] = "qa"
            bump_history(pid)
    except Exception as e:
        st.error(f"Mistral call failed: {e}")
        st.stop()

    st.write("## 5) Speak (ElevenLabs)")
    speak_text = answer[: settings["max_chars_to_speak"]]
    try:
        with st.spinner("Generating speech…"):
            audio_out = elevenlabs_tts(speak_text, settings["voice_id"])
        bucket["tts_last_audio"] = audio_out
        st.audio(audio_out, format="audio/mpeg")
    except Exception as e:
        st.error(f"ElevenLabs TTS failed: {e}")
        st.info("You can still use transcript + answer; TTS is optional for the demo.")


def render_library():
    bucket = u_bucket()
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