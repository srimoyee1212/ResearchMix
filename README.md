# ResearchMix 🎧🎙️  
**Your Spotify for Research Papers**  

Discover → Summarize → Ask → Listen

Team: Fair Game

ResearchMix is an audio-first, interactive research experience powered by live arXiv data.
It transforms static research papers into structured summaries, deep dives, voice conversations, and connected discovery graphs — so you can learn hands-free while commuting, cooking, or coding.
---

## Demo (30 seconds)
1. Home → Open a recommended paper (personalized via arXiv + user interests)

2. Paper → Click Generate + Narrate Summary

3. Click Ask about this paper → ask by voice

4. “What’s the core contribution?”

5. “What are the limitations?”

6. Scroll to Connected Papers Graph → click a node → jump instantly

7. Show Library → listening history + saved papers

---

## What it does 
### 🎶 Smart Paper Discovery (Live arXiv)
- Real-time arXiv API integration
- Personalized recommendations based on user interests + listening history
- Trending research (cs.AI, cs.LG, cs.CL, cs.CV)
- Full-text arXiv search
- Continue listening to last played paper

### 📄 Paper Intelligence
- **Quick Summary**: TL;DR + key bullets + “when to use”
- **Deep Dive**: structured explanation designed for narration
- **Voice Q&A**: ask questions about the selected paper and get spoken answers back

### 🎙️ Voice-first pipeline
- **Voice input** (in-browser recorder)
- **Voxtral STT** for speech → text
- **Mistral LLM** for summaries, deep dives, recommendations, and Q&A
- **ElevenLabs TTS** for spoken narration

### 🕸️ Connected Paper Graph
- Visualizes topical relationships between papers
- Based on shared arXiv categories
- Clickable nodes for instant navigation
- Turns linear reading into graph-based exploration

---

## 🧠 Agent-Inspired Workflow
ResearchMix uses an **agentic workflow pattern**: the app orchestrates multiple specialized “roles” (agents) in a deterministic pipeline:

- **Curator Agent** (LLM): generates trending/recommended playlists and synthesizes paper metadata for discovery
- **Summarizer Agent** (LLM): produces structured quick summaries
- **Explainer Agent** (LLM): produces deep-dive narrations optimized for audio
- **Q&A Research Assistant Agent** (LLM + paper-aware system prompt): answers user questions grounded in the selected paper context
- **Voice IO Tools**: Voxtral (STT) + ElevenLabs (TTS) used as tools within the agentic flow

While the MVP runs in a single Streamlit app, the structure maps cleanly to a multi-agent setup (e.g., routing between summarizer vs explainer vs Q&A agent based on user intent).

---

## Why this exists (problem)
Research papers are valuable but high-friction:
- Reading demands sustained attention and screen time
- Skimming abstracts doesn’t build intuition
- Typing questions breaks flow

ResearchMix turns papers into an **interactive audio experience**—listen first, then ask.

---

## Architecture
**Streamlit UI**  
→ **Browser Audio Recorder**  
→ **Voxtral STT** (`/v1/audio/transcriptions`)  
→ **Mistral Chat** (`/v1/chat/completions`)  
→ **ElevenLabs TTS** (`/v1/text-to-speech/{voice_id}`)

### High-level flow
1. User selects a paper (Recommended/Trending/Search)
2. User chooses mode: Summary / Deep Dive / Q&A / Connected Papers
3. If voice: STT transcribes → LLM responds → TTS narrates

---

## Tech stack
- **Streamlit** (UI)
- **Mistral Voxtral** (Speech-to-Text)
- **Mistral LLM** (playlists, summaries, deep dives, Q&A)
- **ElevenLabs** (Text-to-Speech)

---

## Getting started (local)
### 1) Create venv + install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
### 2) Configure environment

Create a .env file:
```bash
MISTRAL_API_KEY=your_mistral_key
ELEVENLABS_API_KEY=your_elevenlabs_key

# Optional (defaults shown)
ELEVENLABS_VOICE_ID=your_choice_of_voice
MISTRAL_LLM_MODEL=mistral-tiny-latest
MISTRAL_STT_MODEL=voxtral-mini-transcribe-2507
```
### Run the app
```bash
streamlit run app.py
```
> Once you run this, you can also open the same on your phone using the computer/laptop's IP and correct port.
---


## Cost/safety controls

- Max characters to speak slider limits TTS cost and avoids overly long audio

- Prompts encourage concise, structured answers and avoid overclaiming

- Session state isolates user interactions within a run

---

## Team Fair Game

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/srimoyee1212">
        <img src="https://github.com/srimoyee1212.png" width="100px;" alt="srimoyee1212"/>
        <br />
        <sub><b>Srimoyee Mukhopadhyay</b></sub>
      </a>
    </td>
  </tr>
</table>
