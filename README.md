# ResearchMix 🎧🎙️  
**Talk to research papers like they’re a playlist.**  
Record → **Voxtral** transcribes → **Mistral** reasons → **ElevenLabs** narrates.

**Team:** Fair Game

ResearchMix is an audio-first “Spotify for research papers” experience: it remembers what you last played, recommends a paper playlist, and lets you ask questions by voice—so you can learn hands-free while commuting, cooking, or coding.

---

## Demo (30 seconds)
1. Open **Home** → click **Play** on a recommended/trending paper  
2. Go to **Paper** → generate **Quick Summary** → click **Narrate summary**  
3. Go to **Chat** → ask by voice:  
   - “What’s the main contribution?”  
   - “What are the limitations?”  
   - “How would I implement this?”  
4. Show **Library** → saved papers + listening history

---

## What it does (Hackathon MVP)
### 🎶 Paper Player UX
- **Now Playing / Continue**: resume your last played paper instantly
- **Recommended playlist**: personalized suggestions (LLM-generated metadata for MVP)
- **Trending playlist**: great default for new users (LLM-generated metadata for MVP)
- **Library**: history + saved papers

### 📄 Paper modes
- **Quick Summary**: TL;DR + key bullets + “when to use”
- **Deep Dive**: structured explanation designed for narration
- **Voice Q&A**: ask questions about the selected paper and get spoken answers back

### 🎙️ Voice-first pipeline
- **Voice input** (in-browser recorder)
- **Voxtral STT** for speech → text
- **Mistral LLM** for summaries, deep dives, recommendations, and Q&A
- **ElevenLabs TTS** for spoken narration

---

## Agentic / Agentic Workflows (highlight)
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
2. User chooses mode: Summary / Deep Dive / Q&A
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

---

## How to use

### Home

- See Continue (last played)

- Browse Recommended and Trending

- Use Search (LLM-synth MVP) to generate relevant papers

### Paper

- Generate Quick Summary and Deep Dive

- Click Narrate to listen

### Chat

- Ask questions via voice or text

- Responses are paper-scoped to the selected paper

### Library

- History: previously played papers

- Saved: bookmarked papers

- Quick access to playlists

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
