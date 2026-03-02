# researchmix/llm_client.py
import requests
from typing import Dict, Any, Optional

from researchmix.config import get_env

# NOTE: these calls are identical to your working ones.


def mistral_chat_raw(messages, model: str, temperature: float = 0.2, max_tokens: int = 900) -> str:
    env = get_env()
    url = "https://api.mistral.ai/v1/chat/completions"
    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    r = requests.post(
        url,
        headers={"Authorization": f"Bearer {env['MISTRAL_API_KEY']}", "Content-Type": "application/json"},
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
    topics_str = ", ".join((topics or [])[:8]) if topics else "unknown"

    return (
        base
        + "The user is currently focused on the following paper.\n"
        + f"TITLE: {title}\n"
        + f"TOPICS: {topics_str}\n"
        + (f"ABSTRACT: {abstract}\n" if abstract else "ABSTRACT: (not available; rely on title + user context)\n")
        + "Answer primarily about this paper; if asked generally, respond normally.\n"
    )


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