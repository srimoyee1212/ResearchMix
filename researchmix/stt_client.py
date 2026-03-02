# researchmix/stt_client.py
import requests
from researchmix.config import get_env


def voxtral_transcribe(audio_bytes: bytes, filename: str, model: str) -> str:
    env = get_env()
    url = "https://api.mistral.ai/v1/audio/transcriptions"
    files = {"file": (filename, audio_bytes)}
    data = {"model": model}
    r = requests.post(
        url,
        headers={"Authorization": f"Bearer {env['MISTRAL_API_KEY']}"},
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