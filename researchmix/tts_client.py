# researchmix/tts_client.py
import requests
from researchmix.config import get_env


def elevenlabs_tts(text: str, voice_id: str) -> bytes:
    env = get_env()
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    r = requests.post(
        url,
        headers={
            "xi-api-key": env["ELEVENLABS_API_KEY"],
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        },
        json={"text": text, "model_id": "eleven_multilingual_v2"},
        timeout=120,
    )
    if r.status_code != 200:
        raise RuntimeError(f"ElevenLabs TTS error {r.status_code}: {r.text}")
    return r.content