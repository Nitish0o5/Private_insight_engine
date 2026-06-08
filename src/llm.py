from __future__ import annotations

import requests

from src.config import OLLAMA_HOST, OLLAMA_MODEL_NAME


def ask_ollama(prompt: str, model_name: str = OLLAMA_MODEL_NAME, host: str = OLLAMA_HOST) -> str:
    response = requests.post(
        f"{host}/api/generate",
        json={"model": model_name, "prompt": prompt, "stream": False},
        timeout=120,
    )
    response.raise_for_status()
    payload = response.json()
    answer = payload.get("response", "").strip()
    if not answer:
        raise RuntimeError("Ollama returned an empty response")
    return answer
