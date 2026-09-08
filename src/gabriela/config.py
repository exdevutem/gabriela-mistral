"""Configuración central: API key y modelos."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from google import genai

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "assets"
WEB = ROOT / "web"

# override: gana el .env del proyecto por sobre variables exportadas en el shell
load_dotenv(ROOT / ".env", override=True)

CHAT_MODEL = "gemini-3.1-flash-lite"  # 2.6s con thinking apagado; gemini-3.6-flash tarda ~24s
TTS_MODEL = "gemini-3.1-flash-tts-preview"  # los *-2.5-*-tts ya no responden con esta cuenta
# Voz femenina grave. Alternativas a probar de oído: "Kore", "Sulafat", "Leda".
TTS_VOICE = "Gacrux"
TTS_LANGUAGE = "es-US"


@lru_cache(maxsize=1)
def client() -> genai.Client:
    """Cacheado: si el Client se recolecta, cierra su conexión HTTP y todo falla."""
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("Falta GEMINI_API_KEY en .env")
    return genai.Client(api_key=key)
