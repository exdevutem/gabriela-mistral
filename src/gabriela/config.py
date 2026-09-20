"""Configuración central: modelos locales y rutas."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "assets"
WEB = ROOT / "web"
VOZ = ASSETS / "voz"

# override: gana el .env del proyecto por sobre variables exportadas en el shell
load_dotenv(ROOT / ".env", override=True)

# --- Chat: cualquier API compatible con OpenAI ---
# Por defecto Groq, que tiene nivel gratuito sin tarjeta. Apuntando LLM_URL a un
# llama-server local, el mismo código funciona sin cuenta ni internet: es la
# salida si el museo se queda sin red.
LLM_URL = os.getenv("LLM_URL", "https://api.groq.com/openai/v1/chat/completions")
# Groq retira modelos cada pocos meses (llama-3.3-70b-versatile murió en agosto
# de 2026). Si empieza a responder 404, mira qué hay vivo con:
#   curl -s https://api.groq.com/openai/v1/models -H "Authorization: Bearer $LLM_API_KEY"
LLM_MODEL = os.getenv("LLM_MODEL", "qwen/qwen3.8-27b")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")  # vacío para un llama-server local

# --- Voz: F5-TTS afinado en español (clonación a partir de un audio de referencia) ---
F5_REPO = "jpgallegoar/F5-Spanish"
F5_CKPT = "model_1200000.safetensors"
F5_ARQ = "F5TTS_Base"  # arquitectura del checkpoint; el vocab.txt del repo la reetiqueta a español
REF_AUDIO = VOZ / "referencia.wav"
REF_TEXTO = VOZ / "referencia.txt"

# Lo que dice mientras piensa. Se sintetizan una vez y se guardan: son siempre
# las mismas, y pagarlas en cada pregunta sería añadir espera a la espera.
MULETILLAS_DIR = Path(os.getenv("MULETILLAS_DIR", VOZ / "muletillas"))
MULETILLAS = [
    "Déjame pensar.",
    "Mmm. Espera un momento.",
    "A ver cómo te lo digo.",
    "Buena pregunta, esa.",
]

# El mundo físico necesita perillas: en un M2 la síntesis es lo que marca la
# latencia, y bajar nfe_step la acorta a cambio de algo de calidad.
# CPU a propósito, también en Apple Silicon. En MPS la síntesis va 4 veces más
# rápida (11,6 s frente a 44 s) pero el proceso *muere sin traza* en cuanto F5
# parte el texto en más de un bloque, que es cualquier respuesta de dos frases.
# ponytail: si algún día MPS deja de caerse, F5_DEVICE=mps y a correr.
DEVICE = os.getenv("F5_DEVICE", "cpu")
NFE_STEP = int(os.getenv("F5_NFE_STEP", "16"))
VELOCIDAD = float(os.getenv("F5_VELOCIDAD", "0.9"))  # <1 = más pausada
# Semilla fija: la voz sale igual en cada ejecución. Además, sin ella F5-TTS
# sortea un entero enorme y lo escribe en PYTHONHASHSEED, que solo admite
# valores de 32 bits, y el intérprete lo escupe en cada llamada.
SEED = int(os.getenv("F5_SEED", "0"))
