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

# --- Voz: NeuTTS nano-spanish (clonación a partir de un audio de referencia) ---
# Reemplazó a F5-TTS el 20 de septiembre de 2026: misma voz de referencia, 2,8
# veces más rápido en el mismo M2 (medido: RTF 4,08 -> 1,45).
NEUTTS_REPO = os.getenv("NEUTTS_REPO", "neuphonic/neutts-nano-spanish")
NEUTTS_CODEC = os.getenv("NEUTTS_CODEC", "neuphonic/neucodec")
REF_AUDIO = VOZ / "referencia.wav"
REF_TEXTO = VOZ / "referencia.txt"

# Ambos repos son *gated*: hay que entrar con la cuenta de Hugging Face a
#   huggingface.co/neuphonic/neutts-nano-spanish  y  .../neucodec
# y aceptar los términos una vez, con HF_TOKEN en el entorno. Sin eso la
# descarga responde 403 y el servidor arranca sin voz.

# Lo que dice mientras piensa. Se sintetizan una vez y se guardan: son siempre
# las mismas, y pagarlas en cada pregunta sería añadir espera a la espera.
MULETILLAS_DIR = Path(os.getenv("MULETILLAS_DIR", VOZ / "muletillas"))
MULETILLAS = [
    "Déjame pensar.",
    "Mmm. Espera un momento.",
    "A ver cómo te lo digo.",
    "Buena pregunta, esa.",
]

# El mundo físico necesita perillas.
# CPU a propósito, también en Apple Silicon: no se ha medido que MPS gane aquí,
# y con F5 la caída era silenciosa.
DEVICE = os.getenv("NEUTTS_DEVICE", "cpu")
# NeuTTS no tiene perilla de velocidad como la tenía F5 (`speed=0.9`). El ritmo
# sale del audio de referencia: si habla apurada, la referencia es lo que hay
# que cambiar.
TEMPERATURA = float(os.getenv("NEUTTS_TEMPERATURA", "1.0"))
# Sale más bajo que F5 (RMS medido 0,044 frente a 0,099 en la misma frase).
# Se normaliza al nivel que tenía la voz anterior para no tocar el volumen del
# navegador ni el de la sala.
RMS_OBJETIVO = float(os.getenv("NEUTTS_RMS", "0.09"))
# Semilla fija: la voz sale igual en cada ejecución.
SEED = int(os.getenv("NEUTTS_SEED", "0"))
