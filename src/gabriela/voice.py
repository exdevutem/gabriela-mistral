"""Texto -> WAV con el TTS de Gemini."""
from __future__ import annotations

import io
import wave

from google.genai import types

from .config import TTS_LANGUAGE, TTS_MODEL, TTS_VOICE, client

SAMPLE_RATE = 24_000  # Gemini TTS devuelve PCM 16-bit mono a 24 kHz
SAMPLE_WIDTH = 2

# Instrucción de estilo: el modelo TTS la interpreta, no la lee en voz alta.
ESTILO = (
    "Di lo siguiente con voz de mujer chilena mayor, grave y pausada, "
    "con la cadencia serena de una maestra que escoge cada palabra:\n\n"
)


def sintetizar(texto: str) -> bytes:
    """Devuelve PCM crudo (16-bit mono, 24 kHz)."""
    resp = client().models.generate_content(
        model=TTS_MODEL,
        contents=ESTILO + texto,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                language_code=TTS_LANGUAGE,
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=TTS_VOICE)
                ),
            ),
        ),
    )
    return resp.candidates[0].content.parts[0].inline_data.data


def a_wav(pcm: bytes) -> bytes:
    """Envuelve el PCM en una cabecera WAV para que el navegador lo reproduzca."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(SAMPLE_WIDTH)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm)
    return buf.getvalue()


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("texto")
    p.add_argument("--out", default="/tmp/gabriela.wav")
    a = p.parse_args()
    open(a.out, "wb").write(a_wav(sintetizar(a.texto)))
    print(f"escrito {a.out}")
