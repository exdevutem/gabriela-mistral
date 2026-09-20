"""Conversación con un modelo local servido por llama.cpp, con la persona de Gabriela."""
from __future__ import annotations

import re

import httpx

from .config import LLM_API_KEY, LLM_MODEL, LLM_URL
from .persona import SYSTEM


def _hasta_la_ultima_frase(texto: str) -> str:
    """Descarta la frase a medias que deja el tope de tokens.

    Da igual perder media idea: hablada, una frase cortada en seco suena a fallo.
    """
    corte = max(texto.rfind(c) for c in ".!?…")
    return texto[: corte + 1] if corte > 0 else texto


class Conversacion:
    """Un hilo de chat. Guarda el historial en memoria."""

    def __init__(self) -> None:
        self._mensajes = [{"role": "system", "content": SYSTEM}]
        cabeceras = {"Authorization": f"Bearer {LLM_API_KEY}"} if LLM_API_KEY else {}
        self._http = httpx.Client(timeout=120, headers=cabeceras)

    def anotar(self, pregunta: str, respuesta: str) -> None:
        """Deja constancia de un intercambio que no pasó por el modelo.

        Las preguntas frecuentes se responden con audio ya grabado. Si no se
        anotaran, un «¿y eso por qué?» a continuación llegaría al modelo sin
        saber de qué se habló.
        """
        self._mensajes.append({"role": "user", "content": pregunta})
        self._mensajes.append({"role": "assistant", "content": respuesta})

    def responder(self, texto: str) -> str:
        self._mensajes.append({"role": "user", "content": texto})
        r = self._http.post(LLM_URL, json={
            "model": LLM_MODEL,
            "messages": self._mensajes,
            "temperature": 0.9,
            # Corto a propósito, y no por ahorrar tokens: cada frase de más es
            # un bloque más que sintetizar, y la síntesis es el cuello de
            # botella. Medido: cuatro frases son 135 s de espera, dos son la
            # mitad. El tope va junto con la orden de brevedad de la persona,
            # porque ninguna de las dos basta sola —el modelo se pasa igual, y
            # cortar sin más deja la frase a medias—.
            "max_tokens": 90,
        })
        if r.status_code == 429:
            # El nivel gratuito se agota. Decirlo por su nombre ahorra media hora
            # de buscar el fallo en otra parte.
            raise RuntimeError("Cuota del proveedor agotada (429). Espera o cambia de modelo.")
        if r.status_code == 404:
            raise RuntimeError(
                f"El proveedor no conoce el modelo '{LLM_MODEL}' (404). Groq retira "
                "modelos con frecuencia; lista los vivos y ajusta LLM_MODEL."
            )
        if r.status_code == 401:
            raise RuntimeError("El proveedor rechazó la clave (401). Revisa LLM_API_KEY.")
        if r.status_code >= 400:
            raise RuntimeError(f"{LLM_URL} respondió {r.status_code}: {r.text[:200]}")
        eleccion = r.json()["choices"][0]
        crudo = eleccion["message"]["content"]
        # Un GGUF con razonamiento (Qwen3 y parientes) antepone <think>…</think>:
        # sin esto el TTS lo leería en voz alta.
        respuesta = re.sub(r"<think>.*?</think>", "", crudo, flags=re.S).strip()
        if eleccion.get("finish_reason") == "length":
            respuesta = _hasta_la_ultima_frase(respuesta)
        self._mensajes.append({"role": "assistant", "content": respuesta})
        return respuesta


if __name__ == "__main__":
    import sys

    c = Conversacion()
    print(c.responder(" ".join(sys.argv[1:]) or "¿Quién eres?"))
