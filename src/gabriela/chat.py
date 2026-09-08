"""Conversación con Gemini, con la persona de Gabriela."""
from __future__ import annotations

from google.genai import types

from .config import CHAT_MODEL, client
from .persona import SYSTEM


class Conversacion:
    """Un hilo de chat. Guarda el historial en memoria."""

    def __init__(self) -> None:
        self._chat = client().chats.create(
            model=CHAT_MODEL,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM,
                temperature=0.9,
                max_output_tokens=400,
                # Sin razonamiento previo: baja la latencia de ~11s a ~2.6s, y una
                # conversación hablada no tolera esperas largas.
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )

    def responder(self, texto: str) -> str:
        return self._chat.send_message(texto).text.strip()


if __name__ == "__main__":
    import sys

    c = Conversacion()
    print(c.responder(" ".join(sys.argv[1:]) or "¿Quién eres?"))
