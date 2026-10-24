"""Arranca el servidor y elige quién contesta.

    uv run gabriela          # todo local: frecuentes grabadas, documentos y LLM local
    uv run gabriela --ia     # todo lo contesta Claude Haiku; lo local no se carga

En modo IA sólo se cargan Whisper (para el micrófono) y la voz en vivo: ni el
LLM local, ni los documentos, ni el elector, ni se graban las frecuentes. En
Docker, o con uvicorn a mano, `MODO=ia` hace lo mismo que `--ia`.
"""
from __future__ import annotations

import argparse


def main() -> None:
    p = argparse.ArgumentParser(prog="gabriela", description=__doc__.splitlines()[0])
    p.add_argument("--ia", action="store_true",
                   help="contesta todo Claude Haiku (pide ANTHROPIC_KEY) y no carga el modelo local")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    a = p.parse_args()

    import uvicorn

    from . import config

    config.IA = config.IA or a.ia
    from .server import app

    uvicorn.run(app, host=a.host, port=a.port)


if __name__ == "__main__":
    main()
