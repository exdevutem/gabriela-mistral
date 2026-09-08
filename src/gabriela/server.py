"""Servidor: sirve el visor y responde por WebSocket con voz y visemas."""
from __future__ import annotations

import base64
import logging

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .chat import Conversacion
from .config import ASSETS, WEB
from .visemes import VISEMAS, timeline
from .voice import SAMPLE_RATE, a_wav, sintetizar

log = logging.getLogger("gabriela")
app = FastAPI(title="Gabriela Mistral")
app.mount("/assets", StaticFiles(directory=ASSETS), name="assets")
app.mount("/static", StaticFiles(directory=WEB), name="static")


@app.middleware("http")
async def sin_cache(peticion, siguiente):
    """Evita que el navegador sirva de caché el .glb o los módulos JS.

    Regenerar el modelo y ver la versión anterior en pantalla cuesta más tiempo
    de diagnóstico del que ahorra la caché en una aplicación que se sirve en
    local.
    """
    respuesta = await siguiente(peticion)
    respuesta.headers["Cache-Control"] = "no-store"
    return respuesta


@app.get("/")
def inicio() -> FileResponse:
    return FileResponse(WEB / "index.html")


@app.get("/debug")
def depuracion() -> FileResponse:
    return FileResponse(WEB / "debug.html")


@app.websocket("/ws")
async def conversar(ws: WebSocket) -> None:
    await ws.accept()
    charla = Conversacion()  # una conversación por conexión: el historial vive aquí
    await ws.send_json({"tipo": "listo", "visemas": VISEMAS})
    try:
        while True:
            pregunta = (await ws.receive_text()).strip()
            if not pregunta:
                continue
            await ws.send_json({"tipo": "pensando"})
            try:
                texto = charla.responder(pregunta)
                await ws.send_json({"tipo": "texto", "texto": texto})
                pcm = sintetizar(texto)
            except Exception:
                log.exception("falló la respuesta")
                await ws.send_json({"tipo": "error",
                                    "texto": "No pude responder. Inténtalo de nuevo."})
                continue
            await ws.send_json({
                "tipo": "habla",
                "texto": texto,
                "audio": base64.b64encode(a_wav(pcm)).decode(),
                "visemas": [[round(t, 3), v, round(w, 3)]
                            for t, v, w in timeline(pcm, texto, SAMPLE_RATE)],
            })
    except WebSocketDisconnect:
        pass
