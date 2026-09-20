"""Servidor: sirve el visor y responde por WebSocket con voz y visemas."""
from __future__ import annotations

import asyncio
import base64
import contextlib
import logging
import random

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .chat import Conversacion
from .config import ASSETS, WEB
from .visemes import VISEMAS, timeline
from .voice import (SAMPLE_RATE, a_wav, calentar, muletillas, sintetizar,
                    trozos)

log = logging.getLogger("gabriela")


@contextlib.asynccontextmanager
async def ciclo(_app):
    """Carga F5-TTS y lo hace hablar una vez antes de aceptar visitas.

    Son ~1,3 GB de pesos más la primera inferencia, que sola costaba 329 s. Todo
    eso se paga aquí y no en la primera pregunta.
    """
    try:
        log.info("calentando el modelo de voz…")
        await asyncio.to_thread(calentar)
        log.info("modelo de voz listo")
    except Exception:
        log.exception("no pude precalentar el modelo de voz")
    yield


app = FastAPI(title="Gabriela Mistral", lifespan=ciclo)
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


def _habla(texto: str, pcm: bytes, *, fin: bool) -> dict:
    """Un mensaje de voz: audio y boca para un trozo de frase."""
    return {
        "tipo": "habla",
        "texto": texto,
        "audio": base64.b64encode(a_wav(pcm)).decode(),
        "visemas": [[round(t, 3), v, round(w, 3)]
                    for t, v, w in timeline(pcm, texto, SAMPLE_RATE)],
        "fin": fin,
    }


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
            # Una muletilla ya grabada, de inmediato: la síntesis de verdad
            # tarda medio minuto, y medio minuto de estatua muda se lee como que
            # el programa se colgó.
            relleno = muletillas()
            if relleno:
                frase, pcm = random.choice(relleno)
                await ws.send_json(_habla(frase, pcm, fin=False))
            try:
                # En hilo aparte: quema CPU y en el bucle de eventos congelaría
                # al resto de conexiones.
                texto = await asyncio.to_thread(charla.responder, pregunta)
                partes = trozos(texto)
                # `partes` viaja con el texto para que la interfaz pueda decir
                # "preparando la voz, 2 de 3" en vez de un girador sin fondo.
                await ws.send_json({"tipo": "texto", "texto": texto,
                                    "partes": len(partes)})
                # Frase a frase: la síntesis tarda varias veces el tiempo del
                # audio que produce, así que esperar a tenerlo todo son minutos
                # de silencio. Mandando cada trozo en cuanto está, ella empieza
                # a hablar mientras se generan los siguientes.
                for i, parte in enumerate(partes):
                    pcm = await asyncio.to_thread(sintetizar, parte)
                    await ws.send_json(_habla(parte, pcm, fin=i == len(partes) - 1))
            except Exception:
                log.exception("falló la respuesta")
                await ws.send_json({"tipo": "error",
                                    "texto": "No pude responder. Inténtalo de nuevo."})
                continue
    except WebSocketDisconnect:
        pass
