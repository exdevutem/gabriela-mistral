"""Servidor: sirve el visor y responde por WebSocket con voz y visemas.

No hay chat. Lo que el visitante escribe o dice se compara con las preguntas
frecuentes (entender.py) y suena la respuesta grabada que corresponda: siempre
la misma, verificada y al instante.
"""
from __future__ import annotations

import asyncio
import base64
import contextlib
import logging

from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import entender
from .config import ASSETS, RESPALDO, WEB
from .visemes import VISEMAS, timeline
from .voice import SAMPLE_RATE, a_wav, calentar, frecuentes, respaldo

log = logging.getLogger("gabriela")
# Uvicorn sólo configura sus propios loggers: sin esto no se ve qué entendió
# Whisper ni qué respuesta eligió, que es lo primero que hay que mirar cuando
# ella contesta otra cosa.
logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")


@contextlib.asynccontextmanager
async def ciclo(_app):
    """Graba lo que falte y carga Whisper antes de aceptar visitas.

    Grabar es largo sólo la primera vez (unos siete minutos con NeuTTS); después
    queda en disco. Whisper y el elector son ~1,5 GB que no deben cargarse con
    alguien hablándole.
    """
    for nombre, paso in (("grabar las respuestas", calentar),
                         ("cargar Whisper y el elector", entender.calentar)):
        try:
            log.info("preparando: %s…", nombre)
            await asyncio.to_thread(paso)
        except Exception:
            log.exception("no pude %s", nombre)
    log.info("listo para visitas")
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
        "relleno": False,
    }


async def _responder(ws: WebSocket, dicho: str) -> None:
    """Hace sonar la frecuente que corresponde a `dicho`, o el respaldo."""
    grabadas = frecuentes()
    # Un badge manda la pregunta tal cual: no hace falta preguntarle a nadie.
    pregunta = dicho if dicho in grabadas else await asyncio.to_thread(
        entender.elegir, dicho, grabadas)
    partes = grabadas.get(pregunta) or respaldo()
    texto = " ".join(f for f, _ in partes) or RESPALDO
    await ws.send_json({"tipo": "texto", "texto": texto, "partes": len(partes),
                        "pregunta": pregunta})
    if not partes:
        # Sin el respaldo grabado no hay voz que mandar: queda escrito.
        await ws.send_json({"tipo": "error", "texto": texto})
    for i, (frase, pcm) in enumerate(partes):
        await ws.send_json(_habla(frase, pcm, fin=i == len(partes) - 1))


@app.websocket("/ws")
async def conversar(ws: WebSocket) -> None:
    """Texto: una pregunta escrita o un badge. Binario: lo que dijo al
    micrófono, PCM de 16 bits mono a 16 kHz."""
    await ws.accept()
    # Sólo se ofrecen las que están grabadas: un badge que lleva a una respuesta
    # a medio grabar es peor que no ofrecerlo.
    await ws.send_json({"tipo": "listo", "visemas": VISEMAS,
                        "frecuentes": list(frecuentes())})
    while True:
        m = await ws.receive()
        if m["type"] == "websocket.disconnect":
            return
        try:
            if m.get("bytes"):
                await ws.send_json({"tipo": "pensando"})
                dicho = await asyncio.to_thread(entender.transcribir, m["bytes"])
                await ws.send_json({"tipo": "oido", "texto": dicho})
            else:
                dicho = (m.get("text") or "").strip()
                if not dicho:
                    continue
            await _responder(ws, dicho)
        except Exception:
            log.exception("falló la respuesta")
            await ws.send_json({"tipo": "error",
                                "texto": "No pude responder. Inténtalo de nuevo."})
