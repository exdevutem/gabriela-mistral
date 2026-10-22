"""Servidor: sirve el visor y responde por WebSocket con voz y visemas.

No hay chat. Lo que el visitante escribe o dice se compara con las preguntas
frecuentes (entender.py) y suena la respuesta grabada que corresponda: siempre
la misma, verificada y al instante.
"""
from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging

from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import entender, vivo
from .config import ASSETS, RESPALDO, WEB
from .visemes import VISEMAS, timeline
from .voice import SAMPLE_RATE, a_wav, calentar, frecuentes, respaldo

log = logging.getLogger("gabriela")
# Uvicorn sólo configura sus propios loggers: sin esto no se ve qué entendió
# Whisper ni qué respuesta eligió, que es lo primero que hay que mirar cuando
# ella contesta otra cosa.
logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
# Tope de lo que se sintetiza de una vez: unos 40 s de voz. El servidor no es
# un lector de libros, y un mensaje enorme ocuparía la GPU a todos los demás.
MAX_DECIR = 600


@contextlib.asynccontextmanager
async def ciclo(_app):
    """Graba lo que falte y carga Whisper antes de aceptar visitas.

    Grabar es largo sólo la primera vez (unos siete minutos con NeuTTS); después
    queda en disco. Whisper y el elector son ~1,5 GB que no deben cargarse con
    alguien hablándole.
    """
    for nombre, paso in (("grabar las respuestas", calentar),
                         ("cargar Whisper y el elector", entender.calentar),
                         ("cargar la voz en vivo", vivo.calentar)):
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


def _habla(partes: list[tuple[str, bytes]], continuacion: bool = False,
           fin: bool = True) -> dict:
    """La respuesta entera como un solo audio, con la boca de cada frase
    corrida a donde empieza esa frase.

    Se graban por frase, pero se mandan juntas: el reproductor del visor
    muestra el tiempo de la respuesta completa, y pausar o adelantar en un
    solo audio es trivial; en una cola de frases, no.
    """
    pcm, visemas, inicio = b"", [], 0.0
    for frase, trozo in partes:
        visemas += [[round(inicio + t, 3), v, round(w, 3)]
                    for t, v, w in timeline(trozo, frase, SAMPLE_RATE)]
        pcm += trozo
        inicio += len(trozo) / 2 / SAMPLE_RATE
    return {
        "tipo": "habla",
        "texto": " ".join(f for f, _ in partes),
        "audio": base64.b64encode(a_wav(pcm)).decode(),
        "visemas": visemas,
        "duracion": round(inicio, 3),
        "fin": fin,
        "continuacion": continuacion,
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
    await ws.send_json({"tipo": "texto", "texto": texto, "pregunta": pregunta})
    if partes:
        await ws.send_json(_habla(partes))
    else:
        # Sin el respaldo grabado no hay voz que mandar: queda escrito.
        await ws.send_json({"tipo": "error", "texto": texto})


async def _decir(ws: WebSocket, texto: str) -> None:
    """Dice `texto` con su voz, sintetizada al momento.

    Manda una `habla` por unidad según se sintetiza: el visor las encadena, así
    que empieza a sonar la primera mientras se hacen las demás. De dónde sale el
    texto (hoy, quien lo mande; mañana, el modelo conversacional) no es asunto
    de este camino.
    """
    unidades = vivo.unidades(texto)
    await ws.send_json({"tipo": "texto", "texto": " ".join(unidades)})
    for i, frase in enumerate(unidades):
        pcm = await asyncio.to_thread(vivo.sintetizar, frase)
        await ws.send_json(_habla([(frase, pcm)], continuacion=i > 0,
                                  fin=i == len(unidades) - 1))


async def _parcial(ws: WebSocket, pcm: bytes) -> None:
    """Lo que Whisper entiende de lo grabado hasta ahora, para mostrarlo
    mientras el visitante sigue hablando."""
    if texto := await asyncio.to_thread(entender.transcribir, pcm):
        await ws.send_json({"tipo": "parcial", "texto": texto})


@app.websocket("/ws")
async def conversar(ws: WebSocket) -> None:
    """Mensajes de texto, en JSON:

    - `{"tipo": "pregunta", "texto": …}`: escrita, o un badge.
    - `{"tipo": "decir", "texto": …}`: que lo diga con su voz, en vivo.
    - `{"tipo": "fin"}`: el visitante terminó de hablar al micrófono.

    Mensajes binarios: trozos de lo que va diciendo, PCM de 16 bits mono a
    16 kHz. Mientras llegan se transcribe lo acumulado y se devuelve como
    `parcial`; con `fin` se transcribe entero y se responde.
    """
    await ws.accept()
    # Sólo se ofrecen las que están grabadas: un badge que lleva a una respuesta
    # a medio grabar es peor que no ofrecerlo.
    await ws.send_json({"tipo": "listo", "visemas": VISEMAS,
                        "frecuentes": list(frecuentes())})
    voz = bytearray()
    parcial: asyncio.Task | None = None
    while True:
        m = await ws.receive()
        if m["type"] == "websocket.disconnect":
            return
        try:
            if m.get("bytes"):
                voz += m["bytes"]
                # ponytail: se transcribe todo lo acumulado, una pasada tras
                # otra, sin esperar. Con 8 s de tope y whisper-small son ~1 s
                # por pasada; con grabaciones largas habría que transcribir
                # sólo la cola.
                if parcial is None or parcial.done():
                    parcial = asyncio.create_task(_parcial(ws, bytes(voz)))
                continue
            msg = json.loads(m.get("text") or "{}")
            if msg.get("tipo") == "decir":
                if texto := (msg.get("texto") or "").strip()[:MAX_DECIR]:
                    await _decir(ws, texto)
                continue
            if msg.get("tipo") == "fin":
                if parcial:
                    await parcial  # que su texto no llegue después del final
                pcm, voz = bytes(voz), bytearray()
                await ws.send_json({"tipo": "pensando"})
                dicho = await asyncio.to_thread(entender.transcribir, pcm)
                await ws.send_json({"tipo": "oido", "texto": dicho})
            else:
                dicho = (msg.get("texto") or "").strip()
                if not dicho:
                    continue
            await _responder(ws, dicho)
        except Exception:
            log.exception("falló la respuesta")
            await ws.send_json({"tipo": "error",
                                "texto": "No pude responder. Inténtalo de nuevo."})
