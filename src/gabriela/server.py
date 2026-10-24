"""Servidor: sirve el visor y responde por WebSocket con voz y visemas.

Lo que el visitante escribe o dice se compara primero con las preguntas
frecuentes (entender.py) y, si una se parece, suena su respuesta grabada:
siempre la misma, verificada y al instante. Si ninguna se parece, un LLM local
contesta con los documentos oficiales del museo (saber.py) y lo dice con la voz
en vivo (vivo.py); si tampoco ahí hay nada, suena el «no te entendí» grabado.
Con el switch «IA» de la página no se usa nada de eso: todo lo contesta Claude
Haiku, y sólo la voz (y Whisper, si habló al micrófono) sigue siendo local.
"""
from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import time

from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import entender, saber, vivo
from .config import ASSETS, RESPALDO, UMBRAL, VIVO_COLCHON, WEB
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
                         ("cargar los documentos y el LLM", saber.calentar),
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


def _elegir(dicho: str, grabadas) -> str | None:
    """La frecuente que contesta `dicho`. Con documentos, tiene que parecérsele
    más que ellos (saber.piso); sin ellos, basta con pasar de UMBRAL."""
    umbral = UMBRAL
    if saber.disponible():
        umbral = max(umbral, saber.piso(dicho))
    return entender.elegir(dicho, grabadas, umbral)


async def _responder_ia(ws: WebSocket, dicho: str) -> None:
    """Lo que diga Claude, con la voz en vivo. Ni frecuentes ni documentos:
    con el switch «IA» encendido, hasta un badge lo contesta Claude."""
    await ws.send_json({"tipo": "pensando"})
    texto = await asyncio.to_thread(saber.responder_ia, dicho)
    if not texto:
        await ws.send_json({"tipo": "error", "texto": "No supe qué contestarte. "
                                                       "Pregúntame de otra forma."})
        return
    try:
        await _decir(ws, texto)
    except Exception:
        log.exception("no pude decir la respuesta de Claude")
        await ws.send_json({"tipo": "error", "texto": texto})


async def _responder(ws: WebSocket, dicho: str, ia: bool = False) -> None:
    """Hace sonar la frecuente que corresponde a `dicho`; si no hay, lo que
    digan los documentos del museo; y si tampoco, el respaldo. Con `ia`, todo
    lo contesta Claude."""
    if ia:
        await _responder_ia(ws, dicho)
        return
    grabadas = frecuentes()
    # Un badge manda la pregunta tal cual: no hace falta preguntarle a nadie.
    pregunta = dicho if dicho in grabadas else await asyncio.to_thread(_elegir, dicho, grabadas)
    if pregunta is None and saber.disponible():
        await ws.send_json({"tipo": "pensando"})
        try:
            texto, fuentes = await asyncio.to_thread(saber.responder, dicho)
        except Exception:
            # Sin LLM (no se pudo cargar) queda el respaldo grabado, como antes.
            log.exception("el LLM no pudo responder")
            texto, fuentes = None, []
        if texto:
            try:
                await _decir(ws, texto, fuentes)
            except Exception:
                # Sin voz en vivo (falta llama-cpp-python o falló la síntesis)
                # la respuesta igual vale: queda escrita, como el respaldo
                # sin grabar.
                log.exception("no pude decir la respuesta del LLM")
                await ws.send_json({"tipo": "error", "texto": texto})
            return
    partes = grabadas.get(pregunta) or respaldo()
    texto = " ".join(f for f, _ in partes) or RESPALDO
    await ws.send_json({"tipo": "texto", "texto": texto, "pregunta": pregunta})
    if partes:
        await ws.send_json(_habla(partes))
    else:
        # Sin el respaldo grabado no hay voz que mandar: queda escrito.
        await ws.send_json({"tipo": "error", "texto": texto})


async def _decir(ws: WebSocket, texto: str, fuentes: list[str] | None = None) -> None:
    """Dice `texto` con su voz, sintetizada al momento.

    Transmite el audio por unidades, que el visor encadena, pero con un colchón
    al principio: no manda nada hasta que lo ya sintetizado alcance a cubrir lo
    que falta, al ritmo medido en esta misma respuesta (`_alcanza`). Una unidad
    larga, o una que el juez de vivo.py manda a repetir, tarda más de lo que
    dura la anterior, y sin colchón la voz se cortaba a media respuesta. Desde
    que empieza, cada unidad sale apenas está lista. El texto sale junto con el
    primer audio: mientras tanto el visor sigue mostrando que piensa, en vez de
    un texto que no suena.

    De dónde sale el texto (quien lo mande, o un LLM de saber.py) no es asunto
    de este camino; `fuentes` son las URL de los documentos de donde salió.
    """
    unidades = vivo.unidades(texto)
    total, hecho, t0 = sum(map(len, unidades)), 0, time.monotonic()
    pendientes: list[tuple[str, bytes]] = []
    enviadas = 0
    for i, frase in enumerate(unidades):
        pcm = await asyncio.to_thread(vivo.sintetizar, frase)
        pendientes.append((frase, pcm))
        hecho += len(frase)
        ultima = i == len(unidades) - 1
        if not enviadas and not ultima:
            colchon = sum(len(p) for _, p in pendientes) / 2 / SAMPLE_RATE
            if not _alcanza(colchon, time.monotonic() - t0, hecho, total - hecho):
                continue
        if not enviadas:
            await ws.send_json({"tipo": "texto", "texto": " ".join(unidades),
                                "fuentes": fuentes or []})
        await ws.send_json(_habla(pendientes, continuacion=enviadas > 0, fin=ultima))
        enviadas += 1
        pendientes = []


def _alcanza(colchon: float, sintesis: float, hecho: int, falta: int) -> bool:
    """Si `colchon` segundos de audio cubren lo que falta sintetizar (`falta`
    caracteres), al ritmo de `sintesis` segundos por `hecho` caracteres que
    lleva esta respuesta, con VIVO_COLCHON de margen."""
    return colchon >= falta * sintesis / hecho * VIVO_COLCHON


async def _parcial(ws: WebSocket, pcm: bytes) -> None:
    """Lo que Whisper entiende de lo grabado hasta ahora, para mostrarlo
    mientras el visitante sigue hablando."""
    if texto := await asyncio.to_thread(entender.transcribir, pcm):
        await ws.send_json({"tipo": "parcial", "texto": texto})


@app.websocket("/ws")
async def conversar(ws: WebSocket) -> None:
    """Mensajes de texto, en JSON:

    - `{"tipo": "pregunta", "texto": …, "ia": bool}`: escrita, o un badge.
      `ia` es el switch de la página: todo lo contesta Claude.
    - `{"tipo": "decir", "texto": …}`: que lo diga con su voz, en vivo.
    - `{"tipo": "fin", "ia": bool}`: el visitante terminó de hablar al micrófono.

    Mensajes binarios: trozos de lo que va diciendo, PCM de 16 bits mono a
    16 kHz. Mientras llegan se transcribe lo acumulado y se devuelve como
    `parcial`; con `fin` se transcribe entero y se responde.
    """
    await ws.accept()
    # Sólo se ofrecen las que están grabadas: un badge que lleva a una respuesta
    # a medio grabar es peor que no ofrecerlo.
    await ws.send_json({"tipo": "listo", "visemas": VISEMAS,
                        "frecuentes": list(frecuentes()),
                        "ia": saber.ia_disponible()})
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
            ia = bool(msg.get("ia")) and saber.ia_disponible()
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
            await _responder(ws, dicho, ia)
        except Exception:
            log.exception("falló la respuesta")
            await ws.send_json({"tipo": "error",
                                "texto": "No pude responder. Inténtalo de nuevo."})
