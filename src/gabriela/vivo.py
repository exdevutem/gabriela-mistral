"""Texto -> voz de Gabriela al momento, en trozos que empiezan a sonar enseguida.

Es lo mismo que voice.py —NeuTTS clonando `referencia.wav`— pero con el
backbone en GGUF sobre llama.cpp, que usa la GPU (Metal) o, sin ella, la CPU
mucho mejor que transformers: 0,32 de tiempo real en un M5 contra 1,08 en MPS.
voice.py sigue grabando las frecuentes; esto es para lo que no está grabado.

El texto se parte en unidades (ver unidades()) y cada una se sintetiza entera,
no por trozos de audio: el sincronizado de labios (visemes.py) necesita el audio
completo de la frase para repartir las formas de boca, y con unidades de ~3 s y
un tiempo real de 0,32 la siguiente está lista mucho antes de que termine de
sonar la anterior. El primer sonido sale tras la primera unidad, ~1 s.

    uv run python -m gabriela.vivo "Hola, soy Lucila Godoy Alcayaga."
"""
from __future__ import annotations

import atexit
import logging
import re
import threading
import time
from collections.abc import Iterator
from functools import lru_cache

import numpy as np

from . import voice
from .config import (GRABADO_ACEPTABLE, GRABADO_TEMPERATURA, REF_AUDIO, RMS_OBJETIVO,
                     SEED, TEMPERATURA, VIVO_DEVICE, VIVO_GGUF, VIVO_INTENTOS)

log = logging.getLogger(__name__)

# llama.cpp no es reentrante sobre un mismo contexto: una síntesis a la vez.
_candado = threading.Lock()
# La primera unidad se parte en una coma si es larga, para que el primer sonido
# no espere a una frase entera; las demás van por frases (voice.trozos).
PRIMERA_MAX = 60
CLAUSULA_MIN = 25
# Bajo este largo el juez (voice._puntuar) no es fiable —«Bienvenido al museo.»
# puntuó 0,43, 0,56 y 0,00 en tres tomas buenas—, y repetir una frase corta por
# nada es un segundo de latencia perdido.
# ponytail: umbral por largo de texto; si se cuela una voz ajena en frases
# cortas, el siguiente paso es juzgar con otro criterio que no dependa del tono.
JUZGAR_DESDE = 40


def unidades(texto: str) -> list[str]:
    """Las piezas que se sintetizan de una vez, en orden.

    Frases (voice.trozos), y la primera partida en su primera coma o punto y
    coma si pasa de PRIMERA_MAX caracteres y deja al menos CLAUSULA_MIN a cada
    lado: una cláusula suena bien suelta, media palabra no.
    """
    piezas = voice.trozos(" ".join(texto.split()))
    if piezas and len(piezas[0]) > PRIMERA_MAX:
        for m in re.finditer(r"[,;:] ", piezas[0]):
            if m.end() >= CLAUSULA_MIN and len(piezas[0]) - m.end() >= CLAUSULA_MIN:
                primera = piezas[0]
                piezas[:1] = [primera[:m.end()].strip(), primera[m.end():].strip()]
                break
    return piezas


@lru_cache(maxsize=1)
def precargar():
    """Backbone GGUF en la GPU y codec en el mismo dispositivo. Una vez por proceso."""
    from neutts import NeuTTS

    voice._usar_espeak_del_sistema()
    dispositivo = voice._device() if VIVO_DEVICE == "auto" else VIVO_DEVICE
    t = time.time()
    modelo = NeuTTS(backbone_repo=VIVO_GGUF, backbone_device=dispositivo,
                    codec_repo="neuphonic/neucodec", codec_device=dispositivo,
                    language="es", seed=SEED)
    # Sin esto, llama-cpp-python libera el modelo al desmontar el intérprete,
    # cuando ya no puede, y Metal aborta al salir con un volcado de pila.
    atexit.register(modelo.backbone.close)
    log.info("voz en vivo lista en %s (%.1f s)", dispositivo, time.time() - t)
    return modelo


@lru_cache(maxsize=1)
def _referencia():
    ref_texto = voice._referencia()[1]
    return precargar().encode_reference(str(REF_AUDIO)), ref_texto


def calentar() -> None:
    """Carga todo y sintetiza una frase corta: la primera llamada compila los
    kernels de Metal y tarda unos segundos que no deben caerle a un visitante."""
    sintetizar("Hola, soy Gabriela.")


def _una_toma(texto: str, temperatura: float, semilla: int) -> np.ndarray:
    modelo = precargar()
    codigos, ref_texto = _referencia()
    modelo._seed = semilla
    try:
        # Por infer_stream y no infer: con los tokens de audio como «definidos
        # por el usuario» (ver pipeline/convertir_gguf.py) llama-cpp-python sólo
        # los escribe al transmitir; infer no encuentra ningún código.
        trozos = modelo.infer_stream(texto, codigos, ref_texto, temperature=temperatura)
        return np.concatenate([np.asarray(t, dtype=np.float32) for t in trozos])
    finally:
        modelo._seed = SEED


def sintetizar(texto: str) -> bytes:
    """PCM de 16 bits mono a 24 kHz con la voz de la referencia.

    La primera toma es la caliente de siempre (TEMPERATURA); si puntúa mal
    (voz ajena, frase cortada) se repite más fría con otra semilla, y si ninguna
    llega se queda la mejor. Es el mismo juez que usa voice.py para lo grabado.
    """
    mejor, mejor_nota = b"", -1.0
    with _candado:
        for intento in range(max(VIVO_INTENTOS, 1)):
            x = _una_toma(texto, TEMPERATURA if intento == 0 else GRABADO_TEMPERATURA,
                          SEED + intento)
            if len(x) and (rms := float(np.sqrt((x**2).mean()))):
                x = x * (RMS_OBJETIVO / rms)
            pcm = (x.clip(-1, 1) * 32767).astype("<i2").tobytes()
            if len(texto) < JUZGAR_DESDE:
                return pcm
            nota = voice._puntuar(pcm, texto)
            if nota > mejor_nota:
                mejor, mejor_nota = pcm, nota
            if mejor_nota >= GRABADO_ACEPTABLE:
                break
            log.info("toma %d de %r puntuó %.2f; repito", intento + 1, texto, nota)
    return mejor


def hablar(texto: str) -> Iterator[tuple[str, bytes]]:
    """(frase, PCM) por cada unidad, según se van sintetizando."""
    for u in unidades(texto):
        yield u, sintetizar(u)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(description="Habla en vivo y mide la latencia.")
    p.add_argument("texto")
    p.add_argument("--out", default="/tmp/gabriela-vivo.wav")
    a = p.parse_args()
    calentar()
    t0, pcm, primero = time.time(), b"", None
    for frase, trozo in hablar(a.texto):
        primero = primero or time.time() - t0
        pcm += trozo
        print(f"{time.time() - t0:5.2f}s  {len(trozo) / 2 / voice.SAMPLE_RATE:4.1f}s de audio  «{frase}»")
    total, dur = time.time() - t0, len(pcm) / 2 / voice.SAMPLE_RATE
    open(a.out, "wb").write(voice.a_wav(pcm))
    print(f"primer sonido {primero:.2f}s · total {total:.2f}s · audio {dur:.1f}s · "
          f"tiempo real {total / dur:.2f} · {a.out}")
