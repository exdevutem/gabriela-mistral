"""Texto -> WAV con NeuTTS nano-spanish, ejecutándose en local.

NeuTTS clona, igual que el F5-TTS que había antes: no tiene voces prefabricadas.
La voz sale del par `assets/voz/referencia.wav` + `referencia.txt` (unos 10 s de
habla y su transcripción exacta). Cambiar esos dos archivos cambia la voz de
Gabriela.

Medido en un M2 de 8 GB, con la misma referencia y las mismas frases, todo
tras calentar:

| frase                    | F5-TTS | NeuTTS cpu | NeuTTS mps |
|--------------------------|--------|------------|------------|
| «Déjame pensar.»         | 10,7 s |      4,0 s |      2,5 s |
| «Mmm. Espera un momento.»| 13,7 s |      4,3 s |      2,9 s |
| una respuesta de ~9 s    | 30,0 s |     12,5 s |      9,1 s |
| factor de tiempo real    |   4,08 |       1,37 |       1,08 |

Sigue sin ser tiempo real: las muletillas hacen falta igual.
"""
from __future__ import annotations

import io
import platform
import wave
from functools import lru_cache

import numpy as np

from .config import (DEVICE, FRECUENTES, FRECUENTES_DIR, MULETILLAS,
                     MULETILLAS_DIR, NEUTTS_CODEC, NEUTTS_REPO, REF_AUDIO,
                     REF_TEXTO, RMS_OBJETIVO, SEED, TEMPERATURA)

SAMPLE_RATE = 24_000  # NeuCodec entrega 24 kHz, igual que el vocoder de F5
SAMPLE_WIDTH = 2
MAX_BYTES = 135  # tope por trozo; ver trozos()
# Tope para las frases de muletilla, más corto a propósito. Cuando llega la
# respuesta, el visor descarta el relleno pendiente pero deja terminar el trozo
# que está sonando: esa frase es el retraso máximo que el relleno puede costar.
# Con 70 bytes son unos 5 s. Medido: la voz va a unos 13 caracteres por segundo.
MAX_BYTES_MULETILLA = 70


def _usar_espeak_del_sistema() -> None:
    """El espeak-ng que trae `neutts` viene roto en macOS: la dylib empaquetada
    lleva compilada la ruta del directorio temporal donde se construyó, que ya
    no existe, y muere con «Error processing file .../phontab». Ni
    ESPEAK_DATA_PATH ni EspeakWrapper.set_data_path la corrigen.

    Se cambia por la de Homebrew (`brew install espeak-ng`), que sí conoce sus
    datos. Tiene que ser DESPUÉS de importar neutts, porque su import elige la
    empaquetada.

    ponytail: parche de un solo punto y sólo en macOS —en Linux la biblioteca
    empaquetada carga bien—. Cuando publiquen una rueda con la ruta correcta,
    esta función se borra entera.
    """
    if platform.system() != "Darwin":
        return
    import glob

    from phonemizer.backend.espeak.wrapper import EspeakWrapper

    for patron in ("/opt/homebrew/Cellar/espeak-ng/*/lib/libespeak-ng.*.dylib",
                   "/usr/local/Cellar/espeak-ng/*/lib/libespeak-ng.*.dylib"):
        if encontradas := glob.glob(patron):
            EspeakWrapper.set_library(encontradas[0])
            return
    raise RuntimeError(
        "falta espeak-ng del sistema: `brew install espeak-ng`. El que trae "
        "neutts no funciona en macOS."
    )


def _device() -> str:
    """Resuelve DEVICE="auto" a lo que esta máquina tenga.

    Aquí y no en config.py porque importar torch cuesta segundos y config lo
    carga todo el mundo, tests incluidos.
    """
    if DEVICE != "auto":
        return DEVICE
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


@lru_cache(maxsize=1)
def precargar():
    """Carga perezosa: son ~1,5 GB de pesos y tarda en arrancar (10,6 s medidos
    con los modelos ya en caché; la primera vez hay que bajarlos).

    Cacheado además porque mantenerlo en memoria es la diferencia entre
    sintetizar en segundos o en minutos.
    """
    from neutts import NeuTTS

    _usar_espeak_del_sistema()

    device = _device()
    return NeuTTS(
        backbone_repo=NEUTTS_REPO,
        codec_repo=NEUTTS_CODEC,
        backbone_device=device,
        codec_device=device,
        language="es",
        seed=SEED,
    )


@lru_cache(maxsize=1)
def _referencia() -> tuple[str, str]:
    if not REF_AUDIO.exists() or not REF_TEXTO.exists():
        raise RuntimeError(
            f"Falta la voz de referencia. Deja en {REF_AUDIO.parent} un "
            f"'{REF_AUDIO.name}' de unos 10 s y un '{REF_TEXTO.name}' con su "
            "transcripción exacta."
        )
    return str(REF_AUDIO), REF_TEXTO.read_text(encoding="utf-8").strip()


@lru_cache(maxsize=1)
def _codigos_referencia():
    """La referencia, ya pasada por el codec. Son 11,8 s que se pagan una sola
    vez: sin cachear se pagarían en cada frase.
    """
    audio, _ = _referencia()
    return precargar().encode_reference(audio)


def calentar() -> None:
    """Prepara todo lo que no debe pagarse con el visitante delante.

    Cargar los pesos no basta: hay que codificar el audio de referencia y poner
    en marcha los kernels de torch. Aquí se paga una vez, y de paso se graban
    las muletillas que falten, que sirven de calentamiento.
    """
    _codigos_referencia()
    if not grabar_muletillas() + grabar_frecuentes():
        sintetizar("Ay.")  # nada que grabar: hay que calentar igual


def _grabar(directorio, textos: list[str]) -> int:
    """Sintetiza a disco los trozos que falten. Devuelve cuántos grabó.

    Un archivo por *frase*, no por texto, porque el servidor los manda
    troceados: así empieza a sonar en cuanto está la primera en vez de esperar
    a tener los veinte segundos enteros.
    """
    directorio.mkdir(parents=True, exist_ok=True)
    grabados = 0
    for i, texto in enumerate(textos):
        for j, frase in enumerate(trozos(texto)):
            destino = directorio / f"{i}-{j}.wav"
            if destino.exists():
                continue
            destino.write_bytes(a_wav(sintetizar(frase)))
            grabados += 1
    return grabados


def _leer(directorio, textos: list[str]) -> list[list[tuple[str, bytes]]]:
    """Lo grabado, como lista de trozos (frase, PCM) por texto.

    Sólo devuelve los completos: a uno al que le falte un trozo se le cortaría
    la historia a la mitad. Los incompletos salen como lista vacía para que el
    índice siga correspondiendo con `textos`.
    """
    out = []
    for i, texto in enumerate(textos):
        partes = []
        for j, frase in enumerate(trozos(texto)):
            ruta = directorio / f"{i}-{j}.wav"
            if not ruta.exists():
                partes = []
                break
            with wave.open(str(ruta)) as w:
                partes.append((frase, w.readframes(w.getnframes())))
        out.append(partes)
    return out


def grabar_muletillas() -> int:
    """Graba las muletillas que falten. Son textos fijos: generarlos en cada
    pregunta sería añadir espera a la espera que vienen a tapar.
    """
    grabados = _grabar(MULETILLAS_DIR, MULETILLAS)
    if grabados:
        muletillas.cache_clear()
    return grabados


def grabar_frecuentes() -> int:
    """Graba las respuestas a las preguntas frecuentes que falten.

    Son 68 trozos la primera vez, unos cinco minutos. A cambio, un visitante
    que toca un badge oye la respuesta al instante y sin pasar por el LLM.
    """
    grabados = _grabar(FRECUENTES_DIR, [r for _, r in FRECUENTES])
    if grabados:
        frecuentes.cache_clear()
    return grabados


@lru_cache(maxsize=1)
def muletillas() -> list[list[tuple[str, bytes]]]:
    """Las muletillas grabadas y completas. Cacheado: se consultan en cada
    pregunta y son unos cientos de kB.
    """
    return [partes for partes in _leer(MULETILLAS_DIR, MULETILLAS) if partes]


@lru_cache(maxsize=1)
def frecuentes() -> dict[str, list[tuple[str, bytes]]]:
    """Las frecuentes grabadas, por su pregunta. Sólo las completas: una a
    medio grabar se responde mejor con el LLM que a medias.
    """
    grabadas = _leer(FRECUENTES_DIR, [r for _, r in FRECUENTES])
    return {p: partes for (p, _), partes in zip(FRECUENTES, grabadas) if partes}


def trozos(texto: str) -> list[str]:
    """Parte el texto en pedazos que se sintetizan de una pieza.

    Corta en fin de frase, y a la fuerza si un trozo pasa de MAX_BYTES. F5-TTS
    traía su `chunk_text` y este es su reemplazo; el tope de 135 bytes se hereda
    de él porque el servidor manda cada trozo en cuanto está y trozos cortos
    adelantan la primera frase.

    ponytail: NeuTTS rinde *mejor* cuanto más largo el trozo (RTF 1,29 en una
    frase de 10 s contra 1,79 en una de 2 s), así que este tope cuesta algo de
    rendimiento total a cambio de latencia inicial. Si algún día la primera
    frase deja de importar, subirlo es la optimización gratis.
    """
    partes: list[str] = []
    actual: list[str] = []
    largo = 0
    for palabra in texto.split():
        b = len(palabra.encode())
        if actual and largo + 1 + b > MAX_BYTES:
            partes.append(" ".join(actual))
            actual, largo = [], 0
        largo += (1 if actual else 0) + b
        actual.append(palabra)
        if palabra.endswith((".", "!", "?", "…", ".»", ".\"")):
            partes.append(" ".join(actual))
            actual, largo = [], 0
    if actual:
        partes.append(" ".join(actual))
    return partes or [texto]


def sintetizar(texto: str) -> bytes:
    """Devuelve PCM crudo (16-bit mono, 24 kHz)."""
    _, ref_texto = _referencia()
    wav = precargar().infer(
        texto, _codigos_referencia(), ref_texto, temperature=TEMPERATURA
    )
    x = np.asarray(wav, dtype=np.float32)
    # NeuTTS entrega más bajo que F5 y el nivel varía entre frases; sin esto la
    # muletilla suena a la mitad de volumen que la respuesta que la sigue.
    if rms := float(np.sqrt((x**2).mean())):
        x = x * (RMS_OBJETIVO / rms)
    return (x.clip(-1, 1) * 32767).astype("<i2").tobytes()


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
