# /// script
# requires-python = ">=3.12"
# dependencies = ["f5-tts", "soundfile", "numpy", "scipy", "python-dotenv"]
# ///
"""Arma un dataset para afinar Piper con la voz de F5-TTS como maestra.

F5 suena bien pero es lento; Piper es rápido en CPU pero hay que enseñarle la
voz. Se le enseña con horas de F5 diciendo frases variadas: el alumno copia el
timbre del maestro y corre en el clúster sin GPU.

Sale en formato LJSpeech, que es lo que lee `piper_train.preprocess`:

    assets/voz/dataset/wav/000123.wav   22 050 Hz, mono, 16 bits
    assets/voz/dataset/metadata.csv     000123|texto de la frase

Se puede cortar y retomar cuando sea: lo ya grabado se salta.

    PYTORCH_ENABLE_MPS_FALLBACK=1 uv run pipeline/dataset_piper.py --frases 1500

Las frases son las de Common Voice en español (CC0), más las respuestas
frecuentes ya grabadas con F5 —ésas se copian, no se regraban—.
"""
from __future__ import annotations

import argparse
import logging
import random
import re
import urllib.request
import wave
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

# Primero: además de F5, deja `src/` en el path para lo de abajo.
from grabar_frecuentes_f5 import _modelo, _toma
from gabriela.config import FRECUENTES, FRECUENTES_DIR, GRABADO_ACEPTABLE, VOZ
from gabriela.voice import _puntuar, trozos

log = logging.getLogger("dataset")

CORPUS = ("https://raw.githubusercontent.com/common-voice/common-voice/main/"
          "server/data/es/sentence-collector.txt")
SALIDA = VOZ / "dataset"
# Piper "medium" trabaja a 22 050 Hz; F5 entrega 24 000. 22050/24000 = 147/160.
SR_PIPER = 22_050
# Menos que en las frecuentes: aquí una frase que sale mal se descarta y se
# pasa a otra, no hay una respuesta concreta que salvar.
INTENTOS = 4
# Sólo letras del español y puntuación: los números y las siglas los leería
# F5 de una forma y Piper aprendería a escribirlos de otra.
LIMPIA = re.compile(r"[¿¡«]?[A-ZÁÉÍÓÚÑ][A-Za-zÁÉÍÓÚÜÑáéíóúüñ¿¡ ,.;:?!«»()-]*[.?!»]")


def _frases(n: int) -> list[str]:
    cache = SALIDA / "corpus-es.txt"
    if not cache.exists():
        SALIDA.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(CORPUS, cache)
    # Entre 4 palabras y un trozo: más corto no enseña prosodia, más largo lo
    # partiría el servidor igual.
    buenas = sorted({s.strip() for s in cache.read_text(encoding="utf-8").splitlines()
                     if LIMPIA.fullmatch(s.strip()) and len(s.split()) >= 4
                     and len(s.strip().encode()) <= 135})
    random.Random(0).shuffle(buenas)  # orden fijo: retomar sigue la misma lista
    return buenas[:n]


def _escribir(ruta: Path, pcm24k: bytes) -> None:
    x = np.frombuffer(pcm24k, dtype="<i2").astype(np.float32)
    y = resample_poly(x, 147, 160).clip(-32768, 32767).astype("<i2")
    with wave.open(str(ruta), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR_PIPER)
        w.writeframes(y.tobytes())


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--frases", type=int, default=1500)
    a = p.parse_args()

    wavs = SALIDA / "wav"
    wavs.mkdir(parents=True, exist_ok=True)
    metadata = SALIDA / "metadata.csv"
    hechas = dict(l.split("|", 1) for l in metadata.read_text(encoding="utf-8").splitlines()) \
        if metadata.exists() else {}
    # Las descartadas también se anotan, para no reintentarlas al retomar.
    descartes = SALIDA / "descartadas.txt"
    malas = set(descartes.read_text(encoding="utf-8").splitlines()) if descartes.exists() else set()

    def anotar(clave: str, texto: str) -> None:
        with metadata.open("a", encoding="utf-8") as f:
            f.write(f"{clave}|{texto}\n")
        hechas[clave] = texto

    # Las frecuentes ya están grabadas con F5: sólo se remuestrean.
    for i, (_, respuesta) in enumerate(FRECUENTES):
        for j, frase in enumerate(trozos(respuesta)):
            clave, origen = f"frec-{i}-{j}", FRECUENTES_DIR / f"{i}-{j}.wav"
            if clave not in hechas and origen.exists() and len(frase.split()) >= 2:
                with wave.open(str(origen)) as w:
                    _escribir(wavs / f"{clave}.wav", w.readframes(w.getnframes()))
                anotar(clave, frase)

    modelo = None
    for k, frase in enumerate(_frases(a.frases)):
        clave = f"{k:06d}"
        if clave in hechas or clave in malas:
            continue
        modelo = modelo or _modelo()
        mejor, mejor_nota = b"", -1.0
        for intento in range(INTENTOS):
            pcm = _toma(modelo, frase, semilla=intento)
            if (nota := _puntuar(pcm, frase)) > mejor_nota:
                mejor, mejor_nota = pcm, nota
            if mejor_nota >= GRABADO_ACEPTABLE:
                break
        if mejor_nota < GRABADO_ACEPTABLE:
            with descartes.open("a", encoding="utf-8") as f:
                f.write(clave + "\n")
            log.info("%s descartada (%.2f) %s", clave, mejor_nota, frase)
            continue
        _escribir(wavs / f"{clave}.wav", mejor)
        anotar(clave, frase)
        log.info("%s %.2f %s", clave, mejor_nota, frase)


if __name__ == "__main__":
    main()
