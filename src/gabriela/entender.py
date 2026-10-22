"""De lo que dijo el visitante a una de las respuestas grabadas. Todo local.

Dos pasos, porque Whisper sólo sabe transcribir:

1. Whisper pasa la voz a texto (si escribió, este paso se salta).
2. Un modelo de embeddings multilingüe compara el sentido de ese texto con cada
   pregunta frecuente, con su respuesta y con sus VARIANTES, y se queda con la
   más parecida. Con la respuesta también, porque el visitante pregunta con sus
   palabras: «¿en qué año llegó el museo acá?» se parece poco a «¿Cómo empezó
   el museo?» y bastante a «Se mudó a esta casa en el ochenta y uno».

Medido el 21 de octubre de 2026 con 22 preguntas de prueba: MiniLM es el único
de cuatro modelos que separa limpio lo que es del tema (parecido ≥ 0,58) de lo
que no (≤ 0,45). Los e5, más grandes, aciertan parecido pero le dan 0,86 a
«¿qué hora es?», y así no hay umbral que sirva.

Si nada pasa de UMBRAL, devuelve None y ella dice que no entendió.

Los dos modelos van por `transformers`, que ya llega con NeuTTS: no se suma
ninguna dependencia.
"""
from __future__ import annotations

import logging
import threading
from functools import lru_cache

import numpy as np

from .config import (EMBEDDINGS_MODELO, FRECUENTES, UMBRAL, VARIANTES,
                     WHISPER_DEVICE, WHISPER_MODELO)

log = logging.getLogger(__name__)

SAMPLE_RATE = 16_000  # lo que espera Whisper, y lo que manda el visor
# Un pipeline de transformers no es seguro entre hilos, y con la transcripción
# en vivo hay varias pasadas por pregunta, más las de otras conexiones.
_turno = threading.Lock()


@lru_cache(maxsize=1)
def _whisper():
    import torch
    from transformers import pipeline

    device = WHISPER_DEVICE
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    return pipeline("automatic-speech-recognition", model=WHISPER_MODELO, device=device)


def transcribir(pcm: bytes) -> str:
    """PCM de 16 bits mono a 16 kHz -> lo que dijo, o "" si no se oyó nada."""
    x = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768
    # Whisper en silencio alucina, y en español alucina créditos de subtítulos
    # («Subtítulos realizados por la comunidad de Amara.org»). Si no hubo voz,
    # mejor no preguntarle.
    if len(x) < SAMPLE_RATE // 4 or float(np.sqrt((x**2).mean())) < 0.003:
        return ""
    with _turno:
        salida = _whisper()({"raw": x, "sampling_rate": SAMPLE_RATE},
                            generate_kwargs={"language": "spanish", "task": "transcribe"})
    return salida["text"].strip()


@lru_cache(maxsize=1)
def _codificador():
    from transformers import AutoModel, AutoTokenizer

    return (AutoTokenizer.from_pretrained(EMBEDDINGS_MODELO),
            AutoModel.from_pretrained(EMBEDDINGS_MODELO).eval())


def _vectores(textos: list[str]) -> np.ndarray:
    """Un vector de norma 1 por texto: el promedio de sus tokens, que es como
    se entrenó el modelo (mean pooling)."""
    import torch

    tok, modelo = _codificador()
    entrada = tok(textos, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        ocultos = modelo(**entrada).last_hidden_state
    mascara = entrada["attention_mask"].unsqueeze(-1).float()
    v = ((ocultos * mascara).sum(1) / mascara.sum(1)).numpy()
    return v / np.linalg.norm(v, axis=1, keepdims=True)


@lru_cache(maxsize=1)
def _indice() -> tuple[list[str], np.ndarray, np.ndarray]:
    """Las preguntas, un vector por cada texto que las representa —la pregunta,
    su respuesta y sus variantes— y a qué pregunta pertenece cada vector. Se
    calcula una vez: son textos fijos."""
    preguntas = [p for p, _ in FRECUENTES]
    textos, duenos = [], []
    for i, (p, r) in enumerate(FRECUENTES):
        for t in (p, r, *VARIANTES.get(p, [])):
            textos.append(t)
            duenos.append(i)
    return preguntas, _vectores(textos), np.array(duenos)


def mas_parecida(consulta: np.ndarray, vectores: np.ndarray, duenos: np.ndarray,
                 opciones: list[bool], umbral: float = UMBRAL) -> tuple[int | None, float]:
    """El índice de la frecuente que más se parece a `consulta` entre las
    `opciones` disponibles, y su parecido. None si ninguna llega al umbral.

    Cada frecuente puntúa lo de su texto más parecido. Aparte de elegir() para
    poder probarlo sin cargar ningún modelo.
    """
    parecido = np.full(len(opciones), -np.inf)
    np.maximum.at(parecido, duenos, vectores @ consulta)
    parecido = np.where(opciones, parecido, -np.inf)
    i = int(parecido.argmax())
    return (i if parecido[i] >= umbral else None), float(parecido[i])


def elegir(texto: str, disponibles) -> str | None:
    """La pregunta frecuente que mejor responde a `texto`, entre las que están
    grabadas (`disponibles`), o None si ninguna se le parece lo bastante."""
    if not texto.strip():
        return None
    preguntas, vectores, duenos = _indice()
    i, parecido = mas_parecida(_vectores([texto])[0], vectores, duenos,
                               [p in disponibles for p in preguntas])
    log.info("«%s» -> %s (%.2f)", texto, preguntas[i] if i is not None else "nada", parecido)
    return preguntas[i] if i is not None else None


def calentar() -> None:
    """Carga los dos modelos antes de la primera visita: Whisper son ~1 GB y
    tarda unos segundos, que no deben pagarse con alguien hablándole."""
    _indice()
    transcribir((np.sin(np.arange(SAMPLE_RATE) / 8) * 3000).astype("<i2").tobytes())


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for frase in sys.argv[1:]:
        print(f"{frase!r} -> {elegir(frase, {p for p, _ in FRECUENTES})}")
