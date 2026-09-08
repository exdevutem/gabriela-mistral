"""Texto + audio -> timeline de visemas.

La idea: los *tiempos* salen de la energía del audio real y las *formas* de boca
salen del texto. Por separado cada mitad se ve mal —la energía sola da una
mandíbula que sube y baja como una marioneta, el texto solo se desincroniza en
cuanto el TTS respira— pero combinadas dan un labio-sincronizado creíble.
"""
from __future__ import annotations

import unicodedata

import numpy as np

VENTANA_MS = 20.0
REPOSO = "X"

# El español es fonéticamente casi transparente, así que basta un mapa de letras:
# no hace falta espeak ni phonemizer, ni su binario de sistema.
# ponytail: si el sincronizado se ve desfasado en sílabas concretas, el upgrade
# es alineación forzada (Montreal Forced Aligner) en lugar de este mapa.
_MAPA = {
    **{c: "A" for c in "aá"},
    **{c: "E" for c in "eé"},
    **{c: "I" for c in "ií"},
    **{c: "O" for c in "oó"},
    **{c: "U" for c in "uúü"},
    **{c: "MBP" for c in "mbp"},
    **{c: "FV" for c in "fv"},
}
VISEMAS = ["A", "E", "I", "O", "U", "MBP", "FV", "C", REPOSO]


def secuencia(texto: str) -> list[str]:
    """Letras -> visemas, colapsando repeticiones consecutivas."""
    out: list[str] = []
    for ch in unicodedata.normalize("NFC", texto.lower()):
        if not ch.isalpha():
            continue
        v = _MAPA.get(ch, "C")  # cualquier otra consonante: boca entreabierta
        if not out or out[-1] != v:
            out.append(v)
    return out


def envolvente(pcm: bytes, sample_rate: int) -> np.ndarray:
    """RMS por ventana, normalizada a 0..1."""
    x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    n = max(1, int(sample_rate * VENTANA_MS / 1000))
    x = x[: len(x) - len(x) % n].reshape(-1, n)
    rms = np.sqrt((x**2).mean(axis=1))
    pico = rms.max()
    return rms / pico if pico > 0 else rms


def timeline(pcm: bytes, texto: str, sample_rate: int = 24_000,
             umbral: float = 0.06) -> list[tuple[float, str, float]]:
    """Devuelve [(segundo, visema, peso 0..1), ...] ordenado por tiempo.

    Los visemas se reparten según la *energía acumulada*, no según el tiempo:
    así los silencios no consumen visemas y las pausas del TTS no desfasan la boca.
    """
    env = envolvente(pcm, sample_rate)
    if len(env) == 0:
        return [(0.0, REPOSO, 0.0)]
    seq = secuencia(texto) or [REPOSO]

    hablando = env > umbral
    acum = np.cumsum(np.where(hablando, env, 0.0))
    total = acum[-1]
    # posición de cada ventana dentro del habla, en 0..len(seq)
    idx = (acum / total * len(seq)).astype(int).clip(0, len(seq) - 1) if total > 0 \
        else np.zeros(len(env), dtype=int)

    paso = VENTANA_MS / 1000
    fotogramas = [
        (i * paso, seq[k] if h else REPOSO, float(e) if h else 0.0)
        for i, (k, h, e) in enumerate(zip(idx, hablando, env))
    ]
    return _adelgazar(fotogramas)


def _adelgazar(fotogramas, delta: float = 0.08):
    """Deja solo los fotogramas donde cambia el visema o el peso salta.

    El navegador interpola entre keyframes, así que mandar 50 por segundo es
    desperdiciar ancho de banda.
    """
    out = [fotogramas[0]]
    for f in fotogramas[1:]:
        _, v, w = f
        _, pv, pw = out[-1]
        if v != pv or abs(w - pw) > delta:
            out.append(f)
    if out[-1] is not fotogramas[-1]:
        t, _, _ = fotogramas[-1]
        out.append((t + VENTANA_MS / 1000, REPOSO, 0.0))
    return out


if __name__ == "__main__":
    import argparse
    import wave

    p = argparse.ArgumentParser()
    p.add_argument("wav")
    p.add_argument("texto")
    a = p.parse_args()
    with wave.open(a.wav) as w:
        pcm, sr = w.readframes(w.getnframes()), w.getframerate()
    tl = timeline(pcm, a.texto, sr)
    for t, v, peso in tl[:40]:
        print(f"{t:6.2f}s  {v:<4} {'#' * int(peso * 30)}")
    print(f"... {len(tl)} keyframes, audio de {len(pcm) / 2 / sr:.1f}s")
