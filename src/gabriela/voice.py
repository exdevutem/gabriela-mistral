"""Texto -> WAV con F5-TTS afinado en español, ejecutándose en local.

F5-TTS clona: no tiene voces prefabricadas. La voz sale del par
`assets/voz/referencia.wav` + `referencia.txt` (unos 10 s de habla y su
transcripción exacta). Cambiar esos dos archivos cambia la voz de Gabriela.
"""
from __future__ import annotations

import io
import wave
from functools import lru_cache

import numpy as np

from .config import (DEVICE, F5_ARQ, F5_CKPT, F5_REPO, MULETILLAS,
                     MULETILLAS_DIR, NFE_STEP, REF_AUDIO, REF_TEXTO, SEED,
                     VELOCIDAD)

SAMPLE_RATE = 24_000  # el vocoder de F5-TTS entrega 24 kHz, igual que el TTS anterior
SAMPLE_WIDTH = 2


def _leer_wav_con_soundfile() -> None:
    """torchaudio 2.11 lee siempre vía torchcodec, que sólo carga con FFmpeg 4-7
    y revienta contra el 9 de Homebrew. soundfile trae su propia libsndfile y
    abre el WAV de referencia sin depender de nada del sistema.

    ponytail: parche de un solo punto —f5-tts llama a `torchaudio.load` una vez,
    sobre el audio de referencia—. Cuando torchcodec soporte el FFmpeg instalado,
    esta función se borra entera.
    """
    import soundfile as sf
    import torch
    import torchaudio

    def load(uri, *_, channels_first=True, **__):
        datos, sr = sf.read(uri, dtype="float32", always_2d=True)
        t = torch.from_numpy(datos)
        return (t.T if channels_first else t), sr

    torchaudio.load = load


@lru_cache(maxsize=1)
def precargar():
    """Carga perezosa: son ~1,3 GB de pesos y tarda en arrancar.

    Cacheado además porque mantenerlo en memoria es la diferencia entre
    sintetizar en segundos o en minutos.
    """
    _leer_wav_con_soundfile()

    from f5_tts.api import F5TTS
    from huggingface_hub import hf_hub_download

    return F5TTS(
        model=F5_ARQ,
        ckpt_file=hf_hub_download(F5_REPO, F5_CKPT),
        vocab_file=hf_hub_download(F5_REPO, "vocab.txt"),
        device=DEVICE,
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


def calentar() -> None:
    """Prepara todo lo que no debe pagarse con el visitante delante.

    Cargar los pesos no basta: la primera inferencia real paga además la
    preparación del audio de referencia y la puesta en marcha de los kernels de
    torch. Medido, eran **329 s** de más en la primera respuesta. Aquí se pagan
    una vez, y de paso se graban las muletillas que falten, que sirven de
    calentamiento.
    """
    faltan = grabar_muletillas()
    if not faltan:
        sintetizar("Ay.")  # nada que grabar: hay que calentar igual


def grabar_muletillas() -> int:
    """Sintetiza a disco las muletillas que no estén ya grabadas.

    Devuelve cuántas grabó. Son cuatro frases fijas: generarlas en cada pregunta
    sería añadir espera a la espera que vienen a tapar.
    """
    MULETILLAS_DIR.mkdir(parents=True, exist_ok=True)
    grabadas = 0
    for i, frase in enumerate(MULETILLAS):
        destino = MULETILLAS_DIR / f"{i}.wav"
        if destino.exists():
            continue
        destino.write_bytes(a_wav(sintetizar(frase)))
        grabadas += 1
    if grabadas:
        muletillas.cache_clear()
    return grabadas


@lru_cache(maxsize=1)
def muletillas() -> list[tuple[str, bytes]]:
    """Las muletillas grabadas, como (frase, PCM). Vacío si aún no hay ninguna.

    Cacheado: se consultan en cada pregunta y son unos pocos cientos de kB.
    """
    out = []
    for i, frase in enumerate(MULETILLAS):
        ruta = MULETILLAS_DIR / f"{i}.wav"
        if ruta.exists():
            with wave.open(str(ruta)) as w:
                out.append((frase, w.readframes(w.getnframes())))
    return out


def trozos(texto: str) -> list[str]:
    """Parte el texto en pedazos que F5-TTS sintetiza de una pieza.

    Se reutiliza su propio `chunk_text` para que los cortes coincidan con los
    que haría internamente: así cada llamada a `sintetizar` es un solo bloque y
    el servidor puede mandar la primera frase mientras genera las siguientes.

    ponytail: F5 calcula su límite según la duración de la referencia. 135 es su
    valor por defecto y queda por debajo del que sale con una referencia de 7-10
    s. Con una referencia más larga habría que bajarlo, y lo peor que pasa si no
    se hace es que un trozo salga partido en dos: más lento, no roto.
    """
    from f5_tts.infer.utils_infer import chunk_text

    return chunk_text(texto, max_chars=135) or [texto]


def sintetizar(texto: str) -> bytes:
    """Devuelve PCM crudo (16-bit mono, 24 kHz)."""
    ref_audio, ref_texto = _referencia()
    wav, _, _ = precargar().infer(
        ref_file=ref_audio,
        ref_text=ref_texto,
        gen_text=texto,
        nfe_step=NFE_STEP,
        speed=VELOCIDAD,
        seed=SEED,
        show_info=lambda *_: None,  # no ensuciar el log del servidor
    )
    return (np.asarray(wav, dtype=np.float32).clip(-1, 1) * 32767).astype("<i2").tobytes()


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
