# /// script
# requires-python = ">=3.12"
# dependencies = ["f5-tts", "soundfile", "numpy", "python-dotenv"]
# ///
"""Graba las preguntas frecuentes con F5-TTS, offline, en vez de con NeuTTS.

Las frecuentes suenan solas —no las sigue la voz en vivo—, así que pueden
salir de un modelo más lento y mejor sin que se note el cambio de timbre. Las
muletillas NO: a ellas las sigue la respuesta de NeuTTS y el salto se oiría.

Se corre a mano en una máquina con memoria de sobra y el resultado se copia al
volumen del museo. Deja el mismo formato que `voice._grabar` —`i-j.wav` a 24
kHz y `notas.json`—, así que el servidor lo da por revisado y no lo regraba.

    uv run pipeline/grabar_frecuentes_f5.py              # todas
    uv run pipeline/grabar_frecuentes_f5.py --solo 9 13  # sólo esas

F5 no entra en las dependencias del proyecto a propósito: la imagen del
clúster no lo necesita. Por eso el script declara las suyas y `uv run` le arma
su propio entorno.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gabriela.config import (FRECUENTES, FRECUENTES_DIR, GRABADO_ACEPTABLE,  # noqa: E402
                             REF_AUDIO, REF_TEXTO, RMS_OBJETIVO)
from gabriela.voice import _anotar, _notas, _puntuar, a_wav, trozos  # noqa: E402

log = logging.getLogger("f5")

F5_REPO = "jpgallegoar/F5-Spanish"
F5_CKPT = "model_1200000.safetensors"
# Offline el tiempo no importa: 32 es el valor por defecto de F5, cuatro veces
# los pasos que se usaban en vivo (8).
NFE_STEP = int(os.getenv("F5_NFE_STEP", "32"))
VELOCIDAD = float(os.getenv("F5_VELOCIDAD", "0.9"))  # <1 = más pausada
# Más tomas que NeuTTS (6): aquí no hay arranque del servidor esperando.
INTENTOS = int(os.getenv("F5_INTENTOS", "10"))


def _leer_wav_con_soundfile() -> None:
    """torchaudio lee vía torchcodec, que revienta con el FFmpeg de Homebrew.
    Mismo parche que tenía `voice.py` cuando la voz era F5.
    """
    import soundfile as sf
    import torch
    import torchaudio

    def load(uri, *_, channels_first=True, **__):
        datos, sr = sf.read(uri, dtype="float32", always_2d=True)
        t = torch.from_numpy(datos)
        return (t.T if channels_first else t), sr

    torchaudio.load = load


def _modelo():
    _leer_wav_con_soundfile()
    import torch
    from f5_tts.api import F5TTS
    from huggingface_hub import hf_hub_download

    return F5TTS(
        model="F5TTS_Base",
        ckpt_file=hf_hub_download(F5_REPO, F5_CKPT),
        vocab_file=hf_hub_download(F5_REPO, "vocab.txt"),
        # En CPU el Air se calienta y baja el ritmo: de 40 s a 6 min por toma.
        # En MPS son unos 20 s, una vez que Metal compiló sus kernels.
        device=os.getenv("F5_DEVICE", "mps" if torch.backends.mps.is_available() else "cpu"),
    )


def _toma(modelo, texto: str, semilla: int) -> bytes:
    wav, _, _ = modelo.infer(
        ref_file=str(REF_AUDIO),
        ref_text=REF_TEXTO.read_text(encoding="utf-8").strip(),
        gen_text=texto,
        nfe_step=NFE_STEP,
        speed=VELOCIDAD,
        seed=semilla,
        show_info=lambda *_: None,
    )
    x = np.asarray(wav, dtype=np.float32)
    # Al mismo nivel que lo que dice en vivo, o el badge suena más fuerte.
    if rms := float(np.sqrt((x**2).mean())):
        x = x * (RMS_OBJETIVO / rms)
    return (x.clip(-1, 1) * 32767).astype("<i2").tobytes()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--solo", type=int, nargs="*", help="índices de FRECUENTES")
    p.add_argument("--out", type=Path, default=FRECUENTES_DIR)
    a = p.parse_args()

    a.out.mkdir(parents=True, exist_ok=True)
    notas = _notas(a.out)
    modelo = _modelo()
    for i, (pregunta, respuesta) in enumerate(FRECUENTES):
        if a.solo and i not in a.solo:
            continue
        for j, frase in enumerate(trozos(respuesta)):
            mejor, mejor_nota = b"", -1.0
            for intento in range(INTENTOS):
                pcm = _toma(modelo, frase, semilla=intento)
                if (nota := _puntuar(pcm, frase)) > mejor_nota:
                    mejor, mejor_nota = pcm, nota
                if mejor_nota >= GRABADO_ACEPTABLE:
                    break
            nombre = f"{i}-{j}.wav"
            (a.out / nombre).write_bytes(a_wav(mejor))
            _anotar(a.out, notas, nombre, mejor_nota)
            log.info("%s %.2f %s", nombre, mejor_nota, frase)


if __name__ == "__main__":
    main()
