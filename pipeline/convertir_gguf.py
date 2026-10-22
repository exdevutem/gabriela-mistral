"""Convierte el backbone de NeuTTS nano-spanish a GGUF para llama.cpp (Metal/CPU).

Es lo que hace `assets/models/neutts-nano-spanish-f16.gguf`, que usa la voz en
vivo (src/gabriela/vivo.py). Si ya aceptaste los términos de
`neuphonic/neutts-nano-spanish-q4-gguf` en Hugging Face, no hace falta: pon ese
repo en VIVO_GGUF y sale más pequeño y más rápido. Esto existe porque ese repo es
*gated* aparte del modelo base.

    uv run pipeline/convertir_gguf.py

Dos arreglos que el conversor no hace solo, y sin los cuales no suena nada:

- Los 65 536 tokens `<|speech_N|>` salen marcados como de control, y llama.cpp
  no escribe el texto de un token de control: el streaming devuelve cadenas
  vacías y NeuTTS no encuentra ningún código de audio. Se marcan como
  definidos por el usuario, que es como vienen en el GGUF oficial.
- Mismo motivo en el tokenizer.json que lee el conversor (`special: false`).

ponytail: f16 (457 MB) en vez de cuantizar. En un M5 ya va a 0,3 de tiempo real;
cuantizar a Q8_0/Q4_0 con `llama-quantize` es la mejora gratis si en el clúster
(sólo CPU) hace falta.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from gguf import GGUFReader
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
SALIDA = ROOT / "assets" / "models" / "neutts-nano-spanish-f16.gguf"
REPO = "neuphonic/neutts-nano-spanish"
DEFINIDO = 4  # tipo de token «definido por el usuario» en GGUF (3 es «de control»)


def main() -> None:
    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    origen = Path(snapshot_download(REPO))
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        hf = tmp / "hf"
        hf.mkdir()
        for f in ("config.json", "generation_config.json", "model.safetensors",
                  "special_tokens_map.json"):
            (hf / f).symlink_to((origen / f).resolve())
        tok = json.loads((origen / "tokenizer.json").read_text())
        for a in tok["added_tokens"]:
            if a["content"].startswith("<|speech_"):
                a["special"] = False
        (hf / "tokenizer.json").write_text(json.dumps(tok, ensure_ascii=False))
        cfg = json.loads((origen / "tokenizer_config.json").read_text())
        for v in cfg.get("added_tokens_decoder", {}).values():
            if v["content"].startswith("<|speech_"):
                v["special"] = False
        (hf / "tokenizer_config.json").write_text(json.dumps(cfg, ensure_ascii=False))

        llama = tmp / "llama.cpp"
        subprocess.run(["git", "clone", "--depth", "1", "-q",
                        "https://github.com/ggml-org/llama.cpp.git", str(llama)], check=True)
        subprocess.run([sys.executable, str(llama / "convert_hf_to_gguf.py"), str(hf),
                        "--outfile", str(SALIDA), "--outtype", "f16"],
                       check=True, env={"PYTHONPATH": str(llama / "gguf-py"), "PATH": "/usr/bin:/bin"})

    r = GGUFReader(SALIDA, "r+")
    tokens, tipos = r.fields["tokenizer.ggml.tokens"], r.fields["tokenizer.ggml.token_type"]
    n = 0
    for k, i in enumerate(tokens.data):
        if bytes(tokens.parts[i]).startswith(b"<|speech_"):
            tipos.parts[tipos.data[k]][0] = DEFINIDO
            n += 1
    r.data.flush()
    assert n == 65_536, f"esperaba 65 536 tokens de audio y encontré {n}"
    print(f"listo: {SALIDA} ({SALIDA.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
