#!/usr/bin/env bash
# Levanta las dos mitades y muere si cualquiera de las dos cae: un contenedor a
# medias es peor que uno caído, porque el orquestador no lo reinicia.
set -euo pipefail

if [[ ! -f /app/assets/voz/referencia.wav || ! -f /app/assets/voz/referencia.txt ]]; then
    echo "FALTA la voz de referencia: monta assets/voz/ con referencia.wav y referencia.txt" >&2
    exit 1
fi

# El modelo de lenguaje sólo se levanta aquí si no hay proveedor externo. Con
# LLM_API_KEY puesta, el chat sale por la red y el contenedor se ahorra 2,4 GB.
if [[ -z "${LLM_API_KEY:-}" ]]; then
    echo "sin LLM_API_KEY: levantando el modelo local" >&2
    export LLM_URL="${LLM_URL:-http://127.0.0.1:8080/v1/chat/completions}"
    export LLM_MODEL="${LLM_MODEL:-local}"
    llama-server -hf "${LLM_GGUF}" --host 127.0.0.1 --port 8080 \
        -c "${LLM_CTX}" -t "${LLM_THREADS:-$(nproc)}" &
fi

python -m uvicorn gabriela.server:app --host 0.0.0.0 --port 8000 &

wait -n
echo "una de las dos mitades terminó; cerrando el contenedor" >&2
kill 0
