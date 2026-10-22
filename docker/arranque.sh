#!/usr/bin/env bash
# Levanta el servidor. Ya no hay chat, así que tampoco llama-server.
set -euo pipefail

# Sólo NeuTTS clona desde la referencia; Gemini y Piper no la usan.
if [[ "${VOZ_MOTOR:-neutts}" == neutts && ( ! -f /app/assets/voz/referencia.wav || ! -f /app/assets/voz/referencia.txt ) ]]; then
    echo "FALTA la voz de referencia: monta assets/voz/ con referencia.wav y referencia.txt" >&2
    exit 1
fi

exec python -m uvicorn gabriela.server:app --host 0.0.0.0 --port 8000
