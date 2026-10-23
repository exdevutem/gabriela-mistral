# syntax=docker/dockerfile:1
#
# FastAPI sirve el visor, la voz y los visemas. Contesta las frecuentes con
# respuestas grabadas y, si ninguna calza, un LLM pequeño (LFM2.5 1,2B en GGUF)
# contesta con los documentos del museo y lo dice con la voz en vivo. Todo
# corre dentro, sin red ni claves de API.
#
# Los modelos NO se hornean aquí (~3 GB y licencias que no queremos
# redistribuir): se descargan al primer arranque a /modelos, que debe ser un
# volumen persistente. Sin él, cada reinicio vuelve a bajarlos.
#
# La voz (neuphonic/neutts-nano-spanish y neuphonic/neucodec) vive en repos
# *gated*: hace falta HF_TOKEN en el entorno, de una cuenta que haya aceptado
# los términos de ambos en la web. Sin eso el arranque no baja la voz.

# ---- llama-cpp-python ---------------------------------------------------
# La voz en vivo (vivo.py) y el LLM (saber.py) corren sobre llama.cpp por
# llama-cpp-python, que no publica ruedas para Linux con Python 3.13: se compila
# aquí y la imagen final sólo instala la rueda, sin compilador.
FROM python:3.13-slim-trixie AS llama
ARG LLAMA_CPP_PYTHON=0.3.36
# GGML_NATIVE=OFF es obligatorio: con él encendido el compilador usa las
# instrucciones del runner que construye la imagen (AVX-512 en un CI moderno) y
# el binario muere con SIGILL en los Xeon del cluster, que son más viejos. A
# cambio hay que nombrar las extensiones a mano: sin ninguna, llama.cpp queda en
# SSE y va varias veces más lento. AVX2 + FMA las tiene cualquier Xeon desde
# Haswell (2013); si el nodo es anterior, construir con --build-arg
# LLAMA_CPU="-DGGML_AVX=ON".
ARG LLAMA_CPU="-DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON"
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake \
    && rm -rf /var/lib/apt/lists/*
RUN CMAKE_ARGS="-DGGML_NATIVE=OFF ${LLAMA_CPU}" \
    pip wheel --no-cache-dir --no-deps -w /ruedas "llama-cpp-python==${LLAMA_CPP_PYTHON}"

# ---- servicio -----------------------------------------------------------
FROM python:3.13-slim-trixie

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Capa propia para las dependencias: cambiar el código no vuelve a bajar torch.
COPY docker/requirements-linux.txt ./
RUN pip install --no-cache-dir -r requirements-linux.txt
COPY --from=llama /ruedas /ruedas
RUN pip install --no-cache-dir --no-deps /ruedas/*.whl && rm -rf /ruedas

COPY src/ ./src/
COPY web/ ./web/
# gabriela.glb no se versiona: hay que generarlo con el pipeline ANTES de
# construir la imagen (ver docker/README.md). Sin él el COPY falla, que es
# mejor que descubrirlo con el contenedor ya desplegado y sin cara.
COPY assets/gabriela.glb assets/visemes.json ./assets/
# Los documentos del museo para el LLM (pipeline/conocimiento.py).
COPY assets/conocimiento/fragmentos.jsonl ./assets/conocimiento/
COPY docker/arranque.sh /usr/local/bin/arranque.sh
RUN chmod +x /usr/local/bin/arranque.sh

# Todo lo pesado y lo mutable vive fuera de la imagen.
ENV PYTHONPATH=/app/src \
    HF_HOME=/modelos/hf \
    FRECUENTES_DIR=/modelos/frecuentes \
    RESPALDO_DIR=/modelos/respaldo \
    VIVO_GGUF=/modelos/neutts-nano-spanish-f16.gguf \
    NEUTTS_DEVICE=cpu \
    VIVO_DEVICE=cpu \
    LLM_DEVICE=cpu \
    WHISPER_DEVICE=cpu \
    PYTHONUNBUFFERED=1
VOLUME /modelos

# La voz de referencia se monta: no se versiona ni se redistribuye.
# Hacen falta assets/voz/referencia.wav (7-10 s) y referencia.txt.

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=15m --retries=3 \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/').read()"

CMD ["/usr/local/bin/arranque.sh"]
