# syntax=docker/dockerfile:1
#
# FastAPI sirve el visor, la voz y los visemas. El texto lo escribe un proveedor
# externo compatible con OpenAI (Groq por defecto) si hay LLM_API_KEY; si no, se
# levanta el llama-server que viene incluido y todo corre sin red. El segundo
# modo cuesta 2,4 GB más de RAM.
#
# Los modelos NO se hornean aquí (~3,5 GB y licencias que no queremos
# redistribuir): se descargan al primer arranque a /modelos, que debe ser un
# volumen persistente. Sin él, cada reinicio vuelve a bajarlos.
#
# La voz (neuphonic/neutts-nano-spanish y neuphonic/neucodec) vive en repos
# *gated*: hace falta HF_TOKEN en el entorno, de una cuenta que haya aceptado
# los términos de ambos en la web. Sin eso el arranque no baja la voz.

# ---- llama.cpp ----------------------------------------------------------
# Se compila en lugar de copiar un binario publicado: así enlaza contra las
# mismas libc y libcurl que la imagen final, sin cruzar distribuciones.
FROM debian:trixie-slim AS llama
ARG LLAMA_TAG=master
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git ca-certificates libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 --branch "${LLAMA_TAG}" https://github.com/ggml-org/llama.cpp /src
# GGML_NATIVE=OFF es obligatorio: con él encendido el compilador usa las
# instrucciones del runner que construye la imagen (AVX-512 en un CI moderno) y
# el binario muere con SIGILL en los Xeon del cluster, que son más viejos.
# Apagado, llama.cpp detecta las extensiones en tiempo de ejecución.
RUN cmake -S /src -B /build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DLLAMA_CURL=ON \
    && cmake --build /build --target llama-server -j "$(nproc)"

# ---- servicio -----------------------------------------------------------
FROM python:3.13-slim-trixie

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 libcurl4 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=llama /build/bin/llama-server /usr/local/bin/llama-server

WORKDIR /app

# Capa propia para las dependencias: cambiar el código no vuelve a bajar torch.
COPY docker/requirements-linux.txt ./
RUN pip install --no-cache-dir -r requirements-linux.txt

COPY src/ ./src/
COPY web/ ./web/
# gabriela.glb no se versiona: hay que generarlo con el pipeline ANTES de
# construir la imagen (ver docker/README.md). Sin él el COPY falla, que es
# mejor que descubrirlo con el contenedor ya desplegado y sin cara.
COPY assets/gabriela.glb assets/visemes.json ./assets/
COPY docker/arranque.sh /usr/local/bin/arranque.sh
RUN chmod +x /usr/local/bin/arranque.sh

# Todo lo pesado y lo mutable vive fuera de la imagen.
ENV PYTHONPATH=/app/src \
    HF_HOME=/modelos/hf \
    MULETILLAS_DIR=/modelos/muletillas \
    LLAMA_CACHE=/modelos/llama \
    LLM_GGUF=Qwen/Qwen2.5-3B-Instruct-GGUF:Q4_K_M \
    LLM_CTX=2048 \
    NEUTTS_DEVICE=cpu \
    PYTHONUNBUFFERED=1
VOLUME /modelos

# La voz de referencia se monta: no se versiona ni se redistribuye.
# Hacen falta assets/voz/referencia.wav (7-10 s) y referencia.txt.

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=15m --retries=3 \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/').read()"

CMD ["/usr/local/bin/arranque.sh"]
