# Gabriela Mistral — busto 3D conversacional

Un retrato escultórico en 3D que responde por escrito, habla con voz sintetizada y
mueve la cara mientras lo hace. Backend en Python, render en el navegador.

> Es una recreación con inteligencia artificial, no una grabación ni un testimonio
> real de Gabriela Mistral (1889–1957). La interfaz lo indica en pantalla, y el
> prompt le prohíbe inventar versos y atribuírselos.

## Cómo funciona

```
tu mensaje ──> Gemini (texto) ──> Gemini TTS ──> PCM 24 kHz
                                        │
                        envolvente RMS ─┴─ visemas del texto
                                        │
                 {audio, timeline} ──> three.js ──> morph targets
```

El labio-sincronizado combina dos fuentes: los **tiempos** salen de la energía del
audio real y las **formas de boca** del texto. Los visemas se reparten según
energía acumulada, no según tiempo, así que las pausas del TTS no desfasan la boca.

## Requisitos

- Python 3.13 y [uv](https://docs.astral.sh/uv/)
- Una API key de [Google AI Studio](https://aistudio.google.com/apikey) en `.env`

```bash
uv sync
cp .env.example .env    # y pega tu GEMINI_API_KEY
```

Si tienes `GEMINI_API_KEY` exportada en tu shell, el `.env` del proyecto manda:
`config.py` carga con `override=True`.

## Puesta en marcha

```bash
uv run uvicorn gabriela.server:app --port 8000
```

- `http://localhost:8000` — la conversación
- `http://localhost:8000/debug` — un slider por morph target, para afinar visemas

## La cabeza

Ahora mismo el repo trae una **cabeza provisional** (un esferoide con boca modelada
por gaussianas) para poder probar toda la cadena. No se parece a nadie. La cabeza
real se construye así:

### 1. Descargar FLAME (paso manual, requiere registro)

Regístrate en <https://flame.is.tue.mpg.de/> y descarga el modelo. Su licencia es
de investigación y uso no comercial, lo que encaja con un proyecto universitario
pero impide automatizar la descarga. Deja en `assets/flame/`:

- el modelo (`flame2023.pkl` o equivalente)
- `landmark_embedding.npy` del mismo paquete

### 2. Convertir y ajustar

```bash
uv run python pipeline/convert_flame.py --inspeccionar   # ver qué trae el .pkl
uv run python pipeline/convert_flame.py                  # -> assets/flame/flame.npz
uv run python pipeline/fit_face.py assets/fotos/gabriela-referencia.png --preview
```

El `--preview` deja en `/tmp/ajuste.png` la comparación entre los landmarks de la
foto (verde) y los del modelo ajustado (magenta). **Míralo antes de seguir**: si no
se reconoce, sube `REGULARIZACION` en `fit_face.py` o consigue una foto mejor.

La foto incluida es de 250×331 en blanco y negro, que es poco. La Biblioteca
Nacional Digital de Chile y Memoria Chilena tienen material de dominio público en
mejor resolución.

### 3. Exportar

Cuando el ajuste convenza, `pipeline/export_glb.py` escribe el `.glb` con los
visemas como morph targets, y se puede **borrar `pipeline/cabeza_provisional.py`
entero**.

## Verificación

```bash
uv run python tests/test_visemes.py                          # lógica de visemas
uv run python pipeline/landmarks.py assets/fotos/gabriela-referencia.png
uv run python -m gabriela.chat "¿Quién eres?"                # solo texto
uv run python -m gabriela.voice "Hola" --out /tmp/g.wav      # solo voz
uv run python -m gabriela.visemes /tmp/g.wav "Hola"          # timeline
```

## Notas de dependencias

- **mediapipe fijado en 0.10.35**: la 1.x revienta en macOS ARM
  (`DrishtiMetalHelper / Service is unavailable`) porque fuerza Metal.
- Las fotografías históricas vienen en escala de grises, y mediapipe exige tres
  canales: `landmarks.py` convierte a RGB antes de detectar. Sin eso el grafo
  aborta con un error de dimensiones que no dice de dónde viene.
- El modelo de landmarks se descarga aparte:
  ```bash
  curl -sL -o assets/models/face_landmarker.task \
    https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
  ```
- **Modelos de Gemini**: `gemini-3.1-flash-lite` con razonamiento apagado responde
  en ~2,6 s; `gemini-3.6-flash` tarda ~24 s, demasiado para conversar. El TTS
  (`gemini-3.1-flash-tts-preview`) tarda ~15 s en frases largas y no admite
  streaming por esta vía.

## Pendiente

- Voz a voz con micrófono (`gemini-3.1-flash-live-preview`)
- Parpadeo: necesita el morph target de párpados que traerá FLAME
- Volumen de cabello: FLAME no trae pelo y su peinado recogido es reconocible
