# Gabriela Mistral — busto 3D conversacional

Un retrato escultórico en 3D que responde por escrito, habla con voz sintetizada y
mueve la cara mientras lo hace. Backend en Python, render en el navegador.

> Es una recreación con inteligencia artificial, no una grabación ni un testimonio
> real de Gabriela Mistral (1889–1957). La interfaz lo indica en pantalla, y el
> prompt le prohíbe inventar versos y atribuírselos.

## Cómo funciona

```
tu mensaje ──> Groq o llama.cpp ──> F5-TTS español ──> PCM 24 kHz
                                        │
                        envolvente RMS ─┴─ visemas del texto
                                        │
                 {audio, timeline} ──> three.js ──> morph targets
```

Ella habla por frases: el servidor manda cada una en cuanto la sintetiza, en vez
de esperar a tener la respuesta entera. Así empieza a hablar en unos 20 s en
lugar de 70, y termina en 48 en lugar de 85. Y como treinta segundos de estatua muda se leen como que el programa
murió, mientras tanto suelta una **muletilla ya grabada** —«Déjame pensar.»— y
la página dice en qué va: *preparando la voz… 2 de 4*.

El labio-sincronizado combina dos fuentes: los **tiempos** salen de la energía del
audio real y las **formas de boca** del texto. Los visemas se reparten según
energía acumulada, no según tiempo, así que las pausas del TTS no desfasan la boca.

## Requisitos

La voz corre siempre en local. El texto puede venir de un proveedor externo
compatible con OpenAI o de un modelo local; el código es el mismo.

- Python 3.13 y [uv](https://docs.astral.sh/uv/)
- Una voz de referencia en `assets/voz/` (ver más abajo)
- Para el texto, una de dos:
  - una clave gratuita de [Groq](https://console.groq.com/keys) (sin tarjeta), o
  - [llama.cpp](https://github.com/ggml-org/llama.cpp) (`brew install llama.cpp`)
    con un GGUF instruct, si prefieres no depender de la red

```bash
uv sync
cp .env.example .env    # y pega tu LLM_API_KEY
```

### La voz

F5-TTS no tiene voces prefabricadas: **clona** la que le des. Deja en `assets/voz/`
un `referencia.wav` de 7–10 s de habla limpia y un `referencia.txt` con su
transcripción exacta.

**Ojo con la duración**: F5-TTS recorta el audio a 12 s pero usa el texto entero
que le pases, así que un `.wav` largo con su transcripción completa sale
atropellado. Y el largo de la referencia se paga en cada síntesis. Cómo recortar
y transcribir, en `assets/voz/README.md`.

El checkpoint ([`jpgallegoar/F5-Spanish`](https://huggingface.co/jpgallegoar/F5-Spanish),
~1,3 GB, CC BY-NC 4.0) se descarga solo la primera vez a la caché de Hugging Face.

## Puesta en marcha

```bash
uv run uvicorn gabriela.server:app --port 8000
```

Para usar un modelo local en lugar de Groq, deja `LLM_API_KEY` vacía, descomenta
`LLM_URL` y `LLM_MODEL` en el `.env`, y levanta antes:

```bash
llama-server -hf Qwen/Qwen2.5-3B-Instruct-GGUF:Q4_K_M --port 8080 -c 4096
```

Arranca cargando F5-TTS —~1,3 GB, unos segundos—, para no pagarlo en la primera
pregunta con el usuario mirando un «pensando» eterno.

- `http://localhost:8000` — la conversación
- `http://localhost:8000/debug` — un slider por morph target, para afinar visemas

## Despliegue

Hay un `Dockerfile` con las dos mitades en una imagen, pensado para publicarla en
`ghcr.io` y levantarla como contenedor en Proxmox. El build necesita el `.glb` ya
generado y la voz de referencia se monta en vez de hornearse. Requisitos, flujo,
y cuánta RAM y CPU pedirle al nodo: [`docker/README.md`](docker/README.md).

La versión corta: **6 GB de RAM y 8 vCPU** usando Groq para el texto (8 GB si
levantas el modelo local). La latencia de la voz es el problema pendiente: más
de dos minutos por respuesta en el hardware del cluster.

## La cabeza

Se construye a partir de FLAME en cuatro pasos. El modelo no se versiona aquí: su
licencia es de investigación y uso no comercial.

### 1. Descargar FLAME (paso manual, requiere registro)

Regístrate en <https://flame.is.tue.mpg.de/> y descarga el modelo. Su licencia es
de investigación y uso no comercial, lo que encaja con un proyecto universitario
pero impide automatizar la descarga. Deja en `assets/flame/`:

- el modelo (`flame2023.pkl` o equivalente)
- `landmark_embedding.npy` del mismo paquete
- `FLAME_masks.pkl`, que trae las regiones semánticas (cuero cabelludo, cara,
  cuello, orejas). Sin él el cuero cabelludo hay que deducirlo por altura, y el
  borde del peinado sale recto como un flequillo de tazón en lugar de seguir el
  arco del nacimiento del pelo.

### 2. Convertir y ajustar

```bash
uv run python pipeline/convert_flame.py --inspeccionar   # ver qué trae el .pkl
uv run python pipeline/convert_flame.py                  # -> assets/flame/flame.npz
uv run python pipeline/fit_face.py assets/fotos/mistral-1946-frontal.jpg --preview
```

El `--preview` deja en `/tmp/ajuste.png` la comparación entre los landmarks de la
foto (verde) y los del modelo ajustado (magenta). **Míralo antes de seguir**: si no
se reconoce, sube `REGULARIZACION` en `fit_face.py` o consigue una foto mejor.

La referencia es `assets/fotos/mistral-1946-frontal.jpg` (1946, Marcos Chamúdez,
dominio público): 170 px entre ojos y encuadre frontal, contra los 37 px de la
foto original. Procedencia, atribución obligatoria y las candidatas descartadas
—con el motivo de cada descarte— están en `assets/fotos/PROCEDENCIA.md`.

Al elegir otra foto, mide los píxeles útiles de cara y no la resolución del
archivo: un escaneo de 4000 px donde ella sale de cuerpo entero aporta menos que
uno de 1700 px encuadrado en el rostro. Y **no mezcles épocas**: FLAME ajusta
forma facial, así que una foto de juventud y otra de vejez se promedian en una
cara que no es ninguna de las dos.

### 3. Exportar

```bash
uv run python pipeline/flame.py                  # autocomprobación del skinning
uv run python pipeline/export_glb.py             # -> assets/gabriela.glb
```

Los visemas se definen en `assets/visemes.json` como una apertura de mandíbula más
coeficientes del espacio de expresión. Esos componentes son PCA, no tienen
significado propio, así que se identificaron **midiendo** su efecto sobre los
landmarks de la boca: `psi[0]` controla el ancho (±11 mm) y `psi[3]` la apertura de
labios (+6 mm). Para afinarlos, muévelos en `/debug` y vuelve a exportar.

La apertura de la mandíbula es una articulación real, no una expresión: exige
skinning lineal sobre el esqueleto de cinco huesos de FLAME. `pipeline/flame.py` lo
implementa en numpy y se autocomprueba: con pose cero ningún vértice se mueve, y al
rotar la mandíbula baja la barbilla mientras el cráneo queda quieto.

## Verificación

```bash
uv run python tests/test_visemes.py                          # lógica de visemas
uv run python tests/test_local.py                            # chat y formato de audio
uv run python pipeline/landmarks.py assets/fotos/mistral-1946-frontal.jpg
uv run python -m gabriela.chat "¿Quién eres?"                # solo texto
uv run python -m gabriela.voice "Hola" --out /tmp/g.wav      # solo voz
uv run python -m gabriela.visemes /tmp/g.wav "Hola"          # timeline
```

## Notas de dependencias

- **Ni PyTorch ni chumpy**: el .pkl de FLAME envuelve sus arrays en `chumpy`
  (abandonada) y el landmark embedding guarda tensores de PyTorch. Ambos se leen
  con stubs propios —`convert_flame.py` y `torch_pickle.py`— en vez de arrastrar
  2,5 GB de dependencias para una conversión que se hace una sola vez.
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
- **PyTorch sí llega, pero por la voz**: F5-TTS lo arrastra (~2,5 GB). El
  pipeline de la cabeza sigue sin tocarlo —lee FLAME con stubs propios—, así que
  quien solo construya el modelo no necesita instalarlo.
- **Groq retira modelos cada pocos meses** (`llama-3.3-70b-versatile` murió en
  agosto de 2026). Si el chat empieza a dar 404, lista los vivos con
  `curl -s https://api.groq.com/openai/v1/models -H "Authorization: Bearer $LLM_API_KEY"`
  y ajusta `LLM_MODEL`.
- **Elegir el GGUF**, si vas por local: en un M2 de 8 GB un 3B en Q4 convive con
  F5-TTS; un 7B no. Si el modelo razona (Qwen3 y parientes), `chat.py` le quita
  el bloque `<think>` antes de mandarlo al TTS, que si no lo leería en voz alta.
- **Perillas de la voz** (variables de entorno, ver `.env.example`):
  `F5_NFE_STEP` baja la latencia a costa de calidad —16 va casi al doble de
  rápido que 32—, `F5_VELOCIDAD` ajusta la cadencia y `F5_DEVICE` fuerza
  `mps`/`cpu` si la autodetección se equivoca.
- **FFmpeg no hace falta**: `torchaudio` 2.11 lee siempre vía `torchcodec`, que
  solo carga con FFmpeg 4–7 y revienta contra el 9 de Homebrew. `voice.py`
  sustituye `torchaudio.load` por `soundfile`, que trae su propia libsndfile.
  El día que torchcodec soporte el FFmpeg instalado, ese parche se borra.
- **F5-TTS se cae en MPS con respuestas de más de un bloque.** El proceso muere
  sin traza en cuanto el texto da para dos trozos, que es cualquier respuesta de
  dos frases. Por eso `F5_DEVICE` es `cpu` por defecto incluso en Apple Silicon,
  aunque MPS vaya 4 veces más rápido en las frases sueltas que sí sobrevive.
- Medido end-to-end en un M2 con la CPU: la síntesis tarda **5 veces el tiempo
  del audio** que produce, así que todo lo que acorta la respuesta acorta la
  espera. Con `F5_NFE_STEP=8` y respuestas de dos frases, la primera suena a los
  18-26 s y termina a los 48; sin esos dos ajustes eran 70 y 85.
- **`max_tokens` está en 90 a propósito.** No es por ahorrar tokens: cada frase
  de más es un bloque más que sintetizar. Va junto con la orden de brevedad en
  `persona.py`, porque ninguna de las dos basta sola: el modelo se pasa igual, y
  cortar sin más deja la frase a medias (`chat.py` la recorta a la última
  completa).
- **El servidor tarda ~40 s en arrancar porque calienta el modelo**, y no es
  opcional: sin hacerle decir una palabra al inicio, la primera respuesta
  costaba **390 s** en vez de 51. Cargar los pesos no basta.
- Las **muletillas** se graban en ese mismo arranque, la primera vez (unos 70 s),
  y se guardan en `assets/voz/muletillas/`. Si cambias la voz de referencia,
  borra esa carpeta o seguirá titubeando con la voz vieja.

## Pendiente

- **Cabello**: resuelto a medias. El cráneo se engrosa hasta la silueta del
  peinado medida en la foto y se colorea aparte, lo que da volumen y un borde
  que sigue el nacimiento del pelo. Pero es un volumen liso: no hay raya, ondas
  ni mechones, y cubre las orejas más de lo que debería.
- **De perfil no funciona.** La textura es proyectiva: sólo es válida para lo
  que la cámara veía. Los vértices que miran hacia atrás muestrean la cara por
  el otro lado, así que de frente se ve bien y girando el modelo no. Para el uso
  previsto —una conversación cara a cara— alcanza, pero es el límite duro de
  este enfoque.
- **Una sola foto**: la textura hereda la iluminación del original, con sus
  sombras horneadas. Por eso la escena se ilumina de forma plana: sumarle un
  esquema de tres puntos duplicaba las sombras.
- **Acabado escultórico**: `rasgos.py` sigue ahí y genera color por vértice con
  cejas y ojos procedurales, para un busto sin fotografía. Ya no se usa por
  defecto.
- **Parecido**: el ajuste monocular sólo observa 51 landmarks frontales, así que
  recupera proporciones, no rasgos finos. Una segunda vista de perfil ayudaría,
  pero no hay ninguna en dominio público con resolución suficiente.
- Parpadeo, y voz a voz con micrófono (whisper.cpp del lado de la escucha).
