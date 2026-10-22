# Gabriela Mistral — busto 3D conversacional

Un retrato escultórico en 3D que contesta preguntas sobre ella y sobre el museo,
con su voz y moviendo la cara. Backend en Python, render en el navegador.

> Es una recreación con inteligencia artificial, no una grabación ni un testimonio
> real de Gabriela Mistral (1889–1957). La interfaz lo indica en pantalla.

## Cómo funciona

**No hay chat.** Ella sólo dice respuestas escritas y verificadas de antemano
—las preguntas frecuentes de `config.py`—, grabadas con su voz al arrancar. Lo
que el visitante escribe o le dice al micrófono se compara con esas preguntas y
suena la respuesta que más se le parezca:

```
micrófono ──> Whisper (local) ──┐
                                ├──> embeddings (local) ──> respuesta grabada
texto o badge ──────────────────┘          │                  │
                                    si nada se parece:        │
                                    «No te entendí bien…»     │
                                                              ▼
                        envolvente RMS ─┬─ visemas del texto ─> three.js
```

- **Whisper** (`openai/whisper-small`) sólo transcribe. Elegir la respuesta lo
  hace un modelo de embeddings multilingüe
  (`paraphrase-multilingual-MiniLM-L12-v2`) que compara el sentido de lo dicho
  con cada pregunta, con su respuesta y con sus `VARIANTES`. Si nada pasa de
  `UMBRAL` (0,5), dice que no entendió y señala los badges.
- **Por qué así**: en un museo una fecha inventada es una fecha que el visitante
  se lleva a casa creyendo que se la dijo ella. Con respuestas grabadas no hay
  forma de que invente nada, suenan al instante y no hace falta ni red ni
  clave. Se probó primero con Gemini (texto y voz), pero el nivel gratuito del
  TTS da 3 peticiones por minuto y 10 al día.
- **Si contesta otra cosa**: el log muestra qué entendió y qué eligió
  (`gabriela.entender: «…» -> ¿…? (0.72)`). El arreglo es agregar esa forma
  de preguntar a `VARIANTES`.
- **El micrófono** sólo funciona con la página en `localhost` o `https`: el
  navegador no lo ofrece en `http` desde otra máquina. Sin él quedan el texto y
  los badges.
- **Mientras habla el visitante** se ve lo que Whisper va entendiendo: el visor
  manda el audio en trozos de ¼ s y el servidor devuelve una transcripción
  parcial por pasada (~1 s cada una). La respuesta sale de la transcripción
  final, al soltar el botón.
- **Cada respuesta llega en un solo audio**, con un reproductor debajo del
  texto: pausa y reanudar, detener, y una barra con el tiempo que se puede
  arrastrar. Preguntar otra cosa o tocar el micrófono corta la que suena.

El labio-sincronizado combina dos fuentes: los **tiempos** salen de la energía del
audio real y las **formas de boca** del texto. Los visemas se reparten según
energía acumulada, no según tiempo, así que las pausas del TTS no desfasan la boca.

## Requisitos

Todo corre en local; no hace falta ninguna clave de API.

- Python 3.13 y [uv](https://docs.astral.sh/uv/)
- Una voz de referencia en `assets/voz/` para NeuTTS (ver más abajo), y en macOS
  `brew install espeak-ng`
- ~1,5 GB de disco para Whisper y el modelo de embeddings, que se descargan
  solos de Hugging Face la primera vez

```bash
uv sync
cp .env.example .env    # y pega tu HF_TOKEN
```

### La voz en local (NeuTTS)

NeuTTS no tiene voces prefabricadas: **clona** la que le des. Deja en `assets/voz/`
un `referencia.wav` de 7–10 s de habla limpia y un `referencia.txt` con su
transcripción exacta.

**Ojo con la duración**: NeuTTS pide entre 3 y 15 s, y el `.txt` tiene que
transcribir exactamente lo que suena en el `.wav`. Cómo recortar, en
`assets/voz/README.md`.

El modelo ([`neuphonic/neutts-nano-spanish`](https://huggingface.co/neuphonic/neutts-nano-spanish)
y su codec [`neucodec`](https://huggingface.co/neuphonic/neucodec), ~1,5 GB) se
descarga solo la primera vez a la caché de Hugging Face. **Los dos son repos
*gated***: hay que aceptar sus términos una vez en la web y tener `HF_TOKEN` en
el entorno, o la descarga responde 403. Su licencia es la «NeuTTS Open License
1.0» — léela antes de publicar el proyecto.

### La voz en vivo (GPU)

Lo que no está entre las frecuentes —y mañana, lo que escriba el modelo
conversacional— se sintetiza al momento con `src/gabriela/vivo.py`: la misma
NeuTTS y la misma `referencia.wav`, pero con el backbone en **GGUF sobre
llama.cpp**, que usa la GPU (Metal en un Mac) o la CPU con kernels propios. Medido
en un M5 con Metal, tras calentar: **factor de tiempo real 0,33** y el primer
sonido de una respuesta de dos frases a los **1,4 s**. Con el backbone de
transformers en MPS el factor era 1,08 (medido en un M2, ver `config.py`), es
decir, no alcanzaba a hablar en vivo.

El texto se parte en unidades (la primera, en su primera coma si es larga) y se
manda una `habla` por unidad según se sintetiza; el visor las encadena, así que
suena la primera mientras se hacen las demás. Cada toma se puntúa con el mismo
juez que las grabadas y, si suena a otra voz, se repite más fría
(`VIVO_INTENTOS`). Probada con una frase de 11 s: Whisper la transcribe tal cual y
el tono cae a 8 Hz del de la referencia (nota 0,95).

```bash
brew install cmake                       # llama-cpp-python compila llama.cpp
uv sync
uv run pipeline/convertir_gguf.py        # una vez: crea assets/models/*.gguf (467 MB)
uv run python -m gabriela.vivo "Hola, soy Lucila Godoy Alcayaga."   # mide la latencia
```

En la página, `/decir <texto>` en la caja de mensaje lo dice con la voz en vivo;
por WebSocket es `{"tipo": "decir", "texto": "…"}`.

`convertir_gguf.py` existe porque el GGUF oficial
(`neuphonic/neutts-nano-spanish-q4-gguf`) es *gated* aparte del modelo base.
Con ese acceso, `VIVO_GGUF=neuphonic/neutts-nano-spanish-q4-gguf` lo reemplaza y
queda más pequeño y más rápido. **Sin probar todavía en el clúster**: la imagen
de Docker no incluye `llama-cpp-python`, y sin él el servidor arranca igual pero
`decir` responde con error.

## Puesta en marcha

```bash
uv run uvicorn gabriela.server:app --port 8000
```

El primer arranque graba las respuestas que falten con NeuTTS (unos siete
minutos si no hay ninguna) y carga Whisper. Los siguientes sólo cargan Whisper:
si todo está grabado, NeuTTS ni se abre.

- `http://localhost:8000` — la conversación
- `http://localhost:8000/debug` — un slider por morph target, para afinar visemas

## Despliegue

Hay un `Dockerfile` con las dos mitades en una imagen, pensado para publicarla en
`ghcr.io` y levantarla como contenedor en Proxmox. El build necesita el `.glb` ya
generado y la voz de referencia se monta en vez de hornearse. De construirla y
subirla se encarga GitHub Actions al empujar una etiqueta `v*`, o a mano desde la
pestaña *Actions*. Requisitos, flujo, y cuánta RAM y CPU pedirle al nodo:
[`docker/README.md`](docker/README.md).

Sin chat ni síntesis en vivo, la latencia de la voz dejó de ser problema: lo
que cuesta CPU es Whisper transcribiendo, unos segundos por pregunta hablada.
Las cifras de RAM de `docker/README.md` se midieron con NeuTTS en vivo y están
pendientes de re-medir.

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
uv run python pipeline/fit_face.py assets/fotos/mistral-1946-frontal.jpg \
    assets/fotos/mistral-1946-sanfrancisco.jpg assets/fotos/mistral-1946-truman.jpg \
    assets/fotos/mistral-1946-microfono.jpg \
    --perfiles assets/fotos/perfiles.json --preview
uv run python pipeline/hair_volume.py assets/fotos/mistral-1946-frontal.jpg
```

Las fotos comparten una sola forma β; cada una aporta su cámara y su expresión,
para que el ceño de una o la boca apretada de otra no se cuelen en la forma. La
primera, frontal, es la referencia: de ella salen la textura y la silueta del
cráneo. El perfil de `perfiles.json` va anotado a mano —mediapipe no ve caras de
perfil— y aporta la silueta de frente, nariz, labios y mentón (`pipeline/perfil.py`).
`hair_volume.py` se vuelve a correr porque guarda vértices absolutos, calculados
con la forma anterior.

El `--preview` deja en `/tmp/ajuste*.png` la comparación entre los landmarks de
cada foto (verde) y los del modelo ajustado (magenta), y en
`/tmp/ajuste-perfil-*.png` el borde medido del perfil contra la malla. **Míralos
antes de seguir**: si no se reconoce, sube `REGULARIZACION` en `fit_face.py` o
consigue una foto mejor.

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
uv run python tests/test_local.py                            # elector, audio y calidad de tomas
uv run python pipeline/landmarks.py assets/fotos/mistral-1946-frontal.jpg
uv run python -m gabriela.entender "dónde naciste" "hola"    # qué respuesta elige
uv run python -m gabriela.voice "Hola" --out /tmp/g.wav      # solo voz
uv run python -m gabriela.voice "Hola" --grabado             # como lo pre-grabado
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
- **PyTorch sí llega, pero por la voz**: NeuTTS lo arrastra (~2,5 GB). El
  pipeline de la cabeza sigue sin tocarlo —lee FLAME con stubs propios—, así que
  quien solo construya el modelo no necesita instalarlo.
- **Perillas de la voz** (variables de entorno, ver `.env.example`):
  `NEUTTS_RMS` sube o baja el volumen de salida —NeuTTS entrega más bajo que el
  F5 anterior y `voice.py` lo normaliza—, y `NEUTTS_DEVICE` fuerza `mps`/`cpu`.
  Cadencia ya no tiene perilla: el ritmo sale del audio de referencia.
- **En macOS hace falta `brew install espeak-ng`**: el espeak que empaqueta
  `neutts` lleva compilada la ruta del directorio temporal donde se construyó y
  muere con «Error processing file .../phontab». `voice.py` lo cambia por el de
  Homebrew después de importar neutts; el día que publiquen una rueda con la
  ruta correcta, ese parche se borra.
- **Safari exige un gesto reciente para reproducir audio**, y la voz llega
  después del clic (tras transcribir, si se habló). El permiso va atado al elemento `<audio>`, no a la
  página, así que el visor reutiliza **uno solo** para toda la sesión y lo
  desbloquea con 1 ms de silencio al enviar la pregunta. Por eso `avatar.js`
  tiene un `audio` único en vez de crear uno por frase: no es un detalle de
  estilo, es lo que hace que suene en Safari.
- **`NEUTTS_DEVICE` es `auto`**: MPS si la máquina lo tiene, CPU si no. En un
  M2, tras calentar, MPS da un factor de tiempo real de 1,08 contra 1,37 en CPU
  y codifica la referencia en 1,4 s en vez de 11,0 s, a cambio de 35 s más de
  arranque. F5-TTS moría sin traza en MPS; NeuTTS aguantó 27 síntesis seguidas
  y tres conversaciones enteras por WebSocket —que es donde sintetiza desde
  otro hilo—, sin un solo fallo. La imagen de Docker lo fija en `cpu`.
- Medido en un M2 con la CPU (20 de septiembre de 2026): la síntesis tarda
  **1,45 veces el tiempo del audio** que produce —con F5-TTS eran 4,08—, así que
  todo lo que acorta la respuesta sigue acortando la espera. Una respuesta de
  10 s tarda 13 s en salir; antes eran 30. Las cifras end-to-end de esta lista
  se midieron con F5 y **están pendientes de re-medir**.
- Las **respuestas frecuentes** y el «no te entendí» se graban en el
  arranque, la primera vez —unas 90 frases, siete minutos—, en
  `assets/voz/frecuentes/` y `assets/voz/respaldo/`, un archivo por frase. Si
  cambias la voz de referencia, el modelo o el device, borra esas carpetas o
  seguirá hablando con la voz vieja.
- **Lo grabado se repite hasta que sale bien.** NeuTTS falla por muestreo: la
  misma frase sale con su voz o con voz de hombre según la semilla. Cada frase
  grabada se sintetiza hasta
  `NEUTTS_INTENTOS` veces (6) con semilla distinta y se guarda la mejor, medida
  por tono contra `referencia.wav` y por duración contra el texto. Es lo que
  reemplaza al `nfe_step` de F5-TTS: comprar calidad con tiempo, sólo que aquí
  el tiempo se paga una vez y en el arranque, no en cada respuesta.
  Para en cuanto una toma es buena, así que la mayoría de las frases siguen
  costando una sola síntesis. La nota de cada archivo queda en un `notas.json`
  al lado; para que se revise todo otra vez, bórralo. Si una frase no llega en
  seis intentos, queda la menos mala y el log lo dice: óyela y reescríbela.

## Pendiente

- **Cabello**: resuelto a medias. El cráneo se engrosa hasta la silueta del
  peinado medida en la foto y se colorea aparte, lo que da volumen y un borde
  que sigue el nacimiento del pelo. Pero es un volumen liso: no hay raya, ondas
  ni mechones, y cubre las orejas más de lo que debería.
- **De perfil, la foto se acaba.** La textura se hornea en un mapa cilíndrico
  (`pipeline/textura.py`): lo que la foto frontal ve sale de la foto, y la nuca,
  la coronilla y los costados se rellenan con el gris del pelo o el tono de la
  piel que los rodea. Girando el modelo ya no aparece la cara reflejada en la
  nuca, pero detrás de la oreja el pelo pasa de fotografía a tono liso.
- **Una sola foto**: la textura hereda la iluminación del original, con sus
  sombras horneadas. Por eso la escena se ilumina de forma plana: sumarle un
  esquema de tres puntos duplicaba las sombras.
- **Acabado escultórico**: `rasgos.py` sigue ahí y genera color por vértice con
  cejas y ojos procedurales, para un busto sin fotografía. Ya no se usa por
  defecto.
- **Parecido**: el ajuste monocular sólo observa 51 landmarks frontales, así que
  recupera proporciones, no rasgos finos. Una segunda vista de perfil ayudaría,
  pero no hay ninguna en dominio público con resolución suficiente.
- Parpadeo.
- **Voz propia en Piper**: el dataset está hecho (1.566 frases, 93 min), el
  entrenamiento corre en Colab y `VOZ_MOTOR=piper` la usa para grabar. Sin
  síntesis en vivo ya no hace falta por latencia; queda como alternativa a
  NeuTTS. Ver `assets/voz/README.md`.
