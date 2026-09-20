# Despliegue en el cluster de exdev

Una imagen, dos modos según haya o no `LLM_API_KEY`:

- **Con clave** (Groq por defecto): el texto lo escribe un proveedor externo y el
  contenedor sólo carga NeuTTS. **~5 GB de RAM.** Es el modo recomendado.
- **Sin clave**: se levanta el `llama-server` incluido y todo corre sin red.
  **~8 GB de RAM.** La salida si el nodo se queda sin internet.

En ambos, FastAPI sirve el visor, la voz y los visemas.

> [!WARNING]
> **Esta imagen nunca se ha construido.** Se escribió en una máquina sin Docker,
> así que el `Dockerfile` está sin verificar: lo único comprobado es que las
> dependencias resuelven para `linux/amd64` con `torch==2.14.0+cpu` y sin
> paquetes de NVIDIA. Da por hecho que el primer `docker build` va a fallar en
> algo y reserva tiempo para ello. Los puntos más frágiles son la compilación de
> `llama-server` y que `assets/gabriela.glb` exista antes de construir.

## Antes de construir

Dos archivos no están en el repositorio y el build los necesita:

1. **`assets/gabriela.glb`** — el busto. Se genera con el pipeline, que requiere
   descargar FLAME a mano (ver el README principal). Sin él, `docker build` falla
   en el `COPY`, que es mejor que descubrirlo con el contenedor ya desplegado.

   ```bash
   uv run python pipeline/convert_flame.py
   uv run python pipeline/fit_face.py assets/fotos/mistral-1946-frontal.jpg
   uv run python pipeline/export_glb.py
   ```

2. **`assets/voz/referencia.wav` y `referencia.txt`** — la voz de referencia.
   Estos **no** van en la imagen: se montan en tiempo de ejecución, para no
   redistribuir una grabación cuya licencia hay que respetar. El contenedor se
   niega a arrancar sin ellos, con un mensaje que lo dice.

## Construir y publicar

```bash
docker build --platform linux/amd64 -t ghcr.io/exdevutem/gabriela-mistral:0.1.0 .
docker push ghcr.io/exdevutem/gabriela-mistral:0.1.0
```

El build compila `llama-server` desde las fuentes en un stage aparte. Es a
propósito: copiar un binario publicado cruzaría distribuciones, y compilarlo aquí
lo enlaza contra la misma libc que la imagen final. Fija `--build-arg
LLAMA_TAG=bXXXX` para no depender de `master`.

**`GGML_NATIVE=OFF` no se toca.** Con él encendido, el compilador usa las
instrucciones del runner que construye la imagen —AVX-512 en cualquier CI
moderno— y el binario muere con `SIGILL` en los Xeon del cluster, que son más
viejos. Apagado, llama.cpp detecta las extensiones al arrancar.

## Probar en local antes de subirlo

```bash
docker run --rm -p 8000:8000 \
  -e LLM_API_KEY="$LLM_API_KEY" \
  -v gabriela-modelos:/modelos \
  -v "$PWD/assets/voz:/app/assets/voz:ro" \
  ghcr.io/exdevutem/gabriela-mistral:0.1.0
```

El primer arranque descarga 1,5 GB de NeuTTS y su codec (3,5 GB si además
levanta el modelo local) y tarda varios minutos. Los siguientes, unos 45 s, **si
`/modelos` es persistente**.

Los dos repos de la voz son *gated*: el contenedor necesita `-e HF_TOKEN=...` de
una cuenta que haya aceptado sus términos en la web, o el arranque se queda sin
voz.

## En Proxmox 9

Proxmox crea un LXC a partir de la imagen OCI, y eso cambia tres cosas respecto a
Docker:

- **`VOLUME /modelos` se ignora.** Hay que darle un mount point propio en la
  configuración del contenedor, de al menos **6 GB**. Sin él, cada reinicio
  vuelve a bajar los 3,3 GB de modelos.
- **`HEALTHCHECK` se ignora.** Si quieres vigilancia, apunta algo externo a
  `GET /` en el puerto 8000.
- **La voz de referencia** entra como otro mount point sobre `/app/assets/voz`.

Variables que conviene fijar en el contenedor:

| Variable | Por defecto | Para qué |
|---|---|---|
| `LLM_API_KEY` | vacío | Clave del proveedor. Vacía = modelo local. |
| `LLM_MODEL` | `qwen/qwen3.8-27b` | Groq retira modelos cada pocos meses. |
| `LLM_URL` | Groq | Otro proveedor compatible con OpenAI. |
| `NEUTTS_DEVICE` | `cpu` en la imagen, `auto` fuera | `auto` elige mps si lo hay. |
| `NEUTTS_RMS` | `0.09` | Volumen de salida. Súbelo si la sala es ruidosa. |
| `HF_TOKEN` | vacío | **Obligatorio**: la voz vive en repos *gated*. |
| `LLM_CTX` | `2048` | Contexto del LLM. Subirlo cuesta RAM. |
| `LLM_THREADS` | todos los núcleos | Hilos de llama.cpp. |
| `LLM_GGUF` | `Qwen/Qwen2.5-3B-Instruct-GGUF:Q4_K_M` | Otro modelo. |
| `MULETILLAS_DIR` | `/modelos/muletillas` | Dónde se graba el relleno hablado. |
| `FRECUENTES_DIR` | `/modelos/frecuentes` | Dónde se graban las respuestas de los badges. |

## Cuánta RAM y CPU asignarle

### RAM: **4 GB** con Groq, **8 GB** sin él

Medido, no estimado (huella física en un M2, que en Linux es equivalente).
Re-medido el 20 de septiembre de 2026, al cambiar F5-TTS por NeuTTS:

| Componente | En régimen | Pico |
|---|---|---|
| NeuTTS + codec cargados, con Python y FastAPI | 2,10 GB | **2,73 GB** al sintetizar |
| **Con `LLM_API_KEY`** | **~2,1 GB** | **~2,7 GB** |
| `llama-server`, 3B Q4 con `ctx 2048` | ~2,4 GB | ~2,4 GB |
| **Sin clave, todo local** | **~4,5 GB** | **~5,1 GB** |

**Desapareció el pico de arranque.** F5-TTS leía el checkpoint entero antes de
liberar lo que no necesitaba y llegaba a 4,21 GB; NeuTTS no pasa de 2,73 GB, y
su máximo es sintetizando, no cargando. Por eso el modo recomendado baja de 6 GB
a **4 GB**. **8 GB** siguen haciendo falta si levantas el modelo local, que
además mapea 2 GB de GGUF desde disco.

Con menos margen, el sistema mata el proceso de FastAPI a mitad de una síntesis:
ya pasó en la máquina de desarrollo, con 8 GB compartidos con el escritorio.

### CPU: **8 vCPU**, y aun así no será fluido

La síntesis es lo que manda: el texto llega de Groq en menos de un segundo y
luego NeuTTS tarda **1,45 veces el tiempo del audio que produce** (F5-TTS
tardaba 4,08). Se habla por frases, así que lo que cuenta es cuándo empieza a
sonar la primera, no el total.

**Re-medido con NeuTTS** (20 de septiembre de 2026, misma pregunta, dos corridas
contra el WebSocket): la muletilla suena en el acto, la **primera frase de
verdad a los 5,6-9,9 s** y **termina a los 14,8-20,2 s**. Con F5-TTS eran
25-31 s y 35-42 s.

> **La tabla de abajo se midió con F5-TTS.** Se deja como registro de por qué
> el diseño es el que es: hablar por frases y respuestas cortas siguen siendo
> lo que sostiene la latencia.

Medido end-to-end contra el WebSocket, en un M2 con todos los núcleos, sobre la
misma pregunta:

| | Primera palabra | Silencio entre frases | Termina |
|---|---|---|---|
| Como estaba al principio | 70 s | — | 85 s |
| Por frases, `nfe_step=16` | 51 s | 49 s | 120 s |
| Por frases + respuestas cortas, `nfe_step=16` | 47-54 s | 31-34 s | 96-101 s |
| **Lo mismo con `nfe_step=8`** (actual) | **25-31 s** | **ninguno** | **35-42 s** |

Con 8 pasos y respuestas de dos frases, la respuesta suele caber en un solo
bloque, así que además desaparece el silencio intermedio.

**`nfe_step` era la perilla de calidad de F5-TTS y ya no existe.** NeuTTS no
tiene pasos de difusión: genera en un paso autoregresivo y su calidad no se
negocia contra latencia. Lo que sí se hereda es que trozos más largos rinden
mejor (RTF 1,29 en una frase de 10 s contra 1,79 en una de 2 s), así que el tope
de `MAX_BYTES` en `voice.py` cambia latencia inicial por rendimiento total.

Tres cambios acumulados: hablar por frases en vez de esperar la respuesta
entera, ocho pasos de difusión en vez de dieciséis, y un tope de 90 tokens con
la persona ordenando brevedad. Ese último bajó la media de 2-4 bloques por
respuesta a 1,4.

El silencio entre frases no se cierra del todo: mientras la síntesis tarde más
que el audio que produce, la voz no alcanza a la reproducción. Lo tapan una
muletilla grabada que suena al instante y un indicador que dice en qué va.

Esas son cifras de un M2. Los Xeon de un ProLiant con iLO4 son sensiblemente más
lentos por núcleo, así que en el nodo hay que contar bastante más. Lo único que
queda por probar es **una GPU que sirva** (ver abajo): es lo único que cierra el
silencio de verdad.

Estos números salen de un M2; la parte del Xeon es **extrapolación, no medida**.
Para afinarlos hace falta la salida de `lscpu` del nodo y una medición real
contra el contenedor ya desplegado.

### El arranque también cuenta

El contenedor tarda unos **40 s en aceptar visitas**, y el **primer** arranque
sobre un `/modelos` vacío unos 9 minutos, porque además graba las 24 frases de
muletilla y las 68 de las respuestas frecuentes.
Ese calentamiento no es opcional —sin él, la primera respuesta costaba **390 s**
en lugar de 51— así que no lo quites para que el arranque parezca más rápido.

Las muletillas se guardan en `/modelos/muletillas`: es otra razón para que ese
volumen sea persistente. Si cambias la voz de referencia, **bórralas**, o
Gabriela seguirá titubeando con la voz vieja.

### La GTX 750 del `pve0-exdev` no sirve para esto

Tres motivos, cualquiera de ellos bastaría:

- **VRAM.** Una GTX 750 tiene 1 o 2 GB. F5-TTS necesita ~2,75 GB en régimen y
  toca 4,2 GB al cargar. No cabe.
- **Driver.** El nodo usa `nouveau`, que no da CUDA. Haría falta el driver
  propietario, `nvidia-container-toolkit` y passthrough PCI al contenedor.
- **Arquitectura.** Es Maxwell (`sm_50`). PyTorch dejó de compilar para Maxwell
  en las ruedas CUDA 12.8 a partir de la 2.8, y CUDA 13 la elimina del todo: el
  soporte empieza en Turing (`sm_75`).

Por eso la imagen es CPU-only y `torch` viene del índice `+cpu`: así son ~500 MB
en vez de los ~7 GB de las ruedas con CUDA. Si algún día hay una GPU de Turing o
posterior con 6 GB o más, el cambio es el índice de torch, `NEUTTS_DEVICE=cuda` y
compilar llama.cpp con `-DGGML_CUDA=ON`.
