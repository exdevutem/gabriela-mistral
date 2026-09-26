# Briefing — Busto 3D conversacional de Gabriela Mistral

Estado a 8 de septiembre de 2026 · 33 commits en `dev` · ~1.900 líneas

---

## Qué es

Un retrato tridimensional de Gabriela Mistral que responde por escrito, contesta con
voz sintetizada y mueve la cara mientras habla. Backend en Python, render en el
navegador.

**Funciona de punta a punta.** Se le escribe una pregunta, responde en carácter, la
voz suena y la boca acompaña. Verificado en navegador, no solo en tests.

Es una recreación con inteligencia artificial. La interfaz lo dice en pantalla y el
prompt le prohíbe inventar versos y atribuírselos: a un modelo al que se le pide
"recítame un poema tuyo" le sale fabricar estrofas convincentes y falsas, y en un
proyecto universitario sobre una Nobel eso hace más daño que un olvido honesto.

## Cómo se pone en marcha

```bash
uv sync
cp .env.example .env          # y pega tu LLM_API_KEY de Groq
uv run uvicorn gabriela.server:app --port 8000
```

La voz corre siempre en la máquina. El texto sale por Groq —nivel gratuito, sin
tarjeta— o por un `llama-server` local si se prefiere no depender de la red.
Hace falta, eso sí, una voz de referencia en `assets/voz/` — ver *Dependencias
externas*.

- `localhost:8000` — la conversación
- `localhost:8000/debug` — un slider por morph target, para afinar visemas

Reconstruir la cabeza desde cero requiere descargar FLAME (ver *Dependencias
externas*) y ejecutar cuatro pasos:

```bash
uv run python pipeline/convert_flame.py     # .pkl → .npz de numpy puro
uv run python pipeline/fit_face.py assets/fotos/mistral-1946-frontal.jpg --preview
uv run python pipeline/hair_volume.py assets/fotos/mistral-1946-frontal.jpg
uv run python pipeline/export_glb.py        # → assets/gabriela.glb
```

## Arquitectura

Dos mitades que no se tocan en tiempo de ejecución.

**Pipeline offline** — se ejecuta una vez y congela su resultado en un `.glb`:

```
foto → landmarks (mediapipe) → ajuste FLAME (scipy) → volumen de pelo
     → textura proyectiva → GLB con morph targets
```

**Runtime** — FastAPI, WebSocket y three.js:

```
texto → Groq o llama.cpp → F5-TTS español → PCM 24 kHz
                              │
              envolvente RMS ─┴─ visemas del texto
                              │
        {audio, timeline} → three.js → morph targets
```

La clave del labio-sincronizado: los **tiempos** salen de la energía del audio real y
las **formas de boca** del texto. Por separado cada mitad falla —la energía sola da una
mandíbula de marioneta, el texto solo se desfasa en cuanto el TTS respira—. Los
visemas se reparten según energía acumulada y no según tiempo, así que las pausas no
adelantan la boca.

### Mapa de archivos

| Archivo | Qué hace |
|---|---|
| `pipeline/convert_flame.py` | `.pkl` de FLAME → `.npz`, simulando `chumpy` |
| `pipeline/torch_pickle.py` | Lee tensores de PyTorch sin instalar PyTorch |
| `pipeline/landmarks.py` | Foto → 478 landmarks → convención dlib de 68 |
| `pipeline/flame.py` | Evalúa FLAME con skinning de mandíbula |
| `pipeline/fit_face.py` | Ajusta forma y cámara a la fotografía |
| `pipeline/hair_volume.py` | Mide la silueta del peinado y engrosa el cráneo |
| `pipeline/textura.py` | Textura proyectiva desde la foto |
| `pipeline/rasgos.py` | Cejas y ojos procedurales (acabado sin foto) |
| `pipeline/export_glb.py` | Escribe el `.glb` con morph targets y textura |
| `src/gabriela/` | `chat`, `voice`, `visemes`, `persona`, `server` |
| `web/` | Visor three.js y página de ajuste |

## Decisiones, y por qué

Cada número de esta sección salió de medirlo, no de estimarlo.

**La voz en local, el texto donde convenga.** La voz pasó a F5-TTS y ahí se
queda: es lo distintivo del prototipo. El chat habla una API compatible con
OpenAI, y eso deja las dos puertas abiertas con el mismo código —Groq en su nivel
gratuito, o un `llama-server` local— porque el compromiso entre ambas no es
obvio y cambia con el hardware que haya.

**Por defecto, Groq.** Un 3B local cabía en 8 GB pero no seguía la persona:
preguntado "¿quién eres?" respondía con una ficha de enciclopedia en vez de las
dos o tres frases que pide el prompt. Un modelo grande por API lo hace mejor,
responde antes y libera 2,4 GB de RAM en el nodo. A cambio, depende de la red y
de una cuota. Si el GGUF local razona, `chat.py` le corta el bloque `<think>`:
de otro modo el TTS lo leería en voz alta.

**Respuestas cortas por contrato, no por confianza.** `max_tokens` a 90, la
persona ordenando dos frases, y truncado a la última completa. Hacen falta las
tres: el modelo se pasa del límite que le pides por escrito, y cortar sin más
deja la frase colgada, que hablada suena a fallo. Medido, bajó la media de 2-4
bloques por respuesta a **1,4**, y con ella la espera de 135 s a 48. No es por
ahorrar tokens: cada frase de más es un bloque más que sintetizar. Una de ocho
líneas llegó a tumbar el servidor por memoria.

**Voz: clonación, no voz prefabricada.** F5-TTS reproduce el timbre de
`assets/voz/referencia.wav`. Es una libertad que Gemini no daba —se elegía entre
un catálogo— y también una responsabilidad: quien ponga ese audio decide cómo
suena ella, y de dónde salió debe quedar documentado.

**Regularización del ajuste: 10.** Con 2, el error de reproyección baja a 5,3 px pero
el cráneo se aleja un 20 % de la forma media; con 10 el error solo sube a 7,2 px y el
desvío cae al 5 %. Como ningún landmark observa el cráneo, deformarlo le sale gratis
al optimizador: esa ganancia era sobreajuste, no parecido.

**Restricción de silueta.** Los vértices por encima de las cejas no pueden salirse del
contorno de la cabeza medido en la foto. Penalización unilateral: quedarse corto no
es error, solo sobresalir. Bajó el exceso de 27,4 a 8,8 px y el desvío del 4,9 % al
1,7 % sin empeorar el error de landmarks.

**Margen de 15 mm bajo la silueta.** Sin él, el cráneo se ajusta al contorno *del pelo*
y se come el hueco que el pelo debía ocupar: una vista frontal no distingue una cabeza
grande de una cabeza con cabello.

**Visemas medidos, no tanteados.** Los componentes de expresión de FLAME son PCA y no
significan nada por sí mismos, así que se midió su efecto sobre los landmarks de la
boca: `psi[0]` gobierna el ancho (±11 mm) y `psi[3]` la apertura de labios (+6 mm).

**Mapa de letras en vez de fonemas.** El español es fonéticamente casi transparente, y
eso ahorra `espeak`/`phonemizer` y su binario de sistema.

**Sin PyTorch ni chumpy.** El `.pkl` de FLAME envuelve sus arrays en `chumpy`
(abandonada) y el landmark embedding guarda tensores de PyTorch. Ambos se leen con
stubs propios: 2,5 GB de dependencias para una conversión que se hace una sola vez no
se sostiene.

## Lo que costó encontrar

Vale la pena dejarlo por escrito: cada uno de estos se descubrió tarde y ninguno era
evidente desde el código.

**`shapedirs` llegaba vacío.** El stub de `chumpy` descartaba el estado al reconstruir
los objetos, y el array del que depende todo el ajuste salía con forma `(0,)`. Está
en la clave `x` del estado.

**mediapipe 1.x revienta en macOS ARM** con `DrishtiMetalHelper / Service is
unavailable`: fuerza Metal y no hay forma de desactivarlo. El proyecto fija 0.10.35.

**Las fotos históricas son de un solo canal** y mediapipe exige tres. El grafo abortaba
con un error de dimensiones que no delata su causa. Afectaba a cualquier foto de
archivo, que es justo el material de este proyecto.

**Los morph targets necesitaban normales.** Sin ellas la geometría se deforma pero el
sombreado no acompaña: la boca se abre y no se nota.

**Una foto de "1940" resultó ser un cuadro.** 3.217 × 4.879 px del Arquivo Nacional de
Brasil, descrita en su ficha como fotografía. Es un retrato pintado con aerógrafo, con
turbante y geometría idealizada por el artista. Se detectó mirándola, no leyendo la
ficha.

**Doble sombreado con la textura.** La foto trae sus sombras horneadas y el esquema de
tres luces las duplicaba, ennegreciendo medio rostro. La escena se ilumina ahora de
forma plana.

**Caché del navegador.** El servidor entregaba el `avatar.js` nuevo y el navegador
ejecutaba el viejo. Resuelto con `Cache-Control: no-store`.

## Límites actuales

**De perfil no funciona.** La textura es proyectiva: solo vale para lo que la cámara
veía. Los vértices que miran hacia atrás muestrean la cara por el otro lado. Para una
conversación cara a cara alcanza; si se quiere permitir orbitar el modelo, se rompe.

**La latencia la marca el TTS**, y sigue siendo **el problema abierto del
proyecto**. El texto llega de Groq en menos de un segundo; la síntesis tarda
cinco veces el tiempo del audio que produce. En los Xeon del cluster será peor
que en las cifras de aquí abajo, todas medidas en un M2.

**Se habla por frases.** El servidor trocea la respuesta por fin de frase, con
un tope de 135 bytes, y manda cada trozo en cuanto está listo. Lo que importa no
es el total, sino cuándo empieza a sonar.

**El 20 de septiembre de 2026 F5-TTS fue reemplazado por NeuTTS nano-spanish**:
2,8 veces más rápido en banco (RTF 4,08 → 1,45) con la misma voz de referencia.
Medido contra el WebSocket, en dos corridas sobre la misma pregunta, la muletilla
suena en el acto, la **primera frase de verdad a los 5,6-9,9 s** y **termina a
los 14,8-20,2 s**. Con F5-TTS eran 25-31 s y 35-42 s.

> **La tabla y las cifras que siguen se midieron con F5-TTS**, y `nfe_step` ya
> no existe como perilla. Se dejan como registro de por qué el diseño es el que
> es: trocear por frases y acortar las respuestas siguen siendo lo que sostiene
> la latencia.


| | Primera palabra | Silencio entre frases | Termina |
|---|---|---|---|
| Sin trocear | 70 s | — | 85 s |
| Por frases, `nfe_step=16` | 51 s | 49 s | 120 s |
| Por frases + respuestas cortas, `nfe_step=16` | 47-54 s | 31-34 s | 96-101 s |
| **Lo mismo con `nfe_step=8`** (actual) | **25-31 s** | ninguno | **35-42 s** |

**`nfe_step` acabó en 8, y la medición decía 16.** Vale la pena dejar escrito el
desacuerdo. Generando la misma frase con la misma semilla y comparando contra
`nfe=64`:

| `nfe` | corr. envolvente | dif. timbre | coste |
|---|---|---|---|
| 8 | 0,225 | 5,08 dB | 16 s |
| 12 | 0,296 | 3,91 dB | 24 s |
| **16** | **0,974** | **1,89 dB** | **33 s** |
| 20 | 0,606 | 3,23 dB | 39 s |
| 32 | 0,976 | 1,58 dB | 64 s |
| 64 | 1,000 | — | 125 s |

16, 32 y 64 coinciden entre sí; 8, 12 y 20 no. La curva no es monótona porque el
muestreador cae en trayectorias distintas según cómo se discretice, no porque
"menos pasos" sea "peor" de forma proporcional. Con 8 pasos no sale la misma voz
algo degradada: sale otra interpretación, con las sílabas en otros sitios.

Por esos números, 16 era la elección segura: el valor más barato que reproduce
el resultado de los altos. **Pero al escuchar las muestras, la de 8 pasos suena
bien**, y quita la mitad de la espera. Manda el oído.

**La lección, para la próxima vez que haya que elegir un parámetro perceptual:**
la métrica medía consistencia con el modelo convergido, no calidad. Que 8 pasos
produzcan otra interpretación no implica que esa interpretación sea peor: la
energía en altas frecuencias es prácticamente idéntica en 8, 16 y 32 (0,90 %,
0,96 % y 1,00 %), así que no añade aspereza ni ruido; solo dice las cosas de otra
manera. Lo que la medición sí sirvió para descartar fue 12 y 20, que divergen
más que 8 sin ser más baratos.

El silencio entre frases es irreducible mientras la síntesis tarde más que el
audio que produce: la voz nunca alcanza a la reproducción. Con 8 pasos son 8 s,
que se leen como una pausa de quien piensa; con 16 son 49 s, que se leen como
que se colgó.

Las otras perillas, todas medidas: acortar la referencia de 11,8 s a 7,6 s
ahorra un 26 % —F5-TTS genera la referencia y la frase juntas, así que su largo
se paga en cada síntesis—, y `max_tokens` acorta la respuesta.

**Lo grabado se puede permitir lo que lo hablado no.** NeuTTS no tiene una
perilla equivalente a `nfe_step`: genera en un paso autorregresivo, y no falla
por configuración sino por muestreo. Con la misma frase y distinta semilla, una
toma sale con su voz y otra sale con voz de hombre, divagando o cortada a media
palabra —el modo de fallar de cualquier modelo que clona a partir de una
referencia—. En vivo hay que quedarse con la primera toma. Las muletillas y las
respuestas frecuentes no: nadie está esperando, se graban una sola vez y viven
en disco.

Así que ahí se compra calidad con tiempo, que es lo que `nfe_step` compraba:
cada frase se sintetiza hasta seis veces, con semilla distinta y algo más fría
(temperatura 0,7 en vez de 1,0), y se guarda la mejor. Para en cuanto una toma
es lo bastante buena, así que las que salen bien a la primera —la mayoría—
siguen costando una.

Elegir «la mejor» sin oírla se hace con dos medidas, y sólo dos, porque son las
que delatan las tomas malas que se vieron:

- **El tono.** Se compara la fundamental mediana de la toma con la de
  `referencia.wav`, en semitonos. Tres semitonos siguen siendo ella; doce es
  otra persona. Es lo que caza la voz de hombre.
- **El largo.** A trece caracteres por segundo se sabe cuánto debería durar el
  texto. La mitad es una frase cortada; el doble es el modelo divagando.

Lo que esto **no** caza es una toma que diga otra cosa con su voz y en el tiempo
justo; para eso haría falta un reconocedor. La nota de cada archivo queda en un
`notas.json` al lado, que sirve para dos cosas: no volver a revisar en cada
arranque lo ya revisado, y que una frase que sale mal las seis veces no se
repita eternamente —se queda la menos mala, con un aviso en el log para que
alguien la oiga y la reescriba—. Un archivo sin nota es de antes de que esto
existiera: se puntúa una vez y se rehace sólo si está mal, que es como se
arreglan solas las tomas viejas que ya están en el volumen del museo.

**Mientras espera, habla.** Treinta segundos de busto inmóvil no se leen como
"está pensando" sino como "se colgó". Dos cosas lo tapan: una muletilla grabada
—«Déjame pensar.», «Mmm. Espera un momento.»— que suena a 1,5 s de la pregunta,
y un indicador que dice en qué va (*preparando la voz… 2 de 4*) en vez de girar
sin fondo. Las muletillas se graban una vez al arrancar, con la misma voz de
referencia, y reutilizan la misma cola de reproducción que las frases reales:
no hubo que añadir camino nuevo, solo encolar antes.

Verificado en navegador: la muletilla suena a 1,5 s, el indicador aparece a los
2 s, ninguna frase se solapa con otra.

**El audio hay que desbloquearlo con el gesto del usuario.** Safari solo deja
sonar el audio que se arrancó dentro de un gesto reciente, y aquí la voz llega
25-31 s después del clic: para entonces el permiso caducó. Se reproducía la
muletilla —que llega en 1,5 s, dentro de la ventana— y se bloqueaba todo lo
demás, con la boca quieta. El permiso se ata al *elemento* `<audio>`, así que el
visor usa uno solo para toda la sesión y lo estrena con 1 ms de silencio al
enviar la pregunta; las frases posteriores heredan el permiso.

Reproducido y verificado con el WebKit de Playwright, que es el motor de Safari.
En Chromium nunca falló, y ahí estuvo la trampa: la primera verificación se hizo
con `--autoplay-policy=no-user-gesture-required`, que desactiva justo la política
que rompía el producto.

**Calentar el modelo al arrancar no es una optimización, es un requisito.**
Cargar los pesos no basta: la primera inferencia real paga además la preparación
del audio de referencia y la puesta en marcha de los kernels de torch. Medido:
**390 s** la primera respuesta frente a 51 las siguientes. El servidor le hace
decir una palabra antes de aceptar visitas, y por eso tarda ~40 s en levantar.

**MPS no se puede usar.** Con F5-TTS, en Apple Silicon la síntesis iba 4 veces
más rápida (11,6 s frente a 44 s), pero el proceso moría sin traza en cuanto el
texto pasaba de un bloque —cualquier respuesta de dos frases—. Por eso
`NEUTTS_DEVICE` es `cpu` por defecto: un valor rápido que tumba el servidor no
es un valor. Con NeuTTS, que reemplazó a F5 el 20 de septiembre de 2026, **esto
no se ha vuelto a medir**: puede que MPS ya sirva.

**Una sola vista.** El ajuste monocular recupera proporciones, no profundidad. No hay
ninguna foto de perfil suya en dominio público con resolución suficiente; la mejor
candidata (Biblioteca Nacional Digital, ca. 1952) tiene licencia ambigua.

**El peinado es volumen liso.** Sin raya, ondas ni mechones, y cubre las orejas más de
lo que debería.

**Sin memoria entre sesiones.** Cada conexión abre su propia conversación.

## Próximos pasos

En orden de rendimiento por esfuerzo.

**0. Voz propia en Piper (en curso).** Ataca el problema abierto, la latencia
del TTS: Piper corre más rápido que tiempo real en CPU. Se afina con F5-TTS de
maestra —`pipeline/dataset_piper.py` genera ~2 h de audio en un M5, y
`pipeline/entrenar_piper.ipynb` entrena en la T4 gratuita de Colab—. Hecho: el
dataset, en marcha el 26 de septiembre de 2026. **Sin verificar**: el notebook
no se ha corrido, y está por oír si la copia suena lo bastante bien. Las
frecuentes ya se graban con F5 (`pipeline/grabar_frecuentes_f5.py`): la nota media
subió de 0,78 a 0,91. Ojo con la licencia: la voz de Piper sale del audio de
F5-Spanish (CC BY-NC 4.0), así que conviene tratarla también como no comercial.

**1. Voz a voz con micrófono.** Elimina el teclado y hace la interacción
presencial: es el paso que más cambia la experiencia. Dos caminos, y conviene
medir antes de elegir: whisper.cpp en local, que no depende de la red, o
`whisper-large-v3-turbo` en Groq, que ya está en el nivel gratuito de la misma
clave que usa el chat.

**2. Construir y probar la imagen Docker.** Está escrita pero **nunca se ha
construido**: no había Docker en la máquina de desarrollo. Antes de contar con
ella hay que levantarla una vez y ver que arranca.

**3. Parpadeo.** Falta el único gesto involuntario que el modelo no tiene, y es de los
que más separan un rostro vivo de una máscara. FLAME lo permite con un morph target
de párpados.

**4. Segunda vista para el ajuste.** Requiere aclarar la licencia del perfil ca. 1952
con la Biblioteca Nacional, o encontrar otro en dominio público. Resolvería a la vez
la profundidad y el volumen trasero del pelo.

**5. Esculpir el peinado en Blender.** El `.glb` ya se abre ahí. Es trabajo manual y no
reproducible por script, pero es lo único que dará raya y ondas reales.

**6. Textura multivista.** Si aparece el perfil, combinar ambas proyecciones con
mezcla por ángulo resolvería el límite de "solo de frente".

## Dependencias externas

**FLAME** — `flame.is.tue.mpg.de`, requiere registro. Licencia de investigación y uso
no comercial, compatible con un proyecto universitario, pero no automatizable. Hacen
falta tres archivos en `assets/flame/`:

- `flame2023.pkl` — el modelo
- `landmark_embedding.npy` — qué vértices son los 68 landmarks
- `FLAME_masks.pkl` — regiones semánticas; sin él, el cuero cabelludo se deduce por
  altura y el peinado sale con borde recto

**Modelo de landmarks de mediapipe** — se descarga aparte, la orden está en el README.

**Checkpoint de F5-Spanish** (`jpgallegoar/F5-Spanish`, ~1,3 GB, CC BY-NC 4.0) — se
descarga solo a la caché de Hugging Face. Licencia no comercial, igual que FLAME:
encaja con un proyecto universitario.

**Voz de referencia** — `assets/voz/referencia.wav` (7–10 s de habla limpia) y
`referencia.txt` con su transcripción exacta. No se versiona; su procedencia va
documentada junto a ella. **Trampa**: F5-TTS recorta el audio a 12 s pero usa el
texto entero, así que un `.wav` largo con su transcripción completa hace que el
modelo crea que cientos de caracteres caben en 12 s, y la voz sale atropellada.
Recortar a mano y transcribir sólo el trozo; el cómo está en `assets/voz/README.md`.

**Clave de Groq** — gratuita y sin tarjeta en console.groq.com/keys, en `.env`
como `LLM_API_KEY`. Groq retira modelos cada pocos meses, así que `LLM_MODEL` es
configurable y `chat.py` distingue el 404 de "ese modelo ya no existe" del resto.

**Un GGUF instruct**, sólo en modo local — lo descarga `llama-server -hf …`.

## Fotografías y atribución

Todas de dominio público, verificadas una a una en Wikimedia Commons. **La ley chilena
mantiene los derechos morales aun en dominio público: la atribución es obligatoria**
allí donde se muestre el modelo.

| Archivo | Año | Autor | Uso |
|---|---|---|---|
| `mistral-1946-frontal.jpg` | 1946 | Marcos Chamúdez | Referencia del ajuste y textura |
| `mistral-1948-santabarbara.jpg` | ca. 1948 | George F. Weld | Referencia visual del peinado |
| `gabriela-referencia.png` | ca. 1945 | Anna Riwkin-Brick | Primera referencia, ya no se usa |

El detalle completo, incluidas las candidatas descartadas y el motivo de cada
descarte, está en `assets/fotos/PROCEDENCIA.md`.

Una nota metodológica que ahorra trabajo: al elegir otra foto, **mide los píxeles
útiles de cara, no la resolución del archivo**. Un escaneo de 4.000 px donde ella sale
de cuerpo entero aporta menos que uno de 1.700 px encuadrado en el rostro. Y no
mezcles épocas: FLAME ajusta forma facial, y promediar una cara de 33 años con una de
57 da un rostro que no es ninguna de las dos.

## Entorno de referencia

Desarrollado y verificado en un Mac M2 con 8 GB de RAM y Python 3.13. Esa restricción
dio forma al diseño: nada de modelos neurales en tiempo real ni LLM local. El único
paso pesado es el ajuste facial, corre offline una vez y su resultado se congela.

## Verificación

```bash
uv run python tests/test_visemes.py     # lógica de visemas, sin red
uv run python pipeline/flame.py         # autocomprobación del skinning
```

`test_visemes.py` cubre la única parte con ramas que puede romperse en silencio: si el
labio-sincronizado se desfasa, el avatar sigue funcionando y solo se ve raro.
`flame.py` verifica que con pose cero ningún vértice se mueve, que al rotar la
mandíbula baja la barbilla y que el cráneo queda quieto.
