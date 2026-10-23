# Voz de referencia

NeuTTS no tiene voces prefabricadas: clona la que le des. Estos dos archivos
deciden cómo suena Gabriela.

- `referencia.wav` — habla limpia, sin música ni ruido de fondo.
- `referencia.txt` — la transcripción **exacta** de ese audio.

## La regla que hay que respetar: 3–15 segundos

NeuTTS pide una referencia de entre 3 y 15 s, mono, sin ruido de fondo. Audio y
texto tienen que corresponderse **exactamente**: el `.txt` es la transcripción
de lo que suena en el `.wav`, ni una palabra más.

(F5-TTS, el modelo anterior, recortaba a 12 s por dentro pero leía el texto
entero, y descuadrar ambos le salía como voz atropellada. NeuTTS no recorta,
pero el par sigue teniendo que cuadrar.)

El actual son los primeros 7,58 s de `referencia-completa.wav`, cortados donde
termina «Soy Lucila Godoy Alcayaga». Para rehacerlo desde el original:

```python
from pydub import AudioSegment, silence
a = AudioSegment.from_file("assets/voz/referencia-completa.wav")
print(silence.detect_nonsilent(a, min_silence_len=300, silence_thresh=a.dBFS - 16))
a[:7580].set_channels(1).set_frame_rate(24000).export("assets/voz/referencia.wav", format="wav")
```

Para transcribir el recorte sin escribirlo a mano hace falta un Whisper; ya no
viene con el paquete de voz (`f5-tts` lo traía, `neutts` no).

## Duración y latencia

La referencia se codifica **una sola vez** al arrancar (11,8 s medidos) y se
cachea: a diferencia de F5-TTS, su largo ya no se paga en cada frase.

Medido en un M2 de 8 GB, con esta misma referencia de 7,58 s (20 de septiembre
de 2026):

| frase                     | audio  | F5-TTS `nfe=8` | NeuTTS |
|---------------------------|--------|----------------|--------|
| «Déjame pensar.»          |  1,2 s |         10,7 s |  4,4 s |
| «Mmm. Espera un momento.» |  1,9 s |         13,7 s |  4,6 s |
| una respuesta de 10 s     | 10,2 s |         30,0 s | 13,1 s |
| **factor de tiempo real** |        |       **4,08** | **1,45** |

Arranque completo (`calentar()`): **43,4 s** con las muletillas ya grabadas.
Grabarlas la primera vez son 174 s más, una sola vez.

NeuTTS habla más pausado que F5 en frases cortas —la misma muletilla le dura el
doble— y no tiene perilla de velocidad: el ritmo sale de la referencia.

## La referencia también es la vara de medir

Del `referencia.wav` sale además el **tono** contra el que se juzga cada frase
grabada: su fundamental mediana. Una toma que se aleje mucho de ella no es ella
—es el modelo que se fue a otro hablante, y suele sonar a hombre— y se repite
con otra semilla hasta que salga bien. Por eso, si cambias la referencia, la
vara cambia con ella y lo grabado con la voz anterior deja de cuadrar: **borra
`muletillas/` y `frecuentes/` enteras**, `notas.json` incluido.

Si el `.wav` no es PCM de 16 bits no se puede medir, y entonces las tomas se
juzgan sólo por su duración: queda dicho en el log al arrancar.

## Muletillas

`muletillas/` es lo que dice mientras piensa, para que la espera no empiece en
silencio. Ya no son titubeos sueltos: cada una cuenta algo del museo —dónde
está parado el visitante, qué fue este edificio, qué se guarda aquí— durante
unos 20 s. Las graba el servidor al arrancar, una sola vez, con esta misma voz
de referencia.

Se guarda **un archivo por frase** (`0-0.wav`, `0-1.wav`, …) porque el servidor
las manda troceadas: el visor descarta las que no alcanzaron a sonar en cuanto
llega la respuesta. Por eso el largo no se paga. Una muletilla a la que le
falte un trozo no se usa: se cortaría a mitad de la historia.

Junto a los `.wav` queda un `notas.json` con lo que puntuó cada uno, de 0 a 1.
Sirve para no revisar en cada arranque lo ya revisado y para no insistir sin fin
con una frase que sale mal; bórralo y todo se revisa otra vez. Si alguna frase
tiene una nota baja, óyela: el modelo no fue capaz de decirla bien ni en seis
intentos y probablemente haya que reescribirla más corta o más simple.

**Las frases tienen que ser cortas**, bajo `MAX_BYTES_MULETILLA` (70 bytes, unos
5 s). El visor deja terminar la frase que está sonando, así que ésa es la espera
máxima que el relleno le añade a la respuesta. Hay un test que lo comprueba.

**Si cambias la referencia, el modelo o el device, borra esa carpeta**: si no,
seguirá titubeando con la voz anterior y cambiando de timbre a mitad de
respuesta. El texto está en `MULETILLAS`, en `config.py`.

## Preguntas frecuentes

`frecuentes/` son las respuestas a los badges de la página, grabadas igual que
las muletillas y con el mismo formato (`0-0.wav`, `0-1.wav`, …). Están en
`FRECUENTES`, en `config.py`, como pares de pregunta y respuesta.

Van grabadas y no las escribe el modelo por una razón concreta: un visitante que
toca «¿desde cuándo existe el museo?» tiene que oír 1941, siempre, y no la fecha
que el modelo recuerde ese día. Además responden en 0,04 s. Sólo se ofrecen como
badge las que están **completas**: si a una le falta un trozo, la pregunta cae en
el LLM, que al menos contesta entera.

### Grabarlas con F5-TTS

Las frecuentes suenan solas —ninguna respuesta en vivo las sigue—, así que no
tienen por qué salir del modelo rápido. `pipeline/grabar_frecuentes_f5.py` las
graba con F5-TTS a `nfe_step=32`, offline y con hasta diez tomas por frase,
en una máquina con memoria de sobra (probado en un M5 de 24 GB, en MPS):

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run pipeline/grabar_frecuentes_f5.py --out assets/voz/frecuentes-f5
```

Deja el mismo formato y el mismo `notas.json`, así que el servidor las da por
revisadas y no las regraba. Óyelas, y si convencen, cámbialas por
`frecuentes/` y cópialas al volumen del museo. Las **muletillas no**: a ellas
las sigue la voz de NeuTTS y el cambio de timbre se oiría.

Las frases de una palabra —«Sí.», «No.»— puntúan bajo con cualquier modelo:
duran un segundo y la nota espera medio. Hay que oírlas, no leer la nota.

### De dónde salen los datos

Todo lo que dice de este museo está verificado contra dos fuentes:

- la historia oficial del museo (`museodelaeducacion.gob.cl`): fundado en 1941
  como Museo Pedagógico de Chile, trasladado en 1981 al edificio de la Escuela
  Normal de Niñas Nº1 Brígida Walker (que funcionó allí de 1886 a 1973), cerrado
  en 1985 por el terremoto y reabierto el 8 de marzo de 2006 con el nombre de
  Gabriela Mistral —porque fue ahí donde ella obtuvo su habilitación como
  profesora primaria, en 1910—;
- *Lucila Gabriela: La voz de la Maestra* (MEGM, 2008), de María Isabel Orellana
  y Pedro Pablo Zegers, que documenta su relación con Brígida Walker y que nunca
  tuvo título de Escuela Normal.

Los datos concretos que se usan hoy: fundación en 1941 como Museo Pedagógico de
Chile; traslado en 1981; escuela de niñas entre 1886 y 1973; terremoto de 1985 y
reapertura el 8 de marzo de 2006; habilitación de Gabriela Mistral en 1910 en
esta casa; nacimiento el 7 de abril de 1889 en Vicuña; Vasconcelos y la reforma
mexicana en 1922; Nobel de Literatura en 1945; muerte el 10 de enero de 1957 en
Nueva York; la escuela nocturna de La Cantera; los liceos de Punta Arenas,
Temuco y el Nº6 de Santiago.

**Si agregas una muletilla o una frecuente, verifica igual.** Habla en primera
persona sobre un museo real: una fecha inventada aquí es una fecha que un
visitante se lleva a casa creyendo que se la dijo Gabriela Mistral.

## Procedencia

`referencia-completa.wav` (74,8 s) es la locución del Museo de la Educación
Gabriela Mistral. No se versiona —`.gitignore` excluye `assets/voz/referencia.*`—
y su licencia y autoría deben quedar anotadas aquí antes de publicar el proyecto,
como en `assets/fotos/PROCEDENCIA.md`.

> La voz es una recreación sintética, no una grabación de Gabriela Mistral.

## Dataset para una voz propia (Piper)

La voz en vivo es lenta en el clúster. El plan para salir de eso es afinar
[Piper](https://github.com/rhasspy/piper) —VITS, más rápido que tiempo real en
CPU— con F5 como maestra: horas de F5 diciendo frases variadas, y Piper aprende
a imitarla.

`pipeline/dataset_piper.py` arma ese dataset en `dataset/`, en formato
LJSpeech (`wav/*.wav` a 22 050 Hz y `metadata.csv` con `id|texto`):

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 nohup caffeinate -i uv run pipeline/dataset_piper.py --frases 1500 > assets/voz/dataset/grabacion.log 2>&1 &
```

- Las frases son las de Common Voice en español (CC0), filtradas a las que no
  tienen números ni siglas: F5 las leería de una forma y Piper aprendería otra.
- Las frecuentes ya grabadas con F5 entran tal cual, remuestreadas.
- Cada frase tiene hasta cuatro tomas; si ninguna llega a la nota, se descarta
  y queda en `descartadas.txt`.
- Se corta y se retoma cuando sea. Medido en el M5, en MPS: unos 20 s por frase.

El entrenamiento no va en el Mac —Piper y MPS no se llevan bien—, sino en una
GPU gratuita de Colab: `pipeline/entrenar_piper.ipynb`
([abrir en Colab](https://colab.research.google.com/github/exdevutem/gabriela-mistral/blob/dev/pipeline/entrenar_piper.ipynb)).
Afina desde `es_MX/ald/medium`, guarda los checkpoints en Drive para retomar
cuando Colab corte, y al final exporta el `.onnx` y mide su factor de tiempo
real en CPU. Sin probar todavía de punta a punta.

### Usarla

Deja el `.onnx` y su `.onnx.json` que exporta el notebook en `assets/voz/piper/`
y arranca con `VOZ_MOTOR=piper`. El resto no cambia: sale a 24 kHz como NeuTTS,
así que visemas, visor y lo ya grabado siguen sirviendo.

- Las muletillas se graban aparte, en `muletillas-piper/`: a ellas las sigue la
  voz en vivo y no pueden tener otro timbre. Las frecuentes se quedan como
  están —las grabó F5, que es a quien imita Piper—.
- Medido en el M5 con la voz publicada `es_MX-ald-medium`, la base del
  afinado: **factor de tiempo real 0,022 en CPU**, unas 45 veces más rápido que
  hablar. NeuTTS está entre 1,08 y 1,45. Falta medirlo en el clúster.

## NeuTTS afinado con su voz

La clonación zero-shot adivina la voz a partir de 7,6 s de referencia, y a
veces adivina mal: la «voz de hombre» que cazan `_puntuar` y `_mejor_toma`.
Afinar el backbone de NeuTTS con horas de esa misma voz hace que la aprenda en
vez de adivinarla. Todo local y gratis, en el M5 (MPS):
`pipeline/afinar_neutts.py`.

```bash
uv run python pipeline/afinar_neutts.py preparar          # 33 min, una vez
uv run python pipeline/afinar_neutts.py entrenar          # 600 pasos, 1 h 20 min; retoma si se corta
uv run python pipeline/afinar_neutts.py exportar --paso 600
uv run pipeline/convertir_gguf.py assets/models/neutts-gabriela assets/models/neutts-gabriela-f16.gguf
uv run python pipeline/afinar_neutts.py evaluar base assets/models/neutts-gabriela
```

### Con qué datos

- `dataset/` — las 1.566 frases de F5-TTS del plan de Piper (arriba), pasadas
  por Whisper: las 31 con más de un 25 % de letras mal entendidas se quedan
  fuera. Quedan 1.481.
- `referencia-completa.wav` — los 67 s de la locutora real que siguen a
  `referencia.wav`, cortados en 8 frases con su transcripción a mano (en el
  script, `LOCUCION`) y repetidas 8 veces: ~10 % del entrenamiento, y es la
  voz de verdad y no la de la maestra.
- Las `frec-*` del dataset **no se entrenan**: son las respuestas de
  `FRECUENTES`, y sirven de validación (66) y de evaluación limpia.

Nada de grabaciones de Gabriela Mistral: la voz sigue siendo la recreación de
la locutora del museo.

### Cómo

La receta de neuphonic (`TRAINING.md` y `examples/finetune.py` en
github.com/neuphonic/neutts): fine-tuning completo del backbone —son 230 M de
parámetros, no hace falta LoRA— con transformers y los códigos de NeuCodec
como objetivo. Dos cambios:

- Cada ejemplo lleva delante **el mismo prompt que la inferencia**:
  `referencia.wav` y su texto, armado con `NeuTTS._apply_chat_template`. La
  pérdida cuenta sólo los códigos de la frase.
- La pérdida se calcula sólo donde hay etiqueta, con gradient checkpointing y
  vaciando la caché de MPS en cada paso: así un lote de 8 cabe en ~6 GB. Sin
  eso pasa de 24 GB y el Mac se va a swap.

AdamW a 2e-5, coseno con 30 pasos de calentamiento, lotes de 8, 600 pasos
(3 épocas), fp32: 7 s por paso. La pérdida de validación bajó de 7,22 a 5,05 y
está plana desde el paso 450. Los checkpoints quedan en
`assets/models/neutts-gabriela-ckpt/paso-*` y el exportado, en bf16 (457 MB),
en `assets/models/neutts-gabriela/` con la licencia de neuphonic al lado
(`LICENCE` y `NOTICE`: es obra derivada y la NeuTTS Open License lo exige).

### Cuánto mejora

Medido el 23 de octubre de 2026 en el M5, con `evaluar`: las 87 frases de
`FRECUENTES` dos veces cada una (semillas 0 y 1), **una sola toma**, sin
repetir. «Pasan» es la nota de `_puntuar` ≥ `GRABADO_ACEPTABLE` (0,6); «tono
ajeno», a más de 4 semitonos de la referencia; el CER es lo que entiende
Whisper (`whisper-small`) contra el texto, con las cifras pasadas a letras.

Backbone en transformers sobre MPS (lo que usa `voice.py` para grabar):

| modelo | temperatura | nota media | pasan a la 1.ª | tono ajeno | CER | tiempo real |
|---|---|---|---|---|---|---|
| base     | 1,0 | 0,73 | 78 % | 4 % | 3,7 % | 0,75 |
| base     | 0,7 | 0,72 | 72 % | 3 % | 2,8 % | 0,79 |
| afinado  | 1,0 | 0,87 | 95 % | 1 % | 1,8 % | 0,77 |
| afinado  | 0,7 | 0,86 | 94 % | 1 % | 2,5 % | 0,76 |

El paso 300 da casi lo mismo (94 %, 0 % de tono ajeno): más pasos no
estropean, pero tampoco hacen falta.

GGUF f16 sobre llama.cpp con Metal (lo que usa `vivo.py`):

| modelo | temperatura | nota media | pasan a la 1.ª | tono ajeno | CER | tiempo real |
|---|---|---|---|---|---|---|
| base     | 1,0 | 0,75 | 79 % | 1 % | 5,0 % | 0,42 |
| base     | 0,7 | 0,72 | 76 % | 2 % | 9,2 % | 0,42 |
| afinado  | 1,0 | 0,89 | 96 % | 0 % | 3,3 % | 0,42 |
| afinado  | 0,7 | 0,88 | 94 % | 0 % | 2,0 % | 0,42 |

En vivo, eso es que la primera toma —la única que oye el visitante sin
esperar— casi siempre sirve: con el base, una de cada cinco frases largas se
repetía (`VIVO_INTENTOS`) y costaba otra síntesis de latencia.

Las tomas que fallan a la primera siguen pasando por `_mejor_toma` y por los
reintentos de `vivo.py`, que quedan igual: ahora casi nunca hacen falta.

### Usarlo

No hay que configurar nada: si existen `assets/models/neutts-gabriela/` y
`assets/models/neutts-gabriela-f16.gguf`, `NEUTTS_REPO` y `VIVO_GGUF` los usan
por defecto; si no, el modelo de neuphonic, como antes. Para volver al base
con los afinados presentes, `NEUTTS_REPO=neuphonic/neutts-nano-spanish` y
`VIVO_GGUF=assets/models/neutts-nano-spanish-f16.gguf`. En Docker, monta las
dos cosas en el volumen y apunta las variables a esas rutas.

Lo ya grabado (`frecuentes/`, `respaldo/`) no se toca: está en disco y el
servidor no lo regraba. Para regrabarlo con la voz afinada, borra la carpeta
con su `notas.json`.
