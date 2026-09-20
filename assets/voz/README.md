# Voz de referencia

F5-TTS no tiene voces prefabricadas: clona la que le des. Estos dos archivos
deciden cómo suena Gabriela.

- `referencia.wav` — habla limpia, sin música ni ruido de fondo.
- `referencia.txt` — la transcripción **exacta** de ese audio.

## La regla que hay que respetar: 12 segundos

F5-TTS recorta internamente el audio de referencia a 12 s, **pero usa el texto
entero que le pases**. Si el `.wav` dura 75 s y el `.txt` los transcribe todos,
el modelo cree que 863 caracteres caben en 12 s y la voz sale atropellada.

Audio y texto tienen que corresponderse tras el recorte. Lo seguro es recortar
uno mismo el `.wav` a 7–10 s, en un silencio entre frases, y transcribir sólo
ese trozo.

El actual son los primeros 7,58 s de `referencia-completa.wav`, cortados donde
termina «Soy Lucila Godoy Alcayaga». Para rehacerlo desde el original:

```python
from pydub import AudioSegment, silence
a = AudioSegment.from_file("assets/voz/referencia-completa.wav")
print(silence.detect_nonsilent(a, min_silence_len=300, silence_thresh=a.dBFS - 16))
a[:7580].set_channels(1).set_frame_rate(24000).export("assets/voz/referencia.wav", format="wav")
```

Y para transcribir el recorte sin escribirlo a mano, con el Whisper que ya trae
`f5-tts`:

```python
from f5_tts.infer.utils_infer import transcribe
print(transcribe("assets/voz/referencia.wav", language="es"))
```

## Duración y latencia

El largo de la referencia se paga en **cada** síntesis: F5-TTS genera la
referencia y la frase juntas. Medido en un M2 de 8 GB, para 6,5 s de audio:

| referencia | `nfe_step=32` | `nfe_step=16` |
|------------|---------------|---------------|
| 11,8 s     | 31,4 s        | 15,8 s        |
| 7,6 s      | 23,2 s        | 11,6 s        |

De ahí los valores actuales: referencia corta y `F5_NFE_STEP=16`.

## Muletillas

`muletillas/` son las frases cortas que dice mientras piensa —«Déjame pensar.»,
«Mmm. Espera un momento.»— para que la espera de medio minuto no empiece en
silencio. Las graba el servidor al arrancar, una sola vez, con esta misma voz de
referencia.

**Si cambias la referencia o `F5_NFE_STEP`, borra esa carpeta**: si no, seguirá
titubeando con la voz anterior y cambiando de timbre a mitad de respuesta. Pasa
igual con los pasos de difusión, porque cambian el timbre lo suficiente para
notarse entre la muletilla y la frase que la sigue. El texto de las frases está
en `MULETILLAS`, en `config.py`.

## Procedencia

`referencia-completa.wav` (74,8 s) es la locución del Museo de la Educación
Gabriela Mistral. No se versiona —`.gitignore` excluye `assets/voz/referencia.*`—
y su licencia y autoría deben quedar anotadas aquí antes de publicar el proyecto,
como en `assets/fotos/PROCEDENCIA.md`.

> La voz es una recreación sintética, no una grabación de Gabriela Mistral.
