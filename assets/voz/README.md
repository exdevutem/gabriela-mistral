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

Arranque completo (`calentar()`, con las cuatro muletillas): **43,4 s**.

NeuTTS habla más pausado que F5 en frases cortas —la misma muletilla le dura el
doble— y no tiene perilla de velocidad: el ritmo sale de la referencia.

## Muletillas

`muletillas/` son las frases cortas que dice mientras piensa —«Déjame pensar.»,
«Mmm. Espera un momento.»— para que la espera no empiece en silencio. Las graba el servidor al arrancar, una sola vez, con esta misma voz de
referencia.

**Si cambias la referencia o el modelo, borra esa carpeta**: si no, seguirá
titubeando con la voz anterior y cambiando de timbre a mitad de respuesta. El
texto de las frases está en `MULETILLAS`, en `config.py`.

## Procedencia

`referencia-completa.wav` (74,8 s) es la locución del Museo de la Educación
Gabriela Mistral. No se versiona —`.gitignore` excluye `assets/voz/referencia.*`—
y su licencia y autoría deben quedar anotadas aquí antes de publicar el proyecto,
como en `assets/fotos/PROCEDENCIA.md`.

> La voz es una recreación sintética, no una grabación de Gabriela Mistral.
