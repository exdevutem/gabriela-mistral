"""Configuración central: modelos locales y rutas."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "assets"
WEB = ROOT / "web"
VOZ = ASSETS / "voz"

# override: gana el .env del proyecto por sobre variables exportadas en el shell
load_dotenv(ROOT / ".env", override=True)

# --- Chat: cualquier API compatible con OpenAI ---
# Por defecto Groq, que tiene nivel gratuito sin tarjeta. Apuntando LLM_URL a un
# llama-server local, el mismo código funciona sin cuenta ni internet: es la
# salida si el museo se queda sin red.
LLM_URL = os.getenv("LLM_URL", "https://api.groq.com/openai/v1/chat/completions")
# Groq retira modelos cada pocos meses (llama-3.3-70b-versatile murió en agosto
# de 2026). Si empieza a responder 404, mira qué hay vivo con:
#   curl -s https://api.groq.com/openai/v1/models -H "Authorization: Bearer $LLM_API_KEY"
LLM_MODEL = os.getenv("LLM_MODEL", "qwen/qwen3.8-27b")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")  # vacío para un llama-server local

# --- Voz: NeuTTS nano-spanish (clonación a partir de un audio de referencia) ---
# Reemplazó a F5-TTS el 20 de septiembre de 2026: misma voz de referencia, 2,8
# veces más rápido en el mismo M2 (medido: RTF 4,08 -> 1,45).
NEUTTS_REPO = os.getenv("NEUTTS_REPO", "neuphonic/neutts-nano-spanish")
NEUTTS_CODEC = os.getenv("NEUTTS_CODEC", "neuphonic/neucodec")
REF_AUDIO = VOZ / "referencia.wav"
REF_TEXTO = VOZ / "referencia.txt"

# Ambos repos son *gated*: hay que entrar con la cuenta de Hugging Face a
#   huggingface.co/neuphonic/neutts-nano-spanish  y  .../neucodec
# y aceptar los términos una vez, con HF_TOKEN en el entorno. Sin eso la
# descarga responde 403 y el servidor arranca sin voz.

# Lo que dice mientras piensa. Se sintetizan una vez y se guardan: son siempre
# las mismas, y pagarlas en cada pregunta sería añadir espera a la espera.
#
# Cada una empieza con una frase que compra tiempo y sigue contando el museo.
# El servidor las manda troceadas por frase y el visor descarta los trozos que
# no alcanzaron a sonar, así que el largo no se paga: si la respuesta llega
# pronto, se oye sólo el principio.
#
# Los números van con letras a propósito: el TTS lee "1941" como puede.
#
# TODO lo que se afirma aquí está verificado contra la historia oficial del
# museo (museodelaeducacion.gob.cl) y contra «Lucila Gabriela: La voz de la
# Maestra» (MEGM, 2008). Si agregas una, verifica igual: ella no debe decir
# cosas falsas sobre un museo real, y menos hablando en primera persona.
MULETILLAS_DIR = Path(os.getenv("MULETILLAS_DIR", VOZ / "muletillas"))
MULETILLAS = [
    "Mientras busco las palabras, déjame contarte dónde estás. "
    "Este edificio fue la Escuela Normal Número Uno. "
    "Aquí rendí mis exámenes de habilitación, el año diez. "
    "Nunca tuve título de normalista. "
    "Me hice maestra leyendo y enseñando. "
    "Casi cien años después, le pusieron mi nombre a esta casa.",

    "Dame un momento, que no quiero contestarte cualquier cosa. "
    "Te cuento algo mientras. "
    "Este museo nació el año cuarenta y uno. "
    "Entonces se llamaba Museo Pedagógico de Chile. "
    "Se mudó a esta casa en el ochenta y uno. "
    "Un terremoto la cerró el ochenta y cinco. "
    "Estuvo veintiún años sin abrir.",

    "Espera, que la respuesta se me está formando. "
    "Mira a tu alrededor mientras tanto. "
    "Los bancos, los mapas, las máquinas de escribir. "
    "Proyectores de otro siglo. "
    "La campana de la primera Escuela de Preceptores de Chile. "
    "Y los instrumentos de castigo, que también son historia.",

    "Déjame pensar un poco. "
    "Esta casa fue escuela de niñas por casi noventa años. "
    "Su primera directora chilena fue Brígida Walker. "
    "La conocí cuando vine a rendir mis exámenes. "
    "Piensa en cuántas niñas cruzaron esta puerta antes que tú.",
]

# Preguntas frecuentes: los badges que la página ofrece bajo la caja de texto.
# La respuesta va grabada, no la escribe el LLM: suena al instante, siempre dice
# lo mismo y no hay forma de que invente una fecha. Es lo que corresponde a un
# museo real —un dato falso aquí se lo lleva el visitante a casa creyendo que se
# lo dijo ella—, y es también la respuesta más rápida que el sistema puede dar.
#
# Todo lo que se afirma está verificado contra las dos fuentes que documenta
# assets/voz/README.md. Si agregas una, verifica igual y deja la fuente anotada.
# Lo que es opinión suya («eso es lo que no hay que hacerle a un niño») va en su
# voz y se distingue de lo que es dato; los datos no se adornan.
FRECUENTES_DIR = Path(os.getenv("FRECUENTES_DIR", VOZ / "frecuentes"))
FRECUENTES = [
    # --- El museo ---
    ("¿Qué fue este edificio?",
     "Esta casa fue la Escuela Normal de Niñas Número Uno. "
     "Funcionó como escuela desde mil ochocientos ochenta y seis. "
     "Dejó de serlo el año setenta y tres. "
     "Después llegó el museo."),

    ("¿Desde cuándo existe el museo?",
     "El museo nació el año cuarenta y uno. "
     "Entonces se llamaba Museo Pedagógico de Chile. "
     "Se mudó a esta casa en el ochenta y uno. "
     "Y tomó mi nombre el ocho de marzo del dos mil seis."),

    ("¿Por qué lleva tu nombre?",
     "Porque fue aquí donde me hice maestra en el papel. "
     "El año diez rendí en esta casa mis exámenes de habilitación. "
     "Yo no venía de la Escuela Normal. "
     "Casi cien años después le pusieron mi nombre a este lugar."),

    ("¿Qué se guarda aquí?",
     "Bancos, mapas, pizarras. "
     "Máquinas de escribir y proyectores de otro siglo. "
     "La campana de la primera Escuela de Preceptores de Chile. "
     "Y también los instrumentos de castigo. "
     "Esos también son parte de esta historia."),

    ("¿Por qué estuvo cerrado?",
     "Por un terremoto. "
     "El del año ochenta y cinco dañó este edificio. "
     "El museo estuvo veintiún años sin abrir. "
     "Volvió el ocho de marzo del dos mil seis."),

    ("¿Quién fue Brígida Walker?",
     "La primera directora chilena de la Escuela Normal de Niñas. "
     "Por ella lleva su nombre esta casa. "
     "La conocí cuando vine a rendir mis exámenes. "
     "Fue cercana conmigo en esos días."),

    ("¿Dónde estamos?",
     "En el barrio Yungay, en Santiago. "
     "Calle Compañía, número tres mil ciento cincuenta. "
     "Es la esquina con Chacabuco. "
     "Esta casa es monumento nacional desde el año ochenta y uno."),

    # --- Ella ---
    ("¿Cuál es tu nombre verdadero?",
     "Lucila Godoy Alcayaga. "
     "Gabriela Mistral es un nombre que me puse yo. "
     "Con ese firmé lo que escribí. "
     "Pero la que enseñaba se llamaba Lucila."),

    ("¿Cuándo naciste?",
     "El siete de abril de mil ochocientos ochenta y nueve. "
     "En Vicuña, en el valle de Elqui. "
     "Ese mismo año se fundó el Instituto Pedagógico. "
     "Buen año para nacer, si una iba a ser maestra."),

    ("¿Tenías título de maestra?",
     "No. Nunca pasé por la Escuela Normal. "
     "Rendí exámenes de habilitación el año diez, en esta casa. "
     "Sin título sólo podía enseñar en escuelas de segunda categoría. "
     "Así que enseñé donde nadie más quería ir."),

    ("¿Dónde enseñaste?",
     "Empecé en escuelas pequeñas y apartadas. "
     "La Cantera, detrás de las dunas, fue una de ellas. "
     "Después dirigí liceos de niñas en Punta Arenas y en Temuco. "
     "Y el Liceo Número Seis de Santiago."),

    ("¿Cómo era tu escuela en La Cantera?",
     "Era una escuela de noche. "
     "De día no venía nadie, porque todos trabajaban. "
     "Niños, hombres y viejos. "
     "Me llevaban camotes, melones y papas, como un diezmo. "
     "A un viejo analfabeto al fin le enseñé a leer."),

    ("¿Qué hiciste en México?",
     "Me llamó José Vasconcelos, el año veintidós. "
     "Era el ministro de educación de México. "
     "Trabajé en su reforma educativa. "
     "Ayudé a crear bibliotecas escolares. "
     "Y preparé lecturas para mujeres."),

    ("¿Ganaste el Premio Nobel?",
     "Sí. El de Literatura, el año cuarenta y cinco. "
     "Pero si me preguntas qué fui, te diré maestra. "
     "Lo otro vino después."),

    ("¿Qué te pasó de niña en la escuela?",
     "Me acusaron de robarme unos útiles. "
     "No era cierto. "
     "Los otros niños me golpearon. "
     "Lo volví a contar el año cincuenta y cuatro, ya vieja. "
     "Eso es lo que no hay que hacerle a un niño."),

    ("¿Cuándo moriste?",
     "El diez de enero de mil novecientos cincuenta y siete. "
     "En un hospital de Nueva York, lejos de Chile. "
     "Un cáncer al páncreas."),
]

# El mundo físico necesita perillas.
# "auto" = mps si la máquina lo tiene, cpu si no. Se resuelve en voice.py, que
# es donde torch ya está cargado. Medido en un M2 el 20 de septiembre de 2026,
# tras calentar: RTF 1,08 en mps contra 1,37 en cpu, y la referencia se codifica
# en 1,4 s en vez de 11,0 s. A cambio, arrancar cuesta 35 s más.
# F5-TTS moría sin traza en mps; NeuTTS aguantó 27 síntesis seguidas y tres
# conversaciones enteras por WebSocket, que es donde sintetiza en otro hilo.
DEVICE = os.getenv("NEUTTS_DEVICE", "auto")
# NeuTTS no tiene perilla de velocidad como la tenía F5 (`speed=0.9`). El ritmo
# sale del audio de referencia: si habla apurada, la referencia es lo que hay
# que cambiar.
TEMPERATURA = float(os.getenv("NEUTTS_TEMPERATURA", "1.0"))
# Sale más bajo que F5 (RMS medido 0,044 frente a 0,099 en la misma frase).
# Se normaliza al nivel que tenía la voz anterior para no tocar el volumen del
# navegador ni el de la sala.
RMS_OBJETIVO = float(os.getenv("NEUTTS_RMS", "0.09"))
# Semilla fija: la voz sale igual en cada ejecución.
SEED = int(os.getenv("NEUTTS_SEED", "0"))

# --- Calidad de lo pre-grabado ---
# F5-TTS tenía `nfe_step`: subirlo compraba calidad con tiempo, y estaba en 8
# porque en vivo el tiempo era lo que faltaba. NeuTTS no tiene esa perilla —no
# hay pasos de difusión que subir, genera en un paso autorregresivo— así que no
# es cosa de cambiarle el número.
#
# Lo que sí varía es la suerte del muestreo. Con la misma frase y distinta
# semilla, una toma sale con su voz y otra sale con voz de hombre, o divagando,
# o cortada a media palabra: es el modo de fallar de un modelo autorregresivo
# que clona a partir de una referencia. En vivo hay que quedarse con la primera
# toma. Lo grabado no: nadie está esperando, así que se sintetiza varias veces
# y se guarda la mejor. Ese es el reemplazo honesto de `nfe_step` —comprar
# calidad con tiempo— y se paga una sola vez, porque queda en disco.
#
# Cuántas tomas como máximo por frase. Se para en cuanto una es lo bastante
# buena, así que las frases que salen bien a la primera —la mayoría— siguen
# costando una. Ver `_mejor_toma` en voice.py.
GRABADO_INTENTOS = int(os.getenv("NEUTTS_INTENTOS", "6"))
# Nota mínima para dar una toma por buena, de 0 a 1. Ver `_puntuar`: 1,0 es
# exactamente el tono de la referencia y exactamente el largo que le toca al
# texto. Subirlo hace que se repitan más frases y el arranque tarde más; por
# debajo de 0,5 ya deja pasar una voz que no es la suya.
GRABADO_ACEPTABLE = float(os.getenv("NEUTTS_ACEPTABLE", "0.6"))
# Las tomas grabadas se muestrean más frías que las de en vivo: con menos
# temperatura el modelo se aparta menos de la referencia, que es de donde salen
# las voces ajenas. A cambio la entonación es algo más plana, y por eso no se
# toca la de en vivo —donde además no habría con qué comparar la toma.
GRABADO_TEMPERATURA = float(os.getenv("NEUTTS_TEMPERATURA_GRABADA", "0.7"))
