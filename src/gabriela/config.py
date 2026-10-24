"""Configuración central: modelos locales, respuestas y rutas."""
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

# --- Voz: qué motor graba las respuestas ---
# Ya no habla en vivo: sólo graba, al arrancar, las respuestas que falten.
# "neutts" clona desde la referencia y es lo probado. "piper" es la voz propia
# afinada con F5-TTS de maestra (pipeline/entrenar_piper.ipynb). "gemini" es el
# TTS de Google AI Studio: no clona, y su nivel gratuito da 3 peticiones por
# minuto y 10 al día (medido el 21 de octubre de 2026), que no alcanzan ni para
# grabar lo fijo una vez.
MOTOR = os.getenv("VOZ_MOTOR", "neutts")
# --- Voz: Gemini TTS (sólo con VOZ_MOTOR=gemini) ---
# Clave gratuita en https://aistudio.google.com/apikey.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
# El TTS más barato de la API (octubre de 2026: 6 USD por millón de tokens de
# audio, ~0,0015 USD por 10 s; gratis en el nivel gratuito). Los *-preview-tts*
# se apagan el 17 de noviembre de 2026: no volver a ellos.
GEMINI_TTS_MODELO = os.getenv("GEMINI_TTS_MODELO", "gemini-3.8-flash-lite-tts")
# Voz prefabricada. Las que mejor le van a una maestra mayor: Gacrux («madura»),
# Sulafat («cálida»), Vindemiatrix («amable»). Lista completa en
# ai.google.dev/gemini-api/docs/speech-generation
GEMINI_VOZ = os.getenv("GEMINI_VOZ", "Gacrux")
# El estilo se le pide con palabras. La edad y el género los pone la voz; el
# acento no está garantizado —la API sólo dice «español»—, así que se pide.
GEMINI_ESTILO = os.getenv(
    "GEMINI_ESTILO",
    "Mujer chilena mayor, maestra. Habla en español de Chile, pausada, grave "
    "y cálida, sin dramatizar.")

# El .onnx que exporta el notebook; su .onnx.json tiene que estar al lado.
PIPER_MODELO = Path(os.getenv("PIPER_MODELO", VOZ / "piper" / "es_419-gabriela-medium.onnx"))

# --- Voz: NeuTTS nano-spanish (clonación a partir de un audio de referencia) ---
# Reemplazó a F5-TTS el 20 de septiembre de 2026: misma voz de referencia, 2,8
# veces más rápido en el mismo M2 (medido: RTF 4,08 -> 1,45).
# Si está el backbone afinado con su voz (pipeline/afinar_neutts.py), ése; si
# no, el de neuphonic. Ruta local o repo de Hugging Face. Ver assets/voz/README.md.
_AFINADO = ASSETS / "models" / "neutts-gabriela"
NEUTTS_REPO = os.getenv("NEUTTS_REPO", str(_AFINADO) if _AFINADO.is_dir()
                        else "neuphonic/neutts-nano-spanish")
NEUTTS_CODEC = os.getenv("NEUTTS_CODEC", "neuphonic/neucodec")
REF_AUDIO = VOZ / "referencia.wav"
REF_TEXTO = VOZ / "referencia.txt"

# Ambos repos son *gated*: hay que entrar con la cuenta de Hugging Face a
#   huggingface.co/neuphonic/neutts-nano-spanish  y  .../neucodec
# y aceptar los términos una vez, con HF_TOKEN en el entorno. Sin eso la
# descarga responde 403 y el servidor arranca sin voz.

# Preguntas frecuentes: lo único que ella responde. No hay chat: lo que el
# visitante escribe o dice se compara con estas preguntas y sus respuestas
# (ver entender.py) y suena la grabada que más se le parezca. La página las
# ofrece además como badges bajo la caja de texto.
#
# La respuesta va grabada, no la escribe un modelo: suena al instante, siempre dice
# lo mismo y no hay forma de que invente una fecha. Es lo que corresponde a un
# museo real —un dato falso aquí se lo lleva el visitante a casa creyendo que se
# lo dijo ella—, y es también la respuesta más rápida que el sistema puede dar.
#
# Todo lo que se afirma está verificado contra las dos fuentes que documenta
# assets/voz/README.md. Si agregas una, verifica igual y deja la fuente anotada.
# Lo que es opinión suya («eso es lo que no hay que hacerle a un niño») va en su
# voz y se distingue de lo que es dato; los datos no se adornan.
FRECUENTES_DIR = Path(os.getenv("FRECUENTES_DIR", VOZ / "frecuentes"))
# Una carpeta por motor: grabadas con motores distintos, las respuestas
# hablarían con voces distintas. Cambiar de motor graba las suyas en el
# siguiente arranque en vez de mezclar voces.
if MOTOR != "neutts":
    FRECUENTES_DIR = FRECUENTES_DIR.with_name(f"{FRECUENTES_DIR.name}-{MOTOR}")
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

    # --- Lo que antes eran muletillas ---
    # Rellenaban la espera del chat. Sin chat no hay espera, y lo que contaban
    # —verificado como todo lo demás— queda como respuesta propia. Los números
    # siguen con letras: el TTS lee "1941" como puede.
    ("¿Qué pasó en esta casa?",
     "Este edificio fue la Escuela Normal Número Uno. "
     "Aquí rendí mis exámenes de habilitación, el año diez. "
     "Nunca tuve título de normalista. "
     "Me hice maestra leyendo y enseñando. "
     "Casi cien años después, le pusieron mi nombre a esta casa."),

    ("¿Cómo empezó el museo?",
     "Este museo nació el año cuarenta y uno. "
     "Entonces se llamaba Museo Pedagógico de Chile. "
     "Se mudó a esta casa en el ochenta y uno. "
     "Un terremoto la cerró el ochenta y cinco. "
     "Estuvo veintiún años sin abrir."),

    ("¿Qué puedo mirar aquí?",
     "Mira a tu alrededor. "
     "Los bancos, los mapas, las máquinas de escribir. "
     "Proyectores de otro siglo. "
     "La campana de la primera Escuela de Preceptores de Chile. "
     "Y los instrumentos de castigo, que también son historia."),

    ("¿Quiénes estudiaron aquí?",
     "Esta casa fue escuela de niñas por casi noventa años. "
     "Su primera directora chilena fue Brígida Walker. "
     "La conocí cuando vine a rendir mis exámenes. "
     "Piensa en cuántas niñas cruzaron esta puerta antes que tú."),
]

# Otras formas de hacer la misma pregunta. El elector compara lo que dijo el
# visitante con la pregunta, con su respuesta y con éstas: sin ellas confunde
# vecinas —«¿dónde queda esto?» se parece más a «¿Qué se guarda aquí?» que a
# «¿Dónde estamos?»—. Si una pregunta real cae en la respuesta equivocada,
# agregarla aquí es el arreglo; la clave tiene que ser una de FRECUENTES.
VARIANTES = {
    "¿Qué fue este edificio?": ["¿Qué era esta casa antes?", "¿Para qué servía este lugar?",
                                "¿Qué era esta casa antes del museo?"],
    "¿Desde cuándo existe el museo?": ["¿Cuántos años tiene el museo?",
                                       "¿Cuándo llegó el museo a esta casa?"],
    "¿Por qué lleva tu nombre?": ["¿Por qué el museo se llama Gabriela Mistral?",
                                  "¿Por qué el museo tiene tu nombre?"],
    "¿Qué se guarda aquí?": ["¿Qué hay en el museo?", "¿Qué cosas tiene el museo?"],
    "¿Por qué estuvo cerrado?": ["¿Por qué cerraron el museo?"],
    "¿Quién fue Brígida Walker?": ["¿Quién era la directora de la escuela?",
                                   "¿Conociste a la directora?"],
    "¿Dónde estamos?": ["¿Dónde queda esto?", "¿Cuál es la dirección del museo?",
                        "¿En qué barrio estamos?"],
    "¿Cuál es tu nombre verdadero?": ["¿Cómo te llamabas?", "¿Quién eres?"],
    "¿Cuándo naciste?": ["¿Dónde naciste?", "¿Qué edad tienes?"],
    "¿Tenías título de maestra?": ["¿Estudiaste para profesora?"],
    "¿Dónde enseñaste?": ["¿Dónde fuiste profesora?", "¿En qué escuelas trabajaste?"],
    "¿Ganaste el Premio Nobel?": ["¿Qué premios ganaste?", "¿Por qué eres famosa?"],
    "¿Qué te pasó de niña en la escuela?": ["¿Cómo fue tu infancia en el colegio?",
                                            "¿Te castigaron de niña?"],
    "¿Qué puedo mirar aquí?": ["¿Qué puedo ver en el museo?", "¿Qué me recomiendas mirar?"],
}

# Lo que dice cuando no reconoce la pregunta entre las frecuentes. Mejor eso
# que contestar otra cosa con toda seguridad: en un museo, una respuesta que no
# viene al caso suena a que ella no escuchó, y una inventada es peor.
RESPALDO = ("No te entendí bien. "
            "Pregúntame por mí o por esta casa. "
            "Abajo tienes algunas preguntas, si quieres.")
RESPALDO_DIR = Path(os.getenv("RESPALDO_DIR", VOZ / "respaldo"))
if MOTOR != "neutts":
    RESPALDO_DIR = RESPALDO_DIR.with_name(f"{RESPALDO_DIR.name}-{MOTOR}")

# --- Entender la pregunta (todo local, ver entender.py) ---
# Whisper pasa la voz a texto; no sabe elegir una respuesta. Eso lo hace un
# modelo de embeddings multilingüe, que compara el sentido de lo dicho con cada
# pregunta frecuente y su respuesta. "small" entiende bien el español de un
# visitante hablándole a un micrófono de sala; "base" es tres veces más liviano
# y se equivoca más con nombres propios.
WHISPER_MODELO = os.getenv("WHISPER_MODELO", "openai/whisper-small")
# "auto" = mps si la máquina lo tiene, cpu si no (el cluster no tiene GPU).
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "auto")
EMBEDDINGS_MODELO = os.getenv(
    "EMBEDDINGS_MODELO", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
# Parecido mínimo (coseno, de -1 a 1) para darle a una pregunta por reconocida.
# Por debajo suena RESPALDO. Subirlo hace que diga «no te entendí» más seguido;
# bajarlo, que conteste algo que no le preguntaron.
UMBRAL = float(os.getenv("UMBRAL", "0.5"))

# --- Lo que sabe fuera de las frecuentes (saber.py) ---
# Si ninguna frecuente pasa de UMBRAL, se buscan los documentos oficiales del
# museo y un LLM local muy pequeño contesta con ellos. Ver saber.py.
CONOCIMIENTO = Path(os.getenv("CONOCIMIENTO", ASSETS / "conocimiento" / "fragmentos.jsonl"))
# Ruta a un .gguf o `repo:archivo` de Hugging Face. Vacío apaga el LLM y todo
# lo que no sea frecuente vuelve a ser «no te entendí».
# LFM2.5 1,2B en Q4_K_M (731 MB). Medido el 23 de octubre de 2026 contra
# Qwen3.5 0,8B y 2B y Gemma 3 1B con diez preguntas del museo: es el que menos
# se aparta de los fragmentos. Gemma convirtió a José Abelardo Núñez en militar;
# Qwen 0,8B contesta «Sí,» a todo; Qwen 2B redacta mejor pero adorna.
LLM_GGUF = os.getenv("LLM_GGUF", "LiquidAI/LFM2.5-1.2B-Instruct-GGUF:LFM2.5-1.2B-Instruct-Q4_K_M.gguf")
# "auto" = todas las capas a la GPU si llama.cpp se compiló con Metal o CUDA;
# "cpu" = ninguna.
LLM_DEVICE = os.getenv("LLM_DEVICE", "auto")
LLM_CTX = int(os.getenv("LLM_CTX", "2048"))
# Cuánto puede escribir: dos o tres frases dichas en voz alta son ~60 tokens.
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "120"))
# Fragmentos que se le dan al modelo. Más, más contexto que leer en CPU.
LLM_FRAGMENTOS = int(os.getenv("LLM_FRAGMENTOS", "4"))
# Parecido mínimo entre la pregunta y un fragmento para que valga la pena
# preguntarle al modelo. Por debajo, la pregunta no es del museo ni de ella.
SABER_UMBRAL = float(os.getenv("SABER_UMBRAL", "0.5"))
# Cuánto más que el mejor documento tiene que parecerse una frecuente para
# contestar ella con lo grabado. Ver saber.piso().
SABER_MARGEN = float(os.getenv("SABER_MARGEN", "0.05"))

# --- IA: Claude Haiku (saber.py, sólo arrancando con `gabriela --ia`) ---
# En modo IA todo lo contesta Claude Haiku —ni frecuentes grabadas, ni
# documentos, ni LLM local, que ni se cargan— y suena con la voz en vivo. Se
# elige al arrancar (ver __main__.py); MODO=ia sirve para uvicorn a mano o Docker.
IA = os.getenv("MODO", "local") == "ia"
# Sin clave, el modo IA no arranca. Es de pago (octubre de 2026: 0,10 USD
# por millón de tokens de entrada y 0,50 de salida, ~0,0002 USD por pregunta).
# La clave ya pertenece a un workspace: no hace falta su id para llamar.
ANTHROPIC_KEY = os.getenv("ANTHROPIC_KEY") or os.getenv("ANTHROPIC_API_KEY", "")
IA_MODELO = os.getenv("IA_MODELO", "claude-haiku-5-5")
# low: contesta antes y para dos o tres frases no hace falta pensar más.
IA_ESFUERZO = os.getenv("IA_ESFUERZO", "low")

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

# --- Voz en vivo (vivo.py) ---
# Lo que no está entre las frecuentes se sintetiza al momento: NeuTTS con el
# backbone en GGUF por llama.cpp (Metal en un Mac, CPU en el clúster) y la misma
# referencia de siempre. Medido en un M5 con el f16: primer sonido a los 0,4 s y
# factor de tiempo real 0,32. Con el backbone de transformers en MPS iba a 1,08.
#
# Ruta a un .gguf, o un repo de Hugging Face. El f16 lo genera
# pipeline/convertir_gguf.py; el cuantizado oficial
# (neuphonic/neutts-nano-spanish-q4-gguf) es gated aparte del modelo base.
# Como NEUTTS_REPO: el afinado si está convertido, si no el base.
_AFINADO_GGUF = ASSETS / "models" / "neutts-gabriela-f16.gguf"
VIVO_GGUF = os.getenv("VIVO_GGUF", str(_AFINADO_GGUF) if _AFINADO_GGUF.exists()
                      else str(ASSETS / "models" / "neutts-nano-spanish-f16.gguf"))
# "auto" = Metal si lo hay, si no CPU. Cualquier cosa distinta de "cpu" manda
# las capas del backbone a la GPU (llama.cpp elige Metal o CUDA según cómo se
# compiló); el codec corre en el mismo dispositivo con torch.
VIVO_DEVICE = os.getenv("VIVO_DEVICE", "auto")
# Si una toma en vivo puntúa menos que GRABADO_ACEPTABLE se repite, más fría,
# hasta esta cantidad de veces en total. Cada repetición cuesta ~0,3 s por
# segundo de audio; sin esto, la «voz de hombre» de NeuTTS llega al visitante.
VIVO_INTENTOS = int(os.getenv("VIVO_INTENTOS", "3"))
# Antes de que empiece a sonar se junta audio hasta que alcance a cubrir lo que
# falta por sintetizar, al ritmo medido en esa misma respuesta y multiplicado
# por esto: mejor esperar al principio que cortarse a media respuesta. Subirlo
# si todavía se corta (p. ej. con la GPU compartida); bajarlo si la espera
# inicial se hace larga.
VIVO_COLCHON = float(os.getenv("VIVO_COLCHON", "1.3"))
