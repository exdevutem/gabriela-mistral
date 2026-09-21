"""Texto -> WAV con NeuTTS nano-spanish, ejecutándose en local.

NeuTTS clona, igual que el F5-TTS que había antes: no tiene voces prefabricadas.
La voz sale del par `assets/voz/referencia.wav` + `referencia.txt` (unos 10 s de
habla y su transcripción exacta). Cambiar esos dos archivos cambia la voz de
Gabriela.

Medido en un M2 de 8 GB, con la misma referencia y las mismas frases, todo
tras calentar:

| frase                    | F5-TTS | NeuTTS cpu | NeuTTS mps |
|--------------------------|--------|------------|------------|
| «Déjame pensar.»         | 10,7 s |      4,0 s |      2,5 s |
| «Mmm. Espera un momento.»| 13,7 s |      4,3 s |      2,9 s |
| una respuesta de ~9 s    | 30,0 s |     12,5 s |      9,1 s |
| factor de tiempo real    |   4,08 |       1,37 |       1,08 |

Sigue sin ser tiempo real: las muletillas hacen falta igual.
"""
from __future__ import annotations

import io
import json
import logging
import platform
import wave
from functools import lru_cache

import numpy as np

from .config import (DEVICE, FRECUENTES, FRECUENTES_DIR, GRABADO_ACEPTABLE,
                     GRABADO_INTENTOS, GRABADO_TEMPERATURA, MULETILLAS,
                     MULETILLAS_DIR, NEUTTS_CODEC, NEUTTS_REPO, REF_AUDIO,
                     REF_TEXTO, RMS_OBJETIVO, SEED, TEMPERATURA)

log = logging.getLogger(__name__)

SAMPLE_RATE = 24_000  # NeuCodec entrega 24 kHz, igual que el vocoder de F5
SAMPLE_WIDTH = 2
MAX_BYTES = 135  # tope por trozo; ver trozos()
# Tope para las frases de muletilla, más corto a propósito. Cuando llega la
# respuesta, el visor descarta el relleno pendiente pero deja terminar el trozo
# que está sonando: esa frase es el retraso máximo que el relleno puede costar.
# Con 70 bytes son unos 5 s. Medido: la voz va a unos 13 caracteres por segundo.
MAX_BYTES_MULETILLA = 70
# A lo que habla ella, medido sobre lo grabado. Sirve para saber cuánto debería
# durar un texto y notar una toma cortada o una que se fue a divagar.
CARACTERES_POR_SEGUNDO = 13
# Lo que puntuó cada archivo grabado, al lado de los archivos. Ver _notas().
NOTAS = "notas.json"
# Medir el tono: marco de análisis y el rango donde puede caer una voz humana.
# El marco son 85 ms, de los que la primera mitad se compara consigo misma
# desplazada hasta 43 ms: de sobra para un período de 70 Hz, que son 14 ms.
MARCO = 2048
TONO_MINIMO, TONO_MAXIMO = 70.0, 400.0


def _usar_espeak_del_sistema() -> None:
    """El espeak-ng que trae `neutts` viene roto en macOS: la dylib empaquetada
    lleva compilada la ruta del directorio temporal donde se construyó, que ya
    no existe, y muere con «Error processing file .../phontab». Ni
    ESPEAK_DATA_PATH ni EspeakWrapper.set_data_path la corrigen.

    Se cambia por la de Homebrew (`brew install espeak-ng`), que sí conoce sus
    datos. Tiene que ser DESPUÉS de importar neutts, porque su import elige la
    empaquetada.

    ponytail: parche de un solo punto y sólo en macOS —en Linux la biblioteca
    empaquetada carga bien—. Cuando publiquen una rueda con la ruta correcta,
    esta función se borra entera.
    """
    if platform.system() != "Darwin":
        return
    import glob

    from phonemizer.backend.espeak.wrapper import EspeakWrapper

    for patron in ("/opt/homebrew/Cellar/espeak-ng/*/lib/libespeak-ng.*.dylib",
                   "/usr/local/Cellar/espeak-ng/*/lib/libespeak-ng.*.dylib"):
        if encontradas := glob.glob(patron):
            EspeakWrapper.set_library(encontradas[0])
            return
    raise RuntimeError(
        "falta espeak-ng del sistema: `brew install espeak-ng`. El que trae "
        "neutts no funciona en macOS."
    )


def _device() -> str:
    """Resuelve DEVICE="auto" a lo que esta máquina tenga.

    Aquí y no en config.py porque importar torch cuesta segundos y config lo
    carga todo el mundo, tests incluidos.
    """
    if DEVICE != "auto":
        return DEVICE
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


@lru_cache(maxsize=1)
def precargar():
    """Carga perezosa: son ~1,5 GB de pesos y tarda en arrancar (10,6 s medidos
    con los modelos ya en caché; la primera vez hay que bajarlos).

    Cacheado además porque mantenerlo en memoria es la diferencia entre
    sintetizar en segundos o en minutos.
    """
    from neutts import NeuTTS

    _usar_espeak_del_sistema()

    device = _device()
    return NeuTTS(
        backbone_repo=NEUTTS_REPO,
        codec_repo=NEUTTS_CODEC,
        backbone_device=device,
        codec_device=device,
        language="es",
        seed=SEED,
    )


@lru_cache(maxsize=1)
def _referencia() -> tuple[str, str]:
    if not REF_AUDIO.exists() or not REF_TEXTO.exists():
        raise RuntimeError(
            f"Falta la voz de referencia. Deja en {REF_AUDIO.parent} un "
            f"'{REF_AUDIO.name}' de unos 10 s y un '{REF_TEXTO.name}' con su "
            "transcripción exacta."
        )
    return str(REF_AUDIO), REF_TEXTO.read_text(encoding="utf-8").strip()


@lru_cache(maxsize=1)
def _codigos_referencia():
    """La referencia, ya pasada por el codec. Son 11,8 s que se pagan una sola
    vez: sin cachear se pagarían en cada frase.
    """
    audio, _ = _referencia()
    return precargar().encode_reference(audio)


def calentar() -> None:
    """Prepara todo lo que no debe pagarse con el visitante delante.

    Cargar los pesos no basta: hay que codificar el audio de referencia y poner
    en marcha los kernels de torch. Aquí se paga una vez, y de paso se graban
    las muletillas que falten, que sirven de calentamiento.
    """
    _codigos_referencia()
    if not grabar_muletillas() + grabar_frecuentes():
        sintetizar("Ay.")  # nada que grabar: hay que calentar igual


def _grabar(directorio, textos: list[str]) -> int:
    """Sintetiza a disco los trozos que falten o que sonaran mal. Devuelve
    cuántos grabó.

    Un archivo por *frase*, no por texto, porque el servidor los manda
    troceados: así empieza a sonar en cuanto está la primera en vez de esperar
    a tener los veinte segundos enteros.

    Cada archivo se graba con `_mejor_toma`, que repite la frase hasta que sale
    bien, y su nota queda anotada al lado. Un archivo ya grabado sin nota viene
    de antes de que esto existiera: se puntúa una vez y sólo se rehace si está
    mal, para que las tomas con voz de hombre que ya están en el volumen del
    museo se arreglen solas en el siguiente arranque.
    """
    directorio.mkdir(parents=True, exist_ok=True)
    notas = _notas(directorio)
    grabados = 0
    for i, texto in enumerate(textos):
        for j, frase in enumerate(trozos(texto)):
            destino = directorio / f"{i}-{j}.wav"
            if destino.exists():
                if destino.name in notas:
                    continue
                if (nota := _puntuar(_pcm(destino), frase)) >= GRABADO_ACEPTABLE:
                    _anotar(directorio, notas, destino.name, nota)
                    continue
                log.info("rehago %s: la toma que había puntuó %.2f", destino.name, nota)
            pcm, nota = _mejor_toma(frase)
            destino.write_bytes(a_wav(pcm))
            _anotar(directorio, notas, destino.name, nota)
            grabados += 1
    return grabados


# --- Calidad de lo grabado ---------------------------------------------------
#
# NeuTTS falla por muestreo, no por configuración: la misma frase sale bien o
# sale con otra voz según la semilla. En vivo eso se aguanta —el visitante está
# esperando— pero en lo grabado no hay por qué: se repite la frase y se escoge.
# Esto es lo que reemplaza al `nfe_step` de F5-TTS, que ya no existe.


def _mejor_toma(texto: str) -> tuple[bytes, float]:
    """Sintetiza el texto varias veces y devuelve la mejor toma y su nota.

    Para en cuanto una toma llega a GRABADO_ACEPTABLE, así que las frases que
    salen bien a la primera —la mayoría— cuestan una sola síntesis y el
    arranque no cambia. Las que salen mal cuestan hasta GRABADO_INTENTOS, que
    es exactamente el trato que se quiere: tiempo de arranque a cambio de que
    no haya un hombre contestando por ella.

    Si ninguna toma llega, devuelve la menos mala y lo deja dicho en el log:
    esa frase hay que oírla y probablemente reescribirla.
    """
    mejor, mejor_nota = b"", -1.0
    for intento in range(max(GRABADO_INTENTOS, 1)):
        # Semilla distinta en cada intento: con la fija, NeuTTS es determinista
        # y repetir la frase devolvería exactamente el mismo error.
        pcm = sintetizar(texto, temperatura=GRABADO_TEMPERATURA, semilla=SEED + intento)
        nota = _puntuar(pcm, texto)
        if nota > mejor_nota:
            mejor, mejor_nota = pcm, nota
        if mejor_nota >= GRABADO_ACEPTABLE:
            break
    else:
        log.warning("la mejor de %d tomas de %r se quedó en %.2f: óyela",
                    max(GRABADO_INTENTOS, 1), texto, mejor_nota)
    return mejor, mejor_nota


def _puntuar(pcm: bytes, texto: str) -> float:
    """Cuánto se parece una toma a ella diciendo ese texto. De 0 a 1.

    No entiende lo que oye —para eso haría falta un reconocedor— así que mira
    las dos cosas que delatan una toma mala sin escucharla:

    - **El tono.** Cuando el modelo se despega de la referencia se va a otro
      hablante, y lo que sale suele estar una octava más abajo: es la «voz de
      hombre». Se compara la fundamental mediana con la de `referencia.wav`,
      en semitonos, porque lo que importa es la distancia relativa.
    - **El largo.** A CARACTERES_POR_SEGUNDO se sabe cuánto debería durar el
      texto. La mitad de eso es una frase cortada a media palabra; el doble es
      el modelo divagando después de haber terminado.

    Las dos son campanas: 1,0 es clavado y cae suave a los lados, con tres
    semitonos de margen —ella misma varía— y con un 40 % de margen en el largo.
    Se multiplican, así que fallar en cualquiera de las dos hunde la nota.

    Lo que no pilla: una toma que diga otra cosa con su voz y en el tiempo
    justo. Eso hay que oírlo.
    """
    x = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768
    if not len(x) or not float(np.sqrt((x**2).mean())):
        return 0.0

    esperado = max(len(texto) / CARACTERES_POR_SEGUNDO, 0.5)
    largo = len(x) / SAMPLE_RATE / esperado
    factor_largo = float(np.exp(-(np.log(largo) ** 2) / (2 * 0.35**2)))

    tono, referencia = _tono(x), _tono_referencia()
    if not tono:
        return 0.0
    if not referencia:
        # Sin referencia medible no hay con qué comparar; se puntúa sólo el
        # largo antes que rechazarlo todo y dejarla muda.
        return factor_largo
    semitonos = 12 * np.log2(tono / referencia)
    factor_tono = float(np.exp(-(semitonos**2) / (2 * 3.0**2)))

    return factor_tono * factor_largo


@lru_cache(maxsize=1)
def _tono_referencia() -> float:
    """La fundamental de la voz que se está clonando, o 0 si no se pudo medir.

    Es el patrón contra el que se compara cada toma, así que se mide del mismo
    archivo que oye el modelo y con el mismo estimador: un sesgo del método se
    va en la resta.
    """
    audio, _ = _referencia()
    x, sr = _leer_wav(REF_AUDIO)
    if not len(x):
        log.warning("no pude medir el tono de %s: las tomas se juzgarán sólo "
                    "por su largo", audio)
        return 0.0
    return _tono(x, sr)


def _tono(x: np.ndarray, sr: int = SAMPLE_RATE) -> float:
    """Frecuencia fundamental mediana del habla que haya en `x`, en Hz, o 0 si
    no hay ninguna (silencio, o ruido sin periodicidad).

    Es YIN, resumido: por cada marco se busca el desfase que hace que la señal
    se parezca más a sí misma, y ese desfase es el período. El paso que le da
    el nombre —dividir la diferencia por su media acumulada— es lo que evita
    que conteste media frecuencia o el doble, que es justo el error que aquí
    saldría caro: confundiría su voz con la de un hombre.

    A mano y no con `librosa.yin` para que esto se pueda probar sin cargar el
    stack de voz entero; son veinte líneas de numpy.
    """
    if len(x) < MARCO:
        return 0.0
    mitad = MARCO // 2
    n = 1 + (len(x) - MARCO) // mitad
    marcos = x[np.arange(MARCO)[None, :] + mitad * np.arange(n)[:, None]]

    # Los silencios contestan cualquier cosa: sólo se miran los marcos que suenan.
    energia = np.sqrt((marcos**2).mean(axis=1))
    suenan = energia > max(energia.max() * 0.3, 1e-4)
    if not suenan.any():
        return 0.0
    marcos = marcos[suenan]

    # d(t) = suma de (x[n] - x[n+t])^2 sobre la primera mitad del marco, que es
    # lo mismo que las dos energías menos el doble de la correlación. Por FFT
    # para no pagar un bucle por cada desfase.
    tam = 1 << int(np.ceil(np.log2(MARCO + mitad)))
    corr = np.fft.irfft(np.fft.rfft(marcos, tam)
                        * np.conj(np.fft.rfft(marcos[:, :mitad], tam)), tam)[:, :mitad]
    cuadrados = np.cumsum(marcos**2, axis=1)
    desplazada = (cuadrados[:, mitad - 1:MARCO - 1]
                  - np.hstack([np.zeros((len(marcos), 1)), cuadrados[:, :mitad - 1]]))
    d = cuadrados[:, mitad - 1][:, None] + desplazada - 2 * corr

    # La normalización acumulada de YIN: sin ella, el desfase 0 siempre gana.
    desfases = np.arange(1, mitad)
    normalizada = d[:, 1:] * desfases / np.maximum(np.cumsum(d, axis=1)[:, 1:], 1e-12)

    # Los desfases que caen dentro del rango de una voz: el más corto es el del
    # tono más agudo que se admite, y el más largo el del más grave.
    agudo, grave = max(int(sr / TONO_MAXIMO), 1), min(int(sr / TONO_MINIMO), mitad - 1)
    tramo = normalizada[:, agudo:grave + 1]
    bajo_umbral = tramo < 0.15
    empieza = np.where(bajo_umbral.any(axis=1), bajo_umbral.argmax(axis=1), tramo.argmin(axis=1))
    # El umbral lo cruza la bajada, no el fondo del hoyo: sin bajar hasta el
    # mínimo, el tono sale medio semitono alto.
    cerca = np.clip(empieza[:, None] + np.arange(12)[None, :], 0, tramo.shape[1] - 1)
    elegido = np.take_along_axis(tramo, cerca, axis=1).argmin(axis=1) + empieza + agudo
    return float(np.median(sr / elegido))


def _notas(directorio) -> dict[str, float]:
    """Qué puntuó cada archivo ya grabado.

    Existe para dos cosas: no volver a puntuar en cada arranque lo que ya se
    revisó, y no repetir eternamente una frase que sale mal por mucho que se
    insista —la nota mala también se anota, y esa frase se da por buena hasta
    que alguien la cambie—. Si el archivo falta o está roto se puntúa todo otra
    vez: cuesta un rato de arranque y no se pierde nada.
    """
    try:
        return json.loads((directorio / NOTAS).read_text())
    except (OSError, ValueError):
        return {}


def _anotar(directorio, notas: dict[str, float], nombre: str, nota: float) -> None:
    notas[nombre] = round(nota, 3)
    # Se escribe frase a frase y no al final a propósito: al contenedor lo mata
    # la memoria a mitad del arranque más de una vez, y lo ya revisado no
    # debería revisarse de nuevo.
    (directorio / NOTAS).write_text(json.dumps(notas, indent=1, sort_keys=True))


def _pcm(ruta) -> bytes:
    """El PCM crudo de un WAV grabado."""
    with wave.open(str(ruta)) as w:
        return w.readframes(w.getnframes())


def _leer_wav(ruta) -> tuple[np.ndarray, int]:
    """Un WAV del disco como señal en float y su frecuencia de muestreo.

    Sólo entiende PCM de 16 bits, que es lo que produce la receta de
    `assets/voz/README.md` y lo que graba este módulo. Con cualquier otra cosa
    devuelve vacío en vez de reventar: esto sirve para juzgar audio, y quedarse
    sin juzgar es preferible a no arrancar.
    """
    try:
        with wave.open(str(ruta)) as w:
            if w.getsampwidth() != SAMPLE_WIDTH:
                return np.array([], np.float32), 0
            crudo = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
            if w.getnchannels() > 1:
                crudo = crudo.reshape(-1, w.getnchannels()).mean(axis=1)
            return crudo.astype(np.float32) / 32768, w.getframerate()
    except (OSError, wave.Error):
        return np.array([], np.float32), 0


def _leer(directorio, textos: list[str]) -> list[list[tuple[str, bytes]]]:
    """Lo grabado, como lista de trozos (frase, PCM) por texto.

    Sólo devuelve los completos: a uno al que le falte un trozo se le cortaría
    la historia a la mitad. Los incompletos salen como lista vacía para que el
    índice siga correspondiendo con `textos`.
    """
    out = []
    for i, texto in enumerate(textos):
        partes = []
        for j, frase in enumerate(trozos(texto)):
            ruta = directorio / f"{i}-{j}.wav"
            if not ruta.exists():
                partes = []
                break
            partes.append((frase, _pcm(ruta)))
        out.append(partes)
    return out


def grabar_muletillas() -> int:
    """Graba las muletillas que falten o que sonaran mal. Son textos fijos:
    generarlos en cada pregunta sería añadir espera a la espera que vienen a
    tapar.
    """
    grabados = _grabar(MULETILLAS_DIR, MULETILLAS)
    if grabados:
        muletillas.cache_clear()
    return grabados


def grabar_frecuentes() -> int:
    """Graba las respuestas a las preguntas frecuentes que falten o que
    sonaran mal.

    Son 68 trozos la primera vez, unos cinco minutos más lo que cueste repetir
    los que salgan mal. A cambio, un visitante que toca un badge oye la
    respuesta al instante, sin pasar por el LLM y con la voz que corresponde.
    """
    grabados = _grabar(FRECUENTES_DIR, [r for _, r in FRECUENTES])
    if grabados:
        frecuentes.cache_clear()
    return grabados


@lru_cache(maxsize=1)
def muletillas() -> list[list[tuple[str, bytes]]]:
    """Las muletillas grabadas y completas. Cacheado: se consultan en cada
    pregunta y son unos cientos de kB.
    """
    return [partes for partes in _leer(MULETILLAS_DIR, MULETILLAS) if partes]


@lru_cache(maxsize=1)
def frecuentes() -> dict[str, list[tuple[str, bytes]]]:
    """Las frecuentes grabadas, por su pregunta. Sólo las completas: una a
    medio grabar se responde mejor con el LLM que a medias.
    """
    grabadas = _leer(FRECUENTES_DIR, [r for _, r in FRECUENTES])
    return {p: partes for (p, _), partes in zip(FRECUENTES, grabadas) if partes}


def trozos(texto: str) -> list[str]:
    """Parte el texto en pedazos que se sintetizan de una pieza.

    Corta en fin de frase, y a la fuerza si un trozo pasa de MAX_BYTES. F5-TTS
    traía su `chunk_text` y este es su reemplazo; el tope de 135 bytes se hereda
    de él porque el servidor manda cada trozo en cuanto está y trozos cortos
    adelantan la primera frase.

    ponytail: NeuTTS rinde *mejor* cuanto más largo el trozo (RTF 1,29 en una
    frase de 10 s contra 1,79 en una de 2 s), así que este tope cuesta algo de
    rendimiento total a cambio de latencia inicial. Si algún día la primera
    frase deja de importar, subirlo es la optimización gratis.
    """
    partes: list[str] = []
    actual: list[str] = []
    largo = 0
    for palabra in texto.split():
        b = len(palabra.encode())
        if actual and largo + 1 + b > MAX_BYTES:
            partes.append(" ".join(actual))
            actual, largo = [], 0
        largo += (1 if actual else 0) + b
        actual.append(palabra)
        if palabra.endswith((".", "!", "?", "…", ".»", ".\"")):
            partes.append(" ".join(actual))
            actual, largo = [], 0
    if actual:
        partes.append(" ".join(actual))
    return partes or [texto]


def sintetizar(texto: str, temperatura: float | None = None,
               semilla: int | None = None) -> bytes:
    """Devuelve PCM crudo (16-bit mono, 24 kHz).

    Los dos argumentos son para grabar, no para hablar en vivo: sin ellos sale
    lo de siempre. `semilla` pide *otra* toma del mismo texto —con la fija,
    NeuTTS devuelve el mismo audio, el mismo fallo incluido, por muchas veces
    que se le pregunte— y `temperatura` la pide más conservadora.
    """
    modelo = precargar()
    _, ref_texto = _referencia()
    # ponytail: NeuTTS sólo acepta la semilla al construirse y la guarda en
    # `_seed`; metérsela así evita rehacer el modelo —10,6 s de pesos— en cada
    # toma. Se devuelve a SEED al salir para que el camino en vivo, que no pasa
    # semilla, siga siendo palabra por palabra el de antes.
    modelo._seed = SEED if semilla is None else semilla
    try:
        wav = modelo.infer(
            texto, _codigos_referencia(), ref_texto,
            temperature=TEMPERATURA if temperatura is None else temperatura,
        )
    finally:
        modelo._seed = SEED
    x = np.asarray(wav, dtype=np.float32)
    # NeuTTS entrega más bajo que F5 y el nivel varía entre frases; sin esto la
    # muletilla suena a la mitad de volumen que la respuesta que la sigue.
    if rms := float(np.sqrt((x**2).mean())):
        x = x * (RMS_OBJETIVO / rms)
    return (x.clip(-1, 1) * 32767).astype("<i2").tobytes()


def a_wav(pcm: bytes) -> bytes:
    """Envuelve el PCM en una cabecera WAV para que el navegador lo reproduzca."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(SAMPLE_WIDTH)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm)
    return buf.getvalue()


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(description="Sintetiza una frase con su voz.")
    p.add_argument("texto")
    p.add_argument("--out", default="/tmp/gabriela.wav")
    p.add_argument("--grabado", action="store_true",
                   help="como se graban las muletillas y las frecuentes: varias "
                        "tomas y se queda la mejor. Tarda más y suena mejor.")
    a = p.parse_args()
    if a.grabado:
        pcm, nota = _mejor_toma(a.texto)
        print(f"nota {nota:.2f} (se da por buena desde {GRABADO_ACEPTABLE})")
    else:
        pcm = sintetizar(a.texto)
    open(a.out, "wb").write(a_wav(pcm))
    print(f"escrito {a.out}")
