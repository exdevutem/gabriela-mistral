"""Comprobaciones de la lógica local. Sin framework: `python tests/test_local.py`.

Ninguna carga NeuTTS, Whisper ni el modelo de embeddings: lo que se prueba es
la lógica que los rodea —elegir respuesta, lo grabado, conversión a PCM—, que
es donde puede romperse en silencio.
"""
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

from gabriela.voice import SAMPLE_RATE, a_wav  # noqa: E402


def test_wav_es_el_formato_que_espera_el_navegador():
    pcm = (np.sin(np.linspace(0, 400, SAMPLE_RATE)) * 16000).astype("<i2").tobytes()
    with wave.open(__import__("io").BytesIO(a_wav(pcm))) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, 24_000)
        assert w.getnframes() == SAMPLE_RATE, "se perdieron muestras"


def test_el_troceo_no_pierde_ni_desborda():
    # Si trozos() devolviera [], el servidor no mandaría audio y ella se
    # quedaría muda sin que nada fallara.
    from gabriela.voice import trozos

    texto = ("Le diría que la escuela no es una cárcel, sino una ventana. "
             "No lo castigues por su miedo, escúchalo. Porque el niño no odia "
             "aprender, odia sentirse pequeño o ignorado por los suyos.")
    partes = trozos(texto)
    assert partes, "sin trozos no hay voz"
    assert all(len(p.encode()) <= 135 for p in partes), [len(p) for p in partes]
    # Las palabras tienen que seguir estando, y en orden. Se unen con espacio
    # porque chunk_text recorta cada trozo por los bordes.
    assert " ".join(partes).split() == texto.split()
    assert trozos("Ay.") == ["Ay."], "una frase corta no debe trocearse"


def test_las_preguntas_frecuentes_no_se_repiten():
    # Se indexan por su texto: dos iguales se pisarían, y el badge de la
    # segunda respondería lo de la primera sin que nada fallara.
    from gabriela.config import FRECUENTES

    preguntas = [p for p, _ in FRECUENTES]
    assert len(set(preguntas)) == len(preguntas), "hay preguntas repetidas"
    assert len(preguntas) >= 10, f"sólo {len(preguntas)} frecuentes"
    for p in preguntas:
        assert p.endswith("?"), f"el badge no es una pregunta: {p!r}"
        assert len(p) <= 40, f"badge demasiado largo para la fila: {p!r}"


def test_una_frecuente_a_medias_no_se_ofrece():
    # Un badge que lleva a media respuesta es peor que no ofrecerlo: mejor que
    # esa pregunta caiga en el LLM, que al menos contesta entera.
    import tempfile
    import wave as W

    import gabriela.voice as voz
    from gabriela.config import FRECUENTES

    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        original, voz.FRECUENTES_DIR = voz.FRECUENTES_DIR, d
        voz.frecuentes.cache_clear()
        try:
            # La primera, completa; la segunda, a la que le falta un trozo.
            for i in (0, 1):
                trozos_i = voz.trozos(FRECUENTES[i][1])
                if i == 1:
                    trozos_i = trozos_i[:-1]
                for j in range(len(trozos_i)):
                    with W.open(str(d / f"{i}-{j}.wav"), "wb") as w:
                        w.setnchannels(1); w.setsampwidth(2); w.setframerate(24_000)
                        w.writeframes(b"\x00\x00" * 2400)
            voz.frecuentes.cache_clear()
            ofrecidas = voz.frecuentes()
            assert FRECUENTES[0][0] in ofrecidas, "la completa debe ofrecerse"
            assert FRECUENTES[1][0] not in ofrecidas, "la incompleta no"
        finally:
            voz.FRECUENTES_DIR = original
            voz.frecuentes.cache_clear()


def test_lo_grabado_se_lee_y_no_se_regraba():
    # Si _grabar() no respetara lo ya grabado y revisado, cada arranque del
    # contenedor cargaría NeuTTS y volvería a sintetizar todo.
    import json
    import tempfile
    import wave as W

    import gabriela.voice as voz

    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        original, voz.RESPALDO_DIR = voz.RESPALDO_DIR, d
        voz.respaldo.cache_clear()
        try:
            assert voz.respaldo() == [], "sin archivos no hay respaldo"
            frases = voz.trozos(voz.RESPALDO)
            for j in range(len(frases)):
                with W.open(str(d / f"0-{j}.wav"), "wb") as w:
                    w.setnchannels(1); w.setsampwidth(2); w.setframerate(24_000)
                    w.writeframes(b"\x00\x00" * 2400)
            voz.respaldo.cache_clear()
            assert [f for f, _ in voz.respaldo()] == frases, "cada trozo con su frase"
            (d / voz.NOTAS).write_text(json.dumps({r.name: 0.9 for r in d.glob("*.wav")}))
            assert voz.grabar_respaldo() == 0, "no debe regrabar lo ya revisado"
        finally:
            voz.RESPALDO_DIR = original
            voz.respaldo.cache_clear()


# --- Calidad de lo pre-grabado ---
#
# Lo que se prueba aquí es el juez, no el modelo: que sepa distinguir una toma
# con su voz de una con voz de hombre, y una frase entera de una cortada. Si el
# juez se equivoca, o se rechaza para siempre audio que estaba bien, o se dan
# por buenas las tomas que el visitante oye mal.


def _habla(f0, segundos=2.0, semilla=0):
    """Algo que se parece a una voz a f0 hercios: armónicos que caen, un poco
    de vibrato y algo de ruido. Devuelve PCM como el que produce sintetizar().
    """
    rng = np.random.default_rng(semilla)
    t = np.arange(int(segundos * 24_000)) / 24_000
    fase = 2 * np.pi * np.cumsum(f0 * (1 + 0.01 * np.sin(2 * np.pi * 3 * t))) / 24_000
    x = sum(np.sin(k * fase) / k for k in range(1, 25))
    x = x * (1 + 0.3 * np.sin(2 * np.pi * 2 * t)) + 0.02 * rng.standard_normal(len(t))
    return (x / np.abs(x).max() * 0.5 * 32767).astype("<i2").tobytes()


def test_el_tono_no_se_equivoca_de_octava():
    # Es el error clásico de medir un período, y aquí saldría caro: contestar
    # la mitad convertiría su voz en la de un hombre y se rechazarían tomas
    # buenas para siempre.
    from gabriela.voice import _tono

    for f0 in (98, 110, 150, 196, 220, 260, 330):
        medido = _tono(np.frombuffer(_habla(f0), dtype="<i2").astype(np.float32) / 32768)
        assert abs(medido - f0) / f0 < 0.05, f"{f0} Hz medidos como {medido:.1f}"


def test_el_silencio_no_tiene_tono():
    # Una toma muda puntuaría lo que fuera si esto contestara un número.
    from gabriela.voice import _tono

    assert _tono(np.zeros(24_000, np.float32)) == 0.0


def test_la_voz_de_hombre_no_pasa_y_la_suya_si():
    # El fallo que se vio en el museo: un badge contestado por otra persona.
    import gabriela.voice as voz

    texto = "a" * 26  # 26 caracteres a 13 por segundo: dos segundos de audio
    real, voz._tono_referencia = voz._tono_referencia, lambda: 210.0
    try:
        assert voz._puntuar(_habla(205), texto) > 0.9, "es ella"
        assert voz._puntuar(_habla(228), texto) > 0.6, "también es ella, otro día"
        assert voz._puntuar(_habla(110), texto) < 0.1, "eso es un hombre"
    finally:
        voz._tono_referencia = real


def test_la_frase_cortada_y_la_divagacion_no_pasan():
    # Las otras dos formas de salir mal: el modelo se calla a media palabra o
    # sigue hablando después de terminar.
    import gabriela.voice as voz

    texto = "a" * 26
    real, voz._tono_referencia = voz._tono_referencia, lambda: 210.0
    try:
        assert voz._puntuar(_habla(210, 2.4), texto) > 0.6, "un 20% de más es normal"
        assert voz._puntuar(_habla(210, 1.0), texto) < 0.3, "se cortó a la mitad"
        assert voz._puntuar(_habla(210, 5.0), texto) < 0.3, "siguió divagando"
        assert voz._puntuar(b"", texto) == 0.0, "no hay audio"
    finally:
        voz._tono_referencia = real


def test_se_repite_la_frase_hasta_que_sale_bien():
    # El trato entero: en lo grabado se paga en tiempo lo que en vivo se paga
    # en suerte. Y se para en cuanto sale bien, o el arranque se iría al doble.
    import gabriela.voice as voz

    tomas = [_habla(110), _habla(110), _habla(210)]  # dos malas y una buena
    semillas = []

    def falso(texto, temperatura=None, semilla=None):
        semillas.append(semilla)
        return tomas[len(semillas) - 1]

    real_s, voz.sintetizar = voz.sintetizar, falso
    real_t, voz._tono_referencia = voz._tono_referencia, lambda: 210.0
    real_m, voz.MOTOR = voz.MOTOR, "neutts"  # el que clona y se puede ir a otra voz
    try:
        pcm, nota = voz._mejor_toma("a" * 26)
        assert pcm == tomas[2], "se quedó con una toma mala"
        assert nota > 0.6, nota
        assert len(semillas) == 3, f"debió parar en la buena, hizo {len(semillas)}"
        assert len(set(semillas)) == 3, f"semillas repetidas: {semillas}"
    finally:
        voz.sintetizar, voz._tono_referencia, voz.MOTOR = real_s, real_t, real_m


def test_lo_que_sonaba_mal_se_rehace_y_lo_bueno_se_deja():
    # Lo grabado vive en un volumen que sobrevive al despliegue, así que las
    # tomas malas de antes hay que arreglarlas sin que nadie entre a borrar.
    import json
    import tempfile

    import gabriela.voice as voz

    texto = "Cada frase es un archivo. Esta es la segunda."
    frases = voz.trozos(texto)
    assert len(frases) == 2, frases

    hechas = []

    def falso(t, temperatura=None, semilla=None):
        hechas.append(t)
        return _habla(210, len(t) / 13)

    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        real_s, voz.sintetizar = voz.sintetizar, falso
        real_t, voz._tono_referencia = voz._tono_referencia, lambda: 210.0
        real_m, voz.MOTOR = voz.MOTOR, "neutts"
        try:
            # La primera ya está y suena a hombre; la segunda ya está y está bien.
            (d / "0-0.wav").write_bytes(voz.a_wav(_habla(110, len(frases[0]) / 13)))
            (d / "0-1.wav").write_bytes(voz.a_wav(_habla(210, len(frases[1]) / 13)))

            assert voz._grabar(d, [texto]) == 1, "sólo debía rehacerse la mala"
            assert hechas == [frases[0]], hechas
            notas = json.loads((d / "notas.json").read_text())
            assert set(notas) == {"0-0.wav", "0-1.wav"}, notas
            assert min(notas.values()) > 0.6, notas

            # Con las notas puestas, un segundo arranque no vuelve a puntuar.
            hechas.clear()
            assert voz._grabar(d, [texto]) == 0 and not hechas, hechas
        finally:
            voz.sintetizar, voz._tono_referencia, voz.MOTOR = real_s, real_t, real_m



def test_piper_sale_a_24_khz_y_entera():
    # Piper entrega 22 050 Hz y a veces en varios trozos. Todo lo de después
    # —visor, visemas, lo ya grabado— supone 24 kHz: un trozo perdido o un
    # remuestreo mal hecho se oye como una voz acelerada o cortada.
    from types import SimpleNamespace

    import gabriela.voice as voz

    uno = np.sin(np.linspace(0, 300, 22_050)).astype(np.float32) * 0.5
    trozos = [SimpleNamespace(audio_float_array=uno, sample_rate=22_050)] * 2
    real, voz._piper = voz._piper, lambda: SimpleNamespace(synthesize=lambda t: trozos)
    try:
        x = voz._sintetizar_piper("dos segundos")
    finally:
        voz._piper = real
    assert len(x) == 2 * SAMPLE_RATE, f"{len(x)} muestras en vez de {2 * SAMPLE_RATE}"


def test_gemini_pide_pcm_y_lo_decodifica_entero():
    # La Interactions API devuelve el audio en base64 dentro de los pasos del
    # modelo. Pedir otro formato (los 3.8 dan WAV por defecto) colaría la
    # cabecera RIFF como un chasquido al principio de cada frase.
    import base64
    from types import SimpleNamespace

    import httpx

    import gabriela.voice as voz

    pcm = (np.sin(np.linspace(0, 300, 24_000)) * 8000).astype("<i2").tobytes()
    pedido = {}

    def falso(url, headers, json, timeout):
        pedido.update(json)
        datos = {"steps": [{"type": "thinking"}, {"type": "model_output", "content": [
            {"type": "audio", "data": base64.b64encode(pcm[:24_000]).decode()},
            {"type": "audio", "data": base64.b64encode(pcm[24_000:]).decode()}]}]}
        return SimpleNamespace(status_code=200, json=lambda: datos, text="")

    real, httpx.post = httpx.post, falso
    try:
        x = voz._sintetizar_gemini("Hola.")
    finally:
        httpx.post = real
    assert pedido["response_format"] == {"type": "audio", "mime_type": "audio/l16",
                                         "sample_rate": 24_000}, pedido
    assert len(x) == 24_000, f"{len(x)} muestras: se perdió o sobró audio"


def test_gemini_graba_una_sola_toma_y_espera_la_cuota():
    # Con Gemini el juez de tono no sirve —compara con referencia.wav, que es
    # otra voz— y cada toma de más es cuota gratuita gastada. Un 429 al grabar
    # se espera en vez de dejar la frecuente sin grabar.
    import gabriela.voice as voz

    llamadas = []

    def falso(texto, temperatura=None, semilla=None):
        llamadas.append(texto)
        if len(llamadas) == 1:
            raise voz.CuotaAgotada("429")
        return b"\x01\x00" * 10

    reales = voz.sintetizar, voz.MOTOR
    voz.sintetizar, voz.MOTOR = falso, "gemini"
    try:
        pcm = voz.con_paciencia("Hola.", esperas=(0,))
        assert pcm == b"\x01\x00" * 10 and len(llamadas) == 2, llamadas
        llamadas.clear()
        voz.sintetizar = lambda t, temperatura=None, semilla=None: llamadas.append(t) or b"\x00\x00"
        assert voz._mejor_toma("Hola.") == (b"\x00\x00", 1.0)
        assert len(llamadas) == 1, f"{len(llamadas)} tomas en vez de una"
    finally:
        voz.sintetizar, voz.MOTOR = reales


# --- Entender la pregunta ---


def test_las_variantes_apuntan_a_preguntas_que_existen():
    # Una clave mal copiada no falla: sus variantes simplemente no cuentan, y
    # la pregunta que debían rescatar sigue cayendo en otra.
    from gabriela.config import FRECUENTES, VARIANTES

    sobran = set(VARIANTES) - {p for p, _ in FRECUENTES}
    assert not sobran, f"variantes de preguntas que no existen: {sobran}"


def test_elige_la_mas_parecida_entre_las_grabadas():
    from gabriela.entender import mas_parecida

    # Tres textos, de las preguntas 0, 1 y 1 (pregunta, respuesta, variante).
    vectores = np.array([[1, 0, 0], [0, 1, 0], [0, .6, .8]], dtype=float)
    duenos = np.array([0, 1, 1])
    consulta = np.array([0, .6, .8])

    assert mas_parecida(consulta, vectores, duenos, [True, True])[0] == 1
    # La 1 no está grabada: no puede elegirse aunque sea la mejor, y la 0 no
    # se le parece lo bastante, así que no hay respuesta.
    assert mas_parecida(consulta, vectores, duenos, [True, False])[0] is None
    # Bajo el umbral no se contesta, por cerca que esté la mejor.
    assert mas_parecida(np.array([.4, .4, 0]), vectores, duenos, [True, True],
                        umbral=0.5)[0] is None


def test_el_silencio_no_llega_a_whisper():
    # Whisper en silencio alucina créditos de subtítulos, y el elector podría
    # encontrarles parecido con algo.
    import gabriela.entender as e

    def no_debe(*a):
        raise AssertionError("cargó Whisper")

    real, e._whisper = e._whisper, no_debe
    try:
        assert e.transcribir(b"") == ""
        assert e.transcribir(b"\x00\x00" * 16_000) == ""
    finally:
        e._whisper = real
    assert e.elegir("   ", set()) is None


def test_la_respuesta_va_en_un_audio_con_la_boca_en_su_sitio():
    # El reproductor muestra el tiempo de la respuesta entera, así que las
    # frases se mandan juntas. Si la boca de la segunda no se corriera a donde
    # empieza la segunda, ella la diría con la boca de la primera.
    import base64
    import io

    from gabriela.server import _habla

    un_segundo = (np.sin(np.arange(SAMPLE_RATE) / 3) * 9000).astype("<i2").tobytes()
    m = _habla([("Ma.", un_segundo), ("Pa.", un_segundo)])
    with wave.open(io.BytesIO(base64.b64decode(m["audio"]))) as w:
        assert w.getnframes() == 2 * SAMPLE_RATE, "se perdió o sobró audio"
    assert m["duracion"] == 2.0 and m["fin"], m["duracion"]
    tiempos = [t for t, _, _ in m["visemas"]]
    assert tiempos == sorted(tiempos), "la boca retrocede"
    assert any(t >= 1.0 for t in tiempos), "la segunda frase quedó sin boca"


def test_la_voz_en_vivo_no_pierde_palabras_y_arranca_corto():
    # La primera unidad es lo que hay que esperar para oírla: si fuera la frase
    # entera, el primer sonido tardaría lo que tarde la más larga.
    from gabriela.vivo import PRIMERA_MAX, unidades

    texto = ("Nací en Vicuña, en el valle de Elqui, y fui maestra rural antes "
             "que poeta. Enseñar es también una forma de cantar.")
    partes = unidades(texto)
    assert " ".join(partes).split() == texto.split(), partes
    assert len(partes[0]) <= PRIMERA_MAX, partes[0]
    assert unidades("Ay.") == ["Ay."], "una frase corta no se parte"
    assert unidades("Hola, buenas tardes y bienvenidos.") == ["Hola, buenas tardes y bienvenidos."], \
        "una coma temprana no debe dejar un trozo de dos palabras"


def test_la_voz_en_vivo_repite_la_toma_mala_pero_no_la_corta():
    import gabriela.vivo as vivo

    llamadas, tomas = [], iter([_habla(110, 3.5), _habla(210, 3.5)])  # voz de hombre y luego la suya

    def falsa(texto, temperatura, semilla):
        llamadas.append((temperatura, semilla))
        return np.frombuffer(next(tomas), "<i2").astype(np.float32) / 32768

    originales = vivo._una_toma, vivo.voice._tono_referencia
    vivo._una_toma, vivo.voice._tono_referencia = falsa, lambda: 210.0
    try:
        texto = "a" * 45
        pcm = vivo.sintetizar(texto)
        assert len(llamadas) == 2, llamadas
        assert llamadas[1][0] < llamadas[0][0] and llamadas[1][1] != llamadas[0][1], llamadas
        assert vivo.voice._puntuar(pcm, texto) > 0.6, "se quedó con la toma mala"
        # una frase corta no se juzga: una sola toma, sea cual sea
        llamadas.clear()
        tomas = iter([_habla(110)])
        vivo.sintetizar("Sí.")
        assert len(llamadas) == 1, llamadas
    finally:
        vivo._una_toma, vivo.voice._tono_referencia = originales


def test_decir_manda_una_habla_por_unidad_y_las_encadena():
    # El visor corta lo que suena con cada `habla` salvo las continuaciones:
    # si la segunda no lo fuera, pisaría a la primera.
    import asyncio

    import gabriela.server as server

    mandados = []

    class WS:
        async def send_json(self, m):
            mandados.append(m)

    un_segundo = (np.sin(np.arange(SAMPLE_RATE) / 3) * 9000).astype("<i2").tobytes()
    real = server.vivo.sintetizar
    server.vivo.sintetizar = lambda frase: un_segundo
    try:
        asyncio.run(server._decir(WS(), "Primera frase de prueba. Segunda frase de prueba."))
    finally:
        server.vivo.sintetizar = real
    hablas = [m for m in mandados if m["tipo"] == "habla"]
    assert [m["continuacion"] for m in hablas] == [False, True], hablas
    assert [m["fin"] for m in hablas] == [False, True]
    assert mandados[0]["tipo"] == "texto"


if __name__ == "__main__":
    fallos = 0
    for nombre, fn in sorted(globals().items()):
        if nombre.startswith("test_"):
            try:
                fn()
                print(f"  ok   {nombre}")
            except AssertionError as e:
                fallos += 1
                print(f"  FALLA {nombre}: {e}")
    print("todo verde" if not fallos else f"{fallos} fallo(s)")
    sys.exit(1 if fallos else 0)
