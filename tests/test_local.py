"""Comprobaciones de las dos piezas locales. Sin framework: `python tests/test_local.py`.

Ninguna carga NeuTTS ni necesita llama-server levantado: lo que se prueba es la
lógica que rodea a ambos —historial, saneado de la respuesta, conversión a PCM—,
que es donde puede romperse en silencio.
"""
import json
import sys
import threading
import wave
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

RESPUESTA = "<think>divagaciones internas</think>\nLa piedra también enseña."


class _Falso(BaseHTTPRequestHandler):
    """Imita /v1/chat/completions de llama-server y guarda lo que recibió."""

    recibido = None
    autorizacion = None
    contenido = RESPUESTA
    motivo = "stop"
    codigo = 200

    def do_POST(self):
        _Falso.autorizacion = self.headers.get("Authorization")
        _Falso.recibido = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if _Falso.codigo != 200:
            self.send_response(_Falso.codigo)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        datos = json.dumps({"choices": [
            {"message": {"content": _Falso.contenido}, "finish_reason": _Falso.motivo}
        ]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(datos)))
        self.end_headers()
        self.wfile.write(datos)

    def log_message(self, *a):
        pass


_srv = HTTPServer(("127.0.0.1", 0), _Falso)
threading.Thread(target=_srv.serve_forever, daemon=True).start()

import os  # noqa: E402  (la URL tiene que estar puesta antes de importar config)

os.environ["LLM_URL"] = f"http://127.0.0.1:{_srv.server_address[1]}/v1/chat/completions"

from gabriela.chat import Conversacion  # noqa: E402
from gabriela.voice import SAMPLE_RATE, a_wav  # noqa: E402


def test_manda_la_persona_y_acumula_historial():
    c = Conversacion()
    c.responder("¿Quién eres?")
    c.responder("¿Y el agua?")
    roles = [m["role"] for m in _Falso.recibido["messages"]]
    assert roles == ["system", "user", "assistant", "user"], roles
    assert "Gabriela Mistral" in _Falso.recibido["messages"][0]["content"]


def test_quita_el_bloque_de_razonamiento():
    # Si se cuela, el TTS lee las divagaciones del modelo en voz alta.
    dicho = Conversacion().responder("hola")
    assert dicho == "La piedra también enseña.", repr(dicho)


def test_manda_la_clave_solo_si_la_hay():
    # Se fija el valor a mano: si no, el .env del desarrollador decide el
    # resultado del test, y pasa o falla según quién lo ejecute.
    import gabriela.chat as chat

    real = chat.LLM_API_KEY
    try:
        chat.LLM_API_KEY = ""
        Conversacion().responder("hola")
        assert _Falso.autorizacion is None, "sin clave no debe mandarse cabecera"

        chat.LLM_API_KEY = "clave-de-prueba"
        Conversacion().responder("hola")
        assert _Falso.autorizacion == "Bearer clave-de-prueba", _Falso.autorizacion
    finally:
        chat.LLM_API_KEY = real


def test_la_cuota_agotada_se_nombra():
    # Un 429 mudo se confunde con un fallo de red y cuesta media hora.
    _Falso.codigo = 429
    try:
        Conversacion().responder("hola")
        raise AssertionError("debió fallar")
    except RuntimeError as e:
        assert "429" in str(e), e
    finally:
        _Falso.codigo = 200


def test_respuesta_truncada_acaba_en_frase_completa():
    # Al tope de tokens el modelo corta a media palabra; hablado suena a fallo.
    _Falso.contenido = "La piedra enseña. El agua tam"
    _Falso.motivo = "length"
    try:
        assert Conversacion().responder("hola") == "La piedra enseña."
    finally:
        _Falso.contenido, _Falso.motivo = RESPUESTA, "stop"


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


def test_muletillas_se_leen_y_no_se_regraban():
    # Si grabar_muletillas() no respetara lo ya grabado, cada arranque del
    # contenedor costaría cuatro síntesis de más.
    import tempfile
    import wave as W

    import gabriela.voice as voz

    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        original, voz.MULETILLAS_DIR = voz.MULETILLAS_DIR, d
        voz.muletillas.cache_clear()
        try:
            assert voz.muletillas() == [], "sin archivos no debe haber muletillas"
            # Se fabrican a mano para no invocar a NeuTTS en un test.
            for i in range(len(voz.MULETILLAS)):
                with W.open(str(d / f"{i}.wav"), "wb") as w:
                    w.setnchannels(1); w.setsampwidth(2); w.setframerate(24_000)
                    w.writeframes(b"\x00\x00" * 2400)
            voz.muletillas.cache_clear()
            leidas = voz.muletillas()
            assert len(leidas) == len(voz.MULETILLAS), leidas
            assert leidas[0][0] == voz.MULETILLAS[0], "la frase debe acompañar al audio"
            assert voz.grabar_muletillas() == 0, "no debe regrabar lo que ya existe"
        finally:
            voz.MULETILLAS_DIR = original
            voz.muletillas.cache_clear()


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
