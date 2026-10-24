"""Lo que ella sabe más allá de las frecuentes. Todo local, sin red.

Cuando ninguna frecuente se parece a la pregunta (entender.elegir da None), en
vez de decir «no te entendí» de entrada se busca en los documentos oficiales
del museo (`assets/conocimiento/fragmentos.jsonl`, ver
pipeline/conocimiento.py) y un modelo de lenguaje muy pequeño, en GGUF sobre
llama.cpp, contesta con lo que dicen esos fragmentos:

1. Los fragmentos se buscan con el mismo modelo de embeddings que elige las
   frecuentes (entender._vectores): no se carga nada nuevo para buscar.
2. Si ninguno pasa de SABER_UMBRAL, la pregunta no es del museo ni de ella y
   suena el «no te entendí» grabado, como antes.
3. El modelo escribe dos o tres frases con los mejores fragmentos delante. Si
   dice que no está en ellos, o si nombra un número que los fragmentos no
   traen, la respuesta se descarta: en un museo, una fecha inventada es peor
   que un «no te entendí». Es el único dato que se puede comprobar sin
   entender el texto, y es justo el que un modelo pequeño inventa.

Con el switch «IA» de la página no pasa nada de esto: responder_ia() le
pregunta directo a Claude Haiku, sin fragmentos ni modelo local.

    uv run python -m gabriela.saber "¿qué es la exposición retrospectiva?"
    uv run python -m gabriela.saber --ia "¿qué opinas de los exámenes?"
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from functools import lru_cache

import numpy as np

from .config import (ANTHROPIC_KEY, CONOCIMIENTO, EMBEDDINGS_MODELO, IA_ESFUERZO, IA_MODELO,
                     LLM_CTX, LLM_DEVICE, LLM_FRAGMENTOS, LLM_GGUF, LLM_MAX_TOKENS,
                     SABER_MARGEN, SABER_UMBRAL)

log = logging.getLogger(__name__)

# Un contexto de llama.cpp, una generación a la vez.
_candado = threading.Lock()
# Lo que contesta el modelo cuando los fragmentos no dicen nada de la pregunta.
NO_SE = "NO_LO_SÉ"
SISTEMA = (
    "Eres Gabriela Mistral (Lucila Godoy Alcayaga, 1889-1957), maestra y poeta "
    "chilena. Un visitante del Museo de la Educación Gabriela Mistral, en "
    "Santiago, te hace una pregunta. Contesta en español, en primera persona, "
    "con dos o tres frases cortas y sencillas, que se van a decir en voz alta. "
    "Usa SÓLO lo que dicen los fragmentos; no agregues fechas, nombres ni cifras "
    "que no estén en ellos. No empieces con «Sí» ni repitas la pregunta, y no "
    "ofrezcas hacer nada por el visitante: sólo cuentas lo que sabes. "
    f"Si los fragmentos no responden la pregunta, contesta sólo {NO_SE}."
)
# Con el switch «IA» contesta Claude solo, con lo que sabe de ella: sin
# fragmentos. Lo que no se le deja es inventar.
SISTEMA_IA = (
    "Eres Gabriela Mistral (Lucila Godoy Alcayaga, 1889-1957), maestra y poeta "
    "chilena. Un visitante del Museo de la Educación Gabriela Mistral, en "
    "Santiago, te habla. Contesta en español de Chile, en primera persona, con "
    "una a tres frases cortas y sencillas, que se van a decir en voz alta: sin "
    "listas, sin formato, sin emojis, y los números escritos con letras («el año "
    "cuarenta y cinco»). Contesta con lo que se sabe con certeza de tu vida, tu "
    "obra y el museo. No inventes fechas, nombres ni cifras: si no lo sabes, "
    "dilo con tus palabras. Si te preguntan algo que no tiene que ver contigo ni con el museo, "
    "contéstalo en una frase, como lo haría ella, y vuelve a la escuela o al "
    "museo. No ofrezcas hacer nada por el visitante: sólo conversas. Trátalo "
    "siempre de tú, nunca de usted («mira», no «mire»), y sin suponer si es "
    "hombre o mujer (nada de «mijito» ni «mijita»)."
)


@lru_cache(maxsize=1)
def _fragmentos() -> list[dict]:
    if not CONOCIMIENTO.exists():
        return []
    return [json.loads(linea) for linea in CONOCIMIENTO.read_text(encoding="utf-8").splitlines()
            if linea.strip()]


@lru_cache(maxsize=1)
def _indice() -> np.ndarray:
    """Un vector por fragmento, cacheado en disco al lado del .jsonl.

    Calcularlos son unos segundos por cada mil fragmentos, que no hay por qué
    pagar en cada arranque. La caché lleva en el nombre la huella del texto y
    del modelo: si cambia cualquiera de los dos, se rehace sola.
    """
    from .entender import _vectores

    fragmentos = _fragmentos()
    huella = hashlib.sha1((EMBEDDINGS_MODELO + "\n" + CONOCIMIENTO.read_text(encoding="utf-8"))
                          .encode()).hexdigest()[:12]
    cache = CONOCIMIENTO.with_name(f"vectores-{huella}.npy")
    if cache.exists():
        return np.load(cache)
    t = time.time()
    # De a 64 para no pedirle memoria de más al tokenizer con miles de textos.
    v = np.vstack([_vectores([f["texto"] for f in fragmentos[i:i + 64]])
                   for i in range(0, len(fragmentos), 64)]).astype(np.float32)
    for viejo in CONOCIMIENTO.parent.glob("vectores-*.npy"):
        viejo.unlink()
    np.save(cache, v)
    log.info("%d fragmentos indexados en %.1f s", len(fragmentos), time.time() - t)
    return v


def piso(pregunta: str) -> float:
    """Lo que tiene que parecerse una frecuente para ganarle a los documentos.

    Un umbral fijo no alcanza: las preguntas nuevas sobre el museo se parecen
    a alguna frecuente tanto como sus paráfrasis («¿cómo eran las escuelas
    rurales?» da 0,72 con «¿Dónde enseñaste?»). Lo que sí las separa es el
    parecido con los documentos. Medido con 19 paráfrasis y 10 preguntas
    nuevas: las paráfrasis se parecen a su frecuente 0,12 o más por encima del
    mejor fragmento; las nuevas, 0,03 o menos. Entonces la frecuente sólo gana
    si le saca SABER_MARGEN al mejor fragmento.
    """
    encontrados = buscar(pregunta, 1)
    return encontrados[0][1] + SABER_MARGEN if encontrados else -1.0


def buscar(pregunta: str, k: int = LLM_FRAGMENTOS) -> list[tuple[dict, float]]:
    """Los `k` fragmentos más parecidos a la pregunta, con su parecido."""
    from .entender import _vectores

    if not _fragmentos():
        return []
    parecido = _indice() @ _vectores([pregunta])[0]
    mejores = np.argsort(-parecido)[:k]
    return [(_fragmentos()[i], float(parecido[i])) for i in mejores]


@lru_cache(maxsize=1)
def _modelo():
    """El LLM en GGUF: una ruta local o `repo:archivo` de Hugging Face, que se
    descarga la primera vez a su caché."""
    from llama_cpp import Llama

    ruta = LLM_GGUF
    if ":" in ruta and not ruta.startswith("/"):
        from huggingface_hub import hf_hub_download

        repo, archivo = ruta.split(":", 1)
        ruta = hf_hub_download(repo, archivo)
    t = time.time()
    # En CPU lo que cuesta es leer los fragmentos (~650 tokens), no escribir
    # (~30), y leer sí aprovecha todos los núcleos: medido en un M5, 10,0 s con
    # 4 hilos y 6,5 s con 8. Escribir se queda con los que elige llama.cpp.
    modelo = Llama(model_path=ruta, n_ctx=LLM_CTX, verbose=False,
                   n_gpu_layers=0 if LLM_DEVICE == "cpu" else -1,
                   n_threads_batch=os.cpu_count())
    log.info("LLM %s listo (%.1f s)", LLM_GGUF, time.time() - t)
    return modelo


def numeros(texto: str) -> set[str]:
    """Los números escritos con cifras, sin puntos de miles («1.941» = 1941)."""
    return {n.replace(".", "") for n in re.findall(r"\d[\d.]*\d|\d", texto)}


def comprobar(respuesta: str, contexto: str) -> str | None:
    """La respuesta limpia si se puede decir, o None si no.

    No se puede si el modelo dijo que no sabe, si quedó vacía, o si trae un
    número que no está en el contexto: ésa es la fecha inventada.
    """
    texto = " ".join(re.sub(r"<think>.*?</think>", "", respuesta, flags=re.S).split())
    texto = texto.strip("\"«» ")
    # Los modelos pequeños no siempre obedecen el NO_LO_SÉ y lo dicen a su
    # manera: «No lo sé.», «No sé qué pensaba de…».
    if not texto or re.search(r"NO_LO_S[EÉ]|^no (lo )?s[eé]\b", texto, re.I):
        return None
    if inventados := numeros(texto) - numeros(contexto):
        log.info("descarto «%s»: %s no está en los fragmentos", texto, inventados)
        return None
    return texto


@lru_cache(maxsize=1)
def _claude():
    import anthropic

    # Un visitante esperando: mejor el «no te entendí» grabado que un minuto de
    # reintentos.
    return anthropic.Anthropic(api_key=ANTHROPIC_KEY, timeout=15.0, max_retries=1)


def responder_ia(pregunta: str) -> str | None:
    """Lo que contesta Claude Haiku a la pregunta tal cual, o None si se negó o
    se cortó: media respuesta no se dice en voz alta.

    ponytail: sin comprobar(): escribe los números con letras, así que el filtro
    no tendría qué mirar. Si alguna vez inventa, se ajusta SISTEMA_IA.
    """
    t = time.time()
    r = _claude().messages.create(
        model=IA_MODELO, max_tokens=2000, system=SISTEMA_IA,
        output_config={"effort": IA_ESFUERZO},
        messages=[{"role": "user", "content": pregunta}])
    texto = " ".join("".join(b.text for b in r.content if b.type == "text").split())
    log.info("«%s» -> «%s» (%s, %s, %.1f s)", pregunta, texto, IA_MODELO, r.stop_reason,
             time.time() - t)
    return texto if r.stop_reason == "end_turn" and texto else None


def responder(pregunta: str) -> tuple[str | None, list[str]]:
    """(respuesta, urls de sus fuentes), o (None, []) si no hay nada que decir
    con seguridad sobre eso."""
    encontrados = [(f, p) for f, p in buscar(pregunta) if p >= SABER_UMBRAL]
    if not encontrados:
        log.info("«%s»: ningún fragmento pasa de %.2f", pregunta, SABER_UMBRAL)
        return None, []
    contexto = "\n\n".join(f"[{i}] {f['texto']}" for i, (f, _) in enumerate(encontrados, 1))
    mensajes = [
        {"role": "system", "content": SISTEMA},
        {"role": "user", "content": f"Fragmentos:\n{contexto}\n\nPregunta del visitante: {pregunta}"},
    ]
    t = time.time()
    with _candado:
        salida = _modelo().create_chat_completion(
            mensajes, max_tokens=LLM_MAX_TOKENS, temperature=0.2, top_p=0.9,
            repeat_penalty=1.1)
    crudo = salida["choices"][0]["message"]["content"] or ""
    texto = comprobar(crudo, contexto)
    log.info("«%s» -> «%s» (%.1f s, mejor fragmento %.2f)", pregunta, texto or crudo,
             time.time() - t, encontrados[0][1])
    fuentes = list(dict.fromkeys(f["url"] for f, _ in encontrados))
    return texto, (fuentes if texto else [])


def calentar() -> None:
    """Carga el índice y el modelo antes de la primera visita."""
    if not _fragmentos():
        log.warning("sin %s: sólo contestará las frecuentes "
                    "(uv run pipeline/conocimiento.py)", CONOCIMIENTO)
        return
    _indice()
    # La primera generación compila los kernels (Metal) y tarda segundos que
    # no deben caerle al primer visitante.
    with _candado:
        _modelo().create_chat_completion([{"role": "user", "content": "Hola"}], max_tokens=1)


def disponible() -> bool:
    return bool(_fragmentos()) and bool(LLM_GGUF)


def ia_disponible() -> bool:
    """Si se puede ofrecer el switch «IA»: basta con la clave."""
    return bool(ANTHROPIC_KEY)


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if sys.argv[1:2] == ["--ia"]:
        for p in sys.argv[2:]:
            print(f"{p!r} -> {responder_ia(p)}")
        sys.exit()
    for p in sys.argv[1:]:
        for f, parecido in buscar(p):
            print(f"  {parecido:.2f}  {f['url']}\n        {f['texto'][:110]}…")
        print(f"{p!r} -> {responder(p)}")
