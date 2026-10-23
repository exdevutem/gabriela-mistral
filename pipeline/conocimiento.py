# /// script
# requires-python = ">=3.12"
# dependencies = ["pypdf"]
# ///
"""Baja los documentos oficiales del museo y los deja en fragmentos para el LLM.

De aquí sale todo lo que el modelo local puede decir fuera de las frecuentes
(ver src/gabriela/saber.py): si no está en estos fragmentos, no lo sabe.

    uv run pipeline/conocimiento.py                  # sitio del museo + «La voz de la maestra»
    uv run pipeline/conocimiento.py --obra-reunida   # + los 8 tomos de su Obra reunida

Sale `assets/conocimiento/fragmentos.jsonl`, una línea por fragmento:
{"url", "titulo", "texto"}. Se versiona: es texto de sitios públicos del
Estado, con su fuente al lado, y sin él el contenedor no sabe nada.

Recorre sólo museodelaeducacion.gob.cl (Servicio Nacional del Patrimonio
Cultural) y las PDF de sus publicaciones que se piden aquí. La Obra reunida
queda fuera por defecto: son miles de páginas de poesía y prosa que ahogan los
datos del museo en la búsqueda, y un modelo de 1.000 millones de parámetros
tiende a citar versos cuando no sabe qué contestar.
"""
from __future__ import annotations

import argparse
import html
import io
import json
import re
import time
import urllib.parse
import urllib.request
from collections import deque
from html.parser import HTMLParser
from pathlib import Path

SITIO = "https://www.museodelaeducacion.gob.cl"
SALIDA = Path(__file__).resolve().parents[1] / "assets" / "conocimiento" / "fragmentos.jsonl"
# Lo que no es contenido: juegos, recursos, buscadores, listados, bibliografías
# (cientos de años que no son datos) y noticias (talleres y convocatorias con
# fecha, que ella diría como vigentes años después).
FUERA = re.compile(r"^/(sites|themes|core|modules|user|juegos|search|noticias|node/)|^//"
                   r"|bibliografia|listado|%20|\.(jpg|png|gif|zip|mp4|mp3|pdf)$")
PDF_MAESTRA = f"{SITIO}/sites/www.museodelaeducacion.gob.cl/files/images/articles-25896_archivo_01.pdf"
PDF_OBRA = [f"{SITIO}/sites/www.museodelaeducacion.gob.cl/files/2026-02/Tomo{i}.pdf" for i in range(1, 9)]
# MiniLM (el modelo de embeddings) se entrenó con 128 tokens: ~500 caracteres
# en español. Más largo, lo que pase de ahí no cuenta para la búsqueda.
LARGO = 500
CORTO = 80
AGENTE = "gabriela-mistral-museo/0.1 (prototipo; exdevutem)"


class _Texto(HTMLParser):
    """El texto de <main>, por bloques, sin menús ni scripts."""

    BLOQUES = {"p", "h1", "h2", "h3", "h4", "li", "blockquote", "td"}

    def __init__(self):
        super().__init__()
        self.dentro = 0      # profundidad dentro de <main>
        self.salto = 0       # dentro de script/style/nav/form
        self.bloques, self.actual, self.titulo, self.en_titulo = [], [], "", False
        self.enlaces = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "a" and a.get("href"):
            self.enlaces.append(a["href"])
        if tag == "title":
            self.en_titulo = True
        if tag == "main":
            self.dentro += 1
        if tag in ("script", "style", "nav", "form", "noscript"):
            self.salto += 1
        if tag in self.BLOQUES or tag == "br":
            self._cortar()

    def handle_endtag(self, tag):
        if tag == "title":
            self.en_titulo = False
        if tag == "main":
            self.dentro -= 1
        if tag in ("script", "style", "nav", "form", "noscript"):
            self.salto -= 1
        if tag in self.BLOQUES:
            self._cortar()

    def handle_data(self, data):
        if self.en_titulo:
            self.titulo += data
        if self.dentro > 0 and self.salto == 0:
            self.actual.append(data)

    def _cortar(self):
        if t := " ".join("".join(self.actual).split()):
            self.bloques.append(t)
        self.actual = []


def _bajar(url: str) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": AGENTE}),
                                timeout=60) as r:
        return r.read()


def fragmentar(parrafos: list[str], largo: int = LARGO) -> list[str]:
    """Junta párrafos hasta ~`largo` caracteres sin partir ninguno; un párrafo
    más largo que eso se parte en frases."""
    piezas = []
    for p in parrafos:
        piezas += [p] if len(p) <= largo else re.split(r"(?<=[.!?])\s+", p)
    out, actual = [], ""
    for p in piezas:
        if actual and len(actual) + 1 + len(p) > largo:
            out.append(actual)
            actual = ""
        actual = f"{actual} {p}".strip()
    if actual:
        # Una cola corta va con el fragmento anterior: sola no sirve para
        # buscar, y tirarla perdería el final del texto.
        if out and len(actual) < CORTO:
            out[-1] += " " + actual
        else:
            out.append(actual)
    # Un pie de foto o una cabecera de página suelta no sirve de nada.
    return [f for f in out if len(f) >= CORTO]


def sitio(limite: int) -> list[dict]:
    """Recorre el sitio a lo ancho desde la portada, hasta `limite` páginas."""
    vistos, cola, out = {"/"}, deque(["/"]), []
    while cola and len(vistos) <= limite:
        ruta = cola.popleft()
        try:
            crudo = _bajar(SITIO + ruta).decode("utf-8", "replace")
        except Exception as e:  # una página caída no detiene el resto
            print(f"  ✗ {ruta}: {e}")
            continue
        p = _Texto()
        p.feed(crudo)
        titulo = html.unescape(p.titulo.split("|")[0]).strip()
        # Menos de cinco palabras es un menú, una solapa o un botón.
        for f in fragmentar([b for b in p.bloques if len(b.split()) >= 5]):
            out.append({"url": SITIO + ruta, "titulo": titulo, "texto": f})
        for h in p.enlaces:
            u = urllib.parse.urlparse(urllib.parse.urljoin(SITIO + ruta, h))
            if u.netloc.endswith("museodelaeducacion.gob.cl") and not u.query \
                    and not FUERA.search(u.path) and u.path not in vistos:
                vistos.add(u.path)
                cola.append(u.path)
        print(f"  {len(p.bloques):3d} bloques  {ruta}")
        time.sleep(0.3)  # es el sitio de un museo, no un CDN
    return out


def pdf(url: str, titulo: str) -> list[dict]:
    from pypdf import PdfReader

    print(f"  pdf  {url}")
    lector = PdfReader(io.BytesIO(_bajar(url)))
    out = []
    for n, pagina in enumerate(lector.pages, 1):
        texto = pagina.extract_text() or ""
        # Las PDF parten líneas a mitad de frase y cortan palabras con guion.
        texto = re.sub(r"-\n(\w)", r"\1", texto)
        parrafos = [" ".join(b.split()) for b in re.split(r"\n\s*\n|(?<=[.!?:])\n", texto)]
        for f in fragmentar([b for b in parrafos if b]):
            out.append({"url": f"{url}#page={n}", "titulo": titulo, "texto": f})
    return out


if __name__ == "__main__":
    a = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    a.add_argument("--paginas", type=int, default=250, help="tope de páginas del sitio")
    a.add_argument("--obra-reunida", action="store_true", help="suma los 8 tomos de la Obra reunida")
    args = a.parse_args()

    fragmentos = sitio(args.paginas)
    fragmentos += pdf(PDF_MAESTRA, "Lucila Gabriela: La voz de la Maestra (MEGM, 2008)")
    if args.obra_reunida:
        for i, u in enumerate(PDF_OBRA, 1):
            fragmentos += pdf(u, f"Gabriela Mistral, Obra reunida, tomo {i} (MEGM)")
    # Las páginas comparten bloques (avisos, pies): uno de cada uno basta.
    unicos = list({f["texto"]: f for f in fragmentos}.values())
    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    SALIDA.write_text("".join(json.dumps(f, ensure_ascii=False) + "\n" for f in unicos), encoding="utf-8")
    print(f"{len(unicos)} fragmentos -> {SALIDA}")
