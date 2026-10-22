"""Textura horneada: la fotografía, llevada a un mapa que envuelve la cabeza.

El color por vértice topa con la resolución de la malla —FLAME pone un vértice
cada 10 px de foto en la zona de la cara— y ahí se acaban las arrugas, los
surcos y los párpados, que es justo lo que separa un rostro de 57 años de uno
de 25.

FLAME no trae coordenadas UV, y la primera versión usaba como UV la proyección
de cada vértice en la foto. Eso sólo vale para lo que la cámara ve: un vértice
de la nuca cae sobre la cara por el otro lado, y al girar la cabeza aparecía un
ojo detrás de la oreja y la cara entera, reflejada, en la nuca. Darle a los
vértices ocultos la UV del visible más cercano tampoco sirvió: los triángulos
entre dos vecinos lejanos en la foto interpolan a través de ella y dejan
parches de camuflaje.

Así que la UV es un cilindro alrededor de la cabeza, ajeno a la foto, y la
textura se hornea texel a texel: cada uno mira su punto de la superficie en la
foto si la cámara lo ve de frente, y si no, hereda por difusión el color de
sus vecinos visibles. Promediar colores, a diferencia de promediar UV, no
cruza la cara: la nuca se llena con el gris del pelo que la rodea.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.spatial.transform import Rotation

Image.MAX_IMAGE_PIXELS = None
# El cilindro mide ~57 cm de perímetro y ~30 cm de alto con el cuello: 2048×1024
# deja unos 3,5 px/mm en ambos ejes, algo más que los ~3 px/mm de la foto.
ANCHO, ALTO = 2048, 1024
# Coseno entre la normal y la dirección a la cámara. Por encima de VISIBLE la foto
# se usa tal cual; por debajo de OCULTO el píxel está tan de canto que es una
# raya estirada, o directamente la pared del fondo, y se descarta. Entre ambos se
# mezcla con el relleno para no dejar costura.
VISIBLE = 0.45
OCULTO = 0.20


def uv_cilindrica(v: np.ndarray, caras: np.ndarray):
    """UV de un cilindro vertical con la costura en la nuca.

    Los triángulos que cruzan la costura tendrían una punta en u≈0 y otra en u≈1,
    e interpolarían a través de todo el mapa. Sus vértices del lado u<0.5 se
    duplican con u+1, y el muestreador repite en horizontal. Devuelve
    (índice al vértice original de cada vértice nuevo, caras nuevas, uv).
    """
    centro = (v.min(axis=0) + v.max(axis=0)) / 2
    u = np.arctan2(v[:, 0] - centro[0], v[:, 2] - centro[2]) / (2 * np.pi) + 0.5
    t = (v[:, 1].max() - v[:, 1]) / np.ptp(v[:, 1])       # 0 arriba, como en glTF
    uv = np.c_[u, t]

    cruza = np.ptp(u[caras], axis=1) > 0.5
    a_duplicar = np.unique(caras[cruza][u[caras[cruza]] < 0.5])
    nuevo = dict(zip(a_duplicar, len(v) + np.arange(len(a_duplicar))))
    caras = caras.copy()
    for k in np.flatnonzero(cruza):
        caras[k] = [nuevo.get(i, i) if u[i] < 0.5 else i for i in caras[k]]
    indice = np.r_[np.arange(len(v)), a_duplicar]
    uv = np.r_[uv, uv[a_duplicar] + [1.0, 0.0]]
    return indice, caras, uv.astype(np.float32)


def _rasterizar(uv: np.ndarray, caras: np.ndarray):
    """Triángulo y coordenadas baricéntricas de cada texel del mapa."""
    tri = np.full((ALTO, ANCHO), -1)
    bary = np.zeros((ALTO, ANCHO, 3))
    px = uv * [ANCHO, ALTO] - 0.5
    for k, (a, b, c) in enumerate(px[caras]):
        x0, y0 = np.floor(np.minimum(np.minimum(a, b), c)).astype(int)
        x1, y1 = np.ceil(np.maximum(np.maximum(a, b), c)).astype(int)
        y0, y1 = max(y0, 0), min(y1, ALTO - 1)
        d = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if y1 < y0 or abs(d) < 1e-12:
            continue
        gx, gy = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
        w0 = ((b[1] - c[1]) * (gx - c[0]) + (c[0] - b[0]) * (gy - c[1])) / d
        w1 = ((c[1] - a[1]) * (gx - c[0]) + (a[0] - c[0]) * (gy - c[1])) / d
        w = np.stack([w0, w1, 1 - w0 - w1], axis=-1)
        dentro = (w >= -1e-6).all(axis=-1)
        ys, xs = gy[dentro], gx[dentro] % ANCHO       # los duplicados de la costura dan la vuelta
        tri[ys, xs] = k
        bary[ys, xs] = w[dentro]
    return tri, bary


def _rellenar(color: np.ndarray, peso: np.ndarray) -> np.ndarray:
    """Difusión push-pull: cada hueco toma el promedio de lo conocido a su alrededor.

    Se reduce a la mitad hasta que no quedan huecos y se vuelve a subir mezclando
    con lo que había. Es el relleno estándar de mapas de textura, y en una
    pasada lineal hace lo que un difusor iterativo tarda miles de pasos en lograr.
    """
    if min(color.shape) <= 2 or peso.min() >= 1.0:
        return color
    w4 = peso[: peso.shape[0] // 2 * 2, : peso.shape[1] // 2 * 2]
    c4 = color[: w4.shape[0], : w4.shape[1]]
    w_baja = w4.reshape(w4.shape[0] // 2, 2, -1, 2).sum(axis=(1, 3))
    c_baja = (c4 * w4).reshape(w4.shape[0] // 2, 2, -1, 2).sum(axis=(1, 3))
    c_baja = np.divide(c_baja, w_baja, out=np.zeros_like(c_baja), where=w_baja > 0)
    gruesa = _rellenar(c_baja, np.minimum(w_baja, 1.0))
    # grid_mode: cada píxel grueso cubre exactamente sus 2×2 finos. Sin él, el
    # zoom alinea esquinas, desplaza cada nivel medio píxel y los dos bordes de
    # la costura salían de distinto tono.
    arriba = ndimage.zoom(gruesa, (color.shape[0] / gruesa.shape[0],
                                   color.shape[1] / gruesa.shape[1]),
                          order=1, grid_mode=True, mode="nearest")
    return peso * color + (1 - peso) * arriba


def _rellenar_envuelto(color: np.ndarray, peso: np.ndarray) -> np.ndarray:
    """Relleno que da la vuelta al cilindro: sin esto la nuca, que está en los dos
    bordes del mapa, se rellena distinta a cada lado y queda una raya al medio."""
    # Tres vueltas completas, no un margen: el hueco de la nuca abarca casi la
    # mitad del perímetro, y con un margen de un cuarto seguía viendo un borde.
    envolver = lambda a: np.concatenate([a, a, a], axis=1)
    return _rellenar(envolver(color), envolver(peso))[:, ANCHO:2 * ANCHO]


def hornear(v: np.ndarray, caras: np.ndarray, uv: np.ndarray, camara: np.ndarray,
            foto: Path, proyectar, pelo: np.ndarray) -> np.ndarray:
    """Mapa en gris (ALTO, ANCHO), 0..1, con la foto donde se ve y relleno donde no.

    `pelo` es, por vértice, cuánto es cuero cabelludo (0 cara, 1 pelo). El relleno
    se hace por separado para cada región: con uno solo, la nuca heredaba el tono
    de la cara, el cuello y la camisa, y salía calva y color piel.
    """
    from export_glb import _normales

    tri, bary = _rasterizar(uv, caras)
    hay = tri >= 0
    t, w = tri[hay], bary[hay]
    punto = np.einsum("nk,nkj->nj", w, v[caras[t]])
    normal = np.einsum("nk,nkj->nj", w, _normales(v, caras)[caras[t]])
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    de_frente = Rotation.from_rotvec(camara[1:4]).apply(normal)[:, 2]

    gris = np.asarray(Image.open(foto).convert("L"), dtype=float) / 255.0
    px = proyectar(punto, camara)
    muestra = ndimage.map_coordinates(gris, [px[:, 1], px[:, 0]], order=1, mode="nearest")

    p = np.clip((de_frente - OCULTO) / (VISIBLE - OCULTO), 0.0, 1.0)
    color, peso, region = (np.zeros((ALTO, ANCHO)) for _ in range(3))
    color[hay], peso[hay] = muestra, p * p * (3 - 2 * p)
    region[hay] = np.einsum("nk,nk->n", w, pelo[caras[t]])
    es_pelo = region > 0.5
    relleno = np.where(es_pelo, _rellenar_envuelto(color, peso * es_pelo),
                       _rellenar_envuelto(color, peso * ~es_pelo))
    return peso * color + (1 - peso) * relleno


def tenir(gris: np.ndarray, tono=(1.00, 0.94, 0.86)) -> Image.Image:
    """Da un tinte cálido a la foto en blanco y negro.

    Una cara en gris puro se lee como piedra, no como piel. El tinte es leve: la
    intención sigue siendo un busto, no un retrato fotográfico coloreado.
    """
    rgb = np.clip(gris[..., None] * np.array(tono), 0, 1)
    return Image.fromarray((rgb * 255).astype(np.uint8))


def preparar(v: np.ndarray, caras: np.ndarray, camara: np.ndarray, foto: Path, proyectar,
             pelo: np.ndarray):
    """Devuelve (imagen, uv, índice de vértices, caras): la malla gana duplicados en la costura."""
    indice, caras2, uv = uv_cilindrica(v, caras)
    mapa = hornear(v[indice], caras2, uv, camara, foto, proyectar, pelo[indice])
    return tenir(mapa), uv, indice, caras2


if __name__ == "__main__":
    # El relleno no debe tocar lo conocido y debe llenar un hueco con el color
    # que lo rodea, no con cero.
    c = np.full((16, 16), 0.7)
    w = np.ones((16, 16))
    w[4:12, 4:12] = 0.0
    c[4:12, 4:12] = 0.0
    r = _rellenar(c, w)
    assert np.allclose(r[w == 1], 0.7) and np.allclose(r, 0.7, atol=1e-6), r
    print("todo verde")
