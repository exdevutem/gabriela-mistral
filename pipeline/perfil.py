"""Silueta de un perfil: la única medida directa de profundidad que dan las fotos.

Los landmarks de mediapipe sobre una cara pequeña o girada salen muy pegados a
su cara media: el detector rellena con su prior lo que no ve. Medido con
validación cruzada, β ajustado con dos fotos en tres cuartos predecía la foto
frontal PEOR que la cara media de FLAME.

El borde de un perfil contra un fondo liso, en cambio, no lo inventa nadie:
frente, puente, punta de la nariz, labios y mentón quedan dibujados por el
contraste. Aquí se mide ese borde fila a fila y se compara con la silueta de la
malla vista desde la misma cámara.

Lo que se midió, para no venderlo de más: tras ajustar la cámara, la silueta de
la cara media de FLAME ya queda a unos 2 mm de su perfil real, y las versiones
del ajuste con y sin perfil difieren menos que el ruido de las anclas puestas a
mano. El perfil no empeora nada y fija lo que una vista frontal no ve, pero su
efecto es pequeño.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

# Alto de la franja en que se agrupan filas para comparar malla y foto. Tiene que
# superar la separación entre vértices proyectados (FLAME pone uno cada 3-4 mm),
# o hay franjas sin vértice y la silueta de la malla sale dentada.
FRANJA_MM = 5.0
# Fracción del salto fondo→piel que ya cuenta como cara.
SENSIBILIDAD = 0.15


def contorno_perfil(foto: Path, lado: int, filas: tuple[int, int],
                    nariz: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Borde cara→fondo en cada fila, entre `filas` (frente y mentón, a mano).

    mediapipe no sirve en un perfil puro —con ella los puntos caían en la oreja
    y el cuello—, así que las filas, el lado al que mira (+1 = derecha) y la
    punta de la nariz se anotan a mano en `perfiles.json`. Con la nariz se
    toman los tonos de referencia: fondo delante de ella, piel detrás.

    Cada fila se recorre DESDE el fondo hacia la cara, no al revés: desde dentro,
    el primer borde fuerte es el ojo o la mejilla, no el perfil.
    """
    gris = ndimage.gaussian_filter(
        np.asarray(Image.open(foto).convert("L"), dtype=float), 2.0)
    alto_img, ancho_img = gris.shape
    nx, ny = int(nariz[0]), int(nariz[1])
    alto = filas[1] - filas[0]
    d = max(4, int(0.04 * alto))
    fondo = np.median(gris[ny - 2 * d:ny + 2 * d, sorted([nx + lado * d, nx + lado * 4 * d])[0]:
                           sorted([nx + lado * d, nx + lado * 4 * d])[1]])
    piel = np.median(gris[ny - d:ny + d, sorted([nx - lado * 6 * d, nx - lado * 3 * d])[0]:
                          sorted([nx - lado * 6 * d, nx - lado * 3 * d])[1]])
    # El umbral va muy cerca del fondo: en un impreso, el dorso de la nariz y la
    # frente iluminados quedan casi tan claros como la pared, y un umbral a medio
    # camino se los salta y corta en línea recta del entrecejo a la punta.
    salto = SENSIBILIDAD * (fondo - piel)
    claro = fondo > piel

    ys = np.arange(max(0, filas[0]), min(alto_img, filas[1]))
    inicio = int(np.clip(nx + lado * 0.25 * alto, 0, ancho_img - 1))
    tramo = np.arange(inicio, int(np.clip(nx - lado * 0.6 * alto, 0, ancho_img - 1)), -lado)
    xs = np.full(len(ys), np.nan)
    for i, y in enumerate(ys):
        valores = gris[y, tramo]
        local = np.median(valores[:d])                    # fondo de esta fila
        cara = (valores < local - salto) if claro else (valores > local - salto)
        if cara.any():
            xs[i] = tramo[np.argmax(cara)]
    ok = np.isfinite(xs)
    return ys[ok], xs[ok]


def submalla_perfil(m: dict) -> dict:
    """Vértices que pueden formar la silueta de perfil: la mitad delantera de la cara."""
    v0 = m["v_template"]
    caras = m["f"][m["lmk_faces_idx"].astype(int)]
    lmk = np.einsum("ijk,ij->ik", v0[caras], m["lmk_bary_coords"])
    ojos = np.r_[m["mask_left_eyeball"], m["mask_right_eyeball"]]
    idx = np.flatnonzero((v0[:, 2] > lmk[8, 2] - 0.06) & (np.abs(v0[:, 0]) < 0.03))
    idx = np.setdiff1d(idx, ojos)
    return {"v_template": v0[idx], "shapedirs": m["shapedirs"][idx]}


def puntos_perfil(sub: dict, beta: np.ndarray, psi: np.ndarray) -> np.ndarray:
    sd = sub["shapedirs"]
    return sub["v_template"] + sd[:, :, :len(beta)] @ beta + sd[:, :, 300:300 + len(psi)] @ psi


def cargar_perfiles(ruta: Path) -> list[dict]:
    """Lee `perfiles.json` y mide el borde de cada foto anotada."""
    import json
    salida = []
    for nombre, a in json.loads(ruta.read_text()).items():
        if nombre.startswith("_"):
            continue
        foto = ruta.parent / nombre
        lado = 1 if a["mira"] == "derecha" else -1
        anclas = {int(k): np.array(v, dtype=float) for k, v in a["anclas"].items()}
        filas, xs = contorno_perfil(foto, lado, tuple(a["filas"]), anclas[30])
        for y0, y1 in a.get("excluir", []):
            fuera = (filas >= y0) & (filas <= y1)
            filas, xs = filas[~fuera], xs[~fuera]
        salida.append({"foto": foto, "lado": lado, "filas": filas, "xs": xs, "anclas": anclas})
    return salida


def exceso_perfil(p2: np.ndarray, filas: np.ndarray, xs: np.ndarray, lado: int,
                  franja_px: float) -> np.ndarray:
    """Diferencia horizontal, por franja, entre la silueta de la malla y la de la foto.

    En cada franja de filas la silueta es el punto más adelantado en la dirección
    en que mira. Se compara extremo con extremo, así que la malla no necesita
    tener un vértice exactamente sobre cada fila.
    """
    bordes = np.arange(filas.min(), filas.max() + franja_px, franja_px)
    franja_foto = np.digitize(filas, bordes)
    franja_malla = np.digitize(p2[:, 1], bordes)
    salida = np.zeros(len(bordes) - 1)
    for k in range(1, len(bordes)):
        f, mm = xs[franja_foto == k], p2[franja_malla == k, 0]
        if len(f) and len(mm):
            salida[k - 1] = lado * ((mm * lado).max() - (f * lado).max())
    return salida


if __name__ == "__main__":
    # Silueta recta en x=100 mirando a la derecha; la malla sobresale 3 px arriba
    # y se queda 2 px corta abajo. Cada franja debe dar exactamente esa diferencia.
    filas = np.arange(0, 20)
    xs = np.full(20, 100.0)
    malla = np.array([[90, 2], [103, 4], [95, 14], [98, 16]], dtype=float)
    d = exceso_perfil(malla, filas, xs, lado=1, franja_px=10)
    assert np.allclose(d, [3, -2]), d
    # mirando a la izquierda sobresalir es tener x menor; el residuo sigue siendo
    # x(malla) - x(foto), así que el espejo da el mismo valor con signo opuesto
    d = exceso_perfil(malla * [-1, 1], filas, -xs, lado=-1, franja_px=10)
    assert np.allclose(d, [-3, 2]), d
    print("todo verde")
