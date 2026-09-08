"""Medición de la silueta del peinado sobre la fotografía.

INTENTO FALLIDO como generador de cabello, útil como instrumento de medida.
Ejecutarlo produce `vertices_con_pelo.npy`, que `export_glb.py` aplica si
existe; por defecto no está, y la cabeza sale calva a propósito.

La idea era engrosar el cráneo hasta la silueta del peinado. La medición mostró
que la premisa era falsa: **el cráneo ajustado ya sobresale de la silueta real
con pelo**, con una mediana de 22 px (unos 8 mm) y hasta 98 px (35 mm) en los
laterales. No falta volumen de pelo; sobra cráneo, porque FLAME parte de una
cabeza media mayor que la suya y los landmarks faciales no la corrigen.

Permitiendo desplazamiento negativo la malla se ciñe a la silueta, pero un campo
radial derivado de una vista frontal no sabe de rayas, ondas ni recogidos: sin
suavizar deja facetas, y con suavizado suficiente para evitarlas el efecto se
disipa hasta ser invisible.

Lo que sí funciona y se conserva: `contorno_pelo` detecta el borde del peinado
con fiabilidad (102 de 106 ángulos) por barrido radial. El camino con
posibilidades es usar ese contorno como restricción dentro de `fit_face.py`
—que la silueta de la malla quede contenida en la de la cabeza— en vez de como
retoque posterior.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))
from export_glb import _normales
from fit_face import proyectar

# Rango angular explorado: 0 es arriba, ±90 los lados. Más allá de ±105 la
# silueta ya no es pelo sino cuello y hombros.
ANGULOS = np.deg2rad(np.arange(-105, 106, 2))


def contorno_pelo(foto: Path, centro: np.ndarray, escala_px: float,
                  suavizado: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    """Radio del borde exterior del pelo para cada ángulo, en píxeles.

    Busca por barrido radial la transición pelo (oscuro) → fondo (claro). Un
    umbral global no sirve: el fondo de estas fotos no es uniforme y la máscara
    se fuga hacia la ropa y hacia las sombras de la pared.
    """
    gris = ndimage.gaussian_filter(
        np.asarray(Image.open(foto).convert("L"), dtype=float), suavizado)
    alto, ancho = gris.shape
    radios = np.arange(0.6 * escala_px, 3.2 * escala_px, 2.0)

    contorno = np.full(len(ANGULOS), np.nan)
    confianza = np.zeros(len(ANGULOS))
    for i, th in enumerate(ANGULOS):
        direccion = np.array([np.sin(th), -np.cos(th)])   # la y de imagen crece hacia abajo
        pts = centro + np.outer(radios, direccion)
        dentro = ((pts[:, 0] >= 0) & (pts[:, 0] < ancho - 1) &
                  (pts[:, 1] >= 0) & (pts[:, 1] < alto - 1))
        if dentro.sum() < 10:
            continue
        perfil = ndimage.map_coordinates(gris, [pts[dentro, 1], pts[dentro, 0]], order=1)
        grad = np.gradient(perfil)
        # El borde que interesa es el MÁS EXTERNO, no el más marcado: su pelo es
        # canoso y claro, así que el gradiente más fuerte suele caer en un
        # mechón interior y deja el contorno metido dentro del peinado.
        fuertes = np.flatnonzero(grad > 0.55 * grad.max())
        k = int(fuertes[-1]) if len(fuertes) else int(np.argmax(grad))
        contorno[i] = radios[dentro][k]
        confianza[i] = float(grad[k])
    return contorno, confianza


def _reparar(contorno: np.ndarray, confianza: np.ndarray,
             factor: float = 0.5) -> np.ndarray:
    """Descarta ángulos poco fiables y los rellena por interpolación."""
    bueno = np.isfinite(contorno) & (confianza > np.median(confianza) * factor)
    if bueno.sum() < 5:
        raise SystemExit("El contorno del pelo no se detectó en casi ningún ángulo.")
    lleno = np.interp(ANGULOS, ANGULOS[bueno], contorno[bueno])
    return ndimage.gaussian_filter1d(lleno, 2.0, mode="nearest")


def peso_cuero(v: np.ndarray, y_cejas: float, y_coronilla: float,
               inicio: float = 0.15, pleno: float = 0.45) -> np.ndarray:
    """0 en la cara, 1 en el cuero cabelludo, con transición suave.

    La frontera va en fracciones del tramo cejas→coronilla, no de la altura
    total de la malla: esa incluye el cuello, y medir desde ahí desplaza el
    nacimiento del pelo por encima de la cabeza, dejando el peso a cero.
    """
    alto = y_coronilla - y_cejas
    t = np.clip((v[:, 1] - (y_cejas + inicio * alto)) / ((pleno - inicio) * alto), 0.0, 1.0)
    return t * t * (3 - 2 * t)  # smoothstep


def engrosar(v: np.ndarray, caras: np.ndarray, p2: np.ndarray, centro: np.ndarray,
             radio_pelo: np.ndarray, escala: float, peso: np.ndarray,
             percentil: float = 92.0) -> np.ndarray:
    """Desplaza el cuero cabelludo por su normal hasta la silueta del peinado."""
    rel = p2 - centro
    angulo = np.arctan2(rel[:, 0], -rel[:, 1])       # 0 = arriba, como en ANGULOS
    radio = np.linalg.norm(rel, axis=1)

    # Radio de la silueta de la malla en cada ángulo: el percentil alto de los
    # vértices de ese sector, que es más estable que el máximo.
    borde = np.full(len(ANGULOS), np.nan)
    ancho_sector = np.deg2rad(6)
    for i, th in enumerate(ANGULOS):
        cerca = np.abs(np.angle(np.exp(1j * (angulo - th)))) < ancho_sector
        if cerca.sum() >= 3:
            borde[i] = np.percentile(radio[cerca], percentil)
    bueno = np.isfinite(borde)
    borde = np.interp(ANGULOS, ANGULOS[bueno], borde[bueno])

    # Con signo, a propósito. La medición mostró que el cráneo ajustado sobresale
    # de la silueta real del peinado: FLAME parte de una cabeza media que resulta
    # mayor que la suya. Permitir valores negativos deja que la malla se ciña a la
    # silueta observada en lugar de crecer todavía más.
    grosor_px = radio_pelo - borde
    grosor = ndimage.gaussian_filter1d(grosor_px, 2.0, mode="nearest") / escala

    por_vertice = np.interp(angulo, ANGULOS, grosor, left=grosor[0], right=grosor[-1])
    campo = _normales(v, caras) * (por_vertice * peso)[:, None]
    return v + _suavizar_campo(campo, caras)


def _suavizar_campo(campo: np.ndarray, caras: np.ndarray, pasos: int = 40) -> np.ndarray:
    """Promedia el desplazamiento con el de los vértices vecinos.

    Un campo derivado de un contorno angular cambia a saltos entre direcciones
    contiguas, y aplicado sobre la normal esos saltos salen como facetas y
    aristas: un cráneo tallado a hachazos. Suavizar el campo —no la malla—
    conserva el volumen buscado sin introducir relieve falso.
    """
    aristas = np.concatenate([caras[:, [0, 1]], caras[:, [1, 2]], caras[:, [2, 0]]])
    aristas = np.concatenate([aristas, aristas[:, ::-1]])      # ambos sentidos
    grado = np.bincount(aristas[:, 0], minlength=len(campo)).clip(1)[:, None]

    suave = campo.copy()
    for _ in range(pasos):
        suma = np.zeros_like(suave)
        np.add.at(suma, aristas[:, 0], suave[aristas[:, 1]])
        suave = 0.5 * suave + 0.5 * (suma / grado)
    return suave


def aplicar(m: dict, beta: np.ndarray, camara: np.ndarray, foto: Path,
            landmarks68: np.ndarray) -> tuple[np.ndarray, dict]:
    """Devuelve (vértices con pelo, informe de lo medido)."""
    sys.path.insert(0, str(Path(__file__).parent))
    from flame import evaluar

    v = evaluar(m, beta)
    p2 = proyectar(v, camara)

    ojos = np.r_[landmarks68[36:42, :2], landmarks68[42:48, :2]].mean(axis=0)
    interocular = np.linalg.norm(landmarks68[42:48, :2].mean(0) -
                                 landmarks68[36:42, :2].mean(0))
    centro = ojos - np.array([0.0, 0.55 * interocular])   # centro del cráneo, en la foto

    crudo, confianza = contorno_pelo(foto, centro, interocular)
    radio_pelo = _reparar(crudo, confianza)

    # las cejas salen de los landmarks 3D del propio modelo (17-26 en dlib68)
    caras_lmk = m["f"][m["lmk_faces_idx"].astype(int)]
    lmk3d = np.einsum("ijk,ij->ik", v[caras_lmk], m["lmk_bary_coords"])
    y_cejas = float(lmk3d[17:27, 1].mean())
    peso = peso_cuero(v, y_cejas, float(v[:, 1].max()))

    escala = camara[0]
    nuevo = engrosar(v, m["f"], p2, centro, radio_pelo, escala, peso)
    d = np.linalg.norm(nuevo - v, axis=1)
    return nuevo, {
        "angulos_fiables": int((confianza > np.median(confianza) * 0.5).sum()),
        "angulos_totales": len(ANGULOS),
        "vertices_con_pelo": int((peso > 0.5).sum()),
        "crecimiento_maximo_mm": float(d.max() * 1000),
        "crecimiento_medio_mm": float(d[peso > 0.5].mean() * 1000) if (peso > 0.5).any()
                                 else 0.0,
    }


if __name__ == "__main__":
    import argparse

    from flame import cargar
    from landmarks import a_dlib68, detectar

    p = argparse.ArgumentParser()
    p.add_argument("foto")
    a = p.parse_args()

    raiz = Path(__file__).parents[1]
    m = cargar()
    aj = np.load(raiz / "assets" / "flame" / "ajuste.npz")
    p68 = a_dlib68(detectar(Path(a.foto))[0])

    v, informe = aplicar(m, aj["beta"], aj["camara"], Path(a.foto), p68)
    np.save(raiz / "assets" / "flame" / "vertices_con_pelo.npy", v)
    for k, valor in informe.items():
        print(f"  {k:<24} {valor}")
    print(f"-> assets/flame/vertices_con_pelo.npy")
