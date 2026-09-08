"""Cejas y ojos: los dos rasgos que FLAME no modela y más pesan en una cara.

La malla trae globos oculares completos pero sin color, así que se leen como
huecos blancos, y no tiene vello de ningún tipo: sin cejas, una cara pierde
buena parte de su expresión y de su edad aparente.

Ninguno de los dos necesita geometría nueva. Los ojos se resuelven pintando el
globo que ya existe según hacia dónde mira cada vértice, y las cejas con una
banda de color sobre la piel más un relieve de milímetro y medio.

Todos los colores van en espacio LINEAL, que es como glTF interpreta COLOR_0.
"""
from __future__ import annotations

import numpy as np

PIEL = np.array([0.72, 0.68, 0.62])
PELO = np.array([0.10, 0.095, 0.10])
CEJA = np.array([0.055, 0.048, 0.045])    # más oscura que el pelo: las suyas son densas
ESCLEROTICA = np.array([0.62, 0.60, 0.57])  # hueso, no blanco puro
IRIS = np.array([0.045, 0.032, 0.022])      # castaño muy oscuro
PUPILA = np.array([0.008, 0.008, 0.008])

# Ángulos desde el centro del globo, en coseno. Un iris humano ocupa unos 30°
# y la pupila unos 13°.
COS_IRIS = 0.86
COS_PUPILA = 0.975


def landmarks_3d(m: dict, v: np.ndarray) -> np.ndarray:
    caras = m["f"][m["lmk_faces_idx"].astype(int)]
    return np.einsum("ijk,ij->ik", v[caras], m["lmk_bary_coords"])


def _densificar(puntos: np.ndarray, n: int = 40) -> np.ndarray:
    """Interpola una polilínea para poder medir distancias a ella sin geometría."""
    t = np.linspace(0, 1, len(puntos))
    tt = np.linspace(0, 1, n)
    return np.stack([np.interp(tt, t, puntos[:, k]) for k in range(3)], axis=1)


def peso_cejas(m: dict, v: np.ndarray, ancho_mm: float = 7.0,
               entrecejo_mm: float = 6.0) -> np.ndarray:
    """Banda alrededor de las dos polilíneas de ceja (landmarks 17-26).

    El perfil es un smoothstep, no una gaussiana: la cola de la gaussiana tiñe
    media frente y la ceja acaba leyéndose como una sombra difusa en vez de como
    una ceja.

    El ancho lo impone la malla, no la anatomía. FLAME tiene 11 vértices a menos
    de 3 mm de la polilínea de una ceja y 25 a menos de 6: por debajo de unos
    7 mm la ceja se queda sin vértices con los que dibujarse. Una ceja más fina
    y definida necesitaría textura, no color por vértice.
    """
    lmk = landmarks_3d(m, v)
    curvas = [_densificar(lmk[17:22]), _densificar(lmk[22:27])]
    radio = ancho_mm / 1000.0

    peso = np.zeros(len(v))
    for curva in curvas:
        d = np.linalg.norm(v[:, None, :] - curva[None, :, :], axis=2).min(axis=1)
        t = np.clip(1.0 - d / radio, 0.0, 1.0)
        peso = np.maximum(peso, t * t * (3 - 2 * t))

    # Las cejas no se juntan en el entrecejo, y ahí pasa la costura de simetría
    # de la malla: pintarla o levantarla deja una raya vertical en la frente.
    peso *= np.clip(np.abs(v[:, 0]) / (entrecejo_mm / 1000.0), 0.0, 1.0)

    # los globos oculares quedan justo detrás: pintarlos de ceja los arruinaría
    peso[m["mask_left_eyeball"]] = 0.0
    peso[m["mask_right_eyeball"]] = 0.0
    return peso


def relieve_cejas(v: np.ndarray, peso: np.ndarray, normales: np.ndarray,
                  altura_mm: float = 0.0) -> np.ndarray:
    """Desplazamiento hacia fuera. Desactivado por defecto.

    Levantar la banda abultaba la frente entera como un ceño permanente, y sobre
    la costura central dejaba una arista. El color solo resuelve la ceja sin
    ninguno de los dos artefactos; se conserva el parámetro por si con una malla
    más densa llega a merecer la pena.
    """
    if altura_mm == 0.0:
        return np.zeros_like(v)
    return normales * (peso * altura_mm / 1000.0)[:, None]


def colores(m: dict, v: np.ndarray, peso_pelo: np.ndarray,
            peso_ceja: np.ndarray) -> np.ndarray:
    """Color por vértice: piel, pelo, cejas, esclerótica, iris y pupila."""
    rgb = np.tile(PIEL, (len(v), 1))

    borde = np.clip((peso_pelo - 0.15) / 0.35, 0.0, 1.0)
    rgb += (PELO - PIEL) * (borde * borde * (3 - 2 * borde))[:, None]

    c = np.clip(peso_ceja, 0.0, 1.0)[:, None]
    rgb = rgb * (1 - c) + CEJA * c

    for lado in ("left", "right"):
        idx = m[f"mask_{lado}_eyeball"]
        p = v[idx]
        direccion = p - p.mean(axis=0)
        direccion /= np.linalg.norm(direccion, axis=1, keepdims=True)
        mirada = direccion[:, 2]          # los ojos de FLAME miran a +Z en reposo
        ojo = np.tile(ESCLEROTICA, (len(idx), 1))
        ojo[mirada > COS_IRIS] = IRIS
        ojo[mirada > COS_PUPILA] = PUPILA
        rgb[idx] = ojo

    return np.concatenate([rgb, np.ones((len(v), 1))], axis=1)
