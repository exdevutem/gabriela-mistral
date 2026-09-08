"""Cabeza de prueba, para tener algo que animar antes de que llegue FLAME.

Un esferoide con una boca modelada por gaussianas. No se parece a nadie, pero
tiene exactamente los mismos morph targets que tendrá la malla real, así que
sirve para dejar cerrado y verificado todo el camino render + sincronía.

BORRAR este archivo entero cuando `fit_face.py` produzca la cabeza definitiva.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from export_glb import escribir_glb

FILAS, COLUMNAS = 64, 64
BOCA = np.array([0.0, -0.30, 0.95])  # centro de la boca sobre la superficie
ESCALA = 2.5  # deformación exagerada: sobre un huevo liso, lo sutil no se ve


def esferoide() -> tuple[np.ndarray, np.ndarray]:
    lat = np.linspace(0, np.pi, FILAS)
    lon = np.linspace(0, 2 * np.pi, COLUMNAS)
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    v = np.stack([np.sin(la) * np.sin(lo), np.cos(la), np.sin(la) * np.cos(lo)], -1)
    v = v.reshape(-1, 3) * np.array([0.86, 1.15, 0.92])  # proporción de cráneo
    v[:, 2] += 0.10 * np.exp(-((v[:, 1] + 0.15) ** 2) / 0.08)  # insinúa el rostro

    idx = np.arange(FILAS * COLUMNAS).reshape(FILAS, COLUMNAS)
    a, b, c, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]
    caras = np.concatenate([np.stack([a, b, c], -1).reshape(-1, 3),
                            np.stack([a, c, d], -1).reshape(-1, 3)])
    return v.astype(np.float32), caras.astype(np.uint32)


def _peso(v: np.ndarray, centro: np.ndarray, radio: float) -> np.ndarray:
    """Influencia gaussiana alrededor de un punto, solo por delante de la cara."""
    d = np.linalg.norm(v - centro, axis=1)
    return np.exp(-(d**2) / (2 * radio**2)) * (v[:, 2] > 0)


def visemas(v: np.ndarray) -> dict[str, np.ndarray]:
    """Cada visema es la malla ya deformada, no el delta (eso lo calcula el exportador)."""
    boca = _peso(v, BOCA, 0.42)[:, None] * ESCALA
    labio_inf = boca * (v[:, 1:2] < BOCA[1])
    ancho = boca * np.sign(v[:, 0:1])

    def d(**ejes) -> np.ndarray:
        salida = v.copy()
        for eje, delta in ejes.items():
            salida[:, "xyz".index(eje)] += delta[:, 0]
        return salida

    return {
        "A":   d(y=-0.16 * labio_inf, z=0.02 * boca),                    # mandíbula abajo
        "E":   d(y=-0.07 * labio_inf, x=0.05 * ancho),                   # entreabierta y estirada
        "I":   d(y=-0.03 * labio_inf, x=0.08 * ancho),                   # estirada
        "O":   d(y=-0.09 * labio_inf, x=-0.07 * ancho, z=0.06 * boca),   # redonda
        "U":   d(y=-0.04 * labio_inf, x=-0.10 * ancho, z=0.09 * boca),   # proyectada
        "MBP": d(y=0.02 * labio_inf, z=-0.02 * boca),                    # labios juntos
        "FV":  d(y=0.04 * labio_inf, z=-0.05 * labio_inf),               # labio bajo adentro
        "C":   d(y=-0.05 * labio_inf),                                   # consonante neutra
        "X":   v.copy(),                                                 # reposo
    }


if __name__ == "__main__":
    v, f = esferoide()
    salida = escribir_glb(v, f, visemas(v),
                          Path(__file__).parents[1] / "assets" / "gabriela.glb")
    print(f"{salida} — {len(v)} vértices, {len(f)} caras, {len(visemas(v))} morph targets")
