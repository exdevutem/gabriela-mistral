"""Textura proyectiva: la fotografía pegada sobre la malla.

El color por vértice topa con la resolución de la malla —FLAME pone un vértice
cada 10 px de foto en la zona de la cara— y ahí se acaban las arrugas, los
surcos y los párpados, que es justo lo que separa un rostro de 57 años de uno
de 25.

No hacen falta coordenadas UV del paquete de FLAME. El ajuste ya calculó la
cámara que lleva cada vértice a su punto de la fotografía: esas mismas
coordenadas, normalizadas, sirven de UV, y la textura es la propia foto.

El precio es que sólo vale para lo que la cámara ve. Los vértices que miran
hacia atrás muestrean la cara por el otro lado, así que de frente se ve bien y
de perfil no.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
MARGEN = 0.18   # aire alrededor de la cabeza, en fracción de su tamaño


def recorte_cabeza(uv_px: np.ndarray, foto: Path) -> tuple[Image.Image, np.ndarray]:
    """Recorta la foto al entorno de la malla proyectada y devuelve las UV en 0..1.

    Ajustar el recorte a la cabeza aprovecha toda la resolución de la textura, y
    hace que el muestreo con CLAMP en los bordes devuelva piel o pelo en vez del
    fondo de la pared.
    """
    img = Image.open(foto).convert("RGB")
    x0, y0 = uv_px.min(axis=0)
    x1, y1 = uv_px.max(axis=0)
    ancho, alto = x1 - x0, y1 - y0
    x0 -= ancho * MARGEN; x1 += ancho * MARGEN
    y0 -= alto * MARGEN;  y1 += alto * MARGEN

    caja = (int(max(0, x0)), int(max(0, y0)),
            int(min(img.width, x1)), int(min(img.height, y1)))
    recorte = img.crop(caja)

    uv = (uv_px - np.array(caja[:2])) / np.array([recorte.width, recorte.height])
    return recorte, uv.astype(np.float32)


def tenir(img: Image.Image, tono=(1.00, 0.94, 0.86)) -> Image.Image:
    """Da un tinte cálido a una foto en blanco y negro.

    Una cara en gris puro se lee como piedra, no como piel. El tinte es leve: la
    intención sigue siendo un busto, no un retrato fotográfico coloreado.
    """
    a = np.asarray(img, dtype=np.float32) / 255.0
    gris = a.mean(axis=2, keepdims=True)
    return Image.fromarray((np.clip(gris * np.array(tono), 0, 1) * 255).astype(np.uint8))


def preparar(v: np.ndarray, camara: np.ndarray, foto: Path,
             proyectar) -> tuple[Image.Image, np.ndarray]:
    uv_px = proyectar(v, camara)
    recorte, uv = recorte_cabeza(uv_px, foto)
    return tenir(recorte), uv
