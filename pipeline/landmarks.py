"""Detección de landmarks faciales sobre una fotografía.

MediaPipe entrega 478 puntos con su propia numeración; FLAME trae su
`landmark_embedding` en la convención de 68 puntos de dlib. Este módulo hace de
puente entre ambas.

Nota de instalación: mediapipe 1.x revienta en macOS ARM ("DrishtiMetalHelper /
Service is unavailable") porque fuerza Metal. Por eso el proyecto fija 0.10.35.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

MODELO = Path(__file__).parents[1] / "assets" / "models" / "face_landmarker.task"
URL_MODELO = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
              "face_landmarker/float16/1/face_landmarker.task")

# MediaPipe (478) -> dlib (68), que es la convención que usa FLAME.
# Verificar con `python pipeline/landmarks.py foto.png --preview`: los puntos
# deben caer sobre mandíbula, cejas, nariz, ojos y labios, en ese orden.
A_DLIB68 = [
    162, 234, 93, 58, 172, 136, 149, 148, 152, 377, 378, 365, 397, 288, 323, 454, 389,  # mandíbula
    71, 63, 105, 66, 107,          # ceja derecha
    336, 296, 334, 293, 301,       # ceja izquierda
    168, 197, 5, 4,                # puente de la nariz
    75, 97, 2, 326, 305,           # base de la nariz
    33, 160, 158, 133, 153, 144,   # ojo derecho
    362, 385, 387, 263, 373, 380,  # ojo izquierdo
    61, 39, 37, 0, 267, 269, 291, 405, 314, 17, 84, 181,   # labio externo
    78, 82, 13, 312, 308, 317, 14, 87,                     # labio interno
]
assert len(A_DLIB68) == 68, f"la tabla debe tener 68 puntos, tiene {len(A_DLIB68)}"

# El contorno de mandíbula depende del ángulo de la foto: en un ajuste 3D
# arrastra la forma hacia errores de perspectiva, así que se descarta.
SIN_MANDIBULA = list(range(17, 68))


def detectar(foto: Path) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve (478 puntos xyz en coordenadas de imagen, 52 blendshapes)."""
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    if not MODELO.exists():
        raise FileNotFoundError(
            f"Falta {MODELO}.\nDescárgalo con:\n  curl -sL -o {MODELO} {URL_MODELO}")

    opciones = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODELO)),
        output_face_blendshapes=True, num_faces=1)
    with vision.FaceLandmarker.create_from_options(opciones) as detector:
        imagen = mp.Image.create_from_file(str(foto))
        r = detector.detect(imagen)
    if not r.face_landmarks:
        raise ValueError(f"No se detectó ninguna cara en {foto}")

    alto, ancho = imagen.height, imagen.width
    # z viene en la misma escala que x, relativo al centro de la cabeza
    puntos = np.array([[p.x * ancho, p.y * alto, p.z * ancho]
                       for p in r.face_landmarks[0]], dtype=np.float64)
    pesos = np.array([b.score for b in r.face_blendshapes[0]], dtype=np.float64)
    return puntos, pesos


def a_dlib68(puntos: np.ndarray) -> np.ndarray:
    return puntos[A_DLIB68]


if __name__ == "__main__":
    import argparse

    from PIL import Image, ImageDraw

    p = argparse.ArgumentParser(description="Detecta y dibuja los landmarks de una foto")
    p.add_argument("foto")
    p.add_argument("--preview", default="/tmp/landmarks.png")
    a = p.parse_args()

    puntos, _ = detectar(Path(a.foto))
    p68 = a_dlib68(puntos)

    img = Image.open(a.foto).convert("RGB")
    img = img.resize((img.width * 3, img.height * 3), Image.LANCZOS)
    d = ImageDraw.Draw(img)
    grupos = [(0, 17, "#ff4444"), (17, 27, "#ffaa00"), (27, 36, "#44ff44"),
              (36, 48, "#44aaff"), (48, 68, "#ff44ff")]
    for ini, fin, color in grupos:
        for i in range(ini, fin):
            x, y = p68[i, 0] * 3, p68[i, 1] * 3
            d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=color)
            d.text((x + 4, y - 5), str(i), fill=color)
    img.save(a.preview)
    print(f"{len(puntos)} landmarks -> 68 dlib. Vista en {a.preview}")
    print("Colores: mandíbula rojo, cejas naranja, nariz verde, ojos azul, boca magenta")
