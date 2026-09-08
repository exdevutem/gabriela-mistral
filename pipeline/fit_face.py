"""Ajusta la forma de FLAME a los landmarks de una fotografía.

FLAME es lineal en la forma:  V(β, ψ) = T + Σ βᵢ·Sᵢ + Σ ψⱼ·Eⱼ
Con la expresión en reposo (ψ = 0) y una foto frontal, lo único no lineal es la
pose de cámara. Eso lo resuelve `scipy.optimize.least_squares` en segundos, sin
necesidad de PyTorch, autograd ni pytorch3d.

AÚN NO EJECUTADO contra el modelo real: falta descargar FLAME (ver README).
La estructura del .pkl varía entre versiones, así que revisa primero
`python pipeline/convert_flame.py --inspeccionar`.

Uso:
    python pipeline/fit_face.py assets/fotos/gabriela-referencia.png --preview
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).parent))
from landmarks import SIN_MANDIBULA, a_dlib68, detectar

FLAME_NPZ = Path(__file__).parents[1] / "assets" / "flame" / "flame.npz"
BETAS = 80          # coeficientes de forma: más que esto solo capta ruido de la foto
REGULARIZACION = 2.0  # tira de β hacia 0; sube si la cara sale deformada


def cargar_flame() -> dict:
    if not FLAME_NPZ.exists():
        raise SystemExit(f"Falta {FLAME_NPZ}. Ejecuta antes:\n"
                         f"  python pipeline/convert_flame.py")
    d = np.load(FLAME_NPZ)
    m = {k: d[k] for k in d.files}
    if "lmk_faces_idx" not in m:
        raise SystemExit(
            "El .npz no trae el landmark embedding de FLAME.\n"
            "Descarga también 'landmark_embedding.npy' del paquete FLAME y déjalo\n"
            "en assets/flame/. Sin él no se sabe qué vértices son los 68 puntos.")
    return m


def vertices(m: dict, beta: np.ndarray) -> np.ndarray:
    """Malla en reposo con la forma dada. shapedirs: (V, 3, 300 forma + 100 expresión)."""
    return m["v_template"] + m["shapedirs"][:, :, :len(beta)] @ beta


def landmarks_3d(m: dict, v: np.ndarray) -> np.ndarray:
    """Los 68 puntos, interpolados baricéntricamente sobre sus triángulos."""
    caras = m["f"][m["lmk_faces_idx"].astype(int)]      # (68, 3) índices de vértice
    bary = m["lmk_bary_coords"]                          # (68, 3) pesos
    return np.einsum("ijk,ij->ik", v[caras], bary)


def proyectar(p3: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Cámara ortográfica escalada: basta para un retrato frontal."""
    escala, rot, desp = params[0], params[1:4], params[4:6]
    return escala * (Rotation.from_rotvec(rot).apply(p3))[:, :2] + desp


def ajustar(m: dict, objetivo_2d: np.ndarray, indices=SIN_MANDIBULA):
    """Optimiza forma y cámara para que los landmarks proyectados calcen con la foto."""
    n = BETAS

    def residuales(x: np.ndarray) -> np.ndarray:
        beta, camara = x[:n], x[n:]
        p2 = proyectar(landmarks_3d(m, vertices(m, beta))[indices], camara)
        return np.concatenate([(p2 - objetivo_2d[indices]).ravel(),
                               REGULARIZACION * beta])

    # arranque: cara media, mirando al frente, escalada al tamaño del rostro
    escala0 = np.ptp(objetivo_2d[:, 1]) / 0.25
    inicial = np.concatenate([np.zeros(n), [escala0, 0, 0, 0], objetivo_2d.mean(axis=0)])
    r = least_squares(residuales, inicial, method="lm", max_nfev=4000)
    return r.x[:n], r.x[n:], r


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("foto")
    p.add_argument("--preview", action="store_true", help="guarda comparación en /tmp")
    a = p.parse_args()

    m = cargar_flame()
    puntos, _ = detectar(Path(a.foto))
    objetivo = a_dlib68(puntos)[:, :2]

    beta, camara, r = ajustar(m, objetivo)
    error = np.sqrt((r.fun[:-BETAS] ** 2).reshape(-1, 2).sum(axis=1)).mean()
    print(f"convergió={r.success}  error medio={error:.2f} px  |β|={np.linalg.norm(beta):.2f}")

    salida = Path(__file__).parents[1] / "assets" / "flame" / "ajuste.npz"
    np.savez(salida, beta=beta, camara=camara)
    print(f"-> {salida}")

    if a.preview:
        from PIL import Image, ImageDraw
        img = Image.open(a.foto).convert("RGB")
        d = ImageDraw.Draw(img)
        p2 = proyectar(landmarks_3d(m, vertices(m, beta)), camara)
        for (x, y) in objetivo:              # foto: verde
            d.ellipse([x - 1, y - 1, x + 1, y + 1], fill="#22ff22")
        for (x, y) in p2:                    # modelo ajustado: magenta
            d.ellipse([x - 1, y - 1, x + 1, y + 1], fill="#ff22ff")
        img.resize((img.width * 3, img.height * 3), Image.LANCZOS).save("/tmp/ajuste.png")
        print("verde = foto, magenta = modelo -> /tmp/ajuste.png")
