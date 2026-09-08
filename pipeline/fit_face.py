"""Ajusta la forma de FLAME a los landmarks de una fotografía.

FLAME es lineal en la forma:  V(β, ψ) = T + Σ βᵢ·Sᵢ + Σ ψⱼ·Eⱼ
Con la expresión en reposo (ψ = 0) y una foto frontal, lo único no lineal es la
pose de cámara. Eso lo resuelve `scipy.optimize.least_squares` en segundos, sin
necesidad de PyTorch, autograd ni pytorch3d.

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
BETAS = 80            # coeficientes de forma: más que esto sólo capta ruido de la foto
# Tira de β hacia la cara media. El valor sale de medir, no de mirar: con reg=2 el
# error de reproyección baja a 5,3 px pero el cráneo se aleja un 20% de la forma
# media; con reg=10 el error sube sólo a 7,2 px y el desvío cae al 5%, una
# variación anatómica plausible. Los landmarks no observan el cráneo, así que
# deformarlo le sale gratis al optimizador: esa ganancia es sobreajuste, no
# parecido. Con reg=30 se recupera la cara media y se pierde toda personalización.
REGULARIZACION = 10.0
# Peso de la restricción de silueta frente a los residuales de landmarks.
PESO_SILUETA = 0.60
MUESTRAS_CRANEO = 250
# Espacio que se reserva bajo la silueta para el peinado. Sin él, el cráneo se
# ajusta al contorno del PELO y se come el hueco que el pelo debía ocupar: una
# vista frontal no distingue "cráneo grande" de "cráneo más cabello".
MARGEN_PELO_MM = 15.0
INTEROCULAR_MM = 63.0   # media adulta, para convertir el margen a píxeles


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
    """Malla completa con la forma dada. shapedirs: (V, 3, 300 forma + 100 expresión)."""
    return m["v_template"] + m["shapedirs"][:, :, :len(beta)] @ beta


def submalla_landmarks(m: dict) -> dict:
    """Precalcula sólo los vértices que tocan los 68 landmarks.

    El optimizador evalúa esto cientos de veces; reconstruir los 5023 vértices
    cada vez para leer 68 puntos es tirar el 96% del cálculo. Bajando a los ~200
    vértices implicados, cada iteración cuesta una fracción.
    """
    caras = m["f"][m["lmk_faces_idx"].astype(int)]        # (68, 3) índices globales
    usados, inverso = np.unique(caras, return_inverse=True)
    return {"v_template": m["v_template"][usados],
            "shapedirs": m["shapedirs"][usados],
            "caras_locales": inverso.reshape(caras.shape),
            "bary": m["lmk_bary_coords"]}


def submalla_craneo(m: dict, n: int = MUESTRAS_CRANEO) -> dict:
    """Muestra de vértices por encima de las cejas, para la restricción de silueta.

    Los landmarks faciales no observan el cráneo, así que el ajuste lo deja del
    tamaño de la cabeza media de FLAME —que resulta mayor que la suya— y la
    malla acaba sobresaliendo de su propio peinado.
    """
    v0 = m["v_template"]
    caras = m["f"][m["lmk_faces_idx"].astype(int)]
    lmk = np.einsum("ijk,ij->ik", v0[caras], m["lmk_bary_coords"])
    candidatos = np.flatnonzero(v0[:, 1] > lmk[17:27, 1].mean())
    idx = candidatos[:: max(1, len(candidatos) // n)]
    return {"v_template": v0[idx], "shapedirs": m["shapedirs"][idx]}


def puntos_craneo(sub: dict, beta: np.ndarray) -> np.ndarray:
    return sub["v_template"] + sub["shapedirs"][:, :, :len(beta)] @ beta


def exceso_silueta(p2: np.ndarray, centro: np.ndarray, angulos: np.ndarray,
                   radios: np.ndarray) -> np.ndarray:
    """Cuánto se sale cada punto del contorno de la cabeza, en píxeles.

    Unilateral a propósito: quedarse corto no penaliza, porque el peinado ocupa
    un espacio que la malla desnuda no tiene por qué llenar. Sólo se castiga
    sobresalir.
    """
    rel = p2 - centro
    angulo = np.arctan2(rel[:, 0], -rel[:, 1])
    radio = np.linalg.norm(rel, axis=1)
    limite = np.interp(angulo, angulos, radios, left=radios[0], right=radios[-1])
    return np.clip(radio - limite, 0, None)


def landmarks_3d(sub: dict, beta: np.ndarray) -> np.ndarray:
    """Los 68 puntos, interpolados baricéntricamente sobre sus triángulos."""
    v = sub["v_template"] + sub["shapedirs"][:, :, :len(beta)] @ beta
    return np.einsum("ijk,ij->ik", v[sub["caras_locales"]], sub["bary"])


def proyectar(p3: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Cámara ortográfica escalada: basta para un retrato frontal.

    La Y se invierte porque en una imagen crece hacia abajo y en FLAME hacia
    arriba. Sin esto el ajuste tendría que descubrir por su cuenta un giro de
    180°, algo que un optimizador local no encuentra partiendo de cero.
    """
    escala, rot, desp = params[0], params[1:4], params[4:6]
    p2 = Rotation.from_rotvec(rot).apply(p3)[:, :2] * [1, -1]
    return escala * p2 + desp


def ajustar(m: dict, objetivo_2d: np.ndarray, indices=SIN_MANDIBULA,
            silueta: tuple | None = None):
    """Optimiza forma y cámara para que los landmarks proyectados calcen con la foto.

    `silueta`, si se pasa, es (centro, ángulos, radios) del contorno de la cabeza
    medido sobre la foto, y añade la restricción de que el cráneo quepa dentro.
    """
    n = BETAS
    sub = submalla_landmarks(m)
    craneo = submalla_craneo(m) if silueta else None

    def residuales(x: np.ndarray) -> np.ndarray:
        beta, camara = x[:n], x[n:]
        p2 = proyectar(landmarks_3d(sub, beta)[indices], camara)
        partes = [(p2 - objetivo_2d[indices]).ravel(), REGULARIZACION * beta]
        if silueta:
            centro, angulos, radios = silueta
            pc = proyectar(puntos_craneo(craneo, beta), camara)
            partes.append(PESO_SILUETA * exceso_silueta(pc, centro, angulos, radios))
        return np.concatenate(partes)

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
    p.add_argument("--sin-silueta", action="store_true",
                   help="ajusta sólo con landmarks, sin restringir el cráneo")
    a = p.parse_args()

    m = cargar_flame()
    puntos, _ = detectar(Path(a.foto))
    p68 = a_dlib68(puntos)
    objetivo = p68[:, :2]

    silueta = None
    if not a.sin_silueta:
        from hair_volume import ANGULOS, _reparar, contorno_pelo
        oi, od = p68[36:42, :2].mean(0), p68[42:48, :2].mean(0)
        interocular = np.linalg.norm(od - oi)
        centro = (oi + od) / 2 - np.array([0.0, 0.55 * interocular])
        margen = MARGEN_PELO_MM * interocular / INTEROCULAR_MM
        silueta = (centro, ANGULOS,
                   _reparar(*contorno_pelo(Path(a.foto), centro, interocular)) - margen)

    beta, camara, r = ajustar(m, objetivo, silueta=silueta)
    n_lmk = len(SIN_MANDIBULA) * 2
    error = np.sqrt((r.fun[:n_lmk] ** 2).reshape(-1, 2).sum(axis=1)).mean()
    print(f"convergió={r.success}  error medio={error:.2f} px  |β|={np.linalg.norm(beta):.2f}")
    if silueta:
        pc = proyectar(puntos_craneo(submalla_craneo(m), beta), camara)
        exc = exceso_silueta(pc, *silueta)
        print(f"cráneo fuera de la silueta: máximo {exc.max():.1f} px")

    salida = Path(__file__).parents[1] / "assets" / "flame" / "ajuste.npz"
    np.savez(salida, beta=beta, camara=camara)
    print(f"-> {salida}")

    if a.preview:
        from PIL import Image, ImageDraw
        img = Image.open(a.foto).convert("RGB")
        d = ImageDraw.Draw(img)
        p2 = proyectar(landmarks_3d(submalla_landmarks(m), beta), camara)
        for (x, y) in objetivo:              # foto: verde
            d.ellipse([x - 1, y - 1, x + 1, y + 1], fill="#22ff22")
        for (x, y) in p2:                    # modelo ajustado: magenta
            d.ellipse([x - 1, y - 1, x + 1, y + 1], fill="#ff22ff")
        img.resize((img.width * 3, img.height * 3), Image.LANCZOS).save("/tmp/ajuste.png")
        print("verde = foto, magenta = modelo -> /tmp/ajuste.png")
