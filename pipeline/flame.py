"""Evaluación del modelo FLAME en numpy puro.

FLAME se compone de tres capas:

    v = LBS( v_template + forma·β + expresión·ψ + correctivos_de_pose(θ),  θ )

La forma y la expresión son sumas lineales. La pose no: la mandíbula es una
articulación real, y abrir la boca exige aplicar skinning lineal (LBS) sobre el
esqueleto de cinco huesos del modelo. Sin esa parte no hay boca que se abra,
sólo una superficie que se estira.

Articulaciones: 0 global, 1 cuello, 2 mandíbula, 3 y 4 ojos.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

N_FORMA = 300          # columnas de shapedirs dedicadas a la forma; el resto, expresión
MANDIBULA = 2          # índice de la articulación de la mandíbula
NPZ = Path(__file__).parents[1] / "assets" / "flame" / "flame.npz"


def cargar(ruta: Path = NPZ) -> dict:
    if not ruta.exists():
        raise SystemExit(f"Falta {ruta}. Ejecuta antes: python pipeline/convert_flame.py")
    d = np.load(ruta)
    m = {k: d[k] for k in d.files}
    m["padres"] = m["kintree_table"][0].astype(int)
    m["padres"][0] = -1
    return m


def con_forma(m: dict, beta: np.ndarray, psi: np.ndarray | None = None) -> np.ndarray:
    """Malla en reposo con forma y expresión aplicadas, aún sin pose."""
    sd = m["shapedirs"]
    v = m["v_template"] + sd[:, :, :len(beta)] @ beta
    if psi is not None and len(psi):
        v = v + sd[:, :, N_FORMA:N_FORMA + len(psi)] @ psi
    return v


def _cadena(rotaciones: np.ndarray, J: np.ndarray, padres: np.ndarray) -> np.ndarray:
    """Transformaciones absolutas de cada articulación, relativas a la pose de reposo."""
    n = len(padres)
    absolutas = np.zeros((n, 4, 4))
    for i in range(n):
        local = np.eye(4)
        local[:3, :3] = rotaciones[i]
        local[:3, 3] = J[i] - (J[padres[i]] if padres[i] >= 0 else 0)
        absolutas[i] = local if padres[i] < 0 else absolutas[padres[i]] @ local
    # descontar la pose de reposo, para que θ=0 no mueva nada
    for i in range(n):
        resta = absolutas[i] @ np.block([[np.zeros((3, 3)), J[i].reshape(3, 1)],
                                         [np.zeros((1, 4))]])
        absolutas[i] = absolutas[i] - resta
    return absolutas


def evaluar(m: dict, beta: np.ndarray, psi: np.ndarray | None = None,
            pose: np.ndarray | None = None) -> np.ndarray:
    """Vértices finales. `pose` son 5 vectores de rotación (5, 3), uno por articulación."""
    v = con_forma(m, beta, psi)
    pose = np.zeros((5, 3)) if pose is None else np.asarray(pose, dtype=float).reshape(5, 3)
    R = Rotation.from_rotvec(pose).as_matrix()

    # correctivos de pose: corrigen el estirado que el skinning produce por sí solo
    rasgo = (R[1:] - np.eye(3)).reshape(-1)
    v = v + m["posedirs"] @ rasgo

    J = m["J_regressor"] @ con_forma(m, beta, psi)
    T = _cadena(R, J, m["padres"])
    T_v = (m["weights"] @ T.reshape(len(J), 16)).reshape(-1, 4, 4)

    homogeneo = np.concatenate([v, np.ones((len(v), 1))], axis=1)
    return np.einsum("vij,vj->vi", T_v, homogeneo)[:, :3]


def pose_mandibula(angulo: float) -> np.ndarray:
    """Pose con sólo la mandíbula rotada. Ángulo en radianes; positivo = boca abierta."""
    p = np.zeros((5, 3))
    p[MANDIBULA, 0] = angulo
    return p


if __name__ == "__main__":
    m = cargar()
    beta = np.zeros(80)

    cerrada = evaluar(m, beta)
    reposo = con_forma(m, beta)
    assert np.allclose(cerrada, reposo, atol=1e-5), \
        "con pose cero el skinning no debe mover ningún vértice"

    abierta = evaluar(m, beta, pose=pose_mandibula(0.35))
    desplazamiento = abierta - cerrada
    movidos = np.linalg.norm(desplazamiento, axis=1)
    barbilla = int(np.argmax(movidos))
    assert movidos.max() > 0.005, "la mandíbula no se movió al rotar la articulación"
    assert desplazamiento[barbilla, 1] < 0, "el punto más movido debería bajar, no subir"
    quietos = (movidos < 1e-6).sum()

    print(f"vértices: {len(cerrada)}")
    print(f"pose cero == reposo: sí")
    print(f"mandíbula a 0.35 rad: {movidos.max()*1000:.1f} mm de desplazamiento máximo")
    print(f"  vértices inmóviles (cráneo): {quietos} de {len(cerrada)}")
    print(f"  el punto más movido baja {-desplazamiento[barbilla,1]*1000:.1f} mm")
    print("todo verde")
