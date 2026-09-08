"""Escritor de GLB con morph targets.

Es la única parte del proyecto con complejidad no trivial deliberada: armar los
accessors a mano cuesta unas cien líneas, pero a cambio `GLTFLoader` de three.js
carga el resultado sin una sola línea de código propio, y el modelo se abre en
Blender para retocarlo a mano.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pygltflib as gl

FLOAT, USHORT, UINT = 5126, 5123, 5125


def _normales(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    n = np.zeros_like(v)
    tri = v[f]
    cara = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    for k in range(3):
        np.add.at(n, f[:, k], cara)
    largo = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.where(largo == 0, 1, largo)


def escribir_glb(
    vertices: np.ndarray,
    caras: np.ndarray,
    morphs: dict[str, np.ndarray],
    salida: Path,
    nombre: str = "gabriela",
) -> Path:
    """`morphs` mapea nombre -> desplazamientos (mismos vértices, deltas absolutos)."""
    v = np.ascontiguousarray(vertices, dtype=np.float32)
    f = np.ascontiguousarray(caras, dtype=np.uint32)
    n = _normales(v, f).astype(np.float32)

    blob = b""
    vistas, accesos = [], []

    def agregar(datos: np.ndarray, tipo: str, comp: int, objetivo: int | None) -> int:
        nonlocal blob
        blob += b"\x00" * (-len(blob) % 4)  # los accessors deben ir alineados a 4
        desplazamiento = len(blob)
        crudo = datos.tobytes()
        blob += crudo
        vistas.append(gl.BufferView(buffer=0, byteOffset=desplazamiento,
                                    byteLength=len(crudo), target=objetivo))
        plano = datos.reshape(len(datos), -1)
        accesos.append(gl.Accessor(
            bufferView=len(vistas) - 1, componentType=comp, count=len(datos), type=tipo,
            min=plano.min(axis=0).tolist(), max=plano.max(axis=0).tolist()))
        return len(accesos) - 1

    a_pos = agregar(v, "VEC3", FLOAT, gl.ARRAY_BUFFER)
    a_nrm = agregar(n, "VEC3", FLOAT, gl.ARRAY_BUFFER)
    a_idx = agregar(f.reshape(-1), "SCALAR", UINT, gl.ELEMENT_ARRAY_BUFFER)

    objetivos, nombres = [], []
    for nom, destino in morphs.items():
        destino = np.ascontiguousarray(destino, dtype=np.float32)
        d_pos = np.ascontiguousarray(destino - v, dtype=np.float32)
        # Las normales también van como morph target. Sin esto la malla se
        # deforma pero el sombreado no acompaña: la boca se abre y no se nota,
        # porque lo que el ojo lee de una superficie mate es la luz, no la silueta.
        d_nrm = np.ascontiguousarray(_normales(destino, f) - n, dtype=np.float32)
        objetivos.append(gl.Attributes(
            POSITION=agregar(d_pos, "VEC3", FLOAT, gl.ARRAY_BUFFER),
            NORMAL=agregar(d_nrm, "VEC3", FLOAT, gl.ARRAY_BUFFER)))
        nombres.append(nom)

    prim = gl.Primitive(attributes=gl.Attributes(POSITION=a_pos, NORMAL=a_nrm),
                        indices=a_idx, mode=4, targets=objetivos)
    malla = gl.Mesh(primitives=[prim], weights=[0.0] * len(objetivos), name=nombre)
    # three.js lee targetNames de extras para armar morphTargetDictionary.
    malla.extras = {"targetNames": nombres}

    modelo = gl.GLTF2(
        scene=0, scenes=[gl.Scene(nodes=[0])], nodes=[gl.Node(mesh=0, name=nombre)],
        meshes=[malla], accessors=accesos, bufferViews=vistas,
        buffers=[gl.Buffer(byteLength=len(blob))],
        asset=gl.Asset(generator="gabriela-mistral"),
        materials=[gl.Material(
            pbrMetallicRoughness=gl.PbrMetallicRoughness(
                baseColorFactor=[0.82, 0.79, 0.74, 1.0], metallicFactor=0.0,
                roughnessFactor=0.85),
            name="piedra")],
    )
    prim.material = 0
    modelo.set_binary_blob(blob)
    salida.parent.mkdir(parents=True, exist_ok=True)
    modelo.save_binary(str(salida))
    return salida


def cargar_visemas(ruta: Path) -> dict:
    import json
    return json.loads(ruta.read_text())["visemas"]


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from flame import cargar, evaluar, pose_mandibula

    raiz = Path(__file__).parents[1]
    m = cargar()
    beta = np.load(raiz / "assets" / "flame" / "ajuste.npz")["beta"]

    base = evaluar(m, beta)
    morphs = {}
    for nombre, cfg in cargar_visemas(raiz / "assets" / "visemes.json").items():
        psi = np.zeros(100)
        for i, valor in cfg.get("psi", {}).items():
            psi[int(i)] = valor
        morphs[nombre] = evaluar(m, beta, psi=psi, pose=pose_mandibula(cfg["mandibula"]))

    destino = escribir_glb(base, m["f"], morphs, raiz / "assets" / "gabriela.glb")
    tam = destino.stat().st_size / 1e6
    print(f"{destino} ({tam:.1f} MB)")
    print(f"  {len(base)} vértices, {len(m['f'])} caras, {len(morphs)} visemas")
    for nombre, v in morphs.items():
        d = np.linalg.norm(v - base, axis=1).max() * 1000
        print(f"  {nombre:<4} desplazamiento máximo {d:5.1f} mm")
