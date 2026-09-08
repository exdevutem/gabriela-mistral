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
    colores: np.ndarray | None = None,
    uv: np.ndarray | None = None,
    imagen=None,
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
    a_col = (agregar(np.ascontiguousarray(colores, dtype=np.float32), "VEC4",
                     FLOAT, gl.ARRAY_BUFFER) if colores is not None else None)
    a_uv = (agregar(np.ascontiguousarray(uv, dtype=np.float32), "VEC2",
                    FLOAT, gl.ARRAY_BUFFER) if uv is not None else None)

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

    atributos = gl.Attributes(POSITION=a_pos, NORMAL=a_nrm)
    if a_col is not None:
        atributos.COLOR_0 = a_col
    if a_uv is not None:
        atributos.TEXCOORD_0 = a_uv
    prim = gl.Primitive(attributes=atributos, indices=a_idx, mode=4, targets=objetivos)
    malla = gl.Mesh(primitives=[prim], weights=[0.0] * len(objetivos), name=nombre)
    # three.js lee targetNames de extras para armar morphTargetDictionary.
    malla.extras = {"targetNames": nombres}

    # La imagen va dentro del propio .glb, en el mismo buffer binario: un GLB con
    # textura externa deja de ser un archivo que se puede mover solo.
    imagenes, muestreadores, texturas = [], [], []
    if imagen is not None:
        import io as _io
        png = _io.BytesIO()
        imagen.save(png, format="PNG", optimize=True)
        crudo = png.getvalue()
        blob += b"\x00" * (-len(blob) % 4)
        vistas.append(gl.BufferView(buffer=0, byteOffset=len(blob), byteLength=len(crudo)))
        blob += crudo
        imagenes.append(gl.Image(bufferView=len(vistas) - 1, mimeType="image/png"))
        # CLAMP en los bordes: con el recorte ajustado a la cabeza, lo que hay al
        # borde es piel o pelo, así que repetir es mejor que mostrar el fondo.
        muestreadores.append(gl.Sampler(magFilter=9729, minFilter=9987,
                                        wrapS=33071, wrapT=33071))
        texturas.append(gl.Texture(sampler=0, source=0))

    modelo = gl.GLTF2(
        scene=0, scenes=[gl.Scene(nodes=[0])], nodes=[gl.Node(mesh=0, name=nombre)],
        meshes=[malla], accessors=accesos, bufferViews=vistas,
        buffers=[gl.Buffer(byteLength=len(blob))],
        asset=gl.Asset(generator="gabriela-mistral"),
        images=imagenes, samplers=muestreadores, textures=texturas,
        materials=[gl.Material(
            pbrMetallicRoughness=gl.PbrMetallicRoughness(
                # glTF multiplica factor x COLOR_0 x textura: con cualquiera de
                # los dos últimos presentes, el factor tiene que ser blanco.
                baseColorFactor=([1.0, 1.0, 1.0, 1.0]
                                 if (colores is not None or imagen is not None)
                                 else [0.82, 0.79, 0.74, 1.0]),
                baseColorTexture=(gl.TextureInfo(index=0) if imagen is not None else None),
                metallicFactor=0.0, roughnessFactor=0.85),
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
    aj = np.load(raiz / "assets" / "flame" / "ajuste.npz")
    beta = aj["beta"]
    FOTO = raiz / "assets" / "fotos" / "mistral-1946-frontal.jpg"

    base = evaluar(m, beta)
    # El peinado sólo cambia la geometría de reposo: no se mueve al hablar, así
    # que el mismo desplazamiento se aplica a todos los visemas y los deltas de
    # los morph targets salen idénticos. Basta con sustituir la base.
    con_pelo = raiz / "assets" / "flame" / "vertices_con_pelo.npy"
    desplazamiento = 0.0
    if con_pelo.exists():
        desplazamiento = np.load(con_pelo) - base
        base = base + desplazamiento
        print(f"  peinado aplicado: {np.linalg.norm(desplazamiento, axis=1).max()*1000:.1f} mm máx")

    morphs = {}
    for nombre, cfg in cargar_visemas(raiz / "assets" / "visemes.json").items():
        psi = np.zeros(100)
        for i, valor in cfg.get("psi", {}).items():
            psi[int(i)] = valor
        morphs[nombre] = evaluar(m, beta, psi=psi,
                                 pose=pose_mandibula(cfg["mandibula"])) + desplazamiento

    # La textura sustituye al color por vértice de rasgos.py: la fotografía ya
    # trae pelo, cejas e iris, y además arrugas, surcos y párpados, que ninguna
    # cantidad de color por vértice podía dar con 5023 vértices.
    # rasgos.py se conserva para un acabado escultórico sin fotografía.
    from fit_face import proyectar
    from textura import preparar
    imagen, uv = preparar(base, aj["camara"], FOTO, proyectar)

    destino = escribir_glb(base, m["f"], morphs, raiz / "assets" / "gabriela.glb",
                           uv=uv, imagen=imagen)
    tam = destino.stat().st_size / 1e6
    print(f"{destino} ({tam:.1f} MB)")
    print(f"  {len(base)} vértices, {len(m['f'])} caras, {len(morphs)} visemas")
    for nombre, v in morphs.items():
        d = np.linalg.norm(v - base, axis=1).max() * 1000
        print(f"  {nombre:<4} desplazamiento máximo {d:5.1f} mm")
