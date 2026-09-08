"""FLAME .pkl + landmark_embedding.npy -> un único .npz de numpy puro.

Dos formatos incómodos, ninguno de los cuales justifica una dependencia:

- El .pkl envuelve sus arrays en objetos de `chumpy`, una librería de autodiff
  abandonada que ya no importa en Python moderno. Se simula durante la carga:
  el array que interesa vive en el atributo `x` del estado.
- El .npy guarda tensores de PyTorch. Se leen con `torch_pickle`, un lector
  mínimo del formato legacy, en vez de instalar 2,5 GB para un uso único.

Uso:
    python pipeline/convert_flame.py --inspeccionar
    python pipeline/convert_flame.py
"""
from __future__ import annotations

import pickle
import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from torch_pickle import cargar_npy

FLAME_DIR = Path(__file__).parents[1] / "assets" / "flame"
SALIDA = FLAME_DIR / "flame.npz"
EMBEDDING = FLAME_DIR / "landmark_embedding.npy"
MASCARAS = FLAME_DIR / "FLAME_masks.pkl"
DESCARGA = "https://flame.is.tue.mpg.de/  (requiere registro; licencia no comercial)"

# shapedirs concatena los dos espacios: primero forma, después expresión.
N_FORMA = 300


class _Ch:
    """Stub de chumpy.Ch. Sólo interesa el array que envuelve."""

    def __init__(self, *a, **k):
        self.x = np.asarray(a[0]) if a else np.array([])

    def __setstate__(self, estado):
        # chumpy guarda el ndarray en la clave 'x'; el resto es su maquinaria
        # de autodiff, que aquí sobra.
        self.x = np.asarray(estado["x"]) if isinstance(estado, dict) and "x" in estado \
            else np.array([])

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self.x, dtype=dtype)


def _stub_chumpy() -> None:
    if "chumpy" in sys.modules:
        return
    for nombre in ("chumpy", "chumpy.ch"):
        m = types.ModuleType(nombre)
        m.Ch = _Ch
        sys.modules[nombre] = m
    sys.modules["chumpy"].ch = sys.modules["chumpy.ch"]


def cargar_pkl(ruta: Path) -> dict:
    _stub_chumpy()
    with open(ruta, "rb") as f:
        return pickle.load(f, encoding="latin1")


def buscar_pkl() -> Path:
    # El de máscaras también es un .pkl y ordena antes que flame2023.pkl
    candidatos = sorted(p for p in FLAME_DIR.glob("*.pkl") if p != MASCARAS)
    if not candidatos:
        raise SystemExit(
            f"No hay ningún .pkl en {FLAME_DIR}.\n"
            f"Descarga FLAME desde {DESCARGA}\n"
            f"y deja el archivo del modelo (p. ej. flame2023.pkl) en esa carpeta.")
    return candidatos[0]


def _a_denso(valor) -> np.ndarray | None:
    """Normaliza chumpy, matrices dispersas de scipy y arrays a ndarray."""
    if hasattr(valor, "toarray"):          # csc_matrix / csr_matrix
        return valor.toarray()
    try:
        arr = np.asarray(valor)
    except (ValueError, TypeError):
        return None
    return None if arr.dtype == object or arr.ndim == 0 else arr


def describir(d: dict, sangria: str = "  ") -> None:
    for clave, valor in sorted(d.items()):
        arr = _a_denso(valor)
        if arr is None:
            print(f"{sangria}{clave:<24} {type(valor).__name__}")
        else:
            print(f"{sangria}{clave:<24} {str(arr.shape):<20} {arr.dtype}")


def convertir(ruta: Path) -> Path:
    salida = {}
    for clave, valor in cargar_pkl(ruta).items():
        arr = _a_denso(valor)
        if arr is not None:
            salida[clave] = arr.astype(np.float32) if arr.dtype == np.float64 else arr

    if not EMBEDDING.exists():
        raise SystemExit(
            f"Falta {EMBEDDING}.\nEs parte del mismo paquete de FLAME: sin él no se\n"
            f"sabe qué vértices corresponden a los 68 landmarks de la cara.")
    emb = cargar_npy(EMBEDDING)
    salida["lmk_faces_idx"] = np.asarray(emb["full_lmk_faces_idx"]).reshape(-1)
    salida["lmk_bary_coords"] = np.asarray(
        emb["full_lmk_bary_coords"], dtype=np.float32).reshape(-1, 3)

    # Regiones semánticas: qué vértices son cuero cabelludo, cara, cuello, orejas.
    # Sin ellas hay que deducir el cuero cabelludo por altura, y eso produce un
    # borde horizontal —un flequillo recto— en lugar del arco del nacimiento del
    # pelo.
    if MASCARAS.exists():
        for nombre, indices in cargar_pkl(MASCARAS).items():
            salida[f"mask_{nombre}"] = np.asarray(indices, dtype=np.int64)

    faltan = [k for k in ("v_template", "shapedirs", "f") if k not in salida]
    if faltan:
        raise SystemExit(f"El .pkl no trae {faltan}. Revisa con --inspeccionar.")
    if salida["shapedirs"].shape[-1] <= N_FORMA:
        print(f"AVISO: shapedirs sólo tiene {salida['shapedirs'].shape[-1]} columnas; "
              f"se esperaban más de {N_FORMA} (forma + expresión).")

    np.savez_compressed(SALIDA, **salida)
    return SALIDA


if __name__ == "__main__":
    ruta = buscar_pkl()
    if "--inspeccionar" in sys.argv:
        print(f"{ruta}:")
        describir(cargar_pkl(ruta))
        if EMBEDDING.exists():
            print(f"{EMBEDDING}:")
            describir(cargar_npy(EMBEDDING))
    else:
        destino = convertir(ruta)
        d = np.load(destino)
        print(f"{destino} ({destino.stat().st_size / 1e6:.1f} MB)")
        describir({k: d[k] for k in d.files})
