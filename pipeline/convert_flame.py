"""FLAME .pkl -> .npz de numpy puro.

El paquete oficial de FLAME viene en pickles que dependen de `chumpy`, una
librería de autodiff abandonada que ya no importa en Python moderno (usa
`numpy.bool`, `numpy.object` y demás alias eliminados). En vez de instalarla,
la simulamos durante la carga: solo necesitamos los arrays, no su maquinaria.

Uso:
    python pipeline/convert_flame.py --inspeccionar      # ver qué trae el .pkl
    python pipeline/convert_flame.py                     # convertir a .npz
"""
from __future__ import annotations

import pickle
import sys
import types
from pathlib import Path

import numpy as np

FLAME_DIR = Path(__file__).parents[1] / "assets" / "flame"
SALIDA = FLAME_DIR / "flame.npz"
DESCARGA = "https://flame.is.tue.mpg.de/  (requiere registro; licencia no comercial)"


def _stub_chumpy() -> None:
    """Deja que pickle resuelva chumpy.ch.Ch sin tener chumpy instalado."""
    if "chumpy" in sys.modules:
        return

    class Ch(np.ndarray):
        """Se comporta como el array que envuelve; el resto de chumpy sobra."""
        def __new__(cls, *a, **k):
            return np.asarray(a[0] if a else []).view(cls)

        def __array_finalize__(self, obj):
            pass

        def __reduce__(self):
            return (np.asarray, (np.asarray(self),))

        def __setstate__(self, estado):
            # los pickles de chumpy guardan el array en distintas claves según versión
            if isinstance(estado, dict):
                for clave in ("x", "_data", "data"):
                    if clave in estado:
                        return

    for nombre, mod in (("chumpy", None), ("chumpy.ch", None)):
        m = types.ModuleType(nombre)
        m.Ch = Ch
        sys.modules[nombre] = m
    sys.modules["chumpy"].ch = sys.modules["chumpy.ch"]


def cargar_pkl(ruta: Path) -> dict:
    _stub_chumpy()
    with open(ruta, "rb") as f:
        return pickle.load(f, encoding="latin1")


def buscar_pkl() -> Path:
    candidatos = sorted(FLAME_DIR.glob("*.pkl"))
    if not candidatos:
        raise SystemExit(
            f"No hay ningún .pkl en {FLAME_DIR}.\n"
            f"Descarga FLAME desde {DESCARGA}\n"
            f"y deja el archivo del modelo (p. ej. flame2023.pkl) en esa carpeta.")
    return candidatos[0]


def describir(d: dict, sangria: str = "  ") -> None:
    for clave, valor in sorted(d.items()):
        arr = np.asarray(valor) if not isinstance(valor, (dict, str, int)) else valor
        if isinstance(arr, np.ndarray) and arr.dtype != object:
            print(f"{sangria}{clave:<24} {str(arr.shape):<20} {arr.dtype}")
        else:
            print(f"{sangria}{clave:<24} {type(valor).__name__}")


def convertir(ruta: Path) -> Path:
    crudo = cargar_pkl(ruta)
    salida = {}
    for clave, valor in crudo.items():
        try:
            arr = np.asarray(valor, dtype=np.float64)
        except (ValueError, TypeError):
            continue
        if arr.dtype == object or arr.ndim == 0:
            continue
        salida[clave] = arr.astype(np.float32) if arr.dtype == np.float64 else arr

    faltan = [k for k in ("v_template", "shapedirs", "f") if k not in salida]
    if faltan:
        print(f"AVISO: no se encontraron {faltan}. Claves disponibles:")
        describir(crudo)

    np.savez_compressed(SALIDA, **salida)
    return SALIDA


if __name__ == "__main__":
    ruta = buscar_pkl()
    if "--inspeccionar" in sys.argv:
        print(f"{ruta}:")
        describir(cargar_pkl(ruta))
    else:
        destino = convertir(ruta)
        d = np.load(destino)
        print(f"{destino} ({destino.stat().st_size / 1e6:.1f} MB)")
        describir({k: d[k] for k in d.files})
