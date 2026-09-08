"""Lee tensores de PyTorch sin instalar PyTorch.

El `landmark_embedding.npy` de FLAME guarda tensores torch. Instalar torch
—2,5 GB— para leer 31 KB una sola vez no se sostiene, y el formato legacy de
`torch.save` es simple: unos pickles de cabecera, el objeto cuyos storages son
IDs persistentes, y al final los datos crudos de cada storage.

Sólo cubre lo que hace falta aquí: tensores densos en un archivo legacy. No
pretende ser un reemplazo de torch.load.
"""
from __future__ import annotations

import io
import pickle
import struct

import numpy as np

# nombre de la clase de storage -> dtype de numpy
DTYPES = {
    "FloatStorage": np.dtype("<f4"), "DoubleStorage": np.dtype("<f8"),
    "HalfStorage": np.dtype("<f2"), "LongStorage": np.dtype("<i8"),
    "IntStorage": np.dtype("<i4"), "ShortStorage": np.dtype("<i2"),
    "CharStorage": np.dtype("i1"), "ByteStorage": np.dtype("u1"),
    "BoolStorage": np.dtype("?"),
}


class _Storage:
    """Marcador de un storage aún sin datos; se rellena al final del archivo."""

    def __init__(self, dtype: np.dtype) -> None:
        self.dtype = dtype
        self.datos: np.ndarray | None = None


def _rebuild_tensor(storage, desplazamiento, tamano, zancada, *resto) -> np.ndarray:
    """Equivalente de torch._utils._rebuild_tensor_v2 sobre arrays de numpy."""
    plano = storage.datos
    if not tamano:                       # tensor escalar
        return plano[desplazamiento]
    # las zancadas de torch van en elementos, las de numpy en bytes
    return np.lib.stride_tricks.as_strided(
        plano[desplazamiento:], shape=tuple(tamano),
        strides=tuple(s * plano.itemsize for s in zancada)).copy()


def cargar(datos: bytes):
    """Lee un archivo legacy de torch.save desde bytes."""
    f = io.BytesIO(datos)
    almacenes: dict[str, _Storage] = {}

    class Lector(pickle.Unpickler):
        def find_class(self, modulo, nombre):
            if modulo.startswith("torch") and nombre in DTYPES:
                return DTYPES[nombre]
            if nombre == "_rebuild_tensor_v2":
                return _rebuild_tensor
            if nombre == "_load_from_bytes":
                return cargar          # storages anidados: misma rutina
            return super().find_class(modulo, nombre)

        def persistent_load(self, pid):
            # ('storage', tipo, clave, ubicación, numel[, vista])
            _, tipo, clave, _, _ = pid[:5]
            return almacenes.setdefault(str(clave), _Storage(np.dtype(tipo)))

    pickle.load(f)          # magic number
    pickle.load(f)          # versión de protocolo
    pickle.load(f)          # info del sistema
    objeto = Lector(f).load()

    for clave in Lector(f).load():      # orden en que vienen los datos crudos
        st = almacenes[str(clave)]
        (numel,) = struct.unpack("<q", f.read(8))
        st.datos = np.frombuffer(f.read(numel * st.dtype.itemsize), dtype=st.dtype)

    # los tensores se construyeron antes de tener datos: rehacer con el objeto ya lleno
    f.seek(0)
    pickle.load(f); pickle.load(f); pickle.load(f)
    return Lector(f).load()


def cargar_npy(ruta) -> dict:
    """Lee un .npy que contiene un dict con tensores de torch."""
    with open(ruta, "rb") as f:
        version = np.lib.format.read_magic(f)
        lector = getattr(np.lib.format, f"read_array_header_{version[0]}_{version[1]}")
        lector(f)
        crudo = f.read()

    almacenes: dict[str, _Storage] = {}

    class Lector(pickle.Unpickler):
        def find_class(self, modulo, nombre):
            if nombre == "_rebuild_tensor_v2":
                return _rebuild_tensor
            if nombre == "_load_from_bytes":
                return cargar
            return super().find_class(modulo, nombre)

    obj = Lector(io.BytesIO(crudo), encoding="latin1").load()
    return obj.item() if isinstance(obj, np.ndarray) and obj.dtype == object else obj


if __name__ == "__main__":
    import sys
    d = cargar_npy(sys.argv[1])
    for k, v in d.items():
        a = np.asarray(v)
        print(f"{k:<26} {a.shape} {a.dtype}")
