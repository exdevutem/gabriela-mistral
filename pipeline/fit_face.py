"""Ajusta la forma de FLAME a los landmarks de una fotografía.

FLAME es lineal en la forma:  V(β, ψ) = T + Σ βᵢ·Sᵢ + Σ ψⱼ·Eⱼ
Con la expresión en reposo (ψ = 0) y una foto frontal, lo único no lineal es la
pose de cámara. Eso lo resuelve `scipy.optimize.least_squares` en segundos, sin
necesidad de PyTorch, autograd ni pytorch3d.

Con varias fotos de la misma época la forma β se comparte y cada foto aporta su
propia cámara y su propia expresión ψ. Una sola vista frontal no distingue una
nariz larga de una cabeza inclinada; tres vistas sí, y la expresión por foto
evita que un ceño o una boca apretada se cuelen en la forma.

Uso:
    python pipeline/fit_face.py assets/fotos/mistral-1946-frontal.jpg [otras...] --preview

La primera foto es la referencia: de ella salen la silueta del cráneo y la
textura, así que debe ser frontal.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).parent))
from landmarks import a_dlib68, detectar

FLAME_NPZ = Path(__file__).parents[1] / "assets" / "flame" / "flame.npz"
BETAS = 80            # coeficientes de forma: más que esto sólo capta ruido de la foto
# Tira de β hacia la cara media. El valor sale de medir, no de mirar: con reg=2 el
# error de reproyección baja a 5,3 px pero el cráneo se aleja un 20% de la forma
# media; con reg=10 el error sube sólo a 7,2 px y el desvío cae al 5%, una
# variación anatómica plausible. Los landmarks no observan el cráneo, así que
# deformarlo le sale gratis al optimizador: esa ganancia es sobreajuste, no
# parecido. Con reg=30 se recupera la cara media y se pierde toda personalización.
# Ese valor se calibró en píxeles de una foto con 170 px entre ojos; los
# residuales ahora van en milímetros (1 px = 0,37 mm allí), así que 10 → 3,7. Con
# más fotos el término de datos crece y la cara media pesa proporcionalmente menos.
REGULARIZACION = 3.7
EXPRESIONES = 10      # ψ por foto: absorbe ceño, labios apretados, sonrisa
REG_EXPRESION = 4.0
# El contorno de mandíbula es el menos fiable de los 68 —mediapipe lo adivina y
# FLAME lo ata a vértices fijos—, pero sin él nada fija el ancho de la cara y el
# ajuste devuelve el óvalo estrecho de la cara media. Entra, con menos peso.
# Se probó el contorno dinámico de FLAME (un anillo por giro de cabeza, como en
# DECA) y empeora: el error de mandíbula sube de 4,7 a 5,7 mm en la frontal y de
# 9 a 16 mm en la de Truman. Los puntos de mediapipe son vértices de su malla
# densa, anatómicos, no de silueta: calzan con el anillo fijo.
PESO_MANDIBULA = 0.5
# Fotos de perfil (ver perfil.py): la silueta manda; las anclas, puestas a ojo,
# sólo sostienen la cámara para que la silueta no se deslice a lo largo de sí misma.
PESO_PERFIL = 1.0
PESO_ANCLAS = 0.5
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


def landmarks_3d(sub: dict, beta: np.ndarray, psi: np.ndarray | None = None) -> np.ndarray:
    """Los 68 puntos, interpolados baricéntricamente sobre sus triángulos."""
    v = sub["v_template"] + sub["shapedirs"][:, :, :len(beta)] @ beta
    if psi is not None:
        v = v + sub["shapedirs"][:, :, 300:300 + len(psi)] @ psi
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


def _interocular(p68: np.ndarray) -> float:
    return float(np.linalg.norm(p68[42:48].mean(0) - p68[36:42].mean(0)))


def _camara_inicial(sub: dict, objetivo: np.ndarray, pesos: np.ndarray) -> np.ndarray:
    """Cámara de arranque probando varios giros con la cara media.

    Partiendo siempre de frente, un perfil o un tres cuartos caen en el mínimo
    local del espejo: el optimizador gira la cabeza hacia el lado equivocado y
    deforma la cara para compensar.
    """
    p3 = landmarks_3d(sub, np.zeros(1))
    escala0 = np.ptp(objetivo[:, 1]) / 0.25
    mejor = None
    for giro in np.deg2rad([-60, -30, 0, 30, 60]):
        x0 = np.array([escala0, 0, giro, 0, *objetivo.mean(axis=0)])
        r = least_squares(lambda c: ((proyectar(p3, c) - objetivo) * pesos[:, None]).ravel(),
                          x0, method="lm")
        if mejor is None or r.cost < mejor.cost:
            mejor = r
    return mejor.x


def _camara_perfil(sub: dict, anclas: dict, lado: int) -> np.ndarray:
    """Cámara de arranque de un perfil, girando 90° hacia donde mira, con las anclas."""
    idx = sorted(anclas)
    obj = np.array([anclas[i] for i in idx])
    p3 = landmarks_3d(sub, np.zeros(1))[idx]
    escala0 = np.ptp(obj[:, 1]) / np.ptp(p3[:, 1])
    x0 = np.array([escala0, 0, lado * np.pi / 2, 0, *obj.mean(axis=0)])
    return least_squares(lambda c: (proyectar(p3, c) - obj).ravel(), x0, method="lm").x


def ajustar(m: dict, objetivos: list[np.ndarray], silueta: tuple | None = None,
            perfiles: list[dict] = ()):
    """Optimiza forma, cámaras y expresiones para que los landmarks calcen.

    `objetivos` son los 68 puntos 2D de cada foto. Los residuales se pasan a
    milímetros con la distancia entre ojos de cada foto, para que una foto de
    alta resolución no mande sobre las demás sólo por tener más píxeles.

    `silueta`, si se pasa, es (centro, ángulos, radios) del contorno de la cabeza
    en la PRIMERA foto, y añade la restricción de que el cráneo quepa dentro.

    `perfiles` son fotos de perfil ya medidas por `perfil.cargar_perfiles`: cada
    una añade su cámara y su expresión, y compara su silueta con la de la malla.
    En un perfil la distancia entre ojos no existe, así que sus residuales se
    pasan a milímetros con la escala de su cámara de arranque.
    """
    from perfil import FRANJA_MM, exceso_perfil, puntos_perfil, submalla_perfil
    n, k = BETAS, EXPRESIONES
    nf = len(objetivos) + len(perfiles)
    sub = submalla_landmarks(m)
    craneo = submalla_craneo(m) if silueta else None
    pesos = np.ones(68)
    pesos[:17] = PESO_MANDIBULA
    mm = [INTEROCULAR_MM / _interocular(o) for o in objetivos]

    def partir(x):
        beta = x[:n]
        resto = x[n:].reshape(nf, 6 + k)
        return beta, resto[:, :6], resto[:, 6:]

    def residuales(x: np.ndarray, peso_perfil: float = PESO_PERFIL) -> np.ndarray:
        beta, camaras, psis = partir(x)
        partes = [REGULARIZACION * beta]
        for o, c, psi, esc in zip(objetivos, camaras, psis, mm):  # zip corta en los frontales
            p2 = proyectar(landmarks_3d(sub, beta, psi), c)
            partes += [((p2 - o) * pesos[:, None] * esc).ravel(), REG_EXPRESION * psi]
        for pf, c, psi, esc in zip(perfiles, camaras[len(objetivos):],
                                   psis[len(objetivos):], mm_perfil):
            idx = sorted(pf["anclas"])
            p2 = proyectar(landmarks_3d(sub, beta, psi)[idx], c)
            obj = np.array([pf["anclas"][i] for i in idx])
            silueta_px = exceso_perfil(proyectar(puntos_perfil(sub_perfil, beta, psi), c),
                                       pf["filas"], pf["xs"], pf["lado"], FRANJA_MM / esc)
            partes += [PESO_ANCLAS * esc * (p2 - obj).ravel(),
                       peso_perfil * esc * silueta_px, REG_EXPRESION * psi]
        if silueta:
            centro, angulos, radios = silueta
            pc = proyectar(puntos_craneo(craneo, beta), camaras[0])
            partes.append(PESO_SILUETA * mm[0] * exceso_silueta(pc, centro, angulos, radios))
        return np.concatenate(partes)

    sub_perfil = submalla_perfil(m)
    inicial = [np.zeros(n)]
    for o in objetivos:
        inicial += [_camara_inicial(sub, o, pesos), np.zeros(k)]
    mm_perfil = []
    for pf in perfiles:
        c0 = _camara_perfil(sub, pf["anclas"], pf["lado"])
        mm_perfil.append(1000.0 / c0[0])       # la cámara proyecta metros a píxeles
        inicial += [c0, np.zeros(k)]
    # Dos etapas. La silueta de perfil compara extremos por franja: es escalonada
    # y, partiendo lejos, arrastra la cámara a un mínimo donde la malla crece y se
    # desliza a lo largo del borde. Primero se colocan las cámaras sólo con
    # landmarks y anclas, y la silueta entra después, ya cerca.
    x = np.concatenate(inicial)
    if perfiles:
        x = least_squares(residuales, x, method="lm", max_nfev=6000,
                          kwargs={"peso_perfil": 0.0}).x
    r = least_squares(residuales, x, method="lm", max_nfev=6000)
    beta, camaras, psis = partir(r.x)
    return beta, camaras, psis, r


def errores_por_foto(m: dict, objetivos, beta, camaras, psis) -> list[float]:
    """Error medio de reproyección de cada foto, en milímetros."""
    sub = submalla_landmarks(m)
    salida = []
    for o, c, psi in zip(objetivos, camaras, psis):
        e = np.linalg.norm(proyectar(landmarks_3d(sub, beta, psi), c) - o, axis=1)
        salida.append(float(e.mean() * INTEROCULAR_MM / _interocular(o)))
    return salida


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("fotos", nargs="+", help="la primera, frontal, es la referencia")
    p.add_argument("--preview", action="store_true", help="guarda comparaciones en /tmp")
    p.add_argument("--sin-silueta", action="store_true",
                   help="ajusta sólo con landmarks, sin restringir el cráneo")
    p.add_argument("--perfiles", type=Path,
                   help="perfiles.json con fotos de perfil anotadas (ver perfil.py)")
    a = p.parse_args()

    m = cargar_flame()
    fotos, objetivos = [], []
    for foto in map(Path, a.fotos):
        try:
            # la referencia con la confianza por defecto: ahí un falso positivo
            # arruinaría la textura; las demás, más permisivas
            conf = 0.5 if not fotos else 0.2
            objetivos.append(a_dlib68(detectar(foto, confianza=conf)[0])[:, :2])
            fotos.append(foto)
        except ValueError as e:
            print(f"  se omite: {e}")
    if not fotos or fotos[0] != Path(a.fotos[0]):
        raise SystemExit("La foto de referencia tiene que tener cara detectable.")

    silueta = None
    if not a.sin_silueta:
        from hair_volume import ANGULOS, _reparar, contorno_pelo
        p68 = objetivos[0]
        oi, od = p68[36:42].mean(0), p68[42:48].mean(0)
        interocular = np.linalg.norm(od - oi)
        centro = (oi + od) / 2 - np.array([0.0, 0.55 * interocular])
        margen = MARGEN_PELO_MM * interocular / INTEROCULAR_MM
        silueta = (centro, ANGULOS,
                   _reparar(*contorno_pelo(fotos[0], centro, interocular)) - margen)

    perfiles = []
    if a.perfiles:
        from perfil import cargar_perfiles
        perfiles = cargar_perfiles(a.perfiles)

    beta, camaras, psis, r = ajustar(m, objetivos, silueta=silueta, perfiles=perfiles)
    print(f"convergió={r.success}  |β|={np.linalg.norm(beta):.2f}")
    for foto, c, e in zip(fotos, camaras, errores_por_foto(m, objetivos, beta, camaras, psis)):
        giro = np.rad2deg(Rotation.from_rotvec(c[1:4]).as_euler("yxz"))
        print(f"  {e:4.1f} mm  giro {giro[0]:+5.1f}°  {foto.name}")
    for pf, c, psi in zip(perfiles, camaras[len(fotos):], psis[len(fotos):]):
        from perfil import FRANJA_MM, exceso_perfil, puntos_perfil, submalla_perfil
        mm_px = 1000.0 / c[0]
        e = np.abs(exceso_perfil(proyectar(puntos_perfil(submalla_perfil(m), beta, psi), c),
                                 pf["filas"], pf["xs"], pf["lado"], FRANJA_MM / mm_px))
        giro = np.rad2deg(Rotation.from_rotvec(c[1:4]).as_euler("yxz"))
        print(f"  {e[e > 0].mean() * mm_px:4.1f} mm  giro {giro[0]:+5.1f}°  {pf['foto'].name}"
              f"  (silueta de perfil)")
    if silueta:
        pc = proyectar(puntos_craneo(submalla_craneo(m), beta), camaras[0])
        exc = exceso_silueta(pc, *silueta)
        print(f"cráneo fuera de la silueta: máximo {exc.max():.1f} px")

    # `camara` y `psi` son los de la referencia: con ellos la textura proyecta la
    # foto sobre la cara tal como estaba en la foto, ceño incluido.
    salida = Path(__file__).parents[1] / "assets" / "flame" / "ajuste.npz"
    np.savez(salida, beta=beta, camara=camaras[0], psi=psis[0], camaras=camaras,
             psis=psis, fotos=np.array([f.name for f in fotos] +
                                       [pf["foto"].name for pf in perfiles]))
    print(f"-> {salida}")

    if a.preview:
        from PIL import Image, ImageDraw
        sub = submalla_landmarks(m)
        for i, (foto, o, c, psi) in enumerate(zip(fotos, objetivos, camaras, psis)):
            img = Image.open(foto).convert("RGB")
            d = ImageDraw.Draw(img)
            radio = max(1, _interocular(o) / 80)
            for color, puntos in (("#22ff22", o),        # foto: verde
                                  ("#ff22ff", proyectar(landmarks_3d(sub, beta, psi), c))):
                for (x, y) in puntos:                    # modelo ajustado: magenta
                    d.ellipse([x - radio, y - radio, x + radio, y + radio], fill=color)
            io = _interocular(o)
            (x0, y0), (x1, y1) = o.min(0) - io, o.max(0) + io
            ruta = f"/tmp/ajuste{'' if i == 0 else f'-{i}'}.png"
            img.crop((int(x0), int(y0), int(x1), int(y1))).save(ruta)
            print(f"verde = foto, magenta = modelo -> {ruta}")
        from perfil import puntos_perfil, submalla_perfil
        for i, (pf, c, psi) in enumerate(zip(perfiles, camaras[len(fotos):],
                                             psis[len(fotos):])):
            img = Image.open(pf["foto"]).convert("RGB")
            d = ImageDraw.Draw(img)
            for x, y in zip(pf["xs"], pf["filas"]):               # borde medido: verde
                d.point((x, y), fill="#22ff22")
            for x, y in proyectar(puntos_perfil(submalla_perfil(m), beta, psi), c):
                d.point((x, y), fill="#ff22ff")                   # malla: magenta
            ruta = f"/tmp/ajuste-perfil-{i}.png"
            img.save(ruta)
            print(f"verde = borde del perfil, magenta = malla -> {ruta}")
