"""Afina NeuTTS nano-spanish con la voz de Gabriela, en el Mac (MPS) y gratis.

La clonación zero-shot a veces se va a otro hablante —la «voz de hombre» que
cazan voice._puntuar y voice._mejor_toma—. Afinando, el modelo deja de tener
que adivinar la voz a partir de 7,6 s de referencia: la aprende.

Sigue la receta de neuphonic (`TRAINING.md` y `examples/finetune.py` en
github.com/neuphonic/neutts): fine-tuning completo del backbone con
transformers, códigos de NeuCodec como objetivo. Dos cambios:

- **El prompt de entrenamiento es el de la inferencia**: la referencia
  (`referencia.wav` y su texto) va delante de cada frase, tal cual la arma
  NeuTTS._apply_chat_template, y la pérdida sólo cuenta los códigos de la
  frase. Así se entrena exactamente lo que después se le pide.
- **La pérdida se calcula sólo donde hay etiqueta**: el vocabulario son 194 256
  tokens y sacar logits de todo el prompt no cabe en lotes útiles.

Datos (ver `assets/voz/README.md`):

- `dataset/` — ~1.500 frases de Common Voice dichas por F5-TTS clonando la
  misma referencia (la maestra del plan de Piper). Las que Whisper no entiende
  (CER > CER_MAX) se quedan fuera.
- `referencia-completa.wav` — 67 s de la locutora real, cortados en frases
  (LOCUCION) y repetidos REPETIR_LOCUCION veces: es la voz de verdad.
- Las `frec-*` del dataset NO se entrenan: son las FRECUENTES, y sirven de
  validación y de evaluación limpia.

    uv run python pipeline/afinar_neutts.py preparar     # ~33 min en un M5, una vez
    uv run python pipeline/afinar_neutts.py entrenar     # ~1 h 20 min; retoma si se corta
    uv run python pipeline/afinar_neutts.py exportar --paso 600
    uv run python pipeline/afinar_neutts.py evaluar base assets/models/neutts-gabriela
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import random
import re
import shutil
import time
import unicodedata
from pathlib import Path

import numpy as np

from gabriela import voice
from gabriela.config import (ASSETS, FRECUENTES, GRABADO_ACEPTABLE, GRABADO_TEMPERATURA,
                             REF_AUDIO, RMS_OBJETIVO, SEED, TEMPERATURA, VOZ)

log = logging.getLogger("afinar")

BASE = "neuphonic/neutts-nano-spanish"
DATOS = VOZ / "afinado" / "datos.json"
CKPT = ASSETS / "models" / "neutts-gabriela-ckpt"
SALIDA = ASSETS / "models" / "neutts-gabriela"

# Una frase de F5 que Whisper transcribe con más de un 25 % de letras mal es
# una toma que se comió o cambió palabras: enseñaría a decir otra cosa.
CER_MAX = 0.25
# 67 s de voz real contra 90 min de F5: repetida, pesa ~10 % del entrenamiento.
REPETIR_LOCUCION = 8

# referencia-completa.wav cortada en frases: (desde, hasta) en segundos y lo
# que dice. Tiempos de Whisper por palabra, transcripción a mano —los números
# en letras, como los escriben FRECUENTES—. Los primeros 8 s quedan fuera
# porque son referencia.wav: el modelo no debe aprender a copiar el prompt.
LOCUCION = [
    (8.00, 12.50, "Aunque tal vez me conozcan mejor como Gabriela Mistral."),
    (12.50, 18.53, "Hoy recorreremos este espacio que resguarda la memoria de la enseñanza en Chile."),
    (18.53, 30.24, "Aquí no solo se guarda historia, también se escuchan las voces de las maestras, "
                   "los estudiantes y los sueños que alguna vez poblaron estas aulas."),
    (30.24, 38.76, "Yo misma estuve en este edificio en mil novecientos diez, cuando rendí mis "
                   "exámenes para validarme como maestra."),
    (38.76, 47.50, "Por eso, regresar hoy, aunque sea en esta forma, es reencontrarme con mis "
                   "primeros pasos en la enseñanza."),
    (47.50, 57.90, "Desde mil ochocientos ochenta y seis, este lugar fue la Escuela Normal de Niñas "
                   "de Santiago, donde se formaban profesoras primarias."),
    (57.90, 63.95, "Hasta mil novecientos setenta y tres, funcionó como espacio de aprendizaje."),
    (63.95, 74.76, "Tras el terremoto de mil novecientos ochenta y cinco quedó cerrado, y sólo en "
                   "dos mil seis volvió a abrirse, esta vez como museo de la educación."),
]


# --- Preparar: texto -> prompt de NeuTTS, audio -> códigos de NeuCodec -------


def _codificar(modelo, x16k: np.ndarray) -> list[int]:
    """Lo mismo que NeuTTS.encode_reference, pero desde un array."""
    import torch

    with torch.no_grad():
        t = torch.from_numpy(x16k).float()[None, None, :]
        return modelo.codec.encode_code(audio_or_path=t).squeeze().tolist()


def _corte(x: np.ndarray, t: float, sr: int) -> int:
    """El punto más callado a ±0,25 s de `t`: Whisper redondea los bordes."""
    paso = sr // 100
    a, b = max(int((t - 0.25) * sr), 0), min(int((t + 0.25) * sr), len(x))
    if b - a < paso:
        return min(int(t * sr), len(x))
    energia = [float((x[i:i + paso] ** 2).mean()) for i in range(a, b - paso, paso)]
    return a + paso * int(np.argmin(energia)) + paso // 2


_UNIDADES = ("cero uno dos tres cuatro cinco seis siete ocho nueve diez once doce trece "
             "catorce quince dieciséis diecisiete dieciocho diecinueve veinte veintiuno "
             "veintidós veintitrés veinticuatro veinticinco veintiséis veintisiete "
             "veintiocho veintinueve").split()
_DECENAS = "treinta cuarenta cincuenta sesenta setenta ochenta noventa".split()
_CENTENAS = ("ciento doscientos trescientos cuatrocientos quinientos seiscientos "
             "setecientos ochocientos novecientos").split()


def en_letras(n: int) -> str:
    """Un entero en palabras, hasta 999 999: Whisper escribe «1957» y las
    FRECUENTES «mil novecientos cincuenta y siete»."""
    if n < 30:
        return _UNIDADES[n]
    if n < 100:
        d, u = divmod(n, 10)
        return _DECENAS[d - 3] + (f" y {_UNIDADES[u]}" if u else "")
    if n == 100:
        return "cien"
    if n < 1000:
        c, r = divmod(n, 100)
        return _CENTENAS[c - 1] + (f" {en_letras(r)}" if r else "")
    if n < 1_000_000:
        m, r = divmod(n, 1000)
        return ("mil" if m == 1 else f"{en_letras(m)} mil") + (f" {en_letras(r)}" if r else "")
    return str(n)


def normalizar(texto: str) -> str:
    """Minúsculas, cifras en letras, sin tildes ni puntuación: para comparar lo
    dicho con lo pedido."""
    texto = re.sub(r"\d{1,3}(?:\.\d{3})+|\d+",
                   lambda m: en_letras(int(m.group().replace(".", ""))), texto)
    t = unicodedata.normalize("NFD", texto.lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    return " ".join(re.sub(r"[^a-z0-9ñ ]", " ", t).split())


def cer(esperado: str, oido: str) -> float:
    """Distancia de edición entre los dos textos normalizados, por letra esperada."""
    a, b = normalizar(esperado), normalizar(oido)
    fila = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        previa, fila[0] = fila[0], i
        for j, cb in enumerate(b, 1):
            previa, fila[j] = fila[j], min(fila[j] + 1, fila[j - 1] + 1, previa + (ca != cb))
    return fila[-1] / max(len(a), 1)


def preparar() -> None:
    import librosa

    from gabriela import entender

    modelo = voice.precargar()
    ref_audio, ref_texto = voice._referencia()
    ref_codigos = modelo.encode_reference(ref_audio)
    datos = json.loads(DATOS.read_text()) if DATOS.exists() else {}
    DATOS.parent.mkdir(parents=True, exist_ok=True)

    def anotar(clave, texto, x16k, cer_=0.0):
        datos[clave] = {
            "texto": texto, "cer": round(cer_, 3),
            "prompt": modelo._apply_chat_template(ref_codigos, ref_texto, texto),
            "codigos": _codificar(modelo, x16k),
        }

    x, sr = librosa.load(VOZ / "referencia-completa.wav", sr=16_000, mono=True)
    for k, (desde, hasta, texto) in enumerate(LOCUCION):
        anotar(f"locucion-{k}", texto, x[_corte(x, desde, sr):_corte(x, hasta, sr)])

    filas = (VOZ / "dataset" / "metadata.csv").read_text(encoding="utf-8").splitlines()
    t0 = time.time()
    for n, fila in enumerate(filas):
        clave, texto = fila.split("|", 1)
        if clave in datos:
            continue
        y, _ = librosa.load(VOZ / "dataset" / "wav" / f"{clave}.wav", sr=16_000, mono=True)
        oido = entender.transcribir((y.clip(-1, 1) * 32767).astype("<i2").tobytes())
        anotar(clave, texto, y, cer(texto, oido))
        if n % 100 == 99:
            DATOS.write_text(json.dumps(datos))
            log.info("%d/%d (%.0f s)", n + 1, len(filas), time.time() - t0)
    DATOS.write_text(json.dumps(datos))
    malas = sum(d["cer"] > CER_MAX for d in datos.values())
    log.info("listo: %d frases, %d con CER > %.2f quedan fuera", len(datos), malas, CER_MAX)


# --- Entrenar ----------------------------------------------------------------


def _ejemplos(tokenizer) -> tuple[list, list]:
    """(entrenamiento, validación) como pares (ids, cuántos son prompt)."""
    datos = json.loads(DATOS.read_text())
    speech = np.array(tokenizer.convert_tokens_to_ids([f"<|speech_{i}|>" for i in range(65_536)]))
    fin = tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_END|>")
    entreno, valida = [], []
    for clave, d in datos.items():
        ids = d["prompt"] + speech[d["codigos"]].tolist() + [fin]
        ejemplo = (ids, len(d["prompt"]))
        if clave.startswith("frec-"):
            valida.append(ejemplo)
        elif clave.startswith("locucion-"):
            entreno += [ejemplo] * REPETIR_LOCUCION
        elif d["cer"] <= CER_MAX:
            entreno.append(ejemplo)
    return entreno, valida


def _lotes(ejemplos: list, lote: int, epoca: int) -> list[list]:
    """Lotes de una época, en orden fijo por semilla —retomar sigue el mismo—.
    Ordenados por largo dentro de ventanas de 16 lotes para no rellenar de más."""
    orden = list(range(len(ejemplos)))
    random.Random(SEED * 1000 + epoca).shuffle(orden)
    lotes = []
    for i in range(0, len(orden), lote * 16):
        ventana = sorted(orden[i:i + lote * 16], key=lambda k: len(ejemplos[k][0]))
        lotes += [ventana[j:j + lote] for j in range(0, len(ventana), lote)]
    random.Random(SEED * 1000 + epoca + 1).shuffle(lotes)
    return [l for l in lotes if len(l) == lote]


def _tensores(ejemplos: list, relleno: int, device: str):
    import torch

    largo = max(len(ids) for ids, _ in ejemplos)
    ids = torch.full((len(ejemplos), largo), relleno)
    etiquetas = torch.full_like(ids, -100)
    mascara = torch.zeros_like(ids)
    for i, (e, prompt) in enumerate(ejemplos):
        ids[i, :len(e)] = torch.tensor(e)
        etiquetas[i, prompt:len(e)] = torch.tensor(e[prompt:])
        mascara[i, :len(e)] = 1
    return ids.to(device), etiquetas.to(device), mascara.to(device)


def _perdida(modelo, ids, etiquetas, mascara):
    """Entropía cruzada sólo donde hay etiqueta: los logits de 194 256 tokens
    sobre el prompt entero son gigas que no aportan nada."""
    import torch.nn.functional as F

    h = modelo.model(input_ids=ids, attention_mask=mascara).last_hidden_state
    objetivo = etiquetas[:, 1:]
    hay = objetivo != -100
    logits = modelo.lm_head(h[:, :-1][hay])
    return F.cross_entropy(logits.float(), objetivo[hay])


def _acompañar(destino: Path) -> None:
    """Tokenizer y licencia del modelo base al lado de unos pesos: así cada
    carpeta se carga sola con NeuTTS, y la licencia de neuphonic exige
    acompañar las obras derivadas."""
    from huggingface_hub import snapshot_download

    base = Path(snapshot_download(BASE))
    for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "LICENCE"):
        shutil.copy(base / f, destino / f)


def entrenar(a) -> None:
    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              get_cosine_schedule_with_warmup)

    CKPT.mkdir(parents=True, exist_ok=True)
    logging.getLogger().addHandler(logging.FileHandler(CKPT / "entrenamiento.log"))
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(SEED)

    tokenizer = AutoTokenizer.from_pretrained(BASE)
    pasos = sorted(int(p.name.split("-")[1]) for p in CKPT.glob("paso-*"))
    estado = CKPT / "estado.pt"
    e = torch.load(estado, weights_only=True) if estado.exists() else None
    retomar = bool(pasos) and e is not None and e["paso"] == pasos[-1]
    origen = CKPT / f"paso-{pasos[-1]}" if retomar else BASE
    modelo = AutoModelForCausalLM.from_pretrained(origen, dtype=torch.float32).to(device)
    modelo.config.use_cache = False
    # Sin esto SDPA guarda la atención de las 24 capas para el backward y un
    # lote de 8 pasa de 24 GB; con esto son ~6 GB y un 30 % más de tiempo.
    modelo.gradient_checkpointing_enable()

    entreno, valida = _ejemplos(tokenizer)
    lotes_por_epoca = len(entreno) // a.lote
    log.info("%d ejemplos de entrenamiento (%d lotes por época), %d de validación",
             len(entreno), lotes_por_epoca, len(valida))

    opt = torch.optim.AdamW(modelo.parameters(), lr=a.lr, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, a.calentamiento, a.pasos)
    paso = 0
    if retomar:
        opt.load_state_dict(e["opt"])
        sched.load_state_dict(e["sched"])
        paso = e["paso"]
        log.info("retomo desde el paso %d", paso)

    def validar() -> float:
        modelo.eval()
        with torch.no_grad():
            total = [float(_perdida(modelo, *_tensores(valida[i:i + a.lote],
                                                       tokenizer.pad_token_id, device)))
                     for i in range(0, len(valida), a.lote)]
        modelo.train()
        return sum(total) / len(total)

    if paso == 0:
        log.info("paso 0 · validación %.4f", validar())
    modelo.train()
    t0 = time.time()
    while paso < a.pasos:
        epoca, i = divmod(paso, lotes_por_epoca)
        lote = [entreno[k] for k in _lotes(entreno, a.lote, epoca)[i]]
        perdida = _perdida(modelo, *_tensores(lote, tokenizer.pad_token_id, device))
        perdida.backward()
        torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        if device == "mps":
            # Cada lote tiene otro largo y la caché de MPS crece sin techo
            # (2,8 GB en uso, 21 GB reservados en ocho pasos) hasta tirar a swap.
            torch.mps.empty_cache()
        paso += 1
        if paso % 10 == 0:
            log.info("paso %d · época %d · pérdida %.4f · lr %.2e · %.2f s/paso", paso, epoca,
                     float(perdida), sched.get_last_lr()[0], (time.time() - t0) / 10)
            t0 = time.time()
        if paso % a.validar_cada == 0 or paso == a.pasos:
            log.info("paso %d · validación %.4f", paso, validar())
        if paso % a.guardar_cada == 0 or paso == a.pasos:
            modelo.save_pretrained(CKPT / f"paso-{paso}")
            _acompañar(CKPT / f"paso-{paso}")
            torch.save({"paso": paso, "opt": opt.state_dict(), "sched": sched.state_dict()},
                       estado)
            log.info("guardado paso-%d", paso)


def exportar(paso: int) -> None:
    """Un checkpoint a SALIDA en bf16, que es como lo carga NeuTTS: la mitad de
    disco y lo mismo al oído."""
    import torch
    from transformers import AutoModelForCausalLM

    if SALIDA.exists():
        shutil.rmtree(SALIDA)
    modelo = AutoModelForCausalLM.from_pretrained(CKPT / f"paso-{paso}", dtype=torch.bfloat16)
    modelo.config.use_cache = True
    modelo.save_pretrained(SALIDA)
    _acompañar(SALIDA)
    (SALIDA / "NOTICE").write_text(
        f"Obra derivada de {BASE} (NeuTTS Open License v1.0, ver LICENCE).\n"
        f"Modificado: afinado con la voz de referencia del Museo de la Educación Gabriela "
        f"Mistral con pipeline/afinar_neutts.py, paso {paso}.\n")
    log.info("exportado paso-%d a %s", paso, SALIDA)


# --- Evaluar -----------------------------------------------------------------


def _cargar(ruta: str):
    from neutts import NeuTTS

    voice._usar_espeak_del_sistema()
    dispositivo = voice._device()
    return NeuTTS(backbone_repo=BASE if ruta == "base" else ruta, backbone_device=dispositivo,
                  codec_repo="neuphonic/neucodec", codec_device=dispositivo,
                  language="es", seed=SEED)


def _toma(modelo, codigos, ref_texto, texto, temperatura, semilla=SEED) -> np.ndarray:
    """Como voice._sintetizar_neutts (o vivo._una_toma si es GGUF), con su RMS."""
    modelo._seed = semilla
    if modelo._is_quantized_model:
        x = np.concatenate([np.asarray(t, np.float32) for t in
                            modelo.infer_stream(texto, codigos, ref_texto, temperature=temperatura)])
    else:
        x = np.asarray(modelo.infer(texto, codigos, ref_texto, temperature=temperatura), np.float32)
    if len(x) and (rms := float(np.sqrt((x**2).mean()))):
        x = x * (RMS_OBJETIVO / rms)
    return x.clip(-1, 1)


def evaluar(a) -> None:
    """Cada modelo dice todas las frases de FRECUENTES, una toma por semilla,
    a la temperatura en vivo (TEMPERATURA) y a la de grabar. Mide la nota del
    juez, cuántas pasan GRABADO_ACEPTABLE, cuántas se van a otro tono, qué
    entiende Whisper y el factor de tiempo real."""
    from scipy.signal import resample_poly

    from gabriela import entender

    frases = [f for _, r in FRECUENTES for f in voice.trozos(r)][:a.frases or None]
    muestras = Path(a.muestras)
    resultados = {}
    for ruta in a.modelos:
        modelo = _cargar(ruta)
        ref_texto = voice._referencia()[1]
        codigos = modelo.encode_reference(str(REF_AUDIO))
        _toma(modelo, codigos, ref_texto, "Hola, soy Gabriela.", TEMPERATURA)  # calienta
        nombre = "base" if ruta == "base" else Path(ruta).name
        for temperatura in (TEMPERATURA, GRABADO_TEMPERATURA):
            notas, ajenas, cers, sintesis, audio = [], 0, [], 0.0, 0.0
            for n, (semilla, frase) in enumerate((s, f) for s in range(a.semillas) for f in frases):
                t = time.time()
                x = _toma(modelo, codigos, ref_texto, frase, temperatura, SEED + semilla)
                sintesis += time.time() - t
                audio += len(x) / voice.SAMPLE_RATE
                pcm = (x * 32767).astype("<i2").tobytes()
                notas.append(voice._puntuar(pcm, frase))
                if (tono := voice._tono(x)) and abs(12 * math.log2(tono / voice._tono_referencia())) > 4:
                    ajenas += 1
                x16 = resample_poly(x, 2, 3)
                cers.append(cer(frase, entender.transcribir((x16.clip(-1, 1) * 32767)
                                                            .astype("<i2").tobytes())))
                if n < a.ab:
                    d = muestras / f"t{temperatura:g}"
                    d.mkdir(parents=True, exist_ok=True)
                    (d / f"{n:02d}-{nombre}.wav").write_bytes(voice.a_wav(pcm))
            r = {"nota": float(np.mean(notas)),
                 "pasan": float(np.mean([v >= GRABADO_ACEPTABLE for v in notas])),
                 "ajenas": ajenas / len(notas), "cer": float(np.mean(cers)),
                 "rtf": sintesis / audio}
            resultados[f"{nombre} t={temperatura:g}"] = r
            log.info("%s t=%g: %s", nombre, temperatura, {k: round(v, 3) for k, v in r.items()})
        del modelo
    muestras.mkdir(parents=True, exist_ok=True)
    (muestras / "resultados.json").write_text(json.dumps(resultados, indent=1))
    print(f"\n{len(frases)} frases de FRECUENTES × {a.semillas} semillas, una toma cada vez\n")
    print("| modelo | nota media | pasan a la 1.ª | tono ajeno (>4 st) | CER Whisper | tiempo real |")
    print("|---|---|---|---|---|---|")
    for k, r in resultados.items():
        print(f"| {k} | {r['nota']:.2f} | {r['pasan']:.0%} | {r['ajenas']:.0%} | "
              f"{r['cer']:.1%} | {r['rtf']:.2f} |")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="orden", required=True)
    sub.add_parser("preparar")
    e = sub.add_parser("entrenar")
    e.add_argument("--pasos", type=int, default=600)
    e.add_argument("--lote", type=int, default=8)
    e.add_argument("--lr", type=float, default=2e-5)
    e.add_argument("--calentamiento", type=int, default=30)
    e.add_argument("--validar-cada", type=int, default=50)
    e.add_argument("--guardar-cada", type=int, default=100)
    x = sub.add_parser("exportar")
    x.add_argument("--paso", type=int, required=True)
    v = sub.add_parser("evaluar")
    v.add_argument("modelos", nargs="+", help="«base», una carpeta de transformers o un .gguf")
    v.add_argument("--frases", type=int, default=0, help="sólo las primeras N (0: todas)")
    v.add_argument("--semillas", type=int, default=1, help="tomas independientes por frase")
    v.add_argument("--ab", type=int, default=8, help="cuántas frases dejar como WAV")
    v.add_argument("--muestras", default=str(VOZ / "afinado" / "muestras"))
    a = p.parse_args()

    if a.orden == "preparar":
        preparar()
    elif a.orden == "entrenar":
        entrenar(a)
    elif a.orden == "exportar":
        exportar(a.paso)
    else:
        evaluar(a)


if __name__ == "__main__":
    assert cer("¡Hola, Mundo!", "hola mundo") == 0.0 and cer("abcd", "abed") == 0.25
    assert normalizar("En 1957, calle 3.150") == \
        "en mil novecientos cincuenta y siete calle tres mil ciento cincuenta"
    main()
