"""Comprobaciones de la lógica de visemas. Sin framework: `python tests/test_visemes.py`.

Es la única parte del sistema con ramas que puede romperse en silencio: si el
labio-sincronizado se desfasa, el avatar sigue funcionando y solo se ve raro.
"""
import sys

import numpy as np

sys.path.insert(0, "src")
from gabriela.visemes import REPOSO, VENTANA_MS, envolvente, secuencia, timeline

SR = 24_000


def pcm_de(envolvente_deseada, dur_ventana=VENTANA_MS / 1000):
    """Fabrica PCM con la energía pedida por ventana, sin llamar a la API."""
    n = int(SR * dur_ventana)
    tramos = [
        (np.random.default_rng(0).normal(0, amp, n) * 32767).clip(-32767, 32767)
        for amp in envolvente_deseada
    ]
    return np.concatenate(tramos).astype(np.int16).tobytes()


def test_secuencia_mapea_vocales_y_labiales():
    assert secuencia("mamá") == ["MBP", "A", "MBP", "A"]
    assert secuencia("foo") == ["FV", "O"], "vocales repetidas se colapsan"
    assert secuencia("¿Sí?") == ["C", "I"], "puntuación y tildes no rompen el mapa"
    assert secuencia("") == []


def test_timeline_ordenado_y_cubre_el_audio():
    dur = 2.0
    pcm = pcm_de([0.5] * int(dur / (VENTANA_MS / 1000)))
    tl = timeline(pcm, "una frase cualquiera de prueba", SR)
    tiempos = [t for t, _, _ in tl]
    assert tiempos == sorted(tiempos), "los keyframes deben ir en orden"
    assert tiempos[0] == 0.0, "debe arrancar en cero"
    assert abs(tiempos[-1] - dur) < 0.1, f"debe cubrir los {dur}s de audio, llega a {tiempos[-1]}"
    assert all(0.0 <= w <= 1.0 for _, _, w in tl), "pesos fuera de rango"


def test_silencio_no_mueve_la_boca():
    tl = timeline(pcm_de([0.0] * 50), "hola", SR)
    assert {v for _, v, _ in tl} == {REPOSO}
    assert all(w == 0.0 for _, _, w in tl)


def test_visemas_respetan_el_orden_del_texto():
    pcm = pcm_de([0.5] * 100)
    dichos = [v for _, v, _ in timeline(pcm, "mi papa", SR) if v != REPOSO]
    sin_repetir = [v for i, v in enumerate(dichos) if i == 0 or v != dichos[i - 1]]
    esperado = secuencia("mi papa")
    assert sin_repetir == esperado, f"{sin_repetir} != {esperado}"


def vigente(tl, t):
    """Visema activo en el segundo t: el último keyframe emitido antes de t.

    Hace falta porque el timeline solo guarda cambios; durante un tramo estable
    no hay keyframes que mirar.
    """
    return [v for ts, v, _ in tl if ts <= t][-1]


def test_pausas_no_consumen_visemas():
    """Un silencio largo en medio no debe adelantar la boca: ahí está la gracia
    de repartir por energía acumulada y no por tiempo."""
    pcm = pcm_de([0.5] * 25 + [0.0] * 50 + [0.5] * 25)
    tl = timeline(pcm, "mi papa", SR)
    en_pausa = {vigente(tl, t) for t in (0.7, 1.0, 1.3)}
    assert en_pausa == {REPOSO}, f"la boca se mueve durante el silencio: {en_pausa}"
    hablados = {v for _, v, _ in tl if v != REPOSO}
    assert hablados == set(secuencia("mi papa")), "se perdieron visemas por la pausa"


def test_envolvente_normaliza():
    env = envolvente(pcm_de([0.1, 0.5, 0.25]), SR)
    assert abs(env.max() - 1.0) < 1e-6
    assert env.argmax() == 1


if __name__ == "__main__":
    fallos = 0
    for nombre, fn in sorted(globals().items()):
        if nombre.startswith("test_"):
            try:
                fn()
                print(f"  ok   {nombre}")
            except AssertionError as e:
                fallos += 1
                print(f"  FALLA {nombre}: {e}")
    print("todo verde" if not fallos else f"{fallos} fallo(s)")
    sys.exit(1 if fallos else 0)
