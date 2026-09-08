# Briefing — Busto 3D conversacional de Gabriela Mistral

Estado a 8 de septiembre de 2026 · 33 commits en `dev` · ~1.900 líneas

---

## Qué es

Un retrato tridimensional de Gabriela Mistral que responde por escrito, contesta con
voz sintetizada y mueve la cara mientras habla. Backend en Python, render en el
navegador.

**Funciona de punta a punta.** Se le escribe una pregunta, responde en carácter, la
voz suena y la boca acompaña. Verificado en navegador, no solo en tests.

Es una recreación con inteligencia artificial. La interfaz lo dice en pantalla y el
prompt le prohíbe inventar versos y atribuírselos: a un modelo al que se le pide
"recítame un poema tuyo" le sale fabricar estrofas convincentes y falsas, y en un
proyecto universitario sobre una Nobel eso hace más daño que un olvido honesto.

## Cómo se pone en marcha

```bash
uv sync
cp .env.example .env          # y pega tu GEMINI_API_KEY
uv run uvicorn gabriela.server:app --port 8000
```

- `localhost:8000` — la conversación
- `localhost:8000/debug` — un slider por morph target, para afinar visemas

Reconstruir la cabeza desde cero requiere descargar FLAME (ver *Dependencias
externas*) y ejecutar cuatro pasos:

```bash
uv run python pipeline/convert_flame.py     # .pkl → .npz de numpy puro
uv run python pipeline/fit_face.py assets/fotos/mistral-1946-frontal.jpg --preview
uv run python pipeline/hair_volume.py assets/fotos/mistral-1946-frontal.jpg
uv run python pipeline/export_glb.py        # → assets/gabriela.glb
```

## Arquitectura

Dos mitades que no se tocan en tiempo de ejecución.

**Pipeline offline** — se ejecuta una vez y congela su resultado en un `.glb`:

```
foto → landmarks (mediapipe) → ajuste FLAME (scipy) → volumen de pelo
     → textura proyectiva → GLB con morph targets
```

**Runtime** — FastAPI, WebSocket y three.js:

```
texto → Gemini (chat) → Gemini TTS → PCM 24 kHz
                              │
              envolvente RMS ─┴─ visemas del texto
                              │
        {audio, timeline} → three.js → morph targets
```

La clave del labio-sincronizado: los **tiempos** salen de la energía del audio real y
las **formas de boca** del texto. Por separado cada mitad falla —la energía sola da una
mandíbula de marioneta, el texto solo se desfasa en cuanto el TTS respira—. Los
visemas se reparten según energía acumulada y no según tiempo, así que las pausas no
adelantan la boca.

### Mapa de archivos

| Archivo | Qué hace |
|---|---|
| `pipeline/convert_flame.py` | `.pkl` de FLAME → `.npz`, simulando `chumpy` |
| `pipeline/torch_pickle.py` | Lee tensores de PyTorch sin instalar PyTorch |
| `pipeline/landmarks.py` | Foto → 478 landmarks → convención dlib de 68 |
| `pipeline/flame.py` | Evalúa FLAME con skinning de mandíbula |
| `pipeline/fit_face.py` | Ajusta forma y cámara a la fotografía |
| `pipeline/hair_volume.py` | Mide la silueta del peinado y engrosa el cráneo |
| `pipeline/textura.py` | Textura proyectiva desde la foto |
| `pipeline/rasgos.py` | Cejas y ojos procedurales (acabado sin foto) |
| `pipeline/export_glb.py` | Escribe el `.glb` con morph targets y textura |
| `src/gabriela/` | `chat`, `voice`, `visemes`, `persona`, `server` |
| `web/` | Visor three.js y página de ajuste |

## Decisiones, y por qué

Cada número de esta sección salió de medirlo, no de estimarlo.

**Modelo de lenguaje: `gemini-3.1-flash-lite` con razonamiento apagado.** Responde en
2,6 s frente a los 24 s de `gemini-3.6-flash`. Una conversación hablada no tolera
esperas largas. `gemini-2.5-flash` ya no está disponible para cuentas nuevas.

**Regularización del ajuste: 10.** Con 2, el error de reproyección baja a 5,3 px pero
el cráneo se aleja un 20 % de la forma media; con 10 el error solo sube a 7,2 px y el
desvío cae al 5 %. Como ningún landmark observa el cráneo, deformarlo le sale gratis
al optimizador: esa ganancia era sobreajuste, no parecido.

**Restricción de silueta.** Los vértices por encima de las cejas no pueden salirse del
contorno de la cabeza medido en la foto. Penalización unilateral: quedarse corto no
es error, solo sobresalir. Bajó el exceso de 27,4 a 8,8 px y el desvío del 4,9 % al
1,7 % sin empeorar el error de landmarks.

**Margen de 15 mm bajo la silueta.** Sin él, el cráneo se ajusta al contorno *del pelo*
y se come el hueco que el pelo debía ocupar: una vista frontal no distingue una cabeza
grande de una cabeza con cabello.

**Visemas medidos, no tanteados.** Los componentes de expresión de FLAME son PCA y no
significan nada por sí mismos, así que se midió su efecto sobre los landmarks de la
boca: `psi[0]` gobierna el ancho (±11 mm) y `psi[3]` la apertura de labios (+6 mm).

**Mapa de letras en vez de fonemas.** El español es fonéticamente casi transparente, y
eso ahorra `espeak`/`phonemizer` y su binario de sistema.

**Sin PyTorch ni chumpy.** El `.pkl` de FLAME envuelve sus arrays en `chumpy`
(abandonada) y el landmark embedding guarda tensores de PyTorch. Ambos se leen con
stubs propios: 2,5 GB de dependencias para una conversión que se hace una sola vez no
se sostiene.

## Lo que costó encontrar

Vale la pena dejarlo por escrito: cada uno de estos se descubrió tarde y ninguno era
evidente desde el código.

**`shapedirs` llegaba vacío.** El stub de `chumpy` descartaba el estado al reconstruir
los objetos, y el array del que depende todo el ajuste salía con forma `(0,)`. Está
en la clave `x` del estado.

**mediapipe 1.x revienta en macOS ARM** con `DrishtiMetalHelper / Service is
unavailable`: fuerza Metal y no hay forma de desactivarlo. El proyecto fija 0.10.35.

**Las fotos históricas son de un solo canal** y mediapipe exige tres. El grafo abortaba
con un error de dimensiones que no delata su causa. Afectaba a cualquier foto de
archivo, que es justo el material de este proyecto.

**Los morph targets necesitaban normales.** Sin ellas la geometría se deforma pero el
sombreado no acompaña: la boca se abre y no se nota.

**Una foto de "1940" resultó ser un cuadro.** 3.217 × 4.879 px del Arquivo Nacional de
Brasil, descrita en su ficha como fotografía. Es un retrato pintado con aerógrafo, con
turbante y geometría idealizada por el artista. Se detectó mirándola, no leyendo la
ficha.

**Doble sombreado con la textura.** La foto trae sus sombras horneadas y el esquema de
tres luces las duplicaba, ennegreciendo medio rostro. La escena se ilumina ahora de
forma plana.

**Caché del navegador.** El servidor entregaba el `avatar.js` nuevo y el navegador
ejecutaba el viejo. Resuelto con `Cache-Control: no-store`.

## Límites actuales

**De perfil no funciona.** La textura es proyectiva: solo vale para lo que la cámara
veía. Los vértices que miran hacia atrás muestrean la cara por el otro lado. Para una
conversación cara a cara alcanza; si se quiere permitir orbitar el modelo, se rompe.

**La latencia la marca el TTS.** El chat responde en ~2,6 s, pero la síntesis tarda
~15 s en frases largas y no admite streaming por esta vía.

**Una sola vista.** El ajuste monocular recupera proporciones, no profundidad. No hay
ninguna foto de perfil suya en dominio público con resolución suficiente; la mejor
candidata (Biblioteca Nacional Digital, ca. 1952) tiene licencia ambigua.

**El peinado es volumen liso.** Sin raya, ondas ni mechones, y cubre las orejas más de
lo que debería.

**Sin memoria entre sesiones.** Cada conexión abre su propia conversación.

## Próximos pasos

En orden de rendimiento por esfuerzo.

**1. Voz a voz con micrófono.** `gemini-3.1-flash-live-preview` está disponible con
esta cuenta y es el camino natural: elimina el teclado y hace la interacción
presencial. Es el paso que más cambia la experiencia.

**2. Reducir la latencia del habla.** Trocear la respuesta en frases y sintetizarlas
en cadena, empezando a hablar con la primera mientras se generan las siguientes.
Convierte 15 s de espera en unos 3.

**3. Parpadeo.** Falta el único gesto involuntario que el modelo no tiene, y es de los
que más separan un rostro vivo de una máscara. FLAME lo permite con un morph target
de párpados.

**4. Segunda vista para el ajuste.** Requiere aclarar la licencia del perfil ca. 1952
con la Biblioteca Nacional, o encontrar otro en dominio público. Resolvería a la vez
la profundidad y el volumen trasero del pelo.

**5. Esculpir el peinado en Blender.** El `.glb` ya se abre ahí. Es trabajo manual y no
reproducible por script, pero es lo único que dará raya y ondas reales.

**6. Textura multivista.** Si aparece el perfil, combinar ambas proyecciones con
mezcla por ángulo resolvería el límite de "solo de frente".

## Dependencias externas

**FLAME** — `flame.is.tue.mpg.de`, requiere registro. Licencia de investigación y uso
no comercial, compatible con un proyecto universitario, pero no automatizable. Hacen
falta tres archivos en `assets/flame/`:

- `flame2023.pkl` — el modelo
- `landmark_embedding.npy` — qué vértices son los 68 landmarks
- `FLAME_masks.pkl` — regiones semánticas; sin él, el cuero cabelludo se deduce por
  altura y el peinado sale con borde recto

**Modelo de landmarks de mediapipe** — se descarga aparte, la orden está en el README.

**API de Google AI Studio** — clave en `.env`. Ojo: `config.py` carga con
`override=True` a propósito, porque una variable exportada en el shell ganaba sobre el
`.env` del proyecto y devolvía 400.

## Fotografías y atribución

Todas de dominio público, verificadas una a una en Wikimedia Commons. **La ley chilena
mantiene los derechos morales aun en dominio público: la atribución es obligatoria**
allí donde se muestre el modelo.

| Archivo | Año | Autor | Uso |
|---|---|---|---|
| `mistral-1946-frontal.jpg` | 1946 | Marcos Chamúdez | Referencia del ajuste y textura |
| `mistral-1948-santabarbara.jpg` | ca. 1948 | George F. Weld | Referencia visual del peinado |
| `gabriela-referencia.png` | ca. 1945 | Anna Riwkin-Brick | Primera referencia, ya no se usa |

El detalle completo, incluidas las candidatas descartadas y el motivo de cada
descarte, está en `assets/fotos/PROCEDENCIA.md`.

Una nota metodológica que ahorra trabajo: al elegir otra foto, **mide los píxeles
útiles de cara, no la resolución del archivo**. Un escaneo de 4.000 px donde ella sale
de cuerpo entero aporta menos que uno de 1.700 px encuadrado en el rostro. Y no
mezcles épocas: FLAME ajusta forma facial, y promediar una cara de 33 años con una de
57 da un rostro que no es ninguna de las dos.

## Entorno de referencia

Desarrollado y verificado en un Mac M2 con 8 GB de RAM y Python 3.13. Esa restricción
dio forma al diseño: nada de modelos neurales en tiempo real ni LLM local. El único
paso pesado es el ajuste facial, corre offline una vez y su resultado se congela.

## Verificación

```bash
uv run python tests/test_visemes.py     # lógica de visemas, sin red
uv run python pipeline/flame.py         # autocomprobación del skinning
```

`test_visemes.py` cubre la única parte con ramas que puede romperse en silencio: si el
labio-sincronizado se desfasa, el avatar sigue funcionando y solo se ve raro.
`flame.py` verifica que con pose cero ningún vértice se mueve, que al rotar la
mandíbula baja la barbilla y que el cráneo queda quieto.
