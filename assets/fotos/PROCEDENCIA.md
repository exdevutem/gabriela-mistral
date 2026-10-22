# Procedencia de las fotografías

Todas de dominio público o con licencia libre, verificadas una por una en
Wikimedia Commons y en el Nationaal Archief de los Países Bajos. La ley
chilena mantiene los derechos morales aun en dominio público, así que **la
atribución es obligatoria**: hay que acreditar autor y fuente allí donde se
muestre o publique el modelo.

| Archivo | Año | Autor | Licencia | Interocular |
|---|---|---|---|---|
| `mistral-1946-frontal.jpg` | 1946 | Marcos Chamúdez (1907–1989) | Dominio público en EE.UU. (publicada 1931–1977 sin aviso de copyright) | **170 px** |
| `mistral-1948-santabarbara.jpg` | ca. 1948 | George F. Weld | Dominio público (ley chilena) — Memoria Chilena, Biblioteca Nacional de Chile | no detectada |
| `gabriela-referencia.png` | ca. 1945 | Anna Riwkin-Brick (1908–1970) | Dominio público | 37 px |
| `mistral-1946-sanfrancisco.jpg` | 1946 | Ministerio de Relaciones Exteriores de Chile — [Commons](https://commons.wikimedia.org/wiki/File:Arnaldo_Tapia,_Gabriela_Mistral,_Juan_Guzman.jpg) | CC BY 2.0 CL (atribución obligatoria) | ~50 px |
| `mistral-1946-truman.jpg` | 16-03-1946 | Desconocido, Casa Blanca — [Commons](https://commons.wikimedia.org/wiki/File:Harry_Truman-Gabriela_Mistral-Mora-D%C3%ADez.jpg) | CC0 | ~60 px, semitono impreso |
| `mistral-1946-microfono.jpg` | 13-03-1946 | INP, Nationaal Archief — [ficha](https://hdl.handle.net/10648/a89a04ec-d0b4-102d-bcf8-003048976d84) | Public Domain Mark 1.0 | ~170 px |
| `mistral-perfil-nationaalarchief.jpg` | archivada 14-12-1945; retrato de los años 30 | INP, Nationaal Archief — [ficha](https://hdl.handle.net/10648/a88f52d6-d0b4-102d-bcf8-003048976d84) | Public Domain Mark 1.0 | perfil, sin interocular |

Los cuatro últimos son recortes de los originales, centrados en su cara; el del
perfil está además reducido a la mitad (la copia en papel trae 1200 px de cara,
más de lo que el ajuste aprovecha).

## Cuál usar para qué

- **`mistral-1946-frontal.jpg` es la referencia para el ajuste.** 170 px entre
  ojos frente a los 37 px de la original: 4,6× más detalle facial, encuadre
  frontal (giro +0.06) y la edad correcta, 57 años.
- **Las tres de 1946 en tres cuartos** (San Francisco, Truman, micrófono)
  entran al ajuste junto a la frontal: comparten la forma β y cada una trae su
  cámara y su expresión. Son fotos de prensa con la cara pequeña o tapada en
  parte —el micrófono le cubre la boca, la boina la frente—, así que aportan
  poco cada una, pero ninguna empeora el resultado medido.
- **`mistral-perfil-nationaalarchief.jpg` es la única medida directa de
  profundidad.** Perfil derecho puro contra fondo liso. mediapipe no detecta
  caras de perfil, así que lleva anotaciones a mano en `perfiles.json`. Es un
  retrato de los años 30, más joven que la referencia; se usa sólo su silueta,
  que depende del hueso —frente, nariz, mentón— más que de la edad.
- `mistral-1948-santabarbara.jpg` sirve como referencia visual del peinado, el
  porte y la ropa. mediapipe no le detecta la cara a ninguna resolución: está
  girada, en sombra y hablando. No sirve para el ajuste, sí para modelar el pelo.

## Descartadas, y por qué

- **Página del *Forward* (25-11-1945)**, en Commons: trae el mismo retrato de
  perfil que el Nationaal Archief, pero impreso en semitono y a un sexto de la
  resolución. Se usó primero y se reemplazó.
- **`Gabriela Mistral sonriendo ca. 1938.jpg`** (Biblioteca Nacional de Chile,
  CC0): perfil a unos 75°, nítido pero sonriendo, y el fondo tiene una franja
  oscura justo a la altura del entrecejo. Se dejó fuera del ajuste para usarla
  como **validación**: medir si el modelo predice la silueta de una foto que no
  vio. No discrimina: todas las versiones del ajuste quedan entre 1,1 y 1,3 mm.
- **Uppsala, diciembre de 1945** (Upplands Museum, en DigitaltMuseum): cuatro
  fotos de grupo con la cara de unos 120 px y capucha. Licencia CC BY-NC-ND 4.0,
  más restrictiva que el resto; no aportan lo suficiente para justificarla.
- **Visita a Vicuña, 1954** (Museo Gabriela Mistral, fotografia.surdoc.cl,
  registros 19-30, 19-157, 19-66, 19-165): "prohibida la reproducción sin
  autorización de la Dirección del Museo". Con el convenio MEGM–UTEM es la vía
  natural para conseguir fotos de su última época; la 19-30, de pie ante un
  micrófono en la Plaza de Armas, es la más prometedora.
- **Martha Holmes / *Life*, Nueva York 1953** (Memoria Chilena): probablemente el
  mejor retrato de su vejez, con derechos reservados.

- **`Gabriela Mistral (1922).jpg`** (4206×5625, 789 px interoculares, la de mayor
  calidad con diferencia): tiene **33 años** en ella. FLAME ajusta forma facial, y
  promediar una cara de 33 con una de 57 da un rostro que no se parece a ninguna
  de las dos. Hay que elegir una época y no salirse.
- **`Gabriela Mistral (1940).tif`** (3217×4879, Arquivo Nacional de Brasil): pese a
  la resolución y a que la ficha la describe como fotografía, **es un retrato
  pintado o retocado con aerógrafo**, con un turbante que tapa frente y pelo. La
  geometría facial está idealizada por el artista. Verificado mirándola.
- **Museo Histórico Nacional** (fotografiapatrimonial.cl): prohíbe expresamente
  redistribuir en alta resolución y exige autorización previa. No es dominio
  público utilizable aquí.
- **Retrato de perfil ca. 1952** (Biblioteca Nacional Digital, primer plano, 17,4 ×
  23,8 cm): sería ideal como segunda vista para recuperar profundidad, pero la
  ficha dice "algunos derechos reservados" sin especificar licencia. Queda
  pendiente de aclarar antes de usarlo.
- **`Gabriela Mistral 1945.jpg`** (Anna Riwkin-Brick): es un perfil real, pero mide
  280×396 y sólo 26 px entre ojos. Demasiado poco para aportar nada.
