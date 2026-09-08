# Procedencia de las fotografías

Todas de dominio público, verificadas una por una en Wikimedia Commons. La ley
chilena mantiene los derechos morales aun en dominio público, así que **la
atribución es obligatoria**: hay que acreditar autor y fuente allí donde se
muestre o publique el modelo.

| Archivo | Año | Autor | Licencia | Interocular |
|---|---|---|---|---|
| `mistral-1946-frontal.jpg` | 1946 | Marcos Chamúdez (1907–1989) | Dominio público en EE.UU. (publicada 1931–1977 sin aviso de copyright) | **170 px** |
| `mistral-1948-santabarbara.jpg` | ca. 1948 | George F. Weld | Dominio público (ley chilena) — Memoria Chilena, Biblioteca Nacional de Chile | no detectada |
| `gabriela-referencia.png` | ca. 1945 | Anna Riwkin-Brick (1908–1970) | Dominio público | 37 px |

## Cuál usar para qué

- **`mistral-1946-frontal.jpg` es la referencia para el ajuste.** 170 px entre
  ojos frente a los 37 px de la original: 4,6× más detalle facial, encuadre
  frontal (giro +0.06) y la edad correcta, 57 años.
- `mistral-1948-santabarbara.jpg` sirve como referencia visual del peinado, el
  porte y la ropa. mediapipe no le detecta la cara a ninguna resolución: está
  girada, en sombra y hablando. No sirve para el ajuste, sí para modelar el pelo.

## Descartadas, y por qué

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
