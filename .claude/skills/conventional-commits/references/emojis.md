# Emojis

Opcional y **basado en el contenido**, no en el type. El type ya dice la
categoría; el emoji dice de qué se trata el cambio: un `fix` puede ser 🐛 (bug
normal), 🚑 (hotfix en producción), 🔒 (agujero de seguridad) o 💚 (CI roja).

**Cuándo usarlos**: solo si el repo ya los usa. Mira el historial antes de decidir:

```bash
git log --format=%s -50 | grep -cP '^\w+(\([^)]*\))?!?: [\x{1F300}-\x{1FAFF}\x{2600}-\x{27BF}]'
```

Si la mayoría de los commits recientes los trae, síguelos. Si ninguno, no los
introduzcas por tu cuenta — pregunta. Y si empiezas, es para todos los commits:
un historial a medias es peor que uno sin emojis.

**Dónde va** — después de los dos puntos, antes de la descripción, seguido de un
espacio:

```
feat(auth): ✨ add refresh token rotation
```

Nunca antes del type (`✨ feat(auth): ...` o `:sparkles: feat: ...` al estilo
gitmoji clásico): el regex de `@commitlint/config-conventional`, de
semantic-release y de release-please parsea desde el inicio de la línea y esos
commits quedan fuera del CHANGELOG y del cálculo de versión.

**Uno solo por commit.** Si dudas entre dos, el commit probablemente tiene dos
temas — sepáralo. Y el emoji ocupa su espacio dentro de los 72 caracteres: el
presupuesto de la descripción baja a ~69.

**Selección por contenido** (vocabulario de [gitmoji](https://gitmoji.dev), recortado a lo que se usa):

| Emoji | El cambio… | Types típicos |
|---|---|---|
| 💥 | rompe compatibilidad — **gana sobre cualquier otro** | cualquiera con `!` |
| ✨ | introduce una funcionalidad nueva | `feat` |
| 🐛 | corrige un bug | `fix` |
| 🚑 | es un hotfix urgente en producción | `fix` |
| 🩹 | corrige algo menor, no crítico | `fix` |
| 🔒 | tapa un problema de seguridad | `fix` |
| 🔐 | toca secretos, credenciales o llaves | `chore`, `ci` |
| ⚡️ | mejora el rendimiento | `perf` |
| ♻️ | reorganiza código sin cambiar comportamiento | `refactor` |
| 🔥 | elimina código o archivos | `refactor`, `chore` |
| 🏗️ | cambia la arquitectura o la estructura del proyecto | `refactor` |
| 💄 | toca UI, estilos o diseño | `feat`, `style` |
| 🎨 | mejora formato o estructura del código, sin lógica | `style` |
| 🚨 | apaga warnings del linter o del compilador | `style`, `fix` |
| 📝 | escribe o actualiza documentación | `docs` |
| ✅ | agrega o corrige tests | `test` |
| 👷 | toca el pipeline de CI | `ci` |
| 💚 | arregla la CI que estaba roja | `ci`, `fix` |
| 🔧 | cambia archivos de configuración | `chore`, `build` |
| ⬆️ / ⬇️ | sube / baja versiones de dependencias | `build`, `chore` |
| ➕ / ➖ | agrega / elimina una dependencia | `build` |
| 📦 | toca el empaquetado o la publicación del artefacto | `build`, `chore` |
| 🗃️ | cambia esquema, migraciones o datos de la BD | `chore`, `feat` |
| 🌐 | agrega o corrige internacionalización | `feat`, `fix` |
| ♿️ | mejora accesibilidad | `feat`, `fix` |
| 🔊 / 🔇 | agrega / quita logs o trazas | `chore`, `feat` |
| 🚀 | tiene que ver con despliegue | `ci`, `chore` |
| ⏪️ | revierte un commit previo | `revert` |
| 🔖 | marca un release o una versión | `chore` |
| 🚧 | es trabajo a medias (solo en ramas, nunca en la principal) | cualquiera |

Si el contenido no calza con ninguno, usa el genérico del type (✨ `feat`, 🐛
`fix`, 📝 `docs`, ♻️ `refactor`, ⚡️ `perf`, ✅ `test`, 🔧 `build`, 👷 `ci`, 🧹
`chore`) antes que forzar un emoji que confunda.

**Ejemplos con contenido distinto bajo el mismo type**

```
fix(api): 🔒 reject tokens signed with the none algorithm
fix(api): 🚑 restore payment webhook dropped by the last deploy
fix(build): ⬆️ bump axios to 1.7.9 to clear CVE-2024-28849
fix(ci): 💚 pin runner image so the e2e job stops flaking
```

```
feat(reportes): 🗃️ agrega tabla de agregados mensuales
feat(ui): 💄 agrega estado vacío en el listado de trámites
feat(api)!: 💥 exige API key en todos los endpoints v1
```
