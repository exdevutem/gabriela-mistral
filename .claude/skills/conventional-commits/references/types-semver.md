# Tipos e impacto en SemVer

## Impacto en SemVer

Esto es lo que el estándar existe para automatizar:

| Header | Bump |
|---|---|
| `fix:` | PATCH |
| `feat:` | MINOR |
| cualquier type con `!` o footer `BREAKING CHANGE:` | MAJOR |
| el resto (`docs`, `chore`, `test`, …) | ninguno |

Elegir el type es elegir la versión que se va a publicar. Un `fix` que en realidad
agrega comportamiento se publica como parche y rompe a quien confió en SemVer.

## Tipos

La spec solo obliga `feat` y `fix`. El resto viene de la convención de Angular,
que es la que asumen commitlint, semantic-release y release-please:

| Type | Cuándo usarlo |
|---|---|
| `feat` | Nueva funcionalidad (MINOR) |
| `fix` | Corrección de bug (PATCH) |
| `docs` | Solo documentación |
| `style` | Formato, lint, espacios, comillas (sin cambiar lógica) |
| `refactor` | Reorganizar código sin cambiar comportamiento ni arreglar bugs |
| `perf` | Mejoras de rendimiento |
| `test` | Agregar o corregir pruebas |
| `build` | Sistema de build o dependencias externas |
| `ci` | Pipelines, jobs, configuración de CI |
| `chore` | Tareas que no tocan `src` ni tests |
| `revert` | Revertir un commit previo |

Casos que se confunden seguido:

- Cambio que arregla algo **y** agrega comportamiento nuevo → sepáralos. Si no se puede, `feat` (el bump mayor manda) y menciona el arreglo en el cuerpo.
- Actualizar una dependencia por un CVE → `fix` (impacta al usuario, merece PATCH), no `chore`.
- Renombrar variables sin tocar comportamiento → `refactor`, no `style`.
- Tests de una feature nueva → van en el mismo `feat` si nacen juntos; si son tests de código ya existente, `test`.
- Migración SQL → `chore(db)` o el scope que use el repo; el código que la consume, `feat`/`fix` aparte.

No inventes types nuevos sin revisar el `type-enum` del repo: un type desconocido
hace que semantic-release ignore el commit en el CHANGELOG.
