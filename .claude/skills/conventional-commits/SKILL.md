---
name: conventional-commits
description: Genera y ejecuta commits siguiendo Conventional Commits v1.0.0 (conventionalcommits.org), en inglés o español según el repo, y nombra ramas con el formato `<type>/[<scope>-][<ticket>-]<slug>` — inspecciona git status/diff, agrupa cambios en commits atómicos y redacta headers `<type>[(scope)][!]: <description>` con cuerpo, footers y BREAKING CHANGE. Úsala SIEMPRE que se pida "commitea", "haz el commit", "arma los commits", "commit this", "write a commit message", "qué mensaje le pongo a esto", "revisa mi commit", "divide esto en commits", "conventional commit", "crea la rama", "qué nombre le pongo a la rama", "branch name", o cuando haya cambios sin commitear en un repo que ya usa Conventional Commits (headers `feat:`, `fix:`, `chore:` en el historial) o commitlint/semantic-release/changesets. NO la uses en repos con otra convención propia —p. ej. repos SISEI/UTEM, que tienen commits-sisei— ni si el historial es de formato libre.
---

# Conventional Commits

Convierte un working tree sucio en una secuencia de commits limpios que cumplen
[Conventional Commits v1.0.0](https://www.conventionalcommits.org/en/v1.0.0/).
La regla que gobierna todo lo demás: **un cambio por commit**. El historial se lee
después, muchas veces, por gente que no estuvo ahí — y por las herramientas que
generan el CHANGELOG y calculan la versión SemVer.

## Idioma del mensaje / Message language

El estándar no define idioma. Esta skill trabaja en **inglés y español**:

1. Mira `git log --format=%s -30`. Si el repo ya escribe en un idioma, úsalo. Sin excepciones.
2. Repo nuevo o historial mixto: **inglés por defecto** (es lo que espera cualquier lector externo y lo que asume el tooling de CHANGELOG).
3. Si el usuario pide explícitamente un idioma, manda el usuario.

Los `type` y los tokens de footer (`BREAKING CHANGE`, `Refs`, `Closes`,
`Reviewed-by`) van **siempre en inglés**, en cualquier idioma que esté el
resumen. Los identificadores de código van tal cual (`user_id`, `X-Api-Key`).

## Componentes

Lee lo que la tarea necesite; [workflow.md](references/workflow.md) siempre que vayas a commitear.

| Archivo | Contenido | Cuándo leerlo |
|---|---|---|
| [workflow.md](references/workflow.md) | Pasos 0–5: rama, estado del repo, agrupar, plan, ejecutar, validar | Siempre antes de commitear |
| [format.md](references/format.md) | Estructura del mensaje, reglas de la spec, imperativo | Al redactar o revisar un mensaje |
| [types-semver.md](references/types-semver.md) | Tabla de types, bump SemVer, casos confusos | Al elegir el type |
| [scopes.md](references/scopes.md) | Cómo elegir o omitir el scope | Al elegir el scope |
| [emojis.md](references/emojis.md) | Cuándo, dónde y cuál emoji | Solo si el repo ya usa emojis |
| [footers.md](references/footers.md) | Footers, BREAKING CHANGE, revert | Con tickets, quiebres o reverts |
| [branches.md](references/branches.md) | Estándar de nombres de rama | Al crear, nombrar o revisar una rama |
| [commitlint.md](references/commitlint.md) | Config base de commitlint | Si piden configurarlo |
| [examples.md](references/examples.md) | Mensajes completos de referencia | Para calibrar el resultado |
| [checklist.md](references/checklist.md) | Verificación final | Antes de cada `git commit` |

## Validador

```bash
python3 scripts/validate_commit.py "feat(auth): add refresh token rotation"
python3 scripts/validate_commit.py --full mensaje.txt
python3 scripts/validate_commit.py --branch feat/auth-214-refresh-token-rotation
```

Detalle en [workflow.md](references/workflow.md#5-validar).
