# Nombramiento de ramas

Adaptación a Conventional Commits del formato de ramas del protocolo SISEI
(`<tipo>/<scope>-<id-ticket>-<slug-corto>`). La idea es la misma: el nombre de la
rama anticipa el header del commit (o del squash merge) que va a producir.

## Formato

```
<type>/[<scope>-][<ticket>-]<slug>
```

| Parte | Regla |
|---|---|
| `type` | Uno de los types de Conventional Commits ([types-semver.md](types-semver.md)): `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`. Si el repo define su `type-enum`, usa ese. |
| `scope` | Opcional. El mismo scope que llevará el commit ([scopes.md](scopes.md)). Si el cambio es transversal, omítelo. |
| `ticket` | Opcional. El ID tal como existe en el tracker: `PROJ-214` para Jira/Linear (mayúsculas, así lo detectan las integraciones), solo el número para issues de GitHub/GitLab (`214`). |
| `slug` | Obligatorio. 2 a 5 palabras en kebab-case que resumen el cambio. |

Separadores: `/` solo una vez, entre type y el resto. Todo lo demás, `-`.

## Reglas

- **Minúsculas** en todo, salvo el ID del ticket.
- **Solo ASCII**: `[a-z0-9-]`. Sin tildes, `ñ`, espacios, `_`, `.` ni emojis — rompen scripts, URLs y shells. `validacion-rut`, no `validación-rut`.
- **Idioma del slug = idioma de los commits** del repo ([format.md](format.md)).
- **Sin verbo conjugado**: el slug es un sustantivo o frase corta (`refresh-token-rotation`, `dv-k-proveedor`); el imperativo queda para el commit.
- **Largo**: ≤50 caracteres en total, idealmente menos. Se muestra truncado en casi todas las UIs.
- **Sin `!`** aunque el cambio sea breaking: el `!` choca con la expansión de historial de bash/zsh. El breaking change se marca en el commit.
- Un tema por rama, igual que un tema por commit. Si el slug necesita "y" (`and`), son dos ramas.

## Ejemplos

```
feat/auth-214-refresh-token-rotation
feat/tramites-VRAC-214-filtro-carrera
fix/conta-312-dv-k-proveedor
fix/api-null-payload-500
docs/deployment-steps
refactor/api-extract-rut-helper
build/bump-axios-1-7-9
ci/deploy-auto-rollback
chore/remove-unused-imports
revert/payments-v2-endpoint
```

| Mal | Bien | Por qué |
|---|---|---|
| `feature/login` | `feat/auth-login-form` | `feature` no es un type |
| `Fix/Conta_DV` | `fix/conta-dv-k-proveedor` | mayúsculas y `_` |
| `feat/tramites/filtro` | `feat/tramites-filtro-carrera` | más de un `/` |
| `feat/api!-require-key` | `feat/api-require-api-key` | `!` en el nombre |
| `fix/corrige-cálculo-dv` | `fix/dv-calculation` | tilde y verbo conjugado |
| `juan/wip` | `feat/reportes-monthly-aggregates` | sin type ni tema |

## Ramas especiales

No siguen el formato anterior:

- Principales: `main` (o `master`), y `develop` si el repo usa git-flow.
- Releases: `release/<version>` → `release/2.4.0`.
- Hotfix en producción: usa `fix/…` saliendo desde la rama/tag de producción. Si el repo ya usa `hotfix/<slug>`, respétalo.
- Ramas generadas por herramientas (`dependabot/…`, `renovate/…`, `release-please--…`, `changeset-release/…`): no se renombran.

Si el historial de ramas del repo (`git branch -a`) muestra otra convención
consistente, **esa gana**.

## De la rama al commit

La rama se lee como el header que viene:

```
feat/auth-214-refresh-token-rotation
  → feat(auth): add refresh token rotation
    Refs: #214
```

En squash merge, el título del PR se convierte en el commit de la rama principal:
redáctalo con el mismo formato `<type>[(scope)][!]: <description>` y valídalo con
`scripts/validate_commit.py`.

## Crear y validar

```bash
git switch -c feat/auth-214-refresh-token-rotation
```

Antes de crearla, verifica el formato:

```bash
python3 scripts/validate_commit.py --branch feat/auth-214-refresh-token-rotation
git branch --format='%(refname:short)' | python3 scripts/validate_commit.py --branch
```

Acepta las ramas especiales de arriba y explica cada falla (type, `/`, `!`,
ASCII, mayúsculas, guiones, slug, largo). Sin Python, el equivalente en shell:

```bash
b='feat/auth-214-refresh-token-rotation'
git check-ref-format --branch "$b" >/dev/null \
  && echo "$b" | grep -Eq '^(feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert)/([a-z0-9]+-)*([A-Z][A-Z0-9]+-[0-9]+-)?[a-z0-9]+(-[a-z0-9]+)*$' \
  && [ ${#b} -le 50 ] && echo ok || echo "nombre inválido"
```

Si el repo quiere forzarlo, el validador o el regex van en un hook `pre-push` o en un job
de CI que revise `github.head_ref`.

Renombrar una rama local mal nombrada (antes de hacer push):

```bash
git branch -m <nombre-viejo> <nombre-nuevo>
```

Si ya está publicada, avisa antes de renombrarla: rompe los PRs abiertos y las
copias locales del resto del equipo.
