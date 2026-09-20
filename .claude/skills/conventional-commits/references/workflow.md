# Flujo de trabajo

## 0. Rama

Si piden crear la rama, o el trabajo nuevo está sobre `main`/`master`/`develop`,
propón una rama con el formato de [branches.md](branches.md) antes de commitear.
No cambies de rama por tu cuenta si hay cambios sin commitear que no son tuyos.

## 1. Levantar el estado real del repo

No adivines lo que cambió: míralo.

```bash
git status --porcelain=v1 -uall
git diff --stat
git diff                 # cambios sin stagear
git diff --cached        # cambios ya stageados
git log --format=%s -30  # convención, idioma y uso de emojis vigentes
```

Si hay archivos nuevos, revisa su contenido antes de decidir el tipo
(`git diff --no-index /dev/null <archivo>` o léelo directamente).

Revisa también si el repo tiene reglas propias que manden sobre las defaults:
`commitlint.config.*`, `.commitlintrc*`, `.czrc`, `.versionrc`, `.changeset/`,
`release-please-config.json`, `CONTRIBUTING.md`. Si existen, **esas reglas ganan**
(tipos permitidos, scopes, largo máximo).

## 2. Agrupar en commits atómicos

Recorre el diff y agrupa los hunks por **intención**, no por archivo. Señales de
que hay que separar:

- Un archivo trae el arreglo del bug *y* de paso el reordenamiento de imports → `fix` + `style`.
- Migración de BD + el endpoint que la usa → dos commits (`chore(db)` / `feat`).
- Bump de dependencia en `package.json`/`pyproject.toml`/`go.mod` → `build` aparte.
- Formateo automático masivo → siempre su propio `style`, nunca mezclado.

Cuando un mismo archivo tiene cambios de dos temas, usa `git add -p` para stagear
por hunk. Si eso se vuelve inmanejable, dilo y propone commitear junto explicando
el motivo, en vez de partir el archivo mal.

## 3. Proponer el plan antes de ejecutar

Muestra el plan y espera confirmación antes del primer `git commit`:

```
1. feat(auth): add refresh token rotation          [MINOR]
   → src/auth/tokens.ts, src/auth/middleware.ts

2. fix(api)!: return 422 instead of 500 on bad payload   [MAJOR]
   → src/api/handlers.ts   BREAKING CHANGE
```

Marca el impacto SemVer de cada commit — es la mitad del valor del estándar. Si
falta información que no sale del diff (el ID del ticket, si un cambio de esquema
requiere migración manual), pregúntala acá, no después de haber commiteado.

## 4. Ejecutar

Stagea explícitamente por ruta (nunca `git add -A` a ciegas, que arrastra archivos
de otro commit) y commitea con `-F -` para que el cuerpo mantenga sus saltos de línea:

```bash
git add src/auth/tokens.ts src/auth/middleware.ts
git commit -F - <<'EOF'
feat(auth): add refresh token rotation

Refresh tokens are now single-use and rotated on every exchange, so a
leaked token stops being valid as soon as the legitimate client refreshes.

Refs: #214
EOF
```

Después de cada commit verifica el working tree (`git status --porcelain`). Al
terminar, muestra `git log --oneline -N`.

**No hagas `git push` salvo que te lo pidan explícitamente.** Tampoco uses
`--no-verify`: si un hook falla, arregla la causa o repórtala.

## 5. Validar

Antes de commitear, pasa cada header por el validador:

```bash
python3 scripts/validate_commit.py "feat(auth): add refresh token rotation"
git log --format=%s -20 | python3 scripts/validate_commit.py
python3 scripts/validate_commit.py --full mensaje.txt   # valida header + footers
python3 scripts/validate_commit.py --branch "$(git branch --show-current)"
```

Detecta idioma solo, revisa la gramática del header, el largo, el `!`, la
posición del emoji, el formato de los footers y que `BREAKING CHANGE` vaya en
mayúsculas. Si el repo tiene
commitlint, corre también `npx commitlint --edit` — esa es la fuente de verdad de CI.
