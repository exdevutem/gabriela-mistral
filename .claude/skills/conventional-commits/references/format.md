# Formato del mensaje

## Estructura

```
<type>[optional scope][!]: <description>

[optional body]

[optional footer(s)]
```

Reglas de la spec v1.0.0 (las que se rompen más seguido):

- `<type>` es un **sustantivo**, seguido de `: ` — dos puntos **y espacio**, obligatorio.
- El scope va entre paréntesis inmediatamente después del type: `feat(parser):`. Es un sustantivo que describe la sección del código.
- `!` va antes de los dos puntos y después del scope: `feat(api)!:`.
- Si el repo usa emojis, van **después** de `: `, nunca antes del type (ver [emojis.md](emojis.md)).
- El cuerpo empieza **una línea en blanco** después de la descripción. Puede tener varios párrafos.
- Los footers empiezan una línea en blanco después del cuerpo. Formato `Token: value` o `Token #value`; el token usa `-` en vez de espacios (`Reviewed-by`), **excepto** `BREAKING CHANGE`.
- Todo es case-insensitive salvo `BREAKING CHANGE`, que **debe ir en mayúsculas**.
- `BREAKING-CHANGE` es sinónimo válido de `BREAKING CHANGE` en el footer.

Convención de git (no es de la spec, pero se asume en todas partes):

- Descripción en **imperativo**, en minúscula, **sin punto final**.
- Header ≤72 caracteres (ideal ≤50). Sobre 100 falla en commitlint por defecto.
- Cuerpo envuelto a ~72 columnas. Explica el *qué* y el *por qué*, no el *cómo* (el cómo está en el diff). Omítelo si el header ya lo dice todo.

## Imperativo, no pasado ni gerundio

El header completa la frase "if applied, this commit will ..." / "este commit ...".

| Mal | Bien |
|---|---|
| `feat(api): added payments endpoint` | `feat(api): add payments endpoint` |
| `fix: fixing the checksum calc` | `fix: correct checksum calculation` |
| `docs: updated the README` | `docs: document deployment steps` |
| `feat(api): agregado endpoint de pagos` | `feat(api): agrega endpoint de pagos` |
| `fix: arreglando el cálculo de dv` | `fix: corrige cálculo de dv` |

Verbos frecuentes — EN: add, remove, fix, correct, update, rename, move, extract,
refactor, simplify, support, prevent, handle, document. ES: agrega, elimina,
corrige, actualiza, renombra, mueve, extrae, refactoriza, simplifica, soporta,
evita, maneja, documenta.
