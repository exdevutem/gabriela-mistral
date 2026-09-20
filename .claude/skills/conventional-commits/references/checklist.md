# Checklist antes de cada commit

- El type corresponde al cambio real y al bump SemVer que implica (no `chore` por comodidad).
- El header calza con `<type>[(scope)][!]: <description>` — dos puntos **y espacio**.
- Descripción en imperativo, minúscula, ≤72 caracteres, sin punto final.
- El idioma coincide con el del historial del repo.
- Emoji: solo si el repo ya los usa; uno, después de `: `, elegido por el contenido (💥 si hay breaking change).
- El commit contiene **un solo** tema.
- Línea en blanco antes del cuerpo y antes de los footers.
- `BREAKING CHANGE` en mayúsculas, con `!` en el header si hay footer.
- Se agregó `Refs`/`Closes` si hay ticket.
- No se están commiteando secretos, `.env`, credenciales, ni artefactos de build. Si aparecen en el diff, deténte y avisa antes de stagear.
- Si hay rama nueva, su nombre sigue [branches.md](branches.md).
