# Configuración de Claude Code

Lo que hay aquí se aplica a quien trabaje en este repositorio con Claude Code.
No afecta a nada del proyecto en ejecución.

## `skills/conventional-commits/`

El estándar de commits del repositorio, como skill. Claude la carga al preparar
commits; una persona puede leer `SKILL.md` y `references/` igual de bien.

Resumen: `<tipo>[(scope)]: <resumen en imperativo y minúsculas>`, máximo 72
caracteres, sin punto final, en español. Tipos en uso: `feat`, `fix`, `docs`,
`build`, `test`, `refactor`, `perf`, `chore`. Scopes en uso: `voz`, `chat`,
`ui`, `api`, `pipeline`, `deploy`.

Para validar un encabezado a mano:

```bash
python3 .claude/skills/conventional-commits/scripts/validate_commit.py "feat(voz): agrega parpadeo"
```

## `hooks/recordar-estado.sh`

Se ejecuta antes de cada `git push`. Si los commits que van a subir tocan código
o infraestructura pero no `README.md` ni `briefing.md`, le recuerda a Claude que
actualice el estado del proyecto antes de empujar.

**Por qué existe.** Lo que falta por hacer y lo ya terminado se olvidan en la
conversación, no en el repositorio. Ya pasó dos veces: los «Próximos pasos»
listaban como pendiente algo terminado dos turnos antes, y que la imagen Docker
nunca se había construido constaba solo en un chat. Quien retoma el proyecto lee
los archivos, no el historial de nadie.

No bloquea el push ni corre en un push de solo documentación. Para revisarlo o
desactivarlo: `/hooks`.
