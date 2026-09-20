#!/usr/bin/env bash
# Antes de un `git push`, avisa si los commits que van a subir tocan código o
# infraestructura pero no el estado del proyecto.
#
# El motivo: lo que queda por hacer y lo ya hecho se olvidan en la conversación
# y no en el repositorio. Ya pasó —los "próximos pasos" listaban como pendiente
# algo terminado dos turnos antes, y que la imagen Docker nunca se había
# construido solo constaba en el chat—. Quien retome el proyecto lee los
# archivos, no el historial de nadie.
#
# No bloquea el push: inyecta un recordatorio en el contexto de Claude. Un push
# de solo documentación, o sin commits pendientes, no dice nada.
set -uo pipefail

cd "${CLAUDE_PROJECT_DIR:-$PWD}" 2>/dev/null || exit 0
git rev-parse --git-dir >/dev/null 2>&1 || exit 0

# Commits que aún no están en el remoto. Sin rama de seguimiento, los del día.
if upstream=$(git rev-parse --abbrev-ref --symbolic-full-name '@{u}' 2>/dev/null); then
    rango="$upstream..HEAD"
else
    rango="HEAD"
fi

tocados=$(git log --name-only --pretty=format: "$rango" 2>/dev/null | sort -u)
[ -z "$tocados" ] && exit 0

codigo=$(printf '%s\n' "$tocados" | grep -Ec '^(src/|web/|pipeline/|tests/|docker/|Dockerfile|pyproject\.toml)' || true)
estado=$(printf '%s\n' "$tocados" | grep -Ec '^(README\.md|briefing\.md|docker/README\.md)' || true)

[ "$codigo" -eq 0 ] && exit 0
[ "$estado" -gt 0 ] && exit 0

cat <<'JSON'
{"hookSpecificOutput":{"hookEventName":"PreToolUse","additionalContext":"Este push lleva cambios de código o infraestructura sin tocar README.md ni briefing.md. Antes de empujar, actualiza el estado del proyecto: mueve a 'realizado' lo que se terminó, quita de 'Próximos pasos' lo que ya está hecho, anota los límites nuevos que se descubrieron y marca lo que quedó sin verificar. Si de verdad no hay nada que actualizar, sigue adelante y dilo en una línea."}}
JSON
