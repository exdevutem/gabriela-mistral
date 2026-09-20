# Footers, breaking changes y revert

## Footers

```
Refs: #214
Closes: #102
Reviewed-by: Z
BREAKING CHANGE: /v1/certificates now requires the X-Api-Key header
```

Marca breaking change (con `!`, con footer, o ambos) siempre que: cambie o se
elimine un contrato de API, haga falta correr una migración antes del deploy,
cambie el formato de un archivo de configuración, o se elimine una variable de
entorno. El texto del footer describe **qué debe hacer quien despliega**, no qué
cambió internamente.

Si usas `!` sin footer, la descripción tiene que explicar el quiebre por sí sola.

## Revertir

La spec deja el comportamiento abierto. La convención:

```
revert: let us never again speak of the noodle incident

Refs: 676104e, a215868
```

`git revert` genera `Revert "<header original>"`; reescríbelo al formato de arriba
si el repo valida con commitlint.
