# Ejemplos completos

**Feature con ticket (inglés)**

```
feat(auth): add refresh token rotation

Refresh tokens are now single-use and rotated on every exchange, so a
leaked token stops being valid as soon as the legitimate client refreshes.

Refs: #214
```

**Bug fix (español)**

```
fix(pagos): corrige cálculo de dv en creación de proveedores

Se ajusta el algoritmo de módulo 11 para RUT con dv=K.
```

**Breaking change con `!` y footer**

```
feat(api)!: require API key on all v1 endpoints

BREAKING CHANGE: every /v1 request now needs the X-Api-Key header.
Clients without a key get 401 instead of being served anonymously.
```

**Breaking change solo con footer**

```
chore(deps): drop support for Node 16

BREAKING CHANGE: minimum supported runtime is now Node 18.
```

**Sin cuerpo, cuando el header basta**

```
refactor(api): extract RUT validation into a helper
```

```
ci(deploy): add automatic rollback on failed healthcheck
```
