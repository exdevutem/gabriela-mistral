# Scopes

Sustantivo, minúsculas, opcional. Sácalos de los que ya existen en el historial:

```bash
git log --format=%s -200 | sed -n 's/^[a-z]*(\([^)]*\)).*/\1/p' | sort | uniq -c | sort -rn
```

Si el cambio es transversal, **omite el scope** antes que inventar uno:
`chore: remove unused imports`.
