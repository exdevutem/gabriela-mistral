# commitlint

Si te piden configurarlo, la base razonable:

```js
// commitlint.config.js
export default {
  extends: ['@commitlint/config-conventional'],
  rules: {
    'header-max-length': [2, 'always', 72],
    'scope-enum': [2, 'always', ['api', 'auth', 'ui', 'db', 'ci', 'deps']],
  },
};
```

`@commitlint/config-conventional` ya trae el `type-enum` de Angular, `subject-case`
y la exigencia de cuerpo separado por línea en blanco.
