import subprocess, sys, pathlib
V = [sys.executable, str(pathlib.Path(__file__).with_name("validate_commit.py"))]
def run(args, stdin=None):
    r = subprocess.run(V + args, capture_output=True, text=True, input=stdin)
    return r.stdout, r.returncode

ok = [
    "feat(auth): add refresh token rotation",
    "fix: correct checksum calculation",
    "fix(pagos): corrige cálculo de dv en creación de proveedores",
    "refactor(api): extract RUT validation into a helper",
    "chore(deps): drop support for Node 16",
    "docs: document deployment steps",
]
for h in ok:
    out, rc = run([h])
    assert rc == 0 and out.startswith("OK"), (h, out)

fail = [
    ("feat(auth):add token", "espacio"),          # sin espacio tras ':'
    ("feat added payments endpoint", "no calza"),  # sin ':'
    ("feat(api): added payments endpoint", "imperativo"),
    ("fix: arreglando el cálculo de dv", "imperativo"),
    ("Feat(api): add endpoint", "minúsculas"),
    ("feat(api): add endpoint.", "punto final"),
    ("feat(): add endpoint", "scope vacío"),
    ("fix: " + "x"*120, "máximo duro"),
]
for h, needle in fail:
    out, rc = run([h])
    assert rc == 1 and needle in out, (h, out)

# avisos no fallan
for h in ["wip(api): add thing", "fix: correct " + "a"*70]:
    out, rc = run([h])
    assert rc == 0 and out.startswith("AVISO"), (h, out)

# --full
good = "feat(auth): add rotation\n\nBody explaining why.\n\nRefs: #214\n"
out, rc = run(["--full"], stdin=good); assert rc == 0, out
noblank = "feat(auth): add rotation\nBody with no blank line\n"
out, rc = run(["--full"], stdin=noblank); assert rc == 1 and "línea en blanco" in out, out
lower = "feat(api)!: require key\n\nbreaking change: needs X-Api-Key\n"
out, rc = run(["--full"], stdin=lower); assert rc == 1 and "mayúsculas" in out, out
upper = "feat(api)!: require key\n\nBREAKING CHANGE: needs X-Api-Key\n"
out, rc = run(["--full"], stdin=upper); assert rc == 0, out

# stdin multilínea
out, rc = run([], stdin="feat: add a\nfix: correct b\n")
assert rc == 0 and out.count("OK") == 2, out
# --- emojis ---
for h in [
    "feat(auth): \u2728 add refresh token rotation",
    "fix(api): \U0001F512 reject tokens signed with the none algorithm",
    "feat(reportes): \U0001F5C3\ufe0f agrega tabla de agregados mensuales",
    "refactor: \u267b\ufe0f extract RUT validation into a helper",
]:
    out, rc = run([h])
    assert rc == 0 and out.startswith("OK"), (h, out)

emoji_fail = [
    ("\u2728 feat(auth): add token rotation", "antes del type"),
    (":sparkles: feat(auth): add token rotation", "antes del type"),
    ("feat(auth): :sparkles: add token rotation", "emoji literal"),
    ("feat(auth): \u2728add token rotation", "falta el espacio"),
    ("feat(auth): \u2728", "no reemplaza la descripci\u00f3n"),
]
for h, needle in emoji_fail:
    out, rc = run([h])
    assert rc == 1 and needle in out, (h, out)

# el ! ya emite su propio aviso, pero con 💥 no agrega el de emoji
out, rc = run(["feat(api)!: \U0001F4A5 require API key on all v1 endpoints"])
assert rc == 0 and "\U0001F4A5" not in out.split("breaking")[-1], out

# avisos de emoji: no fallan
for h, needle in [
    ("feat(auth): \u2728\U0001F389\U0001F680 add token rotation", "un solo emoji"),
    ("feat(auth): add token rotation \u2728", "al inicio"),
    ("feat(api)!: \u2728 require API key on all endpoints", "\U0001F4A5"),
]:
    out, rc = run([h])
    assert rc == 0 and needle in out, (h, out)

# el emoji no rompe la deteccion de imperativo ni de punto final
out, rc = run(["feat(api): \u2728 added payments endpoint"])
assert rc == 1 and "imperativo" in out, out

# --- ramas ---
for b in [
    "feat/auth-214-refresh-token-rotation",
    "feat/tramites-VRAC-214-filtro-carrera",
    "fix/conta-312-dv-k-proveedor",
    "build/bump-axios-1-7-9",
    "docs/deployment-steps",
    "main", "develop", "release/2.4.0", "dependabot/npm_and_yarn/axios-1.7.9",
]:
    out, rc = run(["--branch", b])
    assert rc == 0 and out.startswith("OK"), (b, out)

for b, needle in [
    ("feature/login-form", "no es de Conventional"),
    ("Fix/conta-dv-k", "minúsculas"),
    ("fix/conta_dv", "no permitidos"),
    ("fix/foo~bar", "no permitidos"),
    ("feat/tramites/filtro", "un solo '/'"),
    ("feat/api!-require-key", "'!'"),
    ("fix/corrige-cálculo-dv", "ASCII"),
    ("fix/Conta-dv", "minúsculas"),
    ("wip", "prefijo"),
    ("feat/VRAC-214", "slug"),
    ("feat/auth--rotation", "guiones"),
    ("feat/" + "a" * 50, "caracteres (máximo"),
]:
    out, rc = run(["--branch", b])
    assert rc == 1 and needle in out, (b, out)

out, rc = run(["--branch"], stdin="main\nfeat/auth-login-form\njuan/wip\n")
assert rc == 1 and out.count("OK") == 2 and "juan/wip" in out, out

print("all checks passed")
