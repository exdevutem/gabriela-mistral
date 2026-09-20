#!/usr/bin/env python3
"""Valida mensajes contra Conventional Commits v1.0.0 (EN/ES).

Uso:
    python3 validate_commit.py "feat(auth): add refresh token rotation"
    git log --format=%s -20 | python3 validate_commit.py
    python3 validate_commit.py --full .git/COMMIT_EDITMSG
    python3 validate_commit.py --branch feat/auth-214-refresh-token-rotation
    git branch --format='%(refname:short)' | python3 validate_commit.py --branch

Sin --full valida una línea de header por línea de entrada.
Con --full valida el mensaje completo: header, línea en blanco, cuerpo y footers.
Con --branch valida nombres de rama: <type>/[<scope>-][<ticket>-]<slug>.

FALLA = rompe la spec o la convención dura. AVISO = revisable.
Código de salida 1 si algo FALLA.
"""

import re
import sys

# feat/fix los exige la spec; el resto es la convención de Angular que asumen
# commitlint, semantic-release y release-please.
TYPES = [
    "feat", "fix", "docs", "style", "refactor", "perf",
    "test", "build", "ci", "chore", "revert",
]

MAX_HEADER = 72        # convención de git: aviso
HARD_MAX_HEADER = 100  # default de commitlint: falla

# Emoji al estilo gitmoji: rangos de pictogramas + selector de variación y ZWJ.
EMOJI_RE = re.compile(
    "[\U0001F000-\U0001FAFF\u2190-\u21FF\u2300-\u23FF\u2460-\u24FF"
    "\u25A0-\u27BF\u2B00-\u2BFF\uFE0F\u200D\u20E3]"
)
LEADING_EMOJI_RE = re.compile(f"^(?P<emoji>{EMOJI_RE.pattern}+)(?P<sep>\\s*)")
GITMOJI_CODE_RE = re.compile(r"^:[a-z0-9_+-]+:")

HEADER_RE = re.compile(
    r"^(?P<type>[a-zA-Z]+)(?:\((?P<scope>[^)]*)\))?(?P<bang>!)?: (?P<desc>.+)$"
)
# Token de footer: 'Refs: x', 'Closes #1', 'BREAKING CHANGE: x'.
FOOTER_RE = re.compile(r"^(BREAKING[ -]CHANGE|[A-Za-z][A-Za-z0-9-]*)(: | #)(.+)$")

# Terminaciones que delatan que la descripción no está en imperativo.
NOT_IMPERATIVE_ES = re.compile(
    r"^\w+(ado|ados|ada|adas|ido|idos|ida|idas|ando|endo|iendo|ción|ciones|miento|mientos)$",
    re.IGNORECASE,
)
# En inglés: pasado (-ed), gerundio (-ing) y 3a persona (-s salvo -ss/-us).
NOT_IMPERATIVE_EN = re.compile(r"^\w+(ed|ing)$|^\w+[^su]s$", re.IGNORECASE)
# Falsos positivos frecuentes del patrón de arriba.
EN_OK = {"add", "read", "feed", "seed", "speed", "embed", "bring", "string",
         "build", "hold", "bind", "find", "send", "spend", "extend", "append",
         "need", "fix", "focus", "bump", "drop", "ring", "sync"}
ES_HINT = re.compile(
    r"\b(el|la|los|las|un|una|de|del|en|con|para|por|se|que|y|no|al)\b", re.IGNORECASE
)


def detect_lang(text: str) -> str:
    """es si hay señales claras de español, en si no. Solo para elegir la
    heurística de imperativo — no es un juicio sobre el idioma correcto."""
    if re.search(r"[áéíóúñ¿¡]", text, re.IGNORECASE):
        return "es"
    return "es" if len(ES_HINT.findall(text)) >= 2 else "en"


def check_header(header: str) -> list[str]:
    issues: list[str] = []
    header = header.rstrip("\n")

    if not header.strip():
        return ["header vacío"]

    n = len(header)
    if n > HARD_MAX_HEADER:
        issues.append(f"header de {n} caracteres (máximo duro {HARD_MAX_HEADER})")
    elif n > MAX_HEADER:
        issues.append(f"aviso: header de {n} caracteres (recomendado ≤{MAX_HEADER})")

    lead = LEADING_EMOJI_RE.match(header) or GITMOJI_CODE_RE.match(header)
    if lead:
        issues.append(
            "el emoji va después de ': ', no antes del type: commitlint y "
            "semantic-release no parsean este header"
        )
        header = header[lead.end():].lstrip()

    m = HEADER_RE.match(header)
    if not m:
        if re.match(r"^[a-zA-Z]+(\([^)]*\))?!?:\S", header):
            issues.append("falta el espacio después de los dos puntos: '<type>: <desc>'")
        else:
            issues.append(
                "no calza con '<type>[(scope)][!]: <description>' "
                "(revisa los dos puntos y el espacio)"
            )
        return issues

    ctype, scope, bang, desc = (
        m.group("type"), m.group("scope"), m.group("bang"), m.group("desc")
    )

    if ctype not in TYPES:
        if ctype.lower() in TYPES:
            issues.append(f"type '{ctype}' debe ir en minúsculas")
        else:
            issues.append(
                f"aviso: type '{ctype}' fuera de la convención Angular "
                f"({', '.join(TYPES)}); válido solo si el repo lo declara en su type-enum"
            )

    if scope is not None:
        if scope == "":
            issues.append("scope vacío: omite los paréntesis si no hay scope")
        else:
            if scope != scope.lower():
                issues.append(f"aviso: scope '{scope}' se suele escribir en minúsculas")
            if not re.fullmatch(r"[a-z0-9][a-z0-9._/-]*", scope.lower()):
                issues.append(f"scope '{scope}' con caracteres no esperados")

    emoji = LEADING_EMOJI_RE.match(desc)
    if emoji:
        if not emoji.group("sep"):
            issues.append("falta el espacio entre el emoji y la descripción")
        if len(EMOJI_RE.findall(emoji.group("emoji"))) > 2:  # 2 = pictograma + \uFE0F
            issues.append("aviso: usa un solo emoji por commit")
        desc = desc[emoji.end():]
        if not desc.strip():
            issues.append("el emoji no reemplaza la descripción")
            return issues
    elif GITMOJI_CODE_RE.match(desc):
        issues.append("usa el emoji literal (✨), no su código (:sparkles:)")
        desc = GITMOJI_CODE_RE.sub("", desc).lstrip()
    elif EMOJI_RE.search(desc):
        issues.append("aviso: el emoji va al inicio de la descripción, no en medio")

    if bang and emoji and "\U0001F4A5" not in emoji.group("emoji"):
        issues.append("aviso: en un commit breaking se suele usar 💥")

    if desc.endswith("."):
        issues.append("la descripción no lleva punto final")

    first = desc.split()[0]
    if first[:1].isupper() and first.upper() != first:
        issues.append(f"aviso: la descripción parte en mayúscula ('{first}')")

    lang = detect_lang(desc)
    word = first.lower().strip(",;:")
    bad = (
        NOT_IMPERATIVE_ES.match(word)
        if lang == "es"
        else (NOT_IMPERATIVE_EN.match(word) and word not in EN_OK)
    )
    if bad:
        hint = "agrega, corrige, actualiza" if lang == "es" else "add, fix, update"
        issues.append(f"'{first}' no parece imperativo (usa {hint}...)")

    if len(desc.split()) < 2:
        issues.append("descripción demasiado corta para ser descriptiva")

    if bang:
        issues.append(
            "aviso: marcado como breaking (!); agrega el footer 'BREAKING CHANGE: ...' "
            "si el quiebre no se explica solo"
        )

    return issues


MAX_BRANCH = 50
TICKET_RE = re.compile(r"[A-Z][A-Z0-9]+-[0-9]+")
# Ramas que no siguen el formato: principales, releases y las de bots.
SPECIAL_BRANCH_RE = re.compile(
    r"^(main|master|develop|release/\d+\.\d+\.\d+.*|hotfix/[a-z0-9-]+"
    r"|dependabot/.+|renovate/.+|release-please--.+|changeset-release/.+)$"
)


def check_branch(name: str) -> list[str]:
    name = name.strip()
    if not name:
        return ["nombre de rama vacío"]
    if SPECIAL_BRANCH_RE.match(name):
        return []

    issues: list[str] = []
    if len(name) > MAX_BRANCH:
        issues.append(f"rama de {len(name)} caracteres (máximo {MAX_BRANCH})")
    if "/" not in name:
        return issues + ["falta el prefijo '<type>/'"]

    btype, rest = name.split("/", 1)
    if btype not in TYPES:
        if btype.lower() in TYPES:
            issues.append(f"type '{btype}' debe ir en minúsculas")
        else:
            issues.append(f"type '{btype}' no es de Conventional Commits ({', '.join(TYPES)})")
    if "/" in rest:
        issues.append("usa un solo '/', entre el type y el resto; lo demás separa con '-'")
    if "!" in name:
        issues.append("sin '!' en la rama: el breaking change se marca en el commit")
    if not rest.isascii():
        issues.append("solo ASCII: sin tildes, ñ ni emojis")
    odd = sorted(set(re.findall(r"[^A-Za-z0-9/!\x80-\U0010ffff-]", rest)))  # no-ASCII ya se reportó
    if odd:
        issues.append(f"caracteres no permitidos {''.join(odd)!r}: usa solo a-z, 0-9 y '-'")

    plain = TICKET_RE.sub("", rest)
    if any(c.isupper() for c in plain):
        issues.append("minúsculas en todo salvo el ID del ticket (PROJ-214)")
    if rest.startswith("-") or rest.endswith("-") or "--" in rest:
        issues.append("guiones al inicio, al final o dobles")
    ticket = TICKET_RE.search(rest)
    if not re.search(r"[a-z]", rest[ticket.end():] if ticket else rest):
        issues.append("falta el slug que describe el cambio")

    return issues


def check_full(text: str) -> list[str]:
    lines = text.rstrip("\n").split("\n")
    issues = check_header(lines[0])

    if len(lines) == 1:
        return issues
    if lines[1].strip():
        issues.append("falta la línea en blanco entre el header y el cuerpo")

    rest = lines[2:]
    # Footers = último bloque separado por línea en blanco, si todas sus líneas
    # (o la primera de cada footer) calzan con el formato de trailer.
    blocks, current = [], []
    for ln in rest:
        if ln.strip():
            current.append(ln)
        elif current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)

    for block in blocks:
        for ln in block:
            if re.match(r"^breaking[ -]change", ln, re.IGNORECASE) and not re.match(
                r"^BREAKING[ -]CHANGE(: | #)", ln
            ):
                issues.append(f"'BREAKING CHANGE' debe ir en mayúsculas: {ln[:40]!r}")

    if blocks:
        last = blocks[-1]
        if FOOTER_RE.match(last[0]):
            for ln in last:
                if ln.startswith((" ", "\t")):
                    continue  # continuación de un footer multilínea
                if not FOOTER_RE.match(ln):
                    issues.append(
                        f"aviso: línea del bloque de footers sin formato 'Token: valor': {ln[:40]!r}"
                    )

    for ln in lines[2:]:
        if len(ln) > 100:
            issues.append(f"aviso: línea de cuerpo de {len(ln)} caracteres (envuelve a ~72)")
            break

    return issues


def main() -> int:
    argv = sys.argv[1:]
    full = "--full" in argv
    branch = "--branch" in argv
    args = [a for a in argv if a not in ("--full", "--branch")]

    if branch:
        names = args or sys.stdin.read().split()
        units = [(n, check_branch(n)) for n in names]
        return report(units)

    if args:
        chunks = []
        for a in args:
            try:
                with open(a, encoding="utf-8") as fh:
                    chunks.append(fh.read())
            except OSError:
                chunks.append(a)
        text = "\n".join(chunks)
    else:
        text = sys.stdin.read()

    if full:
        # Ignora comentarios de git (COMMIT_EDITMSG).
        text = "\n".join(ln for ln in text.split("\n") if not ln.startswith("#"))
        units = [(text.split("\n")[0], check_full(text))] if text.strip() else []
    else:
        units = [(ln, check_header(ln)) for ln in text.splitlines() if ln.strip()]

    return report(units)


def report(units: list[tuple[str, list[str]]]) -> int:
    failed = False
    for header, issues in units:
        hard = [p for p in issues if not p.startswith("aviso:")]
        if not issues:
            print(f"OK    {header}")
            continue
        print(f"{'FALLA' if hard else 'AVISO'} {header}")
        for p in issues:
            print(f"        - {p}")
        failed = failed or bool(hard)

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
