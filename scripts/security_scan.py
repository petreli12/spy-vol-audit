#!/usr/bin/env python3
"""Pre-commit security gate: fail on key-shaped literals in the public tree."""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".venv", ".git", "data", "publishing", "__pycache__"}
SKIP_FILES = {"security_scan.py"}

# Load optional local secret literals from parent OptionsEdge .env for cross-check only.
OPTIONSEDGE_ENV = REPO.parent / "OptionsEdge" / ".env"

GENERIC_PATTERNS = [
    (
        re.compile(
            r"(?i)^[A-Z0-9_]*(API_KEY|SECRET|TOKEN|PASSWORD)[A-Z0-9_]*\s*=\s*['\"]?[A-Za-z0-9_\-]{16,}"
        ),
        "generic credential assignment",
    ),
    (re.compile(r"ghp_[A-Za-z0-9]{20,}"), "GitHub PAT"),
    (re.compile(r"gho_[A-Za-z0-9]{20,}"), "GitHub OAuth token"),
    (re.compile(r"\d{8,10}:[A-Za-z0-9_-]{30,}"), "Telegram bot token shape"),
    (re.compile(r"postgresql\+psycopg2://[^\s\"']+"), "database URL"),
]


def _iter_files() -> list[Path]:
    out: list[Path] = []
    for p in REPO.rglob("*"):
        if not p.is_file():
            continue
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        if p.name in SKIP_FILES:
            continue
        if p.suffix in {".png", ".jpg", ".parquet", ".pyc"}:
            continue
        out.append(p)
    return out


def _literal_secrets() -> list[str]:
    literals: list[str] = []
    if OPTIONSEDGE_ENV.exists():
        for line in OPTIONSEDGE_ENV.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            _, _, val = line.partition("=")
            val = val.strip().strip('"').strip("'")
            if len(val) >= 8 and val not in ("", "true", "false"):
                literals.append(val)
    return literals


def main() -> int:
    findings: list[str] = []

    if (REPO / ".env").exists():
        findings.append(".env exists in repo tree (must not be committed)")

    example = REPO / ".env.example"
    if example.exists():
        for i, line in enumerate(example.read_text().splitlines(), 1):
            if "=" not in line or line.strip().startswith("#"):
                continue
            _, _, val = line.partition("=")
            val = val.strip()
            if val and not val.endswith("=") and len(val) >= 12:
                findings.append(f".env.example:{i} non-placeholder value")

    literals = _literal_secrets()
    for path in _iter_files():
        try:
            text = path.read_text(errors="ignore")
        except Exception:
            continue
        rel = path.relative_to(REPO)
        for lit in literals:
            if lit in text:
                findings.append(f"{rel}: contains known secret literal ({lit[:4]}…)")
        for rx, label in GENERIC_PATTERNS:
            for m in rx.finditer(text):
                snippet = m.group(0)[:40]
                findings.append(f"{rel}: {label} ({snippet}…)")

    print("SECURITY SCAN REPORT")
    print(f"  Root: {REPO}")
    print(f"  Files scanned: {len(_iter_files())}")
    print(f"  .env present: {(REPO / '.env').exists()}")
    if findings:
        print(f"  FINDINGS: {len(findings)}")
        for f in findings:
            print(f"    - {f}")
        return 1
    print("  FINDINGS: 0 (clean)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
