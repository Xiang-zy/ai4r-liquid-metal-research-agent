#!/usr/bin/env python3
"""Small pre-commit secret scan with location-only output."""

import argparse
from pathlib import Path
import re
import sys


PATTERNS = {
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "OpenAI-style token": re.compile(r"\bsk-[A-Za-z0-9_-]{16,}"),
    "GitHub token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"),
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "credential assignment": re.compile(
        r"(?i)(?:api[_-]?key|access[_-]?token|secret|password)\s*=\s*['\"](?!<|your|replace|change|\$\{|os\.|None)[^'\"]{8,}['\"]"
    ),
}

TEXT_SUFFIXES = {".py", ".md", ".json", ".html", ".env", ".txt", ".yaml", ".yml", ".toml"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()
    findings = []

    for path in root.rglob("*"):
        if not path.is_file() or ".git" in path.parts or path.name == ".env.example":
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name != ".env":
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for label, pattern in PATTERNS.items():
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                findings.append((path.relative_to(root), line, label))

    if findings:
        for path, line, label in findings:
            print(f"{path}:{line}: possible {label}")
        print(f"FAIL: {len(findings)} possible secret(s) found")
        return 1
    print("OK: no known secret patterns found")
    return 0


if __name__ == "__main__":
    sys.exit(main())
