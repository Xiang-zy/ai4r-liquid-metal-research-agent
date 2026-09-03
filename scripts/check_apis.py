#!/usr/bin/env python3
"""Minimal API compatibility check that never prints credentials or content."""

import argparse
import json
import os
from pathlib import Path
import re
import sys
import urllib.error
import urllib.request


def load_env(path):
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"\''))


def safe_error(error):
    message = str(error)
    message = re.sub(r"sk-[A-Za-z0-9_-]+", "<secret>", message)
    message = re.sub(r"(?i)(token|key|authorization)[^,;}\n]{0,120}", r"\1=<hidden>", message)
    return message[:300]


def request_json(request, timeout):
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def minimax_check():
    key = os.environ.get("MINIMAX_API_KEY")
    if not key:
        print("MiniMax: SKIP (MINIMAX_API_KEY is missing)")
        return False

    chat_url = os.environ.get(
        "MINIMAX_BASE_URL",
        "https://api.minimaxi.com/v1/chat/completions",
    ).rstrip("/")
    models_url = chat_url.rsplit("/chat/completions", 1)[0] + "/models"
    request = urllib.request.Request(
        models_url,
        headers={"Authorization": f"Bearer {key}"},
    )
    data = request_json(request, 30)
    model_ids = [item.get("id") for item in data.get("data", [])]
    selected = os.environ.get("LLM_MODEL", "MiniMax-M3")
    if selected not in model_ids:
        print(f"MiniMax: FAIL (configured model {selected!r} is unavailable)")
        return False
    print(f"MiniMax: OK (auth, endpoint, model={selected})")
    return True


def sciverse_check():
    key = os.environ.get("SCIVERSE_API_KEY")
    if not key:
        print("Sciverse: SKIP (SCIVERSE_API_KEY is missing)")
        return False

    base_url = os.environ.get("SCIVERSE_BASE_URL", "https://api.sciverse.space").rstrip("/")
    payload = json.dumps({"query": "liquid metal", "top_k": 1}).encode("utf-8")
    request = urllib.request.Request(
        base_url + "/agentic-search",
        data=payload,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    data = request_json(request, 45)
    hits = data.get("hits")
    if not isinstance(hits, list):
        print("Sciverse: FAIL (/agentic-search response has no hits list)")
        return False

    doc_id = next((hit.get("doc_id") for hit in hits if hit.get("doc_id")), None)
    if doc_id:
        from urllib.parse import urlencode

        content_request = urllib.request.Request(
            base_url + "/content?" + urlencode({"doc_id": doc_id, "offset": 0, "limit": 1}),
            headers={"Authorization": f"Bearer {key}"},
        )
        content = request_json(content_request, 30)
        if "text" not in content and "content" not in content:
            print("Sciverse: FAIL (/content response has no text field)")
            return False
        print("Sciverse: OK (auth, search, content schema)")
    else:
        print("Sciverse: OK (auth and search; no full-text doc returned for content check)")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, default=Path(__file__).resolve().parents[1] / ".env")
    args = parser.parse_args()
    load_env(args.env_file)

    checks = []
    for name, function in (("MiniMax", minimax_check), ("Sciverse", sciverse_check)):
        try:
            checks.append(function())
        except urllib.error.HTTPError as error:
            print(f"{name}: FAIL (HTTP {error.code})")
            checks.append(False)
        except Exception as error:
            print(f"{name}: FAIL ({safe_error(error)})")
            checks.append(False)
    return 0 if all(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
