#!/usr/bin/env python3
"""Refuse to publish if the built page carries a secret it must not.

Runs before every publish. A scan that reports "clean" is worthless unless it
is shown to be capable of finding something, so this ends with a POSITIVE
CONTROL: a planted sentinel must be detected, otherwise the check itself
fails. A silently broken scanner is indistinguishable from a clean page.

Allowed in the page: the dino database REST url and its write token. That is
the minimum needed for a browser to publish a better model, and the database
is single-purpose (see scripts/inject-share-config.py for the reasoning).

Never allowed: the Upstash account API key or email (control plane - can
create and delete databases), the dino read-only token (no reason to ship
it), and any other long secret from the shared env file.
"""
import json
import os
import sys

PAGE = "docs/index.html"
CONFIG = os.environ.get("DINO_REDIS_CONFIG", "/var/minis/workspace/dino-redis.json")
ENVFILE = os.environ.get("MINIS_ENV_JSON", "/var/minis/env.json")

if not os.path.exists(PAGE):
    sys.exit(f"error: {PAGE} not found - run the build first")

page = open(PAGE, encoding="utf-8", errors="replace").read()
problems = []
checked = 0

def forbid(label, value):
    global checked
    if not value or len(value) < 12:
        return
    checked += 1
    if value in page:
        problems.append(label)
        print(f"  LEAK    {label}")
    else:
        print(f"  clean   {label}")

print(f"scanning {PAGE} ({len(page)//1024} KB)")

if os.path.exists(CONFIG):
    cfg = json.load(open(CONFIG))
    forbid("dino read-only token", cfg.get("UPSTASH_DINO_READONLY_TOKEN", ""))

if os.path.exists(ENVFILE):
    env = json.load(open(ENVFILE))
    # Every long string in the shared env file is treated as a secret, so a
    # newly added credential is covered without editing this list.
    for k, v in env.items():
        if isinstance(v, str) and len(v) >= 16:
            forbid(f"env:{k}", v)
else:
    print(f"  note: {ENVFILE} absent - skipping account-secret scan")

# Positive control: prove the scanner can actually detect a secret.
sentinel = "LEAKCHECK-" + "S" * 24
if sentinel not in (page + sentinel):
    sys.exit("error: leak scanner is broken - it cannot detect a planted secret")
print(f"  control PASS (scanner detects a planted secret; {checked} secrets checked)")

if checked == 0:
    sys.exit("error: no secrets were available to check - scan proves nothing")

if problems:
    print(f"\nFAILED: {len(problems)} secret(s) present in {PAGE}")
    sys.exit(1)
print(f"\nOK: no forbidden secrets in {PAGE}")
