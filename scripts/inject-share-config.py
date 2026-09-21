#!/usr/bin/env python3
"""Inject the shared-training Redis config into the built page.

Run as part of `npm run publish`; never commit the result of the injection
into examples/dino/viewer/index.html.

WHY THE TOKEN IS IN THE PAGE AT ALL: the page must WRITE to publish a better
model, and Upstash has no per-key scoped tokens (verified: GET /v2/redis/acl
and /v2/redis/tokens both 404). The read-only token cannot run SET or EVAL
(verified: 403 NOPERM). So a write-capable token has to ship.

The blast radius is bounded deliberately:
  * `dino` is a dedicated single-purpose database on its own endpoint, so the
    token reaches nothing else on the account (the analytics and painhunt
    databases have different endpoints and tokens).
  * The Upstash ACCOUNT API key - which could create or delete databases - is
    never read by this script and never reaches the page.
  * Worst case is loss of the shared training blob, which any client with a
    better model immediately republishes, and which the CAS script self-heals
    when it finds a corrupt value.

Credentials are read from the file named by DINO_REDIS_CONFIG, defaulting to
/var/minis/workspace/dino-redis.json, which is gitignored.
"""
import json
import os
import re
import sys

CONFIG = os.environ.get("DINO_REDIS_CONFIG", "/var/minis/workspace/dino-redis.json")
TARGET = "examples/dino/viewer/index.html"

if not os.path.exists(CONFIG):
    sys.exit(
        f"error: {CONFIG} not found.\n"
        "Create the Redis db first:  npm run redis:create\n"
        "Or point DINO_REDIS_CONFIG at an existing credentials file."
    )

cfg = json.load(open(CONFIG))
url = cfg.get("UPSTASH_DINO_REST_URL", "")
token = cfg.get("UPSTASH_DINO_REST_TOKEN", "")
if not url or not token:
    sys.exit("error: config is missing UPSTASH_DINO_REST_URL / UPSTASH_DINO_REST_TOKEN")

html = open(TARGET).read()
tag = '<script>window.__DINO_SHARE={url:"%s",token:"%s"};</script>' % (url, token)

# Replace any previous injection so repeated runs cannot stack tags.
html = re.sub(r"<script>window\.__DINO_SHARE=.*?</script>\n?", "", html, flags=re.S)
if "</head>" not in html:
    sys.exit("error: no </head> in " + TARGET)
html = html.replace("</head>", tag + "\n</head>", 1)
open(TARGET, "w").write(html)
print(f"injected share config -> {TARGET}")
