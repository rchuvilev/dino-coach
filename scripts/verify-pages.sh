#!/bin/sh
# Verify the LIVE page is actually the artifact we just built.
#
# An HTTP 200 proves nothing on its own: a host with an SPA fallback returns
# 200 with the wrong page for every path. So this compares the served bytes by
# md5 against docs/index.html and requires a real 404 on a nonsense path as a
# negative control. A cache-buster is used because a fresh deploy can serve
# stale bytes for several minutes.
set -e
cd "$(dirname "$0")/.."

BASE=${PAGES_BASE:-https://rchuvilev.github.io/dino-coach}
LOCAL=docs/index.html
BUST=$(date +%s)

[ -f "$LOCAL" ] || { echo "error: $LOCAL missing - build first" >&2; exit 1; }

tmp=$(mktemp)
code=$(curl -4 -s -o "$tmp" -w '%{http_code}' "$BASE/?bust=$BUST")
echo "GET $BASE/ -> HTTP $code, $(wc -c < "$tmp") bytes"
[ "$code" = "200" ] || { echo "FAILED: expected 200" >&2; rm -f "$tmp"; exit 1; }

served=$(md5sum < "$tmp" | cut -d' ' -f1)
local_md5=$(md5sum < "$LOCAL" | cut -d' ' -f1)
echo "served md5 $served"
echo "local  md5 $local_md5"
rm -f "$tmp"
[ "$served" = "$local_md5" ] || {
  echo "FAILED: served bytes differ from docs/index.html (deploy still propagating?)" >&2
  exit 1
}

# Negative control: a path that does not exist must 404. If it returns 200 the
# host is serving a fallback and the check above proved nothing about routing.
nf=$(curl -4 -s -o /dev/null -w '%{http_code}' "$BASE/not-a-real-path-$BUST/")
echo "negative control /not-a-real-path -> HTTP $nf"
[ "$nf" = "404" ] || { echo "FAILED: expected 404, got $nf (SPA fallback?)" >&2; exit 1; }

echo "OK: live page matches the built artifact"
