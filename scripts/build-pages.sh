#!/bin/sh
# Build the GitHub Pages payload into docs/ — run LOCALLY, commit the result.
#
# No CI, no Actions, no gh-pages branch: GitHub Pages is configured to serve
# main:/docs, so whatever is committed there is what is live. That keeps the
# published artifact reviewable in a normal diff instead of being produced by
# an opaque runner.
#
# The page is a SINGLE self-contained HTML file. ES modules and <img src> are
# blocked by the file:// origin in several browsers, so everything (game,
# clock, evolution code, tfjs, sprites as data URIs) is inlined. That also
# means the published page has zero runtime dependencies and no network calls.
set -e
cd "$(dirname "$0")/.."

VIEWER=examples/dino/viewer

if [ ! -f "$VIEWER/game.js" ]; then
  echo "error: $VIEWER/game.js missing." >&2
  echo "It is third-party and deliberately not vendored. Fetch and patch it:" >&2
  echo "  cd $VIEWER && curl -o game.js 'https://chromedino.com/js/game.js?v=26' \\" >&2
  echo "    && cp game.js game.orig.js && patch game.js < game.patch" >&2
  exit 1
fi

sh "$VIEWER/build.sh"

mkdir -p docs
cp "$VIEWER/dino-standalone.html" docs/index.html

# .nojekyll stops Pages running the output through Jekyll, which would
# otherwise skip files and directories beginning with an underscore.
touch docs/.nojekyll

echo "docs/index.html: $(( $(wc -c < docs/index.html) / 1024 )) KB"
