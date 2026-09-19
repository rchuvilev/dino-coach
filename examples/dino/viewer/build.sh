#!/bin/sh
# Build a single self-contained HTML file that runs in ANY browser with no
# server, no modules and no network. Everything is inlined: the patched game,
# the clock, the evolution code and the sprites as data URIs.
#
# Why inline rather than ship a folder: ES modules and <img src> are blocked
# by the file:// origin in several browsers, and fetch() is blocked on the
# minis:// scheme (measured). A single file with data URIs sidesteps all of it.
set -e
cd "$(dirname "$0")"

OUT=dino-standalone.html

if [ ! -f game.js ]; then
  echo "game.js missing - fetch and patch it first (see SETUP.md)" >&2
  exit 1
fi

# bundle the two ES modules into one classic script
bun build drive.js --outfile /tmp/bundle.js >/dev/null

python3 - "$OUT" <<'PY'
import base64, re, sys

out = sys.argv[1]
html = open('index.html').read()
game = open('game.js').read()
clock = open('clock.js').read()
bundle = open('/tmp/bundle.js').read()

def data_uri(path):
    with open(path, 'rb') as f:
        return 'data:image/png;base64,' + base64.b64encode(f.read()).decode()

# sprites as data URIs so file:// cannot block them
html = html.replace('src="offline-sprite-1x.png"', f'src="{data_uri("offline-sprite-1x.png")}"')
html = html.replace('src="offline-sprite-2x.png"', f'src="{data_uri("offline-sprite-2x.png")}"')

# inline the scripts, preserving load order: clock BEFORE game (a callback
# registered against the real rAF is invisible to our queue), bundle last.
tfjs = open('tf.min.js').read()
knnlib = open('knn-classifier.min.js').read()
html = html.replace('<script src="tf.min.js"></script>', '<script>\n' + tfjs + '\n</script>')
html = html.replace('<script src="knn-classifier.min.js"></script>', '<script>\n' + knnlib + '\n</script>')
html = re.sub(r'<script src="clock\.js[^"]*"></script>', '<script>\n' + clock + '\n</script>', html)
html = re.sub(r'<script src="game\.js[^"]*"></script>', '<script>\n' + game + '\n</script>', html)
html = re.sub(r'<script type="module" src="drive\.js[^"]*"></script>', '<script>\n' + bundle + '\n</script>', html)

open(out, 'w').write(html)
print(f'{out}: {len(html)//1024} KB')
PY
