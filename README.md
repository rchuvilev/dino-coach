# dino-coach

A self-improving agent that plays the real Chrome Dino game in your browser:
perception heads → priority rules → decisions tuned by evolution and two
online learners.

**▶ Live: https://rchuvilev.github.io/dino-coach/**

The page is a single self-contained HTML file. No server, no build step to
view it, no network calls at runtime.

## What it actually does

- **Perception** reads the live game state (gap to the next obstacle, its
  width and height, speed, whether the dino is airborne).
- **Rules** decide jump / duck / fastdrop / run from an evolved genome. The
  takeoff window is speed-linear, so one genome behaves differently early and
  late in a run.
- **Learning happens at two timescales.** During a run, every resolved jump is
  added to a KNN (usable immediately, no retraining) and buffered for an MLP.
  Between episodes the MLP flushes those outcomes into its weights —
  deliberately *before* the next episode starts, so a run never begins on the
  same weights that just failed.
- **Evolution** promotes a challenger only when it beats what the champion
  *typically* scores, not its luckiest run, and re-measures the champion every
  third attempt so one fortunate score cannot become a permanent, unbeatable
  bar.

Speed and distance are coupled rather than fed in as independent numbers: the
MLP sees `ttc = gap / speed`, the buckets split slow/mid/fast, and the genome's
window is a function of speed.

## The overlay

Toggle it to see what the agent sees: bounding boxes for the dino and each
obstacle, a vertical line at the takeoff point, the predicted landing line, the
trajectory arc integrated with the game's own gravity, and a red dot at the
collision point when a run ends.

Overlay geometry is measured against real sprite pixels, not assumed — one game
unit is one CSS pixel, sprites render 2px below their reported `yPos`, and the
visual ground is at y=131 while the collision-space `groundYPos` reports 93.

## Build

Only needed if you change the code. The published page lives in `docs/` and is
committed, so there is no CI and no GitHub Actions workflow.

```sh
# one-time: fetch the third-party game and apply the harness patch
cd examples/dino/viewer
curl -o game.js 'https://chromedino.com/js/game.js?v=26'
cp game.js game.orig.js
patch game.js < game.patch
cd ../../..

npm run build:pages   # writes docs/index.html
npm test              # 120 tests
```

`game.js` is deliberately **not vendored**: upstream ships it without a licence
header, so redistributing it here would be unattributed. The three-hunk patch
in `game.patch` is ours and only makes the game's clock steppable.

TensorFlow.js and the KNN classifier *are* vendored (Apache-2.0, headers
intact) so the build needs no network.

## Publishing

GitHub Pages serves `main` → `/docs`. Run the build, commit `docs/index.html`,
push. The published artifact is a normal reviewable diff rather than the output
of an opaque runner.

## Layout

```
src/                     tick-loop framework (perception, rules, evolve, persist)
test/                    120 tests
examples/dino/viewer/    the dino app: game patch, agent, overlay, build
docs/index.html          the published single-file build
```
