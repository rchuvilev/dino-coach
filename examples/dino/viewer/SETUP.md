# Dino viewer — real game, local host, manual clock

The game itself (`game.js`, ~73KB) is **not vendored**. Fetch it and apply
the harness patch:

```sh
curl -o game.js "https://chromedino.com/js/game.js?v=26"
cp game.js game.orig.js
patch game.js < game.patch
```

## What the patch does (3 hunks, nothing touching game logic)

1. `getTimeStamp()` consults `window.__CLOCK` — every value the game derives
   (deltaTime, distanceRan, obstacle x, jump arc) flows from this one
   function, so owning it makes the real game deterministic and steppable.
2. The single `requestAnimationFrame` call routes through `__CLOCK.queue`.
3. `loadSounds()` returns early when the audio `<template>` is absent. The
   original page ships base64 audio; without the guard a missing template
   threw from inside `onKeyDown`, so the game could not be STARTED at all.

## Load order matters

`clock.js` **must** load before `game.js`. Runner queues its first frame
during boot, and a callback registered against the real rAF is invisible to
our queue — symptom: `raqId=1` while `CLOCK.cbs` is empty, so stepping
advances nothing and every policy scores 0.

## Standalone build

```sh
sh build.sh      # -> dino-standalone.html (~108 KB)
```

One self-contained file: the patched game, the clock, the bundled evolution
code, and both sprites as data URIs. Runs from `file://` in any browser with
no server and no network.

Inlining is not cosmetic. ES modules and `<img src>` are blocked by the
`file://` origin in several browsers, and `fetch()` is blocked on the
`minis://` scheme (measured) — a single file with data URIs sidesteps all of
it. Script order is preserved: clock before game, bundle last.

## State

- **Save / Load** — an explicit localStorage slot, separate from the rolling
  autosave, so an experiment cannot overwrite a checkpoint.
- **Export / Import** — the same state as a `.json` file. Imports run through
  the same migration as stored state, so older exports still load.

## TensorFlow.js

The KNN tier uses `@tensorflow-models/knn-classifier`. Fetch both libraries
into this directory before building:

```sh
npm install @tensorflow/tfjs @tensorflow-models/knn-classifier
cp node_modules/@tensorflow/tfjs/dist/tf.min.js .
cp node_modules/@tensorflow-models/knn-classifier/dist/knn-classifier.min.js .
```

`build.sh` inlines both, so the standalone page stays offline-capable at the
cost of size: **1599 KB**, up from 140 KB.

**The backend is forced to CPU.** Measured in-browser with 300 examples of 4
features: `cpu 0.74 ms/predict` against `webgl 12.14 ms/predict`. WebGL loses
because kernel-launch overhead dominates at this tensor size, and 12 ms does
not fit a 16.7 ms frame that also has to run the game.
