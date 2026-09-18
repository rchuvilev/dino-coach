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
