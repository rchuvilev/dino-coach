/**
 * Chrome Dino adapter — the first REAL target.
 *
 * Why this target: `Runner.instance_` exposes ground-truth state (distance,
 * speed, obstacle positions, crash flag), so a pixel reader can be validated
 * against known-correct values instead of against my own guesses. That is
 * exactly what validateReader() needs and what the synthetic demo could not
 * provide.
 *
 * This file is the BRIDGE ONLY: it defines what to read and what to press,
 * as data. It runs inside the page via browser execute_js; the framework
 * itself stays target-agnostic.
 */

/** Serialisable snapshot of the target, read once per tick. */
export interface DinoState {
  started: boolean;
  crashed: boolean;
  speed: number;
  distance: number;
  tRexY: number;
  jumping: boolean;
  ducking: boolean;
  /** horizontal gap to the nearest obstacle ahead, in canvas px */
  gap: number;
  /** nearest obstacle height, 0 when none */
  obstacleH: number;
  /** nearest obstacle y, used to tell a high bird from a low cactus */
  obstacleY: number;
}

/**
 * The in-page reader, as a string so it can be shipped to execute_js.
 *
 * Kept as one expression returning a flat object: the tick budget is small
 * and a round trip per field would dominate it.
 *
 * 🔴 Obstacles live on `r.horizon.obstacles`, NOT `r.obstacles`. Reading the
 * latter returned an always-empty array, so `gap` was a constant 9999 while
 * the dino crashed into cacti - a reader that looks plausible and returns a
 * uniform value. Caught by a positive control asking whether the array EVER
 * populates (anyObstacle: false while crashed: true). This is precisely the
 * failure validateReader() exists to catch, on the very first real target.
 */
export const READ_STATE_JS = `
(() => {
  const r = window.Runner && window.Runner.instance_;
  if (!r) return null;
  const tx = r.tRex.xPos + 44;
  let gap = 9999, oh = 0, oy = 0;
  for (const o of (r.horizon && r.horizon.obstacles) || []) {
    const d = o.xPos - tx;
    if (d > -20 && d < gap) {
      gap = d;
      oh = (o.typeConfig && o.typeConfig.height) || 0;
      oy = o.yPos;
    }
  }
  return {
    started: !!r.started, crashed: !!r.crashed,
    speed: r.currentSpeed || 0,
    distance: Math.round(r.distanceRan || 0),
    tRexY: r.tRex.yPos, jumping: !!r.tRex.jumping, ducking: !!r.tRex.ducking,
    gap: gap === 9999 ? 9999 : Math.round(gap),
    obstacleH: oh, obstacleY: oy
  };
})()
`;

/**
 * Input, as dispatched KeyboardEvents.
 *
 * Verified working in-page: a dispatched Space keydown/keyup starts the game
 * and triggers a jump, so this is a real actuator and not a mock.
 */
export const ACTION_JS: Record<string, string> = {
  jump: `(() => {
    document.dispatchEvent(new KeyboardEvent('keydown', {code:'Space', keyCode:32, which:32, bubbles:true}));
    setTimeout(() => document.dispatchEvent(new KeyboardEvent('keyup', {code:'Space', keyCode:32, which:32, bubbles:true})), 50);
    return 'jump';
  })()`,
  duck: `(() => {
    document.dispatchEvent(new KeyboardEvent('keydown', {code:'ArrowDown', keyCode:40, which:40, bubbles:true}));
    setTimeout(() => document.dispatchEvent(new KeyboardEvent('keyup', {code:'ArrowDown', keyCode:40, which:40, bubbles:true})), 120);
    return 'duck';
  })()`,
  run: `'run'`,
};

export const RESTART_JS = `
(() => {
  const r = window.Runner && window.Runner.instance_;
  if (!r) return false;
  if (r.crashed) { r.restart(); return true; }
  if (!r.started) {
    document.dispatchEvent(new KeyboardEvent('keydown', {code:'Space', keyCode:32, which:32, bubbles:true}));
    setTimeout(() => document.dispatchEvent(new KeyboardEvent('keyup', {code:'Space', keyCode:32, which:32, bubbles:true})), 50);
    return true;
  }
  return false;
})()
`;

/**
 * Conditions over DinoState.
 *
 * Deliberately authored as a HUMAN would guess them, with plausible but
 * unverified thresholds — the point of the test is whether evolution can
 * improve on a human's first guess against a real game.
 */
export const THRESHOLDS = {
  /** obstacle considered "near" within this many px */
  near: 90,
  /** obstacle considered "very near" within this many px */
  veryNear: 40,
  /** a bird above this y is high enough to run under */
  highBirdY: 60,
};
