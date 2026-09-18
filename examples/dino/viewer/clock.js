/**
 * Manual clock for the real dino game. MUST load before game.js.
 *
 * Everything the game computes - deltaTime, distanceRan, obstacle x, the
 * jump arc - derives from getTimeStamp(), and it schedules frames through a
 * single requestAnimationFrame call. Both are patched in game.js to consult
 * window.__CLOCK, so owning this object makes the REAL game deterministic
 * and steppable without touching its logic.
 */
// Captured BEFORE game.js patches anything, so the driver has a real timer
// to pump from. Using the patched rAF deadlocks: the pump waits on a frame
// only the pump can deliver.
window.__REAL_RAF = window.requestAnimationFrame.bind(window);
window.__CLOCK = {
  realRAF: window.__REAL_RAF,
  manual: true,
  t: Date.now(),
  cbs: [],
  queue(cb) { this.cbs.push(cb); return this.cbs.length; },
  step(ms) {
    this.t += (ms || 1000 / 60);
    const due = this.cbs;
    this.cbs = [];
    for (const cb of due) {
      try { cb(this.t); } catch (e) { /* a throwing frame must not wedge the harness */ }
    }
    return this.cbs.length;
  },
};
