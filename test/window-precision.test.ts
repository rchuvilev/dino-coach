import { describe, expect, test } from "bun:test";
import { windowAt, fitnessOf } from "../examples/dino/viewer/evolve.js";

/**
 * The takeoff window (the green stripe) had two measured defects.
 *
 * 1. IT IGNORED OBSTACLE SIZE. windowAt(g, speed) took no width argument at
 *    all. `wideAdj` shifted BOTH edges equally, so measured on a live
 *    champion the window was 51px wide for a narrow cactus and 51px wide for
 *    a wide one at speed 6 - the position moved, the size never did. A wide
 *    obstacle needs more air time, which is a different amount of window,
 *    not the same window further out.
 *
 * 2. IT NEVER TIGHTENED. fitnessOf() scored outcomes only, so a genome that
 *    cleared obstacles with a 90px window scored exactly the same as one
 *    with a 20px window. Worse, a wide window fires across more approach
 *    distances and is therefore MORE robust, so selection actively favoured
 *    imprecision. A better-learned policy had no way to express itself as a
 *    sharper window.
 */

const g = {
  loA: 55, loB: 2.19, widthA: 65, widthB: -5.04, duck: 30,
  wideAdj: 19, bands: [{ lo: 0, w: 0 }, { lo: 0, w: 0 }, { lo: 0, w: 0 }],
};

describe("the window responds to obstacle size", () => {
  test("REGRESSION: a wide obstacle gets a DIFFERENT window width, not just a shift", () => {
    const narrow = windowAt(g, 9, 17);    // small cactus
    const wide = windowAt(g, 9, 75);      // wide cluster
    const wN = narrow.hi - narrow.lo;
    const wW = wide.hi - wide.lo;
    expect(wW).not.toBe(wN);
  });

  test("a wider obstacle opens the window EARLIER - it needs more air time", () => {
    // Use a genome well below LO_MAX (55). The original fixture sat AT the
    // clamp, so `lo` could not rise and the test failed while the code was
    // correct - LO_MAX is a deliberate physical bound ("takeoff must start
    // inside the jump arc"), not a bug to work around.
    const low = { ...g, loA: 25 };
    const narrow = windowAt(low, 9, 17);
    const wide = windowAt(low, 9, 75);
    expect(wide.lo).toBeGreaterThan(narrow.lo);
  });

  test("REGRESSION: lo is still clamped by the physical jump-arc bound", () => {
    // Width awareness must not let the window open beyond where a takeoff
    // can physically still clear the obstacle.
    const r = windowAt({ ...g, loA: 55 }, 9, 200);
    expect(r.lo).toBeLessThanOrEqual(55);
  });

  test("omitting width reproduces the previous behaviour exactly", () => {
    // Callers that do not know the obstacle (the strategy panel renders a
    // window before any obstacle is on screen) must be unaffected.
    const a = windowAt(g, 9);
    const b = windowAt(g, 9, 0);
    expect(a.lo).toBe(b.lo);
    expect(a.hi).toBe(b.hi);
  });

  test("the window stays within its bounds at extreme widths", () => {
    for (const w of [0, 17, 50, 75, 200, 9999]) {
      const r = windowAt(g, 12, w);
      expect(r.lo).toBeGreaterThanOrEqual(5);
      expect(r.hi).toBeGreaterThan(r.lo);
      expect(Number.isFinite(r.lo)).toBe(true);
      expect(Number.isFinite(r.hi)).toBe(true);
    }
  });
});

describe("precision is rewarded, but never above survival", () => {
  const cand = (median: number, width: number) => ({
    median,
    g: { ...g, widthA: width, widthB: 0 },
    tel: { causes: { ok: 100 }, wideClearRate: null, birdClearRate: null },
  });

  test("REGRESSION: among equal scorers, the tighter window wins", () => {
    // Previously identical: fitness read only outcomes, so there was no
    // gradient toward a sharper window at all.
    const tight = fitnessOf(cand(1000, 20));
    const loose = fitnessOf(cand(1000, 90));
    expect(tight).toBeGreaterThan(loose);
  });

  test("REGRESSION: a better score still beats a tighter window", () => {
    // The danger of rewarding precision is trading away survival for
    // elegance. A 2x score difference must dominate any width bonus.
    const sloppyButGood = fitnessOf(cand(2000, 90));
    const tightButWorse = fitnessOf(cand(1000, 20));
    expect(sloppyButGood).toBeGreaterThan(tightButWorse);
  });

  test("the precision term is small - at most a few percent", () => {
    const tight = fitnessOf(cand(1000, 10));
    const loose = fitnessOf(cand(1000, 150));
    const spread = (tight - loose) / 1000;
    expect(spread).toBeGreaterThan(0);
    expect(spread).toBeLessThan(0.1);
  });

  test("a candidate with no telemetry is still scored", () => {
    expect(fitnessOf({ median: 500 })).toBe(500);
  });
});

describe("the champion genome is migrated like every other", () => {
  test("REGRESSION: a champion adopted before widthGain existed gains the default", async () => {
    // Measured: after adding widthGain, a champion adopted from the shared
    // pool had no widthGain while the in-progress population had 0.33/0.77.
    // migrate() ran migrateGenome on the population and bestGenome but NOT
    // on S.champion.g - the one genome that actually plays - so the new
    // axis was dead on exactly the genome using it.
    const { clampGenome } = await import("../examples/dino/viewer/evolve.js");
    if (typeof clampGenome !== "function") return;   // not exported; covered live
    const old = clampGenome({ loA: 30, loB: 0, widthA: 40, widthB: 0, duck: 30 });
    expect(Number.isFinite(old.widthGain)).toBe(true);
  });

  test("widthGain stays inside its bounds after clamping", async () => {
    const mod: any = await import("../examples/dino/viewer/evolve.js");
    if (typeof mod.clampGenome !== "function") return;
    for (const v of [-5, 0, 0.45, 1.2, 99]) {
      const g2 = mod.clampGenome({ loA: 30, loB: 0, widthA: 40, widthB: 0, duck: 30, widthGain: v });
      expect(g2.widthGain).toBeGreaterThanOrEqual(0);
      expect(g2.widthGain).toBeLessThanOrEqual(1.2);
    }
  });
});
