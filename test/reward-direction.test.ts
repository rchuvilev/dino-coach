import { describe, expect, test } from "bun:test";
import { deltaReward, TraceRecorder } from "../src/kernel/trace.js";

/**
 * Regression tests for a real bug found by running the demo.
 *
 * enemyHp was rated -5 (lower is better), but on death it reset 0 -> 100.
 * That +100 delta scored -500 reward PER KILL, so the rating punished the
 * exact behaviour it was meant to reward: 120 kills scored worse than
 * standing still. Every prop derived from a screenshot has the same hazard.
 */
describe("directional deltas (the respawn-inversion bug)", () => {
  const enemyHp = { rating: -5, countDirection: "down" as const, maxDelta: 50 };

  test("damage to the enemy is rewarded", () => {
    // 100 -> 88 is a decrease of 12, rated -5 => +60
    expect(deltaReward(100, 88, enemyHp)).toBe(60);
  });

  test("REGRESSION: a 0 -> 100 respawn scores 0, not -500", () => {
    expect(deltaReward(0, 100, enemyHp)).toBe(0);
  });

  test("NEGATIVE CONTROL: without the guard the same jump is catastrophic", () => {
    // proves the guard is load-bearing rather than decorative
    expect(deltaReward(0, 100, { rating: -5 })).toBe(-500);
  });

  test("maxDelta alone also suppresses the discontinuity", () => {
    expect(deltaReward(0, 100, { rating: -5, maxDelta: 50 })).toBe(0);
    // but still credits a plausible per-tick change
    expect(deltaReward(100, 88, { rating: -5, maxDelta: 50 })).toBe(60);
  });
});

describe("countDirection", () => {
  test('"up" ignores decreases', () => {
    expect(deltaReward(10, 20, { rating: 1, countDirection: "up" })).toBe(10);
    expect(deltaReward(20, 10, { rating: 1, countDirection: "up" })).toBe(0);
  });

  test('"down" ignores increases', () => {
    expect(deltaReward(20, 10, { rating: 1, countDirection: "down" })).toBe(-10);
    expect(deltaReward(10, 20, { rating: 1, countDirection: "down" })).toBe(0);
  });

  test('"both" is the default and credits either direction', () => {
    expect(deltaReward(10, 20, 1)).toBe(10);
    expect(deltaReward(20, 10, 1)).toBe(-10);
  });

  test("no change yields exactly 0", () => {
    expect(deltaReward(5, 5, { rating: 100 })).toBe(0);
  });
});

describe("TraceRecorder honours reward rules", () => {
  test("a respawn does not pollute the episode total", () => {
    const r = new TraceRecorder("run");
    const spec = { enemyHp: { rating: -5, countDirection: "down" as const, maxDelta: 50 } };
    // realistic per-tick damage: each step is within maxDelta
    r.record(0, {}, { enemyHp: 100 }, null, null, spec);
    r.record(1, {}, { enemyHp: 88 }, null, "shoot", spec); // -12 => +60
    r.record(2, {}, { enemyHp: 76 }, null, "shoot", spec); // -12 => +60
    r.record(3, {}, { enemyHp: 100 }, null, null, spec); // respawn => 0
    expect(r.totalReward()).toBe(120);
  });

  test("numeric shorthand still works", () => {
    const r = new TraceRecorder("run");
    r.record(0, {}, { hp: 100 }, null, null, { hp: 1 });
    expect(r.record(1, {}, { hp: 90 }, null, null, { hp: 1 })).toBe(-10);
  });
});
