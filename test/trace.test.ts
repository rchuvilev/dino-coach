import { describe, expect, test } from "bun:test";
import {
  discountedReturns,
  gammaForHorizon,
  horizonForGamma,
  normalize,
  TraceRecorder,
} from "../src/kernel/trace.js";

describe("discounted returns", () => {
  test("gamma=0 keeps only the current tick - the lookup-table behaviour", () => {
    expect(discountedReturns([1, 2, 3], 0)).toEqual([1, 2, 3]);
  });

  test("gamma>0 propagates a later reward backwards", () => {
    // reward only at the end; earlier ticks must still receive credit
    const g = discountedReturns([0, 0, 10], 0.5);
    expect(g[2]).toBe(10);
    expect(g[1]).toBe(5);
    expect(g[0]).toBe(2.5);
  });

  test("POSITIVE CONTROL: credit actually reaches a 3-tick-earlier action", () => {
    // this is the property that makes delayed causes learnable at all
    const g = discountedReturns([0, 0, 0, 1], 0.9);
    expect(g[0]).toBeGreaterThan(0.7);
    // and gamma=0 must NOT have it, or the test proves nothing
    expect(discountedReturns([0, 0, 0, 1], 0)[0]).toBe(0);
  });
});

describe("horizon <-> gamma", () => {
  test("round-trips", () => {
    const hz = 10;
    const g = gammaForHorizon(1, hz); // 1 second
    expect(horizonForGamma(g, hz)).toBeCloseTo(1, 5);
  });

  test("a longer horizon yields a larger gamma", () => {
    expect(gammaForHorizon(10, 10)).toBeGreaterThan(gammaForHorizon(1, 10));
  });
});

describe("normalize", () => {
  test("zero mean and unit variance", () => {
    const n = normalize([1, 2, 3, 4]);
    const mean = n.reduce((a, b) => a + b, 0) / n.length;
    expect(mean).toBeCloseTo(0, 9);
  });

  test("constant input yields zeros rather than NaN", () => {
    expect(normalize([5, 5, 5])).toEqual([0, 0, 0]);
  });
});

describe("TraceRecorder reward from prop deltas", () => {
  const ratings = { hp: 1, ammo: 0.5 };

  test("first tick has no delta, so no reward", () => {
    const r = new TraceRecorder("run1");
    expect(r.record(0, {}, { hp: 100, ammo: 50 }, null, null, ratings)).toBe(0);
  });

  test("a drop in a positively-rated prop is negative reward", () => {
    const r = new TraceRecorder("run1");
    r.record(0, {}, { hp: 100, ammo: 50 }, null, null, ratings);
    expect(r.record(1, {}, { hp: 90, ammo: 50 }, null, null, ratings)).toBe(-10);
  });

  test("a rise is positive, and ratings scale it", () => {
    const r = new TraceRecorder("run1");
    r.record(0, {}, { hp: 100, ammo: 50 }, null, null, ratings);
    // ammo +10 at rating 0.5 => +5
    expect(r.record(1, {}, { hp: 100, ammo: 60 }, null, null, ratings)).toBe(5);
  });

  test("unrated props contribute nothing", () => {
    const r = new TraceRecorder("run1");
    r.record(0, {}, { hp: 100, score: 0 }, null, null, { hp: 1 });
    expect(r.record(1, {}, { hp: 100, score: 999 }, null, null, { hp: 1 })).toBe(0);
  });

  test("records accumulate and total correctly", () => {
    const r = new TraceRecorder("run1");
    r.record(0, {}, { hp: 100 }, null, null, { hp: 1 });
    r.record(1, {}, { hp: 90 }, null, null, { hp: 1 });
    r.record(2, {}, { hp: 85 }, null, null, { hp: 1 });
    expect(r.length).toBe(3);
    expect(r.totalReward()).toBe(-15);
  });
});
