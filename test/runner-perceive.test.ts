import { describe, expect, test } from "bun:test";
import { perceive, readGap, readHigh, readPlayerY } from "../examples/runner/perceive.js";
import { render, RunnerWorld } from "../examples/runner/world.js";

/**
 * Validate PIXEL readers against the world's ground truth.
 *
 * This is the test the Chrome Dino run never had: there I read the game's
 * state object directly, so a reader pointed at an always-empty array
 * returned a constant 9999 and nothing noticed. Here every reading is
 * checked against `world.truth()`, which is the only reason a silent
 * perception bug cannot survive.
 */

describe("readGap agrees with ground truth", () => {
  test("mean absolute error is small across many frames", () => {
    const w = new RunnerWorld(21);
    let n = 0;
    let sum = 0;
    let worst = 0;
    for (let i = 0; i < 900; i++) {
      if (!w.step("run")) w.reset(21 + i);
      const t = w.truth();
      if (t.gap >= 999) continue; // no obstacle visible
      const got = readGap(render(w));
      if (got >= 999) continue;
      const err = Math.abs(got - t.gap);
      sum += err;
      worst = Math.max(worst, err);
      n++;
    }
    expect(n).toBeGreaterThan(50); // the test must actually have data
    expect(sum / n).toBeLessThan(2.5);
    expect(worst).toBeLessThan(10);
  });

  test("returns 999 when no obstacle is on screen", () => {
    const w = new RunnerWorld(2);
    w.obstacles = [];
    expect(readGap(render(w))).toBe(999);
  });

  test("gap DECREASES as an obstacle approaches", () => {
    // a reader that returns a constant would pass a mean-error test on a
    // static scene; this asserts it tracks movement
    const w = new RunnerWorld(13);
    const seen: number[] = [];
    for (let i = 0; i < 200; i++) {
      w.step("run");
      const g = readGap(render(w));
      if (g < 999) seen.push(g);
    }
    expect(seen.length).toBeGreaterThan(10);
    // at least one strictly decreasing stretch
    let decreasing = 0;
    for (let i = 1; i < seen.length; i++) if (seen[i]! < seen[i - 1]!) decreasing++;
    expect(decreasing).toBeGreaterThan(seen.length / 3);
  });
});

describe("readHigh distinguishes obstacle types", () => {
  test("a ground-level obstacle reads as NOT high", () => {
    const w = new RunnerWorld(1);
    w.obstacles = [{ x: 60, high: false, w: 6, h: 12 }];
    expect(readHigh(render(w))).toBe(false);
  });

  test("a high obstacle reads as high", () => {
    const w = new RunnerWorld(1);
    w.obstacles = [{ x: 60, high: true, w: 10, h: 6 }];
    expect(readHigh(render(w))).toBe(true);
  });

  test("agrees with truth across generated frames", () => {
    const w = new RunnerWorld(33);
    let checked = 0;
    let correct = 0;
    for (let i = 0; i < 1200; i++) {
      if (!w.step("run")) w.reset(33 + i);
      const t = w.truth();
      if (t.gap >= 999 || t.gap > 60) continue;
      if (readHigh(render(w)) === t.high) correct++;
      checked++;
    }
    expect(checked).toBeGreaterThan(40);
    expect(correct / checked).toBeGreaterThan(0.9);
  });
});

describe("readPlayerY tracks the jump", () => {
  test("reads 0 while standing", () => {
    const w = new RunnerWorld(4);
    w.step("run");
    expect(readPlayerY(render(w))).toBeLessThanOrEqual(1);
  });

  test("reads positive while airborne, and agrees with truth", () => {
    const w = new RunnerWorld(4);
    w.step("jump");
    let maxErr = 0;
    let sawAir = false;
    for (let i = 0; i < 25; i++) {
      const t = w.truth();
      const got = readPlayerY(render(w));
      if (t.y > 2) {
        sawAir = true;
        maxErr = Math.max(maxErr, Math.abs(got - t.y));
      }
      w.step("run");
    }
    expect(sawAir).toBe(true);
    expect(maxErr).toBeLessThan(3);
  });
});

describe("perceive bundles the readings", () => {
  test("airborne follows playerY", () => {
    const w = new RunnerWorld(4);
    w.step("jump");
    w.step("run");
    expect(perceive(render(w)).airborne).toBe(true);
  });

  test("a blank frame yields safe defaults rather than throwing", () => {
    const blank: { width: number; height: number; data: Uint8Array } = {
      width: 200,
      height: 60,
      data: new Uint8Array(200 * 60 * 4).fill(255),
    };
    const p = perceive(blank);
    expect(p.gap).toBe(999);
    expect(p.airborne).toBe(false);
  });
});
