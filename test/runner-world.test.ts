import { describe, expect, test } from "bun:test";
import { PLAYER_X, PLAYER_W, render, RunnerWorld } from "../examples/runner/world.js";

/**
 * These tests exist because the Chrome Dino attempt produced a completely
 * plausible score (8361, with near-identical runs that looked like clean
 * determinism) from a FROZEN world. Every assertion here is aimed at making
 * that specific lie impossible.
 */

describe("the simulation actually advances", () => {
  test("obstacles move TOWARD the player, not just the odometer up", () => {
    // the dino bug: distanceRan climbed to 23325 while the obstacle sat at x=346
    const w = new RunnerWorld(7);
    for (let i = 0; i < 80; i++) w.step("run");
    const first = w.obstacles[0];
    expect(first).toBeDefined();
    const x0 = first!.x;
    for (let i = 0; i < 10; i++) w.step("run");
    expect(w.obstacles[0] ? w.obstacles[0].x : -999).toBeLessThan(x0);
  });

  test("distance only rises while the world is live", () => {
    const w = new RunnerWorld(3);
    for (let i = 0; i < 40; i++) w.step("run");
    expect(w.distance).toBeGreaterThan(30);
  });

  test("obstacles are actually spawned", () => {
    const w = new RunnerWorld(11);
    let sawAny = false;
    for (let i = 0; i < 300; i++) {
      w.step("jump");
      if (w.obstacles.length > 0) sawAny = true;
    }
    // positive control: a world that never spawns cannot be a fair test
    expect(sawAny).toBe(true);
  });
});

describe("crashes are detectable", () => {
  test("NEGATIVE CONTROL: doing nothing eventually crashes", () => {
    // this is the assertion the dino target failed - 400 ticks, no crash
    const w = new RunnerWorld(5);
    let crashed = false;
    for (let i = 0; i < 600; i++) {
      if (!w.step("run")) {
        crashed = true;
        break;
      }
    }
    expect(crashed).toBe(true);
    expect(w.crashed).toBe(true);
  });

  test("a crashed world refuses to advance further", () => {
    const w = new RunnerWorld(5);
    while (w.step("run")) {
      /* run to crash */
    }
    const d = w.distance;
    expect(w.step("run")).toBe(false);
    expect(w.distance).toBe(d);
  });

  test("jumping survives longer than standing still", () => {
    // sanity: the game must be winnable by acting, or scores are meaningless
    const stand = new RunnerWorld(5);
    let sd = 0;
    while (stand.step("run") && sd < 3000) sd++;

    const jump = new RunnerWorld(5);
    let jd = 0;
    while (jd < 3000) {
      const o = jump.nearest();
      const gap = o ? o.x - (PLAYER_X + PLAYER_W) : 999;
      if (!jump.step(o && !o.high && gap < 18 ? "jump" : "run")) break;
      jd++;
    }
    expect(jd).toBeGreaterThan(sd);
  });
});

describe("determinism", () => {
  test("the same seed and actions give an identical result", () => {
    const a = new RunnerWorld(42);
    const b = new RunnerWorld(42);
    for (let i = 0; i < 200; i++) {
      a.step("run");
      b.step("run");
    }
    expect(a.distance).toBe(b.distance);
    expect(a.crashed).toBe(b.crashed);
    expect(a.obstacles.map((o) => Math.round(o.x))).toEqual(
      b.obstacles.map((o) => Math.round(o.x)),
    );
  });

  test("different seeds give different worlds", () => {
    // proves the seed is load-bearing, so determinism is not just a frozen world
    const a = new RunnerWorld(1);
    const b = new RunnerWorld(999);
    for (let i = 0; i < 400; i++) {
      a.step("run");
      b.step("run");
    }
    expect(a.distance).not.toBe(b.distance);
  });

  test("reset restores a world to its initial state", () => {
    const w = new RunnerWorld(8);
    for (let i = 0; i < 100; i++) w.step("jump");
    w.reset();
    expect(w.distance).toBe(0);
    expect(w.crashed).toBe(false);
    expect(w.obstacles).toHaveLength(0);
    expect(w.y).toBe(0);
  });
});

describe("physics", () => {
  test("jump leaves the ground and returns to it", () => {
    const w = new RunnerWorld(2);
    w.step("jump");
    expect(w.y).toBeGreaterThan(0);
    for (let i = 0; i < 40; i++) w.step("run");
    expect(w.y).toBe(0);
  });

  test("a second jump mid-air is ignored", () => {
    const w = new RunnerWorld(2);
    w.step("jump");
    const peak1 = w.y;
    w.step("jump");
    // must follow ballistic motion, not get a second impulse
    expect(w.y).toBeGreaterThan(peak1);
    expect(w.vy).toBeLessThan(5.2);
  });

  test("duck lowers the player box and expires", () => {
    const w = new RunnerWorld(2);
    w.step("duck");
    expect(w.ducking).toBe(true);
    for (let i = 0; i < 10; i++) w.step("run");
    expect(w.ducking).toBe(false);
  });
});

describe("render produces readable pixels", () => {
  test("frame has the right shape and is not blank", () => {
    const w = new RunnerWorld(4);
    for (let i = 0; i < 100; i++) w.step("run");
    const f = render(w);
    expect(f.width * f.height * 4).toBe(f.data.length);
    let dark = 0;
    for (let i = 0; i < f.data.length; i += 4) if (f.data[i]! < 128) dark++;
    // ground line alone guarantees some dark pixels; player adds more
    expect(dark).toBeGreaterThan(f.width);
  });

  test("the rendered frame CHANGES as the world advances", () => {
    // the dino failure: 8 samples, 1 unique frame signature
    const w = new RunnerWorld(6);
    const sig = () => {
      const f = render(w);
      let h = 0;
      for (let i = 0; i < f.data.length; i += 37) h = (h * 31 + f.data[i]!) | 0;
      return h;
    };
    // window must cover at least one spawn: with nextSpawn=24 the first
    // obstacle appears at tick 24, and an all-white pre-spawn scene is
    // legitimately identical frame to frame
    const seen = new Set<number>();
    for (let i = 0; i < 120; i++) {
      w.step("run");
      seen.add(sig());
    }
    expect(seen.size).toBeGreaterThan(20);
  });

  test("a jumping player renders higher than a standing one", () => {
    const stand = new RunnerWorld(9);
    stand.step("run");
    const fa = render(stand);
    const jump = new RunnerWorld(9);
    jump.step("jump");
    const fb = render(jump);
    const colDark = (f: { width: number; data: Uint8Array }, x: number) => {
      const rows: number[] = [];
      for (let y = 0; y < 60; y++) {
        const i = (y * f.width + x) * 4;
        if (f.data[i]! < 128) rows.push(y);
      }
      return rows.length ? Math.min(...rows) : 99;
    };
    // topmost dark pixel in the player column must be higher when jumping
    expect(colDark(fb, PLAYER_X + 2)).toBeLessThan(colDark(fa, PLAYER_X + 2));
  });
});
