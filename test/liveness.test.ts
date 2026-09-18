import { describe, expect, test } from "bun:test";
import { assessAgainstControls, verifyLiveness } from "../src/evolve/liveness.js";

/**
 * Each test here reproduces a real failure that cost a full experiment run,
 * and asserts the guard now catches it. A guard is only trustworthy once it
 * has been made to FAIL on the thing it claims to detect.
 */

/** A healthy target: advances, and responds to input. */
function healthyTarget() {
  let d = 0;
  let boost = 0;
  return {
    reset: () => {
      d = 0;
      boost = 0;
    },
    step: (a: string) => {
      if (a === "jump") boost += 1;
      d += 1 + boost * 0.1;
    },
    observe: () => ({ d: Math.round(d * 10) }),
    progress: (s: { d: number }) => s.d,
    probeAction: "jump",
    idleAction: "run",
  };
}

describe("verifyLiveness accepts a working target", () => {
  test("a healthy target is alive", async () => {
    const r = await verifyLiveness(healthyTarget());
    expect(r.alive).toBe(true);
    expect(r.advances).toBe(true);
    expect(r.actionHasEffect).toBe(true);
    expect(r.reasons).toEqual([]);
  });
});

describe("the Chrome Dino failure modes", () => {
  test("REGRESSION: a frozen target is rejected, not scored", async () => {
    // this is the 8361 fake score: state never changes, but a naive harness
    // still reads a plausible-looking number out of it
    let frozenAt = 1376;
    const r = await verifyLiveness({
      reset: () => {},
      step: () => {},
      observe: () => ({ d: frozenAt }),
      progress: (s: { d: number }) => s.d,
      probeAction: "jump",
      idleAction: "run",
    });
    expect(r.alive).toBe(false);
    expect(r.advances).toBe(false);
    expect(r.distinctStates).toBe(1);
    expect(r.reasons.join(" ")).toContain("frozen");
  });

  test("REGRESSION: a target that ignores input is rejected", async () => {
    // the crashed dino: the odometer moved but no action changed anything.
    // A target that advances but cannot be controlled is useless for scoring
    // and must not pass just because progress increases.
    let d = 0;
    const r = await verifyLiveness({
      reset: () => {
        d = 0;
      },
      step: () => {
        d += 1; // action ignored entirely
      },
      observe: () => ({ d }),
      progress: (s: { d: number }) => s.d,
      probeAction: "jump",
      idleAction: "run",
    });
    expect(r.advances).toBe(true); // it DOES advance
    expect(r.actionHasEffect).toBe(false); // but input does nothing
    expect(r.alive).toBe(false); // so it is not usable
    expect(r.reasons.join(" ")).toContain("input is not reaching");
  });

  test("REGRESSION: a world whose odometer rises while nothing moves is rejected", async () => {
    // the fake rAF clock: distanceRan climbed to 23325 while the obstacle sat
    // frozen at x=346. Progress alone is not evidence of a live simulation,
    // so the distinct-state check is what catches it.
    let d = 0;
    const r = await verifyLiveness({
      reset: () => {
        d = 0;
      },
      step: () => {
        d += 1;
      },
      // observe() deliberately omits the odometer, reporting only the world
      observe: () => ({ obstacleX: 346 }),
      progress: () => d,
      probeAction: "jump",
      idleAction: "run",
    });
    expect(r.distinctStates).toBe(1);
    expect(r.alive).toBe(false);
    expect(r.reasons.join(" ")).toContain("frozen");
  });
});

describe("the guard's own defect", () => {
  test("REGRESSION: an action that changes STATE but not PROGRESS still counts", async () => {
    // First version compared only the progress trajectory. On the runner
    // target a jump does not change DISTANCE within a short probe (the first
    // obstacle has not arrived), so a controllable world was rejected with
    // "input is not reaching the target". The observed state showed 10
    // distinct y values vs 1 - evidence already collected and discarded.
    let d = 0;
    let y = 0;
    let vy = 0;
    const r = await verifyLiveness({
      reset: () => {
        d = 0;
        y = 0;
        vy = 0;
      },
      step: (a) => {
        if (a === "jump" && y === 0) vy = 5;
        y = Math.max(0, y + vy);
        vy = y > 0 ? vy - 1 : 0;
        d += 1; // progress is IDENTICAL regardless of action
      },
      observe: () => ({ d, y: Math.round(y) }),
      progress: (s: { d: number }) => s.d,
      probeAction: "jump",
      idleAction: "run",
      ticks: 12,
    });
    expect(r.actionHasEffect).toBe(true);
    expect(r.alive).toBe(true);
  });
});

describe("assessAgainstControls", () => {
  const controls = { doNothing: 132, alwaysJump: 143 };

  test("accepts a policy that beats every control", () => {
    const a = assessAgainstControls(2030, controls);
    expect(a.valid).toBe(true);
    expect(a.best).toBe("alwaysJump");
    expect(a.ratio).toBeCloseTo(14.2, 0);
    expect(a.verdict).toContain("beats best control");
  });

  test("REGRESSION: flags a policy a control matches or beats", () => {
    // the runner target's ceiling-condition bug: an ORACLE scored 129 while
    // do-nothing scored 129. Reporting that as "evolution does not help"
    // would have been wrong - the environment was broken.
    const a = assessAgainstControls(112, controls);
    expect(a.valid).toBe(false);
    expect(a.verdict).toContain("SUSPECT");
    expect(a.verdict).toContain("not the policy");
  });

  test("a tie is suspect, not a pass", () => {
    expect(assessAgainstControls(143, controls).valid).toBe(false);
  });

  test("identifies the strongest control, not the first", () => {
    expect(assessAgainstControls(200, { a: 10, b: 190, c: 50 }).best).toBe("b");
  });
});
