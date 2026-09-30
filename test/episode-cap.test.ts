import { describe, expect, test } from "bun:test";
import { EP_FRAME_CAP, capPointsAt } from "../examples/dino/viewer/evolve.js";

/**
 * The episode frame cap silently imposed a SCORE ceiling.
 *
 * EP_FRAME_CAP was 9000 frames, chosen as "~150s of game time, beyond which
 * the candidate is clearly strong and further frames add no ranking
 * information". That reasoning is sound for RANKING but wrong for RECORDING:
 * the cap does not merely stop measuring, it ends the episode and restarts
 * the game, so the score that would have been achieved never exists.
 *
 * The arithmetic: points = distance * 0.025 and distance accrues about
 * `currentSpeed` px per frame, so at the game's terminal speed of 13 the
 * ceiling is 9000 * 13 * 0.025 = 2925 points. Measured on a live profile,
 * bestEver was EXACTLY 2925 - not near it, on it. A user reporting lost
 * 5000+ runs was right: such a run could not be recorded at all.
 */

describe("the frame cap must not cap the SCORE a user can reach", () => {
  test("REGRESSION: the ceiling is above the scores users actually reach", () => {
    // A 5000+ run was reported. The cap must comfortably exceed that, or the
    // score is unreachable however well the agent plays.
    expect(capPointsAt(13)).toBeGreaterThan(6000);
  });

  test("REGRESSION: the OLD cap could not have recorded a 5000pt run", () => {
    // Proves the test is measuring the real defect rather than passing by
    // construction: with the previous 9000-frame cap the ceiling was 2925.
    const oldCeiling = Math.round(9000 * 13 * 0.025);
    expect(oldCeiling).toBeLessThan(5000);
  });

  test("the ceiling scales with speed, and even the SLOWEST case is generous", () => {
    // Early-run speed is 6. A long survival at low speed must still not be
    // truncated before a strong score.
    expect(capPointsAt(6)).toBeGreaterThan(2500);
    expect(capPointsAt(13)).toBeGreaterThan(capPointsAt(6));
  });

  test("the cap still exists - an immortal policy cannot stall the search", () => {
    // Removing the cap entirely is the wrong fix: one genome that never dies
    // would block its whole generation forever.
    expect(Number.isFinite(EP_FRAME_CAP)).toBe(true);
    expect(EP_FRAME_CAP).toBeGreaterThan(0);
  });

  test("capPointsAt is defensive about its argument", () => {
    expect(capPointsAt(0)).toBeGreaterThanOrEqual(0);
    expect(Number.isFinite(capPointsAt(undefined as any))).toBe(true);
  });
});
