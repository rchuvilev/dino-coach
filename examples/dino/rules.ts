import type { Project } from "../../src/project.js";
import type { DinoState } from "./adapter.js";
import { THRESHOLDS } from "./adapter.js";

/**
 * The dino project: props, conditions, actions and a HUMAN'S FIRST GUESS at
 * the rules. The test is whether evolution improves on that guess against a
 * real game rather than a synthetic world.
 */

export const ACTIONS = ["jump", "duck", "run"] as const;

/** Conditions a person would plausibly write, with guessed thresholds. */
export function conditionsOf(s: DinoState): Record<string, boolean> {
  const has = s.gap < 9999;
  return {
    obstacle_very_near: has && s.gap <= THRESHOLDS.veryNear,
    obstacle_near: has && s.gap <= THRESHOLDS.near,
    high_bird: has && s.obstacleY < THRESHOLDS.highBirdY,
    airborne: s.jumping,
    always: true,
  };
}

/**
 * Reward comes from distance gained, with a large penalty on crashing.
 *
 * Dense by construction: distance rises every tick the dino survives, so the
 * signal does not wait for a terminal event. Measured earlier, dense reward
 * solved a 6-step task (0.94) where sparse terminal reward did not (0.06).
 */
export const PROP_RATINGS = {
  // distance only ever increases within a run; a restart resets it to 0,
  // which countDirection:"up" correctly ignores as a discontinuity
  distance: { rating: 1, countDirection: "up" as const, maxDelta: 200 },
  // crashed goes 0 -> 1 once; rated heavily negative
  crashed: { rating: -2000, countDirection: "up" as const },
};

/** The human's proposed rules. Deliberately imperfect. */
export const SEED_RULES: Project["rules"] = [
  { when: "obstacle_very_near", then: "jump" },
  { when: "high_bird", then: "duck" },
  { when: "obstacle_near", then: "jump" },
  { when: "always", then: "run" },
];
