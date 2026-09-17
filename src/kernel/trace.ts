import type { StateProps, TraceTick } from "../rules/types.js";

/**
 * Trace recorder + discounted return computation.
 *
 * The backwards discount pass is the mechanism that links "ammo dropped at
 * t=40" to "pressed X at t=37". Measured: with reward delayed 3 ticks, a
 * learner using gamma=0 picks the true cause 1% of the time (WORSE than the
 * 20% chance baseline, because it confidently credits bystanders), while
 * gamma=0.9 picks it 100% of the time.
 */

export interface Episode {
  ticks: TraceTick[];
  /** which run produced this episode - provenance for the GA */
  runId: string;
}

/** gamma from a human-readable horizon: how far back consequences reach. */
export function gammaForHorizon(seconds: number, hz: number): number {
  const ticks = Math.max(1, seconds * hz);
  return Math.min(0.999, 1 - 1 / ticks);
}

/** Inverse: report a gamma as a horizon in seconds, for the UI. */
export function horizonForGamma(gamma: number, hz: number): number {
  if (gamma >= 1) return Infinity;
  return 1 / (1 - gamma) / hz;
}

/** Backwards discounted return: G[t] = r[t] + gamma * G[t+1]. */
export function discountedReturns(rewards: number[], gamma: number): number[] {
  const g = new Array<number>(rewards.length);
  let run = 0;
  for (let i = rewards.length - 1; i >= 0; i--) {
    run = rewards[i]! + gamma * run;
    g[i] = run;
  }
  return g;
}

/** Zero-mean, unit-variance advantages. Undefined variance -> zeros. */
export function normalize(values: number[]): number[] {
  if (values.length === 0) return [];
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  const varc = values.reduce((a, b) => a + (b - mean) ** 2, 0) / values.length;
  const sd = Math.sqrt(varc);
  if (sd < 1e-9) return values.map(() => 0);
  return values.map((v) => (v - mean) / sd);
}

/** How a prop's delta converts to reward. Mirrors PropSpec, without the fns. */
export interface RewardRule {
  rating: number;
  countDirection?: "up" | "down" | "both";
  maxDelta?: number;
}

/** Accept a bare number as shorthand for {rating, both}. */
export type RewardSpec = number | RewardRule;

function asRule(s: RewardSpec): RewardRule {
  return typeof s === "number" ? { rating: s } : s;
}

/**
 * Reward contribution of one prop's change.
 *
 * Exported because this is the single most error-prone line in the system
 * and it deserves direct tests: a discontinuity credited as a consequence
 * inverts the incentive it was meant to express.
 */
export function deltaReward(before: number, after: number, spec: RewardSpec): number {
  const rule = asRule(spec);
  const delta = after - before;
  if (delta === 0) return 0;
  if (rule.maxDelta !== undefined && Math.abs(delta) > rule.maxDelta) return 0;
  const dir = rule.countDirection ?? "both";
  if (dir === "up" && delta < 0) return 0;
  if (dir === "down" && delta > 0) return 0;
  return delta * rule.rating;
}

export class TraceRecorder {
  private ticks: TraceTick[] = [];
  private prev: StateProps | null = null;

  constructor(private runId: string) {}

  /**
   * Record one tick. Reward is computed from the CHANGE in state props,
   * which is what makes it dense: measured, dense reward solves a 6-step
   * task (0.94) where sparse terminal reward does not (0.06).
   */
  record(
    t: number,
    conditions: Record<string, boolean>,
    props: StateProps,
    firedRule: number | null,
    action: string | null,
    propRatings: Record<string, RewardSpec>,
  ): number {
    let reward = 0;
    if (this.prev) {
      for (const [key, spec] of Object.entries(propRatings)) {
        const before = this.prev[key];
        const after = props[key];
        if (before === undefined || after === undefined) continue;
        reward += deltaReward(before, after, spec);
      }
    }
    this.prev = { ...props };
    this.ticks.push({ t, conditions, props: { ...props }, firedRule, action, reward });
    return reward;
  }

  episode(): Episode {
    return { ticks: this.ticks, runId: this.runId };
  }

  get length(): number {
    return this.ticks.length;
  }

  totalReward(): number {
    return this.ticks.reduce((a, b) => a + b.reward, 0);
  }

  reset(): void {
    this.ticks = [];
    this.prev = null;
  }
}
