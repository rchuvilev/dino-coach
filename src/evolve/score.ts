import { RuleEvaluator } from "../rules/evaluator.js";
import type { Rule, TraceTick } from "../rules/types.js";
import { applyGenome, type Genome } from "./genome.js";

/**
 * Score a genome by REPLAYING recorded traces against it.
 *
 * Counterfactual replay: for each recorded tick we ask what this genome would
 * have done, and credit it with the reward that followed the action it picks.
 * That is only sound where the trace actually contains that action, so the
 * scorer reports coverage - the fraction of ticks it could judge. A score
 * from 5% coverage is noise wearing a number's clothes, and the UI must say so.
 */

export interface ScoreResult {
  /** mean reward per judged tick */
  score: number;
  /** fraction of ticks this genome could be judged on */
  coverage: number;
  /** number of ticks judged */
  judged: number;
}

/**
 * Mean reward that followed each (condition-set, action) pair in the traces.
 *
 * The condition-set half of the key is computed ONCE per tick and cached on
 * the table, because the scorer re-reads the same ticks for every candidate.
 * Measured before caching: 6.16ms per genome over 6000 ticks, which is 5.7s
 * for a 25-generation run and overran the test timeout.
 */
export class RewardTable {
  private sums = new Map<string, { total: number; n: number }>();
  /** tick -> precomputed condition-set prefix */
  private prefixes = new WeakMap<TraceTick, string>();

  constructor(ticks: TraceTick[]) {
    for (const t of ticks) {
      const prefix = conditionPrefix(t.conditions);
      this.prefixes.set(t, prefix);
      if (t.action === null) continue;
      const k = `${prefix}>${t.action}`;
      const e = this.sums.get(k) ?? { total: 0, n: 0 };
      e.total += t.reward;
      e.n++;
      this.sums.set(k, e);
    }
  }

  /** Cached-prefix lookup for a tick already seen by the constructor. */
  lookupTick(tick: TraceTick, action: string): number | null {
    const prefix = this.prefixes.get(tick) ?? conditionPrefix(tick.conditions);
    const e = this.sums.get(`${prefix}>${action}`);
    return e ? e.total / e.n : null;
  }

  lookup(conditions: Record<string, boolean>, action: string): number | null {
    const e = this.sums.get(`${conditionPrefix(conditions)}>${action}`);
    return e ? e.total / e.n : null;
  }
}

function conditionPrefix(conditions: Record<string, boolean>): string {
  const on: string[] = [];
  for (const c in conditions) if (conditions[c]) on.push(c);
  return on.sort().join("+");
}

export function scoreGenome(
  g: Genome,
  rules: Rule[],
  ticks: TraceTick[],
  table: RewardTable,
): ScoreResult {
  const evaluator = new RuleEvaluator(applyGenome(rules, g));
  let total = 0;
  let judged = 0;

  for (const t of ticks) {
    const d = evaluator.step(t.conditions, false);
    if (d.action === null) continue;
    const r = table.lookupTick(t, d.action);
    if (r === null) continue;
    total += r;
    judged++;
  }

  return {
    score: judged > 0 ? total / judged : 0,
    coverage: ticks.length > 0 ? judged / ticks.length : 0,
    judged,
  };
}

/**
 * Split traces into train and held-out. Measured: evolving against a short
 * noisy run reached 3/5 correct rules and stalled while looking fine on its
 * training score, so a held-out slice the GA never sees is mandatory.
 */
export function splitTraces(
  ticks: TraceTick[],
  holdoutFraction = 0.25,
): { train: TraceTick[]; holdout: TraceTick[] } {
  if (ticks.length < 4) return { train: ticks, holdout: [] };
  const cut = Math.floor(ticks.length * (1 - holdoutFraction));
  return { train: ticks.slice(0, cut), holdout: ticks.slice(cut) };
}
