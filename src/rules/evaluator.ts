import type { Conditions, Decision, RuleSet } from "./types.js";

/**
 * Subsumption evaluator: first matching rule in priority order wins.
 *
 * Two details that are NOT optional, both measured:
 *  - commitment (hysteresis): without it, coupled rules dither on 66% of ticks.
 *  - cooldown + in-progress state: without it, a multi-tick action restarts
 *    itself every tick and never completes.
 *
 * Stateful across ticks by design; one instance per running agent.
 */
export class RuleEvaluator {
  private ruleSet: RuleSet;
  /** ticks remaining on the current commitment */
  private commitLeft = 0;
  /** rule index currently committed to */
  private committedRule: number | null = null;
  /** ruleIndex -> tick when it may fire again */
  private cooldownUntil = new Map<number, number>();
  private tick = 0;

  constructor(ruleSet: RuleSet) {
    this.ruleSet = ruleSet;
  }

  /** Swap the policy without losing tick count (used when evolution promotes a genome). */
  setRuleSet(rs: RuleSet): void {
    this.ruleSet = rs;
    this.commitLeft = 0;
    this.committedRule = null;
  }

  getRuleSet(): RuleSet {
    return this.ruleSet;
  }

  /** Decide one tick. */
  step(conditions: Conditions): Decision {
    const t = this.tick++;
    const { rules, order } = this.ruleSet;

    // 1. Honour an existing commitment, but only while its condition holds.
    //    A commitment that outlives its reason is how agents get stuck.
    if (this.commitLeft > 0 && this.committedRule !== null) {
      const r = rules[this.committedRule];
      if (r && conditions[r.when]) {
        this.commitLeft--;
        return {
          action: r.then,
          firedRule: this.committedRule,
          committed: true,
          suppressed: [],
          cooling: [],
        };
      }
      // condition lapsed -> drop the commitment early
      this.commitLeft = 0;
      this.committedRule = null;
    }

    // 2. Fresh evaluation, highest priority first.
    const suppressed: number[] = [];
    const cooling: number[] = [];
    let chosen: number | null = null;

    for (const idx of order) {
      const r = rules[idx];
      if (!r) continue;
      if (!conditions[r.when]) continue;

      const until = this.cooldownUntil.get(idx) ?? 0;
      if (t < until) {
        cooling.push(idx);
        continue;
      }
      if (chosen === null) chosen = idx;
      else suppressed.push(idx);
    }

    if (chosen === null) {
      return { action: null, firedRule: null, committed: false, suppressed, cooling };
    }

    const rule = rules[chosen]!;
    const commit = rule.commitTicks ?? 0;
    if (commit > 1) {
      this.commitLeft = commit - 1;
      this.committedRule = chosen;
    }
    if (rule.cooldownTicks) {
      this.cooldownUntil.set(chosen, t + commit + rule.cooldownTicks);
    }

    return { action: rule.then, firedRule: chosen, committed: false, suppressed, cooling };
  }

  reset(): void {
    this.commitLeft = 0;
    this.committedRule = null;
    this.cooldownUntil.clear();
    this.tick = 0;
  }
}
