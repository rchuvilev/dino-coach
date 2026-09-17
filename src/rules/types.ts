/** Core types for the rule tier. */

/** A named boolean reading of the world for one tick. */
export type Conditions = Record<string, boolean>;

/** Numeric state props (hp, ammo, ...) read by conditions and reward. */
export type StateProps = Record<string, number>;

/**
 * One `if x do y` rule. Priority is positional (index in RuleSet.order),
 * NOT a field, so a genome can reorder without rewriting rules.
 */
export interface Rule {
  /** condition name, must exist in Conditions */
  when: string;
  /** action name, must exist in the action dict */
  then: string;
  /**
   * Minimum ticks to stay committed once this rule fires.
   * This is the hysteresis that prevents dithering (measured: 66% -> 8%).
   */
  commitTicks?: number;
  /** Ticks this rule cannot re-fire after completing. */
  cooldownTicks?: number;
}

/** A full decision policy: ordered rules, highest priority first. */
export interface RuleSet {
  rules: Rule[];
  /** order[i] = index into rules[]; the genome mutates this */
  order: number[];
}

/** What the evaluator decided this tick, and why. */
export interface Decision {
  action: string | null;
  /** index into RuleSet.rules, or null if nothing matched */
  firedRule: number | null;
  /** true when the action was held by commitment rather than re-chosen */
  committed: boolean;
  /** rules that matched but lost to a higher priority (for the UI trace) */
  suppressed: number[];
  /** rules skipped because they were cooling down */
  cooling: number[];
}

/** One recorded tick. The unit of persistence and of GA scoring. */
export interface TraceTick {
  t: number;
  conditions: Conditions;
  props: StateProps;
  firedRule: number | null;
  action: string | null;
  reward: number;
}
