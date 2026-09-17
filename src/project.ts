import type { Rule, RuleSet, StateProps } from "./rules/types.js";

/**
 * A Project is the ONE source of truth. It is plain data, so:
 *  - code-first: you write it in TS with full type checking and autocomplete
 *  - UI-representable: the editor reads and writes this same object
 *  - persistable: it is JSON, so runs can save and reload it
 *
 * DRY: there is no second schema for the UI. The editor is a view over this.
 * YAGNI: no node graph, no DSL, no plugin system until something needs one.
 */
export interface Project {
  name: string;
  /** tick rate; also converts gamma <-> a horizon in seconds for the UI */
  hz: number;

  /**
   * Step 1: state props. Each is read per tick and rated.
   * The rating IS the reward: delta * rating, summed. Dense by construction,
   * which is what made the difference between 0.06 and 0.94 in measurement.
   */
  props: Record<string, PropSpec>;

  /** Step 2: named boolean readings derived from props. */
  conditions: Record<string, ConditionFn>;

  /** Step 3: the action dict. The ONLY place with side effects. */
  actions: Record<string, ActionFn>;

  /** Step 3: the starting rules. Evolution improves on these. */
  rules: Rule[];

  /** Optional explicit priority order; defaults to declaration order. */
  order?: number[];
}

export interface PropSpec {
  /** reward weight per unit change: +100 for hp, -5 for ammo spent, etc. */
  rating: number;
  /** read the current value from the target system */
  read: () => number | Promise<number>;
  /** for the UI meter */
  min?: number;
  max?: number;
}

export type ConditionFn = (props: StateProps, prev: StateProps | null) => boolean;

export type ActionFn = () => void | Promise<void>;

/** Build the RuleSet the evaluator consumes. */
export function ruleSetOf(p: Project): RuleSet {
  return { rules: p.rules, order: p.order ?? p.rules.map((_, i) => i) };
}

/** The prop -> rating map the TraceRecorder needs. */
export function ratingsOf(p: Project): Record<string, number> {
  return Object.fromEntries(
    Object.entries(p.props).map(([k, v]) => [k, v.rating]),
  );
}

/**
 * Validate a project before running it. Catching a typo'd condition name at
 * startup beats watching a rule that silently never fires.
 */
export function validate(p: Project): string[] {
  const errors: string[] = [];
  if (p.hz <= 0) errors.push(`hz must be positive, got ${p.hz}`);
  if (p.rules.length === 0) errors.push("project has no rules");

  for (const [i, r] of p.rules.entries()) {
    if (!(r.when in p.conditions)) {
      errors.push(`rule ${i} references unknown condition "${r.when}"`);
    }
    if (!(r.then in p.actions)) {
      errors.push(`rule ${i} references unknown action "${r.then}"`);
    }
  }

  const order = p.order ?? p.rules.map((_, i) => i);
  if (order.length !== p.rules.length) {
    errors.push(`order has ${order.length} entries for ${p.rules.length} rules`);
  }
  const seen = new Set(order);
  if (seen.size !== order.length) errors.push("order contains duplicate indices");
  for (const idx of order) {
    if (idx < 0 || idx >= p.rules.length) errors.push(`order index ${idx} out of range`);
  }

  if (Object.keys(p.props).length > 0) {
    const rated = Object.values(p.props).filter((s) => s.rating !== 0);
    if (rated.length === 0) {
      errors.push("no prop has a non-zero rating, so reward is always 0");
    }
  }
  return errors;
}

/**
 * Serialise for the UI / persistence. Functions cannot cross that boundary,
 * so they are represented by name; the UI edits structure and ratings, and
 * code supplies behaviour. This split is deliberate - it keeps the editor
 * simple and keeps side effects in code where they can be reviewed.
 */
export interface ProjectView {
  name: string;
  hz: number;
  props: Record<string, { rating: number; min?: number; max?: number }>;
  conditionNames: string[];
  actionNames: string[];
  rules: Rule[];
  order: number[];
}

export function toView(p: Project): ProjectView {
  return {
    name: p.name,
    hz: p.hz,
    props: Object.fromEntries(
      Object.entries(p.props).map(([k, v]) => [
        k,
        { rating: v.rating, min: v.min, max: v.max },
      ]),
    ),
    conditionNames: Object.keys(p.conditions),
    actionNames: Object.keys(p.actions),
    rules: p.rules,
    order: p.order ?? p.rules.map((_, i) => i),
  };
}

/** Apply UI edits back onto a code-defined project. Structure only. */
export function applyView(p: Project, v: ProjectView): Project {
  const next: Project = { ...p, name: v.name, hz: v.hz, rules: v.rules, order: v.order };
  for (const [k, spec] of Object.entries(v.props)) {
    const existing = p.props[k];
    if (existing) next.props[k] = { ...existing, ...spec };
  }
  return next;
}
