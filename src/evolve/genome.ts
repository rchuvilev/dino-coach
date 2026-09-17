import type { Rule, RuleSet } from "../rules/types.js";

/**
 * A genome is the evolvable part of a policy: the priority ORDER and the
 * condition -> action MAP. Everything else (commit windows, cooldowns) is
 * authored in code and left alone.
 *
 * Discrete search space, so no gradient exists - a GA/hill-climb is the
 * correct tool, and it stays readable, which matters because the user is
 * meant to inspect what evolved.
 */
export interface Genome {
  order: number[];
  /** ruleIndex -> action name, overriding Rule.then */
  actions: string[];
}

export function genomeOf(rs: RuleSet): Genome {
  return { order: [...rs.order], actions: rs.rules.map((r) => r.then) };
}

export function applyGenome(rules: Rule[], g: Genome): RuleSet {
  return {
    rules: rules.map((r, i) => ({ ...r, then: g.actions[i] ?? r.then })),
    order: [...g.order],
  };
}

export function cloneGenome(g: Genome): Genome {
  return { order: [...g.order], actions: [...g.actions] };
}

export function genomeKey(g: Genome): string {
  return `${g.order.join(",")}|${g.actions.join(",")}`;
}

export interface MutateOptions {
  /** every action name the mutator may choose from */
  actionPool: string[];
  /** probability of mutating the order rather than an action */
  orderRate?: number;
  rng?: () => number;
}

/** One random change: swap two priorities, or reassign one rule's action. */
export function mutate(g: Genome, opts: MutateOptions): Genome {
  const rng = opts.rng ?? Math.random;
  const next = cloneGenome(g);
  const orderRate = opts.orderRate ?? 0.5;

  if (next.order.length > 1 && rng() < orderRate) {
    const i = Math.floor(rng() * next.order.length);
    let j = Math.floor(rng() * next.order.length);
    if (j === i) j = (i + 1) % next.order.length;
    [next.order[i], next.order[j]] = [next.order[j]!, next.order[i]!];
  } else if (next.actions.length > 0 && opts.actionPool.length > 0) {
    const i = Math.floor(rng() * next.actions.length);
    const a = opts.actionPool[Math.floor(rng() * opts.actionPool.length)];
    if (a !== undefined) next.actions[i] = a;
  }
  return next;
}

/** Order-preserving crossover: take order from a, actions from b. */
export function crossover(a: Genome, b: Genome): Genome {
  return { order: [...a.order], actions: [...b.actions] };
}
