import { describe, expect, test } from "bun:test";
import { evolve } from "../src/evolve/ga.js";
import { applyGenome, genomeOf, mutate, type Genome } from "../src/evolve/genome.js";
import { RewardTable, scoreGenome, splitTraces } from "../src/evolve/score.js";
import type { Rule, TraceTick } from "../src/rules/types.js";

/** Deterministic RNG so a failure is reproducible. */
function lcg(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 0x100000000;
  };
}

const CONDS = ["hp_low", "ammo_low", "enemy", "door"] as const;
const TRUTH: Record<string, string> = {
  hp_low: "strafe",
  ammo_low: "reload",
  enemy: "shoot",
  door: "goto_door",
};
const URGENCY: Record<string, number> = { hp_low: 100, ammo_low: 60, enemy: 40, door: 10 };
const POOL = ["strafe", "reload", "shoot", "goto_door", "wait"];

const RULES: Rule[] = CONDS.map((c) => ({ when: c, then: "wait" }));

/**
 * Generate traces from an environment with a hidden correct mapping.
 * Reward depends on whether the fired rule's action matches TRUTH.
 */
function makeTraces(n: number, rng: () => number): TraceTick[] {
  const ticks: TraceTick[] = [];
  for (let t = 0; t < n; t++) {
    const conditions: Record<string, boolean> = {};
    for (const c of CONDS) conditions[c] = rng() < 0.5;
    const active = CONDS.filter((c) => conditions[c]);
    if (active.length === 0) continue;
    // sample every (condition-set, action) pair so replay has coverage
    const fired = active[Math.floor(rng() * active.length)]!;
    const action = POOL[Math.floor(rng() * POOL.length)]!;
    const reward = action === TRUTH[fired] ? URGENCY[fired]! : -10;
    ticks.push({
      t,
      conditions,
      props: {},
      firedRule: CONDS.indexOf(fired as (typeof CONDS)[number]),
      action,
      reward,
    });
  }
  return ticks;
}

describe("genome", () => {
  test("applyGenome overrides actions and order", () => {
    const g: Genome = { order: [1, 0], actions: ["shoot", "strafe"] };
    const rs = applyGenome([RULES[0]!, RULES[1]!], g);
    expect(rs.order).toEqual([1, 0]);
    expect(rs.rules[0]!.then).toBe("shoot");
  });

  test("mutate changes exactly one thing", () => {
    const rng = lcg(7);
    const g: Genome = { order: [0, 1, 2, 3], actions: ["a", "b", "c", "d"] };
    for (let i = 0; i < 50; i++) {
      const m = mutate(g, { actionPool: POOL, rng });
      const orderChanged = m.order.join() !== g.order.join();
      const actionsChanged = m.actions.join() !== g.actions.join();
      expect(orderChanged && actionsChanged).toBe(false);
    }
  });

  test("mutate never produces an invalid order (permutation preserved)", () => {
    const rng = lcg(11);
    let g: Genome = { order: [0, 1, 2, 3], actions: ["a", "b", "c", "d"] };
    for (let i = 0; i < 200; i++) {
      g = mutate(g, { actionPool: POOL, rng });
      expect([...g.order].sort((a, b) => a - b)).toEqual([0, 1, 2, 3]);
    }
  });
});

describe("scoring", () => {
  test("splitTraces holds out the tail and never overlaps", () => {
    const ticks = makeTraces(100, lcg(1));
    const { train, holdout } = splitTraces(ticks, 0.25);
    expect(train.length + holdout.length).toBe(ticks.length);
    expect(holdout.length).toBeGreaterThan(0);
    expect(train).not.toContain(holdout[0]);
  });

  test("the TRUE mapping outscores a deliberately wrong one", () => {
    const ticks = makeTraces(4000, lcg(3));
    const table = new RewardTable(ticks);
    const good: Genome = {
      order: [0, 1, 2, 3],
      actions: CONDS.map((c) => TRUTH[c]!),
    };
    const bad: Genome = { order: [0, 1, 2, 3], actions: CONDS.map(() => "wait") };
    const gs = scoreGenome(good, RULES, ticks, table);
    const bs = scoreGenome(bad, RULES, ticks, table);
    expect(gs.score).toBeGreaterThan(bs.score);
    // NEGATIVE CONTROL: the all-wrong genome must actually be negative
    expect(bs.score).toBeLessThan(0);
  });

  test("coverage is reported and is non-trivial", () => {
    const ticks = makeTraces(2000, lcg(5));
    const table = new RewardTable(ticks);
    const r = scoreGenome(genomeOf({ rules: RULES, order: [0, 1, 2, 3] }), RULES, ticks, table);
    expect(r.coverage).toBeGreaterThan(0.5);
    expect(r.judged).toBeGreaterThan(0);
  });

  test("coverage is 0 against empty traces, and score does not NaN", () => {
    const r = scoreGenome(
      genomeOf({ rules: RULES, order: [0, 1, 2, 3] }),
      RULES,
      [],
      new RewardTable([]),
    );
    expect(r.coverage).toBe(0);
    expect(r.score).toBe(0);
  });
});

describe("evolution recovers a mapping the user got wrong", () => {
  test("improves on a partly-wrong seed and finds the true actions", () => {
    const rng = lcg(42);
    const ticks = makeTraces(6000, rng);
    // user's proposal: 2 of 4 correct
    const seed: Genome = {
      order: [0, 1, 2, 3],
      actions: ["strafe", "wait", "shoot", "wait"],
    };
    const res = evolve({
      rules: RULES,
      ticks,
      actionPool: POOL,
      seeds: [seed],
      generations: 25,
      rng,
    });

    const seedScore = res.history[0]!.train;
    expect(res.best.train.score).toBeGreaterThan(seedScore);

    // it must have discovered the mappings the user got wrong
    const acts = res.best.genome.actions;
    expect(acts[1]).toBe("reload"); // ammo_low, user said "wait"
    expect(acts[3]).toBe("goto_door"); // door, user said "wait"
    expect(res.overfitting).toBe(false);
  });

  test("history is returned so a stalled search is visible, not inferred", () => {
    const rng = lcg(9);
    const ticks = makeTraces(2000, rng);
    const res = evolve({
      rules: RULES,
      ticks,
      actionPool: POOL,
      seeds: [genomeOf({ rules: RULES, order: [0, 1, 2, 3] })],
      generations: 8,
      rng,
    });
    expect(res.history).toHaveLength(8);
    expect(res.history[0]!.generation).toBe(1);
  });

  test("externally proposed genomes are scored like any mutation", () => {
    const rng = lcg(13);
    const ticks = makeTraces(4000, rng);
    const perfect: Genome = {
      order: [0, 1, 2, 3],
      actions: CONDS.map((c) => TRUTH[c]!),
    };
    const res = evolve({
      rules: RULES,
      ticks,
      actionPool: POOL,
      seeds: [{ order: [0, 1, 2, 3], actions: CONDS.map(() => "wait") }],
      proposals: [perfect],
      generations: 3,
      rng,
    });
    // a correct proposal should win, and its provenance must be preserved
    expect(res.best.genome.actions).toEqual(perfect.actions);
    expect(res.population.some((c) => c.origin === "proposal")).toBe(true);
  });

  test("selection uses training score, so held-out stays unseen", () => {
    const rng = lcg(21);
    const ticks = makeTraces(3000, rng);
    const res = evolve({
      rules: RULES,
      ticks,
      actionPool: POOL,
      seeds: [genomeOf({ rules: RULES, order: [0, 1, 2, 3] })],
      generations: 10,
      rng,
    });
    // both numbers must be reported for every survivor
    for (const c of res.population) {
      expect(typeof c.train.score).toBe("number");
      expect(typeof c.holdout.score).toBe("number");
    }
  });
});
