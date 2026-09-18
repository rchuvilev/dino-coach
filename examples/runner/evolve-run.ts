/**
 * The real experiment: can evolution improve on a human's first-guess rules,
 * judged by LIVE episodes on a target read through PIXELS?
 *
 * Everything the earlier attempts lacked is wired in here:
 *  - live scoring (replay is off-policy and scored a 0-reward genome at 15.93)
 *  - negative controls (they are what exposed the frozen-world fake score)
 *  - held-out seeds the search never trains on
 *  - the agent sees only rendered frames, never world state
 */
import { evolve } from "../../src/evolve/ga.js";
import { assessAgainstControls, verifyLiveness } from "../../src/evolve/liveness.js";
import { applyGenome, genomeOf, type Genome } from "../../src/evolve/genome.js";
import type { Rule, TraceTick } from "../../src/rules/types.js";
import { RuleEvaluator } from "../../src/rules/evaluator.js";
import { perceive } from "./perceive.js";
import { render, RunnerWorld, type RunnerAction } from "./world.js";

const CONDITIONS = ["obstacle_imminent", "high_obstacle", "obstacle_near", "always"] as const;
const ACTIONS: RunnerAction[] = ["jump", "duck", "run"];

/** A human's plausible first guess, with guessed thresholds. */
const SEED_RULES: Rule[] = [
  { when: "obstacle_imminent", then: "jump" },
  { when: "high_obstacle", then: "duck" },
  { when: "obstacle_near", then: "jump" },
  { when: "always", then: "run" },
];

// A WINDOW, not a ceiling, and the bounds are MEASURED not reasoned.
// A `gap <= N` ceiling fires continuously from N down to 0 and takes off far
// too early; that single mistake made an ORACLE score 129, BELOW the
// do-nothing control, and looked exactly like broken world geometry.
// Oracle sweep over windows (mean distance, do-nothing = 129):
//   6..16 -> 146    8..14 -> 147    2..8  -> 648
//   2..10 -> 1340   4..10 -> 1340
//   4..12 -> 2184   6..12 -> 2184   <- 17x the control
// The crash that exposed it: player bottom 33.1 vs obstacle top 33.0, i.e.
// overlapping by 0.1px while descending 3px/tick - it had peaked too early.
const TH = { windowLo: 4, windowHi: 12, duckAt: 12 };

function conditionsFrom(p: ReturnType<typeof perceive>): Record<string, boolean> {
  const has = p.gap < 999;
  return {
    // in the takeoff window AND on the ground: the only moment a jump helps
    obstacle_imminent: has && !p.high && p.gap >= TH.windowLo && p.gap <= TH.windowHi && !p.airborne,
    high_obstacle: has && p.high && p.gap <= TH.duckAt,
    obstacle_near: has && p.gap <= TH.windowHi + 10,
    always: true,
  };
}

/** Run one episode. The agent reads ONLY pixels. */
function episode(
  genome: Genome,
  seed: number,
  maxTicks: number,
  collect?: TraceTick[],
  explore = 0,
  rand: () => number = Math.random,
): number {
  const world = new RunnerWorld(seed);
  const rs = applyGenome(SEED_RULES, genome);
  const ev = new RuleEvaluator(rs);

  for (let t = 0; t < maxTicks; t++) {
    const p = perceive(render(world));
    const conditions = conditionsFrom(p);
    const d = ev.step(conditions, false);
    let action = (d.action ?? "run") as RunnerAction;

    if (explore > 0 && rand() < explore) {
      action = ACTIONS[Math.floor(rand() * ACTIONS.length)]!;
    }

    const before = world.distance;
    const alive = world.step(action);
    if (collect) {
      collect.push({
        t,
        conditions,
        props: { distance: world.distance },
        firedRule: d.firedRule,
        action,
        reward: alive ? world.distance - before : -50,
      });
    }
    if (!alive) break;
  }
  return world.distance;
}

/** Mean distance over several seeds - a single seed is luck, not skill. */
function scoreOn(genome: Genome, seeds: number[], maxTicks = 4000): number {
  let sum = 0;
  for (const s of seeds) sum += episode(genome, s, maxTicks);
  return Math.round(sum / seeds.length);
}

const TRAIN_SEEDS = [11, 23, 37, 41, 59];
const HOLDOUT_SEEDS = [101, 211, 307, 401, 503];

const seedGenome = genomeOf({ rules: SEED_RULES, order: [0, 1, 2, 3] });

// --- negative controls: these MUST score far below a working policy -------
const doNothing: Genome = { order: [0, 1, 2, 3], actions: ["run", "run", "run", "run"] };
const alwaysJump: Genome = { order: [3, 0, 1, 2], actions: ["jump", "jump", "jump", "jump"] };
const alwaysDuck: Genome = { order: [3, 0, 1, 2], actions: ["duck", "duck", "duck", "duck"] };

// --- liveness gate: prove the target runs AND responds before scoring ----
// Not optional. Skipping this produced a plausible 8361 score from a target
// that was not running, and three downstream wrong conclusions.
{
  let probe = new RunnerWorld(1);
  const live = await verifyLiveness({
    reset: () => {
      probe = new RunnerWorld(1);
    },
    step: (a) => void probe.step(a as RunnerAction),
    observe: () => ({ d: probe.distance, y: Math.round(probe.y), n: probe.obstacles.length }),
    progress: (s) => s.d,
    probeAction: "jump",
    idleAction: "run",
  });
  console.log(`=== liveness: ${live.alive ? "OK" : "FAILED"} (advances=${live.advances}, inputWorks=${live.actionHasEffect}, states=${live.distinctStates}) ===`);
  if (!live.alive) {
    console.error("target is not usable for evaluation:");
    for (const r of live.reasons) console.error(`  - ${r}`);
    process.exit(1);
  }
}

console.log("\n=== baselines (mean distance over 5 train seeds) ===");
const bDo = scoreOn(doNothing, TRAIN_SEEDS);
const bJump = scoreOn(alwaysJump, TRAIN_SEEDS);
const bDuck = scoreOn(alwaysDuck, TRAIN_SEEDS);
const bSeed = scoreOn(seedGenome, TRAIN_SEEDS);
console.log(`  do-nothing   (control): ${bDo}`);
console.log(`  always-jump  (control): ${bJump}`);
console.log(`  always-duck  (control): ${bDuck}`);
console.log(`  human seed rules      : ${bSeed}`);

{
  const a = assessAgainstControls(bSeed, { doNothing: bDo, alwaysJump: bJump, alwaysDuck: bDuck });
  console.log(`\n  ${a.verdict}`);
}

// --- collect traces with exploration, so replay has alternatives ----------
const traces: TraceTick[] = [];
let rs = 12345;
const rand = () => {
  rs = (rs * 1103515245 + 12345) & 0x7fffffff;
  return rs / 0x7fffffff;
};
for (const s of TRAIN_SEEDS) episode(seedGenome, s, 4000, traces, 0.12, rand);
console.log(`\n=== collected ${traces.length} ticks (explore=0.12) ===`);

// --- GA prefilter on replay, then LIVE evaluation -------------------------
const res = evolve({
  rules: SEED_RULES,
  ticks: traces,
  actionPool: ACTIONS,
  seeds: [seedGenome],
  generations: 25,
  survivors: 10,
  rng: rand,
});
console.log(`replay prefilter: coverage ${(res.best.train.coverage * 100).toFixed(0)}%, overfitting=${res.overfitting}`);

const candidates = [seedGenome, ...res.population.map((c) => c.genome)];
const seen = new Set<string>();
const unique = candidates.filter((g) => {
  const k = `${g.order.join(",")}|${g.actions.join(",")}`;
  if (seen.has(k)) return false;
  seen.add(k);
  return true;
});
console.log(`\n=== live evaluation of ${unique.length} unique candidates ===`);

const ranked = unique
  .map((g) => ({ g, train: scoreOn(g, TRAIN_SEEDS) }))
  .sort((a, b) => b.train - a.train);

for (const r of ranked.slice(0, 5)) {
  const isSeed = r.g === seedGenome;
  console.log(`  train ${String(r.train).padStart(5)}  ${r.g.actions.join(",")}  [${r.g.order.join(",")}]${isSeed ? "  <- seed" : ""}`);
}

// --- the honest number: held-out seeds the search never saw --------------
const best = ranked[0]!;
const seedHold = scoreOn(seedGenome, HOLDOUT_SEEDS);
const bestHold = scoreOn(best.g, HOLDOUT_SEEDS);

console.log(`\n=== HELD-OUT seeds (never trained on) ===`);
console.log(`  human seed rules : ${seedHold}`);
console.log(`  evolved best     : ${bestHold}`);
const delta = bestHold - seedHold;
console.log(`  delta: ${delta > 0 ? "+" : ""}${delta} (${((bestHold / seedHold - 1) * 100).toFixed(1)}%)`);
console.log(
  delta > 0
    ? "  => evolution IMPROVED on the human's rules"
    : "  => no improvement on held-out seeds",
);
