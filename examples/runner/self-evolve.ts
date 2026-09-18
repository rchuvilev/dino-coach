/**
 * Self-evolving mode: run repeatedly, carrying the population and the
 * accumulated traces forward on disk, so each run starts from everything
 * learned so far.
 *
 * This is the user's step 4/5: "run and see how it evolves, saving the
 * findings and updated behavior persistently between runs + learn to evolve
 * every run."
 *
 * The persistence unit is the POPULATION, not just the best genome -
 * diversity is what enables later improvement. Also persisted: per-genome
 * provenance (which run found it, its scores) and the growing evaluation set.
 */
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { evolve } from "../../src/evolve/ga.js";
import { applyGenome, genomeOf, type Genome } from "../../src/evolve/genome.js";
import { assessAgainstControls, verifyLiveness } from "../../src/evolve/liveness.js";
import { RuleEvaluator } from "../../src/rules/evaluator.js";
import type { Rule, TraceTick } from "../../src/rules/types.js";
import { perceive } from "./perceive.js";
import { render, RunnerWorld, type RunnerAction } from "./world.js";

const STATE_PATH = join(import.meta.dir, "../../.data/self-evolve.json");

const ACTIONS: RunnerAction[] = ["jump", "duck", "run"];

const RULES: Rule[] = [
  { when: "obstacle_imminent", then: "jump" },
  { when: "high_obstacle", then: "duck" },
  { when: "obstacle_near", then: "jump" },
  { when: "always", then: "run" },
];

const TH = { windowLo: 4, windowHi: 12, duckAt: 12 };

function conditionsFrom(p: ReturnType<typeof perceive>): Record<string, boolean> {
  const has = p.gap < 999;
  return {
    obstacle_imminent:
      has && !p.high && p.gap >= TH.windowLo && p.gap <= TH.windowHi && !p.airborne,
    high_obstacle: has && p.high && p.gap <= TH.duckAt,
    obstacle_near: has && p.gap <= TH.windowHi + 10,
    always: true,
  };
}

function episode(
  genome: Genome,
  seed: number,
  maxTicks: number,
  collect?: TraceTick[],
  explore = 0,
  rand: () => number = Math.random,
): number {
  const world = new RunnerWorld(seed);
  const ev = new RuleEvaluator(applyGenome(RULES, genome));
  for (let t = 0; t < maxTicks; t++) {
    const conditions = conditionsFrom(perceive(render(world)));
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

const score = (g: Genome, seeds: number[], maxTicks = 6000): number =>
  Math.round(seeds.reduce((sum, s) => sum + episode(g, s, maxTicks), 0) / seeds.length);

interface PersistedState {
  run: number;
  population: { genome: Genome; foundInRun: number; holdout: number }[];
  traces: TraceTick[];
  history: { run: number; bestHoldout: number; popSize: number; traceCount: number }[];
}

function load(): PersistedState {
  if (existsSync(STATE_PATH)) {
    return JSON.parse(readFileSync(STATE_PATH, "utf8")) as PersistedState;
  }
  return { run: 0, population: [], traces: [], history: [] };
}

function save(s: PersistedState): void {
  mkdirSync(dirname(STATE_PATH), { recursive: true });
  writeFileSync(STATE_PATH, JSON.stringify(s));
}

// --- seeds: train rotates per run, held-out is FIXED forever --------------
// A held-out set that changes between runs is not held out; it would let a
// lucky seed look like progress.
const HOLDOUT = [101, 211, 307, 401, 503, 601, 701, 809, 907, 1009];
const trainSeedsFor = (run: number) =>
  [0, 1, 2, 3, 4].map((i) => 11 + ((run * 5 + i) * 37) % 900);

const runs = Number(process.argv[2] ?? 3);
const state = load();

// --- liveness gate, every run --------------------------------------------
{
  let probe = new RunnerWorld(1);
  const live = await verifyLiveness({
    reset: () => void (probe = new RunnerWorld(1)),
    step: (a) => void probe.step(a as RunnerAction),
    observe: () => ({ d: probe.distance, y: Math.round(probe.y), n: probe.obstacles.length }),
    progress: (s) => s.d,
    probeAction: "jump",
    idleAction: "run",
  });
  if (!live.alive) {
    console.error(`liveness FAILED: ${live.reasons.join("; ")}`);
    process.exit(1);
  }
  console.log(`liveness OK (advances=${live.advances}, inputWorks=${live.actionHasEffect})`);
}

const seedGenome = genomeOf({ rules: RULES, order: [0, 1, 2, 3] });
const controls = {
  doNothing: score({ order: [0, 1, 2, 3], actions: ["run", "run", "run", "run"] }, HOLDOUT),
  alwaysJump: score({ order: [3, 0, 1, 2], actions: ["jump", "jump", "jump", "jump"] }, HOLDOUT),
};
console.log(`controls on held-out: doNothing=${controls.doNothing} alwaysJump=${controls.alwaysJump}`);
console.log(`human seed rules    : ${score(seedGenome, HOLDOUT)}\n`);

let rs = 987654321;
const rand = () => {
  rs = (rs * 1103515245 + 12345) & 0x7fffffff;
  return rs / 0x7fffffff;
};

for (let i = 0; i < runs; i++) {
  state.run++;
  const train = trainSeedsFor(state.run);

  // seed the population from persisted genomes, plus the human rules
  const seeds: Genome[] =
    state.population.length > 0 ? state.population.map((p) => p.genome) : [seedGenome];

  // collect fresh traces with exploration, appended to the accumulated set
  const best = state.population[0]?.genome ?? seedGenome;
  for (const s of train) episode(best, s, 6000, state.traces, 0.12, rand);
  // bound the trace store so a long-lived run does not grow without limit
  if (state.traces.length > 20000) state.traces = state.traces.slice(-20000);

  const res = evolve({
    rules: RULES,
    ticks: state.traces,
    actionPool: ACTIONS,
    seeds,
    generations: 20,
    survivors: 8,
    rng: rand,
  });

  // LIVE scoring decides; replay only shortlists (replay is off-policy).
  //
  // ELITISM IS MANDATORY HERE. Measured defect: replay-based selection
  // ELIMINATED the champion every run. All 8 survivors scored an identical
  // 0.90 on replay while the champion - worth 2030 live - was culled before
  // live scoring ever saw it. Replay cannot rank policies (a genome scoring
  // 15.93 on held-out replay earned 0 live), so it must never be allowed to
  // DISCARD a genome that live evaluation has already validated.
  const carried = state.population.slice(0, 4).map((p) => p.genome);
  const pool = [...carried, ...res.population.map((c) => c.genome)];
  const poolSeen = new Set<string>();
  const ranked = pool
    .filter((g) => {
      const k = `${g.order.join(",")}|${g.actions.join(",")}`;
      if (poolSeen.has(k)) return false;
      poolSeen.add(k);
      return true;
    })
    .map((g) => ({ g, train: score(g, train) }))
    .sort((a, b) => b.train - a.train);

  const champion = ranked[0]!.g;
  const holdout = score(champion, HOLDOUT);

  // Persist the whole SHORTLIST, not just the champion.
  // Measured defect: keeping only the champion collapsed the population to 2
  // genomes by run 3, so each run restarted the GA from almost nothing and
  // re-derived from scratch (champions of 145 against run 1's 2030). The
  // stored population IS the search state; storing one genome throws the
  // search away and keeps only its answer.
  const fresh = ranked.slice(0, 6).map((r) => ({
    genome: r.g,
    foundInRun: state.run,
    holdout: r.g === champion ? holdout : score(r.g, HOLDOUT),
  }));
  const seen = new Set<string>();
  state.population = fresh
    .concat(state.population)
    .filter((p) => {
      const k = `${p.genome.order.join(",")}|${p.genome.actions.join(",")}`;
      if (seen.has(k)) return false;
      seen.add(k);
      return true;
    })
    .sort((a, b) => b.holdout - a.holdout)
    .slice(0, 10);

  state.history.push({
    run: state.run,
    bestHoldout: state.population[0]!.holdout,
    popSize: state.population.length,
    traceCount: state.traces.length,
  });
  save(state);

  const a = assessAgainstControls(state.population[0]!.holdout, controls);
  console.log(
    `run ${String(state.run).padStart(2)}: trainBest=${String(ranked[0]!.train).padStart(5)} ` +
      `holdout=${String(holdout).padStart(5)} bestEver=${String(state.population[0]!.holdout).padStart(5)} ` +
      `pop=${state.population.length} traces=${state.traces.length}`,
  );
  if (!a.valid) console.log(`         ${a.verdict}`);
}

const top = state.population[0]!;
console.log(`\n=== after ${state.run} cumulative run(s) ===`);
console.log(`best held-out: ${top.holdout} (found in run ${top.foundInRun})`);
const av = assessAgainstControls(top.holdout, controls);
console.log(`${av.verdict}`);
console.log(`\nrules (priority order):`);
for (const idx of top.genome.order) {
  const r = RULES[idx]!;
  const act = top.genome.actions[idx]!;
  console.log(`  ${r.when} => ${act}${act !== r.then ? `  <- was "${r.then}"` : ""}`);
}
console.log(`\nprogress across runs: ${state.history.map((h) => h.bestHoldout).join(" -> ")}`);
console.log(`state: ${STATE_PATH}`);
