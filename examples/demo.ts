/**
 * End-to-end demo: a code-first project, run headless, then evolved.
 *
 * The "target system" here is a tiny simulated shooter so the demo is
 * self-contained. Swap `world` for real sensors and the rest is unchanged.
 */
import { Agent } from "../src/kernel/agent.js";
import { evolve } from "../src/evolve/ga.js";
import { genomeOf } from "../src/evolve/genome.js";
import { rankLive } from "../src/evolve/live.js";
import { ruleSetOf, type Project } from "../src/project.js";

// --- the target system -------------------------------------------------
const world = {
  hp: 100,
  ammo: 30,
  /** ammo reserve; refilled slowly so the episode never flatlines */
  reserve: 60,
  enemyHp: 100,
  distance: 10,
  tick: 0,
  step(action: string | null) {
    this.tick++;
    // enemy shoots at you when close
    if (this.distance < 8 && this.enemyHp > 0) this.hp -= 3;
    switch (action) {
      case "strafe":
        this.distance = Math.min(20, this.distance + 2);
        this.hp = Math.min(100, this.hp + 1);
        break;
      case "shoot":
        if (this.ammo > 0 && this.distance < 12) {
          this.ammo -= 1;
          this.enemyHp -= 12;
        }
        break;
      case "reload":
        // finite reserve, so "reload forever" is not a free win
        if (this.reserve > 0) {
          const take = Math.min(6, this.reserve, 30 - this.ammo);
          this.ammo += take;
          this.reserve -= take;
        }
        break;
      case "advance":
        this.distance = Math.max(1, this.distance - 2);
        break;
    }
    if (this.enemyHp <= 0) {
      this.enemyHp = 100;
      this.distance = 14;
      // killing an enemy yields pickups, so the run stays alive indefinitely
      this.reserve = Math.min(60, this.reserve + 18);
      this.hp = Math.min(100, this.hp + 10);
    }
  },
};

// --- step 1-3: the project, written in code ---------------------------
let pendingAction: string | null = null;

const project: Project = {
  name: "demo-shooter",
  hz: 30,

  // step 1: state props + ratings. The rating IS the reward weight.
  props: {
    hp: { rating: 10, read: () => world.hp, min: 0, max: 100, maxDelta: 50 },
    ammo: { rating: 1, read: () => world.ammo, min: 0, max: 30 },
    // only credit DECREASES: the 0->100 respawn is a discontinuity, not a
    // consequence. Without this the rating punishes killing (-500/kill).
    enemyHp: {
      rating: -5,
      read: () => world.enemyHp,
      min: 0,
      max: 100,
      countDirection: "down",
      maxDelta: 50,
    },
  },

  // step 2: conditions derived from props
  conditions: {
    hp_dropping: (p, prev) => prev !== null && p.hp! < prev.hp!,
    ammo_low: (p) => p.ammo! < 8,
    enemy_close: (p) => world.distance < 12,
    always: () => true,
  },

  // step 3: the action dict - the only side effects
  actions: {
    strafe: () => void (pendingAction = "strafe"),
    shoot: () => void (pendingAction = "shoot"),
    reload: () => void (pendingAction = "reload"),
    advance: () => void (pendingAction = "advance"),
  },

  // step 3: proposed starting rules, deliberately imperfect
  rules: [
    { when: "hp_dropping", then: "strafe", commitTicks: 6 },
    { when: "ammo_low", then: "strafe", cooldownTicks: 4 }, // wrong on purpose
    { when: "enemy_close", then: "shoot" },
    { when: "always", then: "advance" },
  ],
};

// --- step 4: run, then evolve -----------------------------------------
async function run(label: string, p: Project, ticks: number, explore = 0) {
  // full reset: a missed field silently carries state between runs and makes
  // run 2 look better or worse than it is
  Object.assign(world, {
    hp: 100,
    ammo: 30,
    reserve: 60,
    enemyHp: 100,
    distance: 10,
    tick: 0,
  });
  // explore during the learning run: without alternatives in the trace the
  // GA has nothing to compare against (measured: 0 coverage for rivals)
  const agent = new Agent({ project: p, runId: label, explore });
  for (let i = 0; i < ticks; i++) {
    await agent.tick();
    world.step(pendingAction);
    pendingAction = null;
  }
  return agent;
}

const TICKS = 1500;
const first = await run("run-1", project, TICKS, 0.15);
console.log(`\n=== run 1 (user's proposed rules) ===`);
console.log(`total reward: ${first.recorder.totalReward().toFixed(0)}`);
console.log(`final hp=${world.hp} ammo=${world.ammo}`);

const traces = first.recorder.episode().ticks;
const res = evolve({
  rules: project.rules,
  ticks: traces,
  actionPool: Object.keys(project.actions),
  seeds: [genomeOf(ruleSetOf(project))],
  generations: 20,
  // keep a wider, diverse shortlist: the live tier needs candidates to TEST,
  // and replay ranking is too biased to trust for a top-1 pick
  survivors: 8,
});

console.log(`\n=== evolution (${traces.length} recorded ticks) ===`);
console.log(`replay prefilter: train ${res.history[0]!.train.toFixed(2)} -> ${res.best.train.score.toFixed(2)}, coverage ${(res.best.train.coverage * 100).toFixed(0)}%`);

// Replay is OFF-POLICY: it cannot know which states a policy would reach.
// Measured, a genome scoring 15.93 on replay earned 0 live. So replay only
// SHORTLISTS; the live run decides.
const shortlist = res.population.slice(0, 5).map((c) => c.genome);
const reset = () =>
  void Object.assign(world, {
    hp: 100, ammo: 30, reserve: 60, enemyHp: 100, distance: 10, tick: 0,
  });
const stepWorld = () => {
  world.step(pendingAction);
  pendingAction = null;
};
const ranked = await rankLive([genomeOf(ruleSetOf(project)), ...shortlist], {
  project, reset, stepWorld, ticks: TICKS,
});
console.log(`\nlive evaluation of ${ranked.length} candidates (the real scorer):`);
for (const r of ranked) console.log(`  ${r.totalReward.toFixed(0)}`);

const evolved = ranked[0]!.genome;
console.log(`\nbest rules (priority order):`);
for (const idx of evolved.order) {
  const r = project.rules[idx]!;
  const action = evolved.actions[idx]!;
  console.log(`  ${r.when} => ${action}${action !== r.then ? `  <- was "${r.then}"` : ""}`);
}

// --- step 5: re-run with the evolved policy ---------------------------
const tuned: Project = { ...project, order: evolved.order, rules: project.rules.map((r, i) => ({ ...r, then: evolved.actions[i] ?? r.then })) };
const second = await run("run-2", tuned, TICKS);
console.log(`\n=== run 2 (evolved rules) ===`);
console.log(`total reward: ${second.recorder.totalReward().toFixed(0)}`);
console.log(`final hp=${world.hp} ammo=${world.ammo}`);

const delta = second.recorder.totalReward() - first.recorder.totalReward();
console.log(`\ndelta: ${delta > 0 ? "+" : ""}${delta.toFixed(0)}  ${delta > 0 ? "IMPROVED" : "no improvement"}`);
