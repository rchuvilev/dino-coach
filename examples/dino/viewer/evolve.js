/**
 * Unbounded evolution on the REAL dino game.
 *
 * Runs forever until stopped, learns from every crash, and persists the
 * population to localStorage so a reload continues rather than restarts.
 *
 * Invariants carried over from the synthetic target, each one earned by a
 * measured failure:
 *  - ELITISM is mandatory. Selection once culled a champion worth 2030 live
 *    because every rival tied on a replay score. A genome that live
 *    evaluation has validated is never discarded.
 *  - Live scoring decides. Replay is off-policy and cannot rank policies.
 *  - Episodes are scored as a DELTA. r.restart() does NOT zero distanceRan
 *    in this build (measured 9196 right after a restart), so absolute
 *    readings carry over and a do-nothing control "scores" 14346.
 *  - Controls run in the same rotation. A policy that cannot beat
 *    do-nothing means the metric is broken, not the policy.
 */

const STORE_KEY = "dino-evolve-v1";

// ---- genome -------------------------------------------------------------
// Window bounds are EVOLVED, not guessed. Measured geometry: obstacles close
// at ~6px/frame (366 distance per 60 frames at speed 6.14, ratio 0.99 vs the
// game's own expectation) and the jump rise takes ~10 frames, so a usable
// takeoff window sits around 40..100. A 35px window gives only ~6 usable
// frames and whole obstacles pass undecided.
const LO_MIN = 15, LO_MAX = 55;   // takeoff must start inside the jump arc
const HI_MAX = 110;               // beyond this the tRex lands ON the obstacle

function clampGenome(g) {
  g.lo = Math.min(LO_MAX, Math.max(LO_MIN, g.lo));
  g.hi = Math.min(HI_MAX, Math.max(g.lo + 10, g.hi));
  g.duck = Math.min(90, Math.max(5, g.duck));
  return g;
}

function randomGenome(rnd) {
  const lo = Math.round(LO_MIN + rnd() * (LO_MAX - LO_MIN));
  return clampGenome({
    lo,
    hi: lo + Math.round(20 + rnd() * 60),
    duck: Math.round(20 + rnd() * 60),
    // rule order: which check wins when several match
    duckFirst: rnd() < 0.5,
  });
}

function mutate(g, rnd) {
  const n = { ...g };
  const pick = Math.floor(rnd() * 4);
  const nudge = () => Math.round((rnd() - 0.5) * 30);
  if (pick === 0) n.lo = Math.max(0, n.lo + nudge());
  else if (pick === 1) n.hi = Math.max(n.lo + 10, n.hi + nudge());
  else if (pick === 2) n.duck = Math.max(5, n.duck + nudge());
  else n.duckFirst = !n.duckFirst;
  return clampGenome(n);
}

const key = (g) => `${g.lo}|${g.hi}|${g.duck}|${g.duckFirst ? 1 : 0}`;
const label = (g) => `${g.lo}..${g.hi} duck<=${g.duck}${g.duckFirst ? " D1st" : ""}`;

// fixed controls, always evaluated in the rotation
const CONTROLS = [
  { ctrl: "do-nothing", lo: -1, hi: -1, duck: -1, duckFirst: false },
  { ctrl: "always-jump", lo: 0, hi: 99998, duck: -1, duckFirst: false },
];

// ---- rng ----------------------------------------------------------------
let seed = Date.now() & 0x7fffffff;
const rnd = () => {
  seed = (seed * 1103515245 + 12345) & 0x7fffffff;
  return seed / 0x7fffffff;
};

// ---- state --------------------------------------------------------------
function blank() {
  return {
    version: 1,
    sessions: 0,
    startedAt: Date.now(),
    generation: 0,
    episodes: 0,
    population: [], // {g, runs, total, best, mean}
    history: [], // best mean per generation
    bestEver: 0,
    bestGenome: null,
  };
}

function load() {
  try {
    const raw = localStorage.getItem(STORE_KEY);
    if (raw) return JSON.parse(raw);
  } catch {
    /* storage may be unavailable on some schemes */
  }
  return blank();
}

function save(s) {
  try {
    localStorage.setItem(STORE_KEY, JSON.stringify(s));
  } catch {
    /* non-fatal: evolution continues in memory */
  }
}

let S = load();
// count this page load as a session, so progress across sessions is visible
S.sessions = (S.sessions || 0) + 1;
S.lastOpened = Date.now();
save(S);

const POP = 6; // evaluated per generation, plus 2 controls
const EPISODES_PER = 3; // episodes averaged per candidate - one run is luck
const ELITE = 3;

function seedPopulation() {
  const out = [];
  // resume a generation that was interrupted mid-way
  if (S.inProgress && S.inProgress.length) {
    for (const c of S.inProgress) {
      out.push({ g: c.g, runs: c.runs || 0, total: c.total || 0, best: c.best || 0, mean: c.mean || 0, elite: true });
    }
    S.inProgress = null;
    while (out.length < POP) {
      const parent = out[Math.floor(rnd() * out.length)].g;
      const g = mutate(parent, rnd);
      if (!out.some((c) => key(c.g) === key(g))) out.push({ g, runs: 0, total: 0, best: 0, mean: 0 });
    }
    for (const c of CONTROLS) out.push({ g: c, runs: 0, total: 0, best: 0, mean: 0, ctrl: c.ctrl });
    return out;
  }
  // carry elites forward - this is the persistence that makes it cumulative
  for (const e of (S.population || []).slice(0, ELITE)) {
    out.push({ g: e.g, runs: 0, total: 0, best: 0, mean: 0, elite: true });
  }
  while (out.length < POP) {
    const parent = out.length && rnd() < 0.7 ? out[Math.floor(rnd() * out.length)].g : null;
    const g = parent ? mutate(parent, rnd) : randomGenome(rnd);
    if (out.some((c) => key(c.g) === key(g))) continue;
    out.push({ g, runs: 0, total: 0, best: 0, mean: 0 });
  }
  for (const c of CONTROLS) out.push({ g: c, runs: 0, total: 0, best: 0, mean: 0, ctrl: c.ctrl });
  return out;
}

export function currentState() {
  return S;
}

export function resetAll() {
  S = blank();
  save(S);
}

export { seedPopulation, mutate, randomGenome, key, label, ELITE, EPISODES_PER, rnd, clampGenome, save };

export function recordEpisode(cand, dist, pop) {
  cand.runs++;
  cand.total += dist;
  cand.best = Math.max(cand.best, dist);
  cand.mean = Math.round(cand.total / cand.runs);
  S.episodes++;
  if (!cand.ctrl && dist > S.bestEver) {
    S.bestEver = dist;
    S.bestGenome = cand.g;
  }
  // Persist EVERY episode. Saving only at generation close meant a session
  // stopped mid-generation lost all of it - measured 6 episodes and a best of
  // 15382 while storage still read gen 0 / ep 0 / best 0.
  if (pop) {
    S.inProgress = pop
      .filter((c) => !c.ctrl)
      .map((c) => ({ g: c.g, runs: c.runs, total: c.total, best: c.best, mean: c.mean }));
  }
  save(S);
}

/**
 * Close a generation: rank by LIVE mean, keep elites, record history.
 * Controls are ranked alongside but never become parents.
 */
export function closeGeneration(pop) {
  const evolved = pop.filter((c) => !c.ctrl).sort((a, b) => b.mean - a.mean);
  const controls = pop.filter((c) => c.ctrl);
  const bestCtrl = Math.max(0, ...controls.map((c) => c.mean));
  S.generation++;
  S.population = evolved.slice(0, ELITE).map((c) => ({ g: c.g, mean: c.mean, best: c.best }));
  S.history.push({
    gen: S.generation,
    best: evolved[0] ? evolved[0].mean : 0,
    control: bestCtrl,
    episodes: S.episodes,
  });
  if (S.history.length > 200) S.history = S.history.slice(-200);
  save(S);
  return { best: evolved[0], bestCtrl };
}
