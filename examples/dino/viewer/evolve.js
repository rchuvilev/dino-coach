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
  g.loA = Math.min(LO_MAX, Math.max(LO_MIN, g.loA));
  g.loB = Math.max(-8, Math.min(8, g.loB));
  g.widthA = Math.min(HI_MAX - LO_MIN, Math.max(10, g.widthA));
  g.widthB = Math.max(-8, Math.min(8, g.widthB));
  g.duck = Math.min(90, Math.max(5, g.duck));
  return g;
}

function randomGenome(rnd) {
  return clampGenome({
    loA: Math.round(LO_MIN + rnd() * (LO_MAX - LO_MIN)),
    loB: +((rnd() - 0.5) * 8).toFixed(2),
    widthA: Math.round(20 + rnd() * 60),
    widthB: +((rnd() - 0.5) * 8).toFixed(2),
    duck: Math.round(20 + rnd() * 60),
    // rule order: which check wins when several match
    duckFirst: rnd() < 0.5,
  });
}

/** Resolve the genome at a given speed. This is the whole of option A. */
export function windowAt(g, speed) {
  const d = (speed || 6) - 6;
  let lo = g.loA + g.loB * d;
  let width = g.widthA + g.widthB * d;
  // A NaN here means a genome of the wrong shape reached the policy, which
  // reads as "never jump" and is indistinguishable from a bad strategy.
  // Fail loudly-ish: fall back to a known-viable window instead.
  if (!Number.isFinite(lo) || !Number.isFinite(width)) {
    lo = 30;
    width = 30;
  }
  lo = Math.min(LO_MAX, Math.max(LO_MIN, lo));
  width = Math.min(HI_MAX - lo, Math.max(10, width));
  return { lo, hi: lo + width };
}

function mutate(g, rnd) {
  const n = { ...g };
  const pick = Math.floor(rnd() * 5);
  const nudge = () => Math.round((rnd() - 0.5) * 30);
  const slope = () => +((rnd() - 0.5) * 4).toFixed(2);
  if (pick === 0) n.loA = n.loA + nudge();
  else if (pick === 1) n.widthA = n.widthA + nudge();
  else if (pick === 2) n.duck = n.duck + nudge();
  else if (pick === 3) n.duckFirst = !n.duckFirst;
  else {
    // mutate a SLOPE: this is the new axis option A adds
    if (rnd() < 0.5) n.loB = +(n.loB + slope()).toFixed(2);
    else n.widthB = +(n.widthB + slope()).toFixed(2);
  }
  return clampGenome(n);
}

const key = (g) =>
  `${g.loA}|${g.loB}|${g.widthA}|${g.widthB}|${g.duck}|${g.duckFirst ? 1 : 0}`;
const label = (g) => {
  if (g.ctrlLabel) return g.ctrlLabel;
  const sgn = (v) => (v >= 0 ? `+${v}` : `${v}`);
  return `lo ${g.loA}${sgn(g.loB)}v · w ${g.widthA}${sgn(g.widthB)}v · d${g.duck}${g.duckFirst ? " D1st" : ""}`;
};

// fixed controls, always evaluated in the rotation
const CONTROLS = [
  { ctrl: "do-nothing", never: true, duck: -1, duckFirst: false, ctrlLabel: "do-nothing" },
  { ctrl: "always-jump", always: true, duck: -1, duckFirst: false, ctrlLabel: "always-jump" },
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

/** Upgrade a genome written by an older schema. */
function migrateGenome(g) {
  if (!g || typeof g !== "object") return null;
  if (g.ctrl || g.never || g.always) return g;
  if (g.loA !== undefined) return g;              // already current
  if (g.lo !== undefined && g.hi !== undefined) { // v0: constant window
    return clampGenome({
      loA: g.lo,
      loB: 0,
      widthA: Math.max(10, g.hi - g.lo),
      widthB: 0,
      duck: g.duck === undefined ? 40 : g.duck,
      duckFirst: !!g.duckFirst,
    });
  }
  return null;  // unrecognised: drop rather than feed NaN into the policy
}

function migrate(S) {
  if (!S || typeof S !== "object") return blank();
  const fix = (arr) =>
    (arr || [])
      .map((c) => {
        const g = migrateGenome(c.g || c);
        return g ? { ...c, g } : null;
      })
      .filter(Boolean);
  S.population = fix(S.population);
  S.inProgress = S.inProgress ? fix(S.inProgress) : null;
  if (S.bestGenome) S.bestGenome = migrateGenome(S.bestGenome);
  return S;
}

function load() {
  try {
    const raw = localStorage.getItem(STORE_KEY);
    if (raw) return migrate(JSON.parse(raw));
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
// cv measured at 0.32 for a fixed genome, so a 3-6 episode mean carries a
// standard error of 13-18% - larger than the differences being selected on.
const EPISODES_PER = 9;
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
  // keep the distribution: the mean is dragged by a long upper tail
  cand.samples = (cand.samples || []).concat(dist);
  const sorted = [...cand.samples].sort((a, b) => a - b);
  cand.median = sorted[Math.floor(sorted.length / 2)];
  S.episodes++;
  if (!cand.ctrl && dist > S.bestEver) {
    S.bestEver = dist;          // single-episode record: a luck measure
  }
  // bestMedian is the REPRODUCIBLE figure - what this genome typically does.
  // Only credit it once a candidate has enough episodes to be trustworthy.
  if (!cand.ctrl && cand.runs >= EPISODES_PER && cand.median > (S.bestMedian || 0)) {
    S.bestMedian = cand.median;
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
  // MEDIAN, not mean: the score distribution has a long upper tail (12 runs
  // of one genome: 6340..18777), so a mean rewards luck. The median moves
  // only when a genome is genuinely better more than half the time.
  const evolved = pop
    .filter((c) => !c.ctrl)
    .sort((a, b) => (b.median || b.mean) - (a.median || a.mean));
  const controls = pop.filter((c) => c.ctrl);
  const bestCtrl = Math.max(0, ...controls.map((c) => c.median || c.mean));
  S.generation++;
  S.population = evolved
    .slice(0, ELITE)
    .map((c) => ({ g: c.g, mean: c.mean, median: c.median, best: c.best }));
  S.history.push({
    gen: S.generation,
    best: evolved[0] ? evolved[0].median || evolved[0].mean : 0,
    control: bestCtrl,
    episodes: S.episodes,
  });
  if (S.history.length > 200) S.history = S.history.slice(-200);
  save(S);
  return { best: evolved[0], bestCtrl };
}
