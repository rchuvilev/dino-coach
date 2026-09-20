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
  if (!g.bands) g.bands = BANDS.map(() => ({ lo: 0, w: 0 }));
  g.loA = Math.min(LO_MAX, Math.max(LO_MIN, g.loA));
  g.loB = Math.max(-8, Math.min(8, g.loB));
  g.widthA = Math.min(HI_MAX - LO_MIN, Math.max(10, g.widthA));
  g.widthB = Math.max(-8, Math.min(8, g.widthB));
  g.duck = Math.min(90, Math.max(5, g.duck));
  g.panic = Math.min(34, Math.max(0, g.panic === undefined ? 0 : g.panic));
  g.late = Math.min(1, Math.max(0.15, g.late === undefined ? 1 : g.late));
  g.wideAdj = Math.min(35, Math.max(-35, g.wideAdj === undefined ? 0 : g.wideAdj));
  g.dropAt = Math.min(70, Math.max(0, g.dropAt === undefined ? 0 : g.dropAt));
  g.arcLow = Math.min(11, Math.max(5, g.arcLow === undefined ? 8 : g.arcLow));
  g.arcHigh = Math.min(17, Math.max(9, g.arcHigh === undefined ? 12 : g.arcHigh));
  g.arcSwitch = Math.min(180, Math.max(30, g.arcSwitch === undefined ? 90 : g.arcSwitch));
  g.ttcLo = Math.min(12, Math.max(2, g.ttcLo === undefined ? 6 : g.ttcLo));
  g.ttcHi = Math.min(20, Math.max(g.ttcLo + 2, g.ttcHi === undefined ? 13 : g.ttcHi));
  g.ttcPanic = Math.min(6, Math.max(0, g.ttcPanic === undefined ? 3 : g.ttcPanic));
  g.mlpVeto = Math.min(0.45, Math.max(0, g.mlpVeto === undefined ? 0.25 : g.mlpVeto));
  g.mlpRescue = Math.min(0.99, Math.max(0.55, g.mlpRescue === undefined ? 0.8 : g.mlpRescue));
  if (!g.ctxAdj) g.ctxAdj = {};
  return g;
}

function randomGenome(rnd) {
  // Seed from what the session already knows. A fresh genome that starts
  // with the pooled corrections is not starting from zero knowledge - it
  // only has to search the axes the knowledge base cannot fix.
  const seeded = knowledgeOffsets();
  const ctxAdj = {};
  for (const [k, v] of Object.entries(seeded)) ctxAdj[k] = v.delta;
  return clampGenome({
    ctxAdj,
    loA: Math.round(LO_MIN + rnd() * (LO_MAX - LO_MIN)),
    loB: +((rnd() - 0.5) * 8).toFixed(2),
    widthA: Math.round(20 + rnd() * 60),
    widthB: +((rnd() - 0.5) * 8).toFixed(2),
    // Birds (PTERODACTYL) require minSpeed 8.5 in the game source, so they
    // only appear in runs that survive that long - measured 5 of 10 runs for
    // a strong genome. The duck gene therefore looks like dead code until a
    // policy is good enough to reach them.
    duck: Math.round(rnd() * 90),             // wide range, GA decides
    // rule order: which check wins when several match
    duckFirst: rnd() < 0.5,
    // PANIC jump: if the window was missed (typically because the dino was
    // airborne through it) jump anyway below this gap rather than running
    // into the obstacle. 0 disables, so evolution can switch it off.
    panic: Math.round(rnd() * 34),            // wide range, GA decides
    // fraction of the window to actually use, measured from its LOWER edge.
    // 1 = old behaviour (fire on entry), 0.3 = wait until the obstacle is
    // close. Firing on entry left the dino airborne 56% of all frames.
    late: +(0.15 + rnd() * 0.85).toFixed(2),  // wide range, GA decides
    // Takeoff shift for WIDE obstacles (>=50px). Measured: they dominate the
    // "jumped early, descended onto it" class, and at speed 6 they are not
    // clearable at all (102px of clearance travel against 119px needed), so
    // the correct response differs from a narrow cactus.
    wideAdj: Math.round((rnd() - 0.5) * 70),   // wide range, GA decides
    // gap at which to ABORT a jump with fastdrop. 0 = never. The action
    // itself is an engine capability; this only says when to reach for it.
    dropAt: Math.round(rnd() * 60),
    // TIME-TO-COLLISION window, in frames. Measured optimum around 6..13
    // (median 1670 vs 602 for the pixel-gap baseline), but the range is
    // wide so the GA can move it rather than inherit my hand-picked value.
    ttcLo: 0,   // OFF by default: integration freezes the loop, see note
    ttcHi: +(9 + rnd() * 9).toFixed(1),
    ttcPanic: +(1 + rnd() * 4).toFixed(1),
    // how much to trust the trained model. veto below this P(clear),
    // rescue above the other. The GA decides whether the model helps.
    mlpVeto: +(0.1 + rnd() * 0.3).toFixed(2),
    mlpRescue: +(0.7 + rnd() * 0.25).toFixed(2),
    // JUMP ARC genes. Measured: velocity 8 gives a 57px/28-frame arc,
    // velocity 12 gives 126px/42 frames. Low recovers sooner (fixes the
    // "missed" class), high covers more ground (fixes "wide_early").
    arcLow: +(6 + rnd() * 4).toFixed(1),
    arcHigh: +(10 + rnd() * 6).toFixed(1),
    // gap between consecutive obstacles below which the low arc is chosen
    arcSwitch: Math.round(50 + rnd() * 90),
    // start at zero offsets: identical to option A until evolution finds a
    // reason to differentiate a band
    bands: BANDS.map(() => ({ lo: 0, w: 0 })),
  });
}

/** Speed bands. Boundaries sit where the measured optimum shifts. */
export const BANDS = [
  { name: "slow", max: 8 },
  { name: "mid", max: 10.5 },
  { name: "fast", max: 99 },
];

export function bandOf(speed) {
  const s = speed || 6;
  for (let i = 0; i < BANDS.length; i++) if (s < BANDS[i].max) return i;
  return BANDS.length - 1;
}

/** Resolve the genome at a given speed. This is the whole of option A. */
export function windowAt(g, speed) {
  const d = (speed || 6) - 6;
  // OPTION B: a per-band offset layered on the speed-linear base. Offsets of
  // zero reduce exactly to option A, so the richer space CONTAINS it and
  // cannot score worse given enough search - the same containment argument
  // that justified A over the constant genome.
  const band = bandOf(speed);
  const off = (g.bands && g.bands[band]) || { lo: 0, w: 0 };
  let lo = g.loA + g.loB * d + off.lo;
  let width = g.widthA + g.widthB * d + off.w;
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


/**
 * Uniform crossover: each gene comes from one parent or the other.
 *
 * Mutation alone explores a NEIGHBOURHOOD; it cannot combine a parent with
 * good takeoff timing and a parent with good arc selection. Crossover mixes
 * traits discovered independently - the "take the good parts from the good
 * runs" behaviour that mutation-only search cannot produce.
 *
 * ctxAdj merges per-situation rather than wholesale, because those entries
 * are independent per-state corrections and a child should be able to
 * inherit the better one for each state separately.
 */
export function crossover(a, b, rnd) {
  const child = {};
  for (const k of new Set([...Object.keys(a), ...Object.keys(b)])) {
    if (k === "ctxAdj" || k === "bands" || k.startsWith("__")) continue;
    child[k] = rnd() < 0.5 ? a[k] : b[k];
  }
  const ca = a.ctxAdj || {};
  const cb = b.ctxAdj || {};
  child.ctxAdj = {};
  for (const k of new Set([...Object.keys(ca), ...Object.keys(cb)])) {
    const va = ca[k];
    const vb = cb[k];
    child.ctxAdj[k] = va === undefined ? vb : vb === undefined ? va : (rnd() < 0.5 ? va : vb);
  }
  if (a.bands || b.bands) {
    const ba = a.bands || [];
    const bb = b.bands || [];
    child.bands = (ba.length ? ba : bb).map((_, i) =>
      rnd() < 0.5 ? { ...(ba[i] || { lo: 0, w: 0 }) } : { ...(bb[i] || { lo: 0, w: 0 }) });
  }
  return clampGenome(child);
}

function mutate(g, rnd) {
  const n = { ...g };
  // 15% of mutations are LARGE. With only small nudges the population never
  // escaped the genome found in generation 1 - measured flat for 27
  // generations. A heavy tail in the step size is the standard fix.
  // Stagnation-scaled step size: 1x while improving, up to 6x after a long
  // plateau. A fixed step size cannot both refine a good genome and escape a
  // local optimum - measured, small steps alone froze the score for 10+
  // generations with a fully diverse population.
  const boost = (typeof S !== "undefined" && S && S.mutBoost) || 1;
  const heavy = rnd() < 0.15;
  const scale = (heavy ? 3.5 : 1) * boost;
  const pick = Math.floor(rnd() * 18);
  const nudge = () => Math.round((rnd() - 0.5) * 30 * scale);
  const slope = () => +((rnd() - 0.5) * 4 * scale).toFixed(2);
  if (pick === 0) n.loA = n.loA + nudge();
  else if (pick === 1) n.widthA = n.widthA + nudge();
  else if (pick === 2) n.duck = n.duck + nudge();
  else if (pick === 3) n.duckFirst = !n.duckFirst;
  else if (pick === 4) {
    // mutate a SLOPE: the axis option A adds
    if (rnd() < 0.5) n.loB = +(n.loB + slope()).toFixed(2);
    else n.widthB = +(n.widthB + slope()).toFixed(2);
  } else if (pick === 6) {
    n.panic = Math.max(0, Math.min(34, (n.panic || 0) + Math.round((rnd() - 0.5) * 16)));
  } else if (pick === 7) {
    n.late = +Math.max(0.15, Math.min(1, (n.late === undefined ? 1 : n.late) + (rnd() - 0.5) * 0.4)).toFixed(2);
  } else if (pick === 8) {
    n.wideAdj = Math.max(-35, Math.min(35, (n.wideAdj || 0) + Math.round((rnd() - 0.5) * 20)));
  } else if (pick === 9) {
    n.dropAt = Math.max(0, Math.min(70, (n.dropAt || 0) + Math.round((rnd() - 0.5) * 30)));
  } else if (pick === 10) {
    n.arcLow = +Math.max(5, Math.min(11, (n.arcLow || 8) + (rnd() - 0.5) * 3 * scale)).toFixed(1);
  } else if (pick === 11) {
    n.arcHigh = +Math.max(9, Math.min(17, (n.arcHigh || 12) + (rnd() - 0.5) * 4 * scale)).toFixed(1);
  } else if (pick === 12) {
    n.arcSwitch = Math.max(30, Math.min(180, (n.arcSwitch || 90) + Math.round((rnd() - 0.5) * 50 * scale)));
  } else if (pick === 13) {
    n.ttcLo = +Math.max(2, Math.min(12, (n.ttcLo || 6) + (rnd() - 0.5) * 3 * scale)).toFixed(1);
  } else if (pick === 14) {
    n.ttcHi = +Math.max(4, Math.min(20, (n.ttcHi || 13) + (rnd() - 0.5) * 4 * scale)).toFixed(1);
  } else if (pick === 15) {
    n.ttcPanic = +Math.max(0, Math.min(6, (n.ttcPanic || 3) + (rnd() - 0.5) * 2 * scale)).toFixed(1);
  } else if (pick === 16) {
    n.mlpVeto = +Math.max(0, Math.min(0.45, (n.mlpVeto ?? 0.25) + (rnd() - 0.5) * 0.2 * scale)).toFixed(2);
  } else if (pick === 17) {
    n.mlpRescue = +Math.max(0.55, Math.min(0.99, (n.mlpRescue ?? 0.8) + (rnd() - 0.5) * 0.2 * scale)).toFixed(2);
  } else {
    // mutate ONE BAND's offset: the axis option B adds
    n.bands = (n.bands || BANDS.map(() => ({ lo: 0, w: 0 }))).map((b) => ({ ...b }));
    const i = Math.floor(rnd() * n.bands.length);
    if (rnd() < 0.5) n.bands[i].lo = Math.max(-30, Math.min(30, n.bands[i].lo + nudge()));
    else n.bands[i].w = Math.max(-30, Math.min(30, n.bands[i].w + nudge()));
  }
  return clampGenome(n);
}

const key = (g) =>
  `${g.loA}|${g.loB}|${g.widthA}|${g.widthB}|${g.duck}|${g.duckFirst ? 1 : 0}|${g.panic || 0}|` +
  (g.bands || []).map((b) => `${b.lo},${b.w}`).join(";");
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
    /**
     * The reigning model: the one every run targets. Replaced only when a
     * challenger measurably beats it, otherwise reused unchanged.
     */
    champion: null,      // { g, hash, best, runs, changeHash }
    challenger: null,    // { g, hash, changeHash } currently under test
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
  if (g.loA !== undefined) {
    if (!g.bands) g.bands = BANDS.map(() => ({ lo: 0, w: 0 }));
    return g;
  }
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

/** Structural validation: a state can parse and still be unusable. */
function validate(S) {
  if (!S || typeof S !== "object") return false;
  for (const list of [S.population, S.inProgress]) {
    if (!list) continue;
    if (!Array.isArray(list)) return false;
    for (const c of list) {
      if (!c || typeof c !== "object") return false;
      const g = c.g || c;
      // every genome MUST carry a numeric takeoff base, or decide() throws
      if (!g || typeof g !== "object") return false;
      if (g.loA !== undefined && typeof g.loA !== "number") return false;
    }
  }
  if (S.generation !== undefined && typeof S.generation !== "number") return false;
  return true;
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
    if (raw) {
      const parsed = migrate(JSON.parse(raw));
      if (validate(parsed)) return parsed;
      // Structurally broken: discard rather than run on it. Silently
      // returning a bad state is what makes a run appear stuck.
      try { localStorage.removeItem(STORE_KEY); } catch { /* ignore */ }
      if (typeof console !== "undefined") {
        console.warn("[dino] stored state was invalid and has been discarded");
      }
    }
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

// 24 candidates x 4 episodes = 96 episodes per generation. At the measured
// ~366 episodes/sec ceiling that is still sub-second of pure simulation, so
// the limit is DOM/paint, not the search. Larger population beats more
// episodes per candidate here: with cv 0.32 the median of 4 is noisy, but
// selecting the best of 24 noisy estimates still moves faster than the best
// of 6 precise ones.
const POP = 8;
// cv measured at 0.32 for a fixed genome, so a 3-episode mean carries a
// standard error too large to select on. 5 gives a usable median while
// keeping a generation (5 x 6 = 30 episodes) observable: at 9 episodes per
// candidate a generation took many minutes and gen stayed 0.
const EPISODES_PER_FULL = 8;
/** Episodes for a given generation. Generation 1 is short so the chart shows
 *  a datapoint in reasonable time at the default 1x rate; later generations
 *  use the full budget for a trustworthy median. */
export function episodesFor(gen) {
  // ONE run per candidate, always. The model is corrected after every
  // episode, so a candidate re-evaluated later is judged by DIFFERENT
  // weights - repeating it 8 times in a row just averages away the
  // adjustment we made in between.
  return 1;
}
// With one episode per candidate a "median" is that single score. Noise is
// handled by re-encountering good genomes across generations via elitism
// rather than by repeating them back-to-back.
const EPISODES_PER = 1;
const ELITE = 4;

/**
 * CHAMPION / CHALLENGER.
 *
 * Every run targets the best model found so far. A challenger is that model
 * plus exactly one change; it is promoted only if it beats the champion's
 * best, otherwise the champion is reused unchanged and a different change is
 * tried. This is hill-climbing with an explicit "keep what worked" step,
 * which population-wide breeding did not have - there a good genome could be
 * diluted by unrelated candidates and nothing guaranteed reversion.
 */
export function nextCandidate() {
  // first run of a session: the champion IS the candidate
  if (!S.champion) {
    const g = randomGenome(rnd);
    S.champion = { g, hash: genomeHash(g), best: 0, runs: 0, changeHash: "initial" };
    S.challenger = null;
    return { g, role: "champion", meta: S.champion };
  }
  // propose a challenger: champion + one mutation
  const mutated = mutate(S.champion.g, rnd);
  S.challenger = {
    g: mutated,
    hash: genomeHash(mutated),
    changeHash: changeHash(S.champion.g, mutated),
  };
  return { g: mutated, role: "challenger", meta: S.challenger };
}

/**
 * Judge the run that just finished.
 * @returns {{promoted:boolean, champion:object, result:number}}
 */
export function judgeRun(role, score) {
  if (role === "champion") {
    S.champion.runs++;
    if (score > (S.champion.best || 0)) S.champion.best = score;
    save(S);
    return { promoted: false, champion: S.champion, result: score, reused: true };
  }
  const ch = S.challenger;
  if (!ch) return { promoted: false, champion: S.champion, result: score };
  const beat = score > (S.champion.best || 0);
  if (beat) {
    // the change helped: it becomes the model every future run targets
    S.champion = {
      g: ch.g,
      hash: ch.hash,
      best: score,
      runs: 1,
      changeHash: ch.changeHash,
    };
  } else {
    // it did not help: champion is reused unchanged, and its run count
    // reflects that another attempt was spent on it
    S.champion.runs++;
  }
  S.challenger = null;
  save(S);
  return { promoted: beat, champion: S.champion, result: score, reused: !beat };
}

export function championInfo() {
  return S.champion;
}

function seedPopulation() {
  const out = [];
  // resume a generation that was interrupted mid-way
  if (S.inProgress && S.inProgress.length) {
    for (const c of S.inProgress) {
      // Trim to the episode budget. Samples beyond it are from an earlier
      // generation of the same genome and must not inflate the statistic.
      const samples = (c.samples || []).slice(-EPISODES_PER);
      const sorted = [...samples].sort((a, b) => a - b);
      out.push({
        g: c.g,
        runs: samples.length,
        total: samples.reduce((a, b) => a + b, 0),
        best: samples.length ? Math.max(...samples) : 0,
        mean: samples.length ? Math.round(samples.reduce((a, b) => a + b, 0) / samples.length) : 0,
        samples,
        median: samples.length ? sorted[Math.floor(sorted.length / 2)] : undefined,
        // telemetry must ride along: fitnessOf() needs the failure mix, and
        // dropping it here made the whole adjustment a no-op (fit === best).
        tel: c.tel,
        elite: true,
      });
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
  let elites = S.population || [];
  if (S.forceCull && elites.length > 1) {
    // remove the longest-serving elite: it is the anchor holding the
    // population at a local optimum
    elites = elites.slice(0, -1);
    S.forceCull = false;
  }
  for (const e of elites.slice(0, ELITE)) {
    // genome carries forward, MEASUREMENTS DO NOT. Keeping runs/samples made
    // an elite's stale median win every generation without being re-tested -
    // 27 generations reported the identical best of 18255.
    // median deliberately NOT carried: an elite must re-earn its rank from
    // fresh episodes each generation. Carrying it let a lucky batch reign
    // for 42 consecutive generations without being retested.
    out.push({ g: e.g, runs: 0, total: 0, best: 0, mean: 0, samples: [], elite: true });
  }
  // Fill by TOURNAMENT: pick 2 elites at random, breed from the better one.
  // Truncation to the top ELITE at POP=24 throws away too much diversity;
  // tournament keeps selection pressure while letting mid-rank genomes
  // reproduce. A fraction stays fully random to avoid premature convergence.
  let guard = 0;
  while (out.length < POP && guard++ < POP * 20) {
    let g;
    if (out.length >= 2 && rnd() < 0.72) {
      // two independent tournaments -> two winners -> MIX them, then mutate.
      // Picking a single winner and mutating explores one neighbourhood;
      // crossing two winners combines traits discovered separately.
      const pick = () => {
        const a = out[Math.floor(rnd() * out.length)];
        const b = out[Math.floor(rnd() * out.length)];
        return (a.median || a.mean || 0) >= (b.median || b.mean || 0) ? a : b;
      };
      const p1 = pick();
      const p2 = pick();
      g = rnd() < 0.6 && p1 !== p2 ? crossover(p1.g, p2.g, rnd) : mutate(p1.g, rnd);
      if (rnd() < 0.5) g = mutate(g, rnd);
    } else {
      g = randomGenome(rnd);
    }
    if (out.some((c) => key(c.g) === key(g))) continue;
    out.push({ g, runs: 0, total: 0, best: 0, mean: 0, samples: [] });
  }
  for (const c of CONTROLS) out.push({ g: c, runs: 0, total: 0, best: 0, mean: 0, ctrl: c.ctrl });
  return out;
}

export function currentState() {
  return S;
}

const SLOT_KEY = STORE_KEY + "-slot";

/** Explicit save: a named snapshot the rolling autosave cannot overwrite. */
export function saveSlot() {
  try {
    localStorage.setItem(SLOT_KEY, JSON.stringify({ ...S, savedAt: Date.now() }));
    return true;
  } catch {
    return false;
  }
}

/** Restore the explicit snapshot. Returns false when there is none. */
export function loadSlot() {
  try {
    const raw = localStorage.getItem(SLOT_KEY);
    if (!raw) return false;
    S = migrate(JSON.parse(raw));
    save(S);
    return true;
  } catch {
    return false;
  }
}

export function slotInfo() {
  try {
    const raw = localStorage.getItem(SLOT_KEY);
    if (!raw) return null;
    const d = JSON.parse(raw);
    return { gen: d.generation, ep: d.episodes, typical: d.bestMedian, savedAt: d.savedAt };
  } catch {
    return null;
  }
}

/** Serialise for download. */
export function exportState() {
  return JSON.stringify({ ...S, exportedAt: Date.now() }, null, 2);
}

/** Load from an imported file. Migrated, so old exports still work. */
export function importState(text) {
  const parsed = JSON.parse(text);
  if (!parsed || typeof parsed !== "object") throw new Error("not a state object");
  S = migrate(parsed);
  save(S);
  return S;
}

export function resetAll() {
  // Mutate IN PLACE. Reassigning S left other closures holding the old
  // object, which was then written back by the next autosave - measured,
  // reset left generation 7386 and 125675 episodes intact.
  const fresh = blank();
  for (const k of Object.keys(S)) delete S[k];
  Object.assign(S, fresh);
  save(S);
}

export { seedPopulation, mutate, randomGenome, key, label, ELITE, EPISODES_PER, POP, rnd, clampGenome, save };

export function recordEpisode(cand, dist, pop, bandDist, telemetry) {
  cand.runs++;
  cand.total += dist;
  cand.best = Math.max(cand.best, dist);
  cand.mean = Math.round(cand.total / cand.runs);
  // keep the distribution: the mean is dragged by a long upper tail
  // bounded by the episode budget: samples beyond it belong to a previous
  // generation and would make mean/median disagree wildly (observed 44690
  // vs 8300 for one candidate with 33 samples against a budget of 5).
  cand.samples = (cand.samples || []).concat(dist).slice(-EPISODES_PER);
  // per-band credit: distance earned WHILE each band was active. This is what
  // makes option B more than extra parameters - a band's offset is judged on
  // the progress it actually produced, not on the episode total.
  // --- accumulate telemetry so failure CLASSES become measured data rather
  // than something a human sweeps by hand.
  if (telemetry) {
    const t = (cand.tel = cand.tel || {
      causes: {}, ctx: {}, jumps: 0, panics: 0, ducks: 0,
      framesAir: 0, frames: 0, missedWindow: 0,
      wideSeen: 0, wideCleared: 0, birdsSeen: 0, birdsCleared: 0,
    });
    t.causes[telemetry.cause] = (t.causes[telemetry.cause] || 0) + 1;
    for (const k of ["jumps","panics","ducks","framesAir","frames","missedWindow",
                     "wideSeen","wideCleared","birdsSeen","birdsCleared"]) {
      t[k] += telemetry[k] || 0;
    }
    // MERGE the per-situation outcome map. It is a nested object, so the
    // numeric loop above dropped it entirely - measured ctxPresent 0 on every
    // candidate while hasTel was true, which made learnedOffsets a no-op.
    if (telemetry.ctx) {
      t.ctx = t.ctx || {};
      for (const [k, e] of Object.entries(telemetry.ctx)) {
        const dst = (t.ctx[k] = t.ctx[k] || { ok: 0, fail: 0, okGap: [], failGap: [], noJump: 0 });
        dst.ok += e.ok || 0;
        dst.fail += e.fail || 0;
        dst.noJump = (dst.noJump || 0) + (e.noJump || 0);
        if (e.okGap) dst.okGap = dst.okGap.concat(e.okGap).slice(-30);
        if (e.failGap) dst.failGap = dst.failGap.concat(e.failGap).slice(-30);
      }
    }
    if (telemetry.lastFailure) t.lastFailure = telemetry.lastFailure;
    // derived rates - the numbers worth looking at
    t.pctAir = t.frames ? +(100 * t.framesAir / t.frames).toFixed(0) : 0;
    t.wideClearRate = t.wideSeen ? +(t.wideCleared / t.wideSeen).toFixed(2) : null;
    t.birdClearRate = t.birdsSeen ? +(t.birdsCleared / t.birdsSeen).toFixed(2) : null;
    // GLOBAL rollup, so the UI can show what kills runs overall
    S.causes = S.causes || {};
    S.causes[telemetry.cause] = (S.causes[telemetry.cause] || 0) + 1;
  }
  // pool this episode's situation outcomes into session knowledge
  if (telemetry && telemetry.ctx) absorbKnowledge({ ctx: telemetry.ctx });
  if (bandDist) {
    cand.bandTotals = (cand.bandTotals || bandDist.map(() => 0)).map(
      (v, i) => v + (bandDist[i] || 0),
    );
  }
  const sorted = [...cand.samples].sort((a, b) => a - b);
  cand.median = sorted[Math.floor(sorted.length / 2)];
  S.episodes++;
  if (!cand.ctrl && dist > S.bestEver) {
    S.bestEver = dist;          // single-episode record: a luck measure
  }
  // bestMedian must be REPRODUCIBLE, not a lucky batch. A single 8-episode
  // median claimed 19139 while 12 fresh runs of the same genome gave 10245 -
  // enshrining the discovery value makes it unbeatable by construction.
  // A challenger is recorded as PENDING and only crowned when a later,
  // independent batch confirms it.
  if (!cand.ctrl && cand.runs >= EPISODES_PER) {
    const pending = S.pendingBest;
    if (pending && key(pending.g) === key(cand.g)) {
      // second independent batch for the same genome: confirm with the WORSE
      // of the two medians, so a champion is never credited above what it
      // has repeated.
      const confirmed = Math.min(pending.median, cand.median);
      if (confirmed > (S.bestMedian || 0)) {
        S.bestMedian = confirmed;
        S.bestGenome = cand.g;
      }
      S.pendingBest = null;
    } else if (cand.median > (S.bestMedian || 0)) {
      S.pendingBest = { g: cand.g, median: cand.median };
    }
  }
  // Persist EVERY episode. Saving only at generation close meant a session
  // stopped mid-generation lost all of it - measured 6 episodes and a best of
  // 15382 while storage still read gen 0 / ep 0 / best 0.
  if (pop) {
    S.inProgress = pop
      .filter((c) => !c.ctrl)
      .map((c) => ({
        g: c.g,
        runs: c.runs,
        total: c.total,
        best: c.best,
        mean: c.mean,
        // samples MUST persist: the median is computed from them, and
        // dropping them made median undefined on reload, which silently
        // disabled both ranking and generation close.
        samples: (c.samples || []).slice(-EPISODES_PER),
        bandTotals: c.bandTotals,
        tel: c.tel,
        median: c.median,
      }));
  }
  save(S);
}

/**
 * Close a generation: rank by LIVE mean, keep elites, record history.
 * Controls are ranked alongside but never become parents.
 */
/** True when every non-control candidate has completed its episode budget. */
export function generationComplete(pop) {
  return pop.filter((c) => !c.ctrl).every((c) => (c.runs || 0) >= EPISODES_PER);
}

/**
 * Fitness = median distance, adjusted by the failure profile.
 *
 * Rationale: at cv 0.50 the median of 4 episodes cannot order genomes whose
 * true medians differ by less than ~30%, so the search stalls on noise. The
 * death-cause mix is measured over every episode and is far less noisy than
 * the score, so it can break those ties toward genomes that fail in
 * recoverable ways rather than unrecoverable ones.
 *
 * `missed` (still airborne when the window opened) is weighted worst: it is
 * the class the agent has no answer to. `early` is penalised less because a
 * genome that at least attempts the jump is one mutation from correct timing.
 */
/**
 * Per-situation corrections learned from recorded outcomes.
 *
 * For each situation class we hold the takeoff gaps that worked and those
 * that did not. If a class has enough of both, the gap between their medians
 * is a DIRECTED correction for that class - the thing "38% missed" could
 * never tell us. Requires MIN_SAMPLES on each side, because a single failure
 * is noise at cv 0.50 and would otherwise swing the offset wildly.
 */
const MIN_SAMPLES = 4;

/**
 * Merge one candidate's per-situation outcomes into the SESSION knowledge
 * base. Pooling across all candidates matters: a single genome sees a class
 * a handful of times, but the population as a whole sees it hundreds of
 * times, which is the difference between a noisy offset and a usable one.
 */
export function absorbKnowledge(tel) {
  if (!tel || !tel.ctx) return;
  S.knowledge = S.knowledge || {};
  for (const [k, e] of Object.entries(tel.ctx)) {
    const dst = (S.knowledge[k] = S.knowledge[k] || {
      ok: 0, fail: 0, okGap: [], failGap: [], noJump: 0,
    });
    dst.ok += e.ok || 0;
    dst.fail += e.fail || 0;
    dst.noJump = (dst.noJump || 0) + (e.noJump || 0);
    // keep a bounded window so old, obsolete timings age out
    if (e.okGap) dst.okGap = dst.okGap.concat(e.okGap).slice(-60);
    if (e.failGap) dst.failGap = dst.failGap.concat(e.failGap).slice(-60);
  }
}

/** Offsets derived from the POOLED session knowledge, not one genome. */
export function knowledgeOffsets() {
  return learnedOffsets({ ctx: S.knowledge || {} });
}

export function learnedOffsets(tel) {
  if (!tel || !tel.ctx) return {};
  const med = (a) => {
    if (!a || !a.length) return null;
    const s2 = [...a].sort((x, y) => x - y);
    return s2[Math.floor(s2.length / 2)];
  };
  const out = {};
  for (const [k, e] of Object.entries(tel.ctx)) {
    if (!e.okGap || !e.failGap) continue;
    if (e.okGap.length < MIN_SAMPLES || e.failGap.length < MIN_SAMPLES) continue;
    // an offset derived from a handful of failures is noise at cv 0.50
    if ((e.fail || 0) < MIN_SAMPLES) continue;
    const mo = med(e.okGap);
    const mf = med(e.failGap);
    if (mo === null || mf === null) continue;
    // reject sentinels and impossible gaps - a takeoff beyond the screen
    // cannot be a real observation
    if (Math.abs(mo) > 500 || Math.abs(mf) > 500) continue;
    // positive = successful takeoffs happened at a LARGER gap (earlier)
    const delta = mo - mf;
    if (Math.abs(delta) < 3) continue;   // below measurement resolution
    out[k] = {
      delta: Math.max(-25, Math.min(25, Math.round(delta))),
      okN: e.okGap.length,
      failN: e.failGap.length,
      rate: +(e.ok / (e.ok + e.fail)).toFixed(2),
    };
  }
  return out;
}

/**
 * Stable short hash of a genome, so a model can be NAMED in the log and the
 * same parameters always produce the same name across sessions.
 */
export function genomeHash(g) {
  if (!g) return "------";
  const str = JSON.stringify(g, Object.keys(g).sort());
  let h = 2166136261;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return (h >>> 0).toString(36).padStart(6, "0").slice(-6);
}

/** Hash of only the fields a mutation changed, for the "last change" id. */
export function changeHash(before, after) {
  if (!before || !after) return "------";
  const diff = {};
  for (const k of new Set([...Object.keys(before), ...Object.keys(after)])) {
    if (k.startsWith("__")) continue;
    const a = JSON.stringify(before[k]);
    const b = JSON.stringify(after[k]);
    if (a !== b) diff[k] = [before[k], after[k]];
  }
  if (!Object.keys(diff).length) return "nochange";
  return genomeHash(diff);
}

export function fitnessOf(c) {
  const base = c.median || 0;
  const t = c.tel;
  if (!t || !t.causes) return base;
  const total = Object.values(t.causes).reduce((a, b) => a + b, 0);
  if (!total) return base;
  const rate = (k) => (t.causes[k] || 0) / total;
  const missed = rate("missed") + rate("wide_missed");
  const early = rate("early") + rate("wide_early");
  // clearance rates are direct evidence the policy HANDLES a class
  const wideBonus = t.wideClearRate === null || t.wideClearRate === undefined ? 0 : t.wideClearRate;
  const birdBonus = t.birdClearRate === null || t.birdClearRate === undefined ? 0 : t.birdClearRate;
  // multiplicative, so it scales with the score rather than swamping it
  const factor = 1 - 0.25 * missed - 0.10 * early + 0.08 * wideBonus + 0.05 * birdBonus;
  return Math.round(base * Math.max(0.5, Math.min(1.25, factor)));
}

/** Generations since the record last moved. Drives the escape mechanisms. */
function stagnation(S2) {
  return (S2.generation || 0) - (S2.lastImproveGen || 0);
}

export function closeGeneration(pop) {
  // MEDIAN, not mean: the score distribution has a long upper tail (12 runs
  // of one genome: 6340..18777), so a mean rewards luck. The median moves
  // only when a genome is genuinely better more than half the time.
  const evolved = pop
    .filter((c) => !c.ctrl)
    .sort((a, b) => fitnessOf(b) - fitnessOf(a));
  const controls = pop.filter((c) => c.ctrl);
  const bestCtrl = Math.max(0, ...controls.map((c) => c.median || c.mean));
  S.generation++;
  // Only the genome survives. A median is a MEASUREMENT of one generation's
  // episodes and must never be carried into the next - doing so froze the
  // reported best at 33055 for 31 consecutive generations while real
  // candidates were scoring 10012..22436.
  // Bake each elite's learned per-situation corrections into its genome, so
  // the next generation INHERITS what the failures taught rather than
  // rediscovering it. This is the step that makes failure analysis change
  // behaviour instead of only changing a score.
  for (const c of evolved.slice(0, ELITE)) {
    // pooled knowledge first, then this genome's own experience on top -
    // its own data is more specific to how IT plays, so it wins ties
    const pooled = knowledgeOffsets();
    const own = learnedOffsets(c.tel);
    const off = { ...pooled, ...own };
    if (!Object.keys(off).length) continue;

    c.g = { ...c.g, ctxAdj: { ...(c.g.ctxAdj || {}) } };
    const prevRates = c.g.__ctxRate || {};
    const nowRates = {};
    const ctx = (c.tel && c.tel.ctx) || {};

    for (const [k, v] of Object.entries(off)) {
      const e = ctx[k];
      // observed success rate for this class THIS generation
      const tried = e ? e.ok + e.fail + (e.noJump || 0) : 0;
      const rate = tried ? e.ok / tried : null;
      if (rate !== null) nowRates[k] = +rate.toFixed(3);

      const prev = c.g.ctxAdj[k] || 0;
      const before = prevRates[k];

      // VERDICT on the offset already in place
      if (prev !== 0 && before !== undefined && rate !== null) {
        if (rate < before - 0.02) {
          // it made this class WORSE - roll back toward zero rather than
          // pushing further in the same direction
          c.g.ctxAdj[k] = Math.round(prev * 0.5);
          continue;
        }
        if (Math.abs(rate - before) <= 0.02) {
          // no measurable effect: hold, do not accumulate drift
          continue;
        }
      }
      // either no offset yet, or the last one helped - advance it
      c.g.ctxAdj[k] = Math.max(-30, Math.min(30, Math.round(prev + v.delta * 0.4)));
    }
    c.g.__ctxRate = nowRates;
  }
  S.population = evolved.slice(0, ELITE).map((c) => ({
    g: c.g,
    lastMedian: c.median,   // provenance only, never used for ranking
    fromGen: S.generation,
  }));
  // --- ADAPTIVE ESCAPE ---------------------------------------------------
  // Track whether this generation actually beat the running best. Measured
  // stagnation was 10+ generations at an identical score with a diverse
  // population, which means selection pressure - not exploration - is stuck.
  const topFit = evolved.length ? fitnessOf(evolved[0]) : 0;
  if (topFit > (S.bestFitEver || 0) * 1.02) {
    S.bestFitEver = topFit;
    S.lastImproveGen = S.generation;
  }
  const stale = stagnation(S);
  // Mutation strength rises with stagnation and resets on improvement, so
  // the search widens only when it has evidence it is stuck.
  S.mutBoost = stale >= 5 ? Math.min(6, 1 + (stale - 4) * 0.5) : 1;
  // After a long plateau, cull the oldest elite outright. Keeping every
  // elite forever is what pins the population to one point.
  if (stale >= 12 && evolved.length > 1) {
    S.forceCull = true;
    S.lastImproveGen = S.generation;  // give the new shape time to prove out
  }

  S.history.push({
    gen: S.generation,
    // only count candidates with real episodes this generation
    best: (() => {
      const measured = evolved.filter((c) => (c.runs || 0) > 0 && c.median !== undefined);
      return measured.length ? measured[0].median : 0;
    })(),
    // fitness of that winner, so a flat median with improving failure mix is
    // still visible as progress
    fit: (() => {
      const measured = evolved.filter((c) => (c.runs || 0) > 0 && c.median !== undefined);
      return measured.length ? fitnessOf(measured[0]) : 0;
    })(),
    control: bestCtrl,
    episodes: S.episodes,
    stale,
  });
  if (S.history.length > 200) S.history = S.history.slice(-200);
  save(S);
  return { best: evolved[0], bestCtrl };
}
