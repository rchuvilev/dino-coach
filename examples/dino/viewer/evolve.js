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
  return g;
}

function randomGenome(rnd) {
  return clampGenome({
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

function mutate(g, rnd) {
  const n = { ...g };
  // 15% of mutations are LARGE. With only small nudges the population never
  // escaped the genome found in generation 1 - measured flat for 27
  // generations. A heavy tail in the step size is the standard fix.
  const heavy = rnd() < 0.15;
  const scale = heavy ? 3.5 : 1;
  const pick = Math.floor(rnd() * 10);
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

// 24 candidates x 4 episodes = 96 episodes per generation. At the measured
// ~366 episodes/sec ceiling that is still sub-second of pure simulation, so
// the limit is DOM/paint, not the search. Larger population beats more
// episodes per candidate here: with cv 0.32 the median of 4 is noisy, but
// selecting the best of 24 noisy estimates still moves faster than the best
// of 6 precise ones.
const POP = 12;
// cv measured at 0.32 for a fixed genome, so a 3-episode mean carries a
// standard error too large to select on. 5 gives a usable median while
// keeping a generation (5 x 6 = 30 episodes) observable: at 9 episodes per
// candidate a generation took many minutes and gen stayed 0.
const EPISODES_PER_FULL = 8;
/** Episodes for a given generation. Generation 1 is short so the chart shows
 *  a datapoint in reasonable time at the default 1x rate; later generations
 *  use the full budget for a trustworthy median. */
export function episodesFor(gen) {
  return gen === 0 ? 3 : EPISODES_PER_FULL;
}
const EPISODES_PER = EPISODES_PER_FULL;
const ELITE = 4;

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
  for (const e of (S.population || []).slice(0, ELITE)) {
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
      const a = out[Math.floor(rnd() * out.length)];
      const b = out[Math.floor(rnd() * out.length)];
      const winner = (a.median || a.mean || 0) >= (b.median || b.mean || 0) ? a : b;
      g = mutate(winner.g, rnd);
      if (rnd() < 0.3) g = mutate(g, rnd);   // occasional double step
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
  S = blank();
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
      causes: {}, jumps: 0, panics: 0, ducks: 0,
      framesAir: 0, frames: 0, missedWindow: 0,
      wideSeen: 0, wideCleared: 0, birdsSeen: 0, birdsCleared: 0,
    });
    t.causes[telemetry.cause] = (t.causes[telemetry.cause] || 0) + 1;
    for (const k of ["jumps","panics","ducks","framesAir","frames","missedWindow",
                     "wideSeen","wideCleared","birdsSeen","birdsCleared"]) {
      t[k] += telemetry[k] || 0;
    }
    // derived rates - the numbers worth looking at
    t.pctAir = t.frames ? +(100 * t.framesAir / t.frames).toFixed(0) : 0;
    t.wideClearRate = t.wideSeen ? +(t.wideCleared / t.wideSeen).toFixed(2) : null;
    t.birdClearRate = t.birdsSeen ? +(t.birdsCleared / t.birdsSeen).toFixed(2) : null;
    // GLOBAL rollup, so the UI can show what kills runs overall
    S.causes = S.causes || {};
    S.causes[telemetry.cause] = (S.causes[telemetry.cause] || 0) + 1;
  }
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
  S.population = evolved.slice(0, ELITE).map((c) => ({
    g: c.g,
    lastMedian: c.median,   // provenance only, never used for ranking
    fromGen: S.generation,
  }));
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
  });
  if (S.history.length > 200) S.history = S.history.slice(-200);
  save(S);
  return { best: evolved[0], bestCtrl };
}
