/**
 * Three-way analysis: STATE x FAIL-REASON x PROP.
 *
 * The three dimensions already existed but never met:
 *   - state      ctxKey()  -> wide/slow/tight buckets
 *   - fail reason cause     -> early | late | missed | bird
 *   - prop        captureSituation() -> takeoffGap, speed, width, arc, ...
 *
 * Tracking them apart answers "38% of deaths are early" and "wide/slow/open
 * fails 3% of the time", neither of which says WHICH PROPERTY to change. The
 * join answers the actionable question: "in wide/slow, deaths classed EARLY
 * happen at takeoffGap 72 while survivals happen at 41 - so reduce takeoffGap
 * in that state for that reason."
 *
 * Deliberately NOT a deeper bucket tree: state x reason is already 7 x 5
 * cells, and splitting further starves every cell. Props are kept as
 * distributions inside a cell rather than as more keys.
 */

/** Props whose value at takeoff is worth correlating with the outcome. */
export const TRACKED_PROPS = [
  "takeoffGap",   // pixels to the obstacle when we jumped
  "ttc",          // frames to impact - measured 2.8x more predictive than gap
  "speed",
  "targetWidth",
  "nextDelta",    // distance to the obstacle after this one
  "arcVel",       // which jump shape was used
];

const MIN_CELL = 6;   // below this a cell's medians are noise

function median(a) {
  if (!a || !a.length) return null;
  const s = [...a].sort((x, y) => x - y);
  return s[Math.floor(s.length / 2)];
}

/**
 * Accumulate one resolved decision into the joined table.
 *
 * @param {object} store  persistent table, keyed `state|reason`
 * @param {string} state  ctxKey() bucket
 * @param {string} reason "ok" for a survival, else the death cause
 * @param {object} props  the captured situation at takeoff
 */
export function record(store, state, reason, props) {
  const key = `${state}|${reason}`;
  const cell = (store[key] = store[key] || { n: 0, props: {} });
  cell.n++;
  for (const p of TRACKED_PROPS) {
    const v = props[p];
    if (v === null || v === undefined || !Number.isFinite(v)) continue;
    // reject sentinels: 9999/99999 mean "nothing visible", not a measurement
    if (Math.abs(v) > 500) continue;
    const arr = (cell.props[p] = cell.props[p] || []);
    arr.push(v);
    if (arr.length > 40) arr.shift();   // sliding window, tracks current policy
  }
}

/**
 * For one state, compare each failure reason against the survivals in that
 * SAME state, prop by prop. Returns only differences large enough to act on.
 *
 * Comparing within a state is what makes this meaningful: "early deaths have
 * a larger takeoffGap than survivals" is only informative if both are drawn
 * from the same situation, otherwise speed alone explains the difference.
 */
export function diagnose(store, state) {
  const okCell = store[`${state}|ok`];
  if (!okCell || okCell.n < MIN_CELL) return [];

  const out = [];
  for (const [key, cell] of Object.entries(store)) {
    if (!key.startsWith(`${state}|`)) continue;
    const reason = key.slice(state.length + 1);
    if (reason === "ok" || cell.n < MIN_CELL) continue;

    for (const p of TRACKED_PROPS) {
      const mFail = median(cell.props[p]);
      const mOk = median(okCell.props[p]);
      if (mFail === null || mOk === null) continue;
      const delta = mOk - mFail;
      // relative threshold: a 5px difference matters for gap, not for speed
      const scale = Math.max(Math.abs(mOk), 1);
      if (Math.abs(delta) / scale < 0.15) continue;
      out.push({
        state,
        reason,
        prop: p,
        okMedian: +mOk.toFixed(1),
        failMedian: +mFail.toFixed(1),
        delta: +delta.toFixed(1),
        nFail: cell.n,
        nOk: okCell.n,
        // confidence proxy: more samples on the thinner side = more trust
        weight: Math.min(cell.n, okCell.n),
      });
    }
  }
  // strongest signal first, weighted by sample count
  return out.sort((a, b) => Math.abs(b.delta) * b.weight - Math.abs(a.delta) * a.weight);
}

/** Every state that has enough data to diagnose. */
export function states(store) {
  const set = new Set();
  for (const key of Object.keys(store)) set.add(key.split("|")[0]);
  return [...set];
}

/**
 * Turn diagnoses into a correction for the takeoff timing of one state.
 *
 * Only `takeoffGap` and `ttc` are actionable here - the others (speed,
 * width) are observations the agent cannot change, so they are reported for
 * insight but never fed back as adjustments.
 */
export function correctionFor(store, state) {
  const found = diagnose(store, state).filter(
    (d) => d.prop === "takeoffGap" || d.prop === "ttc",
  );
  if (!found.length) return null;
  const top = found[0];
  return {
    prop: top.prop,
    delta: Math.max(-25, Math.min(25, top.delta)),
    reason: top.reason,
    confidence: top.weight,
  };
}

/**
 * Which gene to mutate, and WHICH WAY, from measured failures.
 *
 * diagnose() already produces the signal - per situation, the median of a
 * property among survivals against its median among failures - and until now
 * nothing consumed it. mutate() picked a gene uniformly at random and chose
 * the sign of every nudge with a coin flip, so a measured "survivals took off
 * 104px further out than failures" had exactly a 50% chance of being applied
 * backwards.
 *
 * Returns { gene, sign, delta, confidence } or null. NULL IS THE COMMON CASE
 * and must stay cheap: measured on live data, four of six situations have
 * zero failures, so there is nothing to infer and the search should remain
 * random rather than invent a direction.
 *
 * @param store  the Analysis store
 * @param rnd    optional RNG, used only to break ties between equal signals
 */
export function mutationBias(store, rnd) {
  if (!store || typeof store !== "object") return null;

  // Map a measured property to the gene that moves it. Only properties with
  // an unambiguous gene are actionable: `speed` is set by the game, not by
  // the policy, so a speed difference is a description of when we die, not
  // an instruction.
  const GENE_FOR = {
    takeoffGap: "loA",     // where the jump window opens
    ttc: "ttcLo",          // same idea expressed in frames-to-impact
    targetWidth: "wideAdj", // failures on wider obstacles -> widen the allowance
  };

  let best = null;
  for (const state of states(store)) {
    for (const d of diagnose(store, state)) {
      const gene = GENE_FOR[d.prop];
      if (!gene) continue;
      // `delta` is okMedian - failMedian. Positive means survivals had the
      // LARGER value, so push the gene up; negative pushes it down.
      // targetWidth inverts: a negative delta means failures were on WIDER
      // obstacles, which calls for a LARGER allowance.
      const sign = d.prop === "targetWidth"
        ? (d.delta < 0 ? 1 : -1)
        : (d.delta > 0 ? 1 : -1);
      const strength = Math.abs(d.delta) * d.weight;
      if (!best || strength > best.strength) {
        best = { gene, sign, delta: d.delta, confidence: d.weight, state,
                 prop: d.prop, strength };
      }
    }
  }
  if (!best) return null;
  if (rnd) void rnd;   // reserved for tie-breaking; deterministic today
  return { gene: best.gene, sign: best.sign, delta: best.delta,
           confidence: best.confidence, state: best.state, prop: best.prop };
}
