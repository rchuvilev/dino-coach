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
