/**
 * Continuous-feature KNN over takeoff outcomes — the Teachable Machine
 * approach, implemented directly rather than via tfjs.
 *
 * Why not @tensorflow-models/knn-classifier: it pulls tf.min.js at 1.47MB
 * against a 134KB self-contained page — an 11x blowup for what KNN actually
 * needs here, which is euclidean distance plus a top-k vote. Measured
 * earlier in this project, that library ran 1.26ms/call; this does the same
 * work on 4 dimensions in microseconds and keeps the file offline-capable.
 *
 * What it replaces: hard situation buckets (`wide/slow/open`). Those put
 * speed 7.9 and 8.1 in different classes sharing nothing, while treating
 * speed 6.0 and 7.9 as identical. Distance-weighted neighbours handle the
 * continuum properly.
 */

/** Feature scales, so no single axis dominates the distance metric. */
const SCALE = { gap: 60, speed: 6, width: 50, next: 120 };

export function featurize(s) {
  return [
    (s.gap ?? 0) / SCALE.gap,
    (s.speed ?? 6) / SCALE.speed,
    (s.width ?? 0) / SCALE.width,
    (s.next === null || s.next === undefined ? 999 : s.next) / SCALE.next,
  ];
}

export class TakeoffKNN {
  /**
   * @param {number} cap  max stored examples; oldest are evicted so the
   *                      model tracks the CURRENT policy rather than
   *                      accumulating outcomes from long-dead genomes.
   */
  constructor(cap = 600) {
    this.cap = cap;
    this.xs = [];   // feature vectors
    this.ys = [];   // 1 = cleared, 0 = failed
    this.gaps = []; // the takeoff gap used, for deriving a correction
  }

  get size() {
    return this.xs.length;
  }

  /** Incremental: one observed takeoff, no retraining. */
  add(features, ok, gap) {
    this.xs.push(features);
    this.ys.push(ok ? 1 : 0);
    this.gaps.push(gap);
    if (this.xs.length > this.cap) {
      this.xs.shift();
      this.ys.shift();
      this.gaps.shift();
    }
  }

  /**
   * Distance-weighted vote over the k nearest takeoffs.
   * Returns null when there is too little evidence to speak, which matters:
   * a confident answer from 2 neighbours is worse than no answer.
   */
  predict(features, k = 12, minN = 20) {
    if (this.xs.length < minN) return null;
    const n = this.xs.length;
    const dists = new Array(n);
    for (let i = 0; i < n; i++) {
      const v = this.xs[i];
      let d = 0;
      for (let j = 0; j < features.length; j++) {
        const diff = features[j] - v[j];
        d += diff * diff;
      }
      dists[i] = [d, i];
    }
    dists.sort((a, b) => a[0] - b[0]);
    const kk = Math.min(k, n);

    let wOk = 0;
    let wFail = 0;
    let okGapSum = 0;
    let okGapW = 0;
    let failGapSum = 0;
    let failGapW = 0;

    for (let i = 0; i < kk; i++) {
      const [d2, idx] = dists[i];
      // inverse-distance weight, floored so an exact match cannot divide by 0
      const w = 1 / (Math.sqrt(d2) + 0.05);
      if (this.ys[idx] === 1) {
        wOk += w;
        okGapSum += this.gaps[idx] * w;
        okGapW += w;
      } else {
        wFail += w;
        failGapSum += this.gaps[idx] * w;
        failGapW += w;
      }
    }

    const total = wOk + wFail;
    if (total <= 0) return null;
    return {
      pOk: wOk / total,
      okGap: okGapW > 0 ? okGapSum / okGapW : null,
      failGap: failGapW > 0 ? failGapSum / failGapW : null,
      neighbours: kk,
    };
  }

  /**
   * Suggested takeoff-gap correction for this situation.
   *
   * Only speaks when the local neighbourhood contains BOTH outcomes and the
   * failure rate is material — otherwise there is nothing to correct and a
   * suggestion would just add noise, which is how the previous hard-bucket
   * version produced offsets from single failures.
   */
  suggest(features, k = 12) {
    const p = this.predict(features, k);
    if (!p || p.okGap === null || p.failGap === null) return null;
    if (p.pOk > 0.9) return null;          // already reliable here
    const delta = p.okGap - p.failGap;
    if (Math.abs(delta) < 3) return null;  // below measurement resolution
    return { delta: Math.max(-25, Math.min(25, delta)), pOk: p.pOk };
  }

  toJSON() {
    return { cap: this.cap, xs: this.xs, ys: this.ys, gaps: this.gaps };
  }

  static fromJSON(o) {
    const m = new TakeoffKNN(o && o.cap ? o.cap : 600);
    if (o && Array.isArray(o.xs)) {
      m.xs = o.xs;
      m.ys = o.ys || [];
      m.gaps = o.gaps || [];
    }
    return m;
  }

  clear() {
    this.xs = [];
    this.ys = [];
    this.gaps = [];
  }
}
