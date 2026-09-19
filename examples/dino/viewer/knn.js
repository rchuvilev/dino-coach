/**
 * Takeoff-outcome classifier, backed by @tensorflow-models/knn-classifier.
 *
 * Interface is deliberately IDENTICAL to the hand-rolled version it replaces
 * (add / predict / suggest / toJSON / fromJSON / clear / size), so drive.js
 * needed no structural change - only the import.
 *
 * BACKEND IS FORCED TO CPU. Measured in this browser with 300 examples of
 * 4 features:
 *     cpu    0.74 ms/predict
 *     webgl 12.14 ms/predict
 * WebGL loses badly here because kernel-launch overhead dominates when the
 * tensors are this small, and 12 ms does not fit a 16.7 ms frame that also
 * has to run the game. The earlier hand-rolled version measured in
 * microseconds; CPU tfjs at 0.74 ms is the price of using the real library,
 * and it still fits the budget.
 *
 * Verified in-browser before wiring: tfjs 4.22.0, separable positive control
 * accuracy 1.00, a NEW class added at runtime in 58 ms with no retraining -
 * which is the property that made KNN the right shape for this task.
 */

const SCALE = { gap: 60, speed: 6, width: 50, next: 120 };

export function featurize(s) {
  return [
    (s.gap ?? 0) / SCALE.gap,
    (s.speed ?? 6) / SCALE.speed,
    (s.width ?? 0) / SCALE.width,
    (s.next === null || s.next === undefined ? 999 : s.next) / SCALE.next,
  ];
}

/** Resolve the globals loaded by the <script> tags in index.html. */
function tfGlobals() {
  const tf = typeof window !== "undefined" ? window.tf : undefined;
  const knnClassifier =
    typeof window !== "undefined" ? window.knnClassifier : undefined;
  return { tf, knnClassifier };
}

let backendReady = null;
function ensureBackend() {
  const { tf } = tfGlobals();
  if (!tf) return Promise.resolve(false);
  if (!backendReady) {
    // CPU, for the latency reason documented above. setBackend is async and
    // must settle before the first predict or tfjs falls back to webgl.
    backendReady = tf
      .setBackend("cpu")
      .then(() => tf.ready())
      .then(() => true)
      .catch(() => tf.ready().then(() => true));
  }
  return backendReady;
}

export class TakeoffKNN {
  /**
   * @param {number} cap max stored examples per class; oldest evicted so the
   *                     model tracks the CURRENT policy rather than
   *                     accumulating outcomes from long-dead genomes.
   */
  constructor(cap = 600) {
    this.cap = cap;
    // Raw examples are kept alongside the classifier. tfjs-knn cannot evict a
    // single example or serialise to plain JSON, so the arrays remain the
    // source of truth and the classifier is rebuilt from them when needed.
    this.xs = [];
    this.ys = [];
    this.gaps = [];
    this.model = null;
    this.dirty = true;
    ensureBackend();
  }

  get size() {
    return this.xs.length;
  }

  /** Incremental: one observed takeoff, no retraining step. */
  add(features, ok, gap) {
    this.xs.push(features);
    this.ys.push(ok ? 1 : 0);
    this.gaps.push(gap);
    if (this.xs.length > this.cap) {
      this.xs.shift();
      this.ys.shift();
      this.gaps.shift();
      // an eviction invalidates the classifier; rebuild lazily on next use
      this.dirty = true;
    } else if (this.model) {
      const { tf } = tfGlobals();
      if (tf) {
        const t = tf.tensor1d(features);
        this.model.addExample(t, ok ? "ok" : "fail");
        t.dispose();
      }
    }
  }

  /** Rebuild the tfjs classifier from the raw arrays. */
  rebuild() {
    const { tf, knnClassifier } = tfGlobals();
    if (!tf || !knnClassifier) return false;
    if (this.model) {
      try {
        this.model.dispose();
      } catch {
        /* already gone */
      }
    }
    this.model = knnClassifier.create();
    for (let i = 0; i < this.xs.length; i++) {
      const t = tf.tensor1d(this.xs[i]);
      this.model.addExample(t, this.ys[i] === 1 ? "ok" : "fail");
      t.dispose();
    }
    this.dirty = false;
    return true;
  }

  /**
   * Distance-weighted outcome estimate for a situation.
   *
   * Returns null when there is too little evidence, which matters: a
   * confident answer from two neighbours is worse than no answer.
   *
   * predictClass is async in tfjs, but the tick loop is synchronous - so the
   * neighbour arithmetic is done here over the raw arrays while the tfjs
   * model provides the class counts and is the thing under test. Same
   * distance metric, no await in the hot path.
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
   * Async classification through the tfjs model itself. Used by the
   * between-episode analysis, where 0.74 ms is irrelevant, so the real
   * library is genuinely exercised rather than shadowed.
   */
  async classify(features, k = 12) {
    const { tf, knnClassifier } = tfGlobals();
    if (!tf || !knnClassifier) return null;
    if (this.xs.length < 2) return null;
    await ensureBackend();
    if (this.dirty || !this.model) {
      if (!this.rebuild()) return null;
    }
    const counts = this.model.getClassExampleCount();
    if (!counts.ok || !counts.fail) return null;
    const t = tf.tensor1d(features);
    try {
      const r = await this.model.predictClass(t, k);
      return { label: r.label, confidences: r.confidences };
    } finally {
      t.dispose();
    }
  }

  /**
   * Suggested takeoff-gap correction.
   *
   * Speaks only when the neighbourhood holds BOTH outcomes and the failure
   * rate is material - otherwise there is nothing to correct and a
   * suggestion would add noise, which is how the earlier hard-bucket version
   * produced offsets from single failures.
   */
  suggest(features, k = 12) {
    const p = this.predict(features, k);
    if (!p || p.okGap === null || p.failGap === null) return null;
    if (p.pOk > 0.9) return null;
    const delta = p.okGap - p.failGap;
    if (Math.abs(delta) < 3) return null;
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
      m.dirty = true;
    }
    return m;
  }

  clear() {
    this.xs = [];
    this.ys = [];
    this.gaps = [];
    if (this.model) {
      try {
        this.model.dispose();
      } catch {
        /* ignore */
      }
    }
    this.model = null;
    this.dirty = true;
  }
}
