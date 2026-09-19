/**
 * Jump-outcome model with TRAINABLE WEIGHTS, updated after every resolved
 * jump and every failed run.
 *
 * Why this exists alongside knn.js: KNN is instance-based. It MEMORISES
 * examples and votes over neighbours - there are no weights, nothing to
 * correct, and "learning" is just appending to an array. The request was to
 * train on each success and each failure and correct the model, which means
 * a parametric model updated by gradient descent. That is this file.
 *
 * Measured in-browser before building (tfjs 4.22.0, cpu backend):
 *   trainOnBatch, 1 sample   4.99 ms
 *   predict, 1 sample        0.80 ms
 *   online-learned separable rule: lr 0.2 -> 0.99, lr 0.01 -> 0.97
 * The first attempt at lr 0.05 scored 0.80, which was underfitting rather
 * than a broken harness - confirmed by the sweep above.
 *
 * 4.99 ms does NOT fit inside a 16.7 ms frame that also runs the game, so
 * training is deliberately deferred to episode boundaries and run in small
 * batches rather than per frame. Prediction at 0.80 ms is affordable in the
 * decision path.
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

const tfg = () => (typeof window !== "undefined" ? window.tf : undefined);

let ready = null;
function ensureBackend() {
  const tf = tfg();
  if (!tf) return Promise.resolve(false);
  if (!ready) {
    ready = tf
      .setBackend("cpu")
      .then(() => tf.ready())
      .then(() => true)
      .catch(() => tf.ready().then(() => true));
  }
  return ready;
}

export class JumpModel {
  constructor() {
    this.net = null;
    this.pending = [];       // outcomes awaiting a weight update
    this.trained = 0;        // total samples the weights have seen
    this.lastLoss = null;
    this.building = false;
  }

  get samplesTrained() {
    return this.trained;
  }

  async build() {
    const tf = tfg();
    if (!tf || this.net || this.building) return !!this.net;
    this.building = true;
    await ensureBackend();
    const m = tf.sequential();
    m.add(tf.layers.dense({ units: 8, activation: "relu", inputShape: [4] }));
    m.add(tf.layers.dense({ units: 1, activation: "sigmoid" }));
    // lr 0.2 measured best for online single-sample updates (0.99 vs 0.80
    // at 0.05); high for batch training, correct for one-sample-at-a-time.
    m.compile({ optimizer: tf.train.adam(0.2), loss: "binaryCrossentropy" });
    this.net = m;
    this.building = false;
    return true;
  }

  /**
   * Record one outcome. Cheap and synchronous - the weight update happens
   * later in flush(), because trainOnBatch costs 4.99 ms and the tick loop
   * has 16.7 ms total.
   */
  observe(features, ok) {
    this.pending.push({ x: features, y: ok ? 1 : 0 });
    if (this.pending.length > 256) this.pending.shift();
  }

  /**
   * Apply the queued outcomes to the weights. Called at episode boundaries.
   * Returns the number of samples trained on.
   */
  async flush(maxBatch = 32) {
    const tf = tfg();
    if (!tf || !this.pending.length) return 0;
    if (!this.net && !(await this.build())) return 0;

    const batch = this.pending.splice(0, maxBatch);
    const xs = tf.tensor2d(batch.map((b) => b.x));
    const ys = tf.tensor2d(batch.map((b) => [b.y]));
    try {
      const h = await this.net.trainOnBatch(xs, ys);
      this.lastLoss = Array.isArray(h) ? h[0] : h;
      this.trained += batch.length;
      return batch.length;
    } catch {
      return 0;
    } finally {
      xs.dispose();
      ys.dispose();
    }
  }

  /**
   * P(this jump clears the obstacle). Synchronous read of a tiny tensor;
   * measured 0.80 ms, which fits the decision path.
   *
   * Returns null until the weights have seen enough to be worth consulting -
   * an untrained sigmoid outputs ~0.5 everywhere and would only add noise.
   */
  predictSync(features, minTrained = 40) {
    const tf = tfg();
    if (!tf || !this.net || this.trained < minTrained) return null;
    let out = null;
    try {
      const x = tf.tensor2d([features]);
      const p = this.net.predict(x);
      out = p.dataSync()[0];
      p.dispose();
      x.dispose();
    } catch {
      return null;
    }
    return out;
  }

  /**
   * Weights as plain arrays, so the model persists like every other store.
   * tfjs model.save needs an IO handler; for a 4-8-1 net the raw weights are
   * smaller and simpler than any of them.
   */
  toJSON() {
    if (!this.net) return { trained: this.trained, weights: null };
    try {
      return {
        trained: this.trained,
        weights: this.net.getWeights().map((w) => ({
          shape: w.shape,
          data: Array.from(w.dataSync()),
        })),
      };
    } catch {
      return { trained: this.trained, weights: null };
    }
  }

  async loadFrom(o) {
    if (!o || !o.weights) return false;
    const tf = tfg();
    if (!tf) return false;
    if (!(await this.build())) return false;
    try {
      this.net.setWeights(o.weights.map((w) => tf.tensor(w.data, w.shape)));
      this.trained = o.trained || 0;
      return true;
    } catch {
      return false;
    }
  }

  dispose() {
    try {
      if (this.net) this.net.dispose();
    } catch {
      /* ignore */
    }
    this.net = null;
    this.pending = [];
    this.trained = 0;
    this.lastLoss = null;
  }
}
