import { describe, expect, test } from "bun:test";

/**
 * Class imbalance in the takeoff learners.
 *
 * MEASURED on a live session before this work: 2592 successes against 12
 * failures (0.46% failure rate), and the KNN held 594 positives to 6
 * negatives. A classifier whose only job is spotting the jumps that FAIL
 * scores 99.5% accuracy by predicting "clears" unconditionally, so there is
 * almost no gradient pressure toward recognising failure.
 *
 * Two independent defects produce that:
 *   1. flush() weights every sample equally, so 64 samples of which 63 are
 *      positive teach the network to say yes.
 *   2. the KNN evicts oldest-first, and because failures are rare they are
 *      evicted while abundant successes survive - the ratio gets WORSE the
 *      longer a session runs.
 */

import { classWeights } from "../examples/dino/viewer/mlp.js";
import { TakeoffKNN } from "../examples/dino/viewer/knn.js";

describe("class weighting counteracts the measured imbalance", () => {
  test("REGRESSION: a rare class is up-weighted, the common one is not", () => {
    // The real ratio seen in play: 63 ok, 1 fail in a 64-sample batch.
    const w = classWeights([...Array(63).fill(1), 0]);
    expect(w).not.toBeNull();
    // the failure must count for far more than one success
    expect(w![0]).toBeGreaterThan(w![1] * 10);
  });

  test("a balanced batch is left alone", () => {
    const w = classWeights([1, 0, 1, 0]);
    expect(w).not.toBeNull();
    expect(w![0]).toBeCloseTo(w![1], 5);
  });

  test("a single-class batch yields no weights rather than dividing by zero", () => {
    expect(classWeights([1, 1, 1])).toBeNull();
    expect(classWeights([])).toBeNull();
  });

  test("weights are bounded - one stray failure cannot dominate a batch", () => {
    // Without a cap, 1 failure in 1000 would be weighted 999x and a single
    // mislabelled sample would swamp the update.
    const w = classWeights([...Array(999).fill(1), 0]);
    expect(w![0]).toBeLessThanOrEqual(20);
  });
});

describe("the KNN must not evict its rare class away", () => {
  test("REGRESSION: failures survive a flood of successes", () => {
    // Live measurement: 594 positives vs 6 negatives at the 600 cap.
    const k = new TakeoffKNN(60);
    k.add([0.1, 0.2, 0.3, 0.4], false, 40);   // one precious failure
    for (let i = 0; i < 200; i++) k.add([0.5, 0.5, 0.5, 0.5], true, 60);
    const negatives = k.ys.filter((y: number) => y === 0).length;
    expect(negatives).toBeGreaterThan(0);
  });

  test("the cap is still respected", () => {
    const k = new TakeoffKNN(50);
    for (let i = 0; i < 300; i++) k.add([0.5, 0.5, 0.5, 0.5], i % 2 === 0, 60);
    expect(k.xs.length).toBeLessThanOrEqual(50);
    expect(k.ys.length).toBe(k.xs.length);
    expect(k.gaps.length).toBe(k.xs.length);
  });

  test("a majority sample is still evicted when the buffer is full", () => {
    const k = new TakeoffKNN(20);
    for (let i = 0; i < 100; i++) k.add([i / 100, 0.5, 0.5, 0.5], true, 60);
    expect(k.xs.length).toBe(20);
    // oldest majority samples went first: the first feature value rose
    expect(k.xs[0][0]).toBeGreaterThan(0);
  });
});
