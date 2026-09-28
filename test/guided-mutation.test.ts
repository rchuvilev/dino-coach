import { describe, expect, test } from "bun:test";
import { mutationBias } from "../examples/dino/viewer/analysis.js";

/**
 * Turning measured failures into a mutation DIRECTION.
 *
 * analysis.js already computes the signal - diagnose() returns, per
 * situation, the median of a property among survivals against its median
 * among failures. Measured live on this project:
 *
 *   wide/fast/open   · early: takeoffGap ok@114 vs fail@10   (+104, n=16/516)
 *   narrow/fast/open · early: takeoffGap ok@114 vs fail@4    (+110)
 *   narrow/fast/open · early: targetWidth ok@34 vs fail@46   (-12)
 *
 * The third row says failures happen on WIDER obstacles, i.e. raise wideAdj.
 * None of it reached mutate(), which picked a gene uniformly at random and
 * chose the sign of every nudge with a coin flip. This bridges the two.
 */

/** Build an analysis store the way Analysis.record() does. */
function store(state: string, okProps: Record<string, number[]>, failProps: Record<string, number[]>) {
  return {
    [`${state}|ok`]: { n: okProps.takeoffGap?.length ?? 0, props: okProps },
    [`${state}|early`]: { n: failProps.takeoffGap?.length ?? 0, props: failProps },
  };
}

const many = (v: number, n = 12) => Array.from({ length: n }, () => v);

describe("mutationBias turns failure medians into a gene + direction", () => {
  test("REGRESSION: a late-takeoff failure asks for a LONGER window", () => {
    // survivals took off at 114px, failures at 10px -> jump earlier.
    const s = store("wide/fast/open",
      { takeoffGap: many(114), ttc: many(10.7) },
      { takeoffGap: many(10), ttc: many(0.9) });
    const b = mutationBias(s);
    expect(b).not.toBeNull();
    expect(b!.gene).toBe("loA");
    expect(b!.sign).toBe(1);          // increase the takeoff distance
  });

  test("REGRESSION: the opposite evidence asks for the opposite direction", () => {
    // Guards against a bias that always points one way, which would be
    // indistinguishable from a constant and would pass the test above.
    const s = store("wide/fast/open",
      { takeoffGap: many(20), ttc: many(2) },
      { takeoffGap: many(120), ttc: many(11) });
    const b = mutationBias(s);
    expect(b).not.toBeNull();
    expect(b!.gene).toBe("loA");
    expect(b!.sign).toBe(-1);
  });

  test("failures on wider obstacles point at wideAdj", () => {
    const s = store("narrow/fast/open",
      { takeoffGap: many(114), targetWidth: many(34) },
      { takeoffGap: many(112), targetWidth: many(46) });
    const b = mutationBias(s);
    expect(b).not.toBeNull();
    expect(b!.gene).toBe("wideAdj");
    expect(b!.sign).toBe(1);          // failures are on wider cacti
  });

  test("REGRESSION: a thin cell yields no bias rather than a guess", () => {
    // MIN_CELL is 6. Acting on 2 samples is how a noisy fluke becomes a
    // systematic push in the wrong direction.
    const s = store("wide/mid/open",
      { takeoffGap: many(100, 2) }, { takeoffGap: many(10, 2) });
    expect(mutationBias(s)).toBeNull();
  });

  test("REGRESSION: no failures at all yields no bias", () => {
    // Measured: four of six live contexts have ZERO failures, so the
    // common case must be "stay random", not "invent a direction".
    const s = { "narrow/slow/open|ok": { n: 1436, props: { takeoffGap: many(94, 40) } } };
    expect(mutationBias(s)).toBeNull();
  });

  test("an empty store is safe", () => {
    expect(mutationBias({})).toBeNull();
    expect(mutationBias(null as any)).toBeNull();
  });

  test("a difference below the relative threshold is ignored", () => {
    // diagnose() requires >=15% relative difference; 100 vs 97 is noise.
    const s = store("wide/fast/open",
      { takeoffGap: many(100) }, { takeoffGap: many(97) });
    expect(mutationBias(s)).toBeNull();
  });
});
