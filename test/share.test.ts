import { describe, expect, test, beforeEach } from "bun:test";

/**
 * Tests for the shared-training sync.
 *
 * The rule these all serve: NOTHING that crosses the network may be installed
 * unchecked. A malformed payload reaching tfjs breaks startup for that visitor
 * with no way back, so every guard here is written to fail first.
 */

// Minimal localStorage so the module under test can run outside a browser.
class MemStore {
  private m = new Map<string, string>();
  getItem(k: string) { return this.m.has(k) ? this.m.get(k)! : null; }
  setItem(k: string, v: string) { this.m.set(k, String(v)); }
  removeItem(k: string) { this.m.delete(k); }
  clear() { this.m.clear(); }
  get size() { return this.m.size; }
  keys() { return [...this.m.keys()]; }
}

const store = new MemStore();
(globalThis as any).localStorage = store;
(globalThis as any).window = globalThis;

const share = await import("../examples/dino/viewer/share.js");

/** A payload that passes validation, so each test can break exactly one thing. */
function goodPayload(over: Record<string, any> = {}) {
  const champion = {
    g: { loA: 25, loB: -1.2, widthA: 31, widthB: -1.4, duck: 39 },
    hash: "abc123",
    edition: "ab12cd",
    best: 1800,
    mean: 1500,
    runs: 5,
  };
  return {
    v: 1,
    score: 1500,
    best: 1800,
    runs: 5,
    edition: "ab12cd",
    at: Date.now(),
    evolve: { champion, episodes: 20 },
    mlp: { dim: 384, trained: 400 },
    knn: { xs: [] },
    ...over,
  };
}

beforeEach(() => store.clear());

describe("validate rejects anything malformed", () => {
  test("a well-formed payload passes", () => {
    expect(share.validate(goodPayload()).valid).toBe(true);
  });

  test("REGRESSION: a schema version mismatch is rejected", () => {
    const r = share.validate(goodPayload({ v: 2 }));
    expect(r.valid).toBe(false);
    expect(r.reason).toContain("schema");
  });

  test("REGRESSION: a genome missing a field the policy reads is rejected", () => {
    // A missing loA yields NaN windows, which read as "never jump" - a silent
    // behavioural failure indistinguishable from a bad strategy.
    const p = goodPayload();
    delete (p.evolve.champion.g as any).loA;
    const r = share.validate(p);
    expect(r.valid).toBe(false);
    expect(r.reason).toContain("loA");
  });

  test("REGRESSION: a score disagreeing with its own champion is rejected", () => {
    // Guards against a hand-crafted payload claiming a huge score to win the
    // ranking while carrying a weak genome.
    const r = share.validate(goodPayload({ score: 99999 }));
    expect(r.valid).toBe(false);
    expect(r.reason).toContain("disagrees");
  });

  test("REGRESSION: too few runs is rejected - a mean over 1 run is luck", () => {
    const p = goodPayload({ runs: 1 });
    p.evolve.champion.runs = 1;
    expect(share.validate(p).valid).toBe(false);
  });

  test("REGRESSION: an MLP of the wrong width is rejected", () => {
    const r = share.validate(goodPayload({ mlp: { dim: 128 } }));
    expect(r.valid).toBe(false);
    expect(r.reason).toContain("dim");
  });

  test("null, non-objects and arrays are rejected, not thrown on", () => {
    for (const bad of [null, undefined, 42, "x", [], true]) {
      expect(share.validate(bad as any).valid).toBe(false);
    }
  });

  test("a missing model is allowed - a fresh champion may have neither", () => {
    expect(share.validate(goodPayload({ mlp: null, knn: null })).valid).toBe(true);
  });
});

describe("quality ranks by mean, never by the lucky record", () => {
  test("uses the mean, not best", () => {
    const q = share.quality({ champion: { mean: 500, best: 9000, runs: 5 } });
    expect(q).toBe(500);
  });

  test("REGRESSION: refuses to rank a champion with too few runs", () => {
    // The measured failure: runs:1 best:1466 froze an unbeatable bar.
    expect(share.quality({ champion: { mean: 1466, best: 1466, runs: 1 } })).toBeNull();
  });

  test("missing champion yields null rather than throwing", () => {
    expect(share.quality(null)).toBeNull();
    expect(share.quality({})).toBeNull();
  });
});

describe("install replaces ALL local training", () => {
  test("a better remote wipes local analysis, logs and slots", () => {
    // The defect this catches: installing a foreign genome while keeping the
    // local per-situation failure statistics produces a hybrid that belongs
    // to neither user, and ctxAdj would then correct the WRONG genome.
    store.setItem("dino-analysis-v1", JSON.stringify({ "wide/slow/open|early": { n: 99 } }));
    store.setItem("dino-logs-v1", JSON.stringify({ entries: [1, 2, 3] }));
    store.setItem("dino-evolve-slot-v1", JSON.stringify({ stale: true }));
    store.setItem("dino-mlp-v1", JSON.stringify({ dim: 384, trained: 1 }));

    const res = share.install(goodPayload());
    expect(res.ok).toBe(true);

    expect(store.getItem("dino-analysis-v1")).toBeNull();
    expect(store.getItem("dino-logs-v1")).toBeNull();
    expect(store.getItem("dino-evolve-slot-v1")).toBeNull();
    // and the remote model did land
    expect(JSON.parse(store.getItem("dino-mlp-v1")!).trained).toBe(400);
  });

  test("REGRESSION: an invalid payload installs nothing at all", () => {
    store.setItem("dino-evolve-v1", JSON.stringify({ mine: true }));
    const res = share.install(goodPayload({ v: 7 }));
    expect(res.ok).toBe(false);
    // local state untouched
    expect(JSON.parse(store.getItem("dino-evolve-v1")!).mine).toBe(true);
  });

  test("a remote without a KNN clears any stale local KNN", () => {
    // Otherwise the new genome inherits the previous user's neighbourhood.
    store.setItem("dino-knn-v1", JSON.stringify({ xs: [[1, 2, 3]] }));
    const res = share.install(goodPayload({ knn: null }));
    expect(res.ok).toBe(true);
    expect(store.getItem("dino-knn-v1")).toBeNull();
  });
});

describe("snapshot refuses to publish junk", () => {
  test("returns null when there is nothing worth sharing", () => {
    expect(share.snapshot()).toBeNull();
  });

  test("REGRESSION: refuses to publish a champion with too few runs", () => {
    store.setItem("dino-evolve-v1", JSON.stringify({
      champion: { g: { loA: 25, loB: -1, widthA: 30, widthB: -1, duck: 30 }, mean: 5000, best: 5000, runs: 1 },
    }));
    expect(share.snapshot()).toBeNull();
  });

  test("publishes a well-measured champion, and it validates", () => {
    store.setItem("dino-evolve-v1", JSON.stringify({
      champion: {
        g: { loA: 25, loB: -1, widthA: 30, widthB: -1, duck: 30 },
        mean: 1200, best: 1500, runs: 4, edition: "zz99xx",
      },
    }));
    const snap = share.snapshot();
    expect(snap).not.toBeNull();
    expect(snap!.score).toBe(1200);
    // round-trip: what we publish must be what a reader accepts
    expect(share.validate(JSON.parse(JSON.stringify(snap))).valid).toBe(true);
  });
});

describe("quality accepts corroborated champions (measured: 96% never reach 3 runs)", () => {
  test("a fresh champion with a scored ledger IS publishable", () => {
    // Measured on a real run: only 2 of 53 editions ever reached runs>=3,
    // because `runs` resets to 1 on promotion. Gating on the current streak
    // alone blocked ~96% of publishing and left the pool empty.
    const q = share.quality({
      champion: { mean: 1400, best: 1600, runs: 1 },
      // real ledger rows carry `best`, not `score` - a fixture with the
      // wrong field name passed while the production filter matched 0 rows
      ledger: [
        { hash: "a", edition: "a", best: 900, runs: 1 },
        { hash: "b", edition: "b", best: 1100, runs: 2 },
        { hash: "c", edition: "c", best: 1400, runs: 1 },
      ],
    });
    expect(q).toBe(1400);
  });

  test("REGRESSION: a champion with NO corroboration is still refused", () => {
    expect(share.quality({ champion: { mean: 5000, best: 5000, runs: 1 }, ledger: [] })).toBeNull();
  });

  test("REGRESSION: a thin ledger does not corroborate", () => {
    const q = share.quality({
      champion: { mean: 5000, best: 5000, runs: 1 },
      ledger: [{ hash: "a", best: 10 }, { hash: "b", best: 20 }],
    });
    expect(q).toBeNull();
  });

  test("REGRESSION: ledger entries without a score do not count", () => {
    // Guards against padding the ledger with unscored rows to force a publish.
    const q = share.quality({
      champion: { mean: 5000, best: 5000, runs: 1 },
      ledger: [{ edition: "a" }, { edition: "b" }, { edition: "c" }],
    });
    expect(q).toBeNull();
  });

  test("a zero or negative mean is never publishable", () => {
    expect(share.quality({ champion: { mean: 0, runs: 9 }, ledger: [] })).toBeNull();
  });
});

describe("an impossible mean cannot enter the pool", () => {
  test("REGRESSION: the exact payload that poisoned the live pool is rejected", () => {
    // Real incident, recovered from a client HAR: a hand-written test payload
    // with mean 5000 but best 900 reached Redis, was pulled by a real client,
    // and blocked every genuine publish - that client's own training (ledger
    // best 2287, 11635 trained samples) could never beat a fabricated 5000.
    const poison = goodPayload({ score: 5000, best: 900 });
    poison.evolve.champion.mean = 5000;
    poison.evolve.champion.best = 900;
    const r = share.validate(poison);
    expect(r.valid).toBe(false);
    expect(r.reason).toContain("exceeds best");
  });

  test("a mean equal to best is fine (one measured run)", () => {
    const p = goodPayload({ score: 1500, best: 1500 });
    p.evolve.champion.mean = 1500;
    p.evolve.champion.best = 1500;
    expect(share.validate(p).valid).toBe(true);
  });

  test("a normal mean below best stays valid", () => {
    expect(share.validate(goodPayload()).valid).toBe(true);
  });
});

describe("syncing is not throttled", () => {
  test("REGRESSION: no time-based floor blocks a sync", () => {
    // The user asked for every run to sync; a 15s floor silently skipped
    // publishes and was indistinguishable from a broken integration.
    const src = share._internals;
    expect(Object.keys(src)).not.toContain("SYNC_FLOOR_MS");
  });
});

describe("the pool carries the champion only", () => {
  beforeEach(() => store.clear());

  test("publishes the champion and drops local search history", () => {
    store.setItem("dino-evolve-v1", JSON.stringify({
      version: 1,
      champion: {
        g: { loA: 25, loB: -1, widthA: 30, widthB: -1, duck: 30 },
        hash: "h1", edition: "win001", mean: 1800, best: 2100, runs: 4,
      },
      // all of this is THIS browser's search history, not transferable skill
      ledger: Array.from({ length: 60 }, (_, i) => ({ hash: "x" + i, best: 100 + i })),
      population: [{ junk: true }, { junk: true }],
      inProgress: [{ g: {} }],
      sessions: 12,
      knowledge: { "narrow/slow/open": { ok: 10, fail: 1 } },
    }));
    const snap = share.snapshot();
    expect(snap).not.toBeNull();
    expect(snap!.score).toBe(1800);
    expect(snap!.evolve.champion.edition).toBe("win001");
    // history is stripped
    expect(snap!.evolve.ledger).toEqual([]);
    expect(snap!.evolve.population).toEqual([]);
    expect(snap!.evolve.inProgress).toBeNull();
    // but transferable per-situation knowledge rides along
    expect(snap!.evolve.knowledge["narrow/slow/open"].ok).toBe(10);
    // and it still validates for a receiver
    expect(share.validate(JSON.parse(JSON.stringify(snap))).valid).toBe(true);
  });

  test("REGRESSION: adopting wipes every local key, not just the models", () => {
    for (const k of ["dino-evolve-v1", "dino-mlp-v1", "dino-knn-v1",
                     "dino-analysis-v1", "dino-logs-v1", "dino-evolve-slot-v1"]) {
      store.setItem(k, JSON.stringify({ mine: true }));
    }
    const res = share.install(goodPayload());
    expect(res.ok).toBe(true);
    // only what the remote supplied survives
    expect(store.getItem("dino-analysis-v1")).toBeNull();
    expect(store.getItem("dino-logs-v1")).toBeNull();
    expect(store.getItem("dino-evolve-slot-v1")).toBeNull();
    expect(JSON.parse(store.getItem("dino-evolve-v1")!).champion.edition).toBe("ab12cd");
  });
});
