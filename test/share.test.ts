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

  test("REGRESSION: a champion with ZERO measured runs is rejected", () => {
    // MIN_RUNS is 1 by request: one measured run may publish. Zero still
    // cannot - an unmeasured genome has no score to rank.
    const p = goodPayload({ runs: 0 });
    p.evolve.champion.runs = 0;
    expect(share.validate(p).valid).toBe(false);
  });

  test("a single measured run IS publishable", () => {
    const p = goodPayload({ runs: 1, score: 1500, best: 1500 });
    p.evolve.champion.runs = 1;
    p.evolve.champion.mean = 1500;
    p.evolve.champion.best = 1500;
    expect(share.validate(p).valid).toBe(true);
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

  test("REGRESSION: refuses to rank an UNMEASURED champion", () => {
    // runs:0 has no measurement behind it at all.
    expect(share.quality({ champion: { mean: 1466, best: 1466, runs: 0 } })).toBeNull();
  });

  test("one measured run now ranks", () => {
    expect(share.quality({ champion: { mean: 1466, best: 1466, runs: 1 } })).toBe(1466);
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

  test("REGRESSION: refuses to publish an unmeasured champion", () => {
    store.setItem("dino-evolve-v1", JSON.stringify({
      champion: { g: { loA: 25, loB: -1, widthA: 30, widthB: -1, duck: 30 }, mean: 5000, best: 5000, runs: 0 },
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

  test("REGRESSION: an unmeasured champion is refused even with a ledger", () => {
    expect(share.quality({
      champion: { mean: 5000, best: 5000, runs: 0 },
      ledger: [{ hash: "a", best: 1 }, { hash: "b", best: 2 }, { hash: "c", best: 3 }],
    })).toBeNull();
  });

  test("an impossible mean is refused regardless of run count", () => {
    // mean 5000 with best 900 cannot happen; this guard is independent of
    // MIN_RUNS and must survive lowering it.
    expect(share.quality({
      champion: { mean: 5000, best: 900, runs: 1 }, ledger: [],
    })).toBeNull();
  });

  test("an imported champion still must re-earn its bar locally", () => {
    // This is what makes MIN_RUNS=1 safe: a lucky entry that reaches the
    // pool cannot block anyone, because the receiver refuses to treat an
    // imported score as its own until it has measured it here.
    expect(share.quality({
      champion: { mean: 5000, best: 5200, runs: 1, fromPool: true, localRuns: 0 },
      ledger: [],
    })).toBeNull();
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
    // Search history is stripped, but the champion's OWN row travels with
    // it: without that row a receiving client displayed the adopted
    // champion as a fresh 1-run candidate (runs 1 / best 571) while the
    // champion object said runs 8 / best 2900.
    expect(snap!.evolve.ledger).toHaveLength(1);
    expect(snap!.evolve.ledger[0].edition).toBe("win001");
    expect(snap!.evolve.ledger[0].best).toBe(2100);
    expect(snap!.evolve.ledger[0].runs).toBe(4);
    expect(snap!.evolve.ledger[0].fromPool).toBe(true);
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

describe("local corruption cannot block the pool forever", () => {
  test("REGRESSION: an impossible LOCAL mean scores null, not 5000", () => {
    // Measured live: a client that adopted the poisoned few01 before the
    // wire guard existed logged "keeping local (5000pts avg >= 2600)" and
    // refused every real pool entry. The guard must apply where local state
    // is READ, not only where a payload arrives.
    const poisoned = { champion: { mean: 5000, best: 900, runs: 6 }, ledger: [] };
    expect(share.quality(poisoned)).toBeNull();
  });

  test("a legitimate champion is unaffected", () => {
    expect(share.quality({ champion: { mean: 1900, best: 2287, runs: 4 }, ledger: [] })).toBe(1900);
  });
});

describe("an adopted snapshot must not crash the page", () => {
  test("REGRESSION: the published payload carries every array the UI reads", () => {
    // Live crash: "Cannot read properties of undefined (reading 'length')"
    // at drawChart -> S.history.length. The lean payload stripped history,
    // and because drawChart() runs at MODULE SCOPE the throw killed every
    // statement after it - including the startup shared-best pull, which is
    // why the pool appeared to never reach the client.
    store.setItem("dino-evolve-v1", JSON.stringify({
      version: 1,
      champion: {
        g: { loA: 25, loB: -1, widthA: 30, widthB: -1, duck: 30 },
        hash: "h", edition: "e1", mean: 1400, best: 1600, runs: 4,
      },
      ledger: [], history: [], population: [],
    }));
    const snap = share.snapshot();
    expect(snap).not.toBeNull();
    for (const k of ["ledger", "population", "history"]) {
      expect(Array.isArray((snap!.evolve as any)[k])).toBe(true);
    }
    expect(snap!.evolve.inProgress).toBeNull();
  });

  test("a payload that omits history still installs without throwing", () => {
    const remote = goodPayload();
    delete (remote.evolve as any).history;
    const res = share.install(remote);
    expect(res.ok).toBe(true);
    const stored = JSON.parse(store.getItem("dino-evolve-v1")!);
    // install stores what it was given; evolve.js migration repairs the shape
    expect(stored.champion.edition).toBe("ab12cd");
  });
});

describe("an imported champion must earn its bar locally", () => {
  test("REGRESSION: install marks the champion as fromPool with zero local stats", () => {
    // Measured: an adopted champion imported mean 1625 earned on another
    // machine with that machine's warm model. Locally the model restarts
    // colder, so 24 episodes produced a best run of 1315 and NOTHING could
    // ever beat the bar - the champion was undisplaceable and the pool
    // edition name never changed.
    const res = share.install(goodPayload());
    expect(res.ok).toBe(true);
    const stored = JSON.parse(store.getItem("dino-evolve-v1")!);
    expect(stored.champion.fromPool).toBe(true);
    expect(stored.champion.localRuns).toBe(0);
    expect(stored.champion.localMean).toBe(0);
    // the imported mean is preserved for display, just not trusted as the bar
    expect(stored.champion.mean).toBe(1500);
  });
});

describe("an imported champion is not our score until measured here", () => {
  test("REGRESSION: fromPool with no local runs contributes no score", () => {
    // Measured live: fromPool true, localRuns 0, this device's own best was
    // 1220pts, yet quality() returned the imported 2350 and the UI claimed
    // "yours 2350pts". That republishes another device's number as ours.
    expect(share.quality({
      champion: { mean: 2350, best: 2900, runs: 6, fromPool: true, localRuns: 0, localMean: 0 },
      ledger: [],
    })).toBeNull();
  });

  test("once measured locally it ranks on the LOCAL mean", () => {
    expect(share.quality({
      champion: { mean: 2350, best: 2900, runs: 9, fromPool: true, localRuns: 3, localMean: 1400 },
      ledger: [],
    })).toBe(1400);
  });

  test("a locally evolved champion is unaffected", () => {
    expect(share.quality({
      champion: { mean: 1800, best: 2100, runs: 4 }, ledger: [],
    })).toBe(1800);
  });
});

describe("an explicit sync is never skipped", () => {
  test("REGRESSION: the force path exists and is wired to the throttle", () => {
    // Function.length is 1 because `force = false` is a DEFAULT parameter -
    // asserting arity proved nothing. Assert the source contract instead:
    // the unchanged-quality short-circuit must be guarded by !force, or an
    // explicit Stop would be silently skipped 4 times out of 5.
    const src = String(share.syncNow);
    expect(src).toContain("force");
    expect(/lastSyncedQuality\s*&&\s*!force/.test(src)).toBe(true);
  });

  test("unconfigured sync fails cleanly rather than throwing", async () => {
    const r = await share.syncNow(undefined, true);
    expect(r.ok).toBe(false);
    expect(r.why).toBe("not configured");
  });
});
