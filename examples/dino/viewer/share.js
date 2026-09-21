/**
 * Shared best training over Upstash Redis.
 *
 * Every visitor pulls the best training the pool has seen on load, and pushes
 * its own when it is genuinely better. The result is that a new tab does not
 * start from zero: it starts from the best policy anyone has evolved.
 *
 * RANKING KEY IS THE MEAN, NOT THE RECORD. Measured in this project: one
 * genome re-run 12 times spanned 6930..48436, and a champion sat at runs:1
 * best:1466 while six challengers scoring 181..800 were all rejected. A
 * single lucky run is therefore not evidence of a better policy, and ranking
 * the global pool by `best` would import exactly that fluke and freeze an
 * unbeatable bar for every other user. We rank by the champion's running mean
 * and require MIN_RUNS measurements before a candidate may be published.
 *
 * EVERYTHING CROSSING THE NETWORK IS UNTRUSTED. A payload written by an older
 * build, a different schema version, or a hand-crafted request must never be
 * installed into localStorage unchecked: the model is loaded straight into
 * tfjs and a malformed shape would break startup for that visitor with no way
 * back. Hence validate() below, applied to BOTH directions.
 *
 * The REST API is used directly rather than @upstash/redis because the page is
 * a single self-contained file with no bundler at runtime.
 */

const KEY = "dino:best:v1";
const SCHEMA = 1;          // bump when the payload shape changes incompatibly
const MIN_RUNS = 3;        // a mean over fewer runs is still mostly luck
const MAX_BYTES = 400000;  // Upstash free tier allows 1MB/entry; stay well under
const MLP_DIM = 384;       // feature width the model is built for

/** Read config injected at build time or set by the host page. */
function cfg() {
  const c = (typeof window !== "undefined" && window.__DINO_SHARE) || {};
  const url = (c.url || "").replace(/\/+$/, "");
  return { url, token: c.token || "", enabled: !!(url && c.token) };
}

/**
 * Upstash REST call. Never throws for network reasons — returns a tagged
 * result so callers can distinguish "offline" from "bad data", which matters
 * because one is retryable and the other is permanent.
 */
async function redis(args, timeoutMs = 4000) {
  const { url, token, enabled } = cfg();
  if (!enabled) return { ok: false, code: "unconfigured" };
  let ctl, t;
  try {
    ctl = new AbortController();
    t = setTimeout(() => ctl.abort(), timeoutMs);
    const res = await fetch(url, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${token}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(args),
      signal: ctl.signal,
    });
    if (res.status === 401 || res.status === 403) return { ok: false, code: "auth" };
    if (res.status === 429) return { ok: false, code: "rate_limited" };
    if (!res.ok) return { ok: false, code: "http_" + res.status };
    let j;
    try { j = await res.json(); }
    catch { return { ok: false, code: "bad_json" }; }
    // Upstash reports command errors in-body with HTTP 200.
    if (j && j.error) return { ok: false, code: "redis_error", detail: String(j.error) };
    return { ok: true, result: j ? j.result : null };
  } catch (e) {
    const aborted = e && (e.name === "AbortError" || /abort/i.test(String(e.message || "")));
    return { ok: false, code: aborted ? "timeout" : "network" };
  } finally {
    if (t) clearTimeout(t);
  }
}

const isObj = (x) => !!x && typeof x === "object" && !Array.isArray(x);
const isNum = (x) => typeof x === "number" && Number.isFinite(x);

/**
 * Structural validation of a shared payload.
 *
 * Returns { valid, reason }. Deliberately strict: a payload that merely
 * *looks* plausible still gets rejected unless the parts we actually load
 * (genome fields, model dimensions) are present and the right type. A silent
 * accept here is indistinguishable from a working sync until the model is
 * loaded, which is far too late.
 */
export function validate(o) {
  if (!isObj(o)) return { valid: false, reason: "not an object" };
  if (o.v !== SCHEMA) return { valid: false, reason: `schema ${o.v} != ${SCHEMA}` };
  if (!isNum(o.score) || o.score <= 0) return { valid: false, reason: "score missing or <= 0" };
  if (!isNum(o.runs) || o.runs < MIN_RUNS) return { valid: false, reason: `runs < ${MIN_RUNS}` };
  if (!isNum(o.at) || o.at <= 0) return { valid: false, reason: "timestamp missing" };

  const ev = o.evolve;
  if (!isObj(ev)) return { valid: false, reason: "evolve missing" };
  const ch = ev.champion;
  if (!isObj(ch)) return { valid: false, reason: "champion missing" };
  if (!isObj(ch.g)) return { valid: false, reason: "genome missing" };

  // The genome fields the policy actually reads. A genome missing these
  // yields NaN windows, which read as "never jump" - indistinguishable from
  // a merely bad strategy, so it must be caught here rather than in play.
  for (const k of ["loA", "loB", "widthA", "widthB", "duck"]) {
    if (!isNum(ch.g[k])) return { valid: false, reason: `genome.${k} not a number` };
  }
  // The score we rank by must agree with the champion it claims to describe.
  const mean = isNum(ch.mean) ? ch.mean : ch.best;
  if (!isNum(mean)) return { valid: false, reason: "champion has no mean/best" };
  if (Math.abs(mean - o.score) > 1) {
    return { valid: false, reason: `score ${o.score} disagrees with champion ${mean}` };
  }
  // A mean can never exceed the best single run. This is what let a
  // fabricated mean:5000 / best:900 payload into the pool, where it blocked
  // every real publish until the key was cleared by hand.
  if (isNum(ch.best) && ch.best > 0 && mean > ch.best + 1) {
    return { valid: false, reason: `mean ${mean} exceeds best ${ch.best}` };
  }
  if (isNum(o.best) && o.best > 0 && o.score > o.best + 1) {
    return { valid: false, reason: `score ${o.score} exceeds best ${o.best}` };
  }

  // Models are optional (a fresh champion may have neither) but must be
  // well-formed and the right width when present.
  if (o.mlp != null) {
    if (!isObj(o.mlp)) return { valid: false, reason: "mlp not an object" };
    if (o.mlp.dim != null && o.mlp.dim !== MLP_DIM) {
      return { valid: false, reason: `mlp dim ${o.mlp.dim} != ${MLP_DIM}` };
    }
  }
  if (o.knn != null && !isObj(o.knn)) return { valid: false, reason: "knn not an object" };

  return { valid: true, reason: "" };
}

/**
 * Quality of a training snapshot, as a single comparable number.
 * Returns null when the snapshot has not been measured enough to rank.
 */
export function quality(evolveState) {
  const ch = evolveState && evolveState.champion;
  if (!isObj(ch)) return null;
  const mean = isNum(ch.mean) ? ch.mean : ch.best || 0;
  if (!(mean > 0)) return null;
  // A mean can never exceed the best single run. Local state that violates
  // this is corrupt - self-heal by discarding it, otherwise an impossible
  // score blocks every legitimate pool entry for the life of the profile.
  if (isNum(ch.best) && ch.best > 0 && mean > ch.best + 1) return null;
  // Direct evidence: this champion has been measured enough on its own.
  if ((ch.runs || 0) >= MIN_RUNS) return mean;
  // Otherwise require corroboration from the run history. A champion is
  // promoted for beating the incumbent's MEAN, so a ledger with at least
  // MIN_RUNS scored entries means the number was earned against measured
  // competition rather than drawn once. Without this, 96% of champions
  // could never publish - measured: 2 of 53 editions reached runs>=3.
  const led = (evolveState.ledger || []).filter(
    (e) => isNum(e && (e.score !== undefined ? e.score : e.best)),
  );
  if (led.length >= MIN_RUNS && (ch.runs || 0) >= 1) return mean;
  return null;
}

/** Everything worth sharing. Logs and analysis stay local: they record THIS
 *  browser's session, not transferable skill. */
export function snapshot() {
  const get = (k) => {
    try { return JSON.parse(localStorage.getItem(k) || "null"); }
    catch { return null; }
  };
  const evolve = get("dino-evolve-v1");
  const q = quality(evolve);
  if (q === null) return null;
  const ch = evolve.champion;
  // Evidence behind the score: the champion's own measurements plus the
  // scored run history that corroborates it. Publishing the bare streak
  // made our own validate() reject the snapshot.
  const ledN = (evolve.ledger || []).filter(
    (e) => isNum(e && (e.score !== undefined ? e.score : e.best)),
  ).length;
  const evidence = Math.max(ch.runs || 0, Math.min(ledN, 999));
  // Publish the CHAMPION ONLY. The local ledger, population, in-progress
  // generation and session counters describe this browser's search history,
  // not transferable skill, and shipping them made the payload 48KB of
  // mostly-irrelevant state. A receiving client starts a fresh search FROM
  // the champion, which is the point of a best-only pool.
  const lean = {
    version: evolve.version,
    champion: {
      g: ch.g,
      hash: ch.hash,
      edition: ch.edition || "",
      best: ch.best || 0,
      runs: ch.runs || 0,
      total: ch.total || 0,
      mean: isNum(ch.mean) ? ch.mean : ch.best || 0,
    },
    challenger: null,
    // The champion's OWN ledger row travels with it. Dropping the whole
    // ledger also dropped this, so the receiving client displayed the
    // adopted champion as a fresh 1-run candidate and rebuilt its stats
    // from zero. One row is not "history" - it is the champion's identity.
    ledger: [{
      hash: ch.hash,
      edition: ch.edition || "",
      best: ch.best || 0,
      runs: ch.runs || 0,
      promoted: true,
      last: Date.now(),
      seq: evolve.episodes || 0,
      fromPool: true,
    }],
    population: [],
    history: [],
    inProgress: null,
    generation: evolve.generation || 0,
    episodes: evolve.episodes || 0,
    // knowledge is per-situation takeoff statistics: genuinely transferable,
    // and what lets an adopting client skip re-learning the same failures.
    knowledge: evolve.knowledge || {},
    bestEver: evolve.bestEver || 0,
  };
  const snap = {
    v: SCHEMA,
    score: q,
    best: ch.best || 0,
    runs: evidence,
    edition: ch.edition || "",
    at: Date.now(),
    evolve: lean,
    mlp: get("dino-mlp-v1"),
    knn: get("dino-knn-v1"),
  };
  // Validate our OWN payload before publishing. Shipping a blob that every
  // reader will reject is worse than not publishing: it looks like success.
  const v = validate(snap);
  if (!v.valid) return null;
  if (JSON.stringify(snap).length > MAX_BYTES) {
    // Drop the KNN (the bulk) rather than fail: the genome and MLP alone are
    // still worth sharing.
    const trimmed = { ...snap, knn: null };
    return JSON.stringify(trimmed).length > MAX_BYTES ? null : trimmed;
  }
  return snap;
}

/** Fetch the pool's best. Never throws; reports why on failure. */
export async function fetchBest() {
  const r = await redis(["GET", KEY]);
  if (!r.ok) return { ok: false, code: r.code, detail: r.detail };
  if (!r.result) return { ok: false, code: "empty" };
  let obj;
  try { obj = typeof r.result === "string" ? JSON.parse(r.result) : r.result; }
  catch { return { ok: false, code: "corrupt_json" }; }
  const v = validate(obj);
  if (!v.valid) return { ok: false, code: "invalid", detail: v.reason };
  return { ok: true, value: obj };
}

/**
 * Install a remote snapshot into localStorage, atomically enough that a
 * failure part-way cannot leave a genome from one user with another's model.
 * Re-validates first: install() is exported and must be safe on its own.
 */
export function install(remote) {
  const v = validate(remote);
  if (!v.valid) return { ok: false, code: "invalid", detail: v.reason };

  // FULL REPLACEMENT. Writing only the genome+models and leaving the rest
  // behind produces a hybrid that belongs to neither user: `dino-analysis-v1`
  // holds per-situation failure counts (ctxAdj) that were measured against
  // the LOCAL genome, and applying them to an imported one corrects the wrong
  // policy. The episode log and the autosave slot are likewise a record of a
  // run history that no longer applies.
  const OWNED = [
    "dino-evolve-v1",
    "dino-mlp-v1",
    "dino-knn-v1",
    "dino-analysis-v1",
    "dino-logs-v1",
    "dino-evolve-slot-v1",
  ];
  const prev = {};
  for (const k of OWNED) prev[k] = localStorage.getItem(k);

  try {
    // Clear everything we own first, so a key absent from the remote cannot
    // survive as a leftover from the previous occupant.
    for (const k of OWNED) localStorage.removeItem(k);
    // Mark the champion as imported and clear any locally-observed stats,
    // so evolve.js re-measures it on THIS machine before trusting its mean
    // as the bar every challenger must clear.
    const adopted = remote.evolve && remote.evolve.champion
      ? {
          ...remote.evolve,
          champion: {
            ...remote.evolve.champion,
            fromPool: true,
            localRuns: 0,
            localTotal: 0,
            localMean: 0,
          },
        }
      : remote.evolve;
    localStorage.setItem("dino-evolve-v1", JSON.stringify(adopted));
    if (remote.mlp) localStorage.setItem("dino-mlp-v1", JSON.stringify(remote.mlp));
    if (remote.knn) localStorage.setItem("dino-knn-v1", JSON.stringify(remote.knn));
    return { ok: true, replaced: OWNED.length };
  } catch (e) {
    // Roll back so a quota failure cannot leave a half-installed mixture -
    // which would be worse than either state on its own.
    try {
      for (const k of OWNED) {
        if (prev[k] == null) localStorage.removeItem(k);
        else localStorage.setItem(k, prev[k]);
      }
    } catch { /* nothing further we can do */ }
    const quota = e && /quota|exceed/i.test(String((e && e.name) || (e && e.message) || ""));
    return { ok: false, code: quota ? "quota" : "write_failed" };
  }
}

/**
 * Pull on load: install the remote training only when it genuinely beats what
 * this browser already has. A remote that merely ties is NOT installed —
 * swapping equal-quality state would discard local run history for nothing.
 */
export async function pullIfBetter() {
  if (!cfg().enabled) return { ok: false, why: "not configured" };
  let localQ = null;
  try {
    localQ = quality(JSON.parse(localStorage.getItem("dino-evolve-v1") || "null"));
  } catch { /* corrupt local state must not block the pull */ }

  const got = await fetchBest();
  if (!got.ok) return { ok: false, why: got.code, detail: got.detail, localQ };

  const remote = got.value;
  if (localQ !== null && remote.score <= localQ) {
    return { ok: false, why: "local is as good or better", localQ, remoteQ: remote.score };
  }
  const ins = install(remote);
  if (!ins.ok) return { ok: false, why: ins.code, detail: ins.detail, localQ, remoteQ: remote.score };
  return { ok: true, why: "installed remote", localQ, remoteQ: remote.score, edition: remote.edition };
}

/**
 * Compare-and-set publish, evaluated SERVER-SIDE.
 *
 * A read-then-write from the client is a race: two tabs closing together both
 * read the old value and the weaker one can land last, overwriting the
 * stronger. The Lua script makes the comparison atomic inside Redis.
 *
 * The script also refuses payloads whose stored form is unparseable, so a
 * corrupt existing value cannot permanently block publishing.
 */
const CAS_LUA =
  "local cur = redis.call('GET', KEYS[1]) " +
  "local corrupt = 0 " +
  "if cur then " +
  // Decode failures and shape violations are BOTH corruption. Treating them
  // as 'absent' is what makes the key self-healing: without this a single
  // bad write (truncated value, older incompatible schema, manual edit)
  // would block every future publish forever, because the comparison could
  // never succeed and no client could ever replace it.
  "  local ok, o = pcall(cjson.decode, cur) " +
  "  if not ok or type(o) ~= 'table' then corrupt = 1 " +
  "  elseif tonumber(o.v) ~= tonumber(ARGV[3]) then corrupt = 1 " +
  "  elseif not tonumber(o.score) or tonumber(o.score) <= 0 then corrupt = 1 " +
  "  elseif not tonumber(o.runs) or tonumber(o.runs) < tonumber(ARGV[4]) then corrupt = 1 " +
  "  elseif type(o.evolve) ~= 'table' or type(o.evolve.champion) ~= 'table' " +
  "         or type(o.evolve.champion.g) ~= 'table' then corrupt = 1 " +
  "  end " +
  // Only a VALID incumbent may block the write.
  "  if corrupt == 0 and tonumber(o.score) >= tonumber(ARGV[2]) then return 0 end " +
  "end " +
  "redis.call('SET', KEYS[1], ARGV[1]) " +
  // 2 distinguishes 'healed a corrupt value' from 2 plain publish, so the
  // client can report it instead of silently masking data loss.
  "if corrupt == 1 then return 2 end " +
  "return 1";

export async function pushIfBetter() {
  if (!cfg().enabled) return { ok: false, why: "not configured" };
  const local = snapshot();
  if (!local) return { ok: false, why: `no publishable snapshot (needs ${MIN_RUNS}+ runs)` };
  const r = await redis(["EVAL", CAS_LUA, "1", KEY, JSON.stringify(local), String(local.score), String(SCHEMA), String(MIN_RUNS)]);
  if (!r.ok) return { ok: false, why: r.code, detail: r.detail, localQ: local.score };
  if (r.result === 2) {
    return { ok: true, why: "replaced corrupt remote", healed: true, localQ: local.score };
  }
  return {
    ok: r.result === 1,
    why: r.result === 1 ? "published" : "remote is as good or better",
    localQ: local.score,
  };
}

/**
 * Sync at the end of a run: publish if we improved, adopt if someone else
 * overtook us. Called after every episode.
 *
 * Cheap by construction. Episodes end 16-20 times a minute at max rate, so
 * this must not be a network call per run. It only acts when the publishable
 * quality has actually CHANGED since the last sync - most runs do not move
 * the champion, and those cost nothing but a localStorage read.
 *
 * A floor between network calls still applies, because a rapidly improving
 * champion could otherwise publish on many consecutive episodes.
 */
/** Set by the host so an adopted snapshot is loaded into the live engines
 *  rather than sitting in localStorage unread. Declared before syncNow uses
 *  it so correctness does not depend on call ordering. */
let reloadHook = () => {};
export function onAdopt(fn) { if (typeof fn === "function") reloadHook = fn; }

/** Host-supplied observer for UI. Called for EVERY sync outcome, from any
 *  caller, so the panel cannot go stale when a different path publishes. */
let syncObserver = () => {};
export function onSync(fn) { if (typeof fn === "function") syncObserver = fn; }
function report(kind, r) {
  try { syncObserver(kind, r); } catch { /* UI must never break a sync */ }
  return r;
}

let lastSyncedQuality = null;
/** Episodes since we last asked the pool whether it overtook us. Checking
 *  every episode would be a network call per run for no reason; never
 *  checking means a better pooled model is invisible for a whole session. */
let sincePoolCheck = 0;
const POOL_CHECK_EVERY = 5;

export async function syncNow(log) {
  if (!cfg().enabled) return { ok: false, why: "not configured" };
  const local = snapshot();
  const q = local ? local.score : null;

  // 🔴 A client with nothing to publish must STILL pull. Returning early
  // here meant a fresh visitor - exactly the one with most to gain - never
  // saw the pool after the startup pull, and a pool that filled later was
  // never picked up at all. Measured: the page logged "pool is empty" while
  // diagnose() could read a 1900pt champion from the same endpoint.
  if (q === null) {
    const pulled = await pullIfBetter();
    if (pulled.ok) {
      reloadHook();
      if (log) log(`shared best adopted · ${pulled.remoteQ}pts avg${pulled.edition ? " · " + pulled.edition : ""}`);
    }
    return pulled;
  }

  // Quality unchanged: still worth asking whether someone else overtook us,
  // but only occasionally - this runs at the end of every episode.
  if (q === lastSyncedQuality) {
    sincePoolCheck++;
    if (sincePoolCheck < POOL_CHECK_EVERY) return { ok: false, why: "unchanged" };
    sincePoolCheck = 0;
    const pulled = await pullIfBetter();
    if (pulled.ok) {
      reloadHook();
      lastSyncedQuality = null;   // our state changed; re-evaluate next run
      if (log) log(`shared best adopted · ${pulled.remoteQ}pts avg${pulled.edition ? " · " + pulled.edition : ""}`);
      return pulled;
    }
    return { ok: false, why: "unchanged" };
  }

  try {
    const res = await pushIfBetter();
    if (res.ok) {
      lastSyncedQuality = q;
      report("published", res);
      if (log) {
        log(res.healed
          ? `shared best published · ${q}pts avg (replaced corrupt entry)`
          : `shared best published · ${q}pts avg`);
      }
      return res;
    }
    // Refused because the pool is better: adopt it, so a run that ends
    // behind the pool immediately benefits instead of waiting for a reload.
    if (res.why === "remote is as good or better") {
      lastSyncedQuality = q;
      const pulled = await pullIfBetter();
      if (pulled.ok) {
        reloadHook();
        report("adopted", pulled);
        if (log) log(`shared best adopted · ${pulled.remoteQ}pts avg${pulled.edition ? " · " + pulled.edition : ""}`);
        return pulled;
      }
      report("ahead", res);
      return res;
    }
    return res;
  } catch {
    // Never let a sync failure interrupt the run.
    return { ok: false, why: "error" };
  }
}

/**
 * Push on the way out. `visibilitychange -> hidden` is the only event that
 * reliably fires on mobile; beforeunload/unload do not fire when a tab is
 * swiped away or the OS reclaims the page. pagehide covers bfcache.
 *
 * keepalive lets the request outlive the page, which a normal fetch cannot.
 */
export function installAutoPush(log) {
  if (!cfg().enabled) return false;
  let warnedUnpublishable = false;
  const fire = () => {
    const local = snapshot();
    if (!local) {
      // Nothing to publish - but still ASK the pool, because a tab being
      // hidden is one of only two moments we sync at all.
      pullIfBetter().then((r) => {
        if (r && r.ok) {
          reloadHook();
          if (log) log(`shared best adopted · ${r.remoteQ}pts avg${r.edition ? " · " + r.edition : ""}`);
        }
      }).catch(() => { /* exit-path failure must be silent */ });
      if (log && !warnedUnpublishable) {
        warnedUnpublishable = true;
        let runs = 0;
        try { runs = (JSON.parse(localStorage.getItem("dino-evolve-v1") || "{}").champion || {}).runs || 0; }
        catch { /* ignore */ }
        log(`shared best: not publishing yet · needs ${MIN_RUNS} measured runs (have ${runs})`);
      }
      return;
    }
    const { url, token } = cfg();
    try {
      fetch(url, {
        method: "POST",
        headers: { Authorization: `Bearer ${token}`, "Content-Type": "application/json" },
        body: JSON.stringify(["EVAL", CAS_LUA, "1", KEY, JSON.stringify(local), String(local.score), String(SCHEMA), String(MIN_RUNS)]),
        keepalive: true,
      })
        .then((r) => (r.ok ? r.json() : null))
        .then((j) => {
          if (!j) return;
          if (j.result === 1 || j.result === 2) {
            report("published", { why: "published", localQ: local.score });
            if (log) log(`shared best published · ${local.score}pts avg${j.result === 2 ? " (replaced corrupt entry)" : ""}`);
          } else {
            // Refused: the pool is ahead, so adopt it rather than leave
            // this browser behind until the next reload.
            pullIfBetter().then((p) => {
              if (p && p.ok) {
                reloadHook();
                if (log) log(`shared best adopted · ${p.remoteQ}pts avg${p.edition ? " · " + p.edition : ""}`);
              }
            }).catch(() => {});
          }
        })
        .catch(() => { /* exit-path failure must be silent */ });
    } catch { /* ignore */ }
  };
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "hidden") fire();
  });
  window.addEventListener("pagehide", fire);
  return true;
}

// Exposed for diagnosis from DevTools on the deployed page: a user who sees
// no network activity can run `await __dinoShare.diagnose()` and get the
// actual reason rather than silence.
if (typeof window !== "undefined") {
  window.__dinoShare = {
    cfg,
    quality,
    snapshot,
    fetchBest,
    pullIfBetter,
    pushIfBetter,
    syncNow,
    async diagnose() {
      const c = cfg();
      const out = { configured: c.enabled, url: c.url, tokenLen: c.token.length };
      if (!c.enabled) { out.verdict = "no url/token injected into the page"; return out; }
      let local = null;
      try { local = JSON.parse(localStorage.getItem("dino-evolve-v1") || "null"); } catch {}
      out.localRuns = (local && local.champion && local.champion.runs) || 0;
      out.localQuality = quality(local);
      out.publishable = snapshot() !== null;
      if (!out.publishable) out.whyNotPublishable = `needs ${MIN_RUNS}+ measured runs (have ${out.localRuns})`;
      const t0 = Date.now();
      const got = await fetchBest();
      out.remote = got.ok ? { score: got.value.score, runs: got.value.runs, edition: got.value.edition } : null;
      out.remoteError = got.ok ? null : (got.code + (got.detail ? ": " + got.detail : ""));
      out.roundTripMs = Date.now() - t0;
      out.verdict = got.ok || got.code === "empty"
        ? "network OK"
        : "network/ACL problem: " + out.remoteError;
      return out;
    },
  };
}

export const _internals = { KEY, SCHEMA, MIN_RUNS, MAX_BYTES, MLP_DIM, cfg, CAS_LUA };
