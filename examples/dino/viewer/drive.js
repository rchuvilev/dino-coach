/**
 * Unbounded evolving run on the REAL dino game.
 *
 * Loop: evaluate every candidate in the generation over EPISODES_PER
 * episodes, rank by live mean, keep elites, breed, repeat — forever, until
 * stopped. Population persists to localStorage, so a reload continues.
 */
import { TakeoffKNN, featurize as knnFeaturize } from "./knn.js";
import * as share from "./share.js";
import { featurize } from "./features.js";
import { JumpModel } from "./mlp.js";
import { LogStore } from "./logstore.js";
import * as Analysis from "./analysis.js";
import {
  closeGeneration,
  currentState,
  ELITE,
  EPISODES_PER,
  key,
  label,
  recordEpisode,
  resetAll,
  bandOf,
  BANDS,
  exportState,
  EPISODES_PER as EP_PER,
  episodesFor,
  championInfo,
  currentHash,
  ledgerRows,
  changeHash,
  genomeHash,
  generationComplete,
  judgeRun,
  nextCandidate,
  POP as POP_SIZE,
  importState,
  loadSlot,
  saveSlot,
  seedPopulation,
  slotInfo,
  windowAt,
  reloadFromStorage,
} from "./evolve.js";

const CLOCK = window.__CLOCK;
/**
 * Bucket a decision into a SITUATION class. Coarse on purpose: fine buckets
 * split the data until every cell has one sample and the counts mean nothing.
 * Three axes measured to matter: obstacle width, speed band, and whether a
 * second obstacle is close behind.
 */
/** Add the derived time-to-collision, measured 2.8x more predictive than gap. */
/** Map a captured takeoff situation onto the model's feature input. */
function mlpFeatFrom(d) {
  return {
    gap: d.takeoffGap,
    speed: d.speed,
    width: d.targetWidth,
    next: d.nextDelta,
    high: !!d.high,
    wide: !!d.wide,
    arcVel: d.arcVel ?? 10,
  };
}

function withTtc(d) {
  const ttc = d.speed > 0 && Math.abs(d.takeoffGap) < 500 ? d.takeoffGap / d.speed : null;
  return { ...d, ttc };
}

function ctxKey(d) {
  const w = d.wide ? "wide" : "narrow";
  const sp = d.speed < 8 ? "slow" : d.speed < 10.5 ? "mid" : "fast";
  const n = d.next !== null && d.next < 90 ? "tight" : "open";
  return `${w}/${sp}/${n}`;
}

/**
 * The FULL situation at takeoff, as the user specified: speed, every obstacle
 * currently on screen, and the jump's own properties. ctxKey() buckets this
 * for counting; the raw record is what per-situation tuning reads, because a
 * bucket cannot tell you the takeoff was 8px late.
 */
function captureSituation(s, action, g) {
  const r = R();
  const tx = r && r.tRex ? r.tRex.xPos + 44 : 0;
  const onScreen = [];
  for (const o of (r && r.horizon && r.horizon.obstacles) || []) {
    onScreen.push({
      d: Math.round(o.xPos - tx),
      w: o.width || 0,
      high: o.yPos < 75,
      type: (o.typeConfig && o.typeConfig.type) || "",
    });
  }
  onScreen.sort((a, b) => a.d - b.d);
  return {
    // world
    speed: Math.round(s.speed * 100) / 100,
    obstacles: onScreen.slice(0, 3),
    // the jump itself
    takeoffGap: Math.round(s.gap),
    targetWidth: s.width,
    wide: !!s.wide,
    nextDelta: s.gap2 < 99999 ? Math.round(s.gap2 - s.gap) : null,
    arc: action,
    arcVel: action === "jumpLow" ? lastArc.low : action === "jumpHigh" ? lastArc.high : null,
    // where the window sat when we chose - lets tuning say "8px too early"
    windowLo: g.__wLo,
    windowHi: g.__wHi,
    frame: 0,
  };
}

const $ = (id) => document.getElementById(id);
const R = () => (window.Runner && window.Runner.instance_ ? window.Runner.instance_ : null);

let running = false;
let loop = null;
let rafId = null;
let painter = null;
let pop = [];
let idx = 0;
let epInCand = 0;
let epStart = 0;
let sawCrash = false;
/**
 * Persistent structured log. Replaces a 5-line textContent buffer that
 * vanished on reload and was invisible to Export - which made it impossible
 * to see afterwards WHAT the model weights had been corrected on.
 */
let logs = new LogStore();

function log(msg, kind = "system", data) {
  logs.add(kind, msg, data);
  const el = $("log");
  if (el) el.textContent = logs.render(6);
  saveLogs();
}

function read() {
  const r = R();
  if (!r || !r.tRex) return null;
  const tx = r.tRex.xPos + 44;
  let gap = 99999;
  let high = false;
  let width = 0;
  // gap2: the obstacle AFTER the nearest one. Required by fastdrop, which
  // must not abort the jump that is currently clearing an obstacle.
  let gap2 = 99999;
  const ahead = [];
  for (const o of (r.horizon && r.horizon.obstacles) || []) {
    const d = o.xPos - tx;
    if (d > -30) ahead.push({ d, high: o.yPos < 75, w: o.width || 0 });
  }
  ahead.sort((a, b) => a.d - b.d);
  if (ahead[0]) {
    gap = ahead[0].d;
    high = ahead[0].high;
    width = ahead[0].w;
  }
  if (ahead[1]) gap2 = ahead[1].d;
  return {
    crashed: !!r.crashed,
    started: !!r.started,
    distance: Math.round(r.distanceRan || 0),
    speed: r.currentSpeed || 0,
    airborne: r.tRex.yPos < (r.tRex.groundYPos || 93) - 4,
    gap,
    gap2,
    high,
    width,
    wide: width >= 50,
  };
}

/** Act through the game's own key handlers, so input follows its real path. */
/** Arc velocities for the current genome, set by decide(). */
let lastArc = { low: 8, high: 12 };

// ---- keep-awake ---------------------------------------------------------
let wakeLock = null;
let keepAliveVideo = null;

/** Hold the screen on for the duration of a run. */
async function acquireWakeLock() {
  // preferred: the real API, Chrome on a secure origin
  try {
    if (navigator.wakeLock && !wakeLock) {
      wakeLock = await navigator.wakeLock.request("screen");
      // the OS can revoke it (tab hidden, battery saver) - re-acquire when
      // we become visible again rather than silently losing it
      wakeLock.addEventListener("release", () => {
        wakeLock = null;
      });
      return "api";
    }
  } catch {
    wakeLock = null;
  }
  // fallback: a muted looping video counts as playback and holds the screen
  try {
    if (!keepAliveVideo) {
      const v = document.createElement("video");
      v.setAttribute("playsinline", "");
      v.muted = true;
      v.loop = true;
      v.style.cssText = "position:fixed;width:1px;height:1px;opacity:0.01;pointer-events:none;left:0;bottom:0";
      // Generate the stream instead of embedding a blob: a hand-written
      // base64 webm failed with NotSupportedError, and captureStream() can
      // never have an invalid payload.
      const c = document.createElement("canvas");
      c.width = 2;
      c.height = 2;
      const cx = c.getContext("2d");
      cx.fillStyle = "#000";
      cx.fillRect(0, 0, 2, 2);
      if (c.captureStream) v.srcObject = c.captureStream(1);
      document.body.appendChild(v);
      keepAliveVideo = v;
    }
    await keepAliveVideo.play();
    return "video";
  } catch {
    return "none";
  }
}

function releaseWakeLock() {
  try {
    if (wakeLock) {
      wakeLock.release();
      wakeLock = null;
    }
  } catch { /* already gone */ }
  try {
    if (keepAliveVideo) keepAliveVideo.pause();
  } catch { /* ignore */ }
}

// Re-acquire after the OS revokes it (tab hidden, then visible again).
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "visible" && running) acquireWakeLock();
});

/** True while a jump we issued is still in the air. */
/** The model this run is testing: champion itself, or champion + 1 change. */
let current = null;

let jumpLatched = false;

/** x position where the current takeoff began, for the overlay's jump-start
 *  line and arc. Null while grounded. */
let jumpStartX = null;
/** dino y at takeoff, so the arc can be drawn from its true origin. */
let jumpStartY = null;

function act(a) {
  const r = R();
  if (!r) return;
  // release the latch once we are grounded again
  if (jumpLatched && r.tRex && !r.tRex.jumping) jumpLatched = false;
  const ev = (kc, t) => ({ keyCode: kc, type: t, preventDefault() {}, target: {} });
  if (a === "jump" || a === "jumpLow" || a === "jumpHigh") {
    // EDGE TRIGGER: one keydown per takeoff. Level-triggering re-launched
    // the dino every frame the window stayed open.
    if (jumpLatched || (r.tRex && r.tRex.jumping)) return;
    jumpLatched = true;
    const wasJumping = false;
    // Remember WHERE this takeoff happened so the overlay can draw the
    // jump-start line. Captured at the takeoff frame because tRex.xPos is
    // constant during play - the world moves, not the dino - so it cannot
    // be recovered afterwards.
    jumpStartX = r.tRex.xPos + ((r.tRex.config && r.tRex.config.WIDTH) || 44);
    jumpStartY = r.tRex.yPos;
    r.onKeyDown(ev(38, "keydown"));
    r.onKeyUp(ev(38, "keyup"));
    // Shape the arc AFTER startJump - setting jumpVelocity before it is
    // overwritten by the game's own initialisation.
    // ONLY on the frame the jump begins. Re-applying the velocity while
    // already airborne re-boosts the dino every frame and it never lands.
    if (r.tRex && r.tRex.jumping && !wasJumping) {
      if (a === "jumpLow") r.tRex.jumpVelocity = -(lastArc.low);
      else if (a === "jumpHigh") r.tRex.jumpVelocity = -(lastArc.high);
    }
  } else if (a === "fastdrop") {
    // Abort the current jump via the game's own speed-drop (measured: cuts a
    // 30-frame arc to 13). setSpeedDrop leaves the dino DUCKING once it
    // lands, and a ducking dino cannot jump - that turned the first fastdrop
    // into the last jump of the episode and collapsed scores 1263 -> 34.
    // Releasing ArrowDown clears it through the game's own handler.
    if (r.tRex && r.tRex.jumping) {
      r.tRex.setSpeedDrop();
    } else {
      r.onKeyUp(ev(40, "keyup"));
    }
  } else if (a === "duck") {
    r.onKeyDown(ev(40, "keydown"));
  } else {
    r.onKeyUp(ev(40, "keyup"));
  }
}

/** Decide from the genome. Returns {action, rule}. */
function decide(s, g) {
  const has = s.gap < 99999;
  const canDuck = g.duck > 0 && has && s.high && s.gap <= g.duck;
  // controls keep their fixed behaviour
  if (g.never) return { action: "run", rule: 2 };
  if (g.always) return { action: "jump", rule: 0 };
  // OPTION A: the window is resolved from the CURRENT speed, so one genome
  // expresses a different takeoff distance early and late in the run.
  let w = windowAt(g, s.speed);
  // stash for captureSituation: lets a failure be described as "took off 8px
  // outside the window" rather than just "failed"
  // apply what previous failures taught about THIS situation
  if (g.ctxAdj) {
    const probe = {
      wide: !!s.wide,
      speed: s.speed,
      next: s.gap2 < 99999 ? s.gap2 - s.gap : null,
    };
    const adj = g.ctxAdj[ctxKey(probe)];
    if (adj) w = { lo: Math.max(5, w.lo + adj), hi: Math.max(15, w.hi + adj) };
  }
  // KNN correction from the CONTINUOUS neighbourhood, layered on the
  // bucketed ctxAdj. It only speaks when the local neighbourhood holds both
  // outcomes and the failure rate is material, so it stays silent where
  // there is nothing to fix.
  const sug = knn.suggest(
    knnFeaturize({ gap: s.gap, speed: s.speed, width: s.width,
                   next: s.gap2 < 99999 ? s.gap2 - s.gap : null }),
  );
  if (sug) {
    const shift = Math.round(sug.delta * 0.5);
    w = { lo: Math.max(5, w.lo + shift), hi: Math.max(15, w.hi + shift) };
  }
  g.__wLo = Math.round(w.lo);
  g.__wHi = Math.round(w.hi);
  // WIDTH-AWARE takeoff. Measured taxonomy over 20 deaths: 16 were "jumped
  // early, descended onto the obstacle", dominated by w>=70 cacti. The dino
  // clears 17 frames * speed px while high enough; a w75 needs 119px
  // including its own body, so at speed 6 it covers only 102 and CANNOT
  // clear - the response must differ by width, not just by speed.
  if (s.wide) {
    const adj = g.wideAdj || 0;
    w = { lo: Math.max(5, w.lo + adj), hi: Math.max(15, w.hi + adj) };
  }
  // Latest-safe takeoff: fire in the LOWER part of the window, not on entry.
  // Firing on entry kept the dino airborne 56% of all frames and wasted 22
  // of 35 jumps, which is what caused missed windows for the next obstacle.
  const lateFrac = g.late === undefined ? 1 : g.late;
  const effHi = w.lo + (w.hi - w.lo) * Math.max(0.15, Math.min(1, lateFrac));
  // TTC window when the genome has one, else the legacy pixel window. Both
  // are evolvable; zero ttcLo disables it so the GA can fall back.
  // TIME-TO-COLLISION, computed HERE because this is where it is used.
  // Previously declared inside withTtc() and referenced from this scope,
  // which threw "ttc is not defined" on the first decision.
  const ttc = s.gap < 9000 && s.speed > 0 ? s.gap / s.speed : 9999;
  // MODEL IN THE LOOP: P(this jump clears), from the trained weights.
  // Null until the model has seen enough to be worth consulting.
  // Includes the DECISION (which arc) and the obstacle TYPE, not just the
  // world state - without them the same situation jumped two different ways
  // produced identical features with opposite labels.
  const mlpFeat = {
    gap: s.gap, speed: s.speed, width: s.width,
    next: s.gap2 < 9000 ? s.gap2 - s.gap : null,
    high: s.high, wide: s.wide,
    arcVel: s.wide ? (g.arcHigh ?? 12) : (g.arcLow ?? 8),
  };
  const pClear = mlp.predictSync(featurize(mlpFeat));

  const useTtc = (g.ttcLo || 0) > 0;
  const canJump = useTtc
    ? ttc < 9999 && !s.high && ttc >= g.ttcLo && ttc <= g.ttcHi && !s.airborne
    : has && !s.high && s.gap >= w.lo && s.gap <= effHi && !s.airborne;
  // PANIC: the window was missed (usually because the dino was airborne
  // through it) and the obstacle is now close. Measured: 12 of 14 deaths
  // were "grounded at impact, never jumped" for exactly this reason.
  const panicGap = g.panic || 0;
  const canPanic = useTtc
    ? (g.ttcPanic || 0) > 0 && ttc < 9999 && !s.high && !s.airborne && ttc > 0 && ttc <= g.ttcPanic
    : panicGap > 0 && has && !s.high && !s.airborne && s.gap > 0 && s.gap <= panicGap;
  // VETO: the rules want to jump but the model says this one fails.
  // Threshold is deliberately strict - only override on real confidence.
  const vetoed =
    pClear !== null && canJump && pClear < (g.mlpVeto === undefined ? 0.25 : g.mlpVeto);
  // RESCUE: the rules are NOT jumping, but the model says a jump would
  // clear and the obstacle is close enough that doing nothing kills us.
  // A high obstacle is never a jump target: jumping INTO a pterodactyl is
  // the collision, not the escape. Ducking is the correct response, so the
  // rescue path must stay out of the way for high obstacles.
  const rescue =
    pClear !== null && !canJump && !canDuck && !s.airborne && !s.high &&
    s.gap > 0 && s.gap < 60 &&
    pClear > (g.mlpRescue === undefined ? 0.8 : g.mlpRescue);

  if (rescue) return { action: "jump", rule: 0, byModel: true };

  if (g.duckFirst) {
    if (canDuck) return { action: "duck", rule: 1 };
    if (canJump && !vetoed) return { action: "jump", rule: 0 };
  } else {
    if (canJump && !vetoed) {
      // wide obstacle -> high arc to cover ground; narrow with another
      // obstacle close behind -> low arc to land sooner and stay ready
      const needHigh = s.wide;
      const nextClose = s.gap2 < 99999 && s.gap2 - s.gap < (g.arcSwitch || 90);
      const action = needHigh ? "jumpHigh" : nextClose ? "jumpLow" : "jump";
      return { action, rule: 0 };
    }
    if (canDuck) return { action: "duck", rule: 1 };
  }
  if (canPanic) return { action: "jump", rule: 0 };
  return { action: "run", rule: 2 };
}

/** Pick the model for the next run: champion, or champion + one change. */
function pickCandidate() {
  current = nextCandidate();
  return current;
}

function restartEpisode() {
  const r = R();
  if (!r) return;
  // every run targets the champion, or the champion plus one change
  pickCandidate();
  r.restart();
  for (let i = 0; i < 8; i++) CLOCK.step();
  // A jump starts the game, but the dino must be back on the ground before
  // the episode begins - otherwise the edge-trigger latch blocks the first
  // real decision and it never recovers.
  act("jump");
  jumpLatched = false;
  for (let i = 0; i < 40 && r.tRex && r.tRex.jumping; i++) CLOCK.step();
  for (let i = 0; i < 2; i++) CLOCK.step();
  jumpLatched = false;
  const s = read();
  // DELTA baseline: restart() does not zero distanceRan in this build
  epStart = s ? s.distance : 0;
  tel = newTel();
  lastDist = 0;
  bandDist = BANDS.map(() => 0);
  sawCrash = false;
}

function candName(c) {
  return c.ctrl ? `control: ${c.ctrl}` : label(c.g);
}

function renderRules(fired, s, g) {
  const has = s && s.gap < 99999;
  const w = g.never || g.always ? { lo: -1, hi: -1 } : windowAt(g, s ? s.speed : 6);
  const rows = [
    { when: `gap ${Math.round(w.lo)}..${Math.round(w.hi)} @spd${s ? s.speed.toFixed(1) : "?"}`, then: "jump" },
    { when: `high & gap<=${g.duck}`, then: "duck" },
    { when: `airborne & passed & next<=${g.dropAt || 0}`, then: "fastdrop" },
    { when: "always", then: "run" },
  ];
  const matched = [
    has && !s.high && s.gap >= w.lo && s.gap <= w.hi && !s.airborne,
    has && s.high && g.duck > 0 && s.gap <= g.duck,
    (g.dropAt || 0) > 0 && s.airborne && s.gap <= 0 && s.gap2 < 99999 && s.gap2 <= g.dropAt,
    true,
  ];
  const ul = $("rules");
  ul.innerHTML = "";
  rows.forEach((r, i) => {
    const li = document.createElement("li");
    li.className = i === fired ? "fire" : matched[i] ? "match" : "";
    li.innerHTML =
      `<span>${r.when}</span><span>=&gt;</span><span class="then">${r.then}</span>` +
      `<span class="why">${i === fired ? "FIRING" : matched[i] ? "suppressed" : ""}</span>`;
    ul.appendChild(li);
  });
}

function renderTable() {
  const S = currentState();
  const t = $("tbl");
  // The population table listed candidates that never played once the
  // champion/challenger loop replaced population breeding. This lists every
  // model that has ACTUALLY RUN, newest activity first, with the one
  // currently on the track highlighted.
  t.innerHTML =
    "<tr><th>edition</th><th class='n'>runs</th><th class='n'>best pts</th></tr>";
  const rows = ledgerRows();
  const cur = currentHash();
  // The model on the track right now may not be in the ledger yet - it is
  // recorded only when its run finishes. Show it as a provisional row so
  // "where are the candidates now" has an answer mid-run.
  if (cur && !rows.some((r) => r.hash === cur)) {
    const ci = championInfo();
    rows.unshift({ hash: cur, runs: 0, best: 0,
      edition: ci && ci.hash === cur ? (ci.edition || "initial") : "testing",
      pending: true });
  }
  if (!rows.length) {
    t.innerHTML += "<tr><td colspan='3'>no runs yet</td></tr>";
  }
  for (const r of rows.slice(0, 12)) {
    const tr = document.createElement("tr");
    if (r.hash === cur) tr.className = "best";      // currently running
    else if (r.promoted) tr.className = "ctrl";     // was champion at some point
    // best is stored in POINTS; dividing again produced 0-7 instead of the
    // real hundreds.
    tr.innerHTML =
      `<td>${r.hash === cur ? "▶ " : "&nbsp;&nbsp;"}${r.edition || "initial"}${r.promoted ? " ★" : ""}</td>` +
      `<td class="n">${r.runs}</td>` +
      `<td class="n">${r.pending ? "…" : (r.best || 0)}</td>`;
    t.appendChild(tr);
  }
  const St = currentState();
  const cz = St.causes || {};
  const total = Object.values(cz).reduce((a, b) => a + b, 0) || 1;
  const topCauses = Object.entries(cz).sort((a, b) => b[1] - a[1]).slice(0, 4)
    .map(([k, v]) => `${k} ${Math.round(100 * v / total)}%`).join(" · ");
  // per-situation knowledge: BOTH the winning and the losing timing
  const kEl = $("knowledge");
  if (kEl) {
    const K = St.knowledge || {};
    const med = (a) => (a && a.length ? [...a].sort((x, y) => x - y)[Math.floor(a.length / 2)] : null);
    const rows = Object.entries(K)
      .sort((a, b) => (b[1].ok + b[1].fail) - (a[1].ok + a[1].fail))
      .slice(0, 5)
      .map(([k, e]) => {
        const mo = med(e.okGap), mf = med(e.failGap);
        const rate = e.ok + e.fail ? Math.round((100 * e.ok) / (e.ok + e.fail)) : 0;
        const fix = mo !== null && mf !== null ? ` fix ${mo - mf > 0 ? "+" : ""}${mo - mf}` : "";
        return `${k} ${rate}% ok@${mo ?? "-"} fail@${mf ?? "-"}${fix}`;
      });
    kEl.textContent = rows.length ? rows.join("  ·  ") : "learning...";
  }
  // THREE-WAY DIAGNOSIS: state x reason x prop, the actionable form
  const dEl = $("diagnosis");
  if (dEl) {
    const lines = [];
    for (const st of Analysis.states(analysis)) {
      for (const d of Analysis.diagnose(analysis, st).slice(0, 2)) {
        lines.push(
          `${d.state} · ${d.reason}: ${d.prop} ok@${d.okMedian} vs fail@${d.failMedian} ` +
            `(${d.delta > 0 ? "+" : ""}${d.delta}, n=${d.nFail}/${d.nOk})`,
        );
      }
    }
    dEl.textContent = lines.length ? lines.slice(0, 6).join("\n") : "gathering...";
  }
  const causesEl = $("causes");
  if (causesEl) causesEl.textContent = topCauses || "no deaths recorded yet";
  // POINTS, matching the game's own readout (distance * 0.025). Reporting
  // distance here while the canvas showed points made a 361-point confirmed
  // median look like a ceiling against a 1191-point game record.
  const pts = (d) => Math.round((d || 0) * 0.025);
  // surface the escape state: a flat curve is only worrying if the search
  // is NOT already widening in response to it
  const stale = (S.generation || 0) - (S.lastImproveGen || 0);
  const boost = S.mutBoost && S.mutBoost > 1 ? ` · explore x${S.mutBoost.toFixed(1)}` : "";
  $("evoinfo").textContent =
    `s${S.sessions} · gen ${S.generation} · ep ${S.episodes} · ` +
    `typical ${pts(S.bestMedian)}pts · best ${pts(S.bestEver)}pts` +
    ` · model ${mlp.samplesTrained}` +
    (stale > 0 ? ` · stale ${stale}` : "") + boost;
}

function drawChart() {
  const S = currentState();
  const c = $("chart");
  if (!c) return;
  const ctx = c.getContext("2d");
  ctx.clearRect(0, 0, c.width, c.height);
  const h = S.history;
  if (!h.length) {
    const S2 = currentState();
    const done = S2.episodes || 0;
    const need = POP_SIZE * episodesFor(S2.generation);
    const frac = Math.min(1, done / need);
    ctx.fillStyle = "#7c8a9a";
    ctx.font = "10px monospace";
    ctx.fillText(`generation 1: ${done}/${need} episodes`, 8, 14);
    // progress bar, so "nothing yet" is visibly DIFFERENT from "stuck"
    const w = c.width - 16;
    ctx.strokeStyle = "#232b35";
    ctx.strokeRect(8.5, 24.5, w, 8);
    ctx.fillStyle = "#3ddc84";
    ctx.fillRect(9, 25, Math.max(0, w * frac - 1), 7);
    ctx.fillStyle = "#7c8a9a";
    ctx.fillText(
      frac < 1 ? "use rate 'max (evolve)' to get here in seconds" : "closing...",
      8,
      48,
    );
    return;
  }
  const max = Math.max(...h.map((x) => Math.max(x.best, x.control)), 1);
  const pad = 4;
  const step = h.length > 1 ? (c.width - pad * 2) / (h.length - 1) : 0;
  const y = (v) => c.height - pad - (v / max) * (c.height - pad * 2);
  // control line
  ctx.strokeStyle = "#ffb020";
  ctx.lineWidth = 1;
  ctx.beginPath();
  h.forEach((p, i) => (i ? ctx.lineTo(pad + i * step, y(p.control)) : ctx.moveTo(pad, y(p.control))));
  ctx.stroke();
  // evolved best
  ctx.strokeStyle = "#3ddc84";
  ctx.lineWidth = 2;
  ctx.beginPath();
  h.forEach((p, i) => (i ? ctx.lineTo(pad + i * step, y(p.best)) : ctx.moveTo(pad, y(p.best))));
  ctx.stroke();
  ctx.fillStyle = "#7c8a9a";
  ctx.font = "9px monospace";
  ctx.fillText(`max ${max}`, pad + 2, 10);
  ctx.fillStyle = "#3ddc84";
  ctx.fillText("evolved", pad + 2, c.height - 4);
  ctx.fillStyle = "#ffb020";
  ctx.fillText("control", pad + 54, c.height - 4);
}

/**
 * Ballistic landing prediction for the jump currently in flight.
 *
 * Uses the game's own physics constants rather than a fitted curve: yPos
 * rises by jumpVelocity each frame and jumpVelocity increases by GRAVITY,
 * so time-to-ground solves from the standard quadratic. Horizontal travel
 * is that time times the world speed, because the dino's x is fixed and the
 * WORLD moves - the landing point is where the ground under it will be.
 */
function predictLanding(r) {
  const t = r && r.tRex;
  if (!t || !t.jumping) return null;
  const g = (t.config && t.config.GRAVITY) || 0.6;
  const ground = t.groundYPos || 93;
  const y0 = ground - t.yPos;          // height above ground, positive up
  const v0 = -(t.jumpVelocity || 0);   // upward positive
  const disc = v0 * v0 + 2 * g * y0;
  if (disc < 0) return null;
  const frames = (v0 + Math.sqrt(disc)) / g;
  if (!Number.isFinite(frames) || frames < 0) return null;
  return { frames, dx: frames * (r.currentSpeed || 6) };
}

function overlay(s, g) {
  const r = R();
  const c = $("ov");
  const wrap = $("wrap");
  if (!c || !wrap) return;
  if (c.width !== wrap.clientWidth || c.height !== wrap.clientHeight) {
    c.width = wrap.clientWidth;
    c.height = wrap.clientHeight;
  }
  const ctx = c.getContext("2d");
  ctx.clearRect(0, 0, c.width, c.height);
  if (!r || !r.canvas || !s) return;
  const cr = r.canvas.getBoundingClientRect();
  const wr = wrap.getBoundingClientRect();
  // The game draws at ~2x its own coordinate units into the backing store
  // (measured: dino at game x=50 lands at pixel 104, obstacle at 136 at
  // pixel 274). cssPerGameUnit therefore combines that draw scale with the
  // css downscale of the backing store.
  // MEASURED, not assumed: the backing store is 866x300 shown in 433x150 CSS
  // px while dimensions.WIDTH is 433, so ONE GAME UNIT IS ONE CSS PIXEL.
  // The previous drawScale term double-counted the 2x backing store and
  // inflated every overlay coordinate, which is what pushed the obstacle
  // boxes forward and lifted the lines off the ground.
  const scale = cr.width / ((r.dimensions && r.dimensions.WIDTH) || cr.width);
  const ox0 = cr.left - wr.left;
  const oy0 = cr.top - wr.top;
  // Sprites are drawn 2px below their reported yPos (constant offset,
  // verified over 6 airborne samples with zero variance).
  const SPRITE_DY = 2;
  // Sprite insets MEASURED per obstacle type - a single constant cannot fit
  // both. Cactus ink fills its box (top = yPos+1); the pterodactyl sprite
  // carries transparent padding, so its ink starts 8px down and is only 22px
  // tall inside a 40px box. Using 8 for everything drew every cactus box 7px
  // too low, which is the "shifted down" that was reported.
  const obstacleInset = (o) => {
    const t = (o.typeConfig && o.typeConfig.type) || "";
    if (t === "PTERODACTYL") return { dx: -1, dy: 8, h: 22 };
    return { dx: 1, dy: 1, h: (o.typeConfig && o.typeConfig.height) || 35 };
  };
  // config.HEIGHT (47) is the sprite-sheet CELL height; the dino's visible
  // hull is 35px (measured: top CSS 95, bottom CSS 130 on a grounded frame).
  // Using 47 drew the box 12px through the ground.
  const DINO_H = 35;
  // The VISUAL ground line. Measured at CSS y=131 on a 150-tall canvas;
  // expressed as a ratio so a resize cannot silently invalidate it.
  // groundYPos reports 93, a collision-space value that is NOT where the
  // ground renders - using it put the landing line 38px too high.
  const groundCSS = ((r.dimensions && r.dimensions.HEIGHT) || 150) * (131 / 150);
  const top = oy0;
  // takeoff window of the ACTIVE genome
  if (g && !g.never && !g.always) {
    let w = windowAt(g, s.speed);
  // stash for captureSituation: lets a failure be described as "took off 8px
  // outside the window" rather than just "failed"
  // apply what previous failures taught about THIS situation
  if (g.ctxAdj) {
    const probe = {
      wide: !!s.wide,
      speed: s.speed,
      next: s.gap2 < 99999 ? s.gap2 - s.gap : null,
    };
    const adj = g.ctxAdj[ctxKey(probe)];
    if (adj) w = { lo: Math.max(5, w.lo + adj), hi: Math.max(15, w.hi + adj) };
  }
  // KNN correction from the CONTINUOUS neighbourhood, layered on the
  // bucketed ctxAdj. It only speaks when the local neighbourhood holds both
  // outcomes and the failure rate is material, so it stays silent where
  // there is nothing to fix.
  const sug = knn.suggest(
    knnFeaturize({ gap: s.gap, speed: s.speed, width: s.width,
                   next: s.gap2 < 99999 ? s.gap2 - s.gap : null }),
  );
  if (sug) {
    const shift = Math.round(sug.delta * 0.5);
    w = { lo: Math.max(5, w.lo + shift), hi: Math.max(15, w.hi + shift) };
  }
  g.__wLo = Math.round(w.lo);
  g.__wHi = Math.round(w.hi);
  // WIDTH-AWARE takeoff. Measured taxonomy over 20 deaths: 16 were "jumped
  // early, descended onto the obstacle", dominated by w>=70 cacti. The dino
  // clears 17 frames * speed px while high enough; a w75 needs 119px
  // including its own body, so at speed 6 it covers only 102 and CANNOT
  // clear - the response must differ by width, not just by speed.
  if (s.wide) {
    const adj = g.wideAdj || 0;
    w = { lo: Math.max(5, w.lo + adj), hi: Math.max(15, w.hi + adj) };
  }
    const x0 = cr.left - wr.left + (r.tRex.xPos + 44 + w.lo) * scale;
    ctx.fillStyle = "rgba(61,220,132,.18)";
    ctx.fillRect(x0, top, (w.hi - w.lo) * scale, cr.height);
  }
  // DINO BOUNDING BOX, so the agent's own hull is visible against the
  // obstacle boxes rather than implied by a bare x position.
  {
    const dxp = ox0 + r.tRex.xPos * scale;
    const dyp = oy0 + (r.tRex.yPos + SPRITE_DY) * scale;
    const dw = (r.tRex.config?.WIDTH || 44) * scale;
    const dh = DINO_H * scale;
    ctx.strokeStyle = "rgba(255,214,0,.95)";
    ctx.lineWidth = 1.5;
    ctx.strokeRect(dxp + 0.5, dyp + 0.5, dw, dh);
    ctx.fillStyle = "rgba(255,214,0,.12)";
    ctx.fillRect(dxp, dyp, dw, dh);
    // JUMP-START marker: vertical BLUE line at the x where this jump began,
    // drawn one hull-height tall. Together with the landing line it makes
    // the whole jump arc legible as a span, not a single instant.
    if (r.tRex.jumping && jumpStartX != null) {
      const jx = cr.left - wr.left + jumpStartX * scale;
      // Hull-height vertical line rising from the VISUAL ground. Anchoring
      // it to groundYPos(93) drew it floating well above the ground line.
      const feet = oy0 + groundCSS * scale;
      ctx.strokeStyle = "rgba(80,160,255,.95)";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(jx, feet);
      ctx.lineTo(jx, feet - dh);
      ctx.stroke();
    }
  }

  // TRAJECTORY ARC between the jump-start line and the landing line.
  // Integrated with the game's OWN gravity/velocity rather than a drawn
  // parabola, so the curve is the predicted path, not a decorative shape.
  if (r.tRex.jumping && jumpStartX != null) {
    const grav = (r.tRex.config && r.tRex.config.GRAVITY) || 0.6;
    const ground = r.tRex.groundYPos || 93;
    let y = r.tRex.yPos, v = r.tRex.jumpVelocity || 0, dx = 0;
    const pts = [[0, y]];
    for (let i = 0; i < 260 && (y < ground || v < 0); i++) {
      v += grav; y += v; dx += r.currentSpeed || 6;
      pts.push([dx, Math.min(y, ground)]);
      if (y >= ground) break;
    }
    if (pts.length > 2) {
      const bx = ox0 + (r.tRex.xPos + ((r.tRex.config && r.tRex.config.WIDTH) || 44)) * scale;
      const hw = 0;
      ctx.strokeStyle = "rgba(80,160,255,.9)";
      ctx.lineWidth = 2;
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      pts.forEach((p, i) => {
        const px = bx + hw + p[0] * scale;
        const py = oy0 + (p[1] + SPRITE_DY) * scale;
        i ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
      });
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }

  // LANDING MARKER: where this jump actually puts us down. A marker sitting
  // on top of an obstacle is a mistimed takeoff made visible.
  const land = predictLanding(r);
  if (land) {
    // Landing x is measured from the dino's own left edge, not from a
    // hardcoded +44 that assumed a different origin.
    const lx = ox0 + (r.tRex.xPos + land.dx) * scale;
    const gy = oy0 + groundCSS * scale;
    // does it land ON the obstacle?
    const onObstacle =
      s.gap < 99999 && land.dx > s.gap - 10 && land.dx < s.gap + (s.width || 0) + 10;
    // LIGHT BLUE landing marker. Turns hot pink only when it lands ON the
    // obstacle, which is the one case worth shouting about.
    const landCol = onObstacle ? "rgba(255,61,127,.95)" : "rgba(120,200,255,.95)";
    // HORIZONTAL blue line at the estimated landing spot, spanning the hull
    // width the dino will occupy on touchdown - so overlap with an obstacle
    // box is read off directly instead of inferred from a single tick.
    const hullW = (r.tRex.config?.WIDTH || 44) * scale;
    ctx.strokeStyle = landCol;
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.moveTo(lx - hullW / 2, gy);
    ctx.lineTo(lx + hullW / 2, gy);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(lx, gy, 3.5, 0, Math.PI * 2);
    ctx.fillStyle = landCol;
    ctx.fill();
  }

  // COLLISION POINT on failure: a red dot at the centre of the overlap
  // between the dino's box and the obstacle it hit. Drawn only while
  // crashed, so the frame that killed the run is self-explaining.
  if (r.crashed) {
    const dL = r.tRex.xPos, dR = dL + (r.tRex.config.WIDTH || 44);
    const dT = r.tRex.yPos + SPRITE_DY, dB = dT + DINO_H;
    let hit = null, bestArea = 0;
    for (const o of (r.horizon && r.horizon.obstacles) || []) {
      const oL = o.xPos, oR = oL + (o.width || 10);
      const oi = obstacleInset(o);
      const oT = (o.yPos || 0) + oi.dy;
      const oB = oT + oi.h;
      const iw = Math.min(dR, oR) - Math.max(dL, oL);
      const ih = Math.min(dB, oB) - Math.max(dT, oT);
      // Overlapping boxes are the real hit. If nothing overlaps (the game
      // can crash a frame after separation) fall back to the nearest one.
      const area = iw > 0 && ih > 0 ? iw * ih : 0;
      const score = area > 0 ? area : -Math.abs(oL - dR) / 1000;
      if (hit === null || score > bestArea) { bestArea = score; hit = { oL, oR, oT, oB, area }; }
    }
    if (hit) {
      const cxp = hit.area > 0
        ? (Math.max(dL, hit.oL) + Math.min(dR, hit.oR)) / 2
        : (dR + hit.oL) / 2;
      const cyp = hit.area > 0
        ? (Math.max(dT, hit.oT) + Math.min(dB, hit.oB)) / 2
        : (Math.max(dT, hit.oT) + Math.min(dB, hit.oB)) / 2;
      const px = ox0 + cxp * scale, py = oy0 + cyp * scale;
      ctx.fillStyle = "rgba(255,40,40,.98)";
      ctx.beginPath(); ctx.arc(px, py, 5, 0, Math.PI * 2); ctx.fill();
      ctx.strokeStyle = "rgba(255,255,255,.9)"; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(px, py, 5, 0, Math.PI * 2); ctx.stroke();
    }
  }

  // LIGHT RED over every obstacle currently on screen, so what the agent is
  // trying to clear is visible alongside where it will land.
  const tx = r.tRex.xPos + 44;
  for (const o of (r.horizon && r.horizon.obstacles) || []) {
    const d = o.xPos - tx;
    if (d < -40 || d > 400) continue;
    // TRUE BOUNDING BOX, not a full-height stripe. A stripe cannot show
    // vertical extent, so a high-flying pterodactyl and a ground cactus
    // looked identical - exactly the distinction the jump decision turns on.
    // Draw at the obstacle's OWN xPos. The old form added the dino's xPos
    // plus a hardcoded 44 on top of the gap, which shifted every box
    // forward by roughly a dino width.
    const ins = obstacleInset(o);
    const ox = ox0 + (o.xPos + ins.dx) * scale;
    const ow = Math.max(3, (o.width || 10) * scale);
    const oh = Math.max(3, ins.h * scale);
    const oy = oy0 + ((o.yPos || 0) + ins.dy) * scale;
    ctx.fillStyle = "rgba(255,120,120,.22)";
    ctx.fillRect(ox, oy, ow, oh);
    ctx.strokeStyle = "rgba(255,120,120,.95)";
    ctx.lineWidth = 1;
    ctx.strokeRect(ox + 0.5, oy + 0.5, ow, oh);
  }
}

/** One frame: decide, act, advance the real game by one frame. */
/**
 * Session KNN over takeoff outcomes. Trains incrementally - every resolved
 * jump is one example, no retraining step - and persists across reloads.
 * Replaces hard situation buckets, which split a continuum into classes that
 * shared no information (speed 7.9 vs 8.1 were unrelated; 6.0 vs 7.9 were
 * identical).
 */
let knn = new TakeoffKNN(600);

/**
 * Parametric model with TRAINABLE WEIGHTS, corrected after every outcome.
 * KNN memorises; this one is actually trained. Weight updates are deferred
 * to episode boundaries because trainOnBatch measured 4.99 ms against a
 * 16.7 ms frame budget.
 */
let mlp = new JumpModel();

/**
 * STATE x FAIL-REASON x PROP table. The three dimensions existed separately;
 * this is where they meet, which is what turns "38% early" into "in
 * wide/slow, early deaths jump at gap 72 while survivals jump at 41".
 */
let analysis = {};

let epFrames = 0;
/** Telemetry accumulated within the current episode. */
let tel = null;
/** Context of the most recent takeoff, blamed if the episode ends badly. */
function newDecision() {
  return { gap: null, speed: null, width: null, wide: false, next: null, arc: null, frame: 0 };
}

function newTel() {
  return {
    /** per-situation outcome counts: key -> {ok, fail} */
    ctx: {},
    lastDecision: null,
    jumps: 0, ducks: 0, panics: 0,
    framesAir: 0, frames: 0,
    wideSeen: 0, wideCleared: 0,
    birdsSeen: 0, birdsCleared: 0,
    missedWindow: 0,      // window open but airborne - cannot act
    lastGap: 99999, lastWide: false, lastHigh: false, lastAir: false, lastY: 93, lastSpeed: 6,
  };
}
let bandDist = BANDS.map(() => 0);   // distance travelled per speed band
let lastDist = 0;
const EP_FRAME_CAP = 9000;  // ~150s of game time; beyond this the candidate
                            // is clearly strong and further frames add no
                            // ranking information, only wall-clock delay.

function frame() {
  if (!running) return;
  // The model under test this run. Falls back to the population slot only
  // if the champion loop has not initialised yet.
  if (!current) pickCandidate();
  const cand = current ? { g: current.g } : pop[idx];
  const s = read();
  if (!s) return;

  // treat a capped run as a completed episode rather than letting one
  // immortal policy stall the whole generation
  if (!s.crashed && ++epFrames > EP_FRAME_CAP) {
    const dist = Math.max(0, s.distance - epStart);
    recordEpisode(cand, dist, pop, bandDist, { ...(tel || newTel()), cause: "capped" });
    tel = newTel();
    bandDist = BANDS.map(() => 0);
    lastDist = 0;
    log(`${candName(cand)} -> ${dist} (capped)`);
    tableDirty = true;
    epFrames = 0;
    restartEpisode();
    return;
  }

  if (s.crashed) {
    if (!sawCrash) {
      sawCrash = true;
      const dist = Math.max(0, s.distance - epStart);
      // CLASSIFY the death from the state at impact. This is the taxonomy:
      // early (descending onto it), late (ascending into it), missed (never
      // jumped), bird, or wide-obstacle failure.
      const t = tel || newTel();
      // BLAME: the takeoff that was still in flight when we died
      // BLAME THE RIGHT DECISION.
      // Airborne at death -> the jump in flight caused it, and its takeoff
      // gap is the thing to correct. Grounded at death -> no jump is at
      // fault; the failure is that none was made, so record the window the
      // agent SHOULD have used rather than charging an unrelated success.
      const airborneAtDeath = !!(t.lastAir);
      // cause is computed just below; capture it first so the joined table
      // gets state x reason x props in one place
      let deathCause;
      if (t.lastHigh) deathCause = "bird";
      else if (!t.lastAir) deathCause = "missed";
      else if (t.lastY > 60) deathCause = "early";
      else deathCause = "late";

      if (airborneAtDeath && t.lastDecision) {
        Analysis.record(analysis, ctxKey(t.lastDecision), deathCause, withTtc(t.lastDecision));
        const k = ctxKey(t.lastDecision);
        const e = (t.ctx[k] = t.ctx[k] || { ok: 0, fail: 0, okGap: [], failGap: [], noJump: 0 });
        e.fail++;
        if (e.failGap.length < 30) e.failGap.push(t.lastDecision.takeoffGap);
        knn.add(knnFeaturize(t.lastDecision), false, t.lastDecision.takeoffGap);
        mlp.observe(featurize(mlpFeatFrom(t.lastDecision)), false);
        t.lastFailure = { ...t.lastDecision, kind: "badJump" };
      } else if (!airborneAtDeath) {
        // omission: count it separately so it cannot corrupt the timing stats
        const probe = {
          wide: !!t.lastWide,
          speed: t.lastSpeed || 6,
          next: null,
        };
        const k = ctxKey(probe);
        const e = (t.ctx[k] = t.ctx[k] || { ok: 0, fail: 0, okGap: [], failGap: [], noJump: 0 });
        e.noJump = (e.noJump || 0) + 1;
        // An omission is a NEGATIVE example for this situation: at this
        // gap/speed/width, failing to take off ended the run. Without these
        // the model saw only successes (measured 104 ok / 0 fail, pOk 1.0)
        // and could never fire a correction.
        if (Number.isFinite(t.lastGap) && Math.abs(t.lastGap) < 500) {
          const omission = knnFeaturize({ gap: t.lastGap, speed: t.lastSpeed || 6,
                                          width: t.lastWidth || 0, next: null });
          knn.add(omission, false, Math.round(t.lastGap));
          mlp.observe(
            featurize({ gap: t.lastGap, speed: t.lastSpeed || 6,
                        width: t.lastWidth || 0, next: null,
                        high: t.lastHigh, wide: (t.lastWidth || 0) >= 50,
                        arcVel: 0 }),   // arcVel 0 = no jump was made
            false,
          );
        }
        t.lastFailure = { kind: "noJump", ctx: k, speed: probe.speed, wide: probe.wide };
      }
      let cause;
      if (t.lastHigh) cause = "bird";
      else if (!t.lastAir) cause = "missed";
      else if (t.lastY > 60) cause = "early";
      else cause = "late";
      if (t.lastWide && cause !== "bird") cause = "wide_" + cause;
      recordEpisode(cand, dist, pop, bandDist, { ...t, cause });
      tel = newTel();
      bandDist = BANDS.map(() => 0);
      lastDist = 0;
      epInCand++;
      // JUDGE the run against the reigning champion, then log ONLY the four
      // things asked for: model name, its best, its run count, last change.
      const pts = Math.round(dist * 0.025);
      const verdict = current ? judgeRun(current.role, pts) : null;
      if (verdict) {
        const c = verdict.champion;
        log(
          `${c.edition || "initial"} · avg ${c.mean || c.best}pts · best ${c.best}pts · runs ${c.runs}` +
            (verdict.promoted ? " · PROMOTED" : ` · tried ${pts}pts`),
          "episode",
          { edition: c.edition || "initial", hash: c.hash, best: c.best,
            runs: c.runs, result: pts, promoted: verdict.promoted },
        );
      }
      // persist the learned model with the episode, not only at generation
      // close - otherwise a short session looks like nothing was learned
      saveKnn();
      saveAnalysis();
      // correct the weights with everything this episode observed
      // Flush NOW, before the next episode starts, so the next run really
      // is executed with corrected weights. Batched deferral meant a run
      // could begin on the same weights that had just failed.
      mlp.flush(64).then((n) => {
        if (!n) return;
        saveMlp();
        const loss = mlp.lastLoss;
        log(
          `trained on ${n} outcomes · ${mlp.samplesTrained} total` +
            (loss != null ? ` · loss ${Number(loss).toFixed(3)}` : ""),
          "train",
          { batch: n, total: mlp.samplesTrained, loss },
        );
      });
      tableDirty = true;

      // Drive on cand.runs, which PERSISTS, not on epInCand which resets on
      // reload - that mismatch let candidates reach 28 runs against a budget
      // of 9 while the generation never closed (gen 0 after 63 episodes).
      const budget = episodesFor(currentState().generation);
      if ((cand.runs || 0) >= budget) {
        epInCand = 0;
        // advance to the next candidate that still owes episodes, INCLUDING
        // controls - they are the baseline that makes a score meaningful.
        do { idx++; } while (idx < pop.length && (pop[idx].runs || 0) >= budget);
        if (idx >= pop.length) {
          // generation complete: rank, keep elites, breed the next
          const { best, bestCtrl } = closeGeneration(pop);
          log(
            `gen ${currentState().generation} closed · best ${Math.round((best && best.median || 0) * 0.025)}pts · control ${Math.round(bestCtrl * 0.025)}pts`,
            "generation",
          );
          saveKnn();
          saveAnalysis();
          const S = currentState();
          log(
            `gen ${S.generation}: best ${best ? best.mean : 0} vs control ${bestCtrl}` +
              (best && best.mean <= bestCtrl ? "  [SUSPECT: control wins]" : ""),
          );
          pop = seedPopulation();
          idx = 0;
          drawChart();
          renderTable();
        }
      }
    }
    epFrames = 0;
    restartEpisode();
    return;
  }

  if (!s.started) {
    act("jump");
    CLOCK.step();
    return;
  }

  // ---- telemetry: what the agent SAW and DID, per frame
  if (!tel) tel = newTel();
  tel.frames++;
  if (s.airborne) tel.framesAir++;
  if (s.gap < 99999 && s.gap > -10) {
    if (s.wide && !tel.lastWide) tel.wideSeen++;
    if (s.high && !tel.lastHigh) tel.birdsSeen++;
  }
  // an obstacle that moved from ahead to behind was cleared
  if (tel.lastGap > 0 && s.gap < 0 && s.gap > -30) {
    if (tel.lastWide) tel.wideCleared++;
    if (tel.lastHigh) tel.birdsCleared++;
  }
  tel.lastGap = s.gap; tel.lastWide = s.wide; tel.lastHigh = s.high;
  tel.lastAir = s.airborne; tel.lastY = Math.round(s.y === undefined ? 93 : s.y);
  tel.lastSpeed = s.speed;
  tel.lastWidth = s.width;

  // attribute progress to the band that was active while it was earned
  const here = Math.max(0, s.distance - epStart);
  const gained = Math.max(0, here - lastDist);
  bandDist[bandOf(s.speed)] += gained;
  lastDist = here;

  const d = decide(s, cand.g);
  // safety: never leave the dino stuck ducking on the ground, which silently
  // disables jumping for the rest of the episode
  if (!s.airborne && d.action !== "duck") {
    const rr = R();
    if (rr && rr.tRex && rr.tRex.ducking) {
      rr.onKeyUp({ keyCode: 40, type: "keyup", preventDefault() {}, target: {} });
    }
  }
  // record the takeoff context so a later crash can be attributed to it
  if (
    (d.action === "jump" || d.action === "jumpLow" || d.action === "jumpHigh") &&
    s.gap < 9000   // reject the no-obstacle sentinel: a jump at nothing
                   // carries no timing information and produced fail@99999
                   // with a -99961 "correction"
  ) {
    tel.lastDecision = captureSituation(s, d.action, cand.g);
    tel.lastDecision.frame = tel.frames;
    // keep the fields ctxKey needs
    tel.lastDecision.gap = tel.lastDecision.takeoffGap;
    tel.lastDecision.width = tel.lastDecision.targetWidth;
    tel.lastDecision.next = tel.lastDecision.nextDelta;
  }
  // A landed jump is resolved and can no longer be blamed. Without this a
  // successful takeoff stayed "live" for the rest of the episode and was
  // charged with a death 38 steps later while the dino was on the ground.
  if (tel.lastDecision && !s.airborne && tel.frames - tel.lastDecision.frame > 3) {
    const k = ctxKey(tel.lastDecision);
    const e = (tel.ctx[k] = tel.ctx[k] || { ok: 0, fail: 0, okGap: [], failGap: [], noJump: 0 });
    e.ok++;
    if (e.okGap.length < 30) e.okGap.push(tel.lastDecision.takeoffGap);
    knn.add(knnFeaturize(tel.lastDecision), true, tel.lastDecision.takeoffGap);
    mlp.observe(featurize(mlpFeatFrom(tel.lastDecision)), true);
    Analysis.record(analysis, ctxKey(tel.lastDecision), "ok", withTtc(tel.lastDecision));
    tel.lastDecision = null;
  }
  // a jump is credited as SUCCESSFUL once its obstacle is behind us
  if (tel.lastDecision && s.gap < -10 && tel.frames - tel.lastDecision.frame > 3) {
    const k = ctxKey(tel.lastDecision);
    const e = (tel.ctx[k] = tel.ctx[k] || { ok: 0, fail: 0, okGap: [], failGap: [] });
    e.ok++;
    // remember WHICH takeoff gap worked here - this is what tuning needs
    if (e.okGap.length < 30) e.okGap.push(tel.lastDecision.takeoffGap);
    knn.add(knnFeaturize(tel.lastDecision), true, tel.lastDecision.takeoffGap);
    mlp.observe(featurize(mlpFeatFrom(tel.lastDecision)), true);
    Analysis.record(analysis, ctxKey(tel.lastDecision), "ok", withTtc(tel.lastDecision));
    tel.lastDecision = null;
  }
  if (d.action === "jump") tel.jumps++;
  else if (d.action === "duck") tel.ducks++;
  if (d.panic) tel.panics++;
  // window was open but the dino was airborne, so it could not act
  if (s.gap < 99999 && !s.high && s.airborne) tel.missedWindow++;
  act(d.action);
  CLOCK.step();

  // presentation is decoupled: stash the latest state, paint on a timer
  latest = { s, d, cand };
}

/** Repaint from the most recent simulated frame. Called at ~20fps, not per frame. */
let latest = null;
let tableDirty = false;
function paint() {
  if (!latest) return;
  const { s, d, cand } = latest;
  $("act").textContent = d.action;
  $("dist").textContent = Math.round(Math.max(0, s.distance - epStart) * 0.025);
  $("v-gap").textContent = s.gap < 99999 ? Math.round(s.gap) : "—";
  $("v-spd").textContent = s.speed.toFixed(1);
  $("m-gap").style.width = s.gap < 99999 ? `${Math.max(0, 100 - s.gap / 4)}%` : "0%";
  $("m-spd").style.width = `${Math.min(100, (s.speed / 13) * 100)}%`;
  $("f-high").className = "flag" + (s.high ? " on" : "");
  $("f-air").className = "flag" + (s.airborne ? " on" : "");
  $("pname").textContent = candName(cand);
  renderRules(d.rule, s, cand.g);
  overlay(s, cand.g);
}

/**
 * Liveness gate: refuse to score until the real game is proven to advance
 * AND to move obstacles toward the player. The second half matters - an
 * earlier fake clock advanced the odometer while obstacles drifted away.
 */
function verifyLiveness() {
  const r = R();
  if (!r) return { ok: false, why: "Runner not constructed" };
  r.restart();
  for (let i = 0; i < 6; i++) CLOCK.step();
  act("jump");
  for (let i = 0; i < 10; i++) CLOCK.step();
  const a = read();
  for (let i = 0; i < 40; i++) CLOCK.step();
  const b = read();
  if (!a || !b) return { ok: false, why: "cannot read state" };
  if (b.distance <= a.distance) {
    return { ok: false, why: `not advancing (${a.distance} -> ${b.distance})` };
  }
  const first = (r.horizon.obstacles || [])[0];
  if (first) {
    const x0 = first.xPos;
    for (let i = 0; i < 20; i++) CLOCK.step();
    if ((r.horizon.obstacles || []).includes(first) && first.xPos > x0) {
      return { ok: false, why: "obstacles moving AWAY - clock corrupts the sim" };
    }
  }
  return { ok: true, why: `advancing ${a.distance}->${b.distance}, obstacles approach` };
}

// ---- boot ---------------------------------------------------------------
pop = seedPopulation();
renderTable();
drawChart();

$("run").onclick = () => {
  const v = verifyLiveness();
  $("liveness").textContent = `liveness: ${v.ok ? "OK" : "FAILED"} · ${v.why}`;
  $("liveness").className = "tag " + (v.ok ? "live" : "dead");
  if (!v.ok) {
    log(`refusing to run: ${v.why}`);
    return;
  }
  running = true;
  // hold the screen on for the duration of the run
  acquireWakeLock().then((mode) => {
    if (mode !== "none") log(`screen kept awake (${mode})`, "system");
  });
  jumpLatched = false;
  sawCrash = false;
  epFrames = 0;
  $("run").disabled = true;
  $("stop").disabled = false;
  $("status").textContent = "evolving";
  $("status").className = "tag live";
  restartEpisode();
  // Unbounded: no step cap, no episode cap. Runs until stopped.
  // Driven by BOTH rAF and setInterval because either can be throttled when
  // the tab is backgrounded - measured, a setInterval-only loop froze with
  // distance stuck at 4795 and one frame queued but never stepped.
  let lastSeen = -1;
  let stalls = 0;
  const rateEl = $("rate");
  let lastPump = performance.now();
  // forward reference so the painter can drive the watchdog too
  let watchdogRef = () => {};
  const pump = () => {
    if (!running) return;
    try {
    const rate = rateEl ? Number(rateEl.value) : 12;
    const now = performance.now();
    const elapsed = now - lastPump;
    lastPump = now;
    // frames owed = real elapsed time x rate, at 60fps. Clamped so a long
    // throttled gap cannot produce a thousand-frame burst that looks like a
    // freeze followed by a teleport.
    // rate 0 == MAX: unpaced, CPU-bound. Anything else is wall-clock paced
    // so 1x really is real time.
    // Each frame() call is one DECISION plus one simulated frame, so the
    // agent is never blind between frames no matter how fast we pump.
    // Measured: batching frames between decisions cost 5x performance.
    // Burst is capped at 4: measured, spacing decisions 8 frames apart costs
    // 3.5x performance (median 4348 vs 15398), while 4 is indistinguishable
    // from 1. Max rate therefore pumps MORE OFTEN rather than more per pump.
    // frame() = exactly one decide + one clock step, so a large burst keeps
    // decisions dense (1 per frame) while giving full throughput. The
    // measured 3.5x loss came from stepping the clock 8x per DECISION, which
    // is a different thing entirely and cannot happen here.
    // 600 frames is ~10 seconds of game time per batch: fast enough that a
    // generation completes in seconds, small enough that the event loop gets
    // control back between batches.
    const owed =
      rate === 0
        ? 600
        : Math.min(240, Math.max(1, Math.round((elapsed / (1000 / 60)) * rate)));
      for (let i = 0; i < owed; i++) frame();
    } catch (e) {
      // A throw inside frame() previously killed the loop with no trace: the
      // timer kept firing, pump() kept throwing, and the UI still read
      // "evolving". Surface it and stop cleanly instead.
      running = false;
      releaseWakeLock();
      $("status").textContent = "error";
      $("status").className = "tag dead";
      log(`ERROR: ${String((e && e.message) || e).slice(0, 80)}`, "error");
      $("run").disabled = false;
      $("stop").disabled = true;
    }
  };
  // The harness PATCHES window.requestAnimationFrame to queue into __CLOCK,
  // so using it here deadlocks: the pump waits on a frame that only the pump
  // can deliver. Use the real one captured at boot.
  const realRAF = window.__CLOCK.realRAF || ((cb) => setTimeout(() => cb(performance.now()), 16));
  const rafPump = () => {
    if (!running) return;
    // Only pump from rAF above real time; at 1x the interval alone gives
    // ~60 game frames/sec and adding rAF would stop it being "real speed".
    if (!rateEl || Number(rateEl.value) > 2 || Number(rateEl.value) === 0) pump();
    // Painting here keeps the overlay on the same frame the display shows.
    paint();
    rafId = realRAF(rafPump);
  };
  painter = setInterval(() => {
    watchdogRef();
    paint();
    if (tableDirty) {
      tableDirty = false;
      renderTable();
      drawChart();
    }
  }, 50);
  // At max rate, drive from a 0ms timer so we get many SMALL bursts rather
  // than one large one - same throughput, dense decisions.
  // Self-RESTARTING, not self-terminating: it previously stopped for good if
  // the rate was not 0 at the instant it fired, so switching to max mid-run
  // never resumed the fast path.
  const fastTick = () => {
    if (!running) return;
    if (rateEl && Number(rateEl.value) === 0) {
      // ONE bounded batch, then yield. 40 x 3000 = 120,000 frames in a single
      // synchronous burst froze the page; the budget below keeps the thread
      // responsive while still far outpacing real time.
      pump();
    }
    setTimeout(fastTick, 0);
  };
  setTimeout(fastTick, 0);
  // WATCHDOG: if nothing advanced since the last check, pump directly rather
  // than waiting for a timer that may be throttled. This is what keeps a run
  // alive when the browser starves setTimeout/rAF.
  let lastOdo = -1;
  let idle = 0;
  const watchdog = () => {
    if (!running) return;
    const s = read();
    const odo = s ? s.distance : -1;
    if (odo === lastOdo) {
      idle++;
      // escalate: one missed check is normal jitter, several means starved
      if (idle >= 2) {
        for (let i = 0; i < 300; i++) frame();
        idle = 0;
      }
    } else {
      idle = 0;
    }
    lastOdo = odo;
  };
  watchdogRef = watchdog;

  loop = setInterval(() => {
    pump();
    // Repaint in the SAME turn the world advanced, so the overlay cannot
    // draw a position the game has already scrolled past.
    paint();
    watchdog();
    // watchdog: if the odometer has not moved between ticks, the game lost
    // its queued frame - re-prime it rather than silently freezing.
    const s = read();
    const d = s ? s.distance : -1;
    if (d === lastSeen) {
      stalls++;
      if (stalls > 3) {
        stalls = 0;
        const r = R();
        if (r && !r.crashed) {
          for (let i = 0; i < 4; i++) CLOCK.step();
        } else {
          restartEpisode();
        }
      }
    } else {
      stalls = 0;
    }
    lastSeen = d;
  }, 16);
  rafId = window.requestAnimationFrame(rafPump);
};

$("stop").onclick = () => {
  running = false;
  releaseWakeLock();
  clearInterval(loop);
  clearInterval(painter);
  if (rafId) window.cancelAnimationFrame(rafId);
  paint();
  renderTable();
  $("run").disabled = false;
  $("stop").disabled = true;
  $("status").textContent = "stopped";
  $("status").className = "tag";
};

function refreshAll() {
  pop = seedPopulation();
  idx = 0;
  epInCand = 0;
  renderTable();
  drawChart();
}

/** Persist the learned model. Bounded by its own cap, so this stays small. */
function saveAnalysis() {
  try { localStorage.setItem("dino-analysis-v1", JSON.stringify(analysis)); }
  catch { /* storage may be full */ }
}
function loadAnalysis() {
  try {
    const raw = localStorage.getItem("dino-analysis-v1");
    if (raw) analysis = JSON.parse(raw);
  } catch { analysis = {}; }
}
loadAnalysis();

let logSaveTimer = null;
function saveLogs() {
  // debounced: the log is written on every entry and episodes are frequent
  if (logSaveTimer) return;
  logSaveTimer = setTimeout(() => {
    logSaveTimer = null;
    try { localStorage.setItem("dino-logs-v1", JSON.stringify(logs.toJSON())); }
    catch { /* storage may be full */ }
  }, 500);
}
function loadLogs() {
  try {
    const raw = localStorage.getItem("dino-logs-v1");
    if (raw) logs = LogStore.fromJSON(JSON.parse(raw));
  } catch { /* corrupt log must not block startup */ }
}
loadLogs();

function saveMlp() {
  try { localStorage.setItem("dino-mlp-v1", JSON.stringify(mlp.toJSON())); }
  catch { /* storage may be full */ }
}
function loadMlp() {
  try {
    const raw = localStorage.getItem("dino-mlp-v1");
    if (raw) mlp.loadFrom(JSON.parse(raw));
  } catch { /* corrupt weights must not block startup */ }
}
loadMlp();

function saveKnn() {
  try { localStorage.setItem("dino-knn-v1", JSON.stringify(knn.toJSON())); }
  catch { /* storage may be full or unavailable */ }
}
function loadKnn() {
  try {
    const raw = localStorage.getItem("dino-knn-v1");
    if (raw) knn = TakeoffKNN.fromJSON(JSON.parse(raw));
  } catch { /* corrupt state must not block startup */ }
}
loadKnn();

/**
 * SHARED BEST TRAINING.
 *
 * The local loads above are synchronous and already ran, so a pull that
 * lands later must RE-load the engines from the freshly installed state -
 * otherwise the imported genome would sit in localStorage while the engines
 * kept running the local one, which looks like a working sync and is not.
 *
 * Every failure path is non-fatal: the page must work offline, on a blocked
 * network, and with the share layer unconfigured.
 */
(async () => {
  try {
    const r = await share.pullIfBetter();
    if (r && r.ok) {
      reloadFromStorage();
      loadAnalysis();
      loadMlp();
      loadKnn();
      renderTable();
      log(`shared best loaded · ${r.remoteQ}pts avg${r.edition ? " · " + r.edition : ""}`);
    } else if (r && r.why === "empty") {
      log("shared best: pool is empty - yours will seed it");
    } else if (r && r.why === "local is as good or better") {
      log(`shared best: keeping local (${r.localQ}pts avg >= ${r.remoteQ})`);
    } else if (r && r.why && r.why !== "not configured") {
      log(`shared best unavailable: ${r.why}${r.detail ? " (" + r.detail + ")" : ""}`);
    }
  } catch {
    /* never block startup on the network */
  }
  try { share.installAutoPush(log); } catch { /* ignore */ }
})();

const saveBtn = $("save");
if (saveBtn) {
  saveBtn.onclick = () => {
    const S = currentState();
    log(saveSlot() ? `saved: gen ${S.generation}, ep ${S.episodes}` : "save FAILED (storage unavailable)");
  };
}

const loadBtn = $("load");
if (loadBtn) {
  loadBtn.onclick = () => {
    const info = slotInfo();
    if (!info) {
      log("no saved slot to load");
      return;
    }
    if (!loadSlot()) {
      log("load FAILED");
      return;
    }
    refreshAll();
    log(`loaded: gen ${info.gen}, ep ${info.ep}, typical ${info.typical || 0}`);
  };
}

const exportBtn = $("export");
if (exportBtn) {
  exportBtn.onclick = () => {
    const S = currentState();
    // FULL session snapshot: evolution state, the trained weights, the KNN
    // examples, the joined analysis, and the log. Exporting only the
    // evolution state would drop the model and the record of what it was
    // corrected on.
    const bundle = {
      format: "dino-session-v2",
      exportedAt: Date.now(),
      evolve: JSON.parse(exportState()),
      mlp: mlp.toJSON(),
      knn: knn.toJSON(),
      analysis,
      logs: logs.toJSON(),
    };
    const blob = new Blob([JSON.stringify(bundle, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `dino-evolve-gen${S.generation}-ep${S.episodes}.json`;
    document.body.appendChild(a);
    a.click();
    setTimeout(() => {
      URL.revokeObjectURL(a.href);
      a.remove();
    }, 1000);
    log(
      `exported gen ${S.generation} · ep ${S.episodes} · ${mlp.samplesTrained} trained · ${logs.entries.length} log lines`,
      "system",
    );
  };
}

const importBtn = $("import");
const fileEl = $("file");
if (importBtn && fileEl) {
  importBtn.onclick = () => fileEl.click();
  fileEl.onchange = async () => {
    const f = fileEl.files && fileEl.files[0];
    if (!f) return;
    try {
      const text = await f.text();
      const parsed = JSON.parse(text);
      let S2;
      if (parsed && parsed.format === "dino-session-v2") {
        // full snapshot
        S2 = importState(JSON.stringify(parsed.evolve));
        if (parsed.knn) knn = TakeoffKNN.fromJSON(parsed.knn);
        if (parsed.analysis) analysis = parsed.analysis;
        if (parsed.logs) logs = LogStore.fromJSON(parsed.logs);
        if (parsed.mlp) {
          mlp.dispose();
          mlp = new JumpModel();
          await mlp.loadFrom(parsed.mlp);
        }
        saveKnn();
        saveAnalysis();
        saveMlp();
        saveLogs();
      } else {
        // older export: evolution state only, still accepted
        S2 = importState(text);
      }
      refreshAll();
      const el = $("log");
      if (el) el.textContent = logs.render(6);
      log(
        `imported gen ${S2.generation} · ep ${S2.episodes} · ${mlp.samplesTrained} trained`,
        "system",
      );
    } catch (e) {
      log(`import FAILED: ${String(e.message || e).slice(0, 60)}`);
    }
    fileEl.value = "";
  };
}

const resetBtn = $("reset");
if (resetBtn) {
  resetBtn.onclick = () => {
    // TOTAL reset: the rolling autosave AND the explicit save slot. Leaving
    // the slot behind meant a later Load silently resurrected old learning.
    try {
      localStorage.removeItem("dino-evolve-v1");
      localStorage.removeItem("dino-evolve-v1-slot");
    } catch { /* storage may be unavailable */ }
    // Also clear the GAME's own record. It survives restarts by design, so
    // after a Reset our stats start at zero while the canvas still shows an
    // old HI - which reads as a record that refuses to update.
    const rr = R();
    if (rr) {
      rr.highestScore = 0;
      if (rr.distanceMeter) {
        // clear the live-record cache as well: it is what the HUD renders,
        // and a stale value survives setHighScore alone
        rr.distanceMeter.liveHigh = 0;
        rr.distanceMeter.setHighScore(0);
      }
    }
    // stop first: a running loop re-saves its in-memory state over the wipe
    if (running) {
      running = false;
      releaseWakeLock();
      clearInterval(loop);
      clearInterval(painter);
      if (rafId) window.cancelAnimationFrame(rafId);
      $("run").disabled = false;
      $("stop").disabled = true;
    }
    // clear ALL module-level run state, not just the stores. jumpLatched is
    // released only when the game steps, so a Reset during a jump left it
    // stuck true and the next Run could never take off.
    jumpLatched = false;
    epFrames = 0;
    sawCrash = false;
    tel = null;
    lastDist = 0;
    idx = 0;
    epInCand = 0;
    // the learned model is part of the session's knowledge and must go too
    knn.clear();
    mlp.dispose();
    mlp = new JumpModel();
    analysis = {};
    // cancel the pending debounced write FIRST, or it fires after the
    // removeItem below and resurrects the log
    if (logSaveTimer) {
      clearTimeout(logSaveTimer);
      logSaveTimer = null;
    }
    logs.clear();
    const le = $("log");
    if (le) le.textContent = "";
    try {
      localStorage.removeItem("dino-knn-v1");
      localStorage.removeItem("dino-analysis-v1");
      localStorage.removeItem("dino-mlp-v1");
      localStorage.removeItem("dino-logs-v1");
    } catch { /* ignore */ }
    resetAll();
    pop = seedPopulation();
    idx = 0;
    epInCand = 0;
    renderTable();
    drawChart();
    log("population reset");
  };
}

setTimeout(() => {
  const r = R();
  $("status").textContent = r ? "ready" : "no runner";
  log(r ? `restored: session ${currentState().sessions}, gen ${currentState().generation}, best ever ${currentState().bestEver}` : "Runner missing");
  if (r) {
    for (let i = 0; i < 3; i++) CLOCK.step();
    overlay(read(), pop[0] && pop[0].g);
  }
}, 250);
