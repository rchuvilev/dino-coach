/**
 * Unbounded evolving run on the REAL dino game.
 *
 * Loop: evaluate every candidate in the generation over EPISODES_PER
 * episodes, rank by live mean, keep elites, breed, repeat — forever, until
 * stopped. Population persists to localStorage, so a reload continues.
 */
import {
  closeGeneration,
  currentState,
  ELITE,
  EPISODES_PER,
  key,
  label,
  recordEpisode,
  resetAll,
  seedPopulation,
} from "./evolve.js";

const CLOCK = window.__CLOCK;
const $ = (id) => document.getElementById(id);
const R = () => (window.Runner && window.Runner.instance_ ? window.Runner.instance_ : null);

let running = false;
let loop = null;
let rafId = null;
let pop = [];
let idx = 0;
let epInCand = 0;
let epStart = 0;
let sawCrash = false;
let logLines = [];

const log = (s) => {
  logLines.unshift(s);
  logLines = logLines.slice(0, 5);
  $("log").textContent = logLines.join("\n");
};

function read() {
  const r = R();
  if (!r || !r.tRex) return null;
  const tx = r.tRex.xPos + 44;
  let gap = 99999;
  let high = false;
  for (const o of (r.horizon && r.horizon.obstacles) || []) {
    const d = o.xPos - tx;
    if (d > -30 && d < gap) {
      gap = d;
      high = o.yPos < 60;
    }
  }
  return {
    crashed: !!r.crashed,
    started: !!r.started,
    distance: Math.round(r.distanceRan || 0),
    speed: r.currentSpeed || 0,
    airborne: r.tRex.yPos < (r.tRex.groundYPos || 93) - 4,
    gap,
    high,
  };
}

/** Act through the game's own key handlers, so input follows its real path. */
function act(a) {
  const r = R();
  if (!r) return;
  const ev = (kc, t) => ({ keyCode: kc, type: t, preventDefault() {}, target: {} });
  if (a === "jump") {
    r.onKeyDown(ev(38, "keydown"));
    r.onKeyUp(ev(38, "keyup"));
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
  const canJump = g.lo >= 0 && has && !s.high && s.gap >= g.lo && s.gap <= g.hi && !s.airborne;
  if (g.duckFirst) {
    if (canDuck) return { action: "duck", rule: 1 };
    if (canJump) return { action: "jump", rule: 0 };
  } else {
    if (canJump) return { action: "jump", rule: 0 };
    if (canDuck) return { action: "duck", rule: 1 };
  }
  return { action: "run", rule: 2 };
}

function restartEpisode() {
  const r = R();
  if (!r) return;
  r.restart();
  for (let i = 0; i < 8; i++) CLOCK.step();
  // a jump starts the game
  act("jump");
  for (let i = 0; i < 6; i++) CLOCK.step();
  const s = read();
  // DELTA baseline: restart() does not zero distanceRan in this build
  epStart = s ? s.distance : 0;
  sawCrash = false;
}

function candName(c) {
  return c.ctrl ? `control: ${c.ctrl}` : label(c.g);
}

function renderRules(fired, s, g) {
  const has = s && s.gap < 99999;
  const rows = [
    { when: `gap ${g.lo}..${g.hi} & !airborne`, then: "jump" },
    { when: `high & gap<=${g.duck}`, then: "duck" },
    { when: "always", then: "run" },
  ];
  const matched = [
    has && !s.high && s.gap >= g.lo && s.gap <= g.hi && !s.airborne,
    has && s.high && g.duck > 0 && s.gap <= g.duck,
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
  t.innerHTML =
    "<tr><th>candidate</th><th class='n'>runs</th><th class='n'>mean</th><th class='n'>best</th></tr>";
  const evolved = pop.filter((c) => !c.ctrl);
  const top = Math.max(0, ...evolved.map((c) => c.mean));
  pop.forEach((c) => {
    const tr = document.createElement("tr");
    if (c.ctrl) tr.className = "ctrl";
    else if (c.mean && c.mean === top) tr.className = "best";
    tr.innerHTML =
      `<td>${candName(c)}${c.elite ? " ★" : ""}</td><td class="n">${c.runs}</td>` +
      `<td class="n">${c.mean || "—"}</td><td class="n">${c.best || "—"}</td>`;
    t.appendChild(tr);
  });
  $("evoinfo").textContent =
    `session ${S.sessions} · gen ${S.generation} · ep ${S.episodes} · best ever ${S.bestEver}`;
}

function drawChart() {
  const S = currentState();
  const c = $("chart");
  if (!c) return;
  const ctx = c.getContext("2d");
  ctx.clearRect(0, 0, c.width, c.height);
  const h = S.history;
  if (!h.length) {
    ctx.fillStyle = "#7c8a9a";
    ctx.font = "10px monospace";
    ctx.fillText("no generations yet", 8, 16);
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
  const scale = cr.width / r.canvas.width;
  const top = cr.top - wr.top;
  // takeoff window of the ACTIVE genome
  if (g && g.lo >= 0 && g.hi < 99998) {
    const x0 = cr.left - wr.left + (r.tRex.xPos + 44 + g.lo) * scale;
    ctx.fillStyle = "rgba(61,220,132,.18)";
    ctx.fillRect(x0, top, (g.hi - g.lo) * scale, cr.height);
  }
  if (s.gap < 99999) {
    const x = cr.left - wr.left + (r.tRex.xPos + 44 + s.gap) * scale;
    ctx.strokeStyle = "rgba(78,161,255,.95)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(x, top);
    ctx.lineTo(x, top + cr.height);
    ctx.stroke();
  }
}

/** One frame: decide, act, advance the real game by one frame. */
function frame() {
  if (!running) return;
  const cand = pop[idx];
  const s = read();
  if (!s) return;

  if (s.crashed) {
    if (!sawCrash) {
      sawCrash = true;
      const dist = Math.max(0, s.distance - epStart);
      recordEpisode(cand, dist, pop);
      epInCand++;
      log(`${candName(cand)} -> ${dist}`);
      renderTable();

      if (epInCand >= EPISODES_PER) {
        epInCand = 0;
        idx++;
        if (idx >= pop.length) {
          // generation complete: rank, keep elites, breed the next
          const { best, bestCtrl } = closeGeneration(pop);
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
    restartEpisode();
    return;
  }

  if (!s.started) {
    act("jump");
    CLOCK.step();
    return;
  }

  const d = decide(s, cand.g);
  act(d.action);
  CLOCK.step();

  $("act").textContent = d.action;
  $("dist").textContent = Math.max(0, s.distance - epStart);
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
  const pump = () => {
    if (!running) return;
    for (let i = 0; i < 24; i++) frame();
  };
  const rafPump = () => {
    if (!running) return;
    pump();
    rafId = window.requestAnimationFrame(rafPump);
  };
  loop = setInterval(() => {
    pump();
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
  clearInterval(loop);
  if (rafId) window.cancelAnimationFrame(rafId);
  $("run").disabled = false;
  $("stop").disabled = true;
  $("status").textContent = "stopped";
  $("status").className = "tag";
};

const resetBtn = $("reset");
if (resetBtn) {
  resetBtn.onclick = () => {
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
