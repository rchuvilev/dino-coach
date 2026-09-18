/**
 * Drives the REAL Chrome Dino game, hosted locally with a manual clock.
 *
 * Why this is the right approach, after three failed ones:
 *  - iframe: cross-origin. The page renders but Runner.instance_ is
 *    unreachable ("Blocked a frame with origin minis:// ..."). Verified.
 *  - driving chromedino.com from outside: works, but each execute_js call is
 *    capped at 30s, which is far too little for many episodes.
 *  - overwriting Runner.time from outside: CORRUPTED the sim - obstacles
 *    drifted AWAY from the player, distanceRan read 650845248688.
 *
 * Here we serve the game's OWN game.js (73KB, defining Runner) and its own
 * sprites, with two surgical patches: getTimeStamp() and the single
 * requestAnimationFrame call both consult window.__CLOCK. Everything the
 * game computes - deltaTime, distanceRan, obstacle x, the jump arc - flows
 * from getTimeStamp(), so owning it makes the real game deterministic and
 * steppable WITHOUT touching game logic.
 */

// clock lives in clock.js, loaded BEFORE game.js (see index.html)
const CLOCK = window.__CLOCK;

const RULES = [
  { when: "obstacle_imminent", then: "jump" },
  { when: "high_obstacle", then: "duck" },
  { when: "obstacle_near", then: "jump" },
  { when: "always", then: "run" },
];

// WINDOWS, not ceilings. A `gap <= N` ceiling fires continuously from N down
// to 0 and takes off far too early; on the synthetic target that single
// mistake made a ground-truth ORACLE score BELOW the do-nothing control.
// Windows sized from MEASURED geometry: obstacles close at ~6px per frame
// (366 distance per 60 frames at speed 6.14, ratio 0.99 vs expected), and the
// jump rise takes ~10 frames, so takeoff must happen ~60-100px out. A window
// of 35px only gives ~6 usable frames and is easily missed.
const POLICIES = [
  { label: "window 40..100", lo: 40, hi: 100, duck: 40 },
  { label: "window 60..130", lo: 60, hi: 130, duck: 40 },
  { label: "window 80..160", lo: 80, hi: 160, duck: 40 },
  { label: "human ceiling <=60", lo: 0, hi: 60, duck: 40 },
  { label: "control: do-nothing", lo: -1, hi: -1, duck: -1, ctrl: true },
  { label: "control: always-jump", lo: 0, hi: 99998, duck: -1, ctrl: true },
];

const $ = (id) => document.getElementById(id);
let running = false;
let policyIdx = 0;
let episode = 0;
let best = 0;
let frames = 0;
let sawCrash = false;
let epStart = 0;   // distanceRan at the start of the current episode
const results = POLICIES.map(() => ({ runs: 0, total: 0, best: 0 }));
let logLines = [];

const log = (s) => {
  logLines.unshift(s);
  logLines = logLines.slice(0, 5);
  $("log").textContent = logLines.join("\n");
};

const R = () => (window.Runner && window.Runner.instance_ ? window.Runner.instance_ : null);

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

/** Act through the game's OWN key handlers, so input follows its real path. */
function act(a) {
  const r = R();
  if (!r) return;
  if (a === "jump") {
    r.onKeyDown({ keyCode: 38, type: "keydown", preventDefault() {}, target: {} });
    r.onKeyUp({ keyCode: 38, type: "keyup", preventDefault() {}, target: {} });
  } else if (a === "duck") {
    r.onKeyDown({ keyCode: 40, type: "keydown", preventDefault() {}, target: {} });
  } else {
    r.onKeyUp({ keyCode: 40, type: "keyup", preventDefault() {}, target: {} });
  }
}

const condsOf = (s, p) => {
  const has = s.gap < 99999;
  return {
    obstacle_imminent: has && !s.high && s.gap >= p.lo && s.gap <= p.hi && !s.airborne,
    high_obstacle: has && s.high && p.duck > 0 && s.gap <= p.duck,
    obstacle_near: has && s.gap <= p.hi + 20,
    always: true,
  };
};

const decide = (c) => {
  for (let i = 0; i < RULES.length; i++) if (c[RULES[i].when]) return i;
  return null;
};

function renderRules(fired, conds) {
  const ul = $("rules");
  ul.innerHTML = "";
  RULES.forEach((r, i) => {
    const li = document.createElement("li");
    const m = !!(conds && conds[r.when]);
    li.className = i === fired ? "fire" : m ? "match" : "";
    li.innerHTML =
      `<span>${r.when}</span><span>=&gt;</span><span class="then">${r.then}</span>` +
      `<span class="why">${i === fired ? "FIRING" : m ? "suppressed" : ""}</span>`;
    ul.appendChild(li);
  });
}

function renderTable() {
  const t = $("tbl");
  t.innerHTML = "<tr><th>policy</th><th class='n'>runs</th><th class='n'>mean</th><th class='n'>best</th></tr>";
  const means = results.map((r) => (r.runs ? r.total / r.runs : -1));
  const top = Math.max(...means);
  POLICIES.forEach((p, i) => {
    const r = results[i];
    const mean = r.runs ? Math.round(r.total / r.runs) : 0;
    const tr = document.createElement("tr");
    if (p.ctrl) tr.className = "ctrl";
    else if (r.runs && mean === Math.round(top)) tr.className = "best";
    tr.innerHTML =
      `<td>${p.label}</td><td class="n">${r.runs}</td>` +
      `<td class="n">${mean || "—"}</td><td class="n">${r.best || "—"}</td>`;
    t.appendChild(tr);
  });
}

function overlay(s) {
  const r = R();
  const c = $("ov");
  const wrap = $("wrap");
  if (c.width !== wrap.clientWidth || c.height !== wrap.clientHeight) {
    c.width = wrap.clientWidth;
    c.height = wrap.clientHeight;
  }
  const ctx = c.getContext("2d");
  ctx.clearRect(0, 0, c.width, c.height);
  if (!r || !r.canvas || !s || s.gap >= 99999) return;
  const cr = r.canvas.getBoundingClientRect();
  const wr = wrap.getBoundingClientRect();
  const scale = cr.width / r.canvas.width;
  const x = cr.left - wr.left + (r.tRex.xPos + 44 + s.gap) * scale;
  ctx.strokeStyle = "rgba(78,161,255,.95)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(x, cr.top - wr.top);
  ctx.lineTo(x, cr.top - wr.top + cr.height);
  ctx.stroke();
  // takeoff window of the active policy
  const p = POLICIES[policyIdx];
  if (p.lo >= 0 && p.hi < 99998) {
    const x0 = cr.left - wr.left + (r.tRex.xPos + 44 + p.lo) * scale;
    ctx.fillStyle = "rgba(61,220,132,.18)";
    ctx.fillRect(x0, cr.top - wr.top, (p.hi - p.lo) * scale, cr.height);
  }
}

/** One harness tick: read -> decide -> act -> advance the clock. */
function tick() {
  if (!running) return;
  const p = POLICIES[policyIdx];
  const s = read();

  if (!s) {
    $("status").textContent = "no runner";
    $("status").className = "tag dead";
    return;
  }

  if (s.crashed) {
    if (!sawCrash) {
      sawCrash = true;
      // DELTA, not absolute: restart() does not zero distanceRan in this
      // build (measured 9196 immediately after a restart), so absolute
      // readings carried over and made a do-nothing control "score" 14346
      // against ~1400 for real policies - a control beating everything 10x,
      // which is the signature of a broken metric, not a good control.
      const dist = Math.max(0, s.distance - epStart);
      const res = results[policyIdx];
      res.runs++;
      res.total += dist;
      if (dist > res.best) res.best = dist;
      if (dist > best) best = dist;
      episode++;
      log(`ep${episode} ${p.label} -> ${dist}`);
      renderTable();
      policyIdx = (policyIdx + 1) % POLICIES.length;
      $("policy").selectedIndex = policyIdx;
      $("pname").textContent = POLICIES[policyIdx].label;
      // Restart and VERIFY the odometer actually zeroed before resuming.
      // Measured bug: scores were carrying over between episodes, so a
      // do-nothing control "scored" 14338 against ~1390 for real policies -
      // a control beating every policy 10x, which is the signature of a
      // broken metric rather than a good control.
      const r = R();
      if (r) r.restart();
      for (let i = 0; i < 6; i++) CLOCK.step();
      const chk = read();
      epStart = chk ? chk.distance : 0;   // baseline for the NEXT episode
      sawCrash = false;
    }
  } else if (!s.started) {
    act("jump");
  } else {
    const conds = condsOf(s, p);
    const fired = decide(conds);
    const action = fired === null ? "run" : RULES[fired].then;
    act(action);
    renderRules(fired, conds);
    $("act").textContent = action;
  }

  // advance the REAL game by one frame using our clock
  CLOCK.step();
  frames++;

  $("status").textContent = "running";
  $("status").className = "tag live";
  $("dist").textContent = Math.max(0, s.distance - epStart);
  $("best").textContent = best;
  $("ep").textContent = episode;
  $("frames").textContent = frames;
  $("v-gap").textContent = s.gap < 99999 ? Math.round(s.gap) : "—";
  $("v-spd").textContent = s.speed.toFixed(1);
  $("m-gap").style.width = s.gap < 99999 ? `${Math.max(0, 100 - s.gap / 4)}%` : "0%";
  $("m-spd").style.width = `${Math.min(100, (s.speed / 13) * 100)}%`;
  $("f-high").className = "flag" + (s.high ? " on" : "");
  $("f-air").className = "flag" + (s.airborne ? " on" : "");
  overlay(s);
}

/**
 * Liveness gate. Refuses to score until the real game is proven BOTH to
 * advance under our clock AND to move obstacles toward the player.
 * The second half is the one that matters: the earlier fake-clock attempt
 * advanced distanceRan while obstacles drifted away.
 */
function verifyLiveness() {
  const r = R();
  if (!r) return { ok: false, why: "Runner not constructed" };
  if (r.crashed) r.restart();
  for (let i = 0; i < 5; i++) CLOCK.step();
  act("jump");
  const d0 = read();
  for (let i = 0; i < 40; i++) CLOCK.step();
  const d1 = read();
  if (!d0 || !d1) return { ok: false, why: "cannot read state" };
  if (d1.distance <= d0.distance) {
    return { ok: false, why: `distance not advancing (${d0.distance} -> ${d1.distance})` };
  }
  // obstacle must approach, not recede
  let approached = null;
  const first = (r.horizon.obstacles || [])[0];
  if (first) {
    const x0 = first.xPos;
    for (let i = 0; i < 20; i++) CLOCK.step();
    const same = (r.horizon.obstacles || []).find((o) => o === first);
    if (same) approached = same.xPos < x0;
  }
  if (approached === false) {
    return { ok: false, why: "obstacles moving AWAY - clock corrupts the sim" };
  }
  // leave a clean slate: otherwise episode 1 inherits the probe's distance
  r.restart();
  for (let i = 0; i < 6; i++) CLOCK.step();
  return {
    ok: true,
    why: `advancing ${d0.distance}->${d1.distance}${approached ? ", obstacles approach" : ""}`,
  };
}

// ---- boot ---------------------------------------------------------------
$("policy").innerHTML = POLICIES.map((p, i) => `<option value="${i}">${p.label}</option>`).join("");
$("policy").onchange = (e) => {
  policyIdx = Number(e.target.value);
  $("pname").textContent = POLICIES[policyIdx].label;
};
$("pname").textContent = POLICIES[0].label;
renderRules(null, {});
renderTable();

let loop = null;
$("run").onclick = () => {
  const v = verifyLiveness();
  $("liveness").textContent = `liveness: ${v.ok ? "OK" : "FAILED"} · ${v.why}`;
  $("liveness").className = "tag " + (v.ok ? "live" : "dead");
  if (!v.ok) {
    log(`refusing to run: ${v.why}`);
    return;
  }
  running = true;
  const s0 = read();
  epStart = s0 ? s0.distance : 0;
  $("run").disabled = true;
  $("stop").disabled = false;
  // run many harness ticks per animation frame: the game's clock is ours,
  // so this is bounded by CPU, not by wall time
  loop = setInterval(() => {
    for (let i = 0; i < 8; i++) tick();
  }, 16);
};
$("stop").onclick = () => {
  running = false;
  clearInterval(loop);
  $("run").disabled = false;
  $("stop").disabled = true;
  $("status").textContent = "stopped";
  $("status").className = "tag";
};

// the game boots itself on DOMContentLoaded; give it a frame then report
setTimeout(() => {
  const r = R();
  $("status").textContent = r ? "ready" : "no runner";
  $("evoinfo").textContent = `${POLICIES.length} policies, rotating per episode`;
  log(r ? "real game.js loaded, Runner constructed" : "Runner missing");
  if (r) {
    for (let i = 0; i < 3; i++) CLOCK.step();
    overlay(read());
  }
}, 300);
