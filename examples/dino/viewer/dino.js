/**
 * Live viewer driving the REAL Chrome Dino.
 *
 * Why this works where the earlier attempts did not: the loop runs INSIDE the
 * page on its own timer, so it is not bound by the 30s execute_js ceiling
 * that made a browser-hosted evolution run impossible from the outside. The
 * page stays focused while it is open, so rAF keeps ticking.
 *
 * Corrected understanding from the retraction (the user was right):
 *  - synthetic KeyboardEvents DO work; earlier probes were run on a CRASHED
 *    game, which stops its own rAF loop, so nothing responded to anything.
 *  - the tab IS visible while a call runs; a crashed game faked the freeze.
 * So the load-bearing check is: is the game running RIGHT NOW? Everything
 * else was a symptom of not asking that.
 */

const RULES = [
  { when: "obstacle_imminent", then: "jump" },
  { when: "high_obstacle", then: "duck" },
  { when: "obstacle_near", then: "jump" },
  { when: "always", then: "run" },
];

// Measured on the real dino: jump arc 93 -> 7 -> 93 over ~550ms, and
// obstacles close at ~6px per frame at starting speed. A CEILING condition
// (gap <= N) takes off far too early - that mistake cost three runs on the
// synthetic target - so these are WINDOWS.
const POLICIES = [
  { label: "window 30..70", lo: 30, hi: 70, duckAt: 40 },
  { label: "window 40..90", lo: 40, hi: 90, duckAt: 40 },
  { label: "window 50..110", lo: 50, hi: 110, duckAt: 40 },
  { label: "human ceiling <=60", lo: 0, hi: 60, duckAt: 40 },
  { label: "control: do-nothing", lo: -1, hi: -1, duckAt: -1 },
  { label: "control: always-jump", lo: 0, hi: 9998, duckAt: -1 },
];

const $ = (id) => document.getElementById(id);
const frame = $("game");
let win = null;
let running = false;
let raf = null;
let episode = 0;
let best = 0;
const results = POLICIES.map(() => ({ runs: 0, total: 0, best: 0 }));
let policyIdx = 0;
let logLines = [];

function log(s) {
  logLines.unshift(s);
  logLines = logLines.slice(0, 6);
  $("log").textContent = logLines.join("\n");
}

/** The game object inside the iframe, or null if not reachable/ready. */
function runner() {
  try {
    return win && win.Runner && win.Runner.instance_ ? win.Runner.instance_ : null;
  } catch {
    return null; // cross-origin
  }
}

function read() {
  const r = runner();
  if (!r) return null;
  const tx = r.tRex.xPos + 44;
  let gap = 9999;
  let high = false;
  const obs = (r.horizon && r.horizon.obstacles) || [];
  for (const o of obs) {
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
    y: r.tRex.yPos,
    airborne: r.tRex.yPos < 88,
    gap,
    high,
  };
}

function key(code, kc, type) {
  const d = win.document;
  d.dispatchEvent(new win.KeyboardEvent(type, { code, keyCode: kc, which: kc, bubbles: true }));
}

function act(a) {
  if (a === "jump") {
    key("Space", 32, "keydown");
    setTimeout(() => key("Space", 32, "keyup"), 40);
  } else if (a === "duck") {
    key("ArrowDown", 40, "keydown");
    setTimeout(() => key("ArrowDown", 40, "keyup"), 90);
  }
}

function conditionsOf(s, p) {
  const has = s.gap < 9999;
  return {
    obstacle_imminent: has && !s.high && s.gap >= p.lo && s.gap <= p.hi && !s.airborne,
    high_obstacle: has && s.high && p.duckAt > 0 && s.gap <= p.duckAt,
    obstacle_near: has && s.gap <= p.hi + 20,
    always: true,
  };
}

/** First-match-wins over the rule order - same semantics as RuleEvaluator. */
function decide(conds) {
  for (let i = 0; i < RULES.length; i++) {
    if (conds[RULES[i].when]) return i;
  }
  return null;
}

function renderRules(fired, conds) {
  const ul = $("rules");
  ul.innerHTML = "";
  RULES.forEach((r, i) => {
    const li = document.createElement("li");
    const m = !!conds[r.when];
    li.className = i === fired ? "fire" : m ? "match" : "";
    li.innerHTML = `<span>${r.when}</span><span>=&gt;</span><span class="then">${r.then}</span>` +
      `<span class="why">${i === fired ? "FIRING" : m ? "suppressed" : ""}</span>`;
    ul.appendChild(li);
  });
}

function renderTable() {
  const t = $("tbl");
  t.innerHTML = "<tr><th>policy</th><th class='n'>runs</th><th class='n'>mean</th><th class='n'>best</th></tr>";
  const means = results.map((r) => (r.runs ? r.total / r.runs : 0));
  const top = Math.max(...means);
  POLICIES.forEach((p, i) => {
    const r = results[i];
    const mean = r.runs ? Math.round(r.total / r.runs) : 0;
    const tr = document.createElement("tr");
    if (r.runs && mean === Math.round(top)) tr.className = "best";
    tr.innerHTML = `<td>${p.label}</td><td class="n">${r.runs}</td><td class="n">${mean || "—"}</td><td class="n">${r.best || "—"}</td>`;
    t.appendChild(tr);
  });
}

/** Overlay the gap marker the reader is locked onto, over the live iframe. */
function overlay(s) {
  const c = $("ov");
  const host = $("host");
  if (c.width !== host.clientWidth || c.height !== host.clientHeight) {
    c.width = host.clientWidth;
    c.height = host.clientHeight;
  }
  const ctx = c.getContext("2d");
  ctx.clearRect(0, 0, c.width, c.height);
  if (!s || s.gap >= 9999) return;
  const r = runner();
  if (!r || !r.canvas) return;
  const cr = r.canvas.getBoundingClientRect();
  const fr = frame.getBoundingClientRect();
  const scale = cr.width / (r.canvas.width || 600);
  const x = (cr.left - fr.left) + (r.tRex.xPos + 44 + s.gap) * scale;
  const top = cr.top - fr.top;
  ctx.strokeStyle = "rgba(78,161,255,.95)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(x, top);
  ctx.lineTo(x, top + cr.height);
  ctx.stroke();
}

let sawCrash = false;

function tick() {
  if (!running) return;
  const p = POLICIES[policyIdx];
  const s = read();

  if (!s) {
    $("status").textContent = "unreachable";
    $("status").className = "tag dead";
    raf = requestAnimationFrame(tick);
    return;
  }

  // episode boundary: record and restart
  if (s.crashed) {
    if (!sawCrash) {
      sawCrash = true;
      const r = results[policyIdx];
      r.runs++;
      r.total += s.distance;
      if (s.distance > r.best) r.best = s.distance;
      if (s.distance > best) best = s.distance;
      episode++;
      log(`ep${episode} ${p.label} -> ${s.distance}`);
      renderTable();
      // rotate policies so every one accumulates episodes
      policyIdx = (policyIdx + 1) % POLICIES.length;
      $("policy").selectedIndex = policyIdx;
      $("pname").textContent = POLICIES[policyIdx].label;
      setTimeout(() => {
        const rr = runner();
        if (rr) rr.restart();
        sawCrash = false;
      }, 350);
    }
    $("status").textContent = "crashed";
    $("status").className = "tag dead";
    $("ep").textContent = episode;
    $("best").textContent = best;
    raf = requestAnimationFrame(tick);
    return;
  }

  if (!s.started) {
    act("jump"); // a jump starts the game
  } else {
    const conds = conditionsOf(s, p);
    const fired = decide(conds);
    const action = fired === null ? "run" : RULES[fired].then;
    act(action);
    renderRules(fired, conds);
    $("act").textContent = action;
  }

  $("status").textContent = "running";
  $("status").className = "tag live";
  $("dist").textContent = s.distance;
  $("best").textContent = best;
  $("ep").textContent = episode;
  $("v-gap").textContent = s.gap < 9999 ? Math.round(s.gap) : "—";
  $("v-spd").textContent = s.speed.toFixed(1);
  $("m-gap").style.width = s.gap < 9999 ? `${Math.max(0, 100 - s.gap / 3)}%` : "0%";
  $("m-spd").style.width = `${Math.min(100, (s.speed / 13) * 100)}%`;
  $("f-high").className = "flag" + (s.high ? " on" : "");
  $("f-air").className = "flag" + (s.airborne ? " on" : "");
  overlay(s);

  raf = requestAnimationFrame(tick);
}

/**
 * Liveness gate - the guard distilled from the dino failures.
 * Refuses to score until the target is proven to advance AND respond.
 */
async function verifyLive() {
  const r = runner();
  if (!r) return { ok: false, why: "game object unreachable (cross-origin?)" };
  if (r.crashed) r.restart();
  act("jump");
  await new Promise((s) => setTimeout(s, 600));
  const a = read();
  await new Promise((s) => setTimeout(s, 500));
  const b = read();
  if (!a || !b) return { ok: false, why: "cannot read state" };
  if (b.distance <= a.distance) {
    return { ok: false, why: `not advancing (${a.distance} -> ${b.distance})` };
  }
  return { ok: true, why: `advancing (${a.distance} -> ${b.distance})` };
}

$("policy").innerHTML = POLICIES.map((p, i) => `<option value="${i}">${p.label}</option>`).join("");
$("policy").onchange = (e) => {
  policyIdx = Number(e.target.value);
  $("pname").textContent = POLICIES[policyIdx].label;
};
$("pname").textContent = POLICIES[0].label;
renderTable();

$("run").onclick = async () => {
  const v = await verifyLive();
  $("liveness").textContent = `liveness: ${v.ok ? "OK" : "FAILED"} · ${v.why}`;
  $("liveness").className = "tag " + (v.ok ? "live" : "dead");
  if (!v.ok) {
    log(`refusing to run: ${v.why}`);
    return;
  }
  running = true;
  $("run").disabled = true;
  $("stop").disabled = false;
  tick();
};
$("stop").onclick = () => {
  running = false;
  cancelAnimationFrame(raf);
  $("run").disabled = false;
  $("stop").disabled = true;
  $("status").textContent = "stopped";
  $("status").className = "tag";
};

frame.addEventListener("load", () => {
  try {
    win = frame.contentWindow;
    // touch a property to force a cross-origin throw early
    void win.location.href;
    $("status").textContent = "ready";
    log("iframe loaded, same-origin access OK");
  } catch (e) {
    win = null;
    $("status").textContent = "cross-origin";
    $("status").className = "tag dead";
    log("CROSS-ORIGIN: cannot reach the game inside the iframe");
  }
  $("evoinfo").textContent = `${POLICIES.length} policies, rotating per episode`;
});
