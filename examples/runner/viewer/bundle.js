// examples/runner/world.ts
function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = a + 1831565813 >>> 0;
    let t = Math.imul(a ^ a >>> 15, 1 | a);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}
var W = 200;
var H = 60;
var GROUND = 46;
var PLAYER_X = 24;
var PLAYER_W = 8;

class RunnerWorld {
  seed;
  y = 0;
  vy = 0;
  ducking = false;
  duckLeft = 0;
  obstacles = [];
  distance = 0;
  crashed = false;
  ticks = 0;
  speed = 2;
  rand;
  nextSpawn;
  constructor(seed = 1) {
    this.seed = seed;
    this.rand = rng(seed);
    this.nextSpawn = 20 + Math.floor(this.rand() * 24);
  }
  reset(seed = this.seed) {
    this.seed = seed;
    this.rand = rng(seed);
    this.y = 0;
    this.vy = 0;
    this.ducking = false;
    this.duckLeft = 0;
    this.obstacles = [];
    this.distance = 0;
    this.crashed = false;
    this.ticks = 0;
    this.speed = 2;
    this.nextSpawn = 20 + Math.floor(this.rand() * 24);
  }
  box() {
    const h = this.ducking ? 6 : 14;
    return { x: PLAYER_X, y: GROUND - h - this.y, w: PLAYER_W, h };
  }
  step(action = "run") {
    if (this.crashed)
      return false;
    this.ticks++;
    if (action === "jump" && this.y === 0)
      this.vy = 5.2;
    if (action === "duck") {
      this.ducking = true;
      this.duckLeft = 6;
    }
    if (this.duckLeft > 0) {
      this.duckLeft--;
      if (this.duckLeft === 0)
        this.ducking = false;
    }
    this.y += this.vy;
    this.vy -= 0.75;
    if (this.y <= 0) {
      this.y = 0;
      this.vy = 0;
    }
    this.speed = 2 + Math.min(2, this.ticks / 900);
    for (const o of this.obstacles)
      o.x -= this.speed;
    this.obstacles = this.obstacles.filter((o) => o.x + o.w > -4);
    if (--this.nextSpawn <= 0) {
      const high = this.rand() < 0.3;
      this.obstacles.push(high ? { x: W + 4, high: true, w: 10, h: 6 } : { x: W + 4, high: false, w: 6, h: 10 + Math.floor(this.rand() * 6) });
      this.nextSpawn = Math.round(46 + this.rand() * 40 + this.speed * 6);
    }
    const p = this.box();
    for (const o of this.obstacles) {
      const oy = o.high ? GROUND - 22 : GROUND - o.h;
      if (p.x < o.x + o.w && p.x + p.w > o.x && p.y < oy + o.h && p.y + p.h > oy) {
        this.crashed = true;
        return false;
      }
    }
    this.distance++;
    return true;
  }
  nearest() {
    let best = null;
    for (const o of this.obstacles) {
      const gap = o.x - (PLAYER_X + PLAYER_W);
      if (gap > -PLAYER_W && (best === null || o.x < best.x))
        best = o;
    }
    return best;
  }
  truth() {
    const o = this.nearest();
    return {
      gap: o ? Math.round(o.x - (PLAYER_X + PLAYER_W)) : 999,
      high: o ? o.high : false,
      y: Math.round(this.y)
    };
  }
}
function render(w) {
  const data = new Uint8Array(W * H * 4);
  for (let i = 0;i < W * H; i++) {
    data[i * 4] = 255;
    data[i * 4 + 1] = 255;
    data[i * 4 + 2] = 255;
    data[i * 4 + 3] = 255;
  }
  const put = (x, y) => {
    if (x < 0 || y < 0 || x >= W || y >= H)
      return;
    const i = (y * W + x) * 4;
    data[i] = 20;
    data[i + 1] = 20;
    data[i + 2] = 20;
  };
  const rect = (x, y, rw, rh) => {
    for (let dy = 0;dy < rh; dy++)
      for (let dx = 0;dx < rw; dx++)
        put(Math.round(x + dx), Math.round(y + dy));
  };
  rect(0, GROUND, W, 1);
  const ph = w.ducking ? 6 : 14;
  rect(PLAYER_X, GROUND - ph - w.y, PLAYER_W, ph);
  for (const o of w.obstacles) {
    const oy = o.high ? GROUND - 22 : GROUND - o.h;
    rect(o.x, oy, o.w, o.h);
  }
  return { width: W, height: H, data };
}

// examples/runner/perceive.ts
var DARK = 128;
function isDark(f, x, y) {
  if (x < 0 || y < 0 || x >= f.width || y >= f.height)
    return false;
  return (f.data[(y * f.width + x) * 4] ?? 255) < DARK;
}
function readGap(f) {
  const from = PLAYER_X + PLAYER_W + 1;
  for (let x = from;x < W; x++) {
    for (let y = GROUND - 24;y < GROUND; y++) {
      if (isDark(f, x, y))
        return x - from;
    }
  }
  return 999;
}
function readHigh(f) {
  const from = PLAYER_X + PLAYER_W + 1;
  for (let x = from;x < W; x++) {
    let top = -1;
    let bottom = -1;
    for (let y = GROUND - 24;y < GROUND; y++) {
      if (isDark(f, x, y)) {
        if (top === -1)
          top = y;
        bottom = y;
      }
    }
    if (top !== -1) {
      return bottom < GROUND - 3;
    }
  }
  return false;
}
function readPlayerY(f) {
  const x = PLAYER_X + Math.floor(PLAYER_W / 2);
  for (let y = 0;y < GROUND; y++) {
    if (isDark(f, x, y)) {
      return Math.max(0, GROUND - 14 - y);
    }
  }
  return 0;
}
function perceive(f) {
  const playerY = readPlayerY(f);
  return {
    gap: readGap(f),
    high: readHigh(f),
    playerY,
    airborne: playerY > 1
  };
}

// src/rules/evaluator.ts
class RuleEvaluator {
  ruleSet;
  commitLeft = 0;
  committedRule = null;
  cooldownUntil = new Map;
  tick = 0;
  constructor(ruleSet) {
    this.ruleSet = ruleSet;
  }
  setRuleSet(rs) {
    this.ruleSet = rs;
    this.commitLeft = 0;
    this.committedRule = null;
  }
  getRuleSet() {
    return this.ruleSet;
  }
  step(conditions, trace = true) {
    const t = this.tick++;
    const { rules, order } = this.ruleSet;
    if (this.commitLeft > 0 && this.committedRule !== null) {
      const r = rules[this.committedRule];
      if (r && conditions[r.when]) {
        this.commitLeft--;
        return {
          action: r.then,
          firedRule: this.committedRule,
          committed: true,
          suppressed: [],
          cooling: []
        };
      }
      this.commitLeft = 0;
      this.committedRule = null;
    }
    const suppressed = [];
    const cooling = [];
    let chosen = null;
    for (const idx of order) {
      const r = rules[idx];
      if (!r)
        continue;
      if (!conditions[r.when])
        continue;
      const until = this.cooldownUntil.get(idx) ?? 0;
      if (t < until) {
        if (trace)
          cooling.push(idx);
        continue;
      }
      if (chosen === null) {
        chosen = idx;
        if (!trace)
          break;
      } else if (trace) {
        suppressed.push(idx);
      }
    }
    if (chosen === null) {
      return { action: null, firedRule: null, committed: false, suppressed, cooling };
    }
    const rule = rules[chosen];
    const commit = rule.commitTicks ?? 0;
    if (commit > 1) {
      this.commitLeft = commit - 1;
      this.committedRule = chosen;
    }
    if (rule.cooldownTicks) {
      this.cooldownUntil.set(chosen, t + commit + rule.cooldownTicks);
    }
    return { action: rule.then, firedRule: chosen, committed: false, suppressed, cooling };
  }
  reset() {
    this.commitLeft = 0;
    this.committedRule = null;
    this.cooldownUntil.clear();
    this.tick = 0;
  }
}

// src/evolve/genome.ts
function applyGenome(rules, g) {
  return {
    rules: rules.map((r, i) => ({ ...r, then: g.actions[i] ?? r.then })),
    order: [...g.order]
  };
}

// examples/runner/viewer/app.js
var RULES = [
  { when: "obstacle_imminent", then: "jump" },
  { when: "high_obstacle", then: "duck" },
  { when: "obstacle_near", then: "jump" },
  { when: "always", then: "run" }
];
var TH = { windowLo: 4, windowHi: 12, duckAt: 12 };
var conditionsFrom = (p) => {
  const has = p.gap < 999;
  return {
    obstacle_imminent: has && !p.high && p.gap >= TH.windowLo && p.gap <= TH.windowHi && !p.airborne,
    high_obstacle: has && p.high && p.gap <= TH.duckAt,
    obstacle_near: has && p.gap <= TH.windowHi + 10,
    always: true
  };
};
var $ = (id) => document.getElementById(id);
var screen = $("screen");
var sctx = screen.getContext("2d");
var chart = $("chart");
var cctx = chart.getContext("2d");
var state = null;
var policies = [];
var world = new RunnerWorld(101);
var evaluator = null;
var running = false;
var timer = null;
var seedIdx = 0;
var HOLDOUT = [101, 211, 307, 401, 503, 601, 701, 809, 907, 1009];
async function loadState() {
  if (document.readyState === "loading") {
    await new Promise((r) => document.addEventListener("DOMContentLoaded", r, { once: true }));
  }
  if (window.__STATE)
    return window.__STATE;
  try {
    const r = await fetch("state.json", { cache: "no-store" });
    if (!r.ok)
      throw new Error(String(r.status));
    return await r.json();
  } catch {
    return null;
  }
}
function buildPolicies(st) {
  const out = [
    { label: "human seed rules", genome: { order: [0, 1, 2, 3], actions: RULES.map((r) => r.then) }, kind: "seed" },
    { label: "control: do-nothing", genome: { order: [0, 1, 2, 3], actions: ["run", "run", "run", "run"] }, kind: "ctrl" },
    { label: "control: always-jump", genome: { order: [3, 0, 1, 2], actions: ["jump", "jump", "jump", "jump"] }, kind: "ctrl" }
  ];
  if (st?.population?.length) {
    const seen = new Set;
    for (const p of st.population) {
      const k = `${p.genome.order}|${p.genome.actions}`;
      if (seen.has(k))
        continue;
      seen.add(k);
      out.unshift({
        label: `evolved r${p.foundInRun} · holdout ${p.holdout}`,
        genome: p.genome,
        kind: "evo",
        holdout: p.holdout
      });
    }
    out.sort((a, b) => (b.holdout ?? -1) - (a.holdout ?? -1));
  }
  return out;
}
function setPolicy(i) {
  const p = policies[i];
  if (!p)
    return;
  evaluator = new RuleEvaluator(applyGenome(RULES, p.genome));
  $("pname").textContent = p.label;
  renderRules(null, {});
}
function resetWorld() {
  world = new RunnerWorld(HOLDOUT[seedIdx % HOLDOUT.length]);
  const i = $("policy").selectedIndex;
  setPolicy(i < 0 ? 0 : i);
}
function draw(frame, per, decision) {
  const img = sctx.createImageData(W, H);
  img.data.set(frame.data);
  sctx.putImageData(img, 0, 0);
  sctx.save();
  sctx.strokeStyle = "rgba(78,161,255,.9)";
  sctx.lineWidth = 1;
  if (per.gap < 999) {
    const x = PLAYER_X + 8 + per.gap;
    sctx.beginPath();
    sctx.moveTo(x + 0.5, 0);
    sctx.lineTo(x + 0.5, GROUND);
    sctx.stroke();
  }
  sctx.fillStyle = "rgba(61,220,132,.16)";
  sctx.fillRect(PLAYER_X + 8 + TH.windowLo, GROUND - 26, TH.windowHi - TH.windowLo, 26);
  if (world.crashed) {
    sctx.fillStyle = "rgba(255,93,93,.22)";
    sctx.fillRect(0, 0, W, H);
  }
  sctx.restore();
}
function renderRules(fired, conds) {
  const ul = $("rules");
  const pol = policies[Math.max(0, $("policy").selectedIndex)];
  const g = pol?.genome ?? { order: [0, 1, 2, 3], actions: RULES.map((r) => r.then) };
  ul.innerHTML = "";
  for (const idx of g.order) {
    const r = RULES[idx];
    const act = g.actions[idx] ?? r.then;
    const li = document.createElement("li");
    const matched = !!conds[r.when];
    if (idx === fired)
      li.className = "fire";
    else if (matched)
      li.className = "match";
    li.innerHTML = `<span>${r.when}</span><span class="arrow">=&gt;</span><span class="then">${act}</span>` + `<span class="why">${idx === fired ? "FIRING" : matched ? "suppressed" : ""}</span>`;
    ul.appendChild(li);
  }
}
function renderConds(conds) {
  const ul = $("conds");
  ul.innerHTML = "";
  for (const [k, v] of Object.entries(conds)) {
    const li = document.createElement("li");
    if (v)
      li.className = "on";
    li.innerHTML = `<span>${k}</span><span>${v ? "true" : "false"}</span>`;
    ul.appendChild(li);
  }
}
function tick() {
  if (world.crashed) {
    $("status").textContent = "crashed";
    $("status").className = "tag dead";
    seedIdx++;
    resetWorld();
    return;
  }
  const frame = render(world);
  const per = perceive(frame);
  const conds = conditionsFrom(per);
  const d = evaluator.step(conds);
  const action = d.action ?? "run";
  draw(frame, per, d);
  renderRules(d.firedRule, conds);
  renderConds(conds);
  $("tick").textContent = world.ticks;
  $("dist").textContent = world.distance;
  $("act").textContent = action;
  $("v-gap").textContent = per.gap < 999 ? per.gap : "—";
  $("v-y").textContent = per.playerY;
  $("m-gap").style.width = per.gap < 999 ? `${Math.max(0, 100 - per.gap)}%` : "0%";
  $("m-y").style.width = `${Math.min(100, per.playerY / 22 * 100)}%`;
  $("f-high").className = "flag" + (per.high ? " on" : "");
  $("f-air").className = "flag" + (per.airborne ? " on" : "");
  $("status").textContent = "running";
  $("status").className = "tag live";
  world.step(action);
}
function drawChart(st) {
  const h = st?.history ?? [];
  cctx.clearRect(0, 0, chart.width, chart.height);
  cctx.strokeStyle = "#232b35";
  cctx.beginPath();
  cctx.moveTo(0, chart.height - 20.5);
  cctx.lineTo(chart.width, chart.height - 20.5);
  cctx.stroke();
  if (!h.length) {
    cctx.fillStyle = "#7c8a9a";
    cctx.font = "11px monospace";
    cctx.fillText("no runs yet - run self-evolve.ts", 10, 24);
    return;
  }
  const max = Math.max(...h.map((x) => x.bestHoldout), 1);
  const pad = 26;
  const step = h.length > 1 ? (chart.width - pad * 2) / (h.length - 1) : 0;
  const y = (v) => chart.height - 20 - v / max * (chart.height - 34);
  cctx.strokeStyle = "#3ddc84";
  cctx.lineWidth = 2;
  cctx.beginPath();
  h.forEach((p, i) => i ? cctx.lineTo(pad + i * step, y(p.bestHoldout)) : cctx.moveTo(pad, y(p.bestHoldout)));
  cctx.stroke();
  cctx.fillStyle = "#3ddc84";
  h.forEach((p, i) => cctx.fillRect(pad + i * step - 2, y(p.bestHoldout) - 2, 4, 4));
  cctx.fillStyle = "#7c8a9a";
  cctx.font = "10px monospace";
  cctx.fillText(`best held-out ${max}`, pad, 12);
  cctx.fillText(`run 1`, pad - 8, chart.height - 6);
  cctx.fillText(`run ${h.length}`, chart.width - pad - 14, chart.height - 6);
}
function renderPop(st) {
  const t = $("pop");
  const rows = (st?.population ?? []).slice(0, 8);
  t.innerHTML = "<tr><th>run</th><th>rules</th><th class='n'>held-out</th></tr>";
  if (!rows.length) {
    t.innerHTML += "<tr><td colspan='3'>no persisted population</td></tr>";
    return;
  }
  const best = Math.max(...rows.map((r) => r.holdout));
  for (const r of rows) {
    const tr = document.createElement("tr");
    if (r.holdout === best)
      tr.className = "best";
    tr.innerHTML = `<td>${r.foundInRun}</td><td>${r.genome.actions.join(",")}</td>` + `<td class="n">${r.holdout}</td>`;
    t.appendChild(tr);
  }
}
function start() {
  if (running)
    return;
  running = true;
  $("run").disabled = true;
  $("stop").disabled = false;
  const loop = () => {
    if (!running)
      return;
    tick();
    timer = setTimeout(loop, Number($("speed").value));
  };
  loop();
}
function stop() {
  running = false;
  clearTimeout(timer);
  $("run").disabled = false;
  $("stop").disabled = true;
  $("status").textContent = "stopped";
  $("status").className = "tag";
}
$("run").onclick = start;
$("stop").onclick = stop;
$("step").onclick = () => {
  stop();
  tick();
};
$("policy").onchange = (e) => {
  stop();
  resetWorld();
  setPolicy(e.target.selectedIndex);
};
state = await loadState();
policies = buildPolicies(state);
$("policy").innerHTML = policies.map((p, i) => `<option value="${i}">${p.label}</option>`).join("");
$("evoinfo").textContent = state ? `${state.run} runs · ${state.traces?.length ?? 0} traces · pop ${state.population?.length ?? 0}` : "state.json not found";
drawChart(state);
renderPop(state);
resetWorld();
$("stop").disabled = true;
tick();
if (window.__AUTORUN) {
  $("speed").value = 10;
  for (let i = 0;i < 90; i++)
    tick();
  start();
}
