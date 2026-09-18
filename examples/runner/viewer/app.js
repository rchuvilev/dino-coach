/**
 * Live viewer.
 *
 * DRY: this imports the SAME world, perception and evaluator the tests and
 * the evolution runs use. A viewer with its own copy of the rules would
 * drift from the real system and show a comforting lie - which is exactly
 * the failure class this project keeps running into.
 */
import { RunnerWorld, render, GROUND, PLAYER_X, W, H } from "../world.js";
import { perceive } from "../perceive.js";
import { RuleEvaluator } from "../../../src/rules/evaluator.js";
import { applyGenome } from "../../../src/evolve/genome.js";

const RULES = [
  { when: "obstacle_imminent", then: "jump" },
  { when: "high_obstacle", then: "duck" },
  { when: "obstacle_near", then: "jump" },
  { when: "always", then: "run" },
];
const TH = { windowLo: 4, windowHi: 12, duckAt: 12 };

const conditionsFrom = (p) => {
  const has = p.gap < 999;
  return {
    obstacle_imminent:
      has && !p.high && p.gap >= TH.windowLo && p.gap <= TH.windowHi && !p.airborne,
    high_obstacle: has && p.high && p.gap <= TH.duckAt,
    obstacle_near: has && p.gap <= TH.windowHi + 10,
    always: true,
  };
};

const $ = (id) => document.getElementById(id);
const screen = $("screen");
const sctx = screen.getContext("2d");
const chart = $("chart");
const cctx = chart.getContext("2d");

let state = null; // loaded evolution state
let policies = [];
let world = new RunnerWorld(101);
let evaluator = null;
let running = false;
let timer = null;
let seedIdx = 0;
const HOLDOUT = [101, 211, 307, 401, 503, 601, 701, 809, 907, 1009];

async function loadState() {
  // window.__STATE is injected by a <script> tag for file/minis:// hosting,
  // where fetch() is blocked by the scheme. fetch is the http fallback.
  //
  // Measured: a `type="module"` script was evaluated BEFORE the classic
  // state.js tag above it had run, so __STATE read as undefined and the
  // viewer showed "state.json not found" with only the 3 built-in policies -
  // while window.__STATE was demonstrably present a moment later. Wait for
  // DOM readiness rather than assuming tag order.
  if (document.readyState === "loading") {
    await new Promise((r) => document.addEventListener("DOMContentLoaded", r, { once: true }));
  }
  if (window.__STATE) return window.__STATE;
  try {
    const r = await fetch("state.json", { cache: "no-store" });
    if (!r.ok) throw new Error(String(r.status));
    return await r.json();
  } catch {
    return null;
  }
}

function buildPolicies(st) {
  const out = [
    { label: "human seed rules", genome: { order: [0, 1, 2, 3], actions: RULES.map((r) => r.then) }, kind: "seed" },
    { label: "control: do-nothing", genome: { order: [0, 1, 2, 3], actions: ["run", "run", "run", "run"] }, kind: "ctrl" },
    { label: "control: always-jump", genome: { order: [3, 0, 1, 2], actions: ["jump", "jump", "jump", "jump"] }, kind: "ctrl" },
  ];
  if (st?.population?.length) {
    const seen = new Set();
    for (const p of st.population) {
      const k = `${p.genome.order}|${p.genome.actions}`;
      if (seen.has(k)) continue;
      seen.add(k);
      out.unshift({
        label: `evolved r${p.foundInRun} · holdout ${p.holdout}`,
        genome: p.genome,
        kind: "evo",
        holdout: p.holdout,
      });
    }
    out.sort((a, b) => (b.holdout ?? -1) - (a.holdout ?? -1));
  }
  return out;
}

function setPolicy(i) {
  const p = policies[i];
  if (!p) return;
  evaluator = new RuleEvaluator(applyGenome(RULES, p.genome));
  $("pname").textContent = p.label;
  renderRules(null, {});
}

function resetWorld() {
  world = new RunnerWorld(HOLDOUT[seedIdx % HOLDOUT.length]);
  const i = $("policy").selectedIndex;
  setPolicy(i < 0 ? 0 : i);
}

/** Draw the frame the agent actually reads, plus overlays it does NOT see. */
function draw(frame, per, decision) {
  const img = sctx.createImageData(W, H);
  img.data.set(frame.data);
  sctx.putImageData(img, 0, 0);

  // overlay: perception readouts, drawn AFTER so they cannot pollute pixels
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
  // the takeoff window the rules use
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
    if (idx === fired) li.className = "fire";
    else if (matched) li.className = "match";
    li.innerHTML =
      `<span>${r.when}</span><span class="arrow">=&gt;</span><span class="then">${act}</span>` +
      `<span class="why">${idx === fired ? "FIRING" : matched ? "suppressed" : ""}</span>`;
    ul.appendChild(li);
  }
}

function renderConds(conds) {
  const ul = $("conds");
  ul.innerHTML = "";
  for (const [k, v] of Object.entries(conds)) {
    const li = document.createElement("li");
    if (v) li.className = "on";
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
  $("m-y").style.width = `${Math.min(100, (per.playerY / 22) * 100)}%`;
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
  const y = (v) => chart.height - 20 - (v / max) * (chart.height - 34);

  cctx.strokeStyle = "#3ddc84";
  cctx.lineWidth = 2;
  cctx.beginPath();
  h.forEach((p, i) => (i ? cctx.lineTo(pad + i * step, y(p.bestHoldout)) : cctx.moveTo(pad, y(p.bestHoldout))));
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
    if (r.holdout === best) tr.className = "best";
    tr.innerHTML =
      `<td>${r.foundInRun}</td><td>${r.genome.actions.join(",")}</td>` +
      `<td class="n">${r.holdout}</td>`;
    t.appendChild(tr);
  }
}

function start() {
  if (running) return;
  running = true;
  $("run").disabled = true;
  $("stop").disabled = false;
  const loop = () => {
    if (!running) return;
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
$("policy").innerHTML = policies
  .map((p, i) => `<option value="${i}">${p.label}</option>`)
  .join("");
$("evoinfo").textContent = state
  ? `${state.run} runs · ${state.traces?.length ?? 0} traces · pop ${state.population?.length ?? 0}`
  : "state.json not found";
drawChart(state);
renderPop(state);
resetWorld();
$("stop").disabled = true;
tick();

// Autorun so a single screenshot shows a LIVE frame. Without it, a capture
// taken right after navigate shows tick 0 and nothing interesting.
if (window.__AUTORUN) {
  $("speed").value = 10;
  // warm up past the first spawn so an obstacle is on screen immediately
  for (let i = 0; i < 90; i++) tick();
  start();
}
