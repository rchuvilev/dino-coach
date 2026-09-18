/**
 * Liveness verification for evaluation targets.
 *
 * This module exists because of a specific, expensive mistake. Evaluating
 * policies against Chrome Dino produced a completely plausible score (8361,
 * with near-identical runs that looked like clean determinism) from a target
 * that was not running at all. Three separate wrong conclusions followed from
 * one unchecked precondition:
 *
 *  1. "synthetic key events are ignored" - false; the game was crashed, so
 *     nothing responded to any input.
 *  2. "the tab backgrounds and freezes the game" - false; a crashed dino
 *     stops its own animation loop regardless of visibility.
 *  3. an invalid baseline table that a human observer ("but I could see it
 *     jump") disproved in one sentence.
 *
 * The rule distilled: BEFORE scoring anything, prove the target advances when
 * stepped and prove an action changes its behaviour. Both, every time.
 */

export interface LivenessProbe<S> {
  /** put the target into a fresh, running state */
  reset: () => void | Promise<void>;
  /** advance the target by one tick, applying an action */
  step: (action: string) => void | Promise<void>;
  /** read the target's observable state */
  observe: () => S | Promise<S>;
  /** a monotonic progress measure, e.g. distance or score */
  progress: (s: S) => number;
  /** an action expected to visibly change behaviour (e.g. "jump") */
  probeAction: string;
  /** a neutral action (e.g. "run") */
  idleAction: string;
  /** how many ticks to observe; must exceed any startup delay */
  ticks?: number;
}

export interface LivenessResult {
  /** did progress increase while idling? */
  advances: boolean;
  /** did probeAction produce a different state trajectory than idling? */
  actionHasEffect: boolean;
  /** progress delta observed while idling */
  idleDelta: number;
  /** number of distinct observed states while idling */
  distinctStates: number;
  alive: boolean;
  reasons: string[];
}

/**
 * Prove a target is alive and controllable.
 *
 * `alive` is true only when BOTH checks pass. A target that advances but
 * ignores input is as useless for evaluation as a frozen one - that is the
 * exact configuration that produced the fake 8361.
 */
export async function verifyLiveness<S>(p: LivenessProbe<S>): Promise<LivenessResult> {
  const ticks = p.ticks ?? 30;
  const reasons: string[] = [];

  // --- 1. does it advance at all while idling?
  await p.reset();
  const seen = new Set<string>();
  const first = p.progress(await p.observe());
  for (let i = 0; i < ticks; i++) {
    await p.step(p.idleAction);
    seen.add(JSON.stringify(await p.observe()));
  }
  const idleDelta = p.progress(await p.observe()) - first;
  const advances = idleDelta > 0;
  if (!advances) reasons.push(`target did not advance while idling (delta ${idleDelta})`);
  if (seen.size <= 1) {
    reasons.push("observed state never changed - target is frozen");
  }

  // --- 2. does an action change the trajectory?
  // Compared against idling from the SAME reset state, so a difference can
  // only come from the action itself.
  // Compare the FULL observed state, not just progress. Measured defect in
  // the first version of this guard: on the runner target, jumping does not
  // change DISTANCE within a short probe (the first obstacle has not arrived
  // yet), so a progress-only comparison reported "input is not reaching the
  // target" on a perfectly controllable world. The observed state showed 10
  // distinct y values versus 1 - the evidence was already being collected
  // and thrown away.
  await p.reset();
  const idleTrace: string[] = [];
  for (let i = 0; i < ticks; i++) {
    await p.step(p.idleAction);
    idleTrace.push(JSON.stringify(await p.observe()));
  }

  await p.reset();
  const actTrace: string[] = [];
  for (let i = 0; i < ticks; i++) {
    await p.step(p.probeAction);
    actTrace.push(JSON.stringify(await p.observe()));
  }

  const actionHasEffect = idleTrace.some((v, i) => v !== actTrace[i]);
  if (!actionHasEffect) {
    reasons.push(
      `action "${p.probeAction}" produced an identical trajectory to "${p.idleAction}" - input is not reaching the target`,
    );
  }

  return {
    advances,
    actionHasEffect,
    idleDelta,
    distinctStates: seen.size,
    alive: advances && seen.size > 1 && actionHasEffect,
    reasons,
  };
}

export interface ControlScores {
  /** label -> mean score */
  scores: Record<string, number>;
  best: { label: string; score: number };
}

/**
 * Compare a policy against trivial controls.
 *
 * A policy that cannot beat do-nothing is not evidence of a bad policy - it
 * is evidence the METRIC or the ENVIRONMENT is broken. Measured on the runner
 * target: a ceiling condition made a ground-truth ORACLE score 129 against a
 * do-nothing control's 129, which looked exactly like broken world geometry
 * and was actually a 2-pixel timing window error.
 */
export function assessAgainstControls(
  policyScore: number,
  controls: Record<string, number>,
): { valid: boolean; ratio: number; verdict: string; best: string } {
  const entries = Object.entries(controls);
  let best = entries[0] ?? ["none", 0];
  for (const e of entries) if (e[1] > best[1]) best = e;
  const ratio = best[1] === 0 ? Infinity : policyScore / best[1];
  const valid = policyScore > best[1];
  return {
    valid,
    ratio,
    best: best[0],
    verdict: valid
      ? `policy beats best control "${best[0]}" by ${ratio.toFixed(2)}x`
      : `SUSPECT: control "${best[0]}" (${best[1]}) matches or beats the policy (${policyScore}) - the metric or environment is broken, not the policy`,
  };
}
