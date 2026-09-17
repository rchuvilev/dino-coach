import { Agent } from "../kernel/agent.js";
import type { Project } from "../project.js";
import { applyGenome, type Genome } from "./genome.js";

/**
 * Evaluate a genome by RUNNING it, not by replaying traces.
 *
 * Why this exists: trace replay is off-policy. It answers "given the states
 * I saw, which action paid best?" but cannot answer "which states would this
 * policy have reached?". Measured in the demo - a genome scored 15.93 on
 * held-out replay and produced 0 reward live, because "strafe" looked
 * excellent in states that only existed because the recording policy also
 * shot. An always-strafe policy never reaches them.
 *
 * So the real scorer is a live run. Replay stays useful as a cheap prefilter
 * (3.3ms per genome vs a full episode), but it must never be the final word.
 */

export interface LiveEvalOptions {
  project: Project;
  /** reset the target system to a known state before each evaluation */
  reset: () => void | Promise<void>;
  /** advance the target system by one tick, applying the agent's action */
  stepWorld: () => void | Promise<void>;
  ticks: number;
  explore?: number;
  rng?: () => number;
}

export interface LiveScore {
  genome: Genome;
  totalReward: number;
  ticks: number;
}

/** Run one genome for `ticks` and return the reward it actually earned. */
export async function evaluateLive(
  genome: Genome,
  opts: LiveEvalOptions,
): Promise<LiveScore> {
  const { project, reset, stepWorld, ticks, explore = 0, rng } = opts;
  await reset();

  const rs = applyGenome(project.rules, genome);
  const tuned: Project = { ...project, rules: rs.rules, order: rs.order };
  const agent = new Agent({ project: tuned, runId: "eval", explore, rng });

  for (let i = 0; i < ticks; i++) {
    await agent.tick();
    await stepWorld();
  }

  return {
    genome,
    totalReward: agent.recorder.totalReward(),
    ticks,
  };
}

/**
 * Rank genomes by live reward. Sequential by necessity - they share one
 * target system, so they cannot be evaluated concurrently.
 */
export async function rankLive(
  genomes: Genome[],
  opts: LiveEvalOptions,
): Promise<LiveScore[]> {
  const out: LiveScore[] = [];
  for (const g of genomes) out.push(await evaluateLive(g, opts));
  return out.sort((a, b) => b.totalReward - a.totalReward);
}
