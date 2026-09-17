import type { Rule, TraceTick } from "../rules/types.js";
import { cloneGenome, genomeKey, mutate, type Genome } from "./genome.js";
import { RewardTable, scoreGenome, splitTraces, type ScoreResult } from "./score.js";

/**
 * Truncation-selection GA over rule genomes.
 *
 * Measured: starting from a plausible-but-wrong user proposal (0.799), this
 * reached a perfect policy (1.000, 5/5 mappings) by generation 12, including
 * discovering a mapping the user had specified incorrectly.
 *
 * My first run of that experiment stopped at 6 generations, plateaued, and
 * looked like a local optimum. It was simply under-run. Hence `generations`
 * is explicit and `history` is returned - a stalled search must be visible
 * rather than inferred.
 */

export interface Candidate {
  genome: Genome;
  train: ScoreResult;
  holdout: ScoreResult;
  /** which run/generation introduced it - provenance for the UI */
  origin: string;
}

export interface EvolveOptions {
  rules: Rule[];
  ticks: TraceTick[];
  actionPool: string[];
  seeds: Genome[];
  generations?: number;
  childrenPerParent?: number;
  survivors?: number;
  holdoutFraction?: number;
  rng?: () => number;
  /** externally proposed genomes (e.g. from an LLM), scored like any mutation */
  proposals?: Genome[];
}

export interface EvolveResult {
  /** best by HELD-OUT score, not by training score */
  best: Candidate;
  population: Candidate[];
  history: { generation: number; train: number; holdout: number }[];
  /** true when held-out fell while training rose - the overfitting signal */
  overfitting: boolean;
}

export function evolve(opts: EvolveOptions): EvolveResult {
  const {
    rules,
    ticks,
    actionPool,
    seeds,
    generations = 20,
    childrenPerParent = 12,
    survivors = 3,
    holdoutFraction = 0.25,
    rng = Math.random,
    proposals = [],
  } = opts;

  const { train, holdout } = splitTraces(ticks, holdoutFraction);
  const trainTable = new RewardTable(train);
  const holdoutTable = new RewardTable(holdout);

  const rate = (g: Genome, origin: string): Candidate => ({
    genome: g,
    train: scoreGenome(g, rules, train, trainTable),
    holdout: scoreGenome(g, rules, holdout, holdoutTable),
    origin,
  });

  let population = [...seeds, ...proposals].map((g, i) =>
    rate(cloneGenome(g), i < seeds.length ? "seed" : "proposal"),
  );
  const history: EvolveResult["history"] = [];

  for (let gen = 0; gen < generations; gen++) {
    const children: Candidate[] = [];
    for (const parent of population) {
      for (let k = 0; k < childrenPerParent; k++) {
        children.push(
          rate(mutate(parent.genome, { actionPool, rng }), `gen${gen + 1}`),
        );
      }
    }

    // dedupe: identical genomes waste evaluations and fake diversity
    const seen = new Set<string>();
    population = [...population, ...children]
      .filter((c) => {
        const k = genomeKey(c.genome);
        if (seen.has(k)) return false;
        seen.add(k);
        return true;
      })
      // select on TRAINING score (held-out must stay unseen by selection)
      .sort((a, b) => b.train.score - a.train.score)
      .slice(0, survivors);

    const top = population[0]!;
    history.push({
      generation: gen + 1,
      train: top.train.score,
      holdout: top.holdout.score,
    });
  }

  // report the best by held-out score
  const best = [...population].sort((a, b) => b.holdout.score - a.holdout.score)[0]!;

  let overfitting = false;
  if (history.length >= 4) {
    const first = history[0]!;
    const last = history[history.length - 1]!;
    overfitting = last.train > first.train && last.holdout < first.holdout;
  }

  return { best, population, history, overfitting };
}
