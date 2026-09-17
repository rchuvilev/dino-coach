import { TickLoop, type TickStats } from "./loop.js";
import { TraceRecorder } from "./trace.js";
import { RuleEvaluator } from "../rules/evaluator.js";
import type { Decision, StateProps } from "../rules/types.js";
import { ratingsOf, ruleSetOf, validate, type Project } from "../project.js";

/**
 * The runnable agent: sensors -> conditions -> rules -> actions, once per tick.
 *
 * This is the only place that touches the outside world, via the project's
 * action dict. Keeping side effects in one dict is what makes replay, mocking
 * and a kill switch possible.
 */

export interface AgentSnapshot {
  tick: number;
  props: StateProps;
  conditions: Record<string, boolean>;
  decision: Decision | null;
  reward: number;
  totalReward: number;
  drift: number;
  /** per-tick ms spent in the agent, for the UI budget meter */
  work: number;
  /** true when this tick's action was a random exploration, not the rule's */
  explored: boolean;
}

export interface AgentOptions {
  project: Project;
  runId?: string;
  /** called every tick with everything the UI needs to render */
  onSnapshot?: (s: AgentSnapshot) => void;
  /** set false to decide without executing - dry run / replay */
  execute?: boolean;

  /**
   * Probability per tick of taking a RANDOM action instead of the rule's.
   *
   * This is not optional for learning, and the reason is measured: a fully
   * deterministic policy records only ONE action per condition-set, so
   * counterfactual replay can never judge an alternative and the GA scores
   * every rival at 0 coverage. Exploration is what generates the data that
   * makes evolution possible at all.
   *
   * Default 0 (pure exploitation) so a production run is deterministic;
   * set ~0.1 while learning.
   */
  explore?: number;
  rng?: () => number;
}

export class Agent {
  readonly evaluator: RuleEvaluator;
  readonly recorder: TraceRecorder;
  private loop: TickLoop;
  private project: Project;
  private prevProps: StateProps | null = null;
  private ratings: Record<string, number>;
  private execute: boolean;
  private lastSnapshot: AgentSnapshot | null = null;
  private explore: number;
  private rng: () => number;
  private actionNames: string[];

  constructor(private opts: AgentOptions) {
    const errors = validate(opts.project);
    if (errors.length > 0) {
      throw new Error(`invalid project:\n  ${errors.join("\n  ")}`);
    }
    this.project = opts.project;
    this.evaluator = new RuleEvaluator(ruleSetOf(opts.project));
    this.recorder = new TraceRecorder(opts.runId ?? `run-${Date.now()}`);
    this.ratings = ratingsOf(opts.project);
    this.execute = opts.execute ?? true;
    this.explore = opts.explore ?? 0;
    this.rng = opts.rng ?? Math.random;
    this.actionNames = Object.keys(opts.project.actions);
    this.loop = new TickLoop({
      hz: opts.project.hz,
      onTick: (s) => this.tick(s),
    });
  }

  get snapshot(): AgentSnapshot | null {
    return this.lastSnapshot;
  }

  get isRunning(): boolean {
    return this.loop.isRunning;
  }

  start(): void {
    this.loop.start();
  }

  stop(): void {
    this.loop.stop();
  }

  restart(): void {
    this.loop.reset();
    this.evaluator.reset();
    this.recorder.reset();
    this.prevProps = null;
    this.loop.start();
  }

  /** One tick, exposed so tests and replay can drive it without a timer. */
  async tick(stats?: TickStats): Promise<AgentSnapshot> {
    const t0 = Date.now();

    // 1. read state props
    const props: StateProps = {};
    for (const [name, spec] of Object.entries(this.project.props)) {
      props[name] = await spec.read();
    }

    // 2. derive conditions
    const conditions: Record<string, boolean> = {};
    for (const [name, fn] of Object.entries(this.project.conditions)) {
      conditions[name] = fn(props, this.prevProps);
    }

    // 3. decide
    const decision = this.evaluator.step(conditions);

    // 3b. explore: occasionally substitute a random action so the trace
    // contains alternatives the GA can actually evaluate. Without this the
    // recorded data supports only the policy that produced it.
    let explored = false;
    if (this.explore > 0 && this.actionNames.length > 0 && this.rng() < this.explore) {
      const pick = this.actionNames[Math.floor(this.rng() * this.actionNames.length)];
      if (pick !== undefined && pick !== decision.action) {
        decision.action = pick;
        explored = true;
      }
    }

    // 4. record (reward comes from prop deltas, so it is dense)
    const reward = this.recorder.record(
      this.loop.tickCount,
      conditions,
      props,
      decision.firedRule,
      decision.action,
      this.ratings,
    );

    // 5. act - the only side effect in the system
    if (this.execute && decision.action) {
      const fn = this.project.actions[decision.action];
      if (fn) await fn();
    }

    this.prevProps = props;
    const snap: AgentSnapshot = {
      tick: this.loop.tickCount,
      props,
      conditions,
      decision,
      reward,
      totalReward: this.recorder.totalReward(),
      drift: stats?.drift ?? 0,
      work: Date.now() - t0,
      explored,
    };
    this.lastSnapshot = snap;
    this.opts.onSnapshot?.(snap);
    return snap;
  }
}
