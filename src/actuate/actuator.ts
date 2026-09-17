/**
 * Actuators: turn an emitted action name into a real, reproducible effect.
 *
 * This is the ONLY layer with side effects on the target system, which is
 * what makes three things possible:
 *  - dry runs (swap in a RecordingActuator and nothing touches the target)
 *  - replay (re-issue a recorded action list against a fresh target)
 *  - a kill switch (one place to stop everything)
 */

export interface Actuator {
  /** perform the named action; unknown names must throw, not no-op silently */
  perform(action: string): void | Promise<void>;
  /** release anything held down (keys, buttons) - always safe to call twice */
  release?(): void | Promise<void>;
}

/** An action expressed as data, so it can be logged, diffed and replayed. */
export type ActionSpec =
  | { kind: "key"; key: string; hold?: number }
  | { kind: "keyDown"; key: string }
  | { kind: "keyUp"; key: string }
  | { kind: "click"; x: number; y: number; button?: "left" | "right" }
  | { kind: "move"; dx: number; dy: number }
  | { kind: "wait" }
  | { kind: "custom"; run: () => void | Promise<void> };

/**
 * Records what was asked for without doing it. Used for dry runs and for
 * asserting in tests that a policy emits the intended actions - checking the
 * emitted sequence is far more reliable than checking pixels afterwards.
 */
export class RecordingActuator implements Actuator {
  readonly log: string[] = [];
  private held = new Set<string>();

  constructor(private specs: Record<string, ActionSpec>) {}

  perform(action: string): void {
    const spec = this.specs[action];
    if (!spec) throw new Error(`unknown action "${action}"`);
    this.log.push(action);
    if (spec.kind === "keyDown") this.held.add(spec.key);
    if (spec.kind === "keyUp") this.held.delete(spec.key);
  }

  release(): void {
    this.held.clear();
  }

  get heldKeys(): string[] {
    return [...this.held].sort();
  }

  reset(): void {
    this.log.length = 0;
    this.held.clear();
  }
}

/**
 * Dispatches ActionSpecs through a pluggable backend.
 *
 * The backend is injected rather than imported so the same project runs
 * against xdotool, a game's own API, an emulator, or a mock, with no change
 * to the rules. YAGNI: no backend is bundled until one is needed.
 */
export interface ActuatorBackend {
  keyDown(key: string): void | Promise<void>;
  keyUp(key: string): void | Promise<void>;
  click?(x: number, y: number, button: "left" | "right"): void | Promise<void>;
  move?(dx: number, dy: number): void | Promise<void>;
}

export class SpecActuator implements Actuator {
  private held = new Set<string>();

  constructor(
    private specs: Record<string, ActionSpec>,
    private backend: ActuatorBackend,
  ) {}

  async perform(action: string): Promise<void> {
    const spec = this.specs[action];
    if (!spec) throw new Error(`unknown action "${action}"`);

    switch (spec.kind) {
      case "wait":
        return;
      case "custom":
        await spec.run();
        return;
      case "key": {
        await this.backend.keyDown(spec.key);
        if (spec.hold && spec.hold > 0) {
          await new Promise((r) => setTimeout(r, spec.hold));
        }
        await this.backend.keyUp(spec.key);
        return;
      }
      case "keyDown":
        this.held.add(spec.key);
        await this.backend.keyDown(spec.key);
        return;
      case "keyUp":
        this.held.delete(spec.key);
        await this.backend.keyUp(spec.key);
        return;
      case "click":
        if (!this.backend.click) throw new Error("backend cannot click");
        await this.backend.click(spec.x, spec.y, spec.button ?? "left");
        return;
      case "move":
        if (!this.backend.move) throw new Error("backend cannot move");
        await this.backend.move(spec.dx, spec.dy);
        return;
    }
  }

  /**
   * Release every held key. Must be called when the loop stops, or a
   * keyDown that never got its keyUp leaves the target stuck moving.
   */
  async release(): Promise<void> {
    for (const key of [...this.held]) {
      await this.backend.keyUp(key);
      this.held.delete(key);
    }
  }

  get heldKeys(): string[] {
    return [...this.held].sort();
  }
}

/** Build the project's action dict from specs + an actuator. DRY: one source. */
export function actionsFrom(
  specs: Record<string, ActionSpec>,
  actuator: Actuator,
): Record<string, () => void | Promise<void>> {
  return Object.fromEntries(
    Object.keys(specs).map((name) => [name, () => actuator.perform(name)]),
  );
}
