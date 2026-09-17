import { describe, expect, test } from "bun:test";
import {
  actionsFrom,
  RecordingActuator,
  SpecActuator,
  type ActionSpec,
  type ActuatorBackend,
} from "../src/actuate/actuator.js";

const specs: Record<string, ActionSpec> = {
  shoot: { kind: "key", key: "ctrl" },
  forward_start: { kind: "keyDown", key: "w" },
  forward_stop: { kind: "keyUp", key: "w" },
  wait: { kind: "wait" },
  aim: { kind: "click", x: 400, y: 300 },
};

class MockBackend implements ActuatorBackend {
  readonly calls: string[] = [];
  keyDown(k: string) {
    this.calls.push(`down:${k}`);
  }
  keyUp(k: string) {
    this.calls.push(`up:${k}`);
  }
  click(x: number, y: number, b: "left" | "right") {
    this.calls.push(`click:${x},${y},${b}`);
  }
}

describe("RecordingActuator (dry run)", () => {
  test("logs actions without a backend", () => {
    const a = new RecordingActuator(specs);
    a.perform("shoot");
    a.perform("wait");
    expect(a.log).toEqual(["shoot", "wait"]);
  });

  test("an unknown action throws rather than silently doing nothing", () => {
    const a = new RecordingActuator(specs);
    expect(() => a.perform("moonwalk")).toThrow('unknown action "moonwalk"');
  });

  test("tracks held keys", () => {
    const a = new RecordingActuator(specs);
    a.perform("forward_start");
    expect(a.heldKeys).toEqual(["w"]);
    a.perform("forward_stop");
    expect(a.heldKeys).toEqual([]);
  });
});

describe("SpecActuator", () => {
  test("a key action presses and releases", async () => {
    const b = new MockBackend();
    await new SpecActuator(specs, b).perform("shoot");
    expect(b.calls).toEqual(["down:ctrl", "up:ctrl"]);
  });

  test("keyDown is held until keyUp", async () => {
    const b = new MockBackend();
    const a = new SpecActuator(specs, b);
    await a.perform("forward_start");
    expect(a.heldKeys).toEqual(["w"]);
    expect(b.calls).toEqual(["down:w"]);
    await a.perform("forward_stop");
    expect(a.heldKeys).toEqual([]);
  });

  test("release() frees every held key - the stuck-key guard", async () => {
    const b = new MockBackend();
    const a = new SpecActuator(
      { d1: { kind: "keyDown", key: "w" }, d2: { kind: "keyDown", key: "shift" } },
      b,
    );
    await a.perform("d1");
    await a.perform("d2");
    expect(a.heldKeys).toEqual(["shift", "w"]);
    await a.release();
    expect(a.heldKeys).toEqual([]);
    expect(b.calls).toContain("up:w");
    expect(b.calls).toContain("up:shift");
  });

  test("release() is safe to call twice", async () => {
    const b = new MockBackend();
    const a = new SpecActuator(specs, b);
    await a.perform("forward_start");
    await a.release();
    await expect(a.release()).resolves.toBeUndefined();
  });

  test("wait does nothing at all", async () => {
    const b = new MockBackend();
    await new SpecActuator(specs, b).perform("wait");
    expect(b.calls).toEqual([]);
  });

  test("click routes to the backend", async () => {
    const b = new MockBackend();
    await new SpecActuator(specs, b).perform("aim");
    expect(b.calls).toEqual(["click:400,300,left"]);
  });

  test("a backend that cannot click reports it instead of failing silently", async () => {
    const limited: ActuatorBackend = { keyDown() {}, keyUp() {} };
    const a = new SpecActuator(specs, limited);
    await expect(a.perform("aim")).rejects.toThrow("cannot click");
  });

  test("custom actions run arbitrary code", async () => {
    let ran = false;
    const a = new SpecActuator(
      { custom: { kind: "custom", run: () => void (ran = true) } },
      new MockBackend(),
    );
    await a.perform("custom");
    expect(ran).toBe(true);
  });
});

describe("actionsFrom", () => {
  test("builds a project action dict from one source of specs", () => {
    const rec = new RecordingActuator(specs);
    const dict = actionsFrom(specs, rec);
    expect(Object.keys(dict).sort()).toEqual(
      ["aim", "forward_start", "forward_stop", "shoot", "wait"].sort(),
    );
    dict.shoot!();
    expect(rec.log).toEqual(["shoot"]);
  });
});
