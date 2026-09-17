import { describe, expect, test } from "bun:test";
import {
  applyView,
  ratingsOf,
  ruleSetOf,
  toView,
  validate,
  type Project,
} from "../src/project.js";

const base = (): Project => ({
  name: "demo",
  hz: 10,
  props: {
    hp: { rating: 1, read: () => 100, min: 0, max: 100 },
    ammo: { rating: 0.5, read: () => 50 },
  },
  conditions: {
    hp_low: (p) => (p.hp ?? 100) < 50,
    always: () => true,
  },
  actions: { strafe: () => {}, shoot: () => {} },
  rules: [
    { when: "hp_low", then: "strafe", commitTicks: 8 },
    { when: "always", then: "shoot" },
  ],
});

describe("validate", () => {
  test("a well-formed project has no errors", () => {
    expect(validate(base())).toEqual([]);
  });

  test("catches an unknown condition", () => {
    const p = base();
    p.rules[0]!.when = "typo_here";
    expect(validate(p).join()).toContain('unknown condition "typo_here"');
  });

  test("catches an unknown action", () => {
    const p = base();
    p.rules[1]!.then = "moonwalk";
    expect(validate(p).join()).toContain('unknown action "moonwalk"');
  });

  test("catches a duplicate order index", () => {
    const p = base();
    p.order = [0, 0];
    expect(validate(p).join()).toContain("duplicate");
  });

  test("catches an out-of-range order index", () => {
    const p = base();
    p.order = [0, 7];
    expect(validate(p).join()).toContain("out of range");
  });

  test("catches an all-zero rating set, which would make reward always 0", () => {
    const p = base();
    p.props.hp!.rating = 0;
    p.props.ammo!.rating = 0;
    expect(validate(p).join()).toContain("reward is always 0");
  });

  test("catches a non-positive hz", () => {
    const p = base();
    p.hz = 0;
    expect(validate(p).join()).toContain("hz must be positive");
  });
});

describe("derivations", () => {
  test("ruleSetOf defaults order to declaration order", () => {
    expect(ruleSetOf(base()).order).toEqual([0, 1]);
  });

  test("ratingsOf carries the reward rule, not just the weight", () => {
    const r = ratingsOf(base());
    expect(r.hp!.rating).toBe(1);
    expect(r.ammo!.rating).toBe(0.5);
  });

  test("ratingsOf forwards countDirection and maxDelta", () => {
    const p = base();
    p.props.hp!.countDirection = "down";
    p.props.hp!.maxDelta = 50;
    const r = ratingsOf(p);
    expect(r.hp!.countDirection).toBe("down");
    expect(r.hp!.maxDelta).toBe(50);
  });
});

describe("UI round trip (code-first, UI-representable)", () => {
  test("toView exposes names, not functions", () => {
    const v = toView(base());
    expect(v.conditionNames).toEqual(["hp_low", "always"]);
    expect(v.actionNames).toEqual(["strafe", "shoot"]);
    expect(v.props.hp).toEqual({ rating: 1, min: 0, max: 100 });
  });

  test("a view survives JSON serialisation", () => {
    const v = toView(base());
    expect(JSON.parse(JSON.stringify(v))).toEqual(v);
  });

  test("applyView writes structural edits back", () => {
    const p = base();
    const v = toView(p);
    v.order = [1, 0];
    v.props.hp!.rating = 42;
    const next = applyView(p, v);
    expect(next.order).toEqual([1, 0]);
    expect(next.props.hp!.rating).toBe(42);
  });

  test("applyView preserves behaviour functions", () => {
    const p = base();
    const v = toView(p);
    v.props.hp!.rating = 5;
    const next = applyView(p, v);
    // the read fn must survive a round trip through the UI
    expect(next.props.hp!.read()).toBe(100);
    expect(typeof next.conditions.hp_low).toBe("function");
  });

  test("an edited view still validates", () => {
    const p = base();
    const v = toView(p);
    v.order = [1, 0];
    expect(validate(applyView(p, v))).toEqual([]);
  });
});
