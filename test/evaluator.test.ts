import { describe, expect, test } from "bun:test";
import { RuleEvaluator } from "../src/rules/evaluator.js";
import type { RuleSet } from "../src/rules/types.js";

const rs = (rules: RuleSet["rules"]): RuleSet => ({
  rules,
  order: rules.map((_, i) => i),
});

describe("priority ordering", () => {
  test("first matching rule in order wins", () => {
    const e = new RuleEvaluator(
      rs([
        { when: "hp_low", then: "strafe" },
        { when: "enemy", then: "shoot" },
      ]),
    );
    const d = e.step({ hp_low: true, enemy: true });
    expect(d.action).toBe("strafe");
    expect(d.suppressed).toEqual([1]);
  });

  test("reordering changes the winner - proves order is load-bearing", () => {
    const rules = [
      { when: "hp_low", then: "strafe" },
      { when: "enemy", then: "shoot" },
    ];
    const e = new RuleEvaluator({ rules, order: [1, 0] });
    expect(e.step({ hp_low: true, enemy: true }).action).toBe("shoot");
  });

  test("no match yields a null action, not a crash", () => {
    const e = new RuleEvaluator(rs([{ when: "hp_low", then: "strafe" }]));
    expect(e.step({ hp_low: false }).action).toBeNull();
  });
});

describe("hysteresis (the anti-dithering guarantee)", () => {
  /**
   * Reproduces the measured failure: two rules with coupled effects.
   * Shooting costs hp; strafing restores it. Without commitment the agent
   * flip-flops and neither behaviour ever completes (measured 66% of ticks).
   */
  function simulate(commitTicks: number): { flips: number; ticks: number } {
    const e = new RuleEvaluator(
      rs([
        { when: "hp_low", then: "strafe", commitTicks },
        { when: "always", then: "shoot", commitTicks },
      ]),
    );
    let hp = 100;
    let last: string | null = null;
    let flips = 0;
    const ticks = 300;
    for (let t = 0; t < ticks; t++) {
      const d = e.step({ hp_low: hp < 100, always: true });
      if (last !== null && d.action !== last) flips++;
      last = d.action;
      if (d.action === "strafe") hp = Math.min(100, hp + 8);
      else hp -= 12;
    }
    return { flips, ticks };
  }

  test("NEGATIVE CONTROL: without commitment it dithers badly", () => {
    const { flips, ticks } = simulate(0);
    // measured ~66%; assert it is unambiguously broken
    expect(flips / ticks).toBeGreaterThan(0.3);
  });

  test("with commitment the same rules are stable", () => {
    const { flips, ticks } = simulate(8);
    expect(flips / ticks).toBeLessThan(0.2);
  });

  test("commitment is abandoned when its condition lapses", () => {
    const e = new RuleEvaluator(
      rs([{ when: "hp_low", then: "strafe", commitTicks: 50 }]),
    );
    expect(e.step({ hp_low: true }).action).toBe("strafe");
    // condition gone -> must not stay committed for 49 more ticks
    expect(e.step({ hp_low: false }).action).toBeNull();
  });
});

describe("cooldown", () => {
  test("a rule cannot re-fire while cooling down", () => {
    const e = new RuleEvaluator(
      rs([
        { when: "ammo_low", then: "reload", cooldownTicks: 5 },
        { when: "enemy", then: "shoot" },
      ]),
    );
    const on = { ammo_low: true, enemy: true };
    expect(e.step(on).action).toBe("reload");
    const second = e.step(on);
    expect(second.action).toBe("shoot");
    expect(second.cooling).toContain(0);
  });

  test("it fires again once the cooldown expires", () => {
    const e = new RuleEvaluator(
      rs([{ when: "ammo_low", then: "reload", cooldownTicks: 3 }]),
    );
    // fires at t=0, then blocked for exactly cooldownTicks ticks (t=1,2,3... )
    expect(e.step({ ammo_low: true }).action).toBe("reload"); // t=0
    expect(e.step({ ammo_low: true }).action).toBeNull(); // t=1
    expect(e.step({ ammo_low: true }).action).toBeNull(); // t=2
    expect(e.step({ ammo_low: true }).action).toBe("reload"); // t=3, expired
  });
});
