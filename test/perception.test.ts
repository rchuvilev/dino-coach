import { describe, expect, test } from "bun:test";
import {
  colorDistance,
  features,
  readBar,
  readPresence,
  validateReader,
  type Frame,
} from "../src/perception/readers.js";

/** Build a synthetic frame so perception is testable without a screen. */
function frame(width: number, height: number, fill: [number, number, number] = [0, 0, 0]): Frame {
  const data = new Uint8Array(width * height * 4);
  for (let i = 0; i < width * height; i++) {
    data[i * 4] = fill[0];
    data[i * 4 + 1] = fill[1];
    data[i * 4 + 2] = fill[2];
    data[i * 4 + 3] = 255;
  }
  return { width, height, data };
}

function paint(f: Frame, x: number, y: number, w: number, h: number, c: [number, number, number]) {
  for (let dy = 0; dy < h; dy++) {
    for (let dx = 0; dx < w; dx++) {
      const i = ((y + dy) * f.width + (x + dx)) * 4;
      f.data[i] = c[0];
      f.data[i + 1] = c[1];
      f.data[i + 2] = c[2];
    }
  }
}

const RED: [number, number, number] = [220, 30, 30];
const region = { x: 10, y: 5, w: 100, h: 10 };

describe("readBar", () => {
  test("an empty bar reads 0", () => {
    const f = frame(200, 40);
    expect(readBar(f, { kind: "bar", region, fillColor: RED })).toBe(0);
  });

  test("a full bar reads the full scale", () => {
    const f = frame(200, 40);
    paint(f, 10, 5, 100, 10, RED);
    expect(readBar(f, { kind: "bar", region, fillColor: RED, scale: 100 })).toBe(100);
  });

  test("a half-filled bar reads ~50", () => {
    const f = frame(200, 40);
    paint(f, 10, 5, 50, 10, RED);
    expect(readBar(f, { kind: "bar", region, fillColor: RED, scale: 100 })).toBeCloseTo(50, 0);
  });

  test("reads the LAST filled pixel, so a gloss gap does not undercount", () => {
    const f = frame(200, 40);
    paint(f, 10, 5, 80, 10, RED);
    // a horizontal gloss line through the middle, wrong colour
    paint(f, 10, 9, 80, 2, [255, 255, 255]);
    const v = readBar(f, { kind: "bar", region, fillColor: RED, scale: 100 });
    // majority of the cross-section still matches, so it must still read ~80
    expect(v).toBeCloseTo(80, 0);
  });

  test("a 1px border of the fill colour cannot trip the reading", () => {
    const f = frame(200, 40);
    // only the top row is red across the whole bar
    paint(f, 10, 5, 100, 1, RED);
    expect(readBar(f, { kind: "bar", region, fillColor: RED, scale: 100 })).toBe(0);
  });

  test("works along the y axis", () => {
    const f = frame(40, 200);
    const vert = { x: 5, y: 10, w: 10, h: 100 };
    paint(f, 5, 10, 10, 25, RED);
    expect(
      readBar(f, { kind: "bar", region: vert, fillColor: RED, axis: "y", scale: 100 }),
    ).toBeCloseTo(25, 0);
  });

  test("out-of-bounds regions do not throw", () => {
    const f = frame(20, 20);
    expect(() =>
      readBar(f, { kind: "bar", region: { x: 15, y: 15, w: 50, h: 50 }, fillColor: RED }),
    ).not.toThrow();
  });
});

describe("readPresence", () => {
  test("returns 1 when the region is entirely the colour", () => {
    const f = frame(50, 50);
    paint(f, 0, 0, 10, 10, RED);
    expect(readPresence(f, { kind: "presence", region: { x: 0, y: 0, w: 10, h: 10 }, color: RED })).toBe(1);
  });

  test("returns 0 when absent", () => {
    const f = frame(50, 50);
    expect(readPresence(f, { kind: "presence", region: { x: 0, y: 0, w: 10, h: 10 }, color: RED })).toBe(0);
  });

  test("returns a fraction when partially present", () => {
    const f = frame(50, 50);
    paint(f, 0, 0, 5, 10, RED);
    expect(
      readPresence(f, { kind: "presence", region: { x: 0, y: 0, w: 10, h: 10 }, color: RED }),
    ).toBeCloseTo(0.5, 2);
  });
});

describe("colorDistance", () => {
  test("identical colours are 0 apart", () => {
    expect(colorDistance([1, 2, 3], [1, 2, 3])).toBe(0);
  });

  test("black to white is the maximum", () => {
    expect(colorDistance([0, 0, 0], [255, 255, 255])).toBeCloseTo(441.67, 1);
  });
});

describe("features", () => {
  test("produces size*size values in 0..1", () => {
    const f = frame(64, 64, [128, 128, 128]);
    const v = features(f, { x: 0, y: 0, w: 64, h: 64 }, 8);
    expect(v).toHaveLength(64);
    for (const x of v) {
      expect(x).toBeGreaterThanOrEqual(0);
      expect(x).toBeLessThanOrEqual(1);
    }
  });

  test("distinguishes two visually different regions", () => {
    const a = frame(32, 32, [0, 0, 0]);
    const b = frame(32, 32, [255, 255, 255]);
    const fa = features(a, { x: 0, y: 0, w: 32, h: 32 }, 8);
    const fb = features(b, { x: 0, y: 0, w: 32, h: 32 }, 8);
    const diff = fa.reduce((acc, v, i) => acc + Math.abs(v - fb[i]!), 0);
    expect(diff).toBeGreaterThan(50);
  });
});

describe("validateReader (the gate on LLM-proposed perception)", () => {
  function sampleAt(fill: number): { frame: Frame; expected: number } {
    const f = frame(200, 40);
    if (fill > 0) paint(f, 10, 5, fill, 10, RED);
    return { frame: f, expected: fill };
  }

  test("accepts a correct reader", () => {
    const r = validateReader(
      { prop: "hp", spec: { kind: "bar", region, fillColor: RED, scale: 100 } },
      [sampleAt(100), sampleAt(50), sampleAt(25)],
      2,
    );
    expect(r.accepted).toBe(true);
    expect(r.mae).toBeLessThan(2);
  });

  test("REJECTS a reader pointed at the wrong region", () => {
    const r = validateReader(
      {
        prop: "hp",
        // wrong y: reads empty space, so every frame reads 0
        spec: { kind: "bar", region: { x: 10, y: 30, w: 100, h: 5 }, fillColor: RED, scale: 100 },
      },
      [sampleAt(100), sampleAt(50)],
      2,
    );
    expect(r.accepted).toBe(false);
    expect(r.mae).toBeGreaterThan(50);
  });

  test("refuses to accept with zero samples rather than passing vacuously", () => {
    const r = validateReader(
      { prop: "hp", spec: { kind: "bar", region, fillColor: RED } },
      [],
      999,
    );
    expect(r.accepted).toBe(false);
    expect(r.samples).toBe(0);
  });

  test("reports maxError, so one bad frame cannot hide behind a good mean", () => {
    const r = validateReader(
      { prop: "hp", spec: { kind: "bar", region, fillColor: RED, scale: 100 } },
      [sampleAt(100), sampleAt(50), { frame: frame(200, 40), expected: 100 }],
      2,
    );
    expect(r.maxError).toBeGreaterThan(90);
    expect(r.accepted).toBe(false);
  });
});
