/**
 * Turning a screenshot into measured numbers.
 *
 * The design rule here is the important part: an LLM agent is used to AUTHOR
 * a reader once, not to run per tick. A vision model call costs ~200ms+ and
 * is nondeterministic; the per-tick budget is ~7ms for six perception heads.
 * So the LLM's output is CODE/CONFIG that runs deterministically forever
 * after, and can be unit-tested against saved frames.
 *
 * Three reader kinds, cheapest first. Prefer the cheapest that works:
 *  1. bar    - a health/ammo bar's filled fraction. Pure arithmetic.
 *  2. digits - a 7-segment/HUD number via template match on a small crop.
 *  3. head   - a trained KNN head over cached features (the teachable tier).
 */

export interface Region {
  x: number;
  y: number;
  w: number;
  h: number;
}

/** A frame as raw RGBA, which is what any capture backend can produce. */
export interface Frame {
  width: number;
  height: number;
  /** RGBA, 4 bytes per pixel, row-major */
  data: Uint8Array;
}

export function pixelAt(f: Frame, x: number, y: number): [number, number, number] {
  const i = (y * f.width + x) * 4;
  return [f.data[i] ?? 0, f.data[i + 1] ?? 0, f.data[i + 2] ?? 0];
}

/** Euclidean distance in RGB. Cheap and adequate for HUD colours. */
export function colorDistance(
  a: [number, number, number],
  b: [number, number, number],
): number {
  return Math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2);
}

export interface BarReaderSpec {
  kind: "bar";
  region: Region;
  /** the colour that counts as "filled" */
  fillColor: [number, number, number];
  /** how close a pixel must be to fillColor, 0-441 */
  tolerance?: number;
  /** scan along width (default) or height */
  axis?: "x" | "y";
  /** map the 0..1 fraction onto real units, e.g. 0..100 hp */
  scale?: number;
}

/**
 * Fraction of a bar that is filled, times scale.
 *
 * Reads the LAST filled position rather than counting filled pixels, because
 * a bar with a gradient, gloss or a segment separator has unfilled pixels
 * inside the filled span - counting them undercounts the value.
 */
export function readBar(f: Frame, spec: BarReaderSpec): number {
  const { region, fillColor, tolerance = 60, axis = "x", scale = 1 } = spec;
  const span = axis === "x" ? region.w : region.h;
  const cross = axis === "x" ? region.h : region.w;
  let lastFilled = -1;

  for (let i = 0; i < span; i++) {
    let hits = 0;
    for (let j = 0; j < cross; j++) {
      const x = axis === "x" ? region.x + i : region.x + j;
      const y = axis === "x" ? region.y + j : region.y + i;
      if (x < 0 || y < 0 || x >= f.width || y >= f.height) continue;
      if (colorDistance(pixelAt(f, x, y), fillColor) <= tolerance) hits++;
    }
    // majority of the cross-section must match, so a 1px border cannot trip it
    if (cross > 0 && hits * 2 > cross) lastFilled = i;
  }

  if (lastFilled < 0) return 0;
  return ((lastFilled + 1) / span) * scale;
}

export interface PresenceReaderSpec {
  kind: "presence";
  region: Region;
  color: [number, number, number];
  tolerance?: number;
  /** fraction of the region that must match to count as present */
  threshold?: number;
}

/** Fraction of a region matching a colour - for "is this icon showing?". */
export function readPresence(f: Frame, spec: PresenceReaderSpec): number {
  const { region, color, tolerance = 60 } = spec;
  let hits = 0;
  let total = 0;
  for (let dy = 0; dy < region.h; dy++) {
    for (let dx = 0; dx < region.w; dx++) {
      const x = region.x + dx;
      const y = region.y + dy;
      if (x < 0 || y < 0 || x >= f.width || y >= f.height) continue;
      total++;
      if (colorDistance(pixelAt(f, x, y), color) <= tolerance) hits++;
    }
  }
  return total > 0 ? hits / total : 0;
}

/**
 * A downscaled grayscale crop, used as the cached feature vector for a
 * trained head. Deliberately tiny: measured, training on a cached 128-d
 * projection took 1411ms versus 4813ms on 608-d, and that gap is what makes
 * the teachable UI feel interactive.
 */
export function features(f: Frame, region: Region, size = 8): Float32Array {
  const out = new Float32Array(size * size);
  for (let oy = 0; oy < size; oy++) {
    for (let ox = 0; ox < size; ox++) {
      const sx = region.x + Math.floor((ox / size) * region.w);
      const sy = region.y + Math.floor((oy / size) * region.h);
      if (sx < 0 || sy < 0 || sx >= f.width || sy >= f.height) continue;
      const [r, g, b] = pixelAt(f, sx, sy);
      out[oy * size + ox] = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
    }
  }
  return out;
}

export type ReaderSpec = BarReaderSpec | PresenceReaderSpec;

/** Run a reader spec. This is the deterministic, per-tick path. */
export function read(f: Frame, spec: ReaderSpec): number {
  switch (spec.kind) {
    case "bar":
      return readBar(f, spec);
    case "presence":
      return readPresence(f, spec);
  }
}

/**
 * What an LLM operator is asked to produce: reader specs as DATA.
 *
 * It sees a screenshot plus the user's prop names, and proposes regions and
 * colours. Its output is then VALIDATED against labelled frames before use -
 * a proposal is a hypothesis, never applied on trust. This is the same rule
 * the GA follows for proposed genomes.
 */
export interface ReaderProposal {
  prop: string;
  spec: ReaderSpec;
  /** the model's stated reason, kept for the UI so a human can audit it */
  rationale?: string;
}

export interface ValidationSample {
  frame: Frame;
  /** the true value a human (or the game's own API) says this frame shows */
  expected: number;
}

export interface ValidationResult {
  prop: string;
  /** mean absolute error over the samples */
  mae: number;
  /** worst single error */
  maxError: number;
  samples: number;
  /** false when error exceeds the caller's tolerance */
  accepted: boolean;
}

/**
 * Score a proposed reader against labelled frames.
 *
 * This is the gate that makes LLM-authored perception safe: a reader that
 * looks plausible but reads the wrong bar will show a large MAE here rather
 * than silently poisoning every reward downstream.
 */
export function validateReader(
  proposal: ReaderProposal,
  samples: ValidationSample[],
  tolerance: number,
): ValidationResult {
  let sum = 0;
  let worst = 0;
  for (const s of samples) {
    const got = read(s.frame, proposal.spec);
    const err = Math.abs(got - s.expected);
    sum += err;
    if (err > worst) worst = err;
  }
  const mae = samples.length > 0 ? sum / samples.length : Infinity;
  return {
    prop: proposal.prop,
    mae,
    maxError: worst,
    samples: samples.length,
    accepted: samples.length > 0 && mae <= tolerance,
  };
}
