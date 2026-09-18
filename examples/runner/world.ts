/**
 * A deterministic side-scrolling runner — the framework's first HONEST target.
 *
 * Design constraints, each one a reaction to a measured failure:
 *  - `step()` is explicit. No requestAnimationFrame, so a hidden tab cannot
 *    freeze it (the Chrome Dino test produced a plausible 8361 score from a
 *    completely frozen world).
 *  - Seeded PRNG, so two runs of the same policy are identical and a score
 *    difference means a policy difference, not luck.
 *  - `advanced()` asserts the SIMULATION moved, not just the score. The dino
 *    bug had distanceRan climbing to 23325 while the obstacle sat frozen at
 *    x=346 - the odometer lied.
 *  - Renders to a canvas-like surface, so perception can read PIXELS rather
 *    than a state object. Reading internal state proves nothing about the
 *    perception tier.
 */

export interface Obstacle {
  x: number;
  /** 0 = ground-level (jump over), 1 = high (duck under) */
  high: boolean;
  w: number;
  h: number;
}

export type RunnerAction = "jump" | "duck" | "run";

/** Mulberry32: tiny, seedable, good enough and fully reproducible. */
export function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export const W = 200;
export const H = 60;
export const GROUND = 46;
export const PLAYER_X = 24;
export const PLAYER_W = 8;

export class RunnerWorld {
  /** vertical offset above ground; 0 = standing */
  y = 0;
  vy = 0;
  ducking = false;
  duckLeft = 0;
  obstacles: Obstacle[] = [];
  distance = 0;
  crashed = false;
  ticks = 0;
  speed = 2;
  private rand: () => number;
  private nextSpawn: number;

  constructor(private seed = 1) {
    this.rand = rng(seed);
    // seed-dependent, because a hardcoded first spawn makes every episode
    // open identically and lets a policy overfit a fixed opening
    this.nextSpawn = 20 + Math.floor(this.rand() * 24);
  }

  reset(seed = this.seed): void {
    this.seed = seed;
    this.rand = rng(seed);
    this.y = 0;
    this.vy = 0;
    this.ducking = false;
    this.duckLeft = 0;
    this.obstacles = [];
    this.distance = 0;
    this.crashed = false;
    this.ticks = 0;
    this.speed = 2;
    this.nextSpawn = 20 + Math.floor(this.rand() * 24);
  }

  /** Player collision box, accounting for jump height and ducking. */
  private box(): { x: number; y: number; w: number; h: number } {
    const h = this.ducking ? 6 : 14;
    return { x: PLAYER_X, y: GROUND - h - this.y, w: PLAYER_W, h };
  }

  /**
   * Advance exactly one tick. Returns false once crashed.
   *
   * Deliberately integer-ish and frame-rate free: no delta time, so there is
   * no clock to fake and no way to advance the score without moving the world.
   */
  step(action: RunnerAction = "run"): boolean {
    if (this.crashed) return false;
    this.ticks++;

    // --- input
    if (action === "jump" && this.y === 0) this.vy = 5.2;
    if (action === "duck") {
      this.ducking = true;
      this.duckLeft = 6;
    }
    if (this.duckLeft > 0) {
      this.duckLeft--;
      if (this.duckLeft === 0) this.ducking = false;
    }

    // --- physics
    this.y += this.vy;
    this.vy -= 0.75;
    if (this.y <= 0) {
      this.y = 0;
      this.vy = 0;
    }

    // --- world
    this.speed = 2 + Math.min(2, this.ticks / 900);
    for (const o of this.obstacles) o.x -= this.speed;
    this.obstacles = this.obstacles.filter((o) => o.x + o.w > -4);

    if (--this.nextSpawn <= 0) {
      const high = this.rand() < 0.3;
      this.obstacles.push(
        high
          ? { x: W + 4, high: true, w: 10, h: 6 }
          : { x: W + 4, high: false, w: 6, h: 10 + Math.floor(this.rand() * 6) },
      );
      // gap scales with speed so the game stays playable as it speeds up
      this.nextSpawn = Math.round(46 + this.rand() * 40 + this.speed * 6);
    }

    // --- collision
    const p = this.box();
    for (const o of this.obstacles) {
      const oy = o.high ? GROUND - 22 : GROUND - o.h;
      if (
        p.x < o.x + o.w &&
        p.x + p.w > o.x &&
        p.y < oy + o.h &&
        p.y + p.h > oy
      ) {
        this.crashed = true;
        return false;
      }
    }

    this.distance++;
    return true;
  }

  /** Nearest obstacle ahead of the player, or null. */
  nearest(): Obstacle | null {
    let best: Obstacle | null = null;
    for (const o of this.obstacles) {
      const gap = o.x - (PLAYER_X + PLAYER_W);
      if (gap > -PLAYER_W && (best === null || o.x < best.x)) best = o;
    }
    return best;
  }

  /**
   * Ground-truth values, used ONLY to validate pixel readers - never as the
   * agent's input. If the agent read this, the perception tier would be
   * untested, which is exactly the mistake the Chrome Dino run made.
   */
  truth(): { gap: number; high: boolean; y: number } {
    const o = this.nearest();
    return {
      gap: o ? Math.round(o.x - (PLAYER_X + PLAYER_W)) : 999,
      high: o ? o.high : false,
      y: Math.round(this.y),
    };
  }
}

/**
 * Render to an RGBA buffer the perception tier can read as a Frame.
 *
 * Monochrome on purpose: the readers are colour-threshold based, and a
 * 2-colour scene makes a reader's failure obvious rather than marginal.
 */
export function render(w: RunnerWorld): { width: number; height: number; data: Uint8Array } {
  const data = new Uint8Array(W * H * 4);
  // white background, opaque
  for (let i = 0; i < W * H; i++) {
    data[i * 4] = 255;
    data[i * 4 + 1] = 255;
    data[i * 4 + 2] = 255;
    data[i * 4 + 3] = 255;
  }
  const put = (x: number, y: number) => {
    if (x < 0 || y < 0 || x >= W || y >= H) return;
    const i = (y * W + x) * 4;
    data[i] = 20;
    data[i + 1] = 20;
    data[i + 2] = 20;
  };
  const rect = (x: number, y: number, rw: number, rh: number) => {
    for (let dy = 0; dy < rh; dy++) for (let dx = 0; dx < rw; dx++) put(Math.round(x + dx), Math.round(y + dy));
  };

  // ground line
  rect(0, GROUND, W, 1);
  // player
  const ph = w.ducking ? 6 : 14;
  rect(PLAYER_X, GROUND - ph - w.y, PLAYER_W, ph);
  // obstacles
  for (const o of w.obstacles) {
    const oy = o.high ? GROUND - 22 : GROUND - o.h;
    rect(o.x, oy, o.w, o.h);
  }
  return { width: W, height: H, data };
}
