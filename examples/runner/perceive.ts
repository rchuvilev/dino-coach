import type { Frame } from "../../src/perception/readers.js";
import { GROUND, PLAYER_X, PLAYER_W, W } from "./world.js";

/**
 * Read the world from PIXELS only.
 *
 * This is the part the Chrome Dino attempt never tested: there I read
 * `Runner.instance_` directly, so the perception tier was bypassed entirely
 * and a broken reader (pointed at an always-empty array) went unnoticed.
 * Here the agent gets nothing but a frame.
 */

const DARK = 128;

function isDark(f: Frame, x: number, y: number): boolean {
  if (x < 0 || y < 0 || x >= f.width || y >= f.height) return false;
  return (f.data[(y * f.width + x) * 4] ?? 255) < DARK;
}

/**
 * Horizontal distance from the player's front edge to the nearest dark
 * column ahead. Scans the band ABOVE the ground line so the ground itself
 * cannot be mistaken for an obstacle.
 */
export function readGap(f: Frame): number {
  const from = PLAYER_X + PLAYER_W + 1;
  for (let x = from; x < W; x++) {
    for (let y = GROUND - 24; y < GROUND; y++) {
      if (isDark(f, x, y)) return x - from;
    }
  }
  return 999;
}

/**
 * Is the nearest obstacle a HIGH one (duck under) rather than ground-level?
 *
 * Decided by where the obstacle's dark pixels sit relative to the ground: a
 * high obstacle leaves a clear gap just above the ground line.
 */
export function readHigh(f: Frame): boolean {
  const from = PLAYER_X + PLAYER_W + 1;
  for (let x = from; x < W; x++) {
    let top = -1;
    let bottom = -1;
    for (let y = GROUND - 24; y < GROUND; y++) {
      if (isDark(f, x, y)) {
        if (top === -1) top = y;
        bottom = y;
      }
    }
    if (top !== -1) {
      // ground-level obstacles extend down to the ground line
      return bottom < GROUND - 3;
    }
  }
  return false;
}

/** Player height above the ground, from its topmost dark pixel. */
export function readPlayerY(f: Frame): number {
  const x = PLAYER_X + Math.floor(PLAYER_W / 2);
  for (let y = 0; y < GROUND; y++) {
    if (isDark(f, x, y)) {
      // standing height is 14px, so anything higher means airborne
      return Math.max(0, GROUND - 14 - y);
    }
  }
  return 0;
}

export interface Perceived {
  gap: number;
  high: boolean;
  playerY: number;
  airborne: boolean;
}

export function perceive(f: Frame): Perceived {
  const playerY = readPlayerY(f);
  return {
    gap: readGap(f),
    high: readHigh(f),
    playerY,
    airborne: playerY > 1,
  };
}
