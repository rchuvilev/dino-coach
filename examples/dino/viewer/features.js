/**
 * Feature extraction for the jump-outcome model.
 *
 * The previous 4-feature vector (gap, speed, width, next) could only learn
 * "this SITUATION is dangerous". It could not learn "this DECISION was
 * wrong", because nothing about the decision was in the input - the same
 * situation jumped early and jumped late produced identical features with
 * opposite labels, which is unlearnable by construction.
 *
 * This set separates the three things that matter:
 *   WORLD     what was in front of us      speed, gap, width, type, next
 *   DECISION  what we chose to do          arc velocity, ttc at takeoff
 *   DERIVED   physics the agent cannot see time-to-collision, clearance
 *
 * Time-to-collision is included because it was measured 2.8x more predictive
 * than raw pixel gap (median 1670 vs 602 over 11 runs per arm): the jump arc
 * is time-based, so at speed 13 the same pixel gap is half the time.
 */

// Scales keep every axis in roughly [0,2] so no single one dominates the
// distance metric or the first layer's gradients.
const S = {
  gap: 60,
  speed: 6,
  width: 50,
  next: 120,
  ttc: 12,
  arc: 12,
  clearance: 60,
};

export const FEATURE_NAMES = [
  "gap",            // pixels to the obstacle at takeoff
  "speed",          // world speed, which sets how fast everything closes
  "width",          // obstacle width - wide cacti need more air time
  "next",           // distance to the obstacle AFTER this one
  "ttc",            // frames to impact: gap/speed, the physical quantity
  "ttcNext",        // frames to the next obstacle
  "isHigh",         // bird vs cactus - needs ducking, not jumping
  "isWide",         // clustered cactus
  "arcVel",         // WHICH JUMP we chose: the decision, not the situation
  "airTime",        // frames that arc stays above clearance height
  "reach",          // horizontal distance covered while high enough
  "margin",         // reach minus what is needed to clear - the crux
];

/**
 * @param {object} s situation at takeoff
 *   {gap, speed, width, next, high, arcVel}
 */
export function featurize(s) {
  const gap = s.gap ?? 0;
  const speed = s.speed ?? 6;
  const width = s.width ?? 0;
  const next = s.next === null || s.next === undefined ? 999 : s.next;
  const arcVel = s.arcVel ?? 10;

  const ttc = speed > 0 ? gap / speed : 99;
  const ttcNext = speed > 0 && next < 900 ? next / speed : 99;

  // Ballistic estimate of the jump we actually chose. Measured arcs:
  // velocity 8 -> 57px high / 28 frames, 12 -> 126px / 42 frames, so air
  // time scales roughly linearly with initial velocity under fixed gravity.
  const airTime = arcVel * 3.2;
  // horizontal ground covered while above clearance height
  const reach = airTime * speed * 0.55;
  // what must be covered: the obstacle plus the dino's own body
  const needed = width + 44;
  // THE CRUX: positive margin means this jump can physically clear it.
  const margin = reach - needed;

  return [
    gap / S.gap,
    speed / S.speed,
    width / S.width,
    Math.min(next, 400) / S.next,
    Math.min(ttc, 40) / S.ttc,
    Math.min(ttcNext, 40) / S.ttc,
    s.high ? 1 : 0,
    s.wide || width >= 50 ? 1 : 0,
    arcVel / S.arc,
    airTime / 40,
    reach / 200,
    Math.max(-2, Math.min(2, margin / S.clearance)),
  ];
}

export const FEATURE_COUNT = FEATURE_NAMES.length;

/** Readable breakdown, for the diagnosis panel and for debugging a decision. */
export function explain(s) {
  const f = featurize(s);
  return FEATURE_NAMES.map((n, i) => `${n}=${f[i].toFixed(2)}`).join(" ");
}
