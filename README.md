# ai-mentat-blueprint

Tick-loop agent framework: **perception → priority rules → evolved decisions → actions**.

Code-first (a project is plain TypeScript data), UI-representable (the same
object is what an editor reads and writes), and every tier earns its place by
measurement rather than by assumption.

```
LLM operator (between runs)  propose readers/rules · explain traces · set run policy
        ↓ proposals scored like any mutation, never applied on trust
tier 3  live evaluation      the REAL scorer - runs a candidate and measures reward
tier 2  GA over genomes      replay prefilter (3.3ms/genome) + held-out scoring
tier 1  priority rules       first-match-wins + hysteresis + cooldowns
tier 0  perception           bar/presence readers, 8x8 feature crops
```

## Quick start

```sh
bun install
bun test test/        # 86 tests
bun run examples/demo.ts
```

## The five steps

| step | what you write | where |
|---|---|---|
| 1. state props | `props: { hp: { rating: 10, read: () => ... } }` | `src/project.ts` |
| 2. conditions | `conditions: { hp_dropping: (p, prev) => ... }` | `src/project.ts` |
| 3. actions + rules | `actions: {...}`, `rules: [{ when, then }]` | `src/actuate/` |
| 4. run and evolve | `evolve()` then `rankLive()` | `src/evolve/` |
| 5. inspect / tune | `toView()` / `applyView()` | `src/project.ts` |

Reward is **dense by construction**: it comes from prop deltas times your
ratings, every tick. Measured, that is the difference between a solvable task
(0.94) and an unsolvable one (0.06 for sparse terminal reward).

## Findings that shaped the design

Each of these is a measurement, not a preference. Most were bugs found by
*running* the framework after the unit tests were already green.

### Hysteresis is mandatory
Two rules with coupled effects (shooting costs hp, strafing restores it)
change action on **199/300 = 66% of ticks** without a commitment window, so
neither behaviour ever completes. An 8-tick commitment drops that to **8%**.
A negative-control test asserts the broken case still reproduces.

### Discounting is what links a reward to its cause
With reward delayed 3 ticks and 4 bystander actions firing just as often:

| γ | picks the true cause |
|---|---|
| 0.0 (a lookup table) | **0.01** — worse than the 0.20 chance baseline |
| 0.9 | **1.00** |

γ=0 *confidently credits bystanders*. Exposed to users as a horizon in
seconds (`gammaForHorizon`), not as a Greek letter.

### A rating can invert on a discontinuity
`enemyHp` rated −5 (lower is better) reset 0→100 on each kill, scoring
**−500 reward per kill** — 120 kills did worse than standing still. Hence
`countDirection` and `maxDelta` on every prop. **Any screenshot-derived prop
has this hazard**: respawns, scene changes, HUD occlusion, OCR misreads.

### A deterministic policy generates no data to learn from
It records only ONE action per condition-set, so counterfactual replay scores
every rival genome at **0% coverage**. Exploration took demo coverage
**51% → 100%**. It is not a tuning knob; it is what makes evolution possible.

### Replay scoring is off-policy and cannot rank policies
A genome scored **15.93 on held-out replay and earned 0 live.** `strafe`
looked excellent in states that existed only because the recording policy
also *shot*; an always-strafe policy never reaches them.

> A bandit-style reward table is valid only for policies close to the one
> that generated it.

So replay is a cheap **prefilter** and a live run is the **scorer**.

### A learned tier must prove it earns its place
In a setup rigged to favour learning (a hidden context bit inverting one
rule's payoff), a policy-gradient gate scored **0.965 against the fixed rule
set's 0.964** — recovering none of the 3.6% headroom. Not a broken learner:
the same gate scored **1.000** on a trivial control. The cause was signal
dilution — the ordering choice changed the outcome on only **13.6% of ticks**.

> Before adding a learned tier, measure how often its decision would change
> the outcome.

### Held-out scoring is not optional
Evolving against a short noisy run reached **3/5 correct rules and stalled**
while its training score looked healthy. Long runs reached 5/5. Selection
uses the training score; the winner is reported by held-out score;
`overfitting` flags train-up/holdout-down.

## The LLM operator

Used **between runs, never in the tick loop** — a vision call is ~200ms
against a ~7ms perception budget, and is nondeterministic.

| role | why it fits |
|---|---|
| author reader specs from screenshots | cold-start is where humans stall |
| propose rules / new conditions | a GA can only recombine what exists; it cannot invent a predicate |
| explain traces, flag divergence | diagnosis in prose is what rules and GAs cannot produce |
| set run length / stop criteria | the 60-vs-2000-tick result is a judgment call |

Every proposal enters as **a candidate scored on held-out traces**, exactly
like a mutation, with provenance preserved. A wrong suggestion costs a scored
run, not a silent regression — and you can measure whether the LLM's
candidates beat random mutation at all.

## Perception

An LLM proposes reader specs as *data*; `validateReader()` gates them against
labelled frames, reporting MAE **and** maxError, and **refuses to accept with
zero samples** rather than passing vacuously.

`readBar` takes the **last filled position** rather than counting filled
pixels, so a gloss line or segment gap cannot undercount, and requires a
majority of the cross-section so a 1px border cannot trip it.

## Actuation

`ActionSpec` is data (`key` / `keyDown` / `keyUp` / `click` / `move` / `wait`
/ `custom`) dispatched through an injected backend, so one project runs
against xdotool, a game API, an emulator or a mock unchanged. This is the
only layer with side effects, which is what makes dry runs, replay and a kill
switch possible. `release()` frees held keys — a `keyDown` without its
`keyUp` leaves the target stuck moving.

## Known gaps

- **No real capture backend is wired.** ffmpeg is the intended primitive.
- **The demo world is too simple** for the GA to beat a near-optimal seed, so
  evolution is proven *mechanically* (it discovers `ammo_low → reload`) but
  not yet *usefully* against a real target.
- **Reward hacking is untested.** An attempted probe returned an identical
  result in both arms, which proves the probe discriminated nothing rather
  than proving an absence. Redoing it needs an exploit paying more than the
  intended behaviour.

## Layout

```
src/project.ts        the one source of truth + validate() + toView/applyView
src/kernel/           tick loop, trace recorder, agent
src/rules/            subsumption evaluator
src/evolve/           genome, replay scoring, GA, live evaluation
src/perception/       frame readers + the LLM-proposal validation gate
src/actuate/          ActionSpec, backends, recording actuator
```
