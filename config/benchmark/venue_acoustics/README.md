# Venue Acoustics: RT60-Calibrated Reverb Scenarios

`venue_scenarios.yaml` benchmarks a piano render through four real,
RT60-cited acoustic spaces — a dry recording-studio control room, a
chamber/recital hall, a symphony hall, and a large cathedral — as a
**single-factor `Reverb` ablation**. It replaces the parent directory's
`reverb_sweep.yaml`, whose `wet_level`/`room_size` steps are arbitrary, with
values calibrated against **published RT60 (reverberation time) figures**
via a measured Schroeder backward-integration procedure. It is a companion
study to `config/benchmark/old_recording/` and
`config/benchmark/telephone_channel/` (the same real-world-grounding
methodology, reused here for a different effect family and a genuinely
different category of calibration problem — see Section 2).

**Read this document before drawing any conclusion from a run of this
config.** In particular: RT60-matching calibrates decay **time** only, not
spectral character or spatial impression — this is a stylized, one-parameter
ablation of "how long does the room ring," not a physically accurate room
simulation. See Section 5 for the full confound disclosure.

---

## 1. Scope

**Modelled:** reverberant decay time (`Reverb`) only — genuinely one-factor,
unlike `old_recording`'s holistic per-era chains or `telephone_channel`'s
bandpass+leveling+saturation chains.

**Explicitly NOT modelled:**

- Early-reflection pattern — Freeverb-family algorithms (including
  `pedalboard.Reverb`) generate a diffuse, comb-filter/all-pass tail with no
  distinct discrete early-reflection cluster the way a real room's
  first-order geometric reflections would produce.
- Frequency-dependent absorption — real rooms absorb high frequencies faster
  than low; a single global `damping` value is a crude proxy for what is, in
  a real space, a frequency-dependent phenomenon governed by wall/ceiling
  material.
- Room shape/size beyond what `room_size` crudely encodes — `room_size` is
  an algorithm tuning knob (Section 2), not a physical volume or dimension.
- Source/receiver distance and directivity — no direct/reverberant ratio
  varies with a modelled listener position; only the `wet_level`/`dry_level`
  mix (fixed per Section 6) stands in for this.
- Any material-specific coloration (wood paneling vs. stone vs. plaster vs.
  acoustic tile) — none of these have a distinct spectral signature in this
  model beyond what `damping` crudely applies.

---

## 2. The central problem: `pedalboard.Reverb` is not RT60-parameterized

`pedalboard.Reverb` is a Freeverb-derived comb-filter/all-pass feedback
network. Its `room_size` (0-1) and `damping` (0-1) are **algorithm tuning
knobs, not physical units** — there is no formula in the plugin's
documentation mapping a `(room_size, damping)` pair to a decay time in
seconds (`REFERENCE.md` §2.6). This is architecturally the same category of
problem `old_recording` solved for its filter cutoffs (nominal cutoff vs.
realized -3 dB point, `old_recording/README.md` §5) and `telephone_channel`
reused for its cascaded bandpass cutoffs — just for a different plugin and a
different measured quantity (decay time, not frequency response).

**No formula exists, so every `room_size`/`damping` pair used by this config
was measured, not derived.** The procedure:

1. Render a unit impulse (a single sample at 1.0, all others 0.0) through
   `Pedalboard([Reverb(room_size=..., damping=..., wet_level=1.0,
   dry_level=0.0, width=1.0, freeze_mode=0.0)])` at 44100 Hz — `wet_level=1.0`
   / `dry_level=0.0` isolates the reverb tail from the impulse's own direct
   pass-through, matching the plan's calibration methodology
   (`REFERENCE.md` §2.6).
2. Measure RT60 from the rendered impulse response via **Schroeder backward
   integration**: `10*log10(cumsum(x[::-1]**2)[::-1] / total_energy)`, fit
   the **-5 dB to -35 dB** decay slope with a linear regression, and
   extrapolate to -60 dB (`RT60 = -60 / slope`) — the standard technique
   when a full 60 dB decay isn't the cleanest region of a short render.
3. Grid-search `room_size` (0.0-1.0, step 0.05, refined finer near each
   target) × `damping` (0.0-1.0, step 0.1, refined finer near each target)
   to build a `(room_size, damping) -> RT60` lookup, and pick the pair
   closest to each target.
4. Re-verify the chosen pair against real FluidSynth-rendered piano audio
   (not just the calibration impulse) — Section 5.2.

`freeze_mode` was also probed (values 0.0-1.0 at the grid's longest-RT60
corner, `room_size=1.0, damping=0.0`): below 0.5 it has no measurable effect
on decay time; at 0.5 and above it locks the reverb into a non-decaying hold
state (RT60 measurement does not converge — Schroeder curve stays flat).
`freeze_mode` is therefore **not used** by any condition in this config;
every RT60 target below turned out to be reachable through `room_size`/
`damping` alone.

---

## 3. RT60 targets and grounding

| Condition | RT60 target | Grounding |
|---|---|---|
| `baseline` | — (no reverb / anechoic) | matches `reverb_sweep.yaml`'s own baseline convention |
| `studio_dry` | ~0.35 s | recording-studio control-room target, 0.2-0.4 s (`REFERENCE.md` §2.2) |
| `recital_hall` | ~1.6 s | chamber/recital hall target, 1.4-1.8 s (`REFERENCE.md` §2.3) |
| `symphony_hall` | ~2.0 s | symphony hall occupied target, 1.8-2.2 s; "best halls" cluster 1.8-2.0 s (`REFERENCE.md` §2.3, Beranek) |
| `cathedral` | ~8.0 s | large cathedral range 7-11 s; St. Paul's Cathedral occupied measurement anchor, 7.8 s at 500 Hz (`REFERENCE.md` §2.4) |

---

## 4. Calibration results

### 4.1 Impulse-based lookup (measurement of record)

Measured on installed `pedalboard` 0.9.24, sample rate 44100 Hz,
`wet_level=1.0` / `dry_level=0.0` (isolation mix), 16 s render length, tested
against a stated **±10% tolerance** of each RT60 target:

| Condition | `room_size` | `damping` | Measured RT60 | Target | Error | Within ±10%? |
|---|---|---|---|---|---|---|
| `studio_dry` | 0.0 | 1.0 | 0.517 s | 0.35 s | +47.7% | **No — algorithm floor, see 4.2** |
| `recital_hall` | 0.6 | 0.0 | 1.600 s | 1.6 s | +0.02% | Yes |
| `symphony_hall` | 0.69 | 0.0 | 2.007 s | 2.0 s | +0.33% | Yes |
| `cathedral` | 0.972 | 0.0 | 8.019 s | 8.0 s | +0.24% | Yes |

`recital_hall`, `symphony_hall`, and `cathedral` all land within a fraction
of a percent of their targets. `studio_dry` does not, honestly reported
below.

### 4.2 `studio_dry`: the target is below this algorithm's floor

The full calibration grid's **global minimum** RT60 — swept across every
`room_size` in `[0.0, 1.0]` and `damping` in `[0.0, 1.0]`, including the
boundary values — is **0.517 s**, at `room_size=0.0, damping=1.0`. Both
parameters are already at the extreme of their valid range (`pedalboard`
raises `ValueError` outside `[0.0, 1.0]` for both); there is no
lower-RT60 direction left to search. Diagnostic decay curve at this pair
(Schroeder dB relative to total energy, from the impulse):

| Time | -5.5 dB | -22.0 dB | -37.6 dB | -52.3 dB | -60 dB (fit) |
|---|---|---|---|---|---|
| Elapsed | 0.05 s | 0.20 s | 0.35 s | 0.40 s | 0.517 s |

This is a genuine, reproducible **algorithm floor**, not a measurement
artifact: Freeverb-family algorithms apply a non-zero base feedback
coefficient to their comb filters even at `room_size=0` (this offset is
part of the classic Freeverb formula — `room_size=0` does not mean "zero
feedback," only "minimum feedback"), so the shortest achievable decay is
bounded below by that base coefficient and the comb delay-line lengths, not
by zero.

**This config uses `room_size=0.0, damping=1.0` (RT60 0.517 s) for
`studio_dry`** — the closest achievable value — and discloses the ~48%
deviation from the 0.35 s target here rather than forcing a value the
algorithm cannot reach. Practically, 0.517 s is still solidly inside the
"dry control room" character (well under 1 s, clearly distinguishable from
every other condition in this study), so the condition remains usable as
the driest point on the ladder; it should not be cited as literally
"RT60 0.35 s."

### 4.3 The opposite question: was `cathedral`'s 8 s target even reachable?

Yes, directly, without needing `freeze_mode` or any other workaround — the
plain `room_size`/`damping` grid reaches as high as **11.2 s** at
`room_size=1.0, damping=0.0`, comfortably past the 8 s target, and the
calibration search found `room_size=0.972, damping=0.0` landing at 8.019 s,
well inside tolerance (Section 4.1). Unlike `studio_dry`'s floor, there was
no ceiling problem on the long end of this grid.

One implementation note for anyone re-running this calibration: near
`room_size >= 0.95`, RT60 becomes **extremely sensitive to `damping`** — at
`room_size=0.97`, moving `damping` from 0.0 to just 0.05 drops RT60 from
7.86 s to 6.82 s (a 13% swing for a 0.05 step). This is expected behavior,
not a bug: RT60 in a feedback-comb algorithm scales with the number of
feedback loops needed to reach -60 dB, and at very long decay times that
loop count is large, so a small per-loop attenuation change compounds over
many more iterations than it would for a short decay. All four non-baseline
conditions in this config use `damping=0.0` except `studio_dry` (which
instead uses `damping=1.0`, its calibration direction), avoiding this
sensitive region for `recital_hall`/`symphony_hall`/`cathedral`.

### 4.4 Production wet/dry mix does not break the calibration

The calibration above uses `wet_level=1.0`/`dry_level=0.0` specifically to
**isolate** the reverb tail for measurement — that is a measurement
convenience, not a realistic mix for an audibly-reverberant recording (a
recording that is 100% wet with 0% dry signal does not sound like "a piano
in a room," it sounds like the room alone). `venue_scenarios.yaml` instead
uses `wet_level=0.33`, `dry_level=0.4`, `width=1.0` for every non-baseline
condition — `pedalboard.Reverb`'s own class defaults, used here as a
representative "audibly reverberant but still clearly a piano" production
mix, fixed across all four conditions so `room_size`/`damping` remain the
only manipulated variable per Section 1's one-factor design.

RT60 is, by definition, a property of the **wet signal's own decay rate**,
not of how much of it is mixed in — changing `wet_level`/`dry_level` should
not change RT60 for a fixed `room_size`/`damping`. This was verified, not
assumed: re-running the impulse calibration at the production mix
(`wet_level=0.33`, `dry_level=0.4`) reproduces each RT60 within a few
percent of the isolation-mix measurement:

| Condition | Isolation-mix RT60 (wet=1.0/dry=0.0) | Production-mix RT60 (wet=0.33/dry=0.4) | Difference |
|---|---|---|---|
| `studio_dry` | 0.517 s | 0.471 s | -8.9% |
| `recital_hall` | 1.600 s | 1.591 s | -0.6% |
| `symphony_hall` | 2.007 s | 1.988 s | -0.9% |
| `cathedral` | 8.019 s | 7.974 s | -0.6% |

Three of four conditions move by under 1%. `studio_dry`'s larger relative
swing (-8.9%) is a small **absolute** difference (0.046 s) that looks large
only because its own RT60 is so short — adding a non-zero `dry_level` puts
a single non-decaying direct-pass sample at the very start of the impulse
response, which shifts the Schroeder curve's reference energy measurably
for the fastest-decaying condition but negligibly for the other three. The
assumption holds well enough in practice: none of the four conditions'
RT60 moves outside its own already-reported tolerance band (Section 4.1)
because of the wet/dry mix choice.

### 4.5 Real-piano re-verification

A pink-noise-burst or unit-impulse calibration signal has different
spectral content from a struck piano string — the calibration in Section
4.1-4.4 was re-checked against real FluidSynth-rendered piano audio
(`/usr/share/sounds/sf2/default-GM.sf2`, the same SoundFont this config
uses), not just the synthetic impulse. Two variants were run.

**(a) A single short piano note.** A 0.05 s C4 note (velocity 100) was
rendered via `fluidsynth` (`-R0 -C0`, FluidSynth's own built-in
reverb/chorus disabled so only this config's `Reverb` is under test — real
`sonitra` renders do **not** disable FluidSynth's built-in effects, so this
is a calibration-only deviation, noted for anyone reproducing it), then
passed through each chosen `Reverb` pair at the isolation mix
(`wet_level=1.0`/`dry_level=0.0`) with 20 s of silence appended, and RT60
was re-measured via the same Schroeder method on the resulting tail:

| Condition | Piano-note RT60 | vs. impulse-calibrated | vs. original target |
|---|---|---|---|
| `studio_dry` | 0.647 s | +25.2% | +84.9% |
| `recital_hall` | 1.448 s | -9.5% | -9.5% |
| `symphony_hall` | 1.779 s | -11.1% | -11.4% |
| `cathedral` | 7.878 s | -1.5% | -1.8% |

`recital_hall` lands just inside the ±10% tolerance and `cathedral` well
inside it; `symphony_hall` lands just outside (-11%); `studio_dry` is
furthest off in absolute percentage terms but, as in Section 4.2, this is
against a target the algorithm cannot reach in the first place — the more
meaningful comparison is against the achievable calibrated value (0.517 s),
against which the real-note result is +25.2%, not +84.9%.

**(b) The full `piano1.mid` corpus file, isolated at its final note.**
`/workspace/corpus/test/midi/piano1.mid` (the file specified for this
check) was rendered in full (16.0 s of legato, back-to-back notes with no
rests, `fluidsynth -R0 -C0`), then everything before the score's final note
(a 2 s held note, 14.0-16.0 s) was zeroed out and 20 s of silence appended,
isolating that note's own attack and natural release as the excitation.
Measured RT60 on the resulting tail: `studio_dry` 3.649 s, `recital_hall`
3.785 s, `symphony_hall` 3.847 s, `cathedral` 7.599 s.

These numbers are **not** used as the primary re-verification, and the
reason is itself the finding worth recording: `piano1.mid` has no natural
silence anywhere except after its very last note, and that last note is
held for a full 2 seconds before release. A 2-second continuous excitation
keeps re-injecting energy into the reverb's comb-filter memory throughout
its own sustain, so by the time it releases, three of the four conditions
(`studio_dry`, `recital_hall`, `symphony_hall` — every RT60 shorter than or
comparable to that 2 s sustain) show a measured decay dominated by "how
long it takes the 2-second-long excitation's own accumulated energy to
dissipate," not by the calibrated `Reverb` parameter itself — which is why
all three cluster at a similar 3.6-3.9 s regardless of their very different
true RT60 (0.517 s / 1.600 s / 2.007 s). `cathedral`, whose true RT60
(8.0 s) is already several times longer than the 2 s sustain, is
unaffected by this and lands close to its calibrated value (7.599 s,
-1.8% from calibration) even using this contaminated excitation — precisely
because for `cathedral` the sustain-driven buildup has already decayed away
well before the algorithm's own much-longer tail becomes the dominant term.
The short-note check in (a) above deliberately avoids this by using an
excitation much shorter than any of the four RT60s, and is the
re-verification of record for `studio_dry`/`recital_hall`/`symphony_hall`.

**Conclusion:** the impulse-based calibration (Section 4.1) remains the
authoritative source for every `room_size`/`damping` pair in this config.
The real-piano check confirms the algorithm decays in the expected
direction and rough magnitude on genuine struck-string spectral content,
most cleanly for `cathedral` and `recital_hall` (both within stated
tolerance against a real note), with `symphony_hall` marginally outside
tolerance (-11%) and `studio_dry` — already established as unreachable
against its literal target — showing the same directional agreement
against its calibrated (not target) value.

---

## 5. Confounds to disclose

### 5.1 Algorithm fidelity

RT60-matching calibrates decay **time** only. It does not calibrate
spectral character, spatial/stereo impression, early-reflection density, or
any other perceptual dimension of "what a room sounds like." A listener who
knows what a real cathedral sounds like will very likely still hear
`pedalboard.Reverb`'s Freeverb-derived comb/all-pass character as
synthetic, even at the "correct" RT60. State plainly: **this is not a
physically accurate room simulation.** This is new territory for the
project — `old_recording`'s own filter-cascade calibration (its closest
analogue) never claimed physical room accuracy either, since it was
calibrating frequency response, not a spatial effect.

### 5.2 Onset-masking risk at long RT60

An 8-second `cathedral` tail can overlap multiple note onsets, especially
in dense piano passages. This is a plausible systematic bias on note-onset
F1 that is a property of the **measurement** interacting with reverb
tails — not necessarily "cathedral acoustics are hard for AMT" in some
deeper sense. If `cathedral`'s onset F1 delta looks disproportionately
large relative to `symphony_hall`'s (2 s RT60, no comparable overlap risk),
first check whether the drop tracks note density/tempo (a measurement
artifact) before attributing it to venue acoustics generally. This
scenario's own real-piano re-verification (Section 4.5) is itself an
instance of the same interaction — the 2-second-sustain contamination for
the shorter conditions is a smaller-scale version of exactly this
phenomenon.

### 5.3 Peak normalisation

Same pipeline-wide confound `old_recording` §4 and `telephone_channel` §4
already document: `normalisation.enabled: true`, `mode: peak`,
`target_db: -1.0` runs **after** the `Reverb` effect (this config leaves
`normalisation.pre_effects` at its default, `false` — see the comment in
`venue_scenarios.yaml`; `pre_effects` is a single-stage switch, not
additive, so post-effects normalisation only stays on because pre-effects
normalisation is off), so every condition reaches the transcriber at
-1 dBFS peak regardless of how much reverberant tail energy the effect
added. **These conditions do not model a real venue's actual received
loudness** — a cathedral recording is not literally renormalized to match
a dry studio take's peak level in real life, but it is here, by the
pipeline's design. Only the **decay-time character**, not the absolute
level, is modelled. If a future study wants to measure the level effect
specifically, it needs a config with post-effects normalisation disabled
or reworked — out of scope for this file.

This is not only a modelling confound but a functional necessity for this
particular config: `Reverb`'s added tail energy pushes several conditions'
peak level above `quality_gates.clip_threshold: 1.0` before renormalisation
(observed as high as ~1.75 for `cathedral` on some files) — without
post-effects normalisation active, those renders are rejected by the
quality gate as clipped and the condition silently loses files. Keeping
`pre_effects: false` (so post-effects normalisation runs) is required for
every condition, especially `cathedral`, to reliably produce output at all.

---

## 6. Interpretation constraints

- **Genuinely one-factor**, unlike `old_recording` and `telephone_channel`.
  Each non-baseline condition here moves exactly two coupled parameters
  (`room_size`, `damping`) that jointly determine one thing — RT60. Unlike
  those two studies, a per-condition F1 delta here **can** be attributed to
  "reverb decay time" without needing to disclaim an inability to decompose
  multiple simultaneous factors.
- **`studio_dry` is a floor, not a citation match.** Do not report
  `studio_dry`'s RT60 as "0.35 s" — report it as "0.517 s, the shortest
  decay time this Reverb algorithm can produce," per Section 4.2.
- **Not a severity ladder in the `old_recording` sense**, but *is* a
  monotonic RT60 ladder — `studio_dry` < `recital_hall` < `symphony_hall` <
  `cathedral` is a legitimate ordering to report and plot against, unlike
  `telephone_channel`'s three channels (which are different real-world
  contexts, not points on one axis).
- **Run the onset-masking check (§5.2)** before writing up `cathedral`
  specifically as evidence about "how hard very reverberant recordings are"
  — rule out tail-overlap as the driver first.

---

## 7. Usage

```bash
sonitra benchmark \
  --config config/benchmark/venue_acoustics/venue_scenarios.yaml \
  --dataset <name>
```

Smoke test on a small subset (use `--seed` for a reproducible sample):

```bash
sonitra benchmark \
  --config config/benchmark/venue_acoustics/venue_scenarios.yaml \
  --dataset test --limit 2 --seed 0
```

Results land under `corpus/<name>/benchmark/` (`benchmark_results.jsonl`,
`summary.json`, per-condition transcriptions) — see the parent
`config/benchmark/README.md` for the general output layout and the resume
mechanism.
