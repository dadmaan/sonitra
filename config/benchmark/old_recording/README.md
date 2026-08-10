# Old Recording: Vintage Recording Chains

`vintage_scenarios.yaml` benchmarks three historically distinct recording
chains — 78rpm shellac disc, early reel-to-reel tape, and AM radio broadcast —
each at two severities, against a common baseline. It is a companion study to
the single-axis sweeps in the parent `config/benchmark/` directory
(`reverb_sweep`, `compression_sweep`, `distortion_sweep`), but unlike those it
is a **holistic era emulation**, not a one-factor ablation — see Section 6.

**Read this document before drawing any conclusion from a run of this
config.** In particular: **this is a bandwidth-and-dynamics ablation, not a
measurement of "how hard vintage recordings are for AMT."** A small F1 delta
is a legitimate, expected result here, and must **not** be written up as
"vintage recordings are not hard for AMT" — additive noise (surface noise,
tape hiss, mains hum) is not modelled by this config and is plausibly a
larger driver of transcription failure than anything modelled here.

---

## 1. Scope

**Modelled**, per condition:

- Bandwidth limiting via cascaded highpass/lowpass filter pairs (`pedalboard`
  `HighpassFilter`/`LowpassFilter`, two stages per band — see Section 5).
- Presence/resonance coloration via a single `PeakFilter` (shellac "presence
  bump" around 2.5 kHz, or tape head-bump resonance in the 50-100 Hz range).
- Saturation via `Distortion`.
- Dynamics / broadcast-style level control via `Compressor`.

**Explicitly NOT modelled**:

- Surface noise, crackle, hiss, mains hum — not modelled. This is plausibly
  the larger driver of transcription failure; see the framing note above.
- Wow/flutter (speed instability) — not modelled; not expressible as a
  stationary filter.
- Reduced medium level / dynamic range — cancelled by post-effects peak
  normalisation, which runs after the whole chain. See Section 4.
- The RIAA curve — a vinyl-LP (post-1954) mastering/playback convention, not
  shellac-78; transparent on a calibrated chain and not itself a source of
  vintage coloration.
- Reverb — electrical-era studios were comparatively dry; none of the three
  chains here are differentiated by room ambience.
- Wire recorders and the pre-1926 acoustic-horn recording era — both out of
  scope.

---

## 2. Historical grounding

### 2.1 The three eras

- **78rpm shellac disc, electrical era (~1925-1950).** The 1925 electromechanical
  cutterhead flattened the response compared to the earlier acoustic-horn
  process, but could not reliably extend past ~5 kHz; pre-1937 discs have
  little content above ~10 kHz. Shellac itself contributes broadband surface
  noise and impulsive crackle — a medium property, not a filter, and not
  modelled here.
- **Early reel-to-reel tape, consumer mono decks (1950s-60s, 3.75-7.5 ips).**
  Consumer decks (as opposed to 15 ips professional machines) have a narrower
  top end and worse signal-to-noise (~50-55 dB vs ~60 dB for pro machines).
  They also exhibit head-bump resonance in the 50-100 Hz range (Section 2.3).
  Wow/flutter is measured per DIN 45507; consumer decks are spec'd to
  ±0.2-0.4% vs ~0.02-0.04% for pro machines, but is not modelled here since it
  needs variable-rate resampling rather than a stationary filter.
- **AM radio broadcast, pre-NRSC golden-age radio (1930s-50s).** NRSC-1/2
  only standardized AM at a 10 kHz audio bandwidth in 1986-90, precisely
  because no consistent limit existed before that; even post-standard, many
  stations run 5-6 kHz and most receivers band-limit to ≤5 kHz. Period
  receivers were narrower still (~3.5-5 kHz). Heavy broadcast
  compression/limiting is characteristic of the era; mains hum is not modelled
  here.

### 2.2 Why AM has no highpass at all

The 300-3400 Hz band commonly associated with "radio voice" is the
**telephone** band, not the AM broadcast band. AM was constrained by channel
selectivity and receiver bandwidth at the **top** end (~3.5-5 kHz); the
transmit chain was not rolled off at 300 Hz, and low-frequency response
extended down toward 50-100 Hz. Both `am_bandlimit_*` conditions therefore
carry **no highpass at all** — the AM independent variables are top-end
bandwidth and broadcast dynamics only. (A period receiver's small loudspeaker
does roll off low frequencies, but that is a transducer artefact outside the
recording/transmission chain being modelled, and including it would
reintroduce the reduced-level/dynamic-range confound described in Section 4.)

### 2.3 Why the shellac highpass is 80/120 Hz, not ~250 Hz

Some period literature cites "little response below 200 Hz" for 78rpm discs —
but that figure belongs to the **pre-1926 acoustic-horn** era, which Section 1
explicitly excludes from this config's scope. This config models the
**1925-1950 electrical** era, where the electromechanical cutterhead extended
low-frequency response considerably further down. A 250 Hz highpass would
misrepresent the electrical era as the acoustic one; `shellac_bandlimit_mild`
and `shellac_bandlimit_pronounced` instead target 80 Hz and 120 Hz -3 dB
points respectively.

### 2.4 Tape head-bump direction

Head-bump resonance frequency moves **down**, not up, as tape speed decreases
(bump frequency is proportional to tape speed for a fixed reproduce-head gap
geometry): `tape_bandlimit_mild` (7.5 ips) sits at 100 Hz, and
`tape_bandlimit_pronounced` (3.75 ips) sits at 60 Hz — both inside the
50-100 Hz range cited in Section 2.1. The head bump is modelled with a
`PeakFilter` (slot 5), not a `LowShelfFilter`: it is a broad low-frequency
**resonance** caused by reproduce-head gap geometry, not a shelf that would
also boost everything below the corner, including rumble.

---

## 3. Piano-fundamental constraint

The corpus is solo piano: fundamentals run from A0 = 27.5 Hz up to
C8 = 4186 Hz, with C4 = 261.6 Hz sitting squarely inside the range any
"vintage" highpass would touch. Aggressive highpassing therefore removes the
**fundamental** of a large part of the keyboard — any resulting F1 drop in
that register is an artefact of the filter, not evidence about the recording
era, unless reported as such.

Measured attenuation (dB) of each highpass setting retained in this config, on
piano fundamentals, through the 2-stage cascade:

| Condition (-3 dB point) | A0 27.5 | C1 32.7 | C2 65.4 | E2 82.4 | C3 130.8 | C4 261.6 |
|---|---|---|---|---|---|---|
| `tape_bandlimit_mild` (40 Hz) | -5.5 | -4.2 | -1.3 | -0.8 | -0.3 | -0.1 |
| `tape_bandlimit_pronounced` (50 Hz) | -7.5 | -5.8 | -1.9 | -1.2 | -0.5 | -0.1 |
| `shellac_bandlimit_mild` (80 Hz) | -13.1 | -10.8 | -4.2 | -2.9 | -1.2 | -0.3 |
| `shellac_bandlimit_pronounced` (120 Hz) | -18.9 | -16.3 | -7.6 | -5.5 | -2.6 | -0.7 |

For reference, a *rejected* highpass considered during drafting (two
`HighpassFilter` stages at a 300 Hz nominal cutoff, the "little below 200 Hz"
acoustic-era figure misapplied to AM) attenuated A0 by -41.6 dB and C4 by
-7.3 dB — attenuation severe enough to dominate the whole study on its own.
That measurement is the concrete reason the AM highpass was dropped entirely
(Section 2.2) rather than tuned down.

**Any F1 loss in the lowest two octaves under `shellac_bandlimit_*` is
attributable to the filter itself and must be reported as such**, not
attributed to "how hard vintage shellac recordings are."

---

## 4. Confound disclosure: post-effects normalisation

`normalisation.pre_effects: true` is set in this config (departing from
`config/benchmark/effects_combinations.yaml`, which leaves it `false`) so that
`Distortion` drive and `Compressor` threshold behave deterministically instead
of depending on whatever level FluidSynth happened to produce for a given
file. This is a correctness fix, not the confound.

The confound is that **post-effects peak normalisation stays on**
(`normalisation.enabled: true`, `mode: peak`, `target_db: -1.0`), and it runs
*after* the entire pedalboard effects chain. That means every condition
reaches the transcriber at -1 dBFS peak, regardless of how much energy the
filters removed — the shellac conditions' +3/+5 dB `PeakFilter` boosts are
therefore effectively small *cuts* of everything else in the signal, once
renormalised.

**These conditions explicitly do not model vintage media's reduced absolute
level or reduced dynamic range.** That reduction is a real property of
playback from a physically degraded medium, but it is cancelled here by
design so that transcription-relevant spectral/dynamics changes are not
confounded with a simple gain difference basic-pitch would trivially
compensate for. If a future study wants to measure the level/dynamic-range
effect specifically, it needs a config with post-effects normalisation
disabled or reworked — out of scope for this file.

---

## 5. Filter cascade calibration (worked example)

`pedalboard`'s `HighpassFilter`/`LowpassFilter` are first-order (6 dB/octave),
-3 dB at their nominal `cutoff_frequency_hz`, and expose no other parameter.
Real vintage bandwidth limiting (shellac groove/stylus mechanics, AM channel
selectivity) is steeper than a single first-order section, so every band in
this config cascades **two** first-order instances (slots 0+1 for highpass,
2+3 for lowpass) rather than adding a new plugin type.

Two identical cascaded first-order sections are **-6 dB**, not -3 dB, at the
nominal cutoff — the -3 dB point moves. Solving
`(1 + (f / f_c)^2)^2 = 2` analytically gives the cascade's -3 dB point at:

- **lowpass:** `0.6436 * f_c`
- **highpass:** `1.5538 * f_c`

So a naive "highpass 300 Hz x2 + lowpass 3500 Hz x2" cascade is actually a
**467-2247 Hz** -3 dB passband, not 300-3500 Hz. Every cutoff used by this
config is therefore stated as a **-3 dB target** (the historical spec figure)
plus the **measured nominal per-stage cutoff** that lands the cascade's -3 dB
point on that target.

The analytic factor holds well for the highpass (cutoffs far from Nyquist,
matched to within 0.2 Hz), but pedalboard's filters are digital biquads, so
bilinear warping pulls the realised lowpass -3 dB point below the analytic
prediction as the cutoff approaches Nyquist — the error is a few percent at
3.5 kHz and over 40% at 12 kHz (sample rate 44100 Hz). The lowpass nominals
below were therefore solved numerically (binary search on the measured
cascade response), not derived from the `0.6436` factor alone.

Measured on installed `pedalboard` 0.9.24, sample rate 44100 Hz, every nominal
used in this config, verified through the actual 2-stage cascade:

| Slot pair | nominal per stage | realised -3 dB | target |
|---|---|---|---|
| HP | 25.7 Hz | 40.0 Hz | 40 Hz |
| HP | 32.1 Hz | 50.0 Hz | 50 Hz |
| HP | 51.4 Hz | 80.0 Hz | 80 Hz |
| HP | 77.1 Hz | 120.0 Hz | 120 Hz |
| LP | 5298 Hz | 3500 Hz | 3500 Hz |
| LP | 7370 Hz | 5000 Hz | 5000 Hz |
| LP | 11009 Hz | 8000 Hz | 8000 Hz |
| LP | 14898 Hz | 12000 Hz | 12000 Hz |

**Cost of this approach:** two first-order sections can match the -3 dB point
*or* the asymptotic slope, but not both — pushing the nominal up to place the
-3 dB point on target flattens the transition band relative to an idealized
12 dB/octave filter. Measured attenuation one octave above the -3 dB point for
the lowpass settings above: -9.2 dB (3500 Hz target), -10.2 dB (5000 Hz),
-15.2 dB (8000 Hz) — an ideal 12 dB/octave section would give -15 dB. The
near-band slope is therefore roughly 6-12 dB/octave, reaching a true
12 dB/octave only well into the stopband. A steeper single-instance option
(`LadderFilter` in `LPF24`/`HPF24` mode, a genuine 24 dB/octave) was
evaluated and rejected: it is not -3 dB at its own cutoff either (measured
`0.703 * f_c` lowpass / `1.955 * f_c` highpass), cannot reach a 12 kHz -3 dB
point at 44.1 kHz at any cutoff value, and is a nonlinear Moog emulation — a
poor fit for a study whose point is to isolate linear bandwidth limiting.

---

## 6. Interpretation constraints

Two constraints apply to every result produced by this config. State both
whenever reporting numbers from it.

- **Holistic, not one-factor.** Unlike the single-axis sibling studies in the
  parent directory (`reverb_sweep`, `compression_sweep`, `distortion_sweep`),
  each non-baseline condition here moves 3-6 parameters at once (bandwidth,
  presence/resonance, saturation, dynamics together). The study can say "the
  shellac-pronounced chain costs Y F1" but **cannot decompose** that delta
  into bandwidth vs presence vs saturation vs dynamics contributions
  individually. That is by design — these are era emulations, not per-factor
  ablations.
- **Severity labels are within-era only.** The mild-to-pronounced ladder is
  monotonic within each era, but severity is **not** comparable across eras:
  `shellac_bandlimit_mild`'s compressor is 6:1 @ -24 dB while
  `tape_bandlimit_pronounced`'s is only 3:1 @ -15 dB. Never aggregate "all
  mild vs all pronounced" across the three eras — compare within an era only
  (e.g. `shellac_bandlimit_mild` vs `shellac_bandlimit_pronounced`).

---

## 7. Usage

```bash
sonitra benchmark \
  --config config/benchmark/old_recording/vintage_scenarios.yaml \
  --dataset <name>
```

Smoke test on a small subset (use `--seed` for a reproducible sample):

```bash
sonitra benchmark \
  --config config/benchmark/old_recording/vintage_scenarios.yaml \
  --dataset test --limit 2 --seed 0
```

Results land under `corpus/<name>/benchmark/` (`benchmark_results.jsonl`,
`summary.json`, per-condition transcriptions) — see the parent
`config/benchmark/README.md` for the general output layout and the resume
mechanism.
