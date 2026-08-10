# Rotary Speaker / Leslie: Chorale/Tremolo Character Scenarios

`rotary_scenarios.yaml` benchmarks a piano render through two canonical
Leslie rotary-speaker states — "chorale" (slow rotation) and "tremolo" (fast
rotation) — via `pedalboard.Chorus`, the closest available primitive in this
project's effect registry. It is a companion study to
`config/benchmark/old_recording/`, `config/benchmark/telephone_channel/`, and
`config/benchmark/venue_acoustics/` (the same real-world-grounding
methodology, reused here for a different effect family), and it replaces
nothing in the parent directory's abstract sweeps — there is no existing
`chorus_sweep.yaml` this supersedes.

**Read this document before drawing any conclusion from a run of this
config.** This scenario carries the **largest "stylized approximation"
disclaimer of the three grounded studies in this directory** — read Section 1
in full before treating any result here as evidence about "Leslie speaker
processing" in a strong sense.

---

## 1. Scope caveat — read this first

The Leslie rotary speaker is historically an **electric organ / electric
piano** effect: a Hammond organ, Rhodes electric piano, or Wurlitzer electric
piano driven through a Leslie 122/147-class cabinet, whose spinning horn
(treble) and drum (bass) produce the characteristic sound. This project's
corpus is **acoustic piano, rendered via MIDI + soundfont** — a Leslie was
never historically part of an acoustic piano's recording chain in the way
`old_recording`'s shellac/tape/AM chains genuinely were part of a real piano
recording's signal path.

**This scenario therefore cannot claim "historically authentic recording
chain" the way `old_recording`'s three eras can.** It is a **grounded,
citable processing chain, applied here as a stylized character ablation** —
"what does this piano render sound like run through a chorus-family effect
tuned to Leslie-like rotation rates" — not a claim about how acoustic piano
was actually recorded, or ever would have been. State this to yourself, and
to anyone reading a result from this config, with the same prominence given
to the confound disclosure in Section 3 below. Neither caveat is softened
relative to the other.

---

## 2. Effect chain design

Single slot — this is a genuinely one-factor ablation, like
`venue_acoustics` and unlike `old_recording`'s/`telephone_channel`'s
multi-parameter chains:

```
0: Chorus
```

**Modelled:** LFO-rate-driven amplitude/pitch modulation via `Chorus`,
tuned to Leslie chorale/tremolo rotation rates (`rate_hz`, cited — Section
4.1) plus depth/centre-delay/feedback/mix (not cited — Section 4.2).

**Explicitly NOT modelled:** anything beyond what a single `Chorus` instance
can express — see Section 3's confound disclosure for the specific list of
missing Leslie physics.

---

## 3. Confound disclosure: `Chorus` is not a Leslie simulation

State this as plainly as Section 1's historical-authenticity caveat — it is
not softened.

`pedalboard.Chorus` is an LFO-modulated delay-line effect: it mixes the dry
signal with a copy delayed by a small, sinusoidally-modulated amount, at a
single rate. A real Leslie rotary speaker is a physical rotating horn and
drum inside a cabinet, miked or heard in a room. `Chorus` models **none** of
the following, all of which are real, audible contributors to an actual
Leslie's sound:

- **No Doppler pitch shift.** As the horn rotates, the physical path length
  from the horn's mouth to a fixed microphone or listener changes
  continuously, producing a real, physically-caused pitch modulation (true
  Doppler shift). `Chorus`'s delay-line modulation produces a
  perceptually-similar-sounding but physically-unrelated effect — a varying
  delay time, not a varying propagation distance.
- **No correlated amplitude modulation from horn directivity.** A rotating
  horn's directional radiation pattern sweeps past the listener, producing
  amplitude modulation that is physically coupled to (and phase-locked with)
  the same rotation driving the pitch modulation above. `Chorus`'s `depth`/
  `mix` parameters are independent knobs with no such coupling.
- **No horn/drum crossover filter.** A real Leslie splits the signal between
  the treble horn and bass drum rotors via a crossover network, and the two
  rotors run at genuinely different, independently-motored speeds (Section
  4.1). A single `Chorus` instance has one rate for the entire signal.
- **No cabinet radiation pattern.** The Leslie cabinet's wood construction,
  louvered vents, and the specific geometry of horn-above/drum-below all
  shape the radiated sound in ways no delay-line effect touches.

`Chorus` is simply **the nearest available primitive in the current effect
registry** (`chain_builder.py`) — there is no rotary-speaker-specific plugin
available to this project. Treat every result from this config as "what does
a chorus-family modulation effect, rate-tuned to Leslie figures, do to
transcription" — not as "what does a Leslie do to transcription."

---

## 4. Parameter grounding

### 4.1 `rate_hz` — cited

Leslie 122/147-class cabinets have independently motored horn (treble) and
drum (bass) rotors, each with a distinct rotation speed for the "chorale"
(slow) and "tremolo" (fast) settings:

| State | Horn | Drum |
|---|---|---|
| Chorale (slow) | ~50 RPM (0.83 Hz) | ~40 RPM (0.67 Hz) |
| Tremolo (fast) | ~400 RPM (6.67 Hz) | ~340 RPM (5.67 Hz) |

Source: HammondWiki, Leslie Rotation Speed (`REFERENCE.md` §2.5). The
RPM/Hz figures themselves are solid — an enthusiast reference, but internally
consistent and precise to the RPM.

Since `pedalboard.Chorus` exposes exactly one `rate_hz`, not independent
horn/drum rates, this config **averages** each pair to a single
representative rate — a simplification this plan makes, not something the
source itself asserts:

| Condition | `rate_hz` used | Averaging |
|---|---|---|
| `leslie_chorale` | 0.75 | mean(0.83, 0.67) |
| `leslie_tremolo` | 6.0 | mean(6.67, 5.67) |

This is the only parameter in this scenario with a real historical citation
behind its specific numeric value. Everything else in Section 4.2 is a
judgment call.

### 4.2 `depth`, `centre_delay_ms`, `feedback`, `mix` — judgment call, NOT tuned by ear

**No historical citation exists for these four parameters, and this
implementation could not tune them by ear either** — an unsupervised coding
agent has no audio playback capability. This is a real limitation, stated
plainly rather than glossed over: the values below were chosen by
**engineering judgment from typical `Chorus`-effect parameter ranges**, not
by listening, and not by any cited source. Anyone with the ability to
actually listen to rendered output should treat these four values as a
starting point to sanity-check, not a settled result — same discipline
`old_recording/README.md` applies to its own presence-bump dB figures,
which are also uncited judgment calls (though that implementation, unlike
this one, had a documented ability to iterate on them).

The reasoning applied:

- **Starting point:** `pedalboard.Chorus`'s own class defaults
  (`rate_hz=1.0, depth=0.25, centre_delay_ms=7.0, feedback=0.0, mix=0.5`,
  measured by direct inspection on the installed `pedalboard` 0.9.24 build)
  are the plugin author's own choice of "a reasonable chorus," and this
  scenario's values are modest departures from that baseline rather than an
  arbitrary independent guess.
- **Physical reasoning for the chorale/tremolo split:** a Leslie's "chorale"
  (slow) setting is the smoother, gentler rotation state, typically used for
  ballads and slower material; "tremolo" (fast) is the more aggressive,
  pronounced state used for solos and louder passages. It is physically
  sensible that the fast setting also reads as perceptually "more" — so
  `leslie_tremolo` uses somewhat higher `depth` and `mix` than
  `leslie_chorale`. This difference is kept **modest** deliberately, since
  it is explicitly not cited, and `rate_hz` (Section 4.1) remains the
  primary, cited differentiator between the two conditions — not `depth`/
  `mix`.
- **Values chosen:**

  | Condition | `depth` | `centre_delay_ms` | `feedback` | `mix` |
  |---|---|---|---|---|
  | `leslie_chorale` | 0.35 | 8.0 | 0.10 | 0.45 |
  | `leslie_tremolo` | 0.45 | 10.0 | 0.15 | 0.55 |

  All four sit within the "audible but musical chorus" range typical of the
  `Chorus` plugin family (`depth` 0.3-0.5, `centre_delay_ms` 7-15,
  `feedback` 0.0-0.25, `mix` 0.4-0.6) — none pushed to an extreme that would
  read as a different effect entirely (e.g. flanger-like feedback, or a
  fully-wet signal).

**Do not report these four values, or any F1 delta attributable to them, as
independently cited facts about real Leslie cabinets.** Only `rate_hz`
carries that status.

---

## 5. Confound disclosure: post-effects normalisation, and why `pre_effects` is `false` here

Unlike `old_recording`, `telephone_channel`, and `venue_acoustics`, this
config sets `normalisation.pre_effects: false`. Those three scenarios set it
`true` specifically because `Distortion`'s drive and `Compressor`'s
threshold are **level-dependent** — how much a signal clips or gets
compressed depends on the input level reaching those effects, so
pre-normalising makes their behavior deterministic across files that
FluidSynth happened to render at different levels.

`Chorus` has **no level-dependent parameter**. `rate_hz`, `depth`,
`centre_delay_ms`, `feedback`, and `mix` are all LFO-rate/delay-line/mix
settings whose character does not change with input signal level the way a
compressor's threshold-crossing or a distortion's clipping point does.
Pre-normalising therefore buys no determinism benefit for this scenario, so
`pre_effects` is left `false` (matching the parent directory's
`effects_combinations.yaml` default) rather than set `true` out of habit
copied from the other three scenarios.

The confound that **does** still apply here is the same one every scenario
in this directory shares: **post-effects peak normalisation stays on**
(`normalisation.enabled: true`, `mode: peak`, `target_db: -1.0`), and it
runs *after* the `Chorus` effect. Every condition reaches the transcriber at
-1 dBFS peak regardless of how much the chorus's comb-filter-like summing
changed the signal's peak level. **These conditions do not model any change
in absolute level a rotary speaker's mic'd cabinet output might have.**
Only the modulation character (Section 4) is modelled, not level. If a
future study wants to measure a level effect specifically, it needs a
config with post-effects normalisation disabled or reworked — out of scope
for this file.

---

## 6. Interpretation constraints

- **This is the least physically grounded of the three new scenarios in
  this round.** `telephone_channel`'s bandwidth figures are ITU-T standards
  and `venue_acoustics`'s RT60 targets are measured against published
  figures with a calibrated realization; this scenario's only cited number
  is `rate_hz`, and the effect used to realize it (`Chorus`) is explicitly
  not a Leslie simulation (Section 3). Do not extend the confidence
  appropriate to `telephone_channel`/`venue_acoustics` to this scenario.
- **Genuinely one-factor**, unlike `old_recording`/`telephone_channel`'s
  holistic chains — every non-baseline condition here changes exactly one
  `Chorus` instance's five parameters, all in service of one modulation
  effect. Unlike `venue_acoustics`, though, only one of those five
  parameters (`rate_hz`) carries a citation; treat any F1 delta between
  `leslie_chorale` and `leslie_tremolo` as "faster vs. slower chorus
  modulation, tuned to Leslie-like rates," not as a clean single-variable
  ablation the way `venue_acoustics`'s RT60 ladder is.
- **Not historically applicable to this corpus** (Section 1) — never
  describe a result from this config as evidence about how acoustic piano
  recordings "actually sounded" with rotary-speaker processing; no such
  recording chain existed.

---

## 7. Usage

```bash
sonitra benchmark \
  --config config/benchmark/rotary_speaker/rotary_scenarios.yaml \
  --dataset <name>
```

Smoke test on a small subset (use `--seed` for a reproducible sample):

```bash
sonitra benchmark \
  --config config/benchmark/rotary_speaker/rotary_scenarios.yaml \
  --dataset test --limit 2 --seed 0
```

Results land under `corpus/<name>/benchmark/` (`benchmark_results.jsonl`,
`summary.json`, per-condition transcriptions) — see the parent
`config/benchmark/README.md` for the general output layout and the resume
mechanism.
