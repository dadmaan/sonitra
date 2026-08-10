# Telephone Channel: Voice-Bandwidth Scenarios

`telephone_scenarios.yaml` benchmarks a piano render through three real,
standardized voice-channel bandwidths — ITU-T G.722 wideband VoIP, ITU-T
G.711 narrowband PSTN, and land-mobile-radio (intercom) FM voice practice —
plus AGC-style leveling and, for the narrowest tier only, clipping-type
saturation. It is a companion study to `config/benchmark/old_recording/` (the
same real-world-grounding methodology, reused here for a different effect
family) and to the parent directory's single-axis sweeps (`reverb_sweep`,
`compression_sweep`, `distortion_sweep`).

**Read this document before drawing any conclusion from a run of this
config.**

---

## 1. Scope

**Modelled**, per condition:

- Bandpass limiting via two cascaded highpass/lowpass filter pairs
  (`pedalboard` `HighpassFilter`/`LowpassFilter`, two stages per band — see
  Section 5), reusing `old_recording`'s proven two-stage-cascade technique.
- AGC-style leveling via `Compressor`.
- Clipping-type saturation via `Distortion` (`intercom_lofi` only).

**Explicitly NOT modelled**:

- Codec artifacts — µ-law/A-law quantization noise, packet loss/jitter for
  VoIP. This config is an **analog-channel bandwidth proxy**, not a
  simulation of the actual G.711/G.722 encode/decode pipeline.
- Sidetone/echo-cancellation artifacts, comfort noise.
- Reduced medium level / dynamic range — cancelled by post-effects peak
  normalisation, which runs after the whole chain. See Section 4.

### 1.1 How this differs from `old_recording`'s AM-radio condition

`old_recording`'s `am_bandlimit_*` conditions constrain only the **top** end
(5.3–7.4 kHz -3 dB) with **no highpass at all** — AM broadcast was a
top-end-only constraint (see `old_recording/README.md` Section 2.2).
Telephone and land-mobile-radio voice channels are different in kind, not
just degree: they constrain **both** ends of the band, and land considerably
narrower than any `old_recording` tier. This is why `telephone_scenarios.yaml`
always enables both the highpass and lowpass cascades together (baseline
excepted), where `old_recording` deliberately omits the highpass for its AM
conditions.

---

## 2. Historical / standards grounding

### 2.1 `voip_wideband` — ITU-T G.722

ITU-T G.722 is the standard for wideband speech coding used by modern "HD
Voice" VoIP and some mobile networks: audio bandwidth **50–7000 Hz**, wider
at both ends than classic narrowband telephony. Grounds the -3 dB targets
50 Hz (highpass) / 7000 Hz (lowpass).

### 2.2 `pstn_narrowband` — ITU-T G.711

ITU-T G.711 is the standard for narrowband telephony that the analog PSTN was
built around: audio bandwidth **300–3400 Hz** at 8 kHz sampling, 8-bit
µ-law/A-law quantization (quantization itself is not modelled here — see
Section 1). This is also the industry-standard reference figure for
"telephone bandwidth" generally. Grounds the -3 dB targets 300 Hz (highpass) /
3400 Hz (lowpass).

### 2.3 `intercom_lofi` — land-mobile-radio (two-way radio) voice practice

"Normal FM practice limits the information for voice bandwidth from 300 Hz
to 3 kHz" — this audio-bandwidth figure has stayed constant across land
mobile radio's RF-channel-width evolution (25 kHz legacy → 12.5 kHz
narrowbanding → 6.25 kHz-equivalent P25 Phase 2), since it describes the
voice content, not the RF carrier. Grounds the -3 dB targets 300 Hz
(highpass, same lower bound as `pstn_narrowband` — both share the same
source figure for the bottom of the band) / 3000 Hz (lowpass, narrower top
end than PSTN).

**Confidence note:** the land-mobile-radio figure rests on a single
corroborating source (not a formal standard document directly consulted),
though internally consistent with the widely-repeated "300 Hz–3 kHz" figure
for two-way/intercom voice audio. See `REFERENCE.md` §2.1 (companion plan
document) for the full source list.

### 2.4 Compressor and Distortion values — a judgment call, not a citation

No standard specifies an exact AGC compression ratio or clipping depth for
any of these three channels. The `Compressor` ratio/threshold escalation
(2:1 @ -18 dB → 4:1 @ -20 dB → 8:1 @ -15 dB) and `intercom_lofi`'s 10 dB
`Distortion` drive encode a **directionally-motivated but not directly-cited**
"wideband = gentle, narrowband = moderate, intercom = heavy" escalation —
the same confidence tier `old_recording` assigns its own AM/shellac
compressor settings (`old_recording/README.md` does not claim a citation for
its compressor ratios either). Treat these two parameters as illustrative,
not as a specific claim about any real codec's or radio's actual leveling
behavior.

---

## 3. Piano-fundamental constraint

The corpus is solo piano: fundamentals run from A0 = 27.5 Hz up to
C8 = 4186 Hz. Both `pstn_narrowband` and `intercom_lofi`'s 300 Hz highpass
sit well below the piano's mid-range but still attenuate the lowest two
octaves materially — as with `old_recording`, **any resulting F1 drop in the
bottom of the keyboard for those two tiers is at least partly an artefact of
the filter**, not solely evidence about how "hard" a narrowband channel is
for AMT.

---

## 4. Confound disclosure: post-effects normalisation

`normalisation.pre_effects` is a **single-stage switch, not additive** —
`src/sonitra/normaliser.py`'s `normalise_from_config()` normalises *either*
before the effects chain *or* after, never both (`config/source.yaml:140`:
"true = normalise before effects chain; false = after"). This config leaves
it at its default, `false`, so **post-effects peak normalisation stays on**
(`normalisation.enabled: true`, `mode: peak`, `target_db: -1.0`) and runs
*after* the entire pedalboard effects chain, while pre-effects normalisation
does not run at all. That means every condition reaches the transcriber at
-1 dBFS peak, regardless of how much energy the bandpass cascade removed —
and it also means `Distortion` drive and `Compressor` threshold operate on
whatever level FluidSynth happened to produce for a given file, not a
normalised input level. A real AGC/clipping stage also reacts to whatever
level arrives at it, so this is a reasonable approximation, not a
correctness gap; it is called out here so a reader doesn't assume the
Compressor/Distortion stage sees a fixed input level across files.

**These conditions explicitly do not model a telephone channel's actual
received level.** A real phone call or radio link delivers audio at a level
set by the far end's transmit gain and the channel's loss, not renormalized
to a fixed peak — but that reduced/variable level is cancelled here by
design so that the spectral/dynamics shaping being studied is not confounded
with a simple gain difference basic-pitch would trivially compensate for.
Only the **spectral bandwidth and dynamics character** of each channel are
modelled by this config, not its absolute level. If a future study wants to
measure the level effect specifically, it needs a config with post-effects
normalisation disabled or reworked — out of scope for this file.

---

## 5. Filter cascade calibration

`pedalboard`'s `HighpassFilter`/`LowpassFilter` are first-order (6 dB/octave),
-3 dB at their nominal `cutoff_frequency_hz`, and expose no other parameter.
Real voice-channel bandpassing (channel filters, receiver IF stages) is
steeper than a single first-order section, so every band in this config
cascades **two** first-order instances (slots 0+1 for highpass, 2+3 for
lowpass) — the same technique `old_recording/README.md` Section 5 establishes
and measures on this installed `pedalboard` build; this section reuses that
methodology, not its specific numbers, since the target frequencies differ.

Two identical cascaded first-order sections are **-6 dB**, not -3 dB, at the
nominal cutoff — the -3 dB point moves. Solving `(1 + (f / f_c)^2)^2 = 2`
analytically gives the cascade's -3 dB point at:

- **lowpass:** `0.6436 * f_c`
- **highpass:** `1.5538 * f_c`

`old_recording/README.md` Section 5 already found that this analytic factor
holds well for the highpass side (cutoffs far from Nyquist) but that
pedalboard's filters are digital biquads, so bilinear warping pulls the
realised **lowpass** -3 dB point below the analytic prediction as the cutoff
approaches Nyquist. Every lowpass nominal below was therefore solved
**numerically** — binary search against the actual measured frequency
response of the real 2-stage cascade (impulse response → FFT → -3 dB
crossing, linearly interpolated between the nearest FFT bins), not derived
from the `0.6436` factor alone. The highpass nominals were solved the same
way for consistency, even though the analytic factor alone is close enough
there to be usable directly.

Measured on installed `pedalboard` 0.9.24, sample rate 44100 Hz — matching
`old_recording`'s own measurement conditions so the two tables are directly
comparable — every nominal used in this config, verified through the actual
2-stage cascade:

| Condition | Slot pair | nominal per stage | realised -3 dB | target |
|---|---|---|---|---|
| `voip_wideband` | HP | 32.1 Hz | 49.98 Hz | 50 Hz |
| `voip_wideband` | LP | 9871 Hz | 7000.01 Hz | 7000 Hz |
| `pstn_narrowband` | HP | 192.7 Hz | 299.96 Hz | 300 Hz |
| `pstn_narrowband` | LP | 5152 Hz | 3400.08 Hz | 3400 Hz |
| `intercom_lofi` | HP | 192.7 Hz | 299.96 Hz | 300 Hz |
| `intercom_lofi` | LP | 4571 Hz | 3000.08 Hz | 3000 Hz |

`intercom_lofi` and `pstn_narrowband` share the same highpass -3 dB target
(300 Hz, per Section 2.3), so they share the same measured nominal — this was
**verified**, not assumed: the calibration is a deterministic function of the
target frequency and the cascade topology alone, so re-running the same
binary search against the same 300 Hz target reproduces the same 192.7 Hz
nominal to within the search tolerance (±0.05 Hz).

All six realised -3 dB points land within 0.1 Hz of their target — well
inside any reasonable tolerance for what this study needs.

---

## 6. Interpretation constraints

- **Holistic per tier, not one-factor.** Like `old_recording`'s per-era
  conditions and unlike the parent directory's single-axis sweeps
  (`reverb_sweep`, `compression_sweep`, `distortion_sweep`), each
  non-baseline condition here moves 2-3 parameter groups at once (bandpass
  cascade + Compressor, and additionally Distortion for `intercom_lofi`).
  The study can say "the intercom tier costs Y F1" but **cannot decompose**
  that delta into bandwidth vs. compression vs. saturation contributions
  individually.
- **Tiers are not a severity ladder.** Unlike `old_recording`'s
  mild/pronounced pairs within a single era, `voip_wideband` →
  `pstn_narrowband` → `intercom_lofi` are three **different real-world
  channels**, not three severities of the same channel. It is reasonable to
  compare all three against `baseline` and against each other, but do not
  describe the sequence as a single monotonic "severity" axis the way
  `old_recording`'s mild-to-pronounced ladders are.
- **Compressor/Distortion escalation is a judgment call** (Section 2.4) —
  do not report the compression/saturation differences between tiers as
  independently cited facts about VoIP, PSTN, or land-mobile-radio AGC
  behavior.

---

## 7. Usage

```bash
sonitra benchmark \
  --config config/benchmark/telephone_channel/telephone_scenarios.yaml \
  --dataset <name>
```

Smoke test on a small subset (use `--seed` for a reproducible sample):

```bash
sonitra benchmark \
  --config config/benchmark/telephone_channel/telephone_scenarios.yaml \
  --dataset test --limit 2 --seed 0
```

Results land under `corpus/<name>/benchmark/` (`benchmark_results.jsonl`,
`summary.json`, per-condition transcriptions) — see the parent
`config/benchmark/README.md` for the general output layout and the resume
mechanism.
