# Evaluation Metrics and Systems for Audio-to-MIDI Transcription in Synthetic-Augmentation Pipelines

## Executive overview

Automatic music transcription (AMT) for audio-to-MIDI remains most mature for solo piano and, increasingly, multi-instrument settings; state-of-the-art models are predominantly deep neural networks (Onsets & Frames, Kong-style regression models, Transformers such as MT3 and hFT, and new seq2seq approaches like Aria-AMT).[^1][^2][^3][^4]

Evaluation practice is dominated by information-retrieval (IR) style metrics at frame and note level (precision/recall/F1) with specific onset/offset tolerances, but there is a growing recognition that these are musically impoverished and insufficient for expressive/notation-level use cases.[^2][^5][^1]

For a synthetic-pipeline study that perturbs rendered audio (reverb, noise, EQ, style), suitable metrics fall into three families: (1) standard IR metrics (frame, note, note-with-offset, velocity); (2) musically informed performance metrics (timing/articulation/harmony/dynamics correlations, notation edit distances); and (3) robustness metrics versus controlled augmentations (F1 degradation curves, sensitivity to individual transforms, DTW-based similarity of audio vs re-synthesized transcription).[^6][^7][^5][^4][^2]

Several recent AMT and robustness/augmentation studies (Hawthorne et al. 2018, Kong et al. 2021, Edwards et al. 2024, Hu et al. 2024, Bradshaw et al. 2024) give concrete designs for augmentation pipelines (pitch shift, reverberation, background noise, EQ, synthetic re-performances) and show which metrics are sensitive to these manipulations.[^8][^5][^4][^6][^2]

The most relevant systems to include in a contemporary benchmark that matches your proposed pipeline are: classic strong piano baselines (Onsets & Frames, Kong’s regression model), modern multi-instrument Transformers (MT3, T5-style models, hFT), efficient open-source general tools (Basic Pitch), and new robust seq2seq systems such as Aria-AMT; commercial tools like AnthemScore, Melodyne, ScoreCloud, and AudioScore can act as additional black-box baselines.[^9][^3][^10][^11][^12][^1]


## 1. State of the art AMT in context of synthetic perturbation

### 1.1 Canonical overviews and problem decomposition

Benetos et al. (2019) provide a widely cited high-level overview of AMT, decomposing it into frame-level (multi-pitch estimation), note-level (note tracking), stream-level (voice/instrument grouping), and notation-level transcription, and emphasizing the persistent challenges of polyphony, overlapping harmonics, expressive timing, annotation scarcity, and robustness across acoustic conditions.[^1]

This survey notes that evaluation traditionally focuses on MIREX-style tasks—multi-F0 estimation and note tracking—with IR metrics over carefully curated datasets (e.g., MAPS), and explicitly flags the gap between parametric output (piano-rolls) and genuine music-notation-level quality, where suitable metrics are still largely open.[^1]

A newer systematic survey (2024) reiterates these levels and highlights the dominance of two method families (NMF and neural networks), the central role of MAESTRO and MAPS, and the importance of data augmentation (time-stretch, pitch shift) and dataset integration (MAESTRO + GiantMIDI) for current ML-based AMT.[^13]


### 1.2 Modern piano transcription architectures

#### Onsets & Frames

Hawthorne et al.’s "Onsets and Frames" model introduced a dual-objective architecture with separate onset and frame branches (CNN + BiLSTM), using onset predictions to gate frame-wise activations during inference.[^2]

They evaluate on MAPS with:
- Frame-level precision/recall/F1.
- Note-level F1 with onset-only tolerance (±50 ms).
- Note-with-offset F1 requiring offsets within 20% of duration or 50 ms (whichever larger).
- Velocity-augmented note metrics that add a velocity tolerance of 0.1 in normalized velocity space.[^2]

The authors argue that note-with-offset (and velocity) F1 correlates better with perceptual quality than frame- or onset-only metrics and recommend using this as a primary metric.[^2]

#### High-resolution regression model (Kong et al.)

Kong et al. (2021) propose a high-resolution piano transcription model that regresses onset and offset times and supports pedal estimation; this architecture underlies newer robustness and augmentation studies.[^6][^8]

The model uses CNN + recurrent layers with regression heads for onset/offset timing, and is typically trained and evaluated on MAESTRO and MAPS with note-level F1 metrics (onsets and onsets+offsets).[^8]

#### Transformers and multi-instrument AMT

MT3 (Multi-Task Multitrack Music Transcription) uses a T5-style sequence-to-sequence Transformer to jointly transcribe multiple instruments and datasets, emphasizing multi-task learning and low-resource instruments.[^3][^14]

MT3 is evaluated with note-level F1 metrics (frame, onset, onset+offset, note-with-instrument), and the authors explicitly call out limitations of heterogeneous evaluation metrics across datasets, arguing for more consistent evaluation.[^14][^15]

More recent work from Toyama et al. (hFT: hierarchical frequency-time Transformer) achieves very high note-level F1 on MAESTRO and MAPS using Transformer architecture specialized over frequency and time axes.[^4]

#### Seq2seq/Whisper-style AMT and robustness

Bradshaw et al. (Aria-AMT, 2024) adapt a Whisper-like encoder-decoder to AMT, using heavy data augmentation (RIR, noise, EQ, pitch detuning), synthetic pretraining on Pianoteq-rendered MIDI, and a bootstrapping loop that uses DTW to filter auto-transcribed data.[^4]

They report state-of-the-art F1 scores on MAESTRO and MAPS (including augmented test variants), and systematically study correlations between DTW audio–audio distance and human judgments and mir_eval metrics, arguing DTW is a useful complementary robustness metric.[^4]


### 1.3 Multi-modal and notation-level directions

Recent work explores multimodal AMT combining audio with score images (MUSCAT) and visual piano transcription; these broaden the transcription context but still fall back on standard IR metrics (note F1) and are less central to a synthetic audio-perturbation pipeline.[^16][^1]

For true notation-level evaluation, Cogliati & Duan (2017) propose a metric that treats music notation as a sequence of sets of musical objects aligned over time and defines an edit distance over 12 aspects: barlines, clefs, key signatures, time signatures, notes, spelling, durations, stem directions, beaming/groupings, rests, rest durations, and staff assignment.[^7]

They fit a linear regression from these aspect-wise error counts to human ratings of pitch notation, rhythm notation, and note positioning, showing moderate correlation (R² around 0.53–0.60), and releasing code and data. This gives a notation-level metric that can sit on top of an audio→MIDI→notation pipeline.[^17][^7]


## 2. Metrics currently used in AMT

### 2.1 Standard IR-style metrics

#### 2.1.1 Frame-level metrics

Frame-level evaluation compares binary piano-roll matrices (pitches × time frames, e.g., 10 ms hop) between reference and prediction; precision, recall, and F1 are computed over active vs inactive frames.[^5][^1]

Frame metrics are:
- Simple and widely used (MIREX, Onsets & Frames, MT3, Kong et al.).
- Sensitive to sustain and segmentation but agnostic to note identity as musical events.
- Known to over-reward diffuse note activations and under-penalize short spurious notes.[^5][^2]

#### 2.1.2 Note-level metrics

Note-level metrics treat notes as tuples of (onset time, offset time, pitch[, velocity]), with matching procedures defined by mir_eval.[^5][^2]

Standard variants include:
- **Onset-only note F1**: onsets must lie within ±50 ms of reference onset; offsets ignored.
- **Onset+offset note F1**: onset condition above plus offsets within max(20% of reference duration, 50 ms).
- **Onset+offset+velocity F1**: above plus velocity within tolerance (typically 0.1 in normalized  velocity after linear rescaling).[^13][^5][^2]

These metrics are computed per piece and then averaged; they are the core reporting metrics in Onsets & Frames, Kong et al., MT3, Toyama’s hFT, and most MAESTRO-based work.[^3][^8][^5][^2]

Note-level metrics are more musically meaningful than frame metrics, but still treat all note errors uniformly and ignore aspects like voice assignment, spelling, or musical role (melody vs accompaniment).[^5]


### 2.2 Musically informed and notation-aware metrics

#### 2.2.1 Musically informed performance metrics (mpteval)

Hu et al. (2024) criticize standard IR metrics for ignoring musical dimensions like articulation, dynamics, rhythmic microtiming, and harmonic context, and propose musically informed metrics implemented in the mpteval library.[^18][^5]

Metrics are defined as correlations between time series extracted from reference and predicted MIDI for different expressive dimensions:
- **Timing**: Inter-onset intervals (IOI) for melody and accompaniment streams (Melody IOI, Accompaniment IOI).[^5]
- **Articulation**: Key-overlap ratio (KOR) for melody and bass, plus ratio KOR (melody vs bass legato).[^5]
- **Harmony**: Cloud Diameter and Cloud Momentum based on Chew’s spiral array tonal model, evaluated over sliding windows.[^5]
- **Dynamics**: Loudness ratio between melody and bass using a simple velocity-to-loudness model.[^5]

These metrics yield correlation scores in [−1, 1]; Hu et al. show that models which look similar on F1 can differ substantially in timing/articulation/dynamics quality, and that these metrics are more discriminative under audio perturbations (reverb, noise) than IR metrics.[^5]

#### 2.2.2 Notation accuracy metric (Cogliati–Duan)

As noted above, Cogliati & Duan’s metric defines an edit distance over high-level notation aspects after aligning two scores by pitch content using dynamic time warping.[^7][^17]

This metric provides 12 aspect-wise error counts, which can be normalized and combined linearly to approximate human ratings for pitch notation, rhythm notation, and note positioning; the authors release code and a dataset of transcriptions evaluated by music theorists.[^7]

This is well-suited when the pipeline produces full notation (MusicXML) rather than just MIDI, especially if quantization, spelling, and voice assignment are in scope.

#### 2.2.3 Perceptually informed metrics (PEAMT)

Hu et al. also reference PEAMT (Ycart et al.) as a perceptually informed piano transcription metric; in their experiments, PEAMT correlates most with frame-level F1 and their harmony Cloud Momentum metric, suggesting that listeners weigh harmonic context heavily.[^5]

While PEAMT is not yet standard, it may be useful as an additional reference for perceptual quality if available.


### 2.3 Robustness and augmentation-related metrics

#### 2.3.1 Out-of-distribution F1 and degradation analysis

Edwards et al. (2024) focus explicitly on robustness and data augmentation. They retrain Kong et al.’s regression model on a re-recorded MAESTRO (Studio MAESTRO) and augmented variants, and evaluate out-of-distribution note-onset F1 on MAPS without training on MAPS.[^6][^8]

Key practices and metrics:
- Report note-onset F1 on both in-distribution (MAESTRO, Studio MAESTRO) and OOD (MAPS) test sets.
- Compare models trained with/without augmentation and with different augmentation subsets.
- Quantify degradation from baseline to perturbed audio (e.g., with added noise, EQ, pitch shifts, reverb) at test time.[^6]

They also conduct:
- **Single augmentation experiments**: train with only one augmentation (background noise, pitch shift, reverb, EQ) to measure individual impact on OOD F1.
- **Ablation of augmentation pipeline**: train with full augmentation then omit one component at a time.[^6]

This yields explicit sensitivity metrics: e.g., dropping pitch-shift or reverb reduced OOD note-onset F1 by about 3–4 points on MAPS, whereas EQ and background noise had smaller effects.[^6]

#### 2.3.2 Musically informed robustness under perturbations

Using their MPTEVAL metrics, Hu et al. analyze how Onsets & Frames, Kong’s model, and a Transformer model behave on re-recorded MAESTRO via a Disklavier and on artificially perturbed audio (multiple reverberation and noise levels).[^5]

They show that:
- Standard note F1 degrades under perturbations but less discriminatively than their timing and articulation metrics.
- For example, Melody IOI correlations and KOR metrics reveal substantial differences in timing/articulation preservation across models under reverberation and noise that F1 alone obscures.[^5]

This indicates that a robustness study focused on expressive performance should include such musically informed metrics alongside F1.

#### 2.3.3 Dynamic Time Warping (DTW) based metrics

Bradshaw et al. use DTW between original audio and re-synthesized transcription audio to score transcription quality and to filter synthetic training data.[^4]

They demonstrate that:
- DTW correlates strongly (Spearman −0.88) with human annotations of transcription quality on a 1–5 scale.[^4]
- DTW correlates well with mir_eval F1 metrics, particularly onset F1.[^4]
- DTW is surprisingly robust to recording quality (reverb, noise) in piano recordings, likely because the onset structure dominates.[^4]

DTW is thus a promising complementary metric for your augmentation pipeline, especially to analyze how much the perturbed audio diverges (after transcription and re-synthesis) from the original reference.


## 3. Data augmentation in AMT and their metrics

### 3.1 Augmentation techniques in robust piano transcription

Edwards et al. (2024) give a detailed, empirically grounded augmentation pipeline, implemented via the audiomentations library; the core components are:[^6]
- Two random 7-band parametric EQs.
- Additive background noise from pub/café recordings, with variable SNR.
- Small random pitch shifts (±0.1 semitone) to mitigate overfitting to instrument tuning.
- Reverb derived from multiple real impulse responses.

Training data combines original MAESTRO audio, re-recorded Studio MAESTRO, and six Pianoteq-rendered variants per piece, with sampling probabilities across these sources.[^6]

They show that:
- Without augmentation, a model trained on Studio MAESTRO overfits heavily: note-onset F1 drops from 97.3 (Studio MAESTRO test) to 80.8 (original MAESTRO test) and further for OOD MAPS.[^6]
- Augmentation plus diversified timbre yields state-of-the-art OOD note-onset F1 of 88.4 on MAPS without training on MAPS.[^8]

Single-augmentation and ablation results quantify the relative contribution of each augmentation (pitch shift and reverb being most impactful for robustness, background noise and EQ less so for MAPS).[^6]


### 3.2 Other augmentation-related AMT work

The same paper reviews prior augmentation use:
- MAESTRO’s original paper emphasizes data augmentation (noise, reverb, compression, synthesizer rendering) for training Onsets & Frames though ablations suggested limited gains, likely due to the dataset size and domain.[^2][^6]
- Thickstun et al. and Lu et al. use label-preserving pitch-shift and cross-dataset mixtures for multi-instrument transcription and low-resource instruments.[^6]

Beyond piano, generalized audio data augmentation tutorials (e.g., torchaudio’s) cover RIR-based reverberation simulation and SNR-controlled noise addition, which are directly applicable to your synthesis-plus-perturbation stage.[^19]

Bradshaw et al. go further by combining synthetic Pianoteq rendering with heavy spectrogram masking, RIR, noise, dynamic EQ, and detuning, then bootstrapping more training data, again measured via standard F1 and DTW.[^4]


### 3.3 Metrics used specifically in augmentation studies

Across these augmentation-focused works, the metrics used to report results are:
- **Note-onset precision/recall/F1** on OOD datasets (MAPS, new re-recorded MAESTRO).[^8][^6]
- Occasionally, full note-with-offset F1 on MAESTRO and OOD data.[^8][^5]
- Degradation tables showing F1 drop under specific test-time perturbations (e.g., adding background noise, EQ, pitch shift, reverb individually).[^6]
- For musically informed metrics, timing/articulation/harmony/dynamics correlations before and after perturbations.[^5]
- DTW between original and re-synthesized audio as a continuous quality metric.[^4]

These patterns suggest that for an augmentation study, the main axes are: absolute OOD F1, F1 degradation per augmentation, and expressive-dimension correlations.


## 4. Recommended metric set for your pipeline

Given your core pipeline (symbolic → synthesis → perturbation → transcription → score analysis, plus parameter influence), and existing practice, a metric suite can be defined at three levels:

### 4.1 Symbolic equivalence level (core AMT metrics)

These compare predicted MIDI to original MIDI ignoring notational issues:

- **Frame-level precision/recall/F1** for sanity and comparison with older literature.
- **Note-level metrics** (using mir_eval or equivalent):
  - Onset-only F1 (±50 ms) as a baseline.
  - Onset+offset F1 with 20%/50 ms rule.
  - Onset+offset+velocity F1 where your systems output velocities and you can control dynamics in synthesis.[^20][^2]

For a perturbation study, note-with-offset and note-with-offset+velocity F1 should be emphasized, aligning with Hawthorne et al.'s recommendation and later usage in MAESTRO-based SOTA papers.[^2][^5]


### 4.2 Musically informed expressive metrics

To analyze how perturbations and systems affect expressive content (timing, articulation, dynamics) beyond binary correctness:

- **MPTEVAL metrics** from Hu et al.:
  - Melody IOI and Accompaniment IOI correlations for timing.
  - Melody KOR, Bass KOR, Ratio KOR for articulation.
  - Cloud Diameter and Cloud Momentum for tonal/harmonic aspects.
  - Dynamics (loudness ratio melody/bass).[^18][^5]

These are computed from paired reference and predicted MIDI and are particularly suitable when starting from clean symbolic scores and analyzing performance aspects.

- Optionally, **PEAMT** as a perceptual metric, if you want a single scalar that attempts to aggregate perceptual salience.[^5]


### 4.3 Notation-level and perceptual/robustness metrics

For full notation and global quality/robustness:

- **Notation edit metric** (Cogliati–Duan) applied to MusicXML scores generated from reference and predicted MIDI via the same quantization and engraving pipeline, if you care about notation readability and high-level correctness (key signatures, beaming, staff assignment).[^17][^7]
- **DTW-based similarity** between original synthesized audio and re-synthesized transcription, used as a scalar measure that captures both pitch and rhythmic mismatches at the waveform/spectral level.[^4]

This DTW measure can (a) be used to help filter pathologically bad transcriptions in large-scale experiments, and (b) complement F1 when investigating effect of augmentations that might alter spectral properties without dramatically affecting IR metrics.


### 4.4 Sensitivity and parameter influence analysis

To explicitly study "audio synthesis parameter influence":

- Define controlled axes (e.g., SNR, RT60, EQ tilt, modulation depth) and measure **F1 and musically informed metric degradation curves** across parameter sweeps.
- Compute **partial derivatives** or effect sizes (e.g., F1 drop per dB noise, per 0.1 semitone random detune) to quantify robustness.
- If using DTW, analyze DTW vs parameter curves alongside F1 to examine whether some perturbations primarily affect perceptual similarity but not frame/note metrics.

These analyses follow the pattern of Edwards et al. (data degradation and ablation tables) and Hu et al. (grid search over reverb and noise combination levels).[^6][^5]


## 5. Augmentation methods and metrics in literature

### 5.1 Summary of key augmentation-oriented studies

- **Onsets & Frames (Hawthorne et al.)** acknowledge data augmentation (normalization, reverb, compression, noise, alternative synthesis) but report that it did not substantially change their performance on MAPS; they still recommend better datasets and more diversity rather than more augmentation.[^2]

- **Edwards et al. 2024 (A Data-Driven Analysis of Robust Automatic Piano Transcription):**
  - Introduce Studio MAESTRO (re-recorded Disklavier) and additional synthetic Pianoteq renderings.
  - Focus on data augmentation pipeline (EQ, noise, pitch shift, reverb) and its impact on OOD note-onset F1, particularly on MAPS.
  - Use note-onset F1 and augmentation-specific degradation tables as main metrics.[^8][^6]

- **Bradshaw et al. 2024 (Aria-AMT):**
  - Use extensive augmentations (RIRs, noise, EQ, detuning, spectrogram masking) and large synthetic pretraining.
  - Evaluate with mir_eval F1 metrics on MAESTRO, MAPS, and heavily augmented variants.
  - Introduce DTW as a metric and show strong correlation with human judgment and F1.[^4]

- **Hu et al. 2024 (musically informed metrics):**
  - Use additive noise and multiple reverberation settings on re-recorded MAESTRO and show that their correlation-based metrics are more sensitive to musical quality than F1 alone under perturbation.[^5]


### 5.2 Takeaways for your study

From these works, the following metric-related conclusions emerge:
- Note-onset and note-with-offset F1 remain the primary benchmark for parametric correctness, especially for piano and MAESTRO/MAPS-style data.[^8][^2][^6]
- Musically informed metrics are valuable when the research question involves expressive performance aspects or when the audio perturbations may subtly affect timing/articulation rather than gross note correctness.[^5]
- DTW can effectively complement symbolic metrics in large-scale and robustness settings, especially when scoring synthetic vs perturbed audio.[^4]
- Reporting OOD performance (e.g., training on one corpus, testing on another, or on re-recorded/perturbed audio) is essential to avoid overfitting to specific acoustics.[^6][^5]


## 6. Candidate AMT systems to include

### 6.1 Open-source research systems

For a research-grade evaluation, including representative systems across architecture families is recommended:

- **Onsets & Frames** (Magenta / Google Brain implementation): canonical piano baseline, strong on MAESTRO/MAPS, well-known metrics and open code.[^9][^2]
- **Kong et al.’s high-resolution regression model** (Bytedance piano transcription): strong piano performance and used in robustness studies.[^8][^6]
- **MT3** (multi-task multitrack Transformer): modern multi-instrument AMT with T5 architecture; open-source implementation via Magenta and PyTorch ports.[^21][^22][^3]
- **Basic Pitch** (Spotify): lightweight multi-pitch and note tracking model optimized for speed and general instruments, open-source and usable as a practical baseline.[^23][^11]
- **Aria-AMT** or similar seq2seq/Whisper-style models, if code is available: designed explicitly for robustness and dataset expansion; highly relevant to an augmentation study.[^4]
- **Other recent research systems** (e.g., NoteEM, unaligned supervision methods) that claim strong cross-dataset generalization and robust notes-with-instrument F1.[^24]


### 6.2 Commercial and semi-commercial systems

From Benetos et al. and other sources, commonly cited commercial AMT tools include: Melodyne, AudioScore, ScoreCloud, AnthemScore, and Transcribe!.[^10][^12][^1]

These provide audio-to-MIDI/notation functionality but typically do not expose internal metrics; they are mainly suitable as black-box baselines where you compute your own metrics on their output.

For example:
- **AnthemScore**: dedicated audio-to-notation transcription for full tracks, desktop and web versions.[^10]
- **Melodyne**: widely used in production for pitch/time editing; can export MIDI and is often used for monophonic or polyphonic pitch tracking.[^12]
- **ScoreCloud**: real-time audio-to-notation tool focused on musician-friendly notation output.[^12]

Including one or two of these as external baselines would allow comparison of research models vs tools actual musicians use.


### 6.3 Practical selection tailored to your pipeline

Given the synthetic-piano-centric nature of your pipeline (MIDI/MusicXML → synthesis → perturbation → transcription), systems that focus on piano and multitrack symbolic output make the most sense:
- At least one strong piano-optimized model (Kong, Onsets & Frames, or Edwards’ re-trained variant).[^2][^8][^6]
- At least one multi-instrument model (MT3 or hFT) to test generalization beyond piano synthesis, especially if you later include non-piano instruments.[^3][^4]
- A lightweight open-source general-purpose model (Basic Pitch) as a fast baseline.[^11]
- Optionally, Aria-AMT or similar for robustness-focused seq2seq comparisons.[^4]
- One commercial system (AnthemScore or Melodyne) for a “real-world” baseline.[^10][^12]


## 7. How these metrics map onto your pipeline stages

Given your pipeline:

1. Start with a symbolic corpus (MIDI/MusicXML).
2. Render audio with controllable synthesis parameters (timbre, reverb, mic position, style).
3. Apply controlled perturbations (noise, reverb RIRs, EQ, style transfers, optional source separation).
4. Run transcription systems.
5. Compare transcribed output to original symbolic score and analyze influence of synthesis and perturbation parameters.

The recommended metric layering is:

- **Core correctness**: note-onset/note-with-offset/note-with-offset+velocity F1 + frame F1.
- **Expressive fidelity**: musically informed metrics (IOI timing, KOR articulation, dynamics, harmony) between original and transcribed MIDI.
- **Notation quality**: Cogliati–Duan notation accuracy metric on MusicXML where available.
- **Audio-level similarity**: DTW between original synthesized audio and audio re-synthesized from the transcription.
- **Robustness curves**: F1 and expressive metrics as functions of perturbation parameters (SNR, RT60, EQ slopes, etc.), including ablation of perturbations.

This set is consistent with state-of-the-art AMT and augmentation works while tailored to your synthesis-centric, RL/QD-aware research context.

---

## References

1. [[PDF] Automatic Music Transcription: An Overview](https://www.semanticscholar.org/paper/Automatic-Music-Transcription:-An-Overview-Benetos-Dixon/b0a4c24d1bc96d71402fc8668a823c43d8bc47dc) - Experiments show that this approach significantly outperforms a state-of-the-art music transcription...

2. [ONSETS AND FRAMES: DUAL-OBJECTIVE PIANO ...](https://archives.ismir.net/ismir2018/paper/000019.pdf) - The metrics used to evaluate a model are frame-level and note-level metrics including precision, rec...

3. [MT3: Multi-Task Multitrack Music Transcription](https://openreview.net/forum?id=iMSjopcOn0p) - A general-purpose Transformer model can perform multi-task AMT, jointly transcribing arbitrary combi...

4. [MUSICALLY AWARE AUTOMATIC PIANO ...](https://www.alexander-spangher.com/papers/aria_amt.pdf) - We employ a variety of data augmentation techniques, mostly targeting common recording environments ...

5. [Towards Musically Informed Evaluation of Piano ...](https://arxiv.org/abs/2406.08454) - Abstract page for arXiv paper 2406.08454: Towards Musically Informed Evaluation of Piano Transcripti...

6. [A Data-Driven Analysis of Robust Automatic Piano ...](https://arxiv.org/html/2402.01424v1) - We present several experiments to explore the effect of data augmentation on training piano transcri...

7. [A METRIC FOR MUSIC NOTATION TRANSCRIPTION ...](https://archives.ismir.net/ismir2017/paper/000131.pdf) - A METRIC FOR MUSIC NOTATION TRANSCRIPTION ACCURACY. Andrea Cogliati. University of Rochester. Electr...

8. [A Data-Driven Analysis of Robust Automatic Piano ...](https://zenodo.org/records/10610212) - On the MAESTRO test set, we acheive a note onset of 96.6 F1 score, compared to 96.7 of Kong et al. P...

9. [Onsets and Frames: Dual-Objective Piano Transcription](https://magenta.withgoogle.com/onsets-frames) - Onsets and Frames is our new model for automatic polyphonic piano music transcription. Using this mo...

10. [AnthemScore - Automatic Music Transcription Software](https://www.lunaverus.com) - AnthemScore is software for automatic music transcription using AI. Convert audio files like MP3 and...

11. [Basic Pitch: A lightweight model for multi-pitch, note and ...](https://ressources.ircam.fr/en/media/x6cb24a_basic-pitch-a-lightweight-model-for-multi) - The model is trained to jointly predict frame-wise onsets, multi-pitch and note activations, and we ...

12. [Best Music Transcription Software: What Actually Matters](https://scorecloud.com/learn/best-music-transcription-software/) - ScoreCloud goes from audio to editable notation directly. Other options include AnthemScore (audio t...

13. [Machine Learning Techniques in Automatic Music ...](https://arxiv.org/html/2406.15249v1) - This review critically evaluates both fully automatic and semi-automatic AMT systems, emphasizing th...

14. [[2111.03017] MT3: Multi-Task Multitrack Music Transcription](https://arxiv.org/abs/2111.03017) - We demonstrate that a general-purpose Transformer model can perform multi-task AMT, jointly transcri...

15. [MT3: Multi-Task Multitrack Music Transcription](https://liner.com/review/mt3-multitask-multitrack-music-transcription) - Table 2: Transcription F1 scores for Frame, Onset, and Onset+Offset metrics defined in Section 4.2. ...

16. [MUSCAT: a Multimodal mUSic Collection for Automatic ...](https://openreview.net/forum?id=B3CsOcxXOa) - Our work intends to advance the state of the art for multimodal image and audio music transcription ...

17. [Piano Music Transcription into Music Notation](https://labsites.rochester.edu/air/projects/AMT.html) - Andrea Cogliati and Zhiyao Duan, A metric for Music Notation Transcription Accuracy, in Proc. of Int...

18. [CPJKU/mpteval: Musical piano transcription evaluation](https://github.com/CPJKU/mpteval) - Towards Musically Informed Evaluation of Piano Transcription Models ... This repository provides a s...

19. [Audio Data Augmentation — Torchaudio 2.10.0 ...](https://docs.pytorch.org/audio/stable/tutorials/audio_data_augmentation_tutorial.html) - torchaudio provides a variety of ways to augment audio data. In this tutorial, we look into a way to...

20. [Automatic Music Transcription: An Overview](https://labsites.rochester.edu/air/publications/benetatos19automaticmusic.pdf) - It involves perception (analyzing complex auditory scenes), cog- nition (recognizing musical objects...

21. [magenta/mt3 - Multi-Task Multitrack Music Transcription](https://github.com/magenta/mt3) - MT3 is a multi-instrument automatic music transcription model that uses the T5X framework. This is n...

22. [MT3 (Multi-Task Multitrack Music Transcription)](https://ai4culture.eu/resources/tools/44) - MT3 is a is a multi-instrument automatic music transcription model that can infer musical notes from...

23. [Rachel Bittner on Basic Pitch: An Open Source Tool for ...](https://newsroom.spotify.com/2022-09-01/rachel-bittner-on-basic-pitch-an-open-source-tool-for-musicians/) - We named the project Basic Pitch because it can also detect pitch bends in the notes, which is a par...

24. [Unaligned Supervision for Automatic Music Transcription in ...](https://benadar293.github.io) - Current AMT approaches are restricted to piano and (some) guitar recordings, due to difficult data c...

