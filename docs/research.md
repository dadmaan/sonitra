# Evaluation metrics and systems for audio-to-MIDI transcription in synthetic-augmentation pipelines

This guide sums up past work on AMT testing. AMT means automatic music transcription, turning audio into notes. MIDI is a digital score format that stores notes, timing, and loudness. A synthetic-augmentation pipeline means you make audio from scores, change the sound on purpose, and test how well transcription still works.

## Executive overview

AMT for audio-to-MIDI works best for solo piano today. It is growing for music with many instruments. Top models are mostly deep neural nets. Examples are Onsets and Frames, Kong-style timing models, Transformers such as MT3 and hFT, and new step-by-step models like Aria-AMT.[^1][^2][^3][^4] Tests mostly use information-retrieval scores at frame and note level. Frame means a short slice of sound. Note means a single musical note. These scores use precision, recall, and F1 with set time limits for note starts and ends. But more researchers say these scores miss musical quality and fall short for expressive playing and full notation.[^2][^5][^1]

If you change rendered audio with reverb, noise, EQ, or style shifts, use three score groups. EQ means tone controls that boost or cut bass and treble. Reverb means room echo. First, standard hit-or-miss scores (frame, note, note-with-end, loudness). Second, musical performance scores (timing, playing style, harmony, and loudness links). Third, strength scores against planned sound changes (F1 drop curves, response to each change, DTW similarity of audio vs remade transcription). DTW means dynamic time warping, a way to line up two clips in time and measure the gap.[^6][^7][^5][^4][^2] Past AMT and strength studies (Hawthorne et al. 2018, Kong et al. 2021, Edwards et al. 2024, Hu et al. 2024, Bradshaw et al. 2024) describe sound-change pipelines (pitch shift, room echo, background noise, EQ, remade performances) and show which scores react to these changes.[^8][^5][^4][^6][^2]

Good systems to test span classic piano baselines (Onsets and Frames, Kong's timing model), modern multi-instrument Transformers (MT3, T5-style models, hFT), fast open tools (Basic Pitch), and strong step-by-step systems such as Aria-AMT. Shop tools like AnthemScore, Melodyne, ScoreCloud, and AudioScore give extra real-world baselines.[^9][^3][^10][^11][^12][^1]


## 1. State of the art AMT in context of synthetic perturbation

### 1.1 Canonical overviews and problem decomposition

Benetos et al. (2019) break AMT into four steps. They are frame-level pitch finding, note tracking, voice and instrument grouping, and full notation writing. They stress hard gaps that remain. These are many notes at once, overlapping overtones, expressive timing, few labeled sets, and weak results across rooms and mics.[^1] They note tests still focus on MIREX-style tasks. MIREX is a shared music-test contest. Tasks are multi-pitch finding and note tracking on curated sets such as MAPS. They flag the gap between raw note grids (piano-rolls) and true notation quality, where good scores are still open.[^1]

A newer 2024 survey covers the same steps. It notes two main tool groups (NMF and neural nets), the use of MAESTRO and MAPS piano sets, and the role of sound changes (time-stretch, pitch shift) and set mixes (MAESTRO plus GiantMIDI) for current ML-based AMT.[^13] NMF means non-negative matrix factorization, an older math method for splitting sounds. ML means machine learning.


### 1.2 Modern piano transcription architectures

#### Onsets & Frames

Hawthorne et al.'s "Onsets and Frames" model uses two linked aims. One branch finds note starts. One branch finds active frames. It uses CNN plus BiLSTM layers, which are neural-net parts that read sound pictures and time order. Note-start guesses gate frame activity at test time.[^2] Onset means note start. Frame means a short time slice.

They test on MAPS with:
- Frame-level precision/recall/F1.
- Note-level F1 with onset-only tolerance (±50 ms).
- Note-with-offset F1 requiring offsets within 20% of duration or 50 ms (whichever larger).
- Velocity-augmented note metrics that add a velocity tolerance of 0.1 in normalised velocity space.[^2]

The authors say note-with-end (and loudness) F1 tracks heard quality better than frame or start-only scores. They urge it as a main score.[^2]

#### High-resolution timing model (Kong et al.)

Kong et al. (2021) build a high-detail piano model. It predicts exact start and end times and can track pedal use. This design sits behind newer strength and sound-change studies.[^6][^8] The model uses CNN plus time-based layers with timing outputs. Teams most often train and test it on MAESTRO and MAPS with note-level F1 scores (starts and starts-plus-ends).[^8]

#### Transformers and multi-instrument AMT

MT3 (Multi-Task Multitrack Music Transcription) uses a T5-style step-by-step Transformer. A Transformer is a neural net that tracks long links across time. It transcribes many instruments and sets at once, which helps rare instruments.[^3][^14] It reports note-level F1 scores (frame, start, start-plus-end, note-with-instrument). The authors note mixed scoring rules across sets block fair compare, and they call for steadier tests.[^14][^15]

Newer work from Toyama et al. (hFT, a Transformer split over pitch and time) hits very high note-level F1 on MAESTRO and MAPS.[^4]

#### Seq2seq and Whisper-style AMT and strength

Bradshaw et al. (Aria-AMT, 2024) adapt a Whisper-like encoder-decoder to AMT. An encoder-decoder reads audio in and writes notes out step by step. They use heavy sound changes (room impulse responses, noise, EQ, pitch drift), pretraining on Pianoteq-made MIDI, and a loop that uses DTW to keep good auto-labeled data. RIR means room impulse response, a record of room echo. DTW lines up two clips in time.[^4] They report top F1 scores on MAESTRO and MAPS (with changed test variants). They study links between DTW audio distance, human ratings, and mir_eval scores. Mir_eval is a standard music-scoring library. They urge DTW as a useful extra strength score.[^4]


### 1.3 Multi-modal and notation-level directions

Newer work joins audio with score images (MUSCAT) and camera-based piano tracking. These widen context but still use standard note F1. They matter less for a sound-change pipeline.[^16][^1]

For true notation tests, Cogliati and Duan (2017) treat notation as timed groups of musical items. They set an edit distance over 12 parts: barlines, clefs, key marks, time marks, notes, spelling, lengths, stem sides, beams and groups, rests, rest lengths, and staff slots.[^7] They fit a straight-line model from these error counts to human ratings of pitch writing, rhythm writing, and note place. The fit is fair (R² about 0.53-0.60, where 1.0 is perfect). They share code and data. This gives a notation score you can place on top of an audio-to-MIDI-to-notation chain.[^17][^7]


## 2. Metrics currently used in AMT

### 2.1 Standard hit-or-miss metrics

#### 2.1.1 Frame-level metrics

Frame tests compare note grids (pitches by time steps, for example 10 ms hops) between truth and guess. Precision, recall, and F1 count active vs silent frames. Precision means the share of guessed frames that are right. Recall means the share of true frames you found. F1 blends the two.[^5][^1]

Frame scores are:
- Simple and widely used (MIREX, Onsets and Frames, MT3, Kong et al.).
- Affected by note length and splits, but blind to notes as musical events.
- Known to reward smeared note activity too much and punish short false notes too little.[^5][^2]

#### 2.1.2 Note-level metrics

Note scores treat notes as groups of start time, end time, pitch, and (at times) loudness. Matching rules come from mir_eval.[^5][^2] Mir_eval is a standard music-scoring library.

Standard types are start-only note F1, start-plus-end note F1, and start-plus-end-plus-loudness F1. Start-only means starts must fall within ±50 ms of the true start, and ends are ignored. Ms means milliseconds. Start-plus-end adds ends within the larger of 20% of true length or 50 ms. Start-plus-end-plus-loudness adds loudness within 0.1 in scaled loudness after straight-line rescaling.[^13][^5][^2]

Teams score each piece and then average. These are the main reported scores in Onsets and Frames, Kong et al., MT3, Toyama's hFT, and most MAESTRO-based work.[^3][^8][^5][^2]

Note scores mean more musically than frame scores. But they still treat all note errors the same. They ignore voice links, spelling, or musical role (tune vs backing).[^5]


### 2.2 Musical and notation-aware metrics

#### 2.2.1 Musical performance metrics (mpteval)

Hu et al. (2024) say standard hit-or-miss scores skip musical parts like playing style, loudness, fine timing, and chord context. They offer musical scores in the mpteval library.[^18][^5] Mpteval means musical piano transcription evaluation.

Scores compare time lines drawn from true and guessed MIDI for each expressive part. Timing covers gaps between note starts (IOI means inter-onset interval) for tune and backing lines. Playing style covers key-overlap ratio (KOR means how much notes overlap, a legato measure) for tune and bass, plus tune-vs-bass ratio. Harmony covers Cloud Diameter and Cloud Momentum from Chew's spiral pitch model, scored over sliding windows. Loudness covers tune-to-bass loudness ratio from a simple loudness-from-velocity rule.[^5]

These scores give links from −1 to 1, where 1 means perfect match. Hu et al. show models that look tied on F1 can split wide on timing, style, and loudness. These musical scores also tell models apart better under audio changes (reverb, noise) than hit-or-miss scores.[^5]

#### 2.2.2 Notation accuracy metric (Cogliati-Duan)

As noted above, Cogliati and Duan's score sets an edit distance over high-level notation parts. It first lines up two scores by pitch with dynamic time warping.[^7][^17] Dynamic time warping lines up two time lines that drift.

This score gives 12 per-part error counts. You can scale them and add them with weights to guess human ratings for pitch writing, rhythm writing, and note place. The authors share code and a set of transcriptions rated by music theorists.[^7]

Use this notation score when your chain outputs full notation (MusicXML). It fits most when note values, spelling, and voice slots matter.

#### 2.2.3 Sense-based metrics (PEAMT)

Hu et al. also name PEAMT (Ycart et al.) as a hearing-based piano score. In their tests, PEAMT tracks most with frame-level F1 and their harmony Cloud Momentum score. This hints listeners weigh chord context a lot.[^5]

PEAMT is not yet standard. But it may help as an extra sense-based check if you have it.


### 2.3 Strength scores for changed audio

#### 2.3.1 New-room F1 and drop checks

Edwards et al. (2024) study strength and sound changes. They retrain Kong et al.'s timing model on remade MAESTRO (Studio MAESTRO) and changed variants. They test note-start F1 on MAPS with no MAPS training.[^6][^8] Out-of-distribution (OOD) means tested on a new set the model never trained on.

Common practice:
- Report note-start F1 on both known sets (MAESTRO, Studio MAESTRO) and new sets (MAPS).
- Compare models trained with and without sound changes and with each change subset.
- Measure the drop from clean to changed audio at test time, for example with added noise, EQ, pitch shifts, or reverb.[^6]

They also run one-change tests and leave-one-out tests. One-change tests train with only one change (background noise, pitch shift, reverb, EQ) to gauge each change's pull on new-set F1. Leave-one-out tests train with all changes, then drop one part at a time. Ablation means removing one part to see what it did.[^6]

This gives clear response scores. For example, dropping pitch-shift or reverb cut new-set note-start F1 by about 3-4 points on MAPS. EQ and background noise moved it less.[^6]

#### 2.3.2 Musical strength under sound changes

Hu et al. use their MPTEVAL musical scores to test Onsets and Frames, Kong's model, and a Transformer model. They use MAESTRO remade on a Disklavier player piano and fake-changed audio with several reverb and noise levels.[^5]

They find:
- Standard note F1 falls under changes, but it tells models apart less well than timing and style scores.
- For example, tune timing links and overlap scores show wide gaps in timing and style care across models under echo and noise. F1 alone hides these gaps.[^5] IOI means gap between note starts. KOR means note-overlap ratio.

So a strength study about expressive playing should pair F1 with these musical scores.

#### 2.3.3 Dynamic time warping (DTW) scores

Bradshaw et al. use DTW between source audio and remade transcription audio to score quality and to sift made training data.[^4] DTW lines up two clips in time and measures the gap.

They show:
- DTW tracks human quality ratings from 1 to 5 very well (Spearman −0.88, where −1 means perfect reverse link, since lower DTW is better).[^4]
- DTW tracks mir_eval F1 scores well, most of all start F1.[^4]
- DTW holds up well to recording quality (reverb, noise) in piano clips, likely because note starts rule the score.[^4]

In a sound-change chain, DTW tells how far the changed audio drifts, after transcription and remaking, from the source.


## 3. Sound changes in AMT and their scores

### 3.1 Change methods in strong piano transcription

Edwards et al. (2024) give a full change chain built with the audiomentations library. Audiomentations is a Python sound-change tool. Core parts are:[^6]
- Two random 7-band tone controls (EQ).
- Added background noise from pub and café clips, with varied SNR. SNR means signal-to-noise ratio, how loud music is next to noise.
- Small random pitch shifts (±0.1 semitone, where a semitone is one piano-key step) to stop overfit to tuning.
- Room echo from many real room records (RIR means room impulse response).

Training blends source MAESTRO audio, remade Studio MAESTRO, and six Pianoteq-made versions per piece, picked with set odds.[^6] Pianoteq is a piano sound tool.

They show:
- With no changes, a model trained on Studio MAESTRO fits too tight. Note-start F1 falls from 97.3 (Studio MAESTRO test) to 80.8 (source MAESTRO test) and lower for new-set MAPS.[^6]
- Changes plus varied tone give top new-set note-start F1 of 88.4 on MAPS with no MAPS training.[^8]

One-change and leave-one-out tests weigh each part. Pitch shift and reverb help strength most. Background noise and EQ help MAPS less.[^6]


### 3.2 Other sound-change AMT work

The same paper sums up past change use:
- MAESTRO's first paper stresses sound changes (noise, reverb, squeeze, synth remakes) for training Onsets and Frames. But leave-one-out tests showed small gains, likely due to set size and scope.[^2][^6]
- Thickstun et al. and Lu et al. use pitch-shift that keeps labels true and cross-set mixes for multi-instrument work and rare instruments.[^6]

Past piano, broad audio-change guides (for example torchaudio's) cover room-echo fakes with RIRs and noise set by SNR. These fit your make-plus-change step at once.[^19] RIR means room echo record. SNR sets how loud noise is.

Bradshaw et al. go further. They join Pianoteq-made sound with heavy sound-picture masks, RIRs, noise, live EQ, and tuning drift. Then they grow more training data in a loop, still scored with standard F1 and DTW.[^4]

### 3.3 Scores used in change studies

Across these change-focused works, teams report note-start hit, miss, and blend scores (precision, recall, and F1) on new sets (MAPS, remade MAESTRO). OOD means a set the model never trained on.[^8][^6]
- At times, full note-with-end F1 on MAESTRO and new-set data.[^8][^5]
- Drop tables showing F1 loss under each test-time change on its own, for example background noise, EQ (tone controls), pitch shift, or reverb (room echo) alone.[^6]
- For musical scores, timing, style, harmony, and loudness links before and after changes.[^5]
- DTW between source and remade audio as a smooth quality score.[^4]

So for a sound-change study, report three things. Report plain F1 on new sets. Report F1 drop per change. Report musical link scores.


## 4. Best score set for your chain

Your chain is score file to sound to sound change to transcription to scoring. Symbolic means the score file. Past work points to three score levels.

### 4.1 Core note-match scores

These match guessed MIDI to source MIDI and skip notation style. MIDI is a digital score.

Frame precision, recall, and F1 work for sanity checks and compare with older papers. Note scores use mir_eval or a like tool, a standard music scorer:
  - Start-only F1 (±50 ms) as a base. Ms means milliseconds.
  - Start-plus-end F1 with 20% or 50 ms rule. Use the larger of the two.
  - Start-plus-end-plus-loudness F1 where your tools write loudness and you can steer loudness in sound making.[^20][^2]

For a sound-change study, stress start-plus-end and start-plus-end-plus-loudness F1. This tracks Hawthorne et al.'s urging and later MAESTRO top-paper use.[^2][^5] SOTA means state of the art, the best known.


### 4.2 Musical expressive scores

Use these to see how sound changes and tools shape expressive music past plain right or wrong:

MPTEVAL scores from Hu et al. MPTEVAL means musical piano transcription evaluation:
  - Tune timing links and backing timing links (IOI means gap between note starts).
  - Tune overlap, bass overlap, and their ratio for playing style (KOR means note-overlap ratio).
  - Cloud Diameter and Cloud Momentum for key and chord color.
  - Loudness (tune-to-bass loudness ratio).[^18][^5]

Sonitra draws these from paired true and guessed MIDI. They fit best when you start from clean scores and study playing style.

At will, PEAMT works as a hearing-based score, if you want one number that tries to sum what stands out to ears.[^5]


### 4.3 Notation and whole-track strength scores

For full notation and whole-track quality:

Notation edit score (Cogliati-Duan) on MusicXML scores made from true and guessed MIDI with the same note-rounding and print chain. MusicXML is a notation file format. Use it if you care about read quality and high-level rightness (key marks, beams, staff slots).[^17][^7]
DTW likeness between source made audio and remade transcription audio. DTW lines up two clips in time. Use it as one number that catches both pitch and rhythm slips at sound level.[^4]

This DTW score can (a) help weed out very bad outputs in big runs, and (b) pair with F1 when you ask how sound changes shift tone color with no big hit to note scores.

### 4.4 Response and setting-effect checks

To study "sound-setting effects":

- Set test axes (for example SNR, RT60, EQ slope, wobble depth) and draw F1 and musical score drop curves across setting sweeps. SNR means how loud music is next to noise. RT60 means how long room echo takes to fade 60 dB. EQ slope means tone tilt.
- Work out slopes or effect sizes (for example F1 drop per dB of noise, per 0.1 semitone random drift) to weigh strength. dB means decibel, a loudness step. A semitone is one piano-key step.
- If you use DTW, draw DTW-against-setting curves next to F1. This shows whether some changes mostly shift heard likeness and not note scores.

This tracks Edwards et al. (sound-harm and leave-one-out tables) and Hu et al. (grid test over reverb and noise mixes).[^6][^5]


## 5. Change methods and scores in past work

### 5.1 Key change-focused studies

- Onsets and Frames (Hawthorne et al.) note sound changes (loudness fix, reverb, squeeze, noise, other synth sounds). But they say changes did not move their MAPS scores much. They still urge better sets with more sounds over more changes.[^2] Squeeze means dynamic compression.

- Edwards et al. 2024 (A Data-Driven Analysis of Robust Automatic Piano Transcription):
  - Bring Studio MAESTRO (Disklavier remakes, a player piano) and extra Pianoteq-made sounds. Pianoteq is a piano sound tool.
  - Study the change chain (EQ, noise, pitch shift, reverb) and its pull on new-set note-start F1, most of all on MAPS. EQ means tone controls.
  - Use note-start F1 and per-change drop tables as main scores.[^8][^6]

- Bradshaw et al. 2024 (Aria-AMT):
  - Use wide changes (room records, noise, EQ, tuning drift, sound-picture masks) and large made pretraining. RIR means room echo record.
  - Test with mir_eval F1 scores on MAESTRO, MAPS, and strongly changed variants. Mir_eval is a standard music scorer.
  - Bring DTW as a score and show tight links to human ratings and F1. DTW lines up two clips in time.[^4]

- Hu et al. 2024 (musical scores):
  - Add noise and many reverb settings to remade MAESTRO. Show their link-based scores sense musical quality better than F1 alone under change.[^5]


### 5.2 Takeaways for your study

- Note-start and note-with-end F1 stay the main test for plain note rightness, most of all for piano and MAESTRO or MAPS-like data.[^8][^2][^6]
- Musical scores pay off when your question covers expressive playing. They also pay off when audio changes may bend timing or style more than plain note hits.[^5]
- DTW pairs with note scores in big and strength runs, most of all when you score made vs changed audio.[^4]
- Report new-set results (for example train on one set, test on one more, or on remade or changed audio). This guards against overfit to one room sound.[^6][^5] Overfit means too tied to training sound.


## 6. Candidate AMT systems to test

AMT means turning audio into notes.

### 6.1 Open research tools

For a research-grade test, cover each design group:

- Onsets and Frames (Magenta and Google Brain code): standard piano base, strong on MAESTRO and MAPS, known scores and open code.[^9][^2]
- Kong et al.'s high-detail timing model (Bytedance piano tool): strong piano scores and used in strength studies.[^8][^6]
- MT3 (multi-task multitrack Transformer): modern multi-instrument AMT with T5 design. A Transformer tracks long time links. Open code through Magenta and PyTorch ports.[^21][^22][^3]
- Basic Pitch (Spotify): light pitch and note tracker built for speed and many instruments. Open code and good as a fast base.[^23][^11]
- Aria-AMT or like step-by-step Whisper-style models, if code is open: built for strength and set growth. Strong fit for a sound-change study.[^4] Seq2seq means step-by-step, reading in sound and writing out notes.
- Other newer research tools (for example NoteEM, no-align training methods) that claim strong cross-set results and strong note-with-instrument F1.[^24]


### 6.2 Shop tools

Often named shop AMT tools are Melodyne, AudioScore, ScoreCloud, AnthemScore, and Transcribe!.[^10][^12][^1]

These turn audio into MIDI or notation. But they most often hide inside scores. So use them as closed baselines where you score their outputs yourself. Closed means you see outputs, not inside workings.

For example:
- AnthemScore: transcription for full tracks, with desktop and web versions.[^10]
- Melodyne: widely used in studios for pitch and time fixes. It can save MIDI and often tracks single and chord pitch.[^12]
- ScoreCloud: live audio-to-notation tool that stresses player-friendly scores.[^12]

Adding one or two of these lets you weigh research models against tools players truly use.


### 6.3 Pragmatic pick for your chain

Your chain stresses made piano sound (MIDI and MusicXML to sound to change to transcription). MusicXML is a notation file. MIDI is a digital score. Piano-tuned and multi-track note tools fit best:
- At least one strong piano-tuned model (Kong, Onsets and Frames, or Edwards' retrained form).[^2][^8][^6]
- At least one multi-instrument model (MT3 or hFT) to test reach past piano sound, most of all if you later add more instruments.[^3][^4]
- A light open general tool (Basic Pitch) as a fast base.[^11]
- At will, Aria-AMT or like model for strength-focused step-by-step compare.[^4]
- One shop tool (AnthemScore or Melodyne) for a real-world base.[^10][^12]


## 7. How these scores fit your chain steps

Your chain is:

1. Start with score files (MIDI and MusicXML). MIDI is a digital score. MusicXML is a notation file.
2. Make audio with set sound controls (tone color, room echo, mic place, style). Timbre means tone color.
3. Change the sound on purpose (noise, room-echo records, tone controls, style shifts, at-will track split). RIR means room echo record.
4. Run transcription tools.
5. Match outputs to source scores and weigh how sound and change settings moved results.

Use scores in layers:

- Core rightness: note-start, note-with-end, and note-with-end-plus-loudness F1 plus frame F1. F1 blends missed and extra notes.
- Expressive truth: musical scores (timing gaps called IOI, overlap style called KOR, loudness, harmony) between source and guessed MIDI.
- Notation quality: Cogliati-Duan notation score on MusicXML where you have it.
- Audio likeness: DTW between source made audio and remade audio from the transcription. DTW lines up two clips in time.
- Strength curves: F1 and expressive scores drawn against change settings (SNR, RT60, EQ slopes, and more), with leave-one-out change tests. SNR means music-vs-noise loudness. RT60 means echo fade time. EQ means tone controls. Ablation means dropping one part to test it.

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

19. [Audio Data Augmentation: Torchaudio 2.10.0 ...](https://docs.pytorch.org/audio/stable/tutorials/audio_data_augmentation_tutorial.html) - torchaudio provides a variety of ways to augment audio data. In this tutorial, we look into a way to...

20. [Automatic Music Transcription: An Overview](https://labsites.rochester.edu/air/publications/benetatos19automaticmusic.pdf) - It involves perception (analyzing complex auditory scenes), cog- nition (recognizing musical objects...

21. [magenta/mt3 - Multi-Task Multitrack Music Transcription](https://github.com/magenta/mt3) - MT3 is a multi-instrument automatic music transcription model that uses the T5X framework. This is n...

22. [MT3 (Multi-Task Multitrack Music Transcription)](https://ai4culture.eu/resources/tools/44) - MT3 is a is a multi-instrument automatic music transcription model that can infer musical notes from...

23. [Rachel Bittner on Basic Pitch: An Open Source Tool for ...](https://newsroom.spotify.com/2022-09-01/rachel-bittner-on-basic-pitch-an-open-source-tool-for-musicians/) - We named the project Basic Pitch because it can also detect pitch bends in the notes, which is a par...

24. [Unaligned Supervision for Automatic Music Transcription in ...](https://benadar293.github.io) - Current AMT approaches are restricted to piano and (some) guitar recordings, due to difficult data c...

