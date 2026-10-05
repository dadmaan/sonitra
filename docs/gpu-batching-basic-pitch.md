# GPU batching for basic_pitch

## Question and verdict

The question was whether batched graphics hardware transcribe beats CPU with batch 16 on full wall time for the same long input. The question also asked at which batch size the note output still matches.

Wall time means clock time from start to finish for one transcribe call. It is measured in seconds.

Batch size means how many audio windows go through the model in one group.

Verdict: Outcome 1 with a floor adjusted reading. GPU batching is permitted at batch 8.

Batch 16 had the lowest median wall time. It changed notes, so it was excluded as faster but different. Batch 8 kept notes identical to batch 1 on the same device. Batch 8 beat CPU batch 16 with no overlap in ranges.

The long input lasted 363.0 s with 222 windows. Numeric mode was off.

## Terms used in this report

- Batch size is the count of audio windows grouped in one model call.
- Wall time is clock time from start to finish. Unit is seconds.
- Median is the middle value of the 3 timed runs. Range is the span from min to max. Tables show median with range in parentheses.
- Note-set equality is an exact match of note lists after rounding. Start values round to 3 decimals. Duration values round to 3 decimals. Pitch must match as an integer.
- dF1 is 1 minus onset F1 between two estimated note sets. F1 is a similarity score from 0 to 1. dF1 of 0 means identical output. No ground truth MIDI file exists for the long input, so dF1 here compares estimates to estimates.
- OOM means out of memory. The run fails because data does not fit in memory.
- CUDA sync means an explicit wait for the graphics hardware to finish before the clock stops. It keeps graphics timing honest. CUDA is the NVIDIA software path that runs model code on the graphics card.
- Warmup means one untimed run before measurement. It lets caches and hardware settle. Each setting then ran 3 timed runs.
- End to end (e2e) means the full transcribe call. CPU means the main processor. GPU means the graphics hardware. TF means the TensorFlow library used here.

## Setup fingerprint

- Graphics card was NVIDIA GeForce RTX 4090 Laptop GPU with 13510 MB of memory. Driver version was 581.29.
- TF version was 2.15.0 with CUDA 12.2 and cuDNN 8. Python version was 3.11.16.
- Code commit was 3c53295. The harness check guard_ok was true on every run. No production file was edited.
- Numeric mode was off. The loader was read_audio_basic_pitch. It stayed constant for all runs.
- gpu_memory_growth was false. Batch size came only from the harness setting. The production guard line stayed in place.
- Long input file was inputs/long_render.wav. Checksum md5 was db83146d902aebe516c885471dd6c3d1. Duration was 363.0 s with 222 windows.
- Short set held 20 files under inputs/bsed20/. The files span 5 distortion conditions over piano stems 1 to 4. All 20 checksums differ.
- Each setting ran in one fresh interpreter process. The long input ran first. Batches ran in ascending order. Devices alternated between cpu and cuda.
- All 31 timed configurations finished. No OOM occurred.

## Batch curves

The report splits wall time into parts. Load time (t_load) covers audio reading and prep. Compute time (t_infer) covers the model predict loop only. Post time (t_post) covers note assembly.

Long input e2e medians in seconds for batch 1 to 64:

| Batch | CPU e2e | CUDA e2e |
| ----- | ------- | -------- |
| 1     | 6.38    | 4.51     |
| 2     | 5.26    | 4.39     |
| 4     | 4.72    | 4.14     |
| 8     | 4.47    | 3.89     |
| 16    | 4.12    | 3.89     |
| 32    | 4.32    | 4.09     |
| 64    | 4.34    | 4.23     |

Short set sums in seconds over 20 files per pass for batch 1 to 64:

| Batch | CPU sum | CUDA sum |
| ----- | ------- | -------- |
| 1     | 5.59    | 2.99     |
| 2     | 4.04    | 2.70     |
| 4     | 3.42    | 2.47     |
| 8     | 3.15    | 2.41     |
| 16    | 2.84    | 2.64     |
| 32    | 2.91    | 2.53     |
| 64    | 2.71    | 2.53     |

Compute time fell as batch grew, then it flattened. Load time stayed near 1.26 to 1.30 s on the long input. Post time stayed near 2.2 to 2.3 s on the long input. Post time took 57 to 59 percent of e2e. That share caps the gain from faster compute.

## Head to head

| Setting       | e2e median (min, max) in seconds | t_infer median in seconds |
| ------------- | -------------------------------- | ------------------------- |
| CUDA batch 8  | 3.887 (3.784, 3.987)             | 0.38                      |
| CUDA batch 16 | 3.886 (3.823, 4.043)             | 0.38                      |
| CPU batch 16  | 4.117 (4.106, 4.265)             | 0.70                      |
| CUDA batch 1  | 4.514 (4.436, 4.527)             | 0.95                      |

GPU batch 8 beat CPU batch 16 by 1.059x. The medians were 3.887 s against 4.117 s. Ranges did not overlap.

Batch 8 sat 0.6 ms above batch 16. Ranges overlapped. Fidelity decided the pick.

## Equivalence

Largest activation gap (maxdiff) is the biggest absolute gap in model output values. Note-set equality and dF1 are defined in the terms section above.

On the long input, CPU output was identical at every batch. Largest activation gap stayed at or below 8e-7.

On CUDA, batch 2 to batch 8 matched batch 1 exactly. Largest activation gap stayed at or below 1.1e-3. dF1 was 0.

Batch 16 and batch 32 differed from batch 1. dF1 was 0.0011. Largest activation gap was 1.19e-2.

Batch 64 differed more. dF1 was 0.0014. Largest activation gap was 1.29e-2.

CUDA differed from CPU even at batch 1. The floor gap was dF1 0.00056. It stayed constant for batch 1 to batch 8.

On the 20 file set, CUDA batch 16 and batch 64 each flipped exactly 1 file in 20. The file was drive=12.0__piano2.wav. Counts were 123 notes against 122. dF1 was 0.00408.

Strict numeric mode removed all divergence. It cost plus 30.4 percent (5.067 s against 3.886 s). That cost removes the speed win.

## Memory

MB means megabytes.

Peak graphics memory on the long input by batch:

| Batch | Peak GPU memory in MB |
| ----- | --------------------- |
| 1     | 75                    |
| 2     | 95                    |
| 4     | 157                   |
| 8     | 279                   |
| 16    | 525                   |
| 32    | 1016                  |
| 64    | 1998                  |

Peak memory grew with batch size. Batch 8 used 279 MB. Batch 64 used 1998 MB. No setting ran out of memory.

## Reproduction

Full sweep command:

```
uv run --no-sync python misc/20260922_gpu_batch_measure/sweep.py sweep --input both --numeric off
```

Single setting command:

```
uv run --no-sync python misc/20260922_gpu_batch_measure/sweep.py cell --device cuda --batch 8 --numeric off --input L
```

CUDA runs waited with tf.reduce_sum(tf.zeros((1024,), tf.float32)).numpy() inside tf.device(tf_device) after the predict loop. Predict to numpy materialisation also serialised each batch.

Page cache was not dropped. Provisioning reads were the cold touch. All timed runs were warm.

Data manifest under misc/20260922_gpu_batch_measure/:

- run_header.json at misc/20260922_gpu_batch_measure/run_header.json holds the full fingerprint.
- data/summary.json holds per setting medians with min and max plus speedups.
- data/strict_rerun.json holds the strict rerun timings with zeroing results.
- Long input curve files are data/batch_curve_L_cpu.csv with data/batch_curve_L_cuda.csv.
- Short set curve files are data/batch_curve_B_cpu.csv with data/batch_curve_B_cuda.csv.
- Parity files are data/parity_L.csv with data/parity_B.csv.
- Per run phase rows match data/phase_times_L_cpu_8_off.jsonl style paths, one file per input and device and batch and numeric mode.
- Per pass note files match data/notes_L_cuda_8_off_p0_long_render.wav.json style paths. Activation references are data/refacts_L_cpu.npz with data/refacts_L_cuda.npz plus the short set pair.
- data/launch_order.json records launch order with return codes.
- inputs/long_render.wav with inputs/long_render.meta.json pins the long input bytes with duration and window count.
- inputs/bsed20/ with inputs/bsed20.meta.json pins the 20 short files with stems and conditions.
- Harness file is misc/20260922_gpu_batch_measure/sweep.py with md5 4f59cfa47063588ec809df6cabf01960.
- Production file src/sonitra/transcribe/basic_pitch.py kept the guard line effective_batch = self.batch_size if "cpu" in tf_device.lower() else 1.

## Limitations and follow ups

- Long input was the existing file at /tmp/x.wav, staged as inputs/long_render.wav. Stem and render provenance is unrecoverable. Checksum pins the bytes.
- Short set holds distortion_sweep piano stems. It is not BSED-20. Timing used 20 transcribe calls per pass. Parity reading covers 20 distinct files over 4 stems.
- Each setting ran in one fresh interpreter. Single setting runs had no pair mate. State could not leak across settings.
- Compute time covers the predict loop only. Unwrap and note assembly count toward post time.
- dF1 compares estimates to estimates. No ground truth MIDI exists for the long input.
- Strict equality to CPU is unreachable in off mode. Even CUDA batch 1 differs from CPU by dF1 0.00056.
- Batch 8 won over batch 16 on fidelity. Medians sat 0.6 ms apart. Ranges overlapped.

The rule as built: an unset `batch_size` means 1 on a GPU and 16 on the CPU, and a value set in the config applies on every device with no cap. Each row records the batch that ran as `effective_batch` in `transcriber_metadata`, so a published number can be checked against the setting that produced it.
