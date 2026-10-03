# Copyright (c) 2023 Sony Group Corporation. Licensed under the MIT License; see LICENSE.

"""hFT-Transformer log-mel features, windowed inference and note decoding.

Vendored from the upstream MIT release; see LICENSE. The caller owns the model, its
device and the mel transform, so this module imports torch lazily and stays usable
without the optional extra.

Every function takes the checkpoint's `inference` mapping, extended with the
architecture dimensions the windowing needs (`n_bins`, `margin_b`, `margin_f`,
`num_frame`, `num_note`); a missing key is named in the error.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, Sequence

if TYPE_CHECKING:
    import numpy as np
    import torch
    import torch.nn as nn

__all__ = [
    "make_mel_transform",
    "mpe2note",
    "transcript",
    "transcript_stride",
    "wav2feature",
]

_FEATURE_KEYS = (
    "sr",
    "fft_bins",
    "window_length",
    "hop_sample",
    "pad_mode",
    "mel_bins",
    "mel_norm",
    "melfilter",
)

_WINDOW_KEYS = ("n_bins", "margin_b", "margin_f", "num_frame", "min_value", "num_note")


def _require(config: Mapping[str, Any], keys: Sequence[str], caller: str) -> list[Any]:
    missing = [key for key in keys if key not in config]
    if missing:
        raise KeyError(
            f"{caller}() needs {', '.join(keys)} in its config mapping; "
            f"missing {', '.join(missing)}"
        )
    return [config[key] for key in keys]


def make_mel_transform(inference: Mapping[str, Any], device: str) -> torch.nn.Module:
    """Build the log-mel analysis transform once so it is not rebuilt per file.

    `norm` and `mel_scale` are passed explicitly because they are not settings in
    upstream's configuration file: `norm='slaney'` and the htk mel filter are
    hard-coded at the call site upstream, so both are pinned here to keep the
    measured feature definition with the checkpoint.
    """
    import torchaudio

    (
        sr,
        fft_bins,
        window_length,
        hop_sample,
        pad_mode,
        mel_bins,
        mel_norm,
        melfilter,
    ) = _require(inference, _FEATURE_KEYS, "make_mel_transform")

    # The configuration's `window` key only names torchaudio's default hann analysis
    # window, which MelSpectrogram already selects, so there is nothing to pass.
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=fft_bins,
        win_length=window_length,
        hop_length=hop_sample,
        pad_mode=pad_mode,
        n_mels=mel_bins,
        norm=mel_norm,
        mel_scale=melfilter,
    ).to(device)


def wav2feature(
    wave: torch.Tensor, inference: Mapping[str, Any], *, mel: torch.nn.Module
) -> np.ndarray:
    """Turn one mono float32 waveform at `inference["sr"]` into `[frames, mel_bins]`.

    Upstream loaded the file itself and resampled it here. The waveform arrives
    already mono and already at the target rate, because loading through
    `torchaudio.load` now depends on a separate codec backend that is not installed.
    """
    import torch

    if wave.dim() != 1:
        raise ValueError(
            f"expected a mono waveform of shape [samples], got {tuple(wave.shape)}; "
            "mean over channels before calling wav2feature"
        )
    (log_offset,) = _require(inference, ("log_offset",), "wav2feature")

    mel_spec = mel(wave)
    a_feature = (torch.log(mel_spec + log_offset)).T

    # upstream deviation: the return type is a numpy array rather than the torch tensor
    # it produced, so the result has to be copied off an accelerator first.
    return a_feature.detach().cpu().numpy()


def transcript(
    features: np.ndarray,
    model: torch.nn.Module,
    *,
    inference: Mapping[str, Any],
    device: str,
    batch_size: int = 1,
    mode: str = "combination",
) -> tuple[np.ndarray, ...]:
    """Run the decoder over every `num_frame` window of `features`.

    Returns `(onset, offset, mpe, velocity)` per decoder head — eight arrays for
    `mode="combination"`, four otherwise. The onset, offset and MPE arrays are
    `float32` and the velocity array is the `argmax` over the velocity bins, stored
    as `int8`. `batch_size` groups that many windows into one forward pass; 1
    reproduces the reference output exactly, larger values only change the
    floating-point reduction order.
    """
    import numpy as np
    import torch

    n_bins, margin_b, margin_f, num_frame, min_value, num_note = _require(
        inference, _WINDOW_KEYS, "transcript"
    )
    combination = mode == "combination"
    # a_feature: [num_frame, n_mels]
    a_feature = np.array(features, dtype=np.float32)

    a_tmp_b = np.full([margin_b, n_bins], min_value, dtype=np.float32)
    len_s = int(np.ceil(a_feature.shape[0] / num_frame) * num_frame) - a_feature.shape[0]
    a_tmp_f = np.full([len_s + margin_f, n_bins], min_value, dtype=np.float32)
    a_input = np.concatenate([a_tmp_b, a_feature, a_tmp_f], axis=0)
    # a_input: [margin_b+a_feature.shape[0]+len_s+margin_f, n_bins]

    a_output_onset_A = np.zeros((a_feature.shape[0] + len_s, num_note), dtype=np.float32)
    a_output_offset_A = np.zeros((a_feature.shape[0] + len_s, num_note), dtype=np.float32)
    a_output_mpe_A = np.zeros((a_feature.shape[0] + len_s, num_note), dtype=np.float32)
    a_output_velocity_A = np.zeros((a_feature.shape[0] + len_s, num_note), dtype=np.int8)

    if combination:
        a_output_onset_B = np.zeros((a_feature.shape[0] + len_s, num_note), dtype=np.float32)
        a_output_offset_B = np.zeros((a_feature.shape[0] + len_s, num_note), dtype=np.float32)
        a_output_mpe_B = np.zeros((a_feature.shape[0] + len_s, num_note), dtype=np.float32)
        a_output_velocity_B = np.zeros((a_feature.shape[0] + len_s, num_note), dtype=np.int8)

    window_len = margin_b + num_frame + margin_f

    model.eval()
    for i in range(0, a_feature.shape[0], num_frame):
        # upstream deviation: windows are grouped so one forward sees `batch_size`
        # of them; upstream always passed exactly one, and `batch_size=1` keeps that.
        begins = [i + step * num_frame for step in range(batch_size)]
        begins = [begin for begin in begins if begin < a_feature.shape[0]]
        input_spec = torch.from_numpy(
            np.stack([a_input[begin:begin + window_len].T for begin in begins])
        ).to(device)

        with torch.no_grad():
            if combination:
                # attention sits at index 4 and is discarded; the A/B heads are fused.
                (
                    output_onset_A,
                    output_offset_A,
                    output_mpe_A,
                    output_velocity_A,
                    attention,
                    output_onset_B,
                    output_offset_B,
                    output_mpe_B,
                    output_velocity_B,
                ) = model(input_spec)
                # output_onset: [batch_size, n_frame, n_note]
                # output_offset: [batch_size, n_frame, n_note]
                # output_mpe: [batch_size, n_frame, n_note]
                # output_velocity: [batch_size, n_frame, n_note, n_velocity]
            else:
                output_onset_A, output_offset_A, output_mpe_A, output_velocity_A = model(
                    input_spec
                )

        for step in range(len(begins)):
            index = i + step * num_frame
            rows = slice(index, index + num_frame)
            # indexing one window replaces upstream's `squeeze(0)`: it drops the batch
            # axis, so the assignment into the output arrays is unchanged.
            a_output_onset_A[rows] = (output_onset_A[step]).to("cpu").detach().numpy()
            a_output_offset_A[rows] = (output_offset_A[step]).to("cpu").detach().numpy()
            a_output_mpe_A[rows] = (output_mpe_A[step]).to("cpu").detach().numpy()
            a_output_velocity_A[rows] = (
                output_velocity_A[step].argmax(2)
            ).to("cpu").detach().numpy()

            if combination:
                a_output_onset_B[rows] = (output_onset_B[step]).to("cpu").detach().numpy()
                a_output_offset_B[rows] = (output_offset_B[step]).to("cpu").detach().numpy()
                a_output_mpe_B[rows] = (output_mpe_B[step]).to("cpu").detach().numpy()
                a_output_velocity_B[rows] = (
                    output_velocity_B[step].argmax(2)
                ).to("cpu").detach().numpy()

    if combination:
        return (
            a_output_onset_A,
            a_output_offset_A,
            a_output_mpe_A,
            a_output_velocity_A,
            a_output_onset_B,
            a_output_offset_B,
            a_output_mpe_B,
            a_output_velocity_B,
        )
    return a_output_onset_A, a_output_offset_A, a_output_mpe_A, a_output_velocity_A


def transcript_stride(
    features: np.ndarray,
    model: torch.nn.Module,
    *,
    inference: Mapping[str, Any],
    device: str,
    n_offset: int,
    batch_size: int = 1,
    mode: str = "combination",
) -> tuple[np.ndarray, ...]:
    """Run the decoder every half-window, keeping only `n_offset`..`n_offset+half`.

    Higher overlap makes onsets sharper, at proportionally more compute. `n_offset`
    must satisfy `n_offset + num_frame // 2 <= num_frame`, because the slice taken
    from the model output is `half_frame` frames long.

    The result is never shorter than the input. Upstream rounds the frame count up to
    a whole number of half-windows, so the tail padding reaches the note detector and
    the last note's fallback offset can sit past the end of the audio by up to about
    two seconds. The padding is exactly zero when the input already holds a whole
    number of half-windows.
    """
    import numpy as np
    import torch

    n_bins, margin_b, margin_f, num_frame, min_value, num_note = _require(
        inference, _WINDOW_KEYS, "transcript_stride"
    )
    combination = mode == "combination"
    # a_feature: [num_frame, n_mels]
    a_feature = np.array(features, dtype=np.float32)

    half_frame = int(num_frame / 2)
    a_tmp_b = np.full([margin_b + n_offset, n_bins], min_value, dtype=np.float32)
    tmp_len = a_feature.shape[0] + margin_b + margin_f + half_frame
    len_s = int(np.ceil(tmp_len / half_frame) * half_frame) - tmp_len
    a_tmp_f = np.full(
        [len_s + margin_f + (half_frame - n_offset), n_bins], min_value, dtype=np.float32
    )

    a_input = np.concatenate([a_tmp_b, a_feature, a_tmp_f], axis=0)
    # a_input: [n_offset+margin_b+a_feature.shape[0]+len_s+(half_frame-n_offset)+margin_f, n_bins]

    a_output_onset_A = np.zeros((a_feature.shape[0] + len_s, num_note), dtype=np.float32)
    a_output_offset_A = np.zeros((a_feature.shape[0] + len_s, num_note), dtype=np.float32)
    a_output_mpe_A = np.zeros((a_feature.shape[0] + len_s, num_note), dtype=np.float32)
    a_output_velocity_A = np.zeros((a_feature.shape[0] + len_s, num_note), dtype=np.int8)

    if combination:
        a_output_onset_B = np.zeros((a_feature.shape[0] + len_s, num_note), dtype=np.float32)
        a_output_offset_B = np.zeros((a_feature.shape[0] + len_s, num_note), dtype=np.float32)
        a_output_mpe_B = np.zeros((a_feature.shape[0] + len_s, num_note), dtype=np.float32)
        a_output_velocity_B = np.zeros((a_feature.shape[0] + len_s, num_note), dtype=np.int8)

    window_len = margin_b + num_frame + margin_f

    model.eval()
    for i in range(0, a_feature.shape[0], half_frame):
        # upstream deviation: as in `transcript`, `batch_size` windows share a forward.
        begins = [i + step * half_frame for step in range(batch_size)]
        begins = [begin for begin in begins if begin < a_feature.shape[0]]
        input_spec = torch.from_numpy(
            np.stack([a_input[begin:begin + window_len].T for begin in begins])
        ).to(device)

        with torch.no_grad():
            if combination:
                (
                    output_onset_A,
                    output_offset_A,
                    output_mpe_A,
                    output_velocity_A,
                    attention,
                    output_onset_B,
                    output_offset_B,
                    output_mpe_B,
                    output_velocity_B,
                ) = model(input_spec)
                # output_onset: [batch_size, n_frame, n_note]
                # output_offset: [batch_size, n_frame, n_note]
                # output_mpe: [batch_size, n_frame, n_note]
                # output_velocity: [batch_size, n_frame, n_note, n_velocity]
            else:
                output_onset_A, output_offset_A, output_mpe_A, output_velocity_A = model(
                    input_spec
                )

        for step in range(len(begins)):
            index = i + step * half_frame
            rows = slice(index, index + half_frame)
            # indexing one window replaces upstream's
            # `squeeze(0)[n_offset:n_offset+half_frame]`.
            keep = slice(n_offset, n_offset + half_frame)
            a_output_onset_A[rows] = (
                (output_onset_A[step])[keep]
            ).to("cpu").detach().numpy()
            a_output_offset_A[rows] = (
                (output_offset_A[step])[keep]
            ).to("cpu").detach().numpy()
            a_output_mpe_A[rows] = (
                (output_mpe_A[step])[keep]
            ).to("cpu").detach().numpy()
            a_output_velocity_A[rows] = (
                (output_velocity_A[step])[keep].argmax(2)
            ).to("cpu").detach().numpy()

            if combination:
                a_output_onset_B[rows] = (
                    (output_onset_B[step])[keep]
                ).to("cpu").detach().numpy()
                a_output_offset_B[rows] = (
                    (output_offset_B[step])[keep]
                ).to("cpu").detach().numpy()
                a_output_mpe_B[rows] = (
                    (output_mpe_B[step])[keep]
                ).to("cpu").detach().numpy()
                a_output_velocity_B[rows] = (
                    (output_velocity_B[step])[keep].argmax(2)
                ).to("cpu").detach().numpy()

    if combination:
        return (
            a_output_onset_A,
            a_output_offset_A,
            a_output_mpe_A,
            a_output_velocity_A,
            a_output_onset_B,
            a_output_offset_B,
            a_output_mpe_B,
            a_output_velocity_B,
        )
    return a_output_onset_A, a_output_offset_A, a_output_mpe_A, a_output_velocity_A


def mpe2note(
    onset: np.ndarray,
    offset: np.ndarray,
    mpe: np.ndarray,
    velocity: np.ndarray,
    *,
    inference: Mapping[str, Any],
    onset_threshold: float = 0.5,
    offset_threshold: float = 0.5,
    mpe_threshold: float = 0.5,
    mode_velocity: str = "ignore_zero",
    mode_offset: str = "shorter",
) -> list[dict]:
    """Turn one head's activation arrays into note dicts, as upstream does.

    Pure Python over the arrays passed in: no torch, no numpy and no model. Each
    note dict has `pitch`, `onset`, `offset` and `velocity`, sorted by onset then
    pitch.

    `mode_velocity` is `ignore_zero` (drop notes whose velocity is 0) or `org`
    (keep 0-127). `mode_offset` picks between the offset and minimum-picked-onset
    activations: `shorter` (default), `longer` or `offset`.
    """
    num_note, note_min, hop_sample, sr = _require(
        inference, ("num_note", "note_min", "hop_sample", "sr"), "mpe2note"
    )

    a_note = []
    hop_sec = float(hop_sample / sr)

    for j in range(num_note):
        # find local maximum
        a_onset_detect = []
        for i in range(len(onset)):
            if onset[i][j] >= onset_threshold:
                left_flag = True
                for ii in range(i - 1, -1, -1):
                    if onset[i][j] > onset[ii][j]:
                        left_flag = True
                        break
                    elif onset[i][j] < onset[ii][j]:
                        left_flag = False
                        break
                right_flag = True
                for ii in range(i + 1, len(onset)):
                    if onset[i][j] > onset[ii][j]:
                        right_flag = True
                        break
                    elif onset[i][j] < onset[ii][j]:
                        right_flag = False
                        break
                if (left_flag is True) and (right_flag is True):
                    if (i == 0) or (i == len(onset) - 1):
                        onset_time = i * hop_sec
                    else:
                        if onset[i - 1][j] == onset[i + 1][j]:
                            onset_time = i * hop_sec
                        elif onset[i - 1][j] > onset[i + 1][j]:
                            onset_time = (
                                i * hop_sec
                                - (
                                    hop_sec
                                    * 0.5
                                    * (onset[i - 1][j] - onset[i + 1][j])
                                    / (onset[i][j] - onset[i + 1][j])
                                )
                            )
                        else:
                            onset_time = (
                                i * hop_sec
                                + (
                                    hop_sec
                                    * 0.5
                                    * (onset[i + 1][j] - onset[i - 1][j])
                                    / (onset[i][j] - onset[i - 1][j])
                                )
                            )
                    a_onset_detect.append({"loc": i, "onset_time": onset_time})
        a_offset_detect = []
        for i in range(len(offset)):
            if offset[i][j] >= offset_threshold:
                left_flag = True
                for ii in range(i - 1, -1, -1):
                    if offset[i][j] > offset[ii][j]:
                        left_flag = True
                        break
                    elif offset[i][j] < offset[ii][j]:
                        left_flag = False
                        break
                right_flag = True
                for ii in range(i + 1, len(offset)):
                    if offset[i][j] > offset[ii][j]:
                        right_flag = True
                        break
                    elif offset[i][j] < offset[ii][j]:
                        right_flag = False
                        break
                if (left_flag is True) and (right_flag is True):
                    if (i == 0) or (i == len(offset) - 1):
                        offset_time = i * hop_sec
                    else:
                        if offset[i - 1][j] == offset[i + 1][j]:
                            offset_time = i * hop_sec
                        elif offset[i - 1][j] > offset[i + 1][j]:
                            offset_time = (
                                i * hop_sec
                                - (
                                    hop_sec
                                    * 0.5
                                    * (offset[i - 1][j] - offset[i + 1][j])
                                    / (offset[i][j] - offset[i + 1][j])
                                )
                            )
                        else:
                            offset_time = (
                                i * hop_sec
                                + (
                                    hop_sec
                                    * 0.5
                                    * (offset[i + 1][j] - offset[i - 1][j])
                                    / (offset[i][j] - offset[i - 1][j])
                                )
                            )
                    a_offset_detect.append({"loc": i, "offset_time": offset_time})

        time_next = 0.0
        time_offset = 0.0
        time_mpe = 0.0
        for idx_on in range(len(a_onset_detect)):
            # onset
            loc_onset = a_onset_detect[idx_on]["loc"]
            time_onset = a_onset_detect[idx_on]["onset_time"]

            if idx_on + 1 < len(a_onset_detect):
                loc_next = a_onset_detect[idx_on + 1]["loc"]
                # time_next = loc_next * hop_sec
                time_next = a_onset_detect[idx_on + 1]["onset_time"]
            else:
                loc_next = len(mpe)
                time_next = (loc_next - 1) * hop_sec

            # offset
            loc_offset = loc_onset + 1
            flag_offset = False
            # time_offset = 0###
            for idx_off in range(len(a_offset_detect)):
                if loc_onset < a_offset_detect[idx_off]["loc"]:
                    loc_offset = a_offset_detect[idx_off]["loc"]
                    time_offset = a_offset_detect[idx_off]["offset_time"]
                    flag_offset = True
                    break
            if loc_offset > loc_next:
                loc_offset = loc_next
                time_offset = time_next

            # offset by MPE
            # (1frame longer)
            loc_mpe = loc_onset + 1
            flag_mpe = False
            # time_mpe = 0###
            for ii_mpe in range(loc_onset + 1, loc_next):
                if mpe[ii_mpe][j] < mpe_threshold:
                    loc_mpe = ii_mpe
                    flag_mpe = True
                    time_mpe = loc_mpe * hop_sec
                    break
            pitch_value = int(j + note_min)
            velocity_value = int(velocity[loc_onset][j])

            if (flag_offset is False) and (flag_mpe is False):
                offset_value = float(time_next)
            elif (flag_offset is True) and (flag_mpe is False):
                offset_value = float(time_offset)
            elif (flag_offset is False) and (flag_mpe is True):
                offset_value = float(time_mpe)
            else:
                if mode_offset == "offset":
                    ## (a) offset
                    offset_value = float(time_offset)
                elif mode_offset == "longer":
                    ## (b) longer
                    if loc_offset >= loc_mpe:
                        offset_value = float(time_offset)
                    else:
                        offset_value = float(time_mpe)
                else:
                    ## (c) shorter
                    if loc_offset <= loc_mpe:
                        offset_value = float(time_offset)
                    else:
                        offset_value = float(time_mpe)
            # upstream deviation: no MIDI writer here, so the note list is the output.
            if mode_velocity != "ignore_zero":
                a_note.append(
                    {
                        "pitch": pitch_value,
                        "onset": float(time_onset),
                        "offset": offset_value,
                        "velocity": velocity_value,
                    }
                )
            else:
                if velocity_value > 0:
                    a_note.append(
                        {
                            "pitch": pitch_value,
                            "onset": float(time_onset),
                            "offset": offset_value,
                            "velocity": velocity_value,
                        }
                    )

            # the velocity-0 drop above happens first, so a zero-velocity note can never
            # be the one whose offset this trim rewrites
            if (
                (len(a_note) > 1)
                and (a_note[len(a_note) - 1]["pitch"] == a_note[len(a_note) - 2]["pitch"])
                and (a_note[len(a_note) - 1]["onset"] < a_note[len(a_note) - 2]["offset"])
            ):
                a_note[len(a_note) - 2]["offset"] = a_note[len(a_note) - 1]["onset"]

    a_note = sorted(sorted(a_note, key=lambda x: x["pitch"]), key=lambda x: x["onset"])
    return a_note