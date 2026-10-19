"""Window construction for the OwlyShield forecasting experiments.

A window is INPUT_LENGTH consecutive rows (model input) followed by H rows
(forecast target), all taken from ONE recording. No window crosses a recording.

Training windows are thinned with the same rule CRAFT used for its training rows
(downsample_train_v4b_seq5.py: "blocks of 100 rows per training recording, keep 1 in 16;
recordings shorter than 100 rows kept whole"), applied to window START positions:
inside each recording, the valid start positions are grouped into consecutive blocks
of 100 and blocks 0, 16, 32, ... are kept. A recording with fewer than 100 valid
starts is one block and is kept whole. No randomness.

Validation and test windows are NOT thinned: every start position (stride 1) is used.
"""
import numpy as np

INPUT_LENGTH = 96
HORIZONS = (96, 192, 336, 720)
BLOCK = 100        # same block size as CRAFT's training downsampling
KEEP_EVERY = 16    # same keep ratio as CRAFT's training downsampling


def all_starts(segments, window):
    """All start positions (stride 1) of windows of `window` rows inside each segment.
    Returns (starts, segment_index) as int64 arrays."""
    starts, owner = [], []
    for k, (a, b) in enumerate(np.asarray(segments, dtype=np.int64)):
        n = int(b - a) - window + 1
        if n > 0:
            starts.append(np.arange(a, a + n, dtype=np.int64))
            owner.append(np.full(n, k, dtype=np.int64))
    if not starts:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    return np.concatenate(starts), np.concatenate(owner)


def thinned_starts(segments, window, block=BLOCK, keep_every=KEEP_EVERY):
    """CRAFT-style thinning of training windows: blocks of `block` consecutive start
    positions per recording, keep blocks 0, keep_every, 2*keep_every, ..."""
    starts, owner = [], []
    for k, (a, b) in enumerate(np.asarray(segments, dtype=np.int64)):
        n = int(b - a) - window + 1
        if n <= 0:
            continue
        for i, s in enumerate(range(0, n, block)):
            if i % keep_every == 0:
                e = min(s + block, n)
                starts.append(np.arange(a + s, a + e, dtype=np.int64))
                owner.append(np.full(e - s, k, dtype=np.int64))
    if not starts:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    return np.concatenate(starts), np.concatenate(owner)
