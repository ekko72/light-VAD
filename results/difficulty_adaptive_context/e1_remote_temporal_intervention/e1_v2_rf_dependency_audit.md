# E1 RF384 dependency audit

This audit is derived from the loaded frozen checkpoint and the actual `SparseCausalDepthwiseBranch.forward_sparse` implementation.

## Actual branch topology

- Kernel size: `5`
- Dilations: `[1, 2, 4, 8, 96]`
- Branch tap map: `{'1': [4, 3, 2, 1, 0], '2': [8, 6, 4, 2, 0], '4': [16, 12, 8, 4, 0], '8': [32, 24, 16, 8, 0], '96': [384, 288, 192, 96, 0]}`
- Union of accessed history offsets: `[0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 96, 192, 288, 384]`
- Local support used by the first four branches: `[0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32]`
- Remote support from the final branch: `[96, 192, 288, 384]`

The target frame `t` uses encoded frame `t - q` for every listed history offset `q`; the current frame is `q=0`.

## Padding and boundaries

- Each branch uses causal left zero padding equal to `(kernel_size - 1) * dilation`.
- Nominal and actual maximum lookback: `384` frames.
- For targets `382` and `383`, `t - 384` is left padding, so only remote taps `96`, `192`, and `288` are available.
- For every target `t >= 384`, all four remote taps are available.

## Protocol note

The nominal RF384 span is 384 frames, but the actual accessed set is sparse. In particular, offsets `48` and `64` are not accessed by the RF384 profile. They must not be silently treated as local history. The E1 local support is therefore the exact union of the first four branch taps, and the remote support is exactly the four nonzero taps of the dilation-96 branch.
