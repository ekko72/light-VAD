# -*- coding: utf-8 -*-
"""MarbleNet-3x2x64 in pure PyTorch (no NeMo dependency).

Architecture, kernels and channel counts follow:

* Jia, Majumdar, Ginsburg, "MarbleNet: Deep 1D Time-Channel Separable
  Convolutional Neural Network for Voice Activity Detection", ICASSP 2021.
* NVIDIA NeMo ``examples/asr/conf/marblenet/marblenet_3x2x64.yaml``
  (the reference configuration is documented in this directory's README).

The block structure mirrors NeMo's ``JasperBlock``:

    repeat x [ separable conv -> BN -> ReLU -> Dropout ]   (mconv)
    + 1x1 conv -> BN                                        (residual)
    -> ReLU -> Dropout                                      (mout)

Only differences from NeMo: masked convolutions are dropped (every window is
a fixed 0.63 s segment, so there is no padding to mask), and the decoder is an
explicit average-pool + linear head instead of ``ConvASRDecoderClassification``.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


def compute_new_kernel_size(kernel_size: int, kernel_size_factor: float) -> int:
    """NeMo's ``compute_new_kernel_size``: scale then force an odd size."""
    new_kernel_size = max(int(kernel_size * float(kernel_size_factor)), 1)
    if new_kernel_size % 2 == 0:
        new_kernel_size += 1
    return new_kernel_size


def same_padding(
    kernel_size: int, stride: int = 1, dilation: int = 1
) -> int:
    """NeMo's ``get_same_padding`` for stride 1 / dilation >= 1."""
    if stride > 1 and dilation > 1:
        raise ValueError("Only stride OR dilation may be greater than 1")
    return (dilation * (kernel_size - 1)) // 2


def causal_padding(kernel_size: int, dilation: int = 1) -> int:
    """Return the number of left-only samples needed by a causal conv."""
    return int(dilation * (kernel_size - 1))


class CausalConv1d(nn.Conv1d):
    """Conv1d with optional left-only padding and incremental cache support.

    When ``causal`` is false this behaves exactly like ``nn.Conv1d`` and keeps
    the same state-dict keys.  When true, the convolution sees only the
    current and previous frames.  ``forward_with_cache`` implements the same
    operation for a chunked stream: the cache stores the previous left-padding
    frames and is returned for the next chunk.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        dilation: int = 1,
        padding: int = 0,
        groups: int = 1,
        bias: bool = False,
        causal: bool = False,
    ) -> None:
        self.causal = bool(causal)
        self.left_padding = (
            causal_padding(kernel_size, dilation) if self.causal else 0
        )
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            padding=0 if self.causal else padding,
            groups=groups,
            bias=bias,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.causal and self.left_padding:
            inputs = F.pad(inputs, (self.left_padding, 0))
        return super().forward(inputs)

    def forward_with_cache(
        self,
        inputs: torch.Tensor,
        cache: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply the convolution to a chunk and return the next cache."""
        if not self.causal:
            return self.forward(inputs), None
        if self.left_padding == 0:
            return super().forward(inputs), None
        if cache is None:
            cache = inputs.new_zeros(
                (inputs.shape[0], inputs.shape[1], self.left_padding)
            )
        expected = (inputs.shape[0], inputs.shape[1], self.left_padding)
        if tuple(cache.shape) != expected:
            raise ValueError(
                f"cache shape {tuple(cache.shape)} does not match {expected}"
            )
        combined = torch.cat((cache, inputs), dim=-1)
        output = super().forward(combined)
        next_cache = combined[..., -self.left_padding :].contiguous()
        return output, next_cache


class SeparableConv1d(nn.Module):
    """1D time-channel separable convolution: depthwise + pointwise."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        padding: int = 0,
        bias: bool = False,
        causal: bool = False,
    ) -> None:
        super().__init__()
        self.depthwise = CausalConv1d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding,
            groups=in_channels,
            bias=bias,
            causal=causal,
        )
        self.pointwise = nn.Conv1d(
            in_channels, out_channels, kernel_size=1, bias=bias
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(inputs))

    def forward_with_cache(
        self,
        inputs: torch.Tensor,
        cache: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        depthwise, next_cache = self.depthwise.forward_with_cache(inputs, cache)
        return self.pointwise(depthwise), next_cache


def _conv_bn(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int,
    dilation: int,
    padding: int,
    separable: bool,
    causal: bool = False,
) -> nn.Sequential:
    """NeMo's ``_get_conv_bn_layer``: conv(s) followed by BatchNorm1d."""
    if separable:
        conv: nn.Module = SeparableConv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding,
            causal=causal,
        )
    else:
        conv = CausalConv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding,
            bias=False,
            causal=causal,
        )
    return nn.Sequential(
        conv, nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.1)
    )


class JasperBlock(nn.Module):
    """One MarbleNet/QuartzNet residual block with ``repeat`` sub-blocks."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        repeat: int = 1,
        kernel_size: int = 11,
        stride: int = 1,
        dilation: int = 1,
        dropout: float = 0.0,
        separable: bool = True,
        residual: bool = False,
        kernel_size_factor: float = 1.0,
        causal: bool = False,
    ) -> None:
        super().__init__()
        kernel_size = compute_new_kernel_size(kernel_size, kernel_size_factor)
        padding = (
            causal_padding(kernel_size, dilation)
            if causal
            else same_padding(kernel_size, stride, dilation)
        )
        self.causal = bool(causal)

        layers: list[nn.Module] = []
        current = in_channels
        for _ in range(repeat - 1):
            layers.append(
                _conv_bn(
                    current,
                    out_channels,
                    kernel_size,
                    stride,
                    dilation,
                    padding,
                    separable,
                    causal=causal,
                )
            )
            layers.extend((nn.ReLU(), nn.Dropout(p=dropout)))
            current = out_channels
        layers.append(
            _conv_bn(
                current,
                out_channels,
                kernel_size,
                stride,
                dilation,
                padding,
                separable,
                causal=causal,
            )
        )
        self.mconv = nn.Sequential(*layers)

        if residual:
            # NeMo always projects with a 1x1 conv, even when channel counts
            # already match, so every residual block carries these weights.
            self.res: nn.Module | None = nn.Sequential(
                CausalConv1d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    bias=False,
                    causal=causal,
                ),
                nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.1),
            )
        else:
            self.res = None

        self.mout = nn.Sequential(nn.ReLU(), nn.Dropout(p=dropout))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        out = self.mconv(inputs)
        if self.res is not None:
            out = out + self.res(inputs)
        return self.mout(out)

    @staticmethod
    def _is_conv_bn(module: nn.Module) -> bool:
        return (
            isinstance(module, nn.Sequential)
            and len(module) == 2
            and isinstance(module[0], (nn.Conv1d, SeparableConv1d))
            and isinstance(module[1], nn.BatchNorm1d)
        )

    @staticmethod
    def _conv_bn_left_padding(module: nn.Sequential) -> int:
        conv = module[0]
        if isinstance(conv, SeparableConv1d):
            return int(conv.depthwise.left_padding)
        if isinstance(conv, CausalConv1d):
            return int(conv.left_padding)
        return 0

    @staticmethod
    def _conv_bn_in_channels(module: nn.Sequential) -> int:
        conv = module[0]
        if isinstance(conv, SeparableConv1d):
            return int(conv.depthwise.in_channels)
        return int(conv.in_channels)

    def _conv_bn_modules(self) -> list[nn.Sequential]:
        modules = [layer for layer in self.mconv if self._is_conv_bn(layer)]
        if self.res is not None and self._is_conv_bn(self.res):
            modules.append(self.res)
        return modules

    def init_stream_state(self, template: torch.Tensor) -> list[torch.Tensor | None]:
        """Create zero caches for one batch-shaped streaming call."""
        if not self.causal:
            raise RuntimeError("streaming state is only defined for causal models")
        states: list[torch.Tensor | None] = []
        for module in self._conv_bn_modules():
            left_padding = self._conv_bn_left_padding(module)
            if left_padding <= 0:
                states.append(None)
            else:
                states.append(
                    template.new_zeros(
                        (
                            template.shape[0],
                            self._conv_bn_in_channels(module),
                            left_padding,
                        )
                    )
                )
        return states

    def forward_stream(
        self,
        inputs: torch.Tensor,
        states: list[torch.Tensor | None] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor | None]]:
        """Run one causal chunk while carrying convolution caches."""
        if not self.causal:
            raise RuntimeError("forward_stream requires a causal model")
        conv_modules = self._conv_bn_modules()
        if states is None:
            states = self.init_stream_state(inputs)
        if len(states) != len(conv_modules):
            raise ValueError(
                f"expected {len(conv_modules)} cache entries, got {len(states)}"
            )

        out = inputs
        next_states: list[torch.Tensor | None] = []
        state_index = 0
        for layer in self.mconv:
            if self._is_conv_bn(layer):
                conv, norm = layer[0], layer[1]
                out, next_cache = conv.forward_with_cache(
                    out, states[state_index]
                )
                out = norm(out)
                next_states.append(next_cache)
                state_index += 1
            else:
                out = layer(out)

        if self.res is not None:
            res_out = inputs
            if self._is_conv_bn(self.res):
                conv, norm = self.res[0], self.res[1]
                res_out, next_cache = conv.forward_with_cache(
                    res_out, states[state_index]
                )
                res_out = norm(res_out)
                next_states.append(next_cache)
                state_index += 1
            else:
                res_out = self.res(res_out)
            out = out + res_out

        return self.mout(out), next_states


class MarbleNet(nn.Module):
    """MarbleNet-BxRxC: ``B`` residual blocks, ``R`` sub-blocks, ``C`` channels."""

    def __init__(
        self,
        feat_in: int = 64,
        num_classes: int = 2,
        repeat: int = 2,
        channels: int = 64,
        dropout: float = 0.0,
        kernel_size_factor: float = 1.0,
        causal: bool = False,
    ) -> None:
        super().__init__()
        self.feat_in = feat_in
        self.num_classes = num_classes
        self.causal = bool(causal)
        self.encoder = nn.Sequential(
            # Prologue: Conv1, 128 channels, kernel 11.
            JasperBlock(
                feat_in,
                128,
                repeat=1,
                kernel_size=11,
                separable=True,
                residual=False,
                dropout=dropout,
                kernel_size_factor=kernel_size_factor,
                causal=causal,
            ),
            # B1/B2/B3: 64 channels, kernels 13/15/17, residual.
            JasperBlock(
                128,
                channels,
                repeat=repeat,
                kernel_size=13,
                separable=True,
                residual=True,
                dropout=dropout,
                kernel_size_factor=kernel_size_factor,
                causal=causal,
            ),
            JasperBlock(
                channels,
                channels,
                repeat=repeat,
                kernel_size=15,
                separable=True,
                residual=True,
                dropout=dropout,
                kernel_size_factor=kernel_size_factor,
                causal=causal,
            ),
            JasperBlock(
                channels,
                channels,
                repeat=repeat,
                kernel_size=17,
                separable=True,
                residual=True,
                dropout=dropout,
                kernel_size_factor=kernel_size_factor,
                causal=causal,
            ),
            # Epilogue: Conv2 (dilated) and Conv3.
            JasperBlock(
                channels,
                128,
                repeat=1,
                kernel_size=29,
                dilation=2,
                separable=True,
                residual=False,
                dropout=dropout,
                kernel_size_factor=kernel_size_factor,
                causal=causal,
            ),
            JasperBlock(
                128,
                128,
                repeat=1,
                kernel_size=1,
                separable=False,
                residual=False,
                dropout=dropout,
                kernel_size_factor=kernel_size_factor,
                causal=causal,
            ),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(128, num_classes, bias=True)
        self.apply(init_weights)

    def forward(
        self, features: torch.Tensor, length: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Map ``[B, feat_in, T]`` MFCC features to ``[B, num_classes]`` logits.

        ``length`` is accepted for call-site compatibility with NeMo and is
        ignored: fixed-length windows mean no padding mask is required.
        """
        del length
        encoded = self.encoder(features)
        pooled = self.pool(encoded).flatten(1)
        return self.classifier(pooled)

    def init_stream_state(
        self, features: torch.Tensor
    ) -> list[list[torch.Tensor | None]]:
        """Create per-block convolution caches for a streaming call."""
        if not self.causal:
            raise RuntimeError("streaming state is only defined for causal models")
        return [block.init_stream_state(features) for block in self.encoder]

    def encode_stream(
        self,
        features: torch.Tensor,
        states: list[list[torch.Tensor | None]] | None = None,
    ) -> tuple[torch.Tensor, list[list[torch.Tensor | None]]]:
        """Encode a chunk while carrying all causal convolution caches."""
        if not self.causal:
            raise RuntimeError("encode_stream requires a causal model")
        if states is None:
            states = self.init_stream_state(features)
        if len(states) != len(self.encoder):
            raise ValueError(
                f"expected {len(self.encoder)} block states, got {len(states)}"
            )
        out = features
        next_states: list[list[torch.Tensor | None]] = []
        for block, block_state in zip(self.encoder, states):
            out, next_block_state = block.forward_stream(out, block_state)
            next_states.append(next_block_state)
        return out, next_states

    def forward_stream(
        self,
        features: torch.Tensor,
        states: list[list[torch.Tensor | None]] | None = None,
    ) -> tuple[torch.Tensor, list[list[torch.Tensor | None]]]:
        """Classify one causal feature chunk and return updated caches."""
        encoded, next_states = self.encode_stream(features, states)
        pooled = self.pool(encoded).flatten(1)
        return self.classifier(pooled), next_states

    def iter_block_shapes(
        self, features: torch.Tensor
    ) -> Iterator[tuple[str, tuple[int, ...]]]:
        """Yield the tensor shape after each encoder block (shape tracing)."""
        out = features
        for index, block in enumerate(self.encoder):
            out = block(out)
            yield f"block{index + 1}({type(block).__name__})", tuple(out.shape)
        pooled = self.pool(out)
        yield "avgpool", tuple(pooled.shape)
        yield "logits", tuple(self.classifier(pooled.flatten(1)).shape)


def init_weights(module: nn.Module, mode: str = "xavier_uniform") -> None:
    """NeMo's ``init_weights``: Xavier uniform on conv/linear, BN at identity."""
    if isinstance(module, (nn.Conv1d, nn.Linear)):
        if mode == "xavier_uniform":
            nn.init.xavier_uniform_(module.weight, gain=1.0)
        elif mode == "xavier_normal":
            nn.init.xavier_normal_(module.weight, gain=1.0)
        elif mode == "kaiming_uniform":
            nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
        elif mode == "kaiming_normal":
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        else:
            raise ValueError(f"Unknown initialization mode: {mode}")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm1d):
        if module.track_running_stats:
            module.running_mean.zero_()
            module.running_var.fill_(1)
            module.num_batches_tracked.zero_()
        if module.affine:
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)


def build_marblenet_3x2x64(
    feat_in: int = 64,
    num_classes: int = 2,
    dropout: float = 0.0,
    causal: bool = False,
) -> MarbleNet:
    """The ``MarbleNet-3x2x64`` model from the paper (B=3, R=2, C=64)."""
    return MarbleNet(
        feat_in=feat_in,
        num_classes=num_classes,
        repeat=2,
        channels=64,
        dropout=dropout,
        causal=causal,
    )


class MarbleNetStreaming:
    """Stateful rolling-window wrapper for causal MarbleNet inference.

    ``context_frames`` should match the number of frames used during training
    (64 for the original 0.63 s MFCC window).  The wrapper keeps only the most
    recent encoded frames and therefore never reads future audio.
    """

    def __init__(
        self,
        model: MarbleNet,
        *,
        context_frames: int = 64,
    ) -> None:
        if not model.causal:
            raise ValueError("MarbleNetStreaming requires a causal model")
        if context_frames <= 0:
            raise ValueError("context_frames must be positive")
        self.model = model
        self.context_frames = int(context_frames)
        self.reset()

    def reset(self) -> None:
        self._states: list[list[torch.Tensor | None]] | None = None
        self._encoded_history: torch.Tensor | None = None
        self._batch_size: int | None = None

    @torch.inference_mode()
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return logits for the newest chunk using a rolling context."""
        if features.dim() != 3:
            raise ValueError(
                f"expected [B, feat, T] features, got {tuple(features.shape)}"
            )
        if self._states is None:
            self._states = self.model.init_stream_state(features)
            self._batch_size = int(features.shape[0])
        elif int(features.shape[0]) != self._batch_size:
            raise ValueError(
                "batch size changed; call reset() before using a new batch"
            )

        encoded, self._states = self.model.encode_stream(
            features, self._states
        )
        if self._encoded_history is None:
            self._encoded_history = encoded
        else:
            self._encoded_history = torch.cat(
                (self._encoded_history, encoded), dim=-1
            )
        self._encoded_history = self._encoded_history[
            ..., -self.context_frames :
        ]
        pooled = self._encoded_history.mean(dim=-1)
        return self.model.classifier(pooled)

    @property
    def ready(self) -> bool:
        """Whether at least one full training context is available."""
        return (
            self._encoded_history is not None
            and self._encoded_history.shape[-1] >= self.context_frames
        )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="MarbleNet-3x2x64 shape trace and parameter count."
    )
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--feat-in", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    model = build_marblenet_3x2x64(feat_in=args.feat_in)
    model.eval()
    features = torch.randn(args.batch_size, args.feat_in, args.frames)
    with torch.no_grad():
        for name, shape in model.iter_block_shapes(features):
            print(f"{name:24s} {shape}")

    total = count_parameters(model)
    print(f"\ninput                {tuple(features.shape)}")
    print(f"trainable parameters {total:,} (~{total / 1000:.1f}K)")
    print("paper reference      88K for MarbleNet-3x2x64")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
