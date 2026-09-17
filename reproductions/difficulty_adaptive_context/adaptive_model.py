# -*- coding: utf-8 -*-
"""Shared-encoder model with optional sparse causal temporal refinement.

Phase A3 keeps the deployed Short-RF MarbleNet encoder unchanged and adds a
stateless refinement path that reads cached encoder features.  The refinement
uses several causal depthwise convolution branches with different dilation
rates.  At inference it evaluates only the frames selected by the confidence
gate, so skipped frames do not create a missing recurrent-state problem.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import NamedTuple, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from reproductions.marblenet_vad.model import MarbleNet


@dataclass(frozen=True)
class RefinementConfig:
    """Configuration of the sparse causal refinement path."""

    in_channels: int = 128
    num_classes: int = 2
    kernel_size: int = 5
    dilations: tuple[int, ...] = (1, 2, 4, 8, 16)
    max_residual: float = 2.0

    def __post_init__(self) -> None:
        if self.in_channels <= 0:
            raise ValueError("in_channels must be positive")
        if self.num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if self.kernel_size <= 0:
            raise ValueError("kernel_size must be positive")
        if not self.dilations or any(value <= 0 for value in self.dilations):
            raise ValueError("dilations must be a non-empty sequence of positives")
        if self.max_residual <= 0:
            raise ValueError("max_residual must be positive")

    @property
    def lookback_frames(self) -> int:
        return max(
            (self.kernel_size - 1) * int(dilation)
            for dilation in self.dilations
        )

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["dilations"] = list(self.dilations)
        return result

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "RefinementConfig":
        values = dict(payload)
        if "dilations" in values:
            values["dilations"] = tuple(
                int(value) for value in values["dilations"]  # type: ignore[arg-type]
            )
        return cls(**values)  # type: ignore[arg-type]


class AdaptiveOutput(NamedTuple):
    """Intermediate tensors returned by the shared adaptive model."""

    logits: torch.Tensor
    short_logits: torch.Tensor
    encoded: torch.Tensor
    selected: torch.Tensor
    residual: torch.Tensor


class AdaptiveStreamingOutput(NamedTuple):
    """Intermediate tensors returned by one streaming adaptive chunk."""

    logits: torch.Tensor
    short_logits: torch.Tensor
    selected: torch.Tensor
    residual: torch.Tensor


def speech_probabilities(logits: torch.Tensor) -> torch.Tensor:
    """Return P(speech) for binary frame logits shaped ``[B, 2, T]``."""
    if logits.dim() != 3 or logits.shape[1] != 2:
        raise ValueError(
            f"expected binary logits [B, 2, T], got {tuple(logits.shape)}"
        )
    return torch.softmax(logits, dim=1)[:, 1]


def confidence_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Return ``abs(P(speech) - 0.5)`` for each frame."""
    return (speech_probabilities(logits) - 0.5).abs()


def selection_from_confidence(
    confidence: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """Select frames whose confidence is at or below a fixed threshold."""
    if confidence.dim() != 2:
        raise ValueError(
            f"expected confidence [B, T], got {tuple(confidence.shape)}"
        )
    if not 0.0 <= float(threshold) <= 0.5:
        raise ValueError("threshold must be in [0, 0.5]")
    return confidence <= float(threshold)


class SparseCausalDepthwiseBranch(nn.Module):
    """One causal depthwise convolution with dense and sparse evaluation."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
    ) -> None:
        super().__init__()
        if channels <= 0 or kernel_size <= 0 or dilation <= 0:
            raise ValueError("channels, kernel_size and dilation must be positive")
        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation)
        self.left_padding = (self.kernel_size - 1) * self.dilation
        self.weight = nn.Parameter(
            torch.empty(self.channels, 1, self.kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(self.channels))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        nn.init.zeros_(self.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.shape[1] != self.channels:
            raise ValueError(
                f"expected {self.channels} channels, got {inputs.shape[1]}"
            )
        padded = F.pad(inputs, (self.left_padding, 0))
        return F.conv1d(
            padded,
            self.weight,
            self.bias,
            groups=self.channels,
            dilation=self.dilation,
        )

    def forward_sparse(
        self,
        inputs: torch.Tensor,
        *,
        sample_index: torch.Tensor,
        time_index: torch.Tensor,
        max_left_padding: int,
    ) -> torch.Tensor:
        """Evaluate only selected ``(batch, time)`` positions."""
        if sample_index.numel() == 0:
            return inputs.new_empty((0, self.channels))
        if sample_index.shape != time_index.shape:
            raise ValueError("sample_index and time_index must have equal shape")
        padded = F.pad(inputs, (int(max_left_padding), 0))
        selected = padded.index_select(0, sample_index)
        past = torch.arange(
            self.kernel_size - 1,
            -1,
            -1,
            device=inputs.device,
        ) * self.dilation
        indices = int(max_left_padding) + time_index[:, None] - past[None, :]
        windows = selected.gather(
            2,
            indices[:, None, :].expand(-1, self.channels, -1),
        )
        values = (windows * self.weight[None, :, 0, :]).sum(dim=-1)
        return values + self.bias[None, :]


class SparseCausalMultiScaleRefinement(nn.Module):
    """Stateless multi-scale causal refinement of cached Short features."""

    def __init__(self, config: RefinementConfig | None = None) -> None:
        super().__init__()
        self.config = config or RefinementConfig()
        self.branches = nn.ModuleList(
            SparseCausalDepthwiseBranch(
                self.config.in_channels,
                self.config.kernel_size,
                dilation,
            )
            for dilation in self.config.dilations
        )
        self.activation = nn.GELU()
        self.output = nn.Conv1d(
            self.config.in_channels,
            self.config.num_classes,
            kernel_size=1,
            bias=True,
        )
        # The initial adaptive model is exactly the Short model.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    @property
    def lookback_frames(self) -> int:
        return self.config.lookback_frames

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        """Dense reference path, useful for equivalence checks and benchmarks."""
        branch_sum = self.branches[0](encoded)
        for branch in self.branches[1:]:
            branch_sum = branch_sum + branch(encoded)
        raw = self.output(self.activation(branch_sum))
        return self.config.max_residual * torch.tanh(raw)

    def forward_sparse(
        self,
        encoded: torch.Tensor,
        selected: torch.Tensor,
    ) -> torch.Tensor:
        """Refine only selected frames and return a dense residual tensor.

        ``selected`` is a boolean ``[B, T]`` mask aligned with ``encoded``.
        The function reads only current and previous encoded frames.  It has no
        recurrent state, so a frame skipped by the gate cannot corrupt later
        refinement calls.
        """
        if encoded.dim() != 3:
            raise ValueError(
                f"expected encoded features [B, C, T], got {tuple(encoded.shape)}"
            )
        if encoded.shape[1] != self.config.in_channels:
            raise ValueError(
                f"expected {self.config.in_channels} channels, "
                f"got {encoded.shape[1]}"
            )
        expected_selected_shape = (encoded.shape[0], encoded.shape[-1])
        if tuple(selected.shape) != expected_selected_shape:
            raise ValueError(
                f"selected mask {tuple(selected.shape)} does not match "
                f"expected shape {expected_selected_shape}"
            )
        selected = selected.to(dtype=torch.bool)
        sample_index, time_index = torch.nonzero(selected, as_tuple=True)
        residual = encoded.new_zeros(
            (encoded.shape[0], self.config.num_classes, encoded.shape[-1])
        )
        if sample_index.numel() == 0:
            return residual

        branch_sum = self.branches[0].forward_sparse(
            encoded,
            sample_index=sample_index,
            time_index=time_index,
            max_left_padding=self.lookback_frames,
        )
        for branch in self.branches[1:]:
            branch_sum = branch_sum + branch.forward_sparse(
                encoded,
                sample_index=sample_index,
                time_index=time_index,
                max_left_padding=self.lookback_frames,
            )
        activated = self.activation(branch_sum)
        raw = F.linear(
            activated,
            self.output.weight[:, :, 0],
            self.output.bias,
        )
        bounded = self.config.max_residual * torch.tanh(raw)
        residual[sample_index, :, time_index] = bounded
        return residual

    def estimated_macs_per_selected_frame(self) -> int:
        """Depthwise and output projection MACs for one selected frame."""
        depthwise = sum(
            self.config.in_channels * self.config.kernel_size
            for _ in self.config.dilations
        )
        output = self.config.in_channels * self.config.num_classes
        return int(depthwise + output)


class AdaptiveCausalVAD(nn.Module):
    """Shared Short encoder plus an optional stateless refinement path."""

    def __init__(
        self,
        short_model: MarbleNet,
        refinement: SparseCausalMultiScaleRefinement | None = None,
    ) -> None:
        super().__init__()
        if not short_model.causal:
            raise ValueError("AdaptiveCausalVAD requires a causal Short model")
        if not short_model.frame_output:
            raise ValueError("AdaptiveCausalVAD requires frame-level outputs")
        self.short_model = short_model
        self.refinement = refinement or SparseCausalMultiScaleRefinement()

    def freeze_short_model(self) -> None:
        self.short_model.requires_grad_(False)
        self.short_model.eval()

    def forward_components(
        self,
        features: torch.Tensor,
        *,
        activation_threshold: float | None = None,
        activation_mask: torch.Tensor | None = None,
    ) -> AdaptiveOutput:
        """Run the shared encoder and optional refinement."""
        encoded = self.short_model.encoder(features)
        short_logits = self.short_model.classifier(encoded)
        if activation_mask is None:
            if activation_threshold is None:
                selected = torch.zeros(
                    short_logits.shape[:2],
                    dtype=torch.bool,
                    device=short_logits.device,
                )
            else:
                selected = selection_from_confidence(
                    confidence_from_logits(short_logits),
                    activation_threshold,
                )
        else:
            selected = activation_mask.to(
                device=short_logits.device,
                dtype=torch.bool,
            )
            if selected.shape != short_logits.shape[:2]:
                raise ValueError(
                    "activation_mask must have shape [B, T] matching logits"
                )
        residual = self.refinement.forward_sparse(encoded, selected)
        return AdaptiveOutput(
            logits=short_logits + residual,
            short_logits=short_logits,
            encoded=encoded,
            selected=selected,
            residual=residual,
        )

    def forward(
        self,
        features: torch.Tensor,
        *,
        activation_threshold: float | None = None,
        activation_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.forward_components(
            features,
            activation_threshold=activation_threshold,
            activation_mask=activation_mask,
        ).logits


class AdaptiveStreamingVAD:
    """Streaming wrapper preserving encoder caches and refinement history."""

    def __init__(
        self,
        model: AdaptiveCausalVAD,
        *,
        activation_threshold: float,
    ) -> None:
        self.model = model
        self.activation_threshold = float(activation_threshold)
        self.reset()

    def reset(self) -> None:
        self._encoder_states: list[list[torch.Tensor | None]] | None = None
        self._encoded_history: torch.Tensor | None = None
        self._batch_size: int | None = None

    @property
    def cache_bytes(self) -> int:
        total = 0
        if self._encoded_history is not None:
            total += self._encoded_history.numel() * self._encoded_history.element_size()
        if self._encoder_states is not None:
            for block in self._encoder_states:
                for cache in block:
                    if cache is not None:
                        total += cache.numel() * cache.element_size()
        return int(total)

    @torch.inference_mode()
    def forward_components(
        self,
        features: torch.Tensor,
        *,
        activation_threshold: float | None = None,
    ) -> AdaptiveStreamingOutput:
        """Consume a causal feature chunk and return its intermediate outputs."""
        if features.dim() != 3:
            raise ValueError(
                f"expected [B, feat, T] features, got {tuple(features.shape)}"
            )
        if self._encoder_states is None:
            self._encoder_states = self.model.short_model.init_stream_state(
                features
            )
            self._batch_size = int(features.shape[0])
        elif int(features.shape[0]) != self._batch_size:
            raise ValueError(
                "batch size changed; call reset() before using a new batch"
            )

        encoded, self._encoder_states = self.model.short_model.encode_stream(
            features,
            self._encoder_states,
        )
        short_logits = self.model.short_model.classifier(encoded)
        threshold = (
            self.activation_threshold
            if activation_threshold is None
            else float(activation_threshold)
        )
        selected_current = selection_from_confidence(
            confidence_from_logits(short_logits),
            threshold,
        )

        if self._encoded_history is None:
            combined = encoded
            selected = selected_current
        else:
            combined = torch.cat((self._encoded_history, encoded), dim=-1)
            selected = torch.cat(
                (
                    torch.zeros(
                        (
                            self._encoded_history.shape[0],
                            self._encoded_history.shape[-1],
                        ),
                        dtype=torch.bool,
                        device=encoded.device,
                    ),
                    selected_current,
                ),
                dim=-1,
            )
        residual = self.model.refinement.forward_sparse(combined, selected)
        current_length = int(encoded.shape[-1])
        residual_current = residual[..., -current_length:]
        self._encoded_history = combined[
            ..., -self.model.refinement.lookback_frames :
        ]
        return AdaptiveStreamingOutput(
            logits=short_logits + residual_current,
            short_logits=short_logits,
            selected=selected_current,
            residual=residual_current,
        )

    def forward(
        self,
        features: torch.Tensor,
        *,
        activation_threshold: float | None = None,
    ) -> torch.Tensor:
        """Consume a causal feature chunk and return its refined logits."""
        return self.forward_components(
            features,
            activation_threshold=activation_threshold,
        ).logits
