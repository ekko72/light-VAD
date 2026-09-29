# -*- coding: utf-8 -*-
"""Causal Tiny-GRU / MarbleNet and prior-factored model variants."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from reproductions.marblenet_vad.model import (
    build_marblenet_3x2x64,
)


LOGIT_EPS = 1e-6


def logit(probabilities: torch.Tensor) -> torch.Tensor:
    probabilities = probabilities.clamp(LOGIT_EPS, 1.0 - LOGIT_EPS)
    return torch.log(probabilities) - torch.log1p(-probabilities)


@dataclass(frozen=True)
class TinyGRUConfig:
    input_size: int = 64
    hidden_size: int = 64
    num_layers: int = 2
    num_classes: int = 2
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if min(
            self.input_size,
            self.hidden_size,
            self.num_layers,
            self.num_classes,
        ) <= 0:
            raise ValueError("Tiny-GRU dimensions must be positive")


class TinyGRUVAD(nn.Module):
    """Two-layer causal GRU with a frame-level binary head."""

    def __init__(self, config: TinyGRUConfig | None = None) -> None:
        super().__init__()
        self.config = config or TinyGRUConfig()
        cfg = self.config
        self.encoder = nn.GRU(
            input_size=cfg.input_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(cfg.hidden_size, cfg.num_classes)

    def forward_features(self, features: torch.Tensor) -> torch.Tensor:
        if features.dim() != 3:
            raise ValueError(
                f"expected features [B, C, T], got {tuple(features.shape)}"
            )
        hidden, _ = self.encoder(features.transpose(1, 2))
        return hidden

    def forward_from_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        logits = self.head(hidden)
        return logits.transpose(1, 2)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.forward_from_hidden(self.forward_features(features))


@dataclass(frozen=True)
class MarbleNetConfig:
    input_size: int = 64
    num_classes: int = 2
    dilation_profile: str = "short"


class MarbleNetVAD(nn.Module):
    """Frame-level Short-RF MarbleNet with the B20 model interface."""

    def __init__(self, config: MarbleNetConfig | None = None) -> None:
        super().__init__()
        self.config = config or MarbleNetConfig()
        cfg = self.config
        self.backbone = build_marblenet_3x2x64(
            feat_in=cfg.input_size,
            num_classes=cfg.num_classes,
            causal=True,
            frame_output=True,
            dilation_profile=cfg.dilation_profile,
        )

    def forward_features(self, features: torch.Tensor) -> torch.Tensor:
        if features.dim() != 3:
            raise ValueError(
                f"expected features [B, C, T], got {tuple(features.shape)}"
            )
        hidden = self.backbone.encoder(features)
        return hidden.transpose(1, 2)

    def forward_from_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.dim() != 3:
            raise ValueError(
                f"expected hidden [B, T, C], got {tuple(hidden.shape)}"
            )
        return self.backbone.classifier(hidden.transpose(1, 2))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.forward_from_hidden(self.forward_features(features))


class PriorFactoredTinyGRU(TinyGRUVAD):
    """Tiny-GRU whose classifier is an acoustic evidence score.

    The environment prior is supplied only through an additive logit
    intercept. At deployment it is replaced with the target prior.
    """

    def forward_from_evidence(self, evidence: torch.Tensor) -> torch.Tensor:
        if evidence.dim() != 2:
            raise ValueError(
                f"expected evidence [B, T], got {tuple(evidence.shape)}"
            )
        return evidence

    def forward_environment(
        self,
        hidden: torch.Tensor,
        *,
        prior: float,
    ) -> torch.Tensor:
        if not 0.0 < float(prior) < 1.0:
            raise ValueError("prior must lie strictly inside (0, 1)")
        binary_logits = self.forward_from_hidden(hidden)
        evidence = binary_logits[:, 1] - binary_logits[:, 0]
        intercept = logit(
            torch.tensor(float(prior), device=evidence.device)
        )
        return evidence + intercept

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        raise ValueError(
            "PriorFactoredTinyGRU requires an explicit target prior; "
            "call forward_environment"
        )


class PriorFactoredMarbleNet(MarbleNetVAD):
    """MarbleNet whose classifier is an acoustic evidence score."""

    def forward_from_evidence(self, evidence: torch.Tensor) -> torch.Tensor:
        if evidence.dim() != 2:
            raise ValueError(
                f"expected evidence [B, T], got {tuple(evidence.shape)}"
            )
        return evidence

    def forward_environment(
        self,
        hidden: torch.Tensor,
        *,
        prior: float,
    ) -> torch.Tensor:
        if not 0.0 < float(prior) < 1.0:
            raise ValueError("prior must lie strictly inside (0, 1)")
        binary_logits = self.forward_from_hidden(hidden)
        evidence = binary_logits[:, 1] - binary_logits[:, 0]
        intercept = logit(
            torch.tensor(float(prior), device=evidence.device)
        )
        return evidence + intercept

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        raise ValueError(
            "PriorFactoredMarbleNet requires an explicit target prior; "
            "call forward_environment"
        )
