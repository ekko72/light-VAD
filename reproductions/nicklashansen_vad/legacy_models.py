from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import sys
import weakref

import torch
import torch.nn as nn
import torch.nn.functional as F


BATCH_SIZE = 2048
FRAMES = 30
FEATURES = 24


class Net(nn.Module):
    """Original fully connected RNN baseline."""

    def __init__(self, large: bool = True, lstm: bool = True):
        super().__init__()
        self.large = large
        self.lstm = lstm
        self.relu = nn.ReLU()

        if lstm:
            self.hidden = self.init_hidden()
            self.rnn = nn.LSTM(
                input_size=FEATURES,
                hidden_size=FRAMES,
                num_layers=1,
                batch_first=True,
            )
        else:
            self.rnn = nn.GRU(
                input_size=FEATURES,
                hidden_size=FRAMES,
                num_layers=1,
                batch_first=True,
            )

        if large:
            self.lin1 = nn.Linear(FRAMES**2, 26)
            self.lin2 = nn.Linear(26, 2)
        else:
            self.lin = nn.Linear(FRAMES**2, 2)

        self.softmax = nn.Softmax(dim=1)

    def init_hidden(self):
        h = torch.zeros(1, BATCH_SIZE, FRAMES)
        c = torch.zeros(1, BATCH_SIZE, FRAMES)
        return h, c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "lstm") and self.lstm:
            x, _ = self.rnn(x, self.hidden)
        else:
            x, _ = self.rnn(x)

        x = x.contiguous().view(-1, FRAMES**2)
        if self.large:
            x = self.lin2(self.relu(self.lin1(x)))
        else:
            x = self.lin(x)
        return self.softmax(x)


class BiRNN(nn.Module):
    def __init__(
        self,
        num_in: int,
        num_hidden: int,
        batch_size: int = BATCH_SIZE,
        large: bool = True,
        lstm: bool = False,
        fcl: bool = True,
        bidir: bool = False,
    ):
        super().__init__()
        self.num_hidden = num_hidden
        self.batch_size = batch_size
        self.lstm = lstm
        self.bidir = bidir
        self.layers = 2 if large else 1

        if lstm:
            self.hidden = self.init_hidden()
            self.rnn = nn.LSTM(
                num_in,
                num_hidden,
                num_layers=self.layers,
                bidirectional=self.bidir,
                batch_first=True,
            )
            size = 18 if large else 16
        else:
            self.rnn = nn.GRU(
                num_in,
                num_hidden,
                num_layers=self.layers,
                bidirectional=self.bidir,
                batch_first=True,
            )
            size = 18

        embed_size = num_hidden * 2 if self.bidir or self.layers > 1 else num_hidden
        if not fcl:
            self.embed = nn.Linear(embed_size, 2)
        elif large:
            self.embed = nn.Sequential(
                nn.Linear(embed_size, size + 14),
                nn.BatchNorm1d(size + 14),
                nn.Dropout(p=0.2),
                nn.ReLU(),
                nn.Linear(size + 14, size),
                nn.BatchNorm1d(size),
                nn.Dropout(p=0.2),
                nn.ReLU(),
                nn.Linear(size, 2),
            )
        else:
            self.embed = nn.Sequential(
                nn.Linear(embed_size, size),
                nn.BatchNorm1d(size),
                nn.Dropout(p=0.2),
                nn.ReLU(),
                nn.Linear(size, 2),
            )

    def init_hidden(self):
        num_dir = 2 if self.bidir or self.layers > 1 else 1
        h = torch.zeros(num_dir, self.batch_size, self.num_hidden)
        c = torch.zeros(num_dir, self.batch_size, self.num_hidden)
        return h, c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        if self.lstm:
            x, self.hidden = self.rnn(x, self.hidden)
        else:
            x, self.hidden = self.rnn(x)
        x = self.hidden.view(self.batch_size, -1)
        return self.embed(x)


class GatedConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        padding: bool = True,
    ):
        super().__init__()
        padding_size = int((kernel_size - 1) / 2) if padding else 0
        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=padding_size,
            ),
            nn.BatchNorm1d(out_channels),
            nn.Tanh(),
        )
        self.conv_gate = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=padding_size,
            ),
            nn.BatchNorm1d(out_channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) * self.conv_gate(x)


class Conv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        padding: bool = True,
    ):
        super().__init__()
        padding_size = int((kernel_size - 1) / 2) if padding else 0
        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=padding_size,
            ),
            nn.BatchNorm1d(out_channels),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class GatedResidualConv(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        self.gated_conv = GatedConv(channels, channels)

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None):
        if residual is None:
            residual = x
        out = self.gated_conv(x)
        return out * x, out * residual


class NickNet(nn.Module):
    def __init__(
        self,
        large: bool = True,
        residual_connections: bool = False,
        gated: bool = True,
        lstm: bool = False,
        fcl: bool = True,
        bidir: bool = False,
        frames: int = FRAMES,
        features: int = FEATURES,
    ):
        super().__init__()
        self.large = large
        self.residual_connections = residual_connections

        if large:
            if gated:
                channels = (32, 28, 25, 18)
            else:
                channels = (38, 35, 31, 24)
            channels_out = channels[3]
        else:
            if gated:
                channels = (20, 18, 16)
            else:
                channels = (26, 20, 16)
            channels_out = channels[2]

        if residual_connections:
            channels_3 = channels[1]
            self.conv1 = GatedConv(features, channels_3)
            self.conv2 = GatedResidualConv(channels_3)
            self.conv3 = GatedResidualConv(channels_3)
            if large:
                self.conv4 = GatedResidualConv(channels_3)
        elif gated:
            self.conv1 = GatedConv(features, channels[0])
            self.conv2 = GatedConv(channels[0], channels[1])
            self.conv3 = GatedConv(channels[1], channels[2])
            if large:
                self.conv4 = GatedConv(channels[2], channels[3])
        else:
            self.conv1 = Conv(features, channels[0])
            self.conv2 = Conv(channels[0], channels[1])
            self.conv3 = Conv(channels[1], channels[2])
            if large:
                self.conv4 = Conv(channels[2], channels[3])

        num_hidden = channels_out + (11 if large else 5)
        self.rnn = BiRNN(
            channels_out,
            num_hidden,
            large=large,
            lstm=lstm,
            fcl=fcl,
            bidir=bidir,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.conv1(x)

        if self.residual_connections:
            x, residual = self.conv2(x)
            x, residual = self.conv3(x, residual)
            if self.large:
                x, residual = self.conv4(x, residual)
            x = x * residual
        else:
            x = self.conv2(x)
            x = self.conv3(x)
            if self.large:
                x = self.conv4(x)

        x = self.rnn(x)
        return F.softmax(x, dim=1)


class DenseSingle(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        dropout: float,
        dilation: int,
        padding: int,
        kernel_size: int,
        stride: int,
    ):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(
                input_size,
                output_size,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm1d(output_size),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x, self.layer(x)], 1)


class DenseBlock(nn.Module):
    def __init__(
        self,
        input_size: int,
        n_layers: int,
        growth_rate: int,
        dropout: float,
        dilation: int,
        padding: int,
        kernel_size: int,
        stride: int,
    ):
        super().__init__()
        layers = []
        for index in range(n_layers):
            layers.append(
                DenseSingle(
                    input_size + index * growth_rate,
                    growth_rate,
                    dropout,
                    dilation,
                    padding,
                    kernel_size,
                    stride,
                )
            )
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TransitionBlock(nn.Module):
    def __init__(self, input_size: int, output_size: int, dropout: float):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(input_size, output_size, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(output_size),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class DenseNet(nn.Module):
    def __init__(self, large: bool = False):
        super().__init__()
        dropout = 0.4

        if large:
            self.cnn_in = nn.Sequential(
                nn.Conv1d(
                    in_channels=24,
                    out_channels=48,
                    kernel_size=6,
                    stride=1,
                    padding=0,
                    dilation=4,
                    bias=False,
                ),
                nn.BatchNorm1d(48),
                nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
                nn.LeakyReLU(),
                nn.Dropout(p=dropout),
            )
            self.dense1 = DenseBlock(48, 8, 4, dropout, 1, 1, 3, 1)
            self.trans1 = TransitionBlock(80, 48, dropout)
            self.dense2 = DenseBlock(48, 8, 4, dropout, 1, 1, 3, 1)
            self.cnn_out = nn.Sequential(
                nn.Conv1d(80, 80, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm1d(80),
                nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
                nn.LeakyReLU(),
                nn.Dropout(p=dropout),
            )
            self.out = nn.Linear(80, 2, bias=False)
        else:
            self.cnn_in = nn.Sequential(
                nn.Conv1d(
                    in_channels=24,
                    out_channels=24,
                    kernel_size=6,
                    stride=1,
                    padding=0,
                    dilation=4,
                    bias=False,
                ),
                nn.BatchNorm1d(24),
                nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
                nn.LeakyReLU(),
                nn.Dropout(p=dropout),
            )
            self.dense1 = DenseBlock(24, 6, 3, dropout, 1, 1, 3, 1)
            self.trans1 = TransitionBlock(42, 24, dropout)
            self.dense2 = DenseBlock(24, 6, 3, dropout, 1, 1, 3, 1)
            self.cnn_out = nn.Sequential(
                nn.Conv1d(42, 42, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm1d(42),
                nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
                nn.LeakyReLU(),
                nn.Dropout(p=dropout),
            )
            self.out = nn.Linear(42, 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.cnn_in(x)
        x = self.dense1(x)
        x = self.trans1(x)
        x = self.dense2(x)
        x = self.cnn_out(x)
        x = x.view(x.shape[0], -1)
        return F.softmax(self.out(x), dim=1)


_LEGACY_CLASSES = {
    "Net": Net,
    "BiRNN": BiRNN,
    "GatedConv": GatedConv,
    "Conv": Conv,
    "GatedResidualConv": GatedResidualConv,
    "NickNet": NickNet,
    "DenseSingle": DenseSingle,
    "DenseBlock": DenseBlock,
    "TransitionBlock": TransitionBlock,
    "DenseNet": DenseNet,
}


def _rebuild_flat_weights(module: nn.RNNBase) -> None:
    """Restore fields expected by modern PyTorch in pre-1.8 RNN pickles."""

    if not hasattr(module, "_all_weights") or not module._all_weights:
        num_layers = module.num_layers
        num_directions = 2 if module.bidirectional else 1
        module._all_weights = []
        for layer in range(num_layers):
            for direction in range(num_directions):
                suffix = "_reverse" if direction == 1 else ""
                names = [
                    "weight_ih_l{}{}",
                    "weight_hh_l{}{}",
                    "bias_ih_l{}{}",
                    "bias_hh_l{}{}",
                    "weight_hr_l{}{}",
                ]
                names = [name.format(layer, suffix) for name in names]
                if module.bias:
                    module._all_weights.append(
                        names if module.proj_size > 0 else names[:4]
                    )
                else:
                    module._all_weights.append(
                        names[:2] + names[-1:] if module.proj_size > 0 else names[:2]
                    )

    if module._all_weights and isinstance(module._all_weights[0][0], str):
        module._flat_weights_names = [
            name for weights in module._all_weights for name in weights
        ]
    else:
        raise RuntimeError(
            f"Unsupported legacy {type(module).__name__} checkpoint layout"
        )

    module._flat_weights = [
        getattr(module, name) if hasattr(module, name) else None
        for name in module._flat_weights_names
    ]
    module._flat_weight_refs = [
        weakref.ref(weight) if weight is not None else None
        for weight in module._flat_weights
    ]


@contextmanager
def _rnn_pickle_compat():
    """Bridge RNN internals from the checkpoint's PyTorch to torch 2.x."""

    original_setstate = nn.RNNBase.__setstate__

    def compatible_setstate(module: nn.RNNBase, state: dict) -> None:
        try:
            original_setstate(module, state)
        except AttributeError as exc:
            if "_flat_weights" not in str(exc):
                raise
            nn.Module.__setstate__(module, state)
            if "all_weights" in state:
                module._all_weights = state["all_weights"]
            if "proj_size" not in state:
                module.proj_size = 0
            _rebuild_flat_weights(module)

    nn.RNNBase.__setstate__ = compatible_setstate
    try:
        yield
    finally:
        nn.RNNBase.__setstate__ = original_setstate


def load_legacy_checkpoint(path: str | Path, map_location: str | torch.device = "cpu"):
    """Load an original full-module ``.net`` checkpoint securely enough for this repo.

    The upstream files are full PyTorch pickles whose classes were defined in
    ``__main__``. We expose classes with matching names there before loading.
    ``weights_only=False`` is required because these are object, not state-dict,
    checkpoints. Only use this loader with the pinned upstream revision.
    """

    main_module = sys.modules["__main__"]
    injected = []
    for name, cls in _LEGACY_CLASSES.items():
        if not hasattr(main_module, name):
            setattr(main_module, name, cls)
            injected.append(name)

    try:
        with _rnn_pickle_compat():
            model = torch.load(path, map_location=map_location, weights_only=False)
    finally:
        for name in injected:
            delattr(main_module, name)

    model.eval()
    return model


def num_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
