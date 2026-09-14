# -*- coding: utf-8 -*-
"""G0-03 最小 CNN 分类例子（Fashion-MNIST）。

重点不是刷分，而是看清 tensor shape 在每一层如何变化：
脚本会用 forward hook 记录真实的前向形状，打印并写入 shape_table.md。

维度公式（无 padding、stride=1）：
    Conv2d 输出边长 = (输入边长 - kernel_size) / stride + 1
    MaxPool2d 输出边长 = (输入边长 - kernel_size) / stride + 1

用法：
    python testother/g0-03_cnn.py
    python testother/g0-03_cnn.py --epochs 2 --train-subset 4000   # 更快
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "fashion-mnist"
DEFAULT_OUT_DIR = PROJECT_ROOT / "results" / "g0-03_cnn"


class ToyCNN(nn.Module):
    """1x28x28 -> conv -> pool -> conv -> pool -> fc -> 10"""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 5 * 5, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))   # (B,1,28,28) -> (B,8,13,13)
        x = self.pool(self.relu(self.conv2(x)))   # (B,8,13,13) -> (B,16,5,5)
        x = self.flatten(x)                       # (B,16,5,5)  -> (B,400)
        x = self.relu(self.fc1(x))                # (B,400)     -> (B,64)
        return self.fc2(x)                        # (B,64)      -> (B,10)


def trace_shapes(net: nn.Module, sample: torch.Tensor) -> list[tuple[str, tuple]]:
    """按 forward 的实际顺序逐步记录张量 shape。

    不用 forward hook 是因为 relu / pool 是共享模块，会被调用多次，
    hook 记录的顺序和归属都不直观。
    """
    rows: list[tuple[str, tuple]] = []

    def record(name, tensor):
        rows.append((name, tuple(tensor.shape)))
        return tensor

    net.eval()
    with torch.no_grad():
        z = record("input", sample)
        z = record("conv1", net.conv1(z))
        z = record("relu1", net.relu(z))
        z = record("pool1", net.pool(z))
        z = record("conv2", net.conv2(z))
        z = record("relu2", net.relu(z))
        z = record("pool2", net.pool(z))
        z = record("flatten", net.flatten(z))
        z = record("fc1", net.fc1(z))
        z = record("relu3", net.relu(z))
        z = record("fc2", net.fc2(z))
    return rows


def format_trace(rows, batch: int) -> str:
    lines = [f"输入张量 (batch, channels, height, width) = ({batch}, 1, 28, 28)", ""]
    lines.append(f"{'步骤':<10} {'输出 shape':<20} {'怎么来的':<34}")
    lines.append("-" * 68)
    prev = None
    for name, shape in rows:
        if prev is None:
            how = "数据本身"
        elif name.startswith("conv"):
            how = "3x3 卷积: 边长 -2"
        elif name.startswith("relu"):
            how = "逐元素激活: shape 不变"
        elif name.startswith("pool"):
            how = "2x2 池化: 边长减半"
        elif name == "flatten":
            how = "摊平: C*H*W"
        elif name == "fc1":
            how = "全连接: 最后一维 -> 64"
        elif name == "fc2":
            how = "全连接: -> 10 类"
        else:
            how = ""
        lines.append(f"{name:<10} {str(shape):<20} {how:<34}")
        prev = shape
    lines.append("")
    lines.append("形状变化含义：")
    lines.append("  B    : batch size，整批样本数，全程不变")
    lines.append("  C    : 通道数（特征图个数），conv1 把 1 提到 8，conv2 把 8 提到 16")
    lines.append("  H, W : 空间尺寸，每次 3x3 卷积减 2，每次 2x2 池化减半")
    lines.append("  400  : flatten 后 16*5*5，把特征图摊平成一维向量")
    lines.append("  10   : Fashion-MNIST 的类别数")
    return "\n".join(lines)


def build_loaders(data_dir: Path, batch_size: int, train_subset: int, test_subset: int):
    transform = transforms.ToTensor()
    train_set = datasets.FashionMNIST(root=str(data_dir), train=True, download=True, transform=transform)
    test_set = datasets.FashionMNIST(root=str(data_dir), train=False, download=True, transform=transform)
    if train_subset:
        train_set = Subset(train_set, range(min(train_subset, len(train_set))))
    if test_subset:
        test_set = Subset(test_set, range(min(test_subset, len(test_set))))
    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True),
        DataLoader(test_set, batch_size=batch_size, shuffle=False),
    )


def evaluate(net, data_iter, device) -> float:
    net.eval()
    correct = total = 0
    with torch.no_grad():
        for features, labels in data_iter:
            features, labels = features.to(device), labels.to(device)
            correct += int((net(features).argmax(dim=1) == labels).sum())
            total += labels.numel()
    return correct / total


def main() -> None:
    parser = argparse.ArgumentParser(description="G0-03 最小 CNN 分类例子")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--train-subset", type=int, default=20000, help="0 表示用完整训练集")
    parser.add_argument("--test-subset", type=int, default=5000, help="0 表示用完整测试集")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    net = ToyCNN().to(device)

    print("\n=== tensor shape 追踪 ===")
    sample = torch.randn(args.batch_size, 1, 28, 28, device=device)
    trace = trace_shapes(net, sample)
    report = format_trace(trace, args.batch_size)
    print(report)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    shape_md = args.out_dir / "shape_table.md"
    shape_md.write_text("# CNN 逐层 shape\n\n```text\n" + report + "\n```\n", encoding="utf-8")
    print(f"\nshape 表 -> {shape_md}")

    train_iter, test_iter = build_loaders(args.data_dir, args.batch_size, args.train_subset, args.test_subset)
    print(f"train batches: {len(train_iter)}, test batches: {len(test_iter)}")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr)
    params = sum(p.numel() for p in net.parameters())
    print(f"参数量: {params:,}")

    losses, accs = [], []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        net.train()
        total_loss = correct = total = 0
        for features, labels in train_iter:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(features)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * labels.numel()
            correct += int((outputs.argmax(dim=1) == labels).sum())
            total += labels.numel()
        test_acc = evaluate(net, test_iter, device)
        losses.append(total_loss / total)
        accs.append(test_acc)
        print(f"epoch {epoch}/{args.epochs}  train loss {losses[-1]:.4f}  "
              f"train acc {correct/total:.3f}  test acc {test_acc:.3f}")

    print(f"done in {time.time()-started:.1f}s, final test acc {accs[-1]:.3f}")

    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(range(1, len(losses) + 1), losses, "o-", color="#c0392b", label="train loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("train loss", color="#c0392b")
    ax1.grid(alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(range(1, len(accs) + 1), accs, "s--", color="#27ae60", label="test acc")
    ax2.set_ylabel("test accuracy")
    ax2.set_ylim(0, 1)
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="center right", fontsize=8)
    fig.suptitle("G0-03 Toy CNN (Fashion-MNIST)", fontsize=11)
    fig.tight_layout()
    curve = args.out_dir / "cnn_result.png"
    fig.savefig(curve, dpi=140)
    plt.close(fig)
    print(f"CNN toy result -> {curve}")


if __name__ == "__main__":
    main()