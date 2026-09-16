# -*- coding: utf-8 -*-
"""G0-01 最小 MLP 训练例子：Fashion-MNIST 十分类。

手写训练循环，不依赖 d2l.train_ch3：
- 该方法在 d2l 1.x 已被移除，只在 0.17 存在，跨环境不可移植；
- 它内部用 Animator + IPython.display 出图，在纯脚本里不会产生任何输出文件。

本脚本在 .venv（CUDA）和 d2l-env（CPU）下都能运行，并把训练曲线落盘成 PNG。

用法：
    python testother/test-train.py                  # 默认 10 epoch
    python testother/test-train.py --epochs 3
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg") #无gui，脚步绘图
import matplotlib.pyplot as plt
import torch
from torch import nn#神经网络的函数
from torch.utils.data import DataLoader#批次封装数据的函数
from torchvision import datasets, transforms#数据集，图像转换

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "fashion-mnist"
DEFAULT_OUT_DIR = PROJECT_ROOT / "results" / "g0-01_mlp"


def get_dataloaders(data_dir: Path, batch_size: int):
    """Fashion-MNIST 训练/测试集；首次运行会自动下载到项目内。"""
    transform = transforms.ToTensor() #转换图像形式，对像素值归一化
    train_set = datasets.FashionMNIST(
        root=str(data_dir), train=True, download=True, transform=transform
    )
    test_set = datasets.FashionMNIST(
        root=str(data_dir), train=False, download=True, transform=transform
    ) #分别设置数据集和测试集
    train_iter = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_iter = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_iter, test_iter #对测试集和训练集的封装，shuffle=true是训练集


def build_net(hidden: int = 256, std: float = 0.01) -> nn.Sequential:
    """784 -> hidden -> ReLU -> 10，权重按正态分布初始化。"""
    net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, hidden),
        nn.ReLU(),
        nn.Linear(hidden, 10),
    )#网络的设置

    def init_weights(module):#初始化线性网络层，正态分布，偏置为0
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=std)
            nn.init.zeros_(module.bias)

    net.apply(init_weights)#apply，对net里的所有层遍历，初始化
    return net


def evaluate_accuracy(net, data_iter, device) -> float:
    net.eval()#评估模式，测正确率
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in data_iter:
            features, labels = features.to(device), labels.to(device)
            predictions = net(features).argmax(dim=1)
            correct += int((predictions == labels).sum())#判断相等数量-sum求和-int化为整形加到correct
            total += labels.numel()#标签数量
    return correct / total


def train_one_epoch(net, train_iter, loss_fn, optimizer, device) -> tuple[float, float]:
    net.train()#进入训练模式
    total_loss = 0.0
    correct = 0
    total = 0
    for features, labels in train_iter:
        features, labels = features.to(device), labels.to(device) #转移特征和标签到设备
        optimizer.zero_grad()#清空梯度
        outputs = net(features)#输入网络获得输出
        loss = loss_fn(outputs, labels)#对比计算损失
        loss.backward()#反向传播
        optimizer.step()#更新梯度

        total_loss += float(loss.item()) * labels.numel()
        correct += int((outputs.argmax(dim=1) == labels).sum())
        total += labels.numel()
    return total_loss / total, correct / total


def plot_curves(epochs, train_losses, train_accs, test_accs, out_png: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(6.4, 4.0))
    ax1.plot(epochs, train_losses, "o-", color="#c0392b", label="train loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("train loss", color="#c0392b")
    ax1.tick_params(axis="y", labelcolor="#c0392b")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(epochs, train_accs, "s--", color="#2e86c1", label="train acc")
    ax2.plot(epochs, test_accs, "^-.", color="#27ae60", label="test acc")
    ax2.set_ylabel("accuracy")
    ax2.set_ylim(0.0, 1.0)

    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc="center right", fontsize=8)
    fig.suptitle("Fashion-MNIST MLP (784-256-10)", fontsize=11)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="G0-01 最小 MLP 训练例子")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    train_iter, test_iter = get_dataloaders(args.data_dir, args.batch_size)
    print(f"train batches: {len(train_iter)}, test batches: {len(test_iter)}")

    net = build_net(args.hidden).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr)

    epochs, train_losses, train_accs, test_accs = [], [], [], []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(net, train_iter, loss_fn, optimizer, device)
        test_acc = evaluate_accuracy(net, test_iter, device)
        epochs.append(epoch)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        print(
            f"epoch {epoch:>2}/{args.epochs}  "
            f"train loss {train_loss:.4f}  train acc {train_acc:.3f}  test acc {test_acc:.3f}"
        )

    elapsed = time.time() - started
    print(f"done in {elapsed:.1f}s, final test acc {test_accs[-1]:.3f}")

    curve_png = args.out_dir / "loss_curve.png"
    plot_curves(epochs, train_losses, train_accs, test_accs, curve_png)
    metrics_csv = args.out_dir / "metrics.csv"
    with open(metrics_csv, "w", encoding="utf-8", newline="") as f:
        f.write("epoch,train_loss,train_acc,test_acc\n")
        for e, tl, ta, sa in zip(epochs, train_losses, train_accs, test_accs):
            f.write(f"{e},{tl:.6f},{ta:.6f},{sa:.6f}\n")
    print(f"curve  -> {curve_png}")
    print(f"metrics-> {metrics_csv}")


if __name__ == "__main__":
    main()