# -*- coding: utf-8 -*-
"""G0-04 最小 GRU 序列分类例子。

任务（合成的 toy 任务，刻意设计成"必须看顺序"）：
    序列长度 T=20，特征 F=8。
    一段 4 帧的高能量 burst 出现在前半段 -> 类别 0
    同一段 burst 出现在后半段        -> 类别 1
    两类都含 burst，只有"位置"不同，所以对全序列取平均是分不出来的，
    必须按时间顺序读才能判断。

关注点不是刷分，而是搞清三个东西：
    input         (B, T, F)
    hidden state  (num_layers, B, H)
    output        (B, T, H)

脚本会数值验证 output[:, -1, :] 是否等于 h_n[-1]，并把结论写入 io_table.md。

用法：
    python testother/g0-04_gru.py
    python testother/g0-04_gru.py --epochs 8 --hidden 32
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
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = PROJECT_ROOT / "results" / "g0-04_gru"


def make_dataset(n_samples, seq_len, n_features, burst_len, generator):
    """burst 随机落在前半段(类别0)或后半段(类别1)。"""
    x = torch.randn(n_samples, seq_len, n_features, generator=generator)
    y = torch.randint(0, 2, (n_samples,), generator=generator)
    half = seq_len // 2
    for i in range(n_samples):
        if y[i].item() == 0:
            start = torch.randint(0, half - burst_len + 1, (1,), generator=generator).item()
        else:
            start = torch.randint(half, seq_len - burst_len + 1, (1,), generator=generator).item()
        x[i, start : start + burst_len, :] += 2.0
    return TensorDataset(x, y)


class ToyGRU(nn.Module):
    def __init__(self, n_features: int, hidden: int, n_classes: int = 2, num_layers: int = 1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden, n_classes)

    def forward(self, x):
        output, h_n = self.gru(x)      # output:(B,T,H)  h_n:(num_layers,B,H)
        last = output[:, -1, :]        # 取最后时间步的 hidden
        return self.fc(last), output, h_n


def describe_io(net, x, output, h_n, hidden, num_layers) -> str:
    b, t, f = x.shape
    matches = torch.allclose(output[:, -1, :], h_n[-1], atol=1e-6)
    max_diff = (output[:, -1, :] - h_n[-1]).abs().max().item()

    lines = []
    lines.append("三个张量的形状与含义：")
    lines.append("")
    lines.append(f"  input        : {tuple(x.shape)}  = (B, T, F) = (batch={b}, 时间步={t}, 每帧特征={f})")
    lines.append("                 batch_first=True 时第二维才是时间；若为 False 则是 (T, B, F)")
    lines.append(f"  output       : {tuple(output.shape)}  = (B, T, H) = (batch, 每个时间步, hidden={hidden})")
    lines.append("                 每个时间步都产出一个 hidden，是'逐帧记忆'的完整快照")
    lines.append(f"  h_n          : {tuple(h_n.shape)}  = (num_layers={num_layers}, B, H)")
    lines.append("                 只保留最后一个时间步的 hidden，是'读完整个序列后的记忆'")
    lines.append("")
    lines.append("关键关系：")
    lines.append(f"  output[:, -1, :] 与 h_n[-1] 相等: {matches}（最大差 {max_diff:.2e}）")
    lines.append("  也就是说：output 的最后一帧 == h_n。h_n 只是 output 末帧的便捷写法。")
    lines.append("")
    lines.append("初始 hidden state：")
    lines.append("  不传 h_0 时 GRU 默认用全零，形状 (num_layers, B, H)。")
    lines.append("  它相当于'进入序列前的记忆'，流式推理时要手动在帧间传递它。")
    lines.append("")
    lines.append("分类怎么用：")
    lines.append(f"  取 output[:, -1, :] -> (B, {hidden}) -> Linear({hidden}, 2) -> (B, 2)")
    lines.append("  因为判断需要整段序列的信息，所以用最后一个时间步代表整条序列。")
    return "\n".join(lines)


def evaluate(net, data_iter, device) -> float:
    net.eval()
    correct = total = 0
    with torch.no_grad():
        for features, labels in data_iter:
            features, labels = features.to(device), labels.to(device)
            logits, _, _ = net(features)
            correct += int((logits.argmax(dim=1) == labels).sum())
            total += labels.numel()
    return correct / total


def main() -> None:
    parser = argparse.ArgumentParser(description="G0-04 最小 GRU 序列分类例子")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=20)
    parser.add_argument("--n-features", type=int, default=8)
    parser.add_argument("--burst-len", type=int, default=4)
    parser.add_argument("--n-train", type=int, default=4096)
    parser.add_argument("--n-test", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    g = torch.Generator().manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    train_set = make_dataset(args.n_train, args.seq_len, args.n_features, args.burst_len, g)
    test_set = make_dataset(args.n_test, args.seq_len, args.n_features, args.burst_len, g)
    train_iter = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_iter = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    # 基线：全序列取平均后分类，验证"必须看顺序"
    x_all = train_set.tensors[0]
    y_all = train_set.tensors[1]
    mean_feat = x_all.mean(dim=1)
    probe = nn.Linear(args.n_features, 2).to(device)
    probe_opt = torch.optim.Adam(probe.parameters(), lr=0.05)
    probe_loss = nn.CrossEntropyLoss()
    for _ in range(300):
        probe_opt.zero_grad()
        loss = probe_loss(probe(mean_feat.to(device)), y_all.to(device))
        loss.backward()
        probe_opt.step()
    with torch.no_grad():
        probe_acc = (probe(train_set.tensors[0].mean(1).to(device)).argmax(1)
                     == train_set.tensors[1].to(device)).float().mean().item()
    print(f"对照组（只用全序列平均，不看顺序）train acc: {probe_acc:.3f}   <- 接近 0.5 说明顺序信息是必需的")

    net = ToyGRU(args.n_features, args.hidden).to(device)
    sample_x, _ = next(iter(DataLoader(train_set, batch_size=args.batch_size)))
    sample_x = sample_x.to(device)
    with torch.no_grad():
        _, sample_out, sample_h = net(sample_x)
    io_report = describe_io(net, sample_x, sample_out, sample_h, args.hidden, 1)

    print("\n=== input / hidden state / output ===")
    print(io_report)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    io_md = args.out_dir / "io_table.md"
    io_md.write_text("# GRU 的 input / hidden / output\n\n```text\n" + io_report + "\n```\n", encoding="utf-8")
    print(f"\n说明 -> {io_md}")

    params = sum(p.numel() for p in net.parameters())
    print(f"GRU 参数量: {params:,}")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    losses, accs = [], []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        net.train()
        total_loss = correct = total = 0
        for features, labels in train_iter:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            logits, _, _ = net(features)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * labels.numel()
            correct += int((logits.argmax(dim=1) == labels).sum())
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
    ax2.axhline(0.5, color="#888888", linestyle=":", linewidth=1)
    ax2.set_ylabel("test accuracy")
    ax2.set_ylim(0, 1)
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="center right", fontsize=8)
    fig.suptitle("G0-04 Toy GRU (burst position classification)", fontsize=11)
    fig.tight_layout()
    curve = args.out_dir / "gru_result.png"
    fig.savefig(curve, dpi=140)
    plt.close(fig)
    print(f"GRU toy result -> {curve}")


if __name__ == "__main__":
    main()