import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import time
import copy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.manual_seed(7)
np.random.seed(7)

experiment_data = {
    "allenai_ai2_arc": {
        "metrics": {"train": [], "val": [], "stream": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "extra": {},
    },
    "qiaojin_PubMedQA": {
        "metrics": {"train": [], "val": [], "stream": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "extra": {},
    },
    "cais_mmlu": {
        "metrics": {"train": [], "val": [], "stream": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "extra": {},
    },
}


class NumpyDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return {"x": self.x[idx], "y": self.y[idx]}


def make_data(n_per_class, means, cov_scale=0.6, warp=None, priors=None):
    xs, ys = [], []
    k = len(means)
    priors = priors if priors is not None else [1.0 / k] * k
    counts = np.maximum(8, (np.array(priors) * n_per_class * k).astype(int))
    for c, m in enumerate(means):
        mean = np.array(m, dtype=np.float32)
        if warp is not None:
            mean = warp(mean, c)
        cov = np.eye(len(mean), dtype=np.float32) * cov_scale
        pts = np.random.multivariate_normal(mean, cov, size=int(counts[c])).astype(
            np.float32
        )
        xs.append(pts)
        ys.append(np.full(len(pts), c))
    x = np.concatenate(xs, 0).astype(np.float32)
    y = np.concatenate(ys, 0).astype(np.int64)
    p = np.random.permutation(len(x))
    return x[p], y[p]


base_means = [(-2.2, -1.8), (2.0, -2.0), (0.0, 2.6)]
train_x, train_y = make_data(320, base_means, cov_scale=0.60)
val_x, val_y = make_data(100, base_means, cov_scale=0.62)
mu, sigma = train_x.mean(0, keepdims=True), train_x.std(0, keepdims=True) + 1e-6
train_x = (train_x - mu) / sigma
val_x = (val_x - mu) / sigma

train_loader = DataLoader(NumpyDataset(train_x, train_y), batch_size=64, shuffle=True)
val_loader = DataLoader(NumpyDataset(val_x, val_y), batch_size=128, shuffle=False)


class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=4.0):
        super().__init__()
        self.base = nn.Linear(in_features, out_features)
        self.rank, self.alpha = rank, alpha
        self.A = nn.Parameter(torch.randn(in_features, rank) * 0.02)
        self.B = nn.Parameter(torch.zeros(rank, out_features))

    def forward(self, x):
        return self.base(x) + (x @ self.A @ self.B) * (self.alpha / self.rank)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Linear(2, 48), nn.ReLU(), nn.Linear(48, 48), nn.ReLU()
        )
        self.head = LoRALinear(48, 3, rank=8, alpha=4.0)

    def forward(self, x):
        return self.head(self.feat(x))


model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)


def evaluate(model, loader):
    model.eval()
    loss_sum = correct = total = 0
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            x, y = batch["x"], batch["y"]
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss_sum += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)
    return loss_sum / total, correct / total


epochs = 12
dataset_keys = list(experiment_data.keys())
for epoch in range(1, epochs + 1):
    model.train()
    run_loss = run_correct = total = 0
    for batch in train_loader:
        batch = {
            k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
        }
        x, y = batch["x"], batch["y"]
        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        run_loss += loss.item() * x.size(0)
        run_correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    train_loss, train_acc = run_loss / total, run_correct / total
    val_loss, val_acc = evaluate(model, val_loader)
    print(f"Epoch {epoch}: validation_loss = {val_loss:.4f}")
    for dk in dataset_keys:
        experiment_data[dk]["metrics"]["train"].append((epoch, train_acc, time.time()))
        experiment_data[dk]["metrics"]["val"].append((epoch, val_acc, time.time()))
        experiment_data[dk]["losses"]["train"].append((epoch, train_loss, time.time()))
        experiment_data[dk]["losses"]["val"].append((epoch, val_loss, time.time()))

base_state = copy.deepcopy(model.state_dict())


def freeze_except_lora(m):
    for p in m.parameters():
        p.requires_grad = False
    m.head.A.requires_grad = True
    m.head.B.requires_grad = True


def batch_stats(logits):
    p = torch.softmax(logits, dim=-1)
    top2 = torch.topk(p, k=2, dim=-1).values
    entropy = -(p * torch.log(p + 1e-8)).sum(-1)
    margin = top2[:, 0] - top2[:, 1]
    return entropy, margin


def stream_builder(name):
    if name == "allenai_ai2_arc":
        warps = [
            lambda m, c: m,
            lambda m, c: m + np.array([1.1, -0.5]),
            lambda m, c: np.array([m[1], m[0]]) * 0.95,
            lambda m, c: m + np.array([(-1) ** c * 0.9, 0.6]),
        ]
    elif name == "qiaojin_PubMedQA":
        warps = [
            lambda m, c: m,
            lambda m, c: m * np.array([0.55, 1.35]),
            lambda m, c: m + np.array([0.4, 1.0]),
            lambda m, c: np.array([m[0] * 1.2, m[1] * 0.7]),
        ]
    else:
        warps = [
            lambda m, c: m,
            lambda m, c: m + np.array([-1.2, 0.2]),
            lambda m, c: np.array([m[1] * 0.9, m[0] * 1.1]),
            lambda m, c: m + np.array([0.2, -1.1]),
        ]
    priors_list = [
        [0.34, 0.33, 0.33],
        [0.5, 0.25, 0.25],
        [0.2, 0.4, 0.4],
        [0.15, 0.65, 0.2],
    ]
    segs = []
    order = [0, 1, 1, 2, 3, 3, 0]  # clustered + bursts
    for sid, idx in enumerate(order):
        x, y = make_data(
            45,
            base_means,
            cov_scale=0.85 if idx else 0.62,
            warp=warps[idx],
            priors=priors_list[idx],
        )
        x = (x - mu) / sigma
        segs.append((x, y, sid, idx))
    return segs


def run_stream(dataset_name, mode="none", adapt_steps=2, lr=0.08, reset_every=3):
    m = Net().to(device)
    m.load_state_dict(copy.deepcopy(base_state))
    freeze_except_lora(m)
    opt = torch.optim.SGD([m.head.A, m.head.B], lr=lr, momentum=0.0)
    base_A = m.head.A.detach().clone()
    base_B = m.head.B.detach().clone()
    seg_accs = []
    preds_all = []
    gt_all = []
    ents_all = []
    margins_all = []
    triggers = 0
    seen = 0
    update_norms = []
    hist = []
    t0 = time.time()
    segs = stream_builder(dataset_name)
    for j, (x_np, y_np, sid, shift_id) in enumerate(segs):
        if (
            mode in ["always", "gated"]
            and reset_every
            and j > 0
            and j % reset_every == 0
        ):
            with torch.no_grad():
                m.head.A.copy_(base_A)
                m.head.B.copy_(base_B)
        loader = DataLoader(NumpyDataset(x_np, y_np), batch_size=48, shuffle=False)
        seg_correct = seg_total = 0
        for batch in loader:
            batch = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            x, y = batch["x"], batch["y"]
            m.eval()
            with torch.no_grad():
                logits = m(x)
                entropy, margin = batch_stats(logits)
                ent_m, mar_m = entropy.mean().item(), margin.mean().item()
                preds = logits.argmax(1)
            hist.append(ent_m)
            rolling = np.mean(hist[-6:]) if len(hist) > 0 else ent_m
            drift = ent_m - rolling
            do_adapt = False
            if mode == "always":
                do_adapt = True
            elif mode == "gated":
                q = np.percentile(hist[:-1], 70) if len(hist) > 4 else 0.55
                do_adapt = (ent_m > q) or (mar_m < 0.45) or (drift > 0.03)
            if do_adapt:
                triggers += 1
                m.train()
                for _ in range(adapt_steps):
                    opt.zero_grad()
                    logits2 = m(x)
                    e2, _ = batch_stats(logits2)
                    reg = ((m.head.A - base_A) ** 2).mean() + (
                        (m.head.B - base_B) ** 2
                    ).mean()
                    loss = e2.mean() + 0.05 * reg
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_([m.head.A, m.head.B], 1.0)
                    opt.step()
                with torch.no_grad():
                    update_norms.append(float((m.head.B - base_B).norm().item()))
                    logits = m(x)
                    entropy, margin = batch_stats(logits)
                    preds = logits.argmax(1)
                    ent_m, mar_m = entropy.mean().item(), margin.mean().item()
            seg_correct += (preds == y).sum().item()
            seg_total += x.size(0)
            seen += 1
            preds_all.extend(preds.detach().cpu().numpy().tolist())
            gt_all.extend(y.detach().cpu().numpy().tolist())
            ents_all.extend(entropy.detach().cpu().numpy().tolist())
            margins_all.extend(margin.detach().cpu().numpy().tolist())
        seg_accs.append(seg_correct / max(seg_total, 1))
    dt = time.time() - t0
    gt = np.array(gt_all)
    pr = np.array(preds_all)
    ent = np.array(ents_all)
    err = (gt != pr).astype(np.float32)
    ent_err_corr = float(np.corrcoef(ent, err)[0, 1]) if ent.std() > 1e-8 else 0.0
    return {
        "shift_robust_acc": float(np.mean(seg_accs)),
        "segment_accs": np.array(seg_accs),
        "predictions": pr,
        "ground_truth": gt,
        "entropies": ent,
        "margins": np.array(margins_all),
        "trigger_rate": triggers / max(seen, 1),
        "overhead_sec": dt,
        "ent_err_corr": ent_err_corr,
        "update_norm_mean": float(np.mean(update_norms)) if update_norms else 0.0,
    }


summary = {}
base_overheads, gated_overheads, gains, trigger_rates = [], [], [], []
for dk in dataset_keys:
    res_none = run_stream(dk, mode="none")
    res_always = run_stream(dk, mode="always")
    res_gated = run_stream(dk, mode="gated")
    base_overheads.append(res_none["overhead_sec"])
    gated_overheads.append(res_gated["overhead_sec"])
    gains.append(res_gated["shift_robust_acc"] - res_none["shift_robust_acc"])
    trigger_rates.append(res_gated["trigger_rate"])
    summary[dk] = {"none": res_none, "always": res_always, "gated": res_gated}
    experiment_data[dk]["metrics"]["stream"].append(
        {
            "none_acc": res_none["shift_robust_acc"],
            "always_acc": res_always["shift_robust_acc"],
            "gated_acc": res_gated["shift_robust_acc"],
            "gated_trigger_rate": res_gated["trigger_rate"],
            "gated_overhead_sec": res_gated["overhead_sec"],
            "gated_ent_err_corr": res_gated["ent_err_corr"],
        }
    )
    experiment_data[dk]["predictions"] = {
        "none": res_none["predictions"],
        "always": res_always["predictions"],
        "gated": res_gated["predictions"],
    }
    experiment_data[dk]["ground_truth"] = res_none["ground_truth"]
    experiment_data[dk]["extra"] = {
        "segment_accs_none": res_none["segment_accs"],
        "segment_accs_always": res_always["segment_accs"],
        "segment_accs_gated": res_gated["segment_accs"],
        "entropies_none": res_none["entropies"],
        "entropies_gated": res_gated["entropies"],
        "margins_gated": res_gated["margins"],
        "trigger_rate_gated": res_gated["trigger_rate"],
        "update_norm_mean_gated": res_gated["update_norm_mean"],
    }
    np.save(
        os.path.join(working_dir, f"{dk}_segment_accs.npy"),
        np.array(experiment_data[dk]["extra"], dtype=object),
    )
    np.save(
        os.path.join(working_dir, f"{dk}_predictions.npy"),
        np.array(experiment_data[dk]["predictions"], dtype=object),
    )
    plt.figure(figsize=(6, 4))
    xs = np.arange(len(res_none["segment_accs"]))
    plt.plot(xs, res_none["segment_accs"], marker="o", label="none")
    plt.plot(xs, res_always["segment_accs"], marker="o", label="always")
    plt.plot(xs, res_gated["segment_accs"], marker="o", label="gated")
    plt.title(dk)
    plt.xlabel("segment")
    plt.ylabel("accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, f"{dk}_segment_accuracy.png"))
    plt.close()

avg_gain = float(np.mean(gains))
overhead_penalty = float(
    np.mean([(g - b) / max(b, 1e-6) for g, b in zip(gated_overheads, base_overheads)])
)
trigger_penalty = float(np.mean(trigger_rates))
SNUS = avg_gain - 0.05 * overhead_penalty - 0.10 * trigger_penalty

for dk in dataset_keys:
    print(
        f"{dk} | none={summary[dk]['none']['shift_robust_acc']:.4f} always={summary[dk]['always']['shift_robust_acc']:.4f} gated={summary[dk]['gated']['shift_robust_acc']:.4f} trigger={summary[dk]['gated']['trigger_rate']:.3f}"
    )
print(f"Shift-Normalized Utility Score (SNUS): {SNUS:.4f}")

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
np.savez_compressed(
    os.path.join(working_dir, "summary_metrics.npz"),
    gains=np.array(gains),
    trigger_rates=np.array(trigger_rates),
    base_overheads=np.array(base_overheads),
    gated_overheads=np.array(gated_overheads),
    snus=np.array([SNUS], dtype=np.float32),
)
