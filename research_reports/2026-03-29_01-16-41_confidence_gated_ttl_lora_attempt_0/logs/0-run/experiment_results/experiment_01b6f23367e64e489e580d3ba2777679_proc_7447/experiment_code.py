import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.manual_seed(42)
np.random.seed(42)

experiment_data = {
    "synthetic_shift_stream": {
        "metrics": {"train": [], "val": [], "stream": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "segment_ids": [],
        "entropies": {"no_adapt": [], "always_on": [], "gated": []},
        "segment_accs": {"no_adapt": [], "always_on": [], "gated": []},
        "timestamps": [],
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


def make_gaussian_data(n, means, cov_scale=0.8, shift=None):
    xs, ys = [], []
    for c, m in enumerate(means):
        mean = np.array(m, dtype=np.float32)
        if shift is not None:
            mean = shift(mean, c)
        cov = np.eye(len(mean)) * cov_scale
        pts = np.random.multivariate_normal(mean, cov, size=n)
        xs.append(pts)
        ys.append(np.full(n, c))
    x = np.concatenate(xs, axis=0).astype(np.float32)
    y = np.concatenate(ys, axis=0).astype(np.int64)
    idx = np.random.permutation(len(x))
    return x[idx], y[idx]


base_means = [(-2, -2), (2, -2), (0, 2.5)]
num_classes, input_dim = 3, 2

train_x, train_y = make_gaussian_data(500, base_means, cov_scale=0.7)
val_x, val_y = make_gaussian_data(150, base_means, cov_scale=0.7)

# normalize inputs based on train stats
mu = train_x.mean(axis=0, keepdims=True)
sigma = train_x.std(axis=0, keepdims=True) + 1e-6
train_x = (train_x - mu) / sigma
val_x = (val_x - mu) / sigma

train_loader = DataLoader(NumpyDataset(train_x, train_y), batch_size=64, shuffle=True)
val_loader = DataLoader(NumpyDataset(val_x, val_y), batch_size=128, shuffle=False)


class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=4, alpha=1.0):
        super().__init__()
        self.base = nn.Linear(in_features, out_features)
        self.rank = rank
        self.alpha = alpha
        self.A = nn.Parameter(torch.zeros(in_features, rank))
        self.B = nn.Parameter(torch.zeros(rank, out_features))
        nn.init.normal_(self.A, std=0.02)
        nn.init.zeros_(self.B)

    def forward(self, x):
        return self.base(x) + (x @ self.A @ self.B) * (self.alpha / self.rank)


class SmallNet(nn.Module):
    def __init__(self, in_dim=2, hidden=32, out_dim=3, rank=4):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.head = LoRALinear(hidden, out_dim, rank=rank, alpha=1.0)

    def forward(self, x):
        h = self.feat(x)
        return self.head(h)


model = SmallNet(in_dim=input_dim, out_dim=num_classes, rank=4).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


def evaluate(model, loader):
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            x, y = batch["x"], batch["y"]
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            total_loss += loss.item() * x.size(0)
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total += x.size(0)
    return total_loss / total, total_correct / total


epochs = 20
for epoch in range(1, epochs + 1):
    model.train()
    running_loss, running_correct, total = 0.0, 0, 0
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
        running_loss += loss.item() * x.size(0)
        running_correct += (logits.argmax(dim=1) == y).sum().item()
        total += x.size(0)
    train_loss = running_loss / total
    train_acc = running_correct / total
    val_loss, val_acc = evaluate(model, val_loader)
    print(f"Epoch {epoch}: validation_loss = {val_loss:.4f}")
    experiment_data["synthetic_shift_stream"]["metrics"]["train"].append(
        (epoch, train_acc)
    )
    experiment_data["synthetic_shift_stream"]["metrics"]["val"].append((epoch, val_acc))
    experiment_data["synthetic_shift_stream"]["losses"]["train"].append(
        (epoch, train_loss)
    )
    experiment_data["synthetic_shift_stream"]["losses"]["val"].append((epoch, val_loss))
    experiment_data["synthetic_shift_stream"]["timestamps"].append(time.time())

base_state = copy.deepcopy(model.state_dict())


def freeze_except_lora(m):
    for p in m.parameters():
        p.requires_grad = False
    m.head.A.requires_grad = True
    m.head.B.requires_grad = True


def entropy_from_logits(logits):
    p = torch.softmax(logits, dim=-1)
    return -(p * torch.log(p + 1e-8)).sum(dim=-1).mean()


def build_stream():
    segs = []
    shifts = [
        lambda mean, c: mean,  # in-distribution
        lambda mean, c: mean + np.array([1.0, -0.5]),  # translation
        lambda mean, c: mean * np.array([0.6, 1.3]),  # anisotropic
        lambda mean, c: np.array([mean[1], mean[0]]) * 0.9,  # swap-ish
    ]
    for sid, sh in enumerate(shifts):
        x, y = make_gaussian_data(
            90, base_means, cov_scale=0.95 if sid > 0 else 0.7, shift=sh
        )
        x = (x - mu) / sigma
        segs.append((x, y, sid))
    return segs


stream_segments = build_stream()


def run_stream(adapt_mode="none", entropy_threshold=0.75, adapt_steps=1, lr=5e-2):
    m = SmallNet(in_dim=input_dim, out_dim=num_classes, rank=4).to(device)
    m.load_state_dict(copy.deepcopy(base_state))
    freeze_except_lora(m)
    opt = torch.optim.SGD([m.head.A, m.head.B], lr=lr)
    all_preds, all_gt, all_seg, all_ent = [], [], [], []
    seg_accs = []
    for x_np, y_np, sid in stream_segments:
        ds = NumpyDataset(x_np, y_np)
        loader = DataLoader(ds, batch_size=64, shuffle=False)
        seg_correct, seg_total = 0, 0
        for batch in loader:
            batch = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            x, y = batch["x"], batch["y"]
            m.eval()
            with torch.no_grad():
                logits = m(x)
                ent = entropy_from_logits(logits).item()
                preds = logits.argmax(dim=1)
            do_adapt = (adapt_mode == "always") or (
                adapt_mode == "gated" and ent > entropy_threshold
            )
            if adapt_mode != "none" and do_adapt:
                m.train()
                for _ in range(adapt_steps):
                    opt.zero_grad()
                    logits_adapt = m(x)
                    loss_adapt = entropy_from_logits(logits_adapt)
                    loss_adapt.backward()
                    opt.step()
                m.eval()
                with torch.no_grad():
                    logits = m(x)
                    preds = logits.argmax(dim=1)
                    ent = entropy_from_logits(logits).item()
            seg_correct += (preds == y).sum().item()
            seg_total += x.size(0)
            all_preds.extend(preds.detach().cpu().numpy().tolist())
            all_gt.extend(y.detach().cpu().numpy().tolist())
            all_seg.extend([sid] * x.size(0))
            all_ent.extend([ent] * x.size(0))
        seg_accs.append(seg_correct / seg_total)
    shift_robust_acc = float(np.mean(seg_accs))
    return {
        "shift_robust_acc": shift_robust_acc,
        "predictions": np.array(all_preds),
        "ground_truth": np.array(all_gt),
        "segment_ids": np.array(all_seg),
        "entropies": np.array(all_ent),
        "segment_accs": np.array(seg_accs),
    }


results_none = run_stream("none")
results_always = run_stream("always", adapt_steps=1, lr=5e-2)
results_gated = run_stream("gated", entropy_threshold=0.75, adapt_steps=1, lr=5e-2)

print(f"No adaptation Shift-Robust Accuracy: {results_none['shift_robust_acc']:.4f}")
print(
    f"Always-on LoRA TTA Shift-Robust Accuracy: {results_always['shift_robust_acc']:.4f}"
)
print(
    f"Confidence-gated LoRA TTA Shift-Robust Accuracy: {results_gated['shift_robust_acc']:.4f}"
)

experiment_data["synthetic_shift_stream"]["metrics"]["stream"].append(
    {
        "no_adapt": results_none["shift_robust_acc"],
        "always_on": results_always["shift_robust_acc"],
        "gated": results_gated["shift_robust_acc"],
    }
)
experiment_data["synthetic_shift_stream"]["predictions"] = {
    "no_adapt": results_none["predictions"],
    "always_on": results_always["predictions"],
    "gated": results_gated["predictions"],
}
experiment_data["synthetic_shift_stream"]["ground_truth"] = results_none["ground_truth"]
experiment_data["synthetic_shift_stream"]["segment_ids"] = results_none["segment_ids"]
experiment_data["synthetic_shift_stream"]["entropies"]["no_adapt"] = results_none[
    "entropies"
]
experiment_data["synthetic_shift_stream"]["entropies"]["always_on"] = results_always[
    "entropies"
]
experiment_data["synthetic_shift_stream"]["entropies"]["gated"] = results_gated[
    "entropies"
]
experiment_data["synthetic_shift_stream"]["segment_accs"]["no_adapt"] = results_none[
    "segment_accs"
]
experiment_data["synthetic_shift_stream"]["segment_accs"]["always_on"] = results_always[
    "segment_accs"
]
experiment_data["synthetic_shift_stream"]["segment_accs"]["gated"] = results_gated[
    "segment_accs"
]

np.save(
    os.path.join(working_dir, "train_val_losses.npy"),
    np.array(experiment_data["synthetic_shift_stream"]["losses"], dtype=object),
)
np.save(
    os.path.join(working_dir, "stream_segment_accs.npy"),
    np.array(experiment_data["synthetic_shift_stream"]["segment_accs"], dtype=object),
)
np.save(
    os.path.join(working_dir, "stream_predictions.npy"),
    np.array(experiment_data["synthetic_shift_stream"]["predictions"], dtype=object),
)
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)

epochs_arr = np.arange(1, epochs + 1)
train_losses = np.array(
    [x[1] for x in experiment_data["synthetic_shift_stream"]["losses"]["train"]]
)
val_losses = np.array(
    [x[1] for x in experiment_data["synthetic_shift_stream"]["losses"]["val"]]
)

plt.figure(figsize=(6, 4))
plt.plot(epochs_arr, train_losses, label="train_loss")
plt.plot(epochs_arr, val_losses, label="val_loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(working_dir, "synthetic_shift_train_val_loss.png"))
plt.close()

seg_idx = np.arange(len(results_none["segment_accs"]))
plt.figure(figsize=(6, 4))
plt.plot(seg_idx, results_none["segment_accs"], marker="o", label="no_adapt")
plt.plot(seg_idx, results_always["segment_accs"], marker="o", label="always_on")
plt.plot(seg_idx, results_gated["segment_accs"], marker="o", label="gated")
plt.xlabel("Stream segment")
plt.ylabel("Accuracy")
plt.title("Segment-wise Accuracy under Shift")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(working_dir, "synthetic_shift_segment_accuracy.png"))
plt.close()

plt.figure(figsize=(7, 4))
plt.plot(results_none["entropies"], label="no_adapt", alpha=0.8)
plt.plot(results_always["entropies"], label="always_on", alpha=0.8)
plt.plot(results_gated["entropies"], label="gated", alpha=0.8)
plt.xlabel("Stream sample index")
plt.ylabel("Predictive entropy")
plt.title("Entropy over evaluation stream")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(working_dir, "synthetic_shift_entropy_stream.png"))
plt.close()
