import os
import copy
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

experiment_data = {
    "batch_size_tuning": {
        "synthetic_shift_stream": {
            "config": {
                "batch_sizes": [32, 64, 128],
                "epochs": 20,
                "lr": 1e-3,
                "val_batch_size": 128,
                "stream_batch_size": 64,
                "adapt_steps": 1,
                "adapt_lr": 5e-2,
                "entropy_threshold": 0.75,
                "seed": SEED,
            },
            "runs": {},
            "summary": {
                "batch_sizes": [],
                "best_val_acc": [],
                "final_val_acc": [],
                "shift_robust_acc_no_adapt": [],
                "shift_robust_acc_always_on": [],
                "shift_robust_acc_gated": [],
            },
        }
    }
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


def evaluate(model, loader):
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            total_loss += loss.item() * x.size(0)
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total += x.size(0)
    return total_loss / total, total_correct / total


def freeze_except_lora(m):
    for p in m.parameters():
        p.requires_grad = False
    m.head.A.requires_grad = True
    m.head.B.requires_grad = True


def entropy_from_logits(logits):
    p = torch.softmax(logits, dim=-1)
    return -(p * torch.log(p + 1e-8)).sum(dim=-1).mean()


def build_stream(base_means, mu, sigma):
    segs = []
    shifts = [
        lambda mean, c: mean,
        lambda mean, c: mean + np.array([1.0, -0.5]),
        lambda mean, c: mean * np.array([0.6, 1.3]),
        lambda mean, c: np.array([mean[1], mean[0]]) * 0.9,
    ]
    for sid, sh in enumerate(shifts):
        x, y = make_gaussian_data(
            90, base_means, cov_scale=0.95 if sid > 0 else 0.7, shift=sh
        )
        x = (x - mu) / sigma
        segs.append((x, y, sid))
    return segs


def run_stream(
    base_state,
    stream_segments,
    input_dim,
    num_classes,
    adapt_mode="none",
    entropy_threshold=0.75,
    adapt_steps=1,
    lr=5e-2,
    stream_batch_size=64,
):
    m = SmallNet(in_dim=input_dim, out_dim=num_classes, rank=4).to(device)
    m.load_state_dict(copy.deepcopy(base_state))
    freeze_except_lora(m)
    opt = torch.optim.SGD([m.head.A, m.head.B], lr=lr)

    all_preds, all_gt, all_seg, all_ent = [], [], [], []
    seg_accs = []
    for x_np, y_np, sid in stream_segments:
        ds = NumpyDataset(x_np, y_np)
        loader = DataLoader(ds, batch_size=stream_batch_size, shuffle=False)
        seg_correct, seg_total = 0, 0
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)

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

    return {
        "shift_robust_acc": float(np.mean(seg_accs)),
        "predictions": np.array(all_preds),
        "ground_truth": np.array(all_gt),
        "segment_ids": np.array(all_seg),
        "entropies": np.array(all_ent),
        "segment_accs": np.array(seg_accs),
    }


# data
base_means = [(-2, -2), (2, -2), (0, 2.5)]
num_classes, input_dim = 3, 2
train_x, train_y = make_gaussian_data(500, base_means, cov_scale=0.7)
val_x, val_y = make_gaussian_data(150, base_means, cov_scale=0.7)

mu = train_x.mean(axis=0, keepdims=True)
sigma = train_x.std(axis=0, keepdims=True) + 1e-6
train_x = (train_x - mu) / sigma
val_x = (val_x - mu) / sigma

val_loader = DataLoader(
    NumpyDataset(train_x[:0], train_y[:0]), batch_size=128, shuffle=False
)  # placeholder
val_loader = DataLoader(NumpyDataset(val_x, val_y), batch_size=128, shuffle=False)
stream_segments = build_stream(base_means, mu, sigma)

batch_sizes = experiment_data["batch_size_tuning"]["synthetic_shift_stream"]["config"][
    "batch_sizes"
]
epochs = experiment_data["batch_size_tuning"]["synthetic_shift_stream"]["config"][
    "epochs"
]
train_lr = experiment_data["batch_size_tuning"]["synthetic_shift_stream"]["config"][
    "lr"
]
stream_batch_size = experiment_data["batch_size_tuning"]["synthetic_shift_stream"][
    "config"
]["stream_batch_size"]
adapt_steps = experiment_data["batch_size_tuning"]["synthetic_shift_stream"]["config"][
    "adapt_steps"
]
adapt_lr = experiment_data["batch_size_tuning"]["synthetic_shift_stream"]["config"][
    "adapt_lr"
]
entropy_threshold = experiment_data["batch_size_tuning"]["synthetic_shift_stream"][
    "config"
]["entropy_threshold"]

for bs in batch_sizes:
    print(f"\n=== Training with batch size {bs} ===")
    train_loader = DataLoader(
        NumpyDataset(train_x, train_y), batch_size=bs, shuffle=True
    )

    model = SmallNet(in_dim=input_dim, out_dim=num_classes, rank=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_lr)

    run_data = {
        "metrics": {"train": [], "val": [], "stream": []},
        "losses": {"train": [], "val": []},
        "predictions": {},
        "ground_truth": None,
        "segment_ids": None,
        "entropies": {"no_adapt": [], "always_on": [], "gated": []},
        "segment_accs": {"no_adapt": [], "always_on": [], "gated": []},
        "timestamps": [],
        "config": {
            "train_batch_size": bs,
            "val_batch_size": 128,
            "epochs": epochs,
            "lr": train_lr,
            "adapt_steps": adapt_steps,
            "adapt_lr": adapt_lr,
            "entropy_threshold": entropy_threshold,
            "stream_batch_size": stream_batch_size,
        },
    }

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, running_correct, total = 0.0, 0, 0
        for batch in train_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
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

        print(
            f"Batch {bs} | Epoch {epoch:02d} | "
            f"train_loss {train_loss:.4f} | train_acc {train_acc:.4f} | "
            f"val_loss {val_loss:.4f} | val_acc {val_acc:.4f}"
        )

        run_data["metrics"]["train"].append((epoch, train_acc))
        run_data["metrics"]["val"].append((epoch, val_acc))
        run_data["losses"]["train"].append((epoch, train_loss))
        run_data["losses"]["val"].append((epoch, val_loss))
        run_data["timestamps"].append(time.time())

    base_state = copy.deepcopy(model.state_dict())

    results_none = run_stream(
        base_state,
        stream_segments,
        input_dim,
        num_classes,
        adapt_mode="none",
        entropy_threshold=entropy_threshold,
        adapt_steps=adapt_steps,
        lr=adapt_lr,
        stream_batch_size=stream_batch_size,
    )
    results_always = run_stream(
        base_state,
        stream_segments,
        input_dim,
        num_classes,
        adapt_mode="always",
        entropy_threshold=entropy_threshold,
        adapt_steps=adapt_steps,
        lr=adapt_lr,
        stream_batch_size=stream_batch_size,
    )
    results_gated = run_stream(
        base_state,
        stream_segments,
        input_dim,
        num_classes,
        adapt_mode="gated",
        entropy_threshold=entropy_threshold,
        adapt_steps=adapt_steps,
        lr=adapt_lr,
        stream_batch_size=stream_batch_size,
    )

    print(
        f"Batch {bs} No adaptation Shift-Robust Accuracy: {results_none['shift_robust_acc']:.4f}"
    )
    print(
        f"Batch {bs} Always-on LoRA TTA Shift-Robust Accuracy: {results_always['shift_robust_acc']:.4f}"
    )
    print(
        f"Batch {bs} Confidence-gated LoRA TTA Shift-Robust Accuracy: {results_gated['shift_robust_acc']:.4f}"
    )

    run_data["metrics"]["stream"].append(
        {
            "no_adapt": results_none["shift_robust_acc"],
            "always_on": results_always["shift_robust_acc"],
            "gated": results_gated["shift_robust_acc"],
        }
    )
    run_data["predictions"] = {
        "no_adapt": results_none["predictions"],
        "always_on": results_always["predictions"],
        "gated": results_gated["predictions"],
    }
    run_data["ground_truth"] = results_none["ground_truth"]
    run_data["segment_ids"] = results_none["segment_ids"]
    run_data["entropies"]["no_adapt"] = results_none["entropies"]
    run_data["entropies"]["always_on"] = results_always["entropies"]
    run_data["entropies"]["gated"] = results_gated["entropies"]
    run_data["segment_accs"]["no_adapt"] = results_none["segment_accs"]
    run_data["segment_accs"]["always_on"] = results_always["segment_accs"]
    run_data["segment_accs"]["gated"] = results_gated["segment_accs"]

    key = f"batch_size_{bs}"
    experiment_data["batch_size_tuning"]["synthetic_shift_stream"]["runs"][
        key
    ] = run_data

    val_accs = [x[1] for x in run_data["metrics"]["val"]]
    experiment_data["batch_size_tuning"]["synthetic_shift_stream"]["summary"][
        "batch_sizes"
    ].append(bs)
    experiment_data["batch_size_tuning"]["synthetic_shift_stream"]["summary"][
        "best_val_acc"
    ].append(float(np.max(val_accs)))
    experiment_data["batch_size_tuning"]["synthetic_shift_stream"]["summary"][
        "final_val_acc"
    ].append(float(val_accs[-1]))
    experiment_data["batch_size_tuning"]["synthetic_shift_stream"]["summary"][
        "shift_robust_acc_no_adapt"
    ].append(results_none["shift_robust_acc"])
    experiment_data["batch_size_tuning"]["synthetic_shift_stream"]["summary"][
        "shift_robust_acc_always_on"
    ].append(results_always["shift_robust_acc"])
    experiment_data["batch_size_tuning"]["synthetic_shift_stream"]["summary"][
        "shift_robust_acc_gated"
    ].append(results_gated["shift_robust_acc"])

# save all plottable data to a single required file
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)

# plots
summary = experiment_data["batch_size_tuning"]["synthetic_shift_stream"]["summary"]
bs_arr = np.array(summary["batch_sizes"])
best_val_arr = np.array(summary["best_val_acc"])
final_val_arr = np.array(summary["final_val_acc"])
none_arr = np.array(summary["shift_robust_acc_no_adapt"])
always_arr = np.array(summary["shift_robust_acc_always_on"])
gated_arr = np.array(summary["shift_robust_acc_gated"])

plt.figure(figsize=(7, 4))
for bs in batch_sizes:
    run = experiment_data["batch_size_tuning"]["synthetic_shift_stream"]["runs"][
        f"batch_size_{bs}"
    ]
    epochs_arr = np.array([x[0] for x in run["metrics"]["val"]])
    val_accs = np.array([x[1] for x in run["metrics"]["val"]])
    plt.plot(epochs_arr, val_accs, marker="o", label=f"bs={bs}")
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.title("Validation Accuracy by Train Batch Size")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(working_dir, "batch_size_val_accuracy.png"))
plt.close()

plt.figure(figsize=(7, 4))
plt.plot(bs_arr, best_val_arr, marker="o", label="best_val_acc")
plt.plot(bs_arr, final_val_arr, marker="s", label="final_val_acc")
plt.xlabel("Train Batch Size")
plt.ylabel("Accuracy")
plt.title("Validation Accuracy Summary")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(working_dir, "batch_size_val_summary.png"))
plt.close()

plt.figure(figsize=(7, 4))
plt.plot(bs_arr, none_arr, marker="o", label="no_adapt")
plt.plot(bs_arr, always_arr, marker="o", label="always_on")
plt.plot(bs_arr, gated_arr, marker="o", label="gated")
plt.xlabel("Train Batch Size")
plt.ylabel("Shift-Robust Accuracy")
plt.title("Shift Robustness vs Train Batch Size")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(working_dir, "batch_size_shift_robust_summary.png"))
plt.close()

best_idx = int(np.argmax(gated_arr))
best_bs = int(bs_arr[best_idx])
best_run = experiment_data["batch_size_tuning"]["synthetic_shift_stream"]["runs"][
    f"batch_size_{best_bs}"
]

seg_idx = np.arange(len(best_run["segment_accs"]["no_adapt"]))
plt.figure(figsize=(7, 4))
plt.plot(seg_idx, best_run["segment_accs"]["no_adapt"], marker="o", label="no_adapt")
plt.plot(seg_idx, best_run["segment_accs"]["always_on"], marker="o", label="always_on")
plt.plot(seg_idx, best_run["segment_accs"]["gated"], marker="o", label="gated")
plt.xlabel("Stream segment")
plt.ylabel("Accuracy")
plt.title(f"Segment Accuracy under Shift (best gated bs={best_bs})")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(working_dir, "best_batch_size_segment_accuracy.png"))
plt.close()

plt.figure(figsize=(8, 4))
plt.plot(best_run["entropies"]["no_adapt"], label="no_adapt", alpha=0.8)
plt.plot(best_run["entropies"]["always_on"], label="always_on", alpha=0.8)
plt.plot(best_run["entropies"]["gated"], label="gated", alpha=0.8)
plt.xlabel("Stream sample index")
plt.ylabel("Predictive entropy")
plt.title(f"Entropy over Evaluation Stream (best gated bs={best_bs})")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(working_dir, "best_batch_size_entropy_stream.png"))
plt.close()

print("\n=== Summary ===")
for i, bs in enumerate(bs_arr):
    print(
        f"bs={int(bs)} | best_val={best_val_arr[i]:.4f} | final_val={final_val_arr[i]:.4f} | "
        f"no_adapt={none_arr[i]:.4f} | always_on={always_arr[i]:.4f} | gated={gated_arr[i]:.4f}"
    )
print(f"Saved experiment data to: {os.path.join(working_dir, 'experiment_data.npy')}")
