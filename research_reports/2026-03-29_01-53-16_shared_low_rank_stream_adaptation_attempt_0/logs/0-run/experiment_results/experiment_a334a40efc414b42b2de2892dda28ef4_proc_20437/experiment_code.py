import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.manual_seed(7)
np.random.seed(7)

experiment_data = {
    "synthetic_stream": {
        "metrics": {"train": [], "val": [], "stream": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
        "per_strategy": {},
    }
}


# ---------- Synthetic related task sequence ----------
def make_task(n, d, n_classes, centers, shift_scale=0.0, noise=0.8, seed=0):
    rng = np.random.RandomState(seed)
    X, y = [], []
    shift = rng.randn(n_classes, d) * shift_scale
    for i in range(n):
        c = rng.randint(0, n_classes)
        feat = centers[c] + shift[c] + rng.randn(d) * noise
        X.append(feat)
        y.append(c)
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    return X, y


d, n_classes = 20, 4
base_centers = np.random.randn(n_classes, d).astype(np.float32) * 2.0

tasks = []
for t, s in enumerate([0.0, 0.6, 1.2]):
    X_train, y_train = make_task(
        1000, d, n_classes, base_centers, shift_scale=s, noise=0.9, seed=10 + t
    )
    X_stream, y_stream = make_task(
        300, d, n_classes, base_centers, shift_scale=s, noise=1.0, seed=30 + t
    )
    tasks.append(
        {
            "train": (X_train, y_train),
            "stream": (X_stream, y_stream),
            "name": f"task_{t}",
        }
    )

# Global normalization from source train data
all_train_X = np.concatenate([t["train"][0] for t in tasks], axis=0)
mean = all_train_X.mean(0, keepdims=True)
std = all_train_X.std(0, keepdims=True) + 1e-6


def norm(x):
    return (x - mean) / std


train_X = np.concatenate([norm(t["train"][0]) for t in tasks], axis=0)
train_y = np.concatenate([t["train"][1] for t in tasks], axis=0)

# Split train/val
perm = np.random.permutation(len(train_X))
split = int(0.85 * len(train_X))
tr_idx, va_idx = perm[:split], perm[split:]
X_tr = torch.tensor(train_X[tr_idx], dtype=torch.float32)
y_tr = torch.tensor(train_y[tr_idx], dtype=torch.long)
X_va = torch.tensor(train_X[va_idx], dtype=torch.float32)
y_va = torch.tensor(train_y[va_idx], dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=64, shuffle=True)
val_loader = DataLoader(TensorDataset(X_va, y_va), batch_size=128, shuffle=False)


# ---------- Model ----------
class LowRankAdapter(nn.Module):
    def __init__(self, dim, rank=4):
        super().__init__()
        self.A = nn.Parameter(torch.zeros(dim, rank))
        self.B = nn.Parameter(torch.zeros(rank, dim))
        nn.init.normal_(self.A, std=0.02)
        nn.init.normal_(self.B, std=0.02)

    def forward(self, x):
        return x + x @ self.A @ self.B


class ClassifierWithAdapter(nn.Module):
    def __init__(self, dim, hidden, n_classes, rank=4):
        super().__init__()
        self.adapter = LowRankAdapter(dim, rank)
        self.base = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, n_classes)
        )

    def forward(self, x):
        x = self.adapter(x)
        return self.base(x)


model = ClassifierWithAdapter(d, 64, n_classes, rank=4).to(device)
criterion = nn.CrossEntropyLoss()

# ---------- Base training: freeze adapter, train base ----------
for p in model.adapter.parameters():
    p.requires_grad = False
optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)

best_state = None
best_val = 1e9
for epoch in range(1, 11):
    model.train()
    running_loss, running_acc, n_seen = 0.0, 0.0, 0
    for xb, yb in train_loader:
        batch = {"x": xb.to(device), "y": yb.to(device)}
        optimizer.zero_grad()
        logits = model(batch["x"])
        loss = criterion(logits, batch["y"])
        loss.backward()
        optimizer.step()
        preds = logits.argmax(1)
        running_loss += loss.item() * len(batch["y"])
        running_acc += (preds == batch["y"]).float().sum().item()
        n_seen += len(batch["y"])
    train_loss = running_loss / n_seen
    train_acc = running_acc / n_seen

    model.eval()
    val_loss_sum, val_acc_sum, val_n = 0.0, 0.0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            batch = {"x": xb.to(device), "y": yb.to(device)}
            logits = model(batch["x"])
            loss = criterion(logits, batch["y"])
            preds = logits.argmax(1)
            val_loss_sum += loss.item() * len(batch["y"])
            val_acc_sum += (preds == batch["y"]).float().sum().item()
            val_n += len(batch["y"])
    val_loss = val_loss_sum / val_n
    val_acc = val_acc_sum / val_n
    print(f"Epoch {epoch}: validation_loss = {val_loss:.4f}")

    experiment_data["synthetic_stream"]["metrics"]["train"].append(
        {"epoch": epoch, "acc": train_acc}
    )
    experiment_data["synthetic_stream"]["metrics"]["val"].append(
        {"epoch": epoch, "acc": val_acc}
    )
    experiment_data["synthetic_stream"]["losses"]["train"].append(
        {"epoch": epoch, "loss": train_loss}
    )
    experiment_data["synthetic_stream"]["losses"]["val"].append(
        {"epoch": epoch, "loss": val_loss}
    )
    experiment_data["synthetic_stream"]["timestamps"].append(epoch)

    if val_loss < best_val:
        best_val = val_loss
        best_state = copy.deepcopy(model.state_dict())

model.load_state_dict(best_state)

# ---------- Prepare adaptation versions ----------
base_state = copy.deepcopy(model.state_dict())
base_anchor_A = model.adapter.A.detach().clone()
base_anchor_B = model.adapter.B.detach().clone()


def reset_to_base(m):
    m.load_state_dict(base_state)
    for p in m.base.parameters():
        p.requires_grad = False
    for p in m.adapter.parameters():
        p.requires_grad = True
    return m


def run_stream(
    strategy, batch_size=16, adapt_lr=0.08, anchor_lambda=1e-2, outlier_thresh=2.0
):
    m = ClassifierWithAdapter(d, 64, n_classes, rank=4).to(device)
    m.load_state_dict(base_state)
    for p in m.base.parameters():
        p.requires_grad = False
    for p in m.adapter.parameters():
        p.requires_grad = True
    opt = optim.SGD(m.adapter.parameters(), lr=adapt_lr)

    preds_all, gts_all, cumulative_acc = [], [], []
    batch_losses = []

    stream_X = np.concatenate([norm(t["stream"][0]) for t in tasks], axis=0)
    stream_y = np.concatenate([t["stream"][1] for t in tasks], axis=0)
    X_tensor = torch.tensor(stream_X, dtype=torch.float32)
    y_tensor = torch.tensor(stream_y, dtype=torch.long)
    loader = DataLoader(
        TensorDataset(X_tensor, y_tensor), batch_size=batch_size, shuffle=False
    )

    correct = 0
    total = 0
    running_loss_ref = None

    for i, (xb, yb) in enumerate(loader):
        batch = {"x": xb.to(device), "y": yb.to(device)}

        if strategy == "reset":
            m = reset_to_base(m)
            opt = optim.SGD(m.adapter.parameters(), lr=adapt_lr)

        m.eval()
        with torch.no_grad():
            logits = m(batch["x"])
            preds = logits.argmax(1)
        preds_all.extend(preds.cpu().numpy().tolist())
        gts_all.extend(batch["y"].cpu().numpy().tolist())
        correct += (preds == batch["y"]).float().sum().item()
        total += len(batch["y"])
        cumulative_acc.append(correct / total)

        m.train()
        opt.zero_grad()
        logits_adapt = m(batch["x"])
        loss = criterion(logits_adapt, batch["y"])

        if strategy == "anchored":
            reg = ((m.adapter.A - base_anchor_A.to(device)) ** 2).mean() + (
                (m.adapter.B - base_anchor_B.to(device)) ** 2
            ).mean()
            total_loss = loss + anchor_lambda * reg
            current = loss.item()
            if running_loss_ref is None:
                running_loss_ref = current
            do_update = current <= outlier_thresh * running_loss_ref
            if do_update:
                total_loss.backward()
                opt.step()
                running_loss_ref = 0.9 * running_loss_ref + 0.1 * current
        else:
            loss.backward()
            opt.step()

        batch_losses.append(loss.item())

    return {
        "stream_acc": float(np.mean(np.array(preds_all) == np.array(gts_all))),
        "preds": np.array(preds_all),
        "gts": np.array(gts_all),
        "cumulative_acc": np.array(cumulative_acc),
        "batch_losses": np.array(batch_losses),
    }


results = {}
for strategy in ["reset", "carry", "anchored"]:
    key = "carry" if strategy == "carry" else strategy
    run_key = strategy if strategy != "carry" else "carry"
    if run_key == "carry":
        res = run_stream("carry")
    else:
        res = run_stream(run_key)
    results[strategy] = res
    print(f"{strategy} Stream-Averaged Task Accuracy: {res['stream_acc']:.4f}")
    experiment_data["synthetic_stream"]["per_strategy"][strategy] = {
        "metrics": {"stream_acc": res["stream_acc"]},
        "losses": res["batch_losses"],
        "predictions": res["preds"],
        "ground_truth": res["gts"],
        "cumulative_acc": res["cumulative_acc"],
    }

# Save plottable data
experiment_data["synthetic_stream"]["predictions"] = results["anchored"]["preds"]
experiment_data["synthetic_stream"]["ground_truth"] = results["anchored"]["gts"]
experiment_data["synthetic_stream"]["metrics"]["stream"].append(
    {
        "reset_acc": results["reset"]["stream_acc"],
        "carry_acc": results["carry"]["stream_acc"],
        "anchored_acc": results["anchored"]["stream_acc"],
    }
)

np.save(os.path.join(working_dir, "reset_preds.npy"), results["reset"]["preds"])
np.save(os.path.join(working_dir, "carry_preds.npy"), results["carry"]["preds"])
np.save(os.path.join(working_dir, "anchored_preds.npy"), results["anchored"]["preds"])
np.save(os.path.join(working_dir, "ground_truth.npy"), results["anchored"]["gts"])
np.save(
    os.path.join(working_dir, "reset_cumulative_acc.npy"),
    results["reset"]["cumulative_acc"],
)
np.save(
    os.path.join(working_dir, "carry_cumulative_acc.npy"),
    results["carry"]["cumulative_acc"],
)
np.save(
    os.path.join(working_dir, "anchored_cumulative_acc.npy"),
    results["anchored"]["cumulative_acc"],
)

# Visualization
plt.figure(figsize=(8, 4))
for strategy, color in zip(
    ["reset", "carry", "anchored"], ["tab:blue", "tab:orange", "tab:green"]
):
    plt.plot(results[strategy]["cumulative_acc"], label=strategy, color=color)
plt.xlabel("Stream batch")
plt.ylabel("Cumulative accuracy")
plt.title("Continual Test-Time Adaptation on Synthetic Task Stream")
plt.legend()
plt.tight_layout()
plt.savefig(
    os.path.join(working_dir, "synthetic_stream_cumulative_accuracy.png"), dpi=150
)
plt.close()

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
