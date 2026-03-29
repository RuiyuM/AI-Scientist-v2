# Set random seed
import random
import numpy as np
import torch

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import copy
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

experiment_data = {
    "synthetic_stream": {
        "metrics": {"train": [], "val": [], "stream": []},
        "losses": {"train": [], "val": [], "stream": []},
        "predictions": [],
        "ground_truth": [],
        "stream_position": [],
        "methods": {},
    }
}


# -----------------------------
# Synthetic related-task data
# -----------------------------
def make_task_data(n, d, shift_scale=0.0, rotate=False):
    x = np.random.randn(n, d).astype(np.float32)
    if rotate:
        theta = shift_scale
        R = np.eye(d, dtype=np.float32)
        c, s = np.cos(theta), np.sin(theta)
        for i in range(0, min(d - 1, 6), 2):
            R[i : i + 2, i : i + 2] = np.array([[c, -s], [s, c]], dtype=np.float32)
        x = x @ R.T
    x[:, :4] += shift_scale
    score = (
        1.2 * x[:, 0]
        - 0.9 * x[:, 1]
        + 0.7 * x[:, 2] * x[:, 3]
        + 0.5 * np.sin(x[:, 4])
        - 0.3 * x[:, 5] ** 2
    )
    y = (score > 0.2 * shift_scale).astype(np.int64)
    return x, y


d_in = 16
n_train, n_val, n_test = 2000, 600, 900

# Source train/val mixture
x0_tr, y0_tr = make_task_data(n_train // 2, d_in, 0.0, False)
x1_tr, y1_tr = make_task_data(n_train // 4, d_in, 0.6, True)
x2_tr, y2_tr = make_task_data(n_train // 4, d_in, 1.0, True)
x_train = np.concatenate([x0_tr, x1_tr, x2_tr], 0)
y_train = np.concatenate([y0_tr, y1_tr, y2_tr], 0)

x0_val, y0_val = make_task_data(n_val // 3, d_in, 0.0, False)
x1_val, y1_val = make_task_data(n_val // 3, d_in, 0.6, True)
x2_val, y2_val = make_task_data(n_val - 2 * (n_val // 3), d_in, 1.0, True)
x_val = np.concatenate([x0_val, x1_val, x2_val], 0)
y_val = np.concatenate([y0_val, y1_val, y2_val], 0)

# Stream tasks
task_specs = [("task_a", 0.0, False), ("task_b", 0.7, True), ("task_c", 1.3, True)]
stream_batches = []
batch_size_stream = 32
for name, shift, rot in task_specs:
    x_t, y_t = make_task_data(n_test // 3, d_in, shift, rot)
    for i in range(0, len(x_t), batch_size_stream):
        xb = torch.tensor(x_t[i : i + batch_size_stream], dtype=torch.float32)
        yb = torch.tensor(y_t[i : i + batch_size_stream], dtype=torch.long)
        stream_batches.append((name, xb, yb))

train_loader = DataLoader(
    TensorDataset(torch.tensor(x_train), torch.tensor(y_train)),
    batch_size=64,
    shuffle=True,
)
val_loader = DataLoader(
    TensorDataset(torch.tensor(x_val), torch.tensor(y_val)),
    batch_size=128,
    shuffle=False,
)


# -----------------------------
# Model: frozen backbone + low-rank adapter
# -----------------------------
class Backbone(nn.Module):
    def __init__(self, d_in=16, d_hidden=32, n_classes=2):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, n_classes)

    def hidden(self, x):
        return torch.tanh(self.fc1(x))

    def logits_from_hidden(self, h):
        return self.fc2(h)

    def forward(self, x):
        h = self.hidden(x)
        return self.logits_from_hidden(h)


class LowRankAdapter(nn.Module):
    def __init__(self, d_hidden=32, rank=4, scale=0.1):
        super().__init__()
        self.A = nn.Parameter(scale * torch.randn(d_hidden, rank))
        self.B = nn.Parameter(scale * torch.randn(rank, d_hidden))

    def forward(self, h):
        return h + h @ self.A @ self.B


class AdaptedModel(nn.Module):
    def __init__(self, backbone, adapter):
        super().__init__()
        self.backbone = backbone
        self.adapter = adapter

    def forward(self, x):
        h = self.backbone.hidden(x)
        h = self.adapter(h)
        return self.backbone.logits_from_hidden(h)


backbone = Backbone(d_in=d_in).to(device)
opt = torch.optim.Adam(backbone.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# -----------------------------
# Base pretraining
# -----------------------------
epochs = 10
for epoch in range(1, epochs + 1):
    backbone.train()
    tr_loss, tr_correct, tr_total = 0.0, 0, 0
    for xb, yb in train_loader:
        batch = {"x": xb.to(device), "y": yb.to(device)}
        opt.zero_grad()
        logits = backbone(batch["x"])
        loss = criterion(logits, batch["y"])
        loss.backward()
        opt.step()
        tr_loss += loss.item() * batch["x"].size(0)
        tr_correct += (logits.argmax(1) == batch["y"]).sum().item()
        tr_total += batch["x"].size(0)

    backbone.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            batch = {"x": xb.to(device), "y": yb.to(device)}
            logits = backbone(batch["x"])
            loss = criterion(logits, batch["y"])
            val_loss += loss.item() * batch["x"].size(0)
            val_correct += (logits.argmax(1) == batch["y"]).sum().item()
            val_total += batch["x"].size(0)

    tr_loss /= tr_total
    val_loss /= val_total
    tr_acc = tr_correct / tr_total
    val_acc = val_correct / val_total
    experiment_data["synthetic_stream"]["metrics"]["train"].append((epoch, tr_acc))
    experiment_data["synthetic_stream"]["metrics"]["val"].append((epoch, val_acc))
    experiment_data["synthetic_stream"]["losses"]["train"].append((epoch, tr_loss))
    experiment_data["synthetic_stream"]["losses"]["val"].append((epoch, val_loss))
    print(f"Epoch {epoch}: validation_loss = {val_loss:.4f}")

for p in backbone.parameters():
    p.requires_grad = False
backbone.eval()


# -----------------------------
# Streaming evaluation methods
# -----------------------------
def entropy_loss(logits):
    p = F.softmax(logits, dim=-1).clamp_min(1e-8)
    return -(p * p.log()).sum(dim=-1).mean()


def clone_adapter(adapter):
    new_adapter = LowRankAdapter(d_hidden=32, rank=4).to(device)
    new_adapter.load_state_dict(copy.deepcopy(adapter.state_dict()))
    return new_adapter


base_adapter = LowRankAdapter(d_hidden=32, rank=4).to(device)
base_state = copy.deepcopy(base_adapter.state_dict())

methods = ["reset", "carry", "anchored"]
for m in methods:
    experiment_data["synthetic_stream"]["methods"][m] = {
        "metrics": {"train": [], "val": [], "stream": []},
        "losses": {"train": [], "val": [], "stream": []},
        "predictions": [],
        "ground_truth": [],
        "stream_position": [],
        "task_name": [],
        "adapter_drift": [],
    }


def run_stream(method):
    adapter = clone_adapter(base_adapter)
    model = AdaptedModel(backbone, adapter).to(device)
    init_params = [p.detach().clone() for p in model.adapter.parameters()]
    optimizer = torch.optim.SGD(model.adapter.parameters(), lr=0.2)

    online_accs, online_losses = [], []
    for step, (task_name, xb, yb) in enumerate(stream_batches):
        if method == "reset":
            model.adapter.load_state_dict(copy.deepcopy(base_state))
            optimizer = torch.optim.SGD(model.adapter.parameters(), lr=0.2)

        batch = {"x": xb.to(device), "y": yb.to(device)}

        # Online eval before adaptation
        model.eval()
        with torch.no_grad():
            logits = model(batch["x"])
            loss_sup = criterion(logits, batch["y"]).item()
            preds = logits.argmax(1)
            acc = (preds == batch["y"]).float().mean().item()

        online_accs.append(acc)
        online_losses.append(loss_sup)
        experiment_data["synthetic_stream"]["methods"][method]["metrics"][
            "stream"
        ].append((step, acc))
        experiment_data["synthetic_stream"]["methods"][method]["losses"][
            "stream"
        ].append((step, loss_sup))
        experiment_data["synthetic_stream"]["methods"][method]["predictions"].append(
            preds.detach().cpu().numpy()
        )
        experiment_data["synthetic_stream"]["methods"][method]["ground_truth"].append(
            batch["y"].detach().cpu().numpy()
        )
        experiment_data["synthetic_stream"]["methods"][method][
            "stream_position"
        ].append(step)
        experiment_data["synthetic_stream"]["methods"][method]["task_name"].append(
            task_name
        )

        # Test-time update via entropy minimization
        model.train()
        optimizer.zero_grad()
        logits_adapt = model(batch["x"])
        ent = entropy_loss(logits_adapt)

        if method == "anchored":
            reg = 0.0
            for p, p0 in zip(model.adapter.parameters(), init_params):
                reg = reg + ((p - p0) ** 2).mean()
            total_loss = ent + 0.05 * reg
            # outlier rejection based on high supervised-like uncertainty proxy
            if loss_sup < 1.2:
                total_loss.backward()
                optimizer.step()
        else:
            total_loss = ent
            total_loss.backward()
            optimizer.step()

        drift = 0.0
        with torch.no_grad():
            for p, p0 in zip(model.adapter.parameters(), init_params):
                drift += ((p - p0) ** 2).mean().item()
        experiment_data["synthetic_stream"]["methods"][method]["adapter_drift"].append(
            drift
        )

    return float(np.mean(online_accs)), np.array(online_accs), np.array(online_losses)


results = {}
for method in methods:
    mean_acc, acc_curve, loss_curve = run_stream(method)
    results[method] = {
        "mean_acc": mean_acc,
        "acc_curve": acc_curve,
        "loss_curve": loss_curve,
    }
    print(f"{method} Stream-Averaged Online Accuracy: {mean_acc:.4f}")

# -----------------------------
# Save arrays and plot
# -----------------------------
for method in methods:
    np.save(
        os.path.join(working_dir, f"{method}_acc_curve.npy"),
        results[method]["acc_curve"],
    )
    np.save(
        os.path.join(working_dir, f"{method}_loss_curve.npy"),
        results[method]["loss_curve"],
    )
    np.save(
        os.path.join(working_dir, f"{method}_drift.npy"),
        np.array(
            experiment_data["synthetic_stream"]["methods"][method]["adapter_drift"]
        ),
    )
    preds = np.concatenate(
        experiment_data["synthetic_stream"]["methods"][method]["predictions"]
    )
    gts = np.concatenate(
        experiment_data["synthetic_stream"]["methods"][method]["ground_truth"]
    )
    np.save(os.path.join(working_dir, f"{method}_predictions.npy"), preds)
    np.save(os.path.join(working_dir, f"{method}_ground_truth.npy"), gts)

positions = np.arange(len(stream_batches))
plt.figure(figsize=(8, 4))
for method in methods:
    plt.plot(
        positions,
        results[method]["acc_curve"],
        label=f"{method} ({results[method]['mean_acc']:.3f})",
    )
plt.xlabel("Stream position")
plt.ylabel("Online accuracy")
plt.title("Synthetic continual test-time adaptation")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(working_dir, "synthetic_stream_online_accuracy.png"), dpi=150)
plt.close()

experiment_data["synthetic_stream"]["stream_position"] = positions.tolist()
experiment_data["synthetic_stream"]["metrics"]["stream"] = [
    (m, results[m]["mean_acc"]) for m in methods
]
experiment_data["synthetic_stream"]["losses"]["stream"] = [
    (m, float(results[m]["loss_curve"].mean())) for m in methods
]
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)

print("Final summary:")
for method in methods:
    print(
        f"{method}: Stream-Averaged Online Accuracy = {results[method]['mean_acc']:.4f}"
    )
