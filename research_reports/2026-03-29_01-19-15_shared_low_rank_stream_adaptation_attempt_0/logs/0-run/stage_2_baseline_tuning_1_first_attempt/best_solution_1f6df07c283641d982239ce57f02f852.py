import os
import copy
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

BASE_SEED = 42
torch.manual_seed(BASE_SEED)
np.random.seed(BASE_SEED)
random.seed(BASE_SEED)

experiment_data = {
    "pretraining_epochs_tuning": {
        "synthetic_stream": {
            "epoch_candidates": [],
            "trials": {},
            "selection": {},
            "plots": {},
        }
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

# Fixed synthetic dataset generation
np.random.seed(BASE_SEED)
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

task_specs = [("task_a", 0.0, False), ("task_b", 0.7, True), ("task_c", 1.3, True)]
stream_batches = []
batch_size_stream = 32
for name, shift, rot in task_specs:
    x_t, y_t = make_task_data(n_test // 3, d_in, shift, rot)
    for i in range(0, len(x_t), batch_size_stream):
        xb = torch.tensor(x_t[i : i + batch_size_stream], dtype=torch.float32)
        yb = torch.tensor(y_t[i : i + batch_size_stream], dtype=torch.long)
        stream_batches.append((name, xb, yb))

train_dataset = TensorDataset(
    torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)
)
val_dataset = TensorDataset(
    torch.tensor(x_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long)
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


criterion = nn.CrossEntropyLoss()
methods = ["reset", "carry", "anchored"]
epoch_candidates = [10, 20, 40, 80]
experiment_data["pretraining_epochs_tuning"]["synthetic_stream"][
    "epoch_candidates"
] = epoch_candidates


def make_train_loader(seed, batch_size=64):
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=g)


val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)


def entropy_loss(logits):
    p = F.softmax(logits, dim=-1).clamp_min(1e-8)
    return -(p * p.log()).sum(dim=-1).mean()


def clone_adapter(adapter):
    new_adapter = LowRankAdapter(d_hidden=32, rank=4).to(device)
    new_adapter.load_state_dict(copy.deepcopy(adapter.state_dict()))
    return new_adapter


def pretrain_backbone(num_epochs, init_seed=42, loader_seed=999, lr=1e-3):
    torch.manual_seed(init_seed)
    np.random.seed(BASE_SEED)
    random.seed(BASE_SEED)

    backbone = Backbone(d_in=d_in).to(device)
    opt = torch.optim.Adam(backbone.parameters(), lr=lr)
    train_loader = make_train_loader(loader_seed)

    hist = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
    }

    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_state = copy.deepcopy(backbone.state_dict())

    for epoch in range(1, num_epochs + 1):
        backbone.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = backbone(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * xb.size(0)
            tr_correct += (logits.argmax(1) == yb).sum().item()
            tr_total += xb.size(0)

        backbone.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = backbone(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item() * xb.size(0)
                val_correct += (logits.argmax(1) == yb).sum().item()
                val_total += xb.size(0)

        tr_loss /= tr_total
        val_loss /= val_total
        tr_acc = tr_correct / tr_total
        val_acc = val_correct / val_total

        hist["metrics"]["train"].append((epoch, tr_acc))
        hist["metrics"]["val"].append((epoch, val_acc))
        hist["losses"]["train"].append((epoch, tr_loss))
        hist["losses"]["val"].append((epoch, val_loss))

        if (val_acc > best_val_acc) or (
            math.isclose(val_acc, best_val_acc) and val_loss < best_val_loss
        ):
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_state = copy.deepcopy(backbone.state_dict())

        print(
            f"[pretrain epochs={num_epochs}] epoch {epoch:03d} | "
            f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

    backbone.load_state_dict(best_state)
    for p in backbone.parameters():
        p.requires_grad = False
    backbone.eval()
    return backbone, hist, best_val_acc, best_val_loss


def init_method_store():
    out = {}
    for m in methods:
        out[m] = {
            "metrics": {"train": [], "val": [], "stream": []},
            "losses": {"train": [], "val": [], "stream": []},
            "predictions": [],
            "ground_truth": [],
            "stream_position": [],
            "task_name": [],
            "adapter_drift": [],
        }
    return out


def run_stream(backbone):
    trial_methods_data = init_method_store()
    results = {}

    torch.manual_seed(BASE_SEED)
    base_adapter = LowRankAdapter(d_hidden=32, rank=4).to(device)
    base_state = copy.deepcopy(base_adapter.state_dict())

    for method in methods:
        adapter = clone_adapter(base_adapter)
        model = AdaptedModel(backbone, adapter).to(device)
        init_params = [p.detach().clone() for p in model.adapter.parameters()]
        optimizer = torch.optim.SGD(model.adapter.parameters(), lr=0.2)

        online_accs, online_losses = [], []
        for step, (task_name, xb, yb) in enumerate(stream_batches):
            if method == "reset":
                model.adapter.load_state_dict(copy.deepcopy(base_state))
                optimizer = torch.optim.SGD(model.adapter.parameters(), lr=0.2)

            xb, yb = xb.to(device), yb.to(device)

            model.eval()
            with torch.no_grad():
                logits = model(xb)
                loss_sup = criterion(logits, yb).item()
                preds = logits.argmax(1)
                acc = (preds == yb).float().mean().item()

            online_accs.append(acc)
            online_losses.append(loss_sup)
            trial_methods_data[method]["metrics"]["stream"].append((step, acc))
            trial_methods_data[method]["losses"]["stream"].append((step, loss_sup))
            trial_methods_data[method]["predictions"].append(
                preds.detach().cpu().numpy()
            )
            trial_methods_data[method]["ground_truth"].append(yb.detach().cpu().numpy())
            trial_methods_data[method]["stream_position"].append(step)
            trial_methods_data[method]["task_name"].append(task_name)

            model.train()
            optimizer.zero_grad()
            logits_adapt = model(xb)
            ent = entropy_loss(logits_adapt)

            if method == "anchored":
                reg = 0.0
                for p, p0 in zip(model.adapter.parameters(), init_params):
                    reg = reg + ((p - p0) ** 2).mean()
                total_loss = ent + 0.05 * reg
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
            trial_methods_data[method]["adapter_drift"].append(drift)

        results[method] = {
            "mean_acc": float(np.mean(online_accs)),
            "acc_curve": np.array(online_accs, dtype=np.float32),
            "loss_curve": np.array(online_losses, dtype=np.float32),
        }
        print(
            f"[stream][method={method}] mean_online_acc={results[method]['mean_acc']:.4f}"
        )

    return results, trial_methods_data


# -----------------------------
# Hyperparameter tuning loop
# -----------------------------
summary_rows = []
all_val_curves = {}
all_loss_curves = {}

for num_epochs in epoch_candidates:
    print(f"\n=== Tuning trial: pretraining epochs = {num_epochs} ===")
    backbone, pretrain_hist, best_val_acc, best_val_loss = pretrain_backbone(
        num_epochs=num_epochs, init_seed=BASE_SEED, loader_seed=BASE_SEED + 100
    )
    stream_results, method_data = run_stream(backbone)

    trial_key = f"epochs_{num_epochs}"
    experiment_data["pretraining_epochs_tuning"]["synthetic_stream"]["trials"][
        trial_key
    ] = {
        "metrics": {
            "train": pretrain_hist["metrics"]["train"],
            "val": pretrain_hist["metrics"]["val"],
            "stream": [(m, stream_results[m]["mean_acc"]) for m in methods],
        },
        "losses": {
            "train": pretrain_hist["losses"]["train"],
            "val": pretrain_hist["losses"]["val"],
            "stream": [
                (m, float(stream_results[m]["loss_curve"].mean())) for m in methods
            ],
        },
        "methods": method_data,
        "best_val_acc": float(best_val_acc),
        "best_val_loss": float(best_val_loss),
    }

    val_acc_curve = np.array(
        [v for _, v in pretrain_hist["metrics"]["val"]], dtype=np.float32
    )
    val_loss_curve = np.array(
        [v for _, v in pretrain_hist["losses"]["val"]], dtype=np.float32
    )
    all_val_curves[num_epochs] = val_acc_curve
    all_loss_curves[num_epochs] = val_loss_curve

    for method in methods:
        preds = np.concatenate(method_data[method]["predictions"])
        gts = np.concatenate(method_data[method]["ground_truth"])
        drifts = np.array(method_data[method]["adapter_drift"], dtype=np.float32)
        acc_curve = stream_results[method]["acc_curve"]
        loss_curve = stream_results[method]["loss_curve"]
        np.save(
            os.path.join(working_dir, f"{trial_key}_{method}_predictions.npy"), preds
        )
        np.save(
            os.path.join(working_dir, f"{trial_key}_{method}_ground_truth.npy"), gts
        )
        np.save(os.path.join(working_dir, f"{trial_key}_{method}_drift.npy"), drifts)
        np.save(
            os.path.join(working_dir, f"{trial_key}_{method}_acc_curve.npy"), acc_curve
        )
        np.save(
            os.path.join(working_dir, f"{trial_key}_{method}_loss_curve.npy"),
            loss_curve,
        )

    np.save(
        os.path.join(working_dir, f"{trial_key}_train_acc.npy"),
        np.array(pretrain_hist["metrics"]["train"], dtype=np.float32),
    )
    np.save(
        os.path.join(working_dir, f"{trial_key}_val_acc.npy"),
        np.array(pretrain_hist["metrics"]["val"], dtype=np.float32),
    )
    np.save(
        os.path.join(working_dir, f"{trial_key}_train_loss.npy"),
        np.array(pretrain_hist["losses"]["train"], dtype=np.float32),
    )
    np.save(
        os.path.join(working_dir, f"{trial_key}_val_loss.npy"),
        np.array(pretrain_hist["losses"]["val"], dtype=np.float32),
    )

    row = {
        "epochs": num_epochs,
        "best_val_acc": float(best_val_acc),
        "best_val_loss": float(best_val_loss),
    }
    for method in methods:
        row[f"{method}_mean_acc"] = float(stream_results[method]["mean_acc"])
    summary_rows.append(row)

# -----------------------------
# Select best epoch count
# -----------------------------
summary_rows = sorted(
    summary_rows, key=lambda r: (-r["best_val_acc"], r["best_val_loss"], r["epochs"])
)
best_row = summary_rows[0]
best_epochs = best_row["epochs"]
experiment_data["pretraining_epochs_tuning"]["synthetic_stream"]["selection"] = {
    "criterion": "highest best validation accuracy, tie-broken by lower validation loss",
    "best_epochs": int(best_epochs),
    "best_val_acc": float(best_row["best_val_acc"]),
    "best_val_loss": float(best_row["best_val_loss"]),
}

# -----------------------------
# Save summary arrays
# -----------------------------
epochs_arr = np.array([r["epochs"] for r in summary_rows], dtype=np.int32)
best_val_acc_arr = np.array([r["best_val_acc"] for r in summary_rows], dtype=np.float32)
best_val_loss_arr = np.array(
    [r["best_val_loss"] for r in summary_rows], dtype=np.float32
)
np.save(os.path.join(working_dir, "tuned_epochs.npy"), epochs_arr)
np.save(os.path.join(working_dir, "tuned_best_val_acc.npy"), best_val_acc_arr)
np.save(os.path.join(working_dir, "tuned_best_val_loss.npy"), best_val_loss_arr)

for method in methods:
    arr = np.array([r[f"{method}_mean_acc"] for r in summary_rows], dtype=np.float32)
    np.save(os.path.join(working_dir, f"tuned_{method}_mean_stream_acc.npy"), arr)

# -----------------------------
# Plots
# -----------------------------
plt.figure(figsize=(8, 4))
for num_epochs in epoch_candidates:
    x = np.arange(1, len(all_val_curves[num_epochs]) + 1)
    plt.plot(x, all_val_curves[num_epochs], label=f"{num_epochs} ep")
plt.xlabel("Pretraining epoch")
plt.ylabel("Validation accuracy")
plt.title("Backbone pretraining validation accuracy")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(working_dir, "pretraining_val_accuracy_curves.png"), dpi=150)
plt.close()

plt.figure(figsize=(8, 4))
for num_epochs in epoch_candidates:
    x = np.arange(1, len(all_loss_curves[num_epochs]) + 1)
    plt.plot(x, all_loss_curves[num_epochs], label=f"{num_epochs} ep")
plt.xlabel("Pretraining epoch")
plt.ylabel("Validation loss")
plt.title("Backbone pretraining validation loss")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(working_dir, "pretraining_val_loss_curves.png"), dpi=150)
plt.close()

plt.figure(figsize=(7, 4))
plt.plot(epochs_arr, best_val_acc_arr, marker="o")
plt.xlabel("Pretraining epochs")
plt.ylabel("Best validation accuracy")
plt.title("Validation accuracy vs pretraining epochs")
plt.tight_layout()
plt.savefig(os.path.join(working_dir, "best_val_acc_vs_epochs.png"), dpi=150)
plt.close()

plt.figure(figsize=(7, 4))
for method in methods:
    arr = np.array([r[f"{method}_mean_acc"] for r in summary_rows], dtype=np.float32)
    plt.plot(epochs_arr, arr, marker="o", label=method)
plt.xlabel("Pretraining epochs")
plt.ylabel("Stream-averaged online accuracy")
plt.title("Downstream stream accuracy vs pretraining epochs")
plt.legend()
plt.tight_layout()
plt.savefig(
    os.path.join(working_dir, "stream_accuracy_vs_pretraining_epochs.png"), dpi=150
)
plt.close()

experiment_data["pretraining_epochs_tuning"]["synthetic_stream"]["plots"] = {
    "val_accuracy_curves": "pretraining_val_accuracy_curves.png",
    "val_loss_curves": "pretraining_val_loss_curves.png",
    "best_val_acc_vs_epochs": "best_val_acc_vs_epochs.png",
    "stream_accuracy_vs_pretraining_epochs": "stream_accuracy_vs_pretraining_epochs.png",
}

# Save full experiment data with required filename
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)

# -----------------------------
# Final summary
# -----------------------------
print("\nFinal tuning summary:")
for r in summary_rows:
    msg = (
        f"epochs={r['epochs']:>3d} | best_val_acc={r['best_val_acc']:.4f} "
        f"| best_val_loss={r['best_val_loss']:.4f}"
    )
    for method in methods:
        msg += f" | {method}={r[f'{method}_mean_acc']:.4f}"
    print(msg)

print(
    f"\nSelected best pretraining epochs: {best_epochs} "
    f"(val_acc={best_row['best_val_acc']:.4f}, val_loss={best_row['best_val_loss']:.4f})"
)
for method in methods:
    print(
        f"Best-trial {method} stream-avg online acc: {best_row[f'{method}_mean_acc']:.4f}"
    )
