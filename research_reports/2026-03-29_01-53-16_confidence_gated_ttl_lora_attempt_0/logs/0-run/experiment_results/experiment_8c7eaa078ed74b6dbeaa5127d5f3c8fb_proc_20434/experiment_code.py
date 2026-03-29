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
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.manual_seed(42)
np.random.seed(42)

experiment_data = {
    "synthetic_reasoning_stream": {
        "metrics": {"train": [], "val": [], "test": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
        "stream_results": {},
    }
}


# -----------------------
# Synthetic dataset
# -----------------------
class SyntheticReasoningDataset(Dataset):
    def __init__(self, n=2000, d=20, shift=False):
        self.X, self.y = self.make_data(n, d, shift)

    @staticmethod
    def make_data(n, d, shift=False):
        X = np.random.randn(n, d).astype(np.float32)
        if not shift:
            s0 = X[:, :5].sum(1)
            s1 = X[:, 5:10].sum(1)
            y = (s1 > s0).astype(np.int64)
        else:
            X = (1.3 * X + 0.4).astype(np.float32)
            s0 = 0.6 * X[:, :5].sum(1) + 0.4 * X[:, 10:15].sum(1)
            s1 = 0.7 * X[:, 5:10].sum(1) + 0.3 * X[:, 15:20].sum(1)
            y = (s1 > s0 + 0.2).astype(np.int64)
        # Normalize inputs carefully for model stability
        mu = X.mean(axis=0, keepdims=True)
        std = X.std(axis=0, keepdims=True) + 1e-6
        X = ((X - mu) / std).astype(np.float32)
        return torch.tensor(X), torch.tensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {"x": self.X[idx], "y": self.y[idx]}


train_ds = SyntheticReasoningDataset(n=2500, d=20, shift=False)
val_ds = SyntheticReasoningDataset(n=500, d=20, shift=False)
test_shift_ds = SyntheticReasoningDataset(n=800, d=20, shift=True)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)


# -----------------------
# Model with LoRA layer
# -----------------------
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=4, alpha=1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.rank = rank
        self.alpha = alpha
        self.A = nn.Parameter(torch.zeros(rank, in_features))
        self.B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.normal_(self.A, std=0.02)
        nn.init.zeros_(self.B)

    def forward(self, x):
        delta = (self.B @ self.A) * (self.alpha / self.rank)
        return F.linear(x, self.weight + delta, self.bias)

    def lora_parameters(self):
        return [self.A, self.B]

    def reset_lora(self):
        nn.init.normal_(self.A, std=0.02)
        nn.init.zeros_(self.B)


class SimpleNet(nn.Module):
    def __init__(self, d=20, h=32, num_classes=2, rank=4):
        super().__init__()
        self.fc1 = LoRALinear(d, h, rank=rank, alpha=1.0)
        self.fc2 = nn.Linear(h, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.tanh(x)
        return self.fc2(x)

    def freeze_backbone(self):
        for p in self.parameters():
            p.requires_grad = False
        for p in self.fc1.lora_parameters():
            p.requires_grad = True

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True

    def reset_lora(self):
        self.fc1.reset_lora()


model = SimpleNet().to(device)
criterion = nn.CrossEntropyLoss()
model.unfreeze_all()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# -----------------------
# Train backbone on source
# -----------------------
epochs = 12
best_state = None
best_val = float("inf")

for epoch in range(1, epochs + 1):
    model.train()
    train_loss, train_correct, train_total = 0.0, 0, 0
    for batch in train_loader:
        batch = {
            k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
        }
        x, y = batch["x"], batch["y"]
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * y.size(0)
        preds = logits.argmax(dim=1)
        train_correct += (preds == y).sum().item()
        train_total += y.size(0)

    train_loss /= train_total
    train_acc = train_correct / train_total

    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for batch in val_loader:
            batch = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            x, y = batch["x"], batch["y"]
            logits = model(x)
            loss = criterion(logits, y)
            val_loss += loss.item() * y.size(0)
            preds = logits.argmax(dim=1)
            val_correct += (preds == y).sum().item()
            val_total += y.size(0)
    val_loss /= val_total
    val_acc = val_correct / val_total

    experiment_data["synthetic_reasoning_stream"]["metrics"]["train"].append(
        (epoch, train_acc)
    )
    experiment_data["synthetic_reasoning_stream"]["metrics"]["val"].append(
        (epoch, val_acc)
    )
    experiment_data["synthetic_reasoning_stream"]["losses"]["train"].append(
        (epoch, train_loss)
    )
    experiment_data["synthetic_reasoning_stream"]["losses"]["val"].append(
        (epoch, val_loss)
    )
    experiment_data["synthetic_reasoning_stream"]["timestamps"].append(time.time())

    print(f"Epoch {epoch}: validation_loss = {val_loss:.4f}")
    if val_loss < best_val:
        best_val = val_loss
        best_state = copy.deepcopy(model.state_dict())

model.load_state_dict(best_state)


# -----------------------
# Stream evaluation
# -----------------------
def entropy_from_logits(logits):
    p = F.softmax(logits, dim=-1)
    return -(p * torch.log(p + 1e-8)).sum(dim=-1)


def stream_evaluate(
    base_model, dataset, mode="frozen", entropy_thresh=0.62, reset_every=100, lr=5e-2
):
    stream_model = copy.deepcopy(base_model).to(device)
    stream_model.freeze_backbone()
    stream_model.eval()
    opt = torch.optim.SGD(stream_model.fc1.lora_parameters(), lr=lr)

    preds_all, gt_all, ent_all, triggered = [], [], [], []
    cumulative_acc = []
    correct = 0

    for i in range(len(dataset)):
        batch = dataset[i]
        x = batch["x"].unsqueeze(0).to(device)
        y = batch["y"].unsqueeze(0).to(device)

        stream_model.eval()
        with torch.no_grad():
            logits = stream_model(x)
            ent = entropy_from_logits(logits).item()
            pred = logits.argmax(dim=1)

        preds_all.append(pred.item())
        gt_all.append(y.item())
        ent_all.append(ent)
        correct += int(pred.item() == y.item())
        cumulative_acc.append(correct / (i + 1))

        do_update = False
        if mode == "always":
            do_update = True
        elif mode == "gated":
            do_update = ent > entropy_thresh

        triggered.append(int(do_update))

        if do_update:
            stream_model.train()
            opt.zero_grad()
            logits_u = stream_model(x)
            pseudo = logits_u.detach().argmax(dim=1)
            loss_u = criterion(logits_u, pseudo)
            loss_u.backward()
            opt.step()

        if mode in ["always", "gated"] and (i + 1) % reset_every == 0:
            stream_model.reset_lora()

    acc = np.mean(np.array(preds_all) == np.array(gt_all))
    return {
        "accuracy": float(acc),
        "preds": np.array(preds_all),
        "gt": np.array(gt_all),
        "entropy": np.array(ent_all),
        "triggered": np.array(triggered),
        "cumulative_acc": np.array(cumulative_acc),
    }


frozen_res = stream_evaluate(model, test_shift_ds, mode="frozen")
always_res = stream_evaluate(model, test_shift_ds, mode="always")
gated_res = stream_evaluate(model, test_shift_ds, mode="gated", entropy_thresh=0.62)

experiment_data["synthetic_reasoning_stream"]["stream_results"] = {
    "frozen": {
        "Shifted-Stream Accuracy": frozen_res["accuracy"],
        "trigger_rate": float(frozen_res["triggered"].mean()),
    },
    "always": {
        "Shifted-Stream Accuracy": always_res["accuracy"],
        "trigger_rate": float(always_res["triggered"].mean()),
    },
    "gated": {
        "Shifted-Stream Accuracy": gated_res["accuracy"],
        "trigger_rate": float(gated_res["triggered"].mean()),
    },
}
experiment_data["synthetic_reasoning_stream"]["metrics"]["test"].append(
    (
        0,
        {
            "frozen_acc": frozen_res["accuracy"],
            "always_acc": always_res["accuracy"],
            "gated_acc": gated_res["accuracy"],
        },
    )
)
experiment_data["synthetic_reasoning_stream"]["predictions"] = gated_res[
    "preds"
].tolist()
experiment_data["synthetic_reasoning_stream"]["ground_truth"] = gated_res["gt"].tolist()

print(
    f"Shifted-Stream Accuracy | frozen={frozen_res['accuracy']:.4f} | always={always_res['accuracy']:.4f} | gated={gated_res['accuracy']:.4f}"
)
print(
    f"Adapt trigger rate      | frozen={frozen_res['triggered'].mean():.4f} | always={always_res['triggered'].mean():.4f} | gated={gated_res['triggered'].mean():.4f}"
)

# Save arrays
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
np.save(os.path.join(working_dir, "frozen_preds.npy"), frozen_res["preds"])
np.save(os.path.join(working_dir, "always_preds.npy"), always_res["preds"])
np.save(os.path.join(working_dir, "gated_preds.npy"), gated_res["preds"])
np.save(os.path.join(working_dir, "ground_truth.npy"), gated_res["gt"])
np.save(os.path.join(working_dir, "gated_entropy.npy"), gated_res["entropy"])
np.save(os.path.join(working_dir, "gated_triggered.npy"), gated_res["triggered"])
np.save(os.path.join(working_dir, "frozen_cumacc.npy"), frozen_res["cumulative_acc"])
np.save(os.path.join(working_dir, "always_cumacc.npy"), always_res["cumulative_acc"])
np.save(os.path.join(working_dir, "gated_cumacc.npy"), gated_res["cumulative_acc"])

# Visualization
plt.figure(figsize=(8, 5))
plt.plot(frozen_res["cumulative_acc"], label=f"Frozen ({frozen_res['accuracy']:.3f})")
plt.plot(
    always_res["cumulative_acc"], label=f"Always-LoRA ({always_res['accuracy']:.3f})"
)
plt.plot(gated_res["cumulative_acc"], label=f"Gated-LoRA ({gated_res['accuracy']:.3f})")
plt.xlabel("Shifted stream step")
plt.ylabel("Cumulative accuracy")
plt.title("Synthetic Shifted-Stream Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(working_dir, "synthetic_shifted_stream_accuracy.png"))
plt.close()
