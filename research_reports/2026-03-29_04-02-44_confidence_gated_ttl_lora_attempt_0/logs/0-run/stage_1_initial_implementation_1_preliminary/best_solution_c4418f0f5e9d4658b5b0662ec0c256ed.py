import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

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
        "confidences": [],
        "timestamps": [],
    }
}


# ----------------------------
# Synthetic dataset
# ----------------------------
class SyntheticReasoningDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {"x": self.X[idx], "y": self.y[idx]}


def make_data(n, d=20, shift=False):
    x = np.random.randn(n, d).astype(np.float32)
    # latent rule with interactions; shifted test alters correlations and bias
    score = (
        1.5 * x[:, 0]
        - 1.0 * x[:, 1]
        + 0.8 * x[:, 2] * x[:, 3]
        + 0.5 * np.sin(x[:, 4])
        - 0.3 * x[:, 5] ** 2
        + 0.2 * x[:, 6] * x[:, 7]
    )
    if shift:
        x[:, :8] = 0.8 * x[:, :8] + 0.6  # feature drift
        score = (
            0.7 * x[:, 0]
            - 1.2 * x[:, 1]
            + 1.2 * x[:, 2] * x[:, 3]
            + 0.9 * np.sin(1.5 * x[:, 4])
            - 0.1 * x[:, 5] ** 2
            + 0.5
        )
    y = np.digitize(score, bins=np.quantile(score, [1 / 3, 2 / 3]))
    return x, y


# ----------------------------
# Model with LoRA-style head
# ----------------------------
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=4, alpha=1.0):
        super().__init__()
        self.base = nn.Linear(in_features, out_features)
        self.base.weight.requires_grad = False
        self.base.bias.requires_grad = False
        self.rank = rank
        self.alpha = alpha
        self.A = nn.Parameter(torch.zeros(in_features, rank))
        self.B = nn.Parameter(torch.zeros(rank, out_features))
        nn.init.normal_(self.A, std=0.02)
        nn.init.zeros_(self.B)

    def forward(self, x):
        base_out = self.base(x)
        delta = x @ self.A @ self.B * (self.alpha / self.rank)
        return base_out + delta

    def reset_lora(self):
        with torch.no_grad():
            self.A.zero_()
            self.B.zero_()


class SmallNet(nn.Module):
    def __init__(self, d=20, h=64, num_classes=3, rank=4):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(d, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU()
        )
        self.head = LoRALinear(h, num_classes, rank=rank, alpha=1.0)

    def forward(self, x):
        z = self.backbone(x)
        return self.head(z)

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def lora_parameters(self):
        return [self.head.A, self.head.B]


# ----------------------------
# Data
# ----------------------------
d = 20
X_train, y_train = make_data(1800, d=d, shift=False)
X_val, y_val = make_data(400, d=d, shift=False)

# Stream with mixed domains: first ID, then shifted, then mixed
X_s1, y_s1 = make_data(120, d=d, shift=False)
X_s2, y_s2 = make_data(160, d=d, shift=True)
X_s3a, y_s3a = make_data(60, d=d, shift=False)
X_s3b, y_s3b = make_data(60, d=d, shift=True)
X_stream = np.concatenate([X_s1, X_s2, X_s3a, X_s3b], axis=0)
y_stream = np.concatenate([y_s1, y_s2, y_s3a, y_s3b], axis=0)

train_loader = DataLoader(
    SyntheticReasoningDataset(X_train, y_train), batch_size=64, shuffle=True
)
val_loader = DataLoader(
    SyntheticReasoningDataset(X_val, y_val), batch_size=128, shuffle=False
)

# ----------------------------
# Train source model
# ----------------------------
model = SmallNet(d=d).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

epochs = 12
for epoch in range(1, epochs + 1):
    model.train()
    train_losses, train_correct, train_total = [], 0, 0
    for batch in train_loader:
        batch = {
            k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
        }
        x = (batch["x"] - batch["x"].mean(dim=0, keepdim=True)) / (
            batch["x"].std(dim=0, keepdim=True) + 1e-6
        )
        y = batch["y"]
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        pred = logits.argmax(dim=-1)
        train_correct += (pred == y).sum().item()
        train_total += y.size(0)

    model.eval()
    val_losses, val_correct, val_total = [], 0, 0
    with torch.no_grad():
        for batch in val_loader:
            batch = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            x = (batch["x"] - batch["x"].mean(dim=0, keepdim=True)) / (
                batch["x"].std(dim=0, keepdim=True) + 1e-6
            )
            y = batch["y"]
            logits = model(x)
            loss = criterion(logits, y)
            val_losses.append(loss.item())
            pred = logits.argmax(dim=-1)
            val_correct += (pred == y).sum().item()
            val_total += y.size(0)

    train_loss = float(np.mean(train_losses))
    val_loss = float(np.mean(val_losses))
    train_acc = train_correct / train_total
    val_acc = val_correct / val_total
    print(f"Epoch {epoch}: validation_loss = {val_loss:.4f}")
    experiment_data["synthetic_shift_stream"]["metrics"]["train"].append(
        {"epoch": epoch, "acc": train_acc}
    )
    experiment_data["synthetic_shift_stream"]["metrics"]["val"].append(
        {"epoch": epoch, "acc": val_acc}
    )
    experiment_data["synthetic_shift_stream"]["losses"]["train"].append(
        {"epoch": epoch, "loss": train_loss}
    )
    experiment_data["synthetic_shift_stream"]["losses"]["val"].append(
        {"epoch": epoch, "loss": val_loss}
    )
    experiment_data["synthetic_shift_stream"]["timestamps"].append(time.time())

# Freeze backbone for test-time LoRA
model.freeze_backbone()
base_state = copy.deepcopy(model.state_dict())


# ----------------------------
# Stream evaluation helpers
# ----------------------------
def entropy_from_logits(logits):
    p = F.softmax(logits, dim=-1)
    return -(p * torch.log(p + 1e-8)).sum(dim=-1)


def eval_stream(mode="none", entropy_threshold=0.85, adapt_lr=5e-2):
    m = SmallNet(d=d).to(device)
    m.load_state_dict(base_state)
    m.freeze_backbone()
    opt = torch.optim.SGD(m.lora_parameters(), lr=adapt_lr)

    preds, confs, corrects = [], [], []
    per_ex_losses = []
    m.eval()
    for i in range(len(y_stream)):
        x = torch.tensor(X_stream[i : i + 1], dtype=torch.float32).to(device)
        y = torch.tensor(y_stream[i : i + 1], dtype=torch.long).to(device)
        x = (x - x.mean(dim=0, keepdim=True)) / (x.std(dim=0, keepdim=True) + 1e-6)

        with torch.no_grad():
            logits = m(x)
            probs = F.softmax(logits, dim=-1)
            ent = entropy_from_logits(logits).item()
            pred = logits.argmax(dim=-1)
            conf = probs.max(dim=-1).values.item()
            loss = criterion(logits, y).item()

        do_adapt = (mode == "always") or (mode == "gated" and ent > entropy_threshold)
        if do_adapt:
            m.train()
            opt.zero_grad()
            logits_adapt = m(x)
            p = F.softmax(logits_adapt, dim=-1)
            adapt_loss = (
                -(p * torch.log(p + 1e-8)).sum(dim=-1).mean()
            )  # entropy minimization
            adapt_loss.backward()
            opt.step()
            m.eval()
            with torch.no_grad():
                logits = m(x)
                probs = F.softmax(logits, dim=-1)
                pred = logits.argmax(dim=-1)
                conf = probs.max(dim=-1).values.item()
                loss = criterion(logits, y).item()

        preds.append(int(pred.item()))
        confs.append(float(conf))
        corrects.append(int(pred.item() == y.item()))
        per_ex_losses.append(float(loss))

    return {
        "accuracy": float(np.mean(corrects)),
        "loss": float(np.mean(per_ex_losses)),
        "preds": np.array(preds),
        "confs": np.array(confs),
        "corrects": np.array(corrects),
    }


# Tune threshold on validation uncertainty scale (simple heuristic)
model.eval()
val_ents = []
with torch.no_grad():
    for batch in val_loader:
        batch = {
            k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
        }
        x = (batch["x"] - batch["x"].mean(dim=0, keepdim=True)) / (
            batch["x"].std(dim=0, keepdim=True) + 1e-6
        )
        logits = model(x)
        val_ents.extend(entropy_from_logits(logits).detach().cpu().numpy().tolist())
entropy_threshold = float(np.quantile(val_ents, 0.75))

res_none = eval_stream("none", entropy_threshold=entropy_threshold)
res_always = eval_stream("always", entropy_threshold=entropy_threshold)
res_gated = eval_stream("gated", entropy_threshold=entropy_threshold)

print(
    f"Shifted-Stream Accuracy | no_adapt={res_none['accuracy']:.4f} always_on={res_always['accuracy']:.4f} gated={res_gated['accuracy']:.4f}"
)

experiment_data["synthetic_shift_stream"]["metrics"]["stream"].append(
    {
        "epoch": epochs,
        "Shifted-Stream Accuracy/no_adapt": res_none["accuracy"],
        "Shifted-Stream Accuracy/always_on": res_always["accuracy"],
        "Shifted-Stream Accuracy/gated": res_gated["accuracy"],
        "stream_loss/no_adapt": res_none["loss"],
        "stream_loss/always_on": res_always["loss"],
        "stream_loss/gated": res_gated["loss"],
        "entropy_threshold": entropy_threshold,
    }
)
experiment_data["synthetic_shift_stream"]["predictions"] = {
    "none": res_none["preds"],
    "always": res_always["preds"],
    "gated": res_gated["preds"],
}
experiment_data["synthetic_shift_stream"]["ground_truth"] = y_stream
experiment_data["synthetic_shift_stream"]["confidences"] = {
    "none": res_none["confs"],
    "always": res_always["confs"],
    "gated": res_gated["confs"],
}

# Save arrays
np.save(os.path.join(working_dir, "stream_ground_truth.npy"), y_stream)
np.save(os.path.join(working_dir, "stream_preds_none.npy"), res_none["preds"])
np.save(os.path.join(working_dir, "stream_preds_always.npy"), res_always["preds"])
np.save(os.path.join(working_dir, "stream_preds_gated.npy"), res_gated["preds"])
np.save(os.path.join(working_dir, "stream_confs_none.npy"), res_none["confs"])
np.save(os.path.join(working_dir, "stream_confs_always.npy"), res_always["confs"])
np.save(os.path.join(working_dir, "stream_confs_gated.npy"), res_gated["confs"])
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)

# Visualization
idx = np.arange(len(y_stream))
plt.figure(figsize=(12, 5))
plt.plot(idx, res_none["corrects"], label="No Adapt Correct", alpha=0.8)
plt.plot(idx, res_always["corrects"], label="Always-on Correct", alpha=0.8)
plt.plot(idx, res_gated["corrects"], label="Gated Correct", alpha=0.8)
plt.xlabel("Stream Index")
plt.ylabel("Correct (0/1)")
plt.title("Correctness over Shifted Evaluation Stream")
plt.legend()
plt.tight_layout()
plt.savefig(
    os.path.join(working_dir, "synthetic_shift_stream_correctness.png"), dpi=150
)
plt.close()

plt.figure(figsize=(12, 5))
plt.plot(idx, res_none["confs"], label="No Adapt Confidence", alpha=0.8)
plt.plot(idx, res_always["confs"], label="Always-on Confidence", alpha=0.8)
plt.plot(idx, res_gated["confs"], label="Gated Confidence", alpha=0.8)
plt.axhline(
    np.exp(-entropy_threshold),
    color="k",
    linestyle="--",
    label="Entropy-derived marker",
)
plt.xlabel("Stream Index")
plt.ylabel("Max Probability Confidence")
plt.title("Confidence over Shifted Evaluation Stream")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(working_dir, "synthetic_shift_stream_confidence.png"), dpi=150)
plt.close()
