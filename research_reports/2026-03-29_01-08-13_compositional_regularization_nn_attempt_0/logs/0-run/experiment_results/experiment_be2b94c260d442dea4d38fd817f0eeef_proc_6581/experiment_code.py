# Set random seed
import random
import numpy as np
import torch

seed = 2
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

experiment_data = {
    "synthetic_compositional_baseline": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    },
    "synthetic_compositional_regularized": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    },
}

# Synthetic primitives with two latent binary attributes; target on pair is XOR over attrs.
N_PRIMS = 12
attrs = np.random.randint(0, 2, size=(N_PRIMS, 2))
labels_map = np.zeros((N_PRIMS, N_PRIMS), dtype=np.int64)
for i in range(N_PRIMS):
    for j in range(N_PRIMS):
        labels_map[i, j] = (attrs[i, 0] ^ attrs[j, 1]).astype(np.int64)

all_pairs = [(i, j) for i in range(N_PRIMS) for j in range(N_PRIMS)]
random.shuffle(all_pairs)
train_pairs = all_pairs[: int(0.7 * len(all_pairs))]
val_pairs = all_pairs[int(0.7 * len(all_pairs)) :]


class PairDataset(Dataset):
    def __init__(self, pairs, labels_map):
        self.pairs = pairs
        self.labels_map = labels_map

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        i, j = self.pairs[idx]
        return {
            "a": torch.tensor(i, dtype=torch.long),
            "b": torch.tensor(j, dtype=torch.long),
            "y": torch.tensor(self.labels_map[i, j], dtype=torch.long),
        }


train_ds = PairDataset(train_pairs, labels_map)
val_ds = PairDataset(val_pairs, labels_map)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)


class CompositionalNet(nn.Module):
    def __init__(self, n_prims, emb_dim=16, hid_dim=32):
        super().__init__()
        self.emb = nn.Embedding(n_prims + 1, emb_dim)  # extra slot for null token
        self.null_idx = n_prims
        self.encoder = nn.Sequential(
            nn.Linear(emb_dim * 2, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(hid_dim, 2)

    def encode_pair(self, a, b):
        ea = self.emb(a)
        eb = self.emb(b)
        x = torch.cat([ea, eb], dim=-1)
        return self.encoder(x)

    def forward(self, a, b):
        h = self.encode_pair(a, b)
        return self.classifier(h), h


def accuracy_from_logits(logits, y):
    return (logits.argmax(dim=-1) == y).float().mean().item()


def evaluate(model, loader):
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    all_preds, all_y = [], []
    ce = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            logits, _ = model(batch["a"], batch["b"])
            loss = ce(logits, batch["y"])
            bs = batch["y"].size(0)
            total_loss += loss.item() * bs
            total_acc += (logits.argmax(dim=-1) == batch["y"]).float().sum().item()
            n += bs
            all_preds.append(logits.argmax(dim=-1).cpu().numpy())
            all_y.append(batch["y"].cpu().numpy())
    return (
        total_loss / n,
        total_acc / n,
        np.concatenate(all_preds),
        np.concatenate(all_y),
    )


def train_model(name, reg_lambda=0.0, epochs=40):
    model = CompositionalNet(N_PRIMS).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    ce = nn.CrossEntropyLoss()
    train_acc_hist, val_acc_hist, train_loss_hist, val_loss_hist = [], [], [], []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_acc, n = 0.0, 0.0, 0
        for batch in train_loader:
            batch = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            opt.zero_grad()
            logits, h_ab = model(batch["a"], batch["b"])
            loss = ce(logits, batch["y"])

            if reg_lambda > 0:
                null_tok = torch.full_like(batch["a"], fill_value=model.null_idx).to(
                    device
                )
                h_a0 = model.encode_pair(batch["a"], null_tok)
                h_0b = model.encode_pair(null_tok, batch["b"])
                comp_reg = ((h_ab - (h_a0 + h_0b)) ** 2).mean()
                loss = loss + reg_lambda * comp_reg

            loss.backward()
            opt.step()

            bs = batch["y"].size(0)
            total_loss += loss.item() * bs
            total_acc += (logits.argmax(dim=-1) == batch["y"]).float().sum().item()
            n += bs

        train_loss, train_acc = total_loss / n, total_acc / n
        val_loss, val_acc, val_preds, val_y = evaluate(model, val_loader)

        experiment_data[name]["metrics"]["train"].append((epoch, train_acc))
        experiment_data[name]["metrics"]["val"].append((epoch, val_acc))
        experiment_data[name]["losses"]["train"].append((epoch, train_loss))
        experiment_data[name]["losses"]["val"].append((epoch, val_loss))
        train_acc_hist.append(train_acc)
        val_acc_hist.append(val_acc)
        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_loss)

        print(f"Epoch {epoch}: validation_loss = {val_loss:.4f}")
        print(f"{name} | epoch={epoch} train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

    experiment_data[name]["predictions"] = val_preds
    experiment_data[name]["ground_truth"] = val_y
    return (
        model,
        np.array(train_acc_hist),
        np.array(val_acc_hist),
        np.array(train_loss_hist),
        np.array(val_loss_hist),
    )


baseline_model, btr, bva, btrl, bval = train_model(
    "synthetic_compositional_baseline", reg_lambda=0.0
)
reg_model, rtr, rva, rtrl, rval = train_model(
    "synthetic_compositional_regularized", reg_lambda=0.1
)

print(f"Baseline Compositional Generalization Accuracy: {bva[-1]:.4f}")
print(f"Regularized Compositional Generalization Accuracy: {rva[-1]:.4f}")

epochs = np.arange(1, len(btr) + 1)
plt.figure(figsize=(8, 5))
plt.plot(epochs, btr, label="Baseline Train Acc")
plt.plot(epochs, bva, label="Baseline Val Acc")
plt.plot(epochs, rtr, label="Regularized Train Acc")
plt.plot(epochs, rva, label="Regularized Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Compositional Generalization Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(working_dir, "synthetic_compositional_accuracy.png"))
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(epochs, btrl, label="Baseline Train Loss")
plt.plot(epochs, bval, label="Baseline Val Loss")
plt.plot(epochs, rtrl, label="Regularized Train Loss")
plt.plot(epochs, rval, label="Regularized Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training/Validation Loss")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(working_dir, "synthetic_compositional_loss.png"))
plt.close()

np.save(
    os.path.join(working_dir, "baseline_val_predictions.npy"),
    experiment_data["synthetic_compositional_baseline"]["predictions"],
)
np.save(
    os.path.join(working_dir, "baseline_val_ground_truth.npy"),
    experiment_data["synthetic_compositional_baseline"]["ground_truth"],
)
np.save(
    os.path.join(working_dir, "regularized_val_predictions.npy"),
    experiment_data["synthetic_compositional_regularized"]["predictions"],
)
np.save(
    os.path.join(working_dir, "regularized_val_ground_truth.npy"),
    experiment_data["synthetic_compositional_regularized"]["ground_truth"],
)
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
