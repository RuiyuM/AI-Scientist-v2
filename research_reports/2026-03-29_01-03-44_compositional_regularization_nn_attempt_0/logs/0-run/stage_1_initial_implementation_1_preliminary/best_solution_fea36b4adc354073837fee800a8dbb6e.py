import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

experiment_data = {
    "synthetic_compositional": {
        "metrics": {"train": [], "val": [], "test": []},
        "losses": {"train": [], "val": [], "test": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
}

colors = ["red", "blue", "green", "yellow"]
shapes = ["circle", "square", "triangle", "star"]
n_colors, n_shapes = len(colors), len(shapes)
n_classes = n_colors * n_shapes
input_dim = n_colors + n_shapes

held_out = {(0, 3), (1, 2), (2, 1), (3, 0)}  # unseen compositions at test
seen_pairs = [
    (c, s) for c in range(n_colors) for s in range(n_shapes) if (c, s) not in held_out
]
test_pairs = list(held_out)


def make_feature(c, s, noise=0.05):
    x = np.zeros(input_dim, dtype=np.float32)
    x[c] = 1.0
    x[n_colors + s] = 1.0
    x += np.random.normal(0, noise, size=input_dim).astype(np.float32)
    x = np.clip(x, 0.0, 1.0)
    return x


def make_dataset(pairs, n_per_pair):
    xs, ys, cs, ss = [], [], [], []
    for c, s in pairs:
        for _ in range(n_per_pair):
            xs.append(make_feature(c, s))
            ys.append(c * n_shapes + s)
            cs.append(c)
            ss.append(s)
    xs = np.stack(xs)
    xs = xs / (xs.sum(axis=1, keepdims=True) + 1e-8)  # normalize model input
    return xs.astype(np.float32), np.array(ys), np.array(cs), np.array(ss)


train_x, train_y, train_c, train_s = make_dataset(seen_pairs, 120)
val_x, val_y, val_c, val_s = make_dataset(seen_pairs, 30)
test_x, test_y, test_c, test_s = make_dataset(test_pairs, 80)


class CompDataset(Dataset):
    def __init__(self, x, y, c, s):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.c = torch.tensor(c, dtype=torch.long)
        self.s = torch.tensor(s, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {
            "x": self.x[idx],
            "y": self.y[idx],
            "color": self.c[idx],
            "shape": self.s[idx],
        }


train_loader = DataLoader(
    CompDataset(train_x, train_y, train_c, train_s), batch_size=64, shuffle=True
)
val_loader = DataLoader(
    CompDataset(val_x, val_y, val_c, val_s), batch_size=256, shuffle=False
)
test_loader = DataLoader(
    CompDataset(test_x, test_y, test_c, test_s), batch_size=256, shuffle=False
)


class CompositionalNet(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, bottleneck_dim, n_classes, n_colors, n_shapes
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.classifier = nn.Linear(bottleneck_dim, n_classes)
        self.color_embed = nn.Embedding(n_colors, bottleneck_dim)
        self.shape_embed = nn.Embedding(n_shapes, bottleneck_dim)

    def forward(self, x):
        z = self.encoder(x)
        logits = self.classifier(F.relu(z))
        return logits, z


model = CompositionalNet(
    input_dim,
    hidden_dim=64,
    bottleneck_dim=16,
    n_classes=n_classes,
    n_colors=n_colors,
    n_shapes=n_shapes,
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
ce_loss = nn.CrossEntropyLoss()


def evaluate(loader):
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    preds_all, gts_all = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            logits, z = model(batch["x"])
            target_z = model.color_embed(batch["color"]) + model.shape_embed(
                batch["shape"]
            )
            loss = ce_loss(logits, batch["y"]) + 0.1 * F.mse_loss(z, target_z)
            preds = logits.argmax(dim=1)
            total_loss += loss.item() * batch["y"].size(0)
            total_correct += (preds == batch["y"]).sum().item()
            total += batch["y"].size(0)
            preds_all.append(preds.cpu().numpy())
            gts_all.append(batch["y"].cpu().numpy())
    return (
        total_loss / total,
        total_correct / total,
        np.concatenate(preds_all),
        np.concatenate(gts_all),
    )


epochs = 40
for epoch in range(1, epochs + 1):
    model.train()
    running_loss, running_correct, running_total = 0.0, 0, 0
    for batch in train_loader:
        batch = {
            k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
        }
        optimizer.zero_grad()
        logits, z = model(batch["x"])
        target_z = model.color_embed(batch["color"]) + model.shape_embed(batch["shape"])
        cls = ce_loss(logits, batch["y"])
        comp_reg = F.mse_loss(z, target_z)
        loss = cls + 0.1 * comp_reg
        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=1)
        running_loss += loss.item() * batch["y"].size(0)
        running_correct += (preds == batch["y"]).sum().item()
        running_total += batch["y"].size(0)

    train_loss = running_loss / running_total
    train_acc = running_correct / running_total
    val_loss, val_acc, _, _ = evaluate(val_loader)
    test_loss, test_acc, test_preds, test_gts = evaluate(test_loader)

    experiment_data["synthetic_compositional"]["epochs"].append(epoch)
    experiment_data["synthetic_compositional"]["losses"]["train"].append(
        (epoch, train_loss)
    )
    experiment_data["synthetic_compositional"]["losses"]["val"].append(
        (epoch, val_loss)
    )
    experiment_data["synthetic_compositional"]["losses"]["test"].append(
        (epoch, test_loss)
    )
    experiment_data["synthetic_compositional"]["metrics"]["train"].append(
        (epoch, train_acc)
    )
    experiment_data["synthetic_compositional"]["metrics"]["val"].append(
        (epoch, val_acc)
    )
    experiment_data["synthetic_compositional"]["metrics"]["test"].append(
        (epoch, test_acc)
    )

    print(f"Epoch {epoch}: validation_loss = {val_loss:.4f}")
    print(
        f"Epoch {epoch}: train_acc = {train_acc:.4f}, val_acc = {val_acc:.4f}, compositional_generalization_accuracy = {test_acc:.4f}"
    )

experiment_data["synthetic_compositional"]["predictions"] = test_preds
experiment_data["synthetic_compositional"]["ground_truth"] = test_gts

np.save(os.path.join(working_dir, "synthetic_test_predictions.npy"), test_preds)
np.save(os.path.join(working_dir, "synthetic_test_ground_truth.npy"), test_gts)
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)

epochs_arr = np.array(experiment_data["synthetic_compositional"]["epochs"])
train_accs = np.array(
    [x[1] for x in experiment_data["synthetic_compositional"]["metrics"]["train"]]
)
val_accs = np.array(
    [x[1] for x in experiment_data["synthetic_compositional"]["metrics"]["val"]]
)
test_accs = np.array(
    [x[1] for x in experiment_data["synthetic_compositional"]["metrics"]["test"]]
)
train_losses = np.array(
    [x[1] for x in experiment_data["synthetic_compositional"]["losses"]["train"]]
)
val_losses = np.array(
    [x[1] for x in experiment_data["synthetic_compositional"]["losses"]["val"]]
)
test_losses = np.array(
    [x[1] for x in experiment_data["synthetic_compositional"]["losses"]["test"]]
)

np.save(os.path.join(working_dir, "synthetic_epochs.npy"), epochs_arr)
np.save(os.path.join(working_dir, "synthetic_train_acc.npy"), train_accs)
np.save(os.path.join(working_dir, "synthetic_val_acc.npy"), val_accs)
np.save(os.path.join(working_dir, "synthetic_test_acc.npy"), test_accs)
np.save(os.path.join(working_dir, "synthetic_train_loss.npy"), train_losses)
np.save(os.path.join(working_dir, "synthetic_val_loss.npy"), val_losses)
np.save(os.path.join(working_dir, "synthetic_test_loss.npy"), test_losses)

plt.figure(figsize=(7, 4))
plt.plot(epochs_arr, train_accs, label="Train Acc")
plt.plot(epochs_arr, val_accs, label="Val Acc")
plt.plot(epochs_arr, test_accs, label="Test Compositional Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(working_dir, "synthetic_compositional_accuracy.png"))
plt.close()

plt.figure(figsize=(7, 4))
plt.plot(epochs_arr, train_losses, label="Train Loss")
plt.plot(epochs_arr, val_losses, label="Val Loss")
plt.plot(epochs_arr, test_losses, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(working_dir, "synthetic_compositional_loss.png"))
plt.close()

print(f"Final Compositional Generalization Accuracy: {test_accs[-1]:.4f}")
