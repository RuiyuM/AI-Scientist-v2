import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import time
import copy
import math
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
    "synthetic_multimodal_stream": {
        "metrics": {"train": [], "val": [], "stream": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
        "stream_details": {},
    }
}

NUM_CLASSES = 3
IMG_SHAPE = (3, 8, 8)
Q_DIM = 6


class SyntheticMultiModalDataset(Dataset):
    def __init__(self, n=1200, split="train", shift=False):
        self.samples = []
        for i in range(n):
            y = np.random.randint(NUM_CLASSES)
            qid = np.random.randint(0, 2)  # question type
            q = np.zeros(Q_DIM, dtype=np.float32)
            q[qid] = 1.0

            img = np.random.randn(*IMG_SHAPE).astype(np.float32) * 0.25

            # source rule: each class corresponds to one bright channel block
            img[y, 2:6, 2:6] += 1.5

            # spurious feature
            spur = np.zeros(NUM_CLASSES, dtype=np.float32)
            if not shift:
                spur[y] = 1.0
            else:
                # shifted stream: shortcut conflicts often
                wrong = (y + np.random.randint(1, NUM_CLASSES)) % NUM_CLASSES
                if np.random.rand() < 0.7:
                    spur[wrong] = 1.0
                else:
                    spur[y] = 1.0
                # make image noisier under shift
                img += np.random.randn(*IMG_SHAPE).astype(np.float32) * 0.20

            q[2:5] = spur

            # auxiliary labels with some noise: caption and rationale should mostly match y
            cap = (
                y
                if np.random.rand() > (0.1 if not shift else 0.2)
                else np.random.randint(NUM_CLASSES)
            )
            rat = (
                y
                if np.random.rand() > (0.15 if not shift else 0.25)
                else np.random.randint(NUM_CLASSES)
            )

            self.samples.append((img, q, y, cap, rat))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, q, y, cap, rat = self.samples[idx]
        return {
            "image": torch.tensor(img, dtype=torch.float32),
            "question": torch.tensor(q, dtype=torch.float32),
            "label": torch.tensor(y, dtype=torch.long),
            "caption_label": torch.tensor(cap, dtype=torch.long),
            "rationale_label": torch.tensor(rat, dtype=torch.long),
        }


train_ds = SyntheticMultiModalDataset(n=1200, split="train", shift=False)
val_ds = SyntheticMultiModalDataset(n=300, split="val", shift=False)
stream_ds = SyntheticMultiModalDataset(n=400, split="stream", shift=True)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)


class MultiModalNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod(IMG_SHAPE), 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.q_net = nn.Sequential(
            nn.Linear(Q_DIM, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU()
        )
        self.fuse = nn.Sequential(nn.Linear(48, 32), nn.ReLU())
        self.main_head = nn.Linear(32, NUM_CLASSES)
        self.caption_head = nn.Linear(32, NUM_CLASSES)
        self.rationale_head = nn.Linear(32, NUM_CLASSES)
        self.adapter = nn.Linear(32, NUM_CLASSES, bias=False)

    def forward(self, image, question):
        # normalize inputs
        image = (image - image.mean(dim=(1, 2, 3), keepdim=True)) / (
            image.std(dim=(1, 2, 3), keepdim=True) + 1e-6
        )
        question = question / (question.norm(dim=1, keepdim=True) + 1e-6)
        x1 = self.img_net(image)
        x2 = self.q_net(question)
        z = self.fuse(torch.cat([x1, x2], dim=1))
        main_logits = self.main_head(z) + self.adapter(z)
        cap_logits = self.caption_head(z)
        rat_logits = self.rationale_head(z)
        return main_logits, cap_logits, rat_logits


model = MultiModalNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


def evaluate_loader(model, loader):
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            logits, cap_logits, rat_logits = model(batch["image"], batch["question"])
            loss = (
                F.cross_entropy(logits, batch["label"])
                + 0.3 * F.cross_entropy(cap_logits, batch["caption_label"])
                + 0.3 * F.cross_entropy(rat_logits, batch["rationale_label"])
            )
            total_loss += loss.item() * batch["label"].size(0)
            total_correct += (logits.argmax(1) == batch["label"]).sum().item()
            total += batch["label"].size(0)
    return total_loss / total, total_correct / total


epochs = 8
for epoch in range(1, epochs + 1):
    model.train()
    running_loss, running_correct, total = 0.0, 0, 0
    for batch in train_loader:
        batch = {
            k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
        }
        optimizer.zero_grad()
        logits, cap_logits, rat_logits = model(batch["image"], batch["question"])
        loss = (
            F.cross_entropy(logits, batch["label"])
            + 0.3 * F.cross_entropy(cap_logits, batch["caption_label"])
            + 0.3 * F.cross_entropy(rat_logits, batch["rationale_label"])
        )
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch["label"].size(0)
        running_correct += (logits.argmax(1) == batch["label"]).sum().item()
        total += batch["label"].size(0)

    train_loss = running_loss / total
    train_acc = running_correct / total
    val_loss, val_acc = evaluate_loader(model, val_loader)

    experiment_data["synthetic_multimodal_stream"]["metrics"]["train"].append(
        {"epoch": epoch, "accuracy": train_acc}
    )
    experiment_data["synthetic_multimodal_stream"]["metrics"]["val"].append(
        {"epoch": epoch, "accuracy": val_acc}
    )
    experiment_data["synthetic_multimodal_stream"]["losses"]["train"].append(
        {"epoch": epoch, "loss": train_loss}
    )
    experiment_data["synthetic_multimodal_stream"]["losses"]["val"].append(
        {"epoch": epoch, "loss": val_loss}
    )
    print(f"Epoch {epoch}: validation_loss = {val_loss:.4f}")

base_state = copy.deepcopy(model.state_dict())


def stream_stability_adjusted_accuracy(accs, update_steps, window=25):
    accs = np.array(accs, dtype=np.float32)
    mean_acc = accs.mean() if len(accs) else 0.0
    if len(accs) < window:
        var_penalty = float(accs.std()) if len(accs) else 0.0
    else:
        wins = [
            accs[i : i + window].mean()
            for i in range(0, len(accs) - window + 1, window)
        ]
        var_penalty = float(np.std(wins))
    drop_penalty = 0.0
    for s in update_steps:
        if s + 5 < len(accs):
            pre = accs[max(0, s - 5) : s].mean() if s > 0 else accs[:1].mean()
            post = accs[s : min(len(accs), s + 5)].mean()
            drop_penalty += max(0.0, pre - post)
    drop_penalty = drop_penalty / max(1, len(update_steps))
    return float(mean_acc - 0.5 * var_penalty - 0.5 * drop_penalty)


def run_stream(method="frozen"):
    m = MultiModalNet().to(device)
    m.load_state_dict(base_state)

    # freeze all except adapter for adaptation
    for p in m.parameters():
        p.requires_grad = False
    for p in m.adapter.parameters():
        p.requires_grad = True
    adapt_optim = torch.optim.SGD(m.adapter.parameters(), lr=0.05)

    preds, gts, accs, update_steps = [], [], [], []
    entropies, confidences = [], []
    start = time.time()

    m.eval()
    for idx in range(len(stream_ds)):
        batch = stream_ds[idx]
        batch = {
            k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        with torch.no_grad():
            logits, cap_logits, rat_logits = m(batch["image"], batch["question"])
            probs = F.softmax(logits, dim=1)
            pred = logits.argmax(1)
            conf = probs.max(1).values.item()
            ent = -(probs * (probs.clamp_min(1e-8).log())).sum(1).item()
            cap_pred = cap_logits.argmax(1)
            rat_pred = rat_logits.argmax(1)

        gt = batch["label"].item()
        correct = int(pred.item() == gt)
        preds.append(pred.item())
        gts.append(gt)
        accs.append(correct)
        entropies.append(ent)
        confidences.append(conf)

        do_update = False
        if method == "entropy":
            do_update = ent > 0.7
        elif method == "consistency":
            agree = pred.item() == cap_pred.item() == rat_pred.item()
            do_update = (ent > 0.5) and (conf > 0.45) and agree

        if do_update:
            m.train()
            logits, cap_logits, rat_logits = m(batch["image"], batch["question"])
            probs = F.softmax(logits, dim=1)
            pseudo = probs.detach().argmax(1)
            loss = 0.0
            if method == "entropy":
                loss = -(probs * (probs.clamp_min(1e-8).log())).sum(1).mean()
            else:
                # consistency-gated pseudo-labeling with auxiliary agreement
                loss = (
                    F.cross_entropy(logits, pseudo)
                    + 0.5 * F.cross_entropy(cap_logits, pseudo)
                    + 0.5 * F.cross_entropy(rat_logits, pseudo)
                )
            adapt_optim.zero_grad()
            loss.backward()
            adapt_optim.step()
            update_steps.append(idx)
            m.eval()

    elapsed = time.time() - start
    mean_acc = float(np.mean(accs))
    ssaa = stream_stability_adjusted_accuracy(accs, update_steps)
    result = {
        "accuracy": mean_acc,
        "stream_stability_adjusted_accuracy": ssaa,
        "adaptation_frequency": len(update_steps) / len(stream_ds),
        "latency_seconds": elapsed,
        "predictions": np.array(preds),
        "ground_truth": np.array(gts),
        "acc_curve": np.array(accs),
        "update_steps": np.array(update_steps),
        "entropies": np.array(entropies),
        "confidences": np.array(confidences),
    }
    return result


results = {
    "frozen": run_stream("frozen"),
    "entropy": run_stream("entropy"),
    "consistency": run_stream("consistency"),
}

for name, res in results.items():
    print(
        f"{name} | accuracy={res['accuracy']:.4f} | SSAA={res['stream_stability_adjusted_accuracy']:.4f} | adapt_freq={res['adaptation_frequency']:.4f} | latency={res['latency_seconds']:.2f}s"
    )

experiment_data["synthetic_multimodal_stream"]["stream_details"] = {
    k: {
        "accuracy": v["accuracy"],
        "stream_stability_adjusted_accuracy": v["stream_stability_adjusted_accuracy"],
        "adaptation_frequency": v["adaptation_frequency"],
        "latency_seconds": v["latency_seconds"],
    }
    for k, v in results.items()
}
experiment_data["synthetic_multimodal_stream"]["predictions"] = results["consistency"][
    "predictions"
].tolist()
experiment_data["synthetic_multimodal_stream"]["ground_truth"] = results["consistency"][
    "ground_truth"
].tolist()
experiment_data["synthetic_multimodal_stream"]["timestamps"] = [
    {"time": time.time(), "epoch": e + 1} for e in range(epochs)
]
experiment_data["synthetic_multimodal_stream"]["metrics"]["stream"].append(
    {
        "frozen_ssaa": results["frozen"]["stream_stability_adjusted_accuracy"],
        "entropy_ssaa": results["entropy"]["stream_stability_adjusted_accuracy"],
        "consistency_ssaa": results["consistency"][
            "stream_stability_adjusted_accuracy"
        ],
    }
)

np.save(
    os.path.join(working_dir, "frozen_acc_curve.npy"), results["frozen"]["acc_curve"]
)
np.save(
    os.path.join(working_dir, "entropy_acc_curve.npy"), results["entropy"]["acc_curve"]
)
np.save(
    os.path.join(working_dir, "consistency_acc_curve.npy"),
    results["consistency"]["acc_curve"],
)
np.save(
    os.path.join(working_dir, "frozen_updates.npy"), results["frozen"]["update_steps"]
)
np.save(
    os.path.join(working_dir, "entropy_updates.npy"), results["entropy"]["update_steps"]
)
np.save(
    os.path.join(working_dir, "consistency_updates.npy"),
    results["consistency"]["update_steps"],
)
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)

x = np.arange(len(stream_ds))
plt.figure(figsize=(10, 5))
for name, color in [
    ("frozen", "tab:blue"),
    ("entropy", "tab:orange"),
    ("consistency", "tab:green"),
]:
    curve = results[name]["acc_curve"]
    cum = np.cumsum(curve) / (np.arange(len(curve)) + 1)
    plt.plot(x, cum, label=name, color=color)
plt.xlabel("Stream step")
plt.ylabel("Cumulative accuracy")
plt.title("Synthetic multimodal stream: online performance")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(working_dir, "synthetic_stream_cumulative_accuracy.png"))
plt.close()
