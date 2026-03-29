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

torch.manual_seed(0)
np.random.seed(0)

experiment_data = {
    "synthetic_multimodal": {
        "metrics": {"train": [], "val": [], "stream": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
        "stream_details": {},
    }
}


# ---------- Synthetic data ----------
def render_image(label, noise=0.15, strength=1.0):
    img = np.random.randn(1, 8, 8).astype(np.float32) * noise
    if label == 0:  # vertical bar
        img[0, :, 3:5] += strength
    else:  # horizontal bar
        img[0, 3:5, :] += strength
    img = np.clip(img, -2, 2)
    img = (img - img.mean()) / (img.std() + 1e-6)  # normalize input
    return img.astype(np.float32)


class SyntheticMMDataset(Dataset):
    def __init__(self, n, split="source"):
        self.samples = []
        for _ in range(n):
            y = np.random.randint(0, 2)
            if split == "source":
                q_corr = 0.9
                img_noise, img_strength = 0.18, 0.9
            elif split == "val":
                q_corr = 0.9
                img_noise, img_strength = 0.20, 0.85
            else:  # target shift
                q_corr = 0.1  # shortcut flips
                img_noise, img_strength = 0.45, 0.45  # image degraded
            q_token = y if np.random.rand() < q_corr else 1 - y
            img = render_image(y, noise=img_noise, strength=img_strength)
            caption = y
            rationale = y
            self.samples.append((img, q_token, y, caption, rationale))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, q, y, c, r = self.samples[idx]
        return {
            "image": torch.tensor(img, dtype=torch.float32),
            "question": torch.tensor(q, dtype=torch.long),
            "label": torch.tensor(y, dtype=torch.long),
            "caption": torch.tensor(c, dtype=torch.long),
            "rationale": torch.tensor(r, dtype=torch.long),
        }


train_ds = SyntheticMMDataset(3000, "source")
val_ds = SyntheticMMDataset(600, "val")
target_ds = SyntheticMMDataset(1200, "target")

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)


# ---------- Model ----------
class Adapter(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))

    def forward(self, x):
        return x + 0.3 * self.net(x)


class MMNet(nn.Module):
    def __init__(self, d=32):
        super().__init__()
        self.img_enc = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 8, 3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8 * 8 * 8, d),
            nn.ReLU(),
        )
        self.q_emb = nn.Embedding(2, d)
        self.fuse = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU())
        self.adapter = Adapter(d)
        self.ans_head = nn.Linear(d, 2)
        self.cap_head = nn.Linear(d, 2)
        self.rat_head = nn.Linear(d, 2)

    def forward(self, image, question):
        z_img = self.img_enc(image)
        z_q = self.q_emb(question)
        z = self.fuse(torch.cat([z_img, z_q], dim=1))
        z = self.adapter(z)
        return {
            "answer": self.ans_head(z),
            "caption": self.cap_head(z),
            "rationale": self.rat_head(z),
        }


model = MMNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# ---------- Train ----------
def evaluate(loader, model):
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            out = model(batch["image"], batch["question"])
            loss = (
                F.cross_entropy(out["answer"], batch["label"])
                + 0.5 * F.cross_entropy(out["caption"], batch["caption"])
                + 0.5 * F.cross_entropy(out["rationale"], batch["rationale"])
            )
            pred = out["answer"].argmax(1)
            total_loss += loss.item() * batch["label"].size(0)
            total_acc += (pred == batch["label"]).float().sum().item()
            n += batch["label"].size(0)
    return total_loss / n, total_acc / n


epochs = 8
for epoch in range(1, epochs + 1):
    model.train()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for batch in train_loader:
        batch = {
            k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
        }
        optimizer.zero_grad()
        out = model(batch["image"], batch["question"])
        loss = (
            F.cross_entropy(out["answer"], batch["label"])
            + 0.5 * F.cross_entropy(out["caption"], batch["caption"])
            + 0.5 * F.cross_entropy(out["rationale"], batch["rationale"])
        )
        loss.backward()
        optimizer.step()
        pred = out["answer"].argmax(1)
        total_loss += loss.item() * batch["label"].size(0)
        total_acc += (pred == batch["label"]).float().sum().item()
        n += batch["label"].size(0)
    train_loss, train_acc = total_loss / n, total_acc / n
    val_loss, val_acc = evaluate(val_loader, model)
    print(f"Epoch {epoch}: validation_loss = {val_loss:.4f}")
    experiment_data["synthetic_multimodal"]["losses"]["train"].append(
        (epoch, train_loss)
    )
    experiment_data["synthetic_multimodal"]["losses"]["val"].append((epoch, val_loss))
    experiment_data["synthetic_multimodal"]["metrics"]["train"].append(
        (epoch, train_acc)
    )
    experiment_data["synthetic_multimodal"]["metrics"]["val"].append((epoch, val_acc))
    experiment_data["synthetic_multimodal"]["timestamps"].append(time.time())

base_state = copy.deepcopy(model.state_dict())


# ---------- Stream adaptation ----------
def stream_eval(mode="frozen", ent_thresh=0.60, lr=5e-3, window=50):
    m = MMNet().to(device)
    m.load_state_dict(base_state)
    for p in m.parameters():
        p.requires_grad = False
    for p in m.adapter.parameters():
        p.requires_grad = True
    opt = torch.optim.SGD(m.adapter.parameters(), lr=lr)

    accs, updates, ents = [], [], []
    preds, gts = [], []
    harmful_drops = 0
    prev_window_acc = None

    m.eval()
    for i in range(len(target_ds)):
        batch = target_ds[i]
        batch = {
            k: v.unsqueeze(0).to(device)
            for k, v in batch.items()
            if isinstance(v, torch.Tensor)
        }

        with torch.no_grad():
            out = m(batch["image"], batch["question"])
            p_ans = F.softmax(out["answer"], dim=1)
            ent = -(p_ans * (p_ans + 1e-8).log()).sum(1).item()
            ans_pred = out["answer"].argmax(1).item()
            cap_pred = out["caption"].argmax(1).item()
            rat_pred = out["rationale"].argmax(1).item()
            correct = int(ans_pred == batch["label"].item())

        do_update = False
        if mode == "tent":
            do_update = ent > ent_thresh
        elif mode == "consistency":
            do_update = (ent > ent_thresh) and (ans_pred == cap_pred == rat_pred)

        if do_update:
            m.train()
            out2 = m(batch["image"], batch["question"])
            p2 = F.softmax(out2["answer"], dim=1)
            loss = -(p2 * (p2 + 1e-8).log()).sum(1).mean()
            if mode == "consistency":
                pseudo = torch.tensor([ans_pred], device=device)
                loss = (
                    loss
                    + 0.5 * F.cross_entropy(out2["caption"], pseudo)
                    + 0.5 * F.cross_entropy(out2["rationale"], pseudo)
                )
            opt.zero_grad()
            loss.backward()
            opt.step()
            m.eval()
            updates.append(1)
        else:
            updates.append(0)

        accs.append(correct)
        ents.append(ent)
        preds.append(ans_pred)
        gts.append(batch["label"].item())

        if (i + 1) % window == 0:
            wacc = np.mean(accs[-window:])
            if (
                prev_window_acc is not None
                and updates[-window:].count(1) > 0
                and wacc < prev_window_acc
            ):
                harmful_drops += prev_window_acc - wacc
            prev_window_acc = wacc

    overall_acc = float(np.mean(accs))
    windows = [
        np.mean(accs[i : i + window])
        for i in range(0, len(accs), window)
        if len(accs[i : i + window]) > 0
    ]
    stability_penalty = float(
        np.std(windows) + 0.5 * harmful_drops / max(1, len(windows))
    )
    ssaa = overall_acc - stability_penalty
    return {
        "accuracy": overall_acc,
        "ssaa": ssaa,
        "adapt_freq": float(np.mean(updates)),
        "acc_stream": np.array(accs, dtype=np.float32),
        "update_stream": np.array(updates, dtype=np.float32),
        "entropy_stream": np.array(ents, dtype=np.float32),
        "preds": np.array(preds, dtype=np.int64),
        "gts": np.array(gts, dtype=np.int64),
    }


frozen_res = stream_eval("frozen")
tent_res = stream_eval("tent")
cons_res = stream_eval("consistency")

for name, res in [
    ("frozen", frozen_res),
    ("tent", tent_res),
    ("consistency", cons_res),
]:
    print(
        f'{name}: accuracy={res["accuracy"]:.4f}, SSAA={res["ssaa"]:.4f}, adapt_freq={res["adapt_freq"]:.4f}'
    )

experiment_data["synthetic_multimodal"]["metrics"]["stream"].append(
    {
        "frozen_accuracy": frozen_res["accuracy"],
        "frozen_ssaa": frozen_res["ssaa"],
        "frozen_adapt_freq": frozen_res["adapt_freq"],
        "tent_accuracy": tent_res["accuracy"],
        "tent_ssaa": tent_res["ssaa"],
        "tent_adapt_freq": tent_res["adapt_freq"],
        "cons_accuracy": cons_res["accuracy"],
        "cons_ssaa": cons_res["ssaa"],
        "cons_adapt_freq": cons_res["adapt_freq"],
        "timestamp": time.time(),
    }
)
experiment_data["synthetic_multimodal"]["predictions"] = cons_res["preds"].tolist()
experiment_data["synthetic_multimodal"]["ground_truth"] = cons_res["gts"].tolist()
experiment_data["synthetic_multimodal"]["stream_details"] = {
    "frozen_acc_stream": frozen_res["acc_stream"],
    "tent_acc_stream": tent_res["acc_stream"],
    "cons_acc_stream": cons_res["acc_stream"],
    "tent_update_stream": tent_res["update_stream"],
    "cons_update_stream": cons_res["update_stream"],
    "tent_entropy_stream": tent_res["entropy_stream"],
    "cons_entropy_stream": cons_res["entropy_stream"],
}

# ---------- Save arrays ----------
np.save(os.path.join(working_dir, "frozen_acc_stream.npy"), frozen_res["acc_stream"])
np.save(os.path.join(working_dir, "tent_acc_stream.npy"), tent_res["acc_stream"])
np.save(os.path.join(working_dir, "cons_acc_stream.npy"), cons_res["acc_stream"])
np.save(os.path.join(working_dir, "tent_update_stream.npy"), tent_res["update_stream"])
np.save(os.path.join(working_dir, "cons_update_stream.npy"), cons_res["update_stream"])
np.save(
    os.path.join(working_dir, "tent_entropy_stream.npy"), tent_res["entropy_stream"]
)
np.save(
    os.path.join(working_dir, "cons_entropy_stream.npy"), cons_res["entropy_stream"]
)
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)


# ---------- Plots ----------
def moving_avg(x, k=25):
    x = np.asarray(x, dtype=np.float32)
    if len(x) < k:
        return x
    return np.convolve(x, np.ones(k) / k, mode="valid")


plt.figure(figsize=(8, 4))
plt.plot(moving_avg(frozen_res["acc_stream"]), label="frozen")
plt.plot(moving_avg(tent_res["acc_stream"]), label="entropy")
plt.plot(moving_avg(cons_res["acc_stream"]), label="consistency")
plt.xlabel("stream step")
plt.ylabel("moving accuracy")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(working_dir, "synthetic_stream_accuracy.png"))
plt.close()

plt.figure(figsize=(8, 4))
plt.plot(tent_res["update_stream"], label="entropy updates", alpha=0.7)
plt.plot(cons_res["update_stream"], label="consistency updates", alpha=0.7)
plt.xlabel("stream step")
plt.ylabel("update event")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(working_dir, "synthetic_adaptation_events.png"))
plt.close()
