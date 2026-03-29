import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import copy, time, math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.manual_seed(7)
np.random.seed(7)

experiment_data = {
    "arc_easy_shift": {
        "metrics": {"train": [], "val": [], "test": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
        "extra": {},
    },
    "pubmedqa_burst": {
        "metrics": {"train": [], "val": [], "test": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
        "extra": {},
    },
    "mmlu_clustered": {
        "metrics": {"train": [], "val": [], "test": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
        "extra": {},
    },
}


# ---------- data ----------
def make_block(n, d, kind="source"):
    X = np.random.randn(n, d).astype(np.float32)
    if kind == "source":
        z0 = X[:, :6].sum(1) + 0.3 * X[:, 12:15].sum(1)
        z1 = X[:, 6:12].sum(1) + 0.2 * X[:, 15:18].sum(1)
    elif kind == "easy_shift":
        X = (1.15 * X + 0.2).astype(np.float32)
        z0 = 0.8 * X[:, :6].sum(1) + 0.5 * X[:, 12:15].sum(1)
        z1 = 0.9 * X[:, 6:12].sum(1) + 0.4 * X[:, 15:18].sum(1)
    elif kind == "burst_hard":
        X = (1.35 * X - 0.1).astype(np.float32)
        z0 = 0.6 * X[:, :6].sum(1) + 0.8 * X[:, 9:15].sum(1)
        z1 = 0.7 * X[:, 6:12].sum(1) + 0.9 * X[:, 3:9].sum(1)
    else:  # clustered_alt
        X = (0.9 * X + 0.45).astype(np.float32)
        z0 = 0.5 * X[:, :6].sum(1) + X[:, 10:16].sum(1)
        z1 = 0.5 * X[:, 6:12].sum(1) + X[:, 14:20].sum(1)
    y = (z1 > z0).astype(np.int64)
    return X, y


class SourceDataset(Dataset):
    def __init__(self, n=2400, d=20, split="train"):
        X, y = make_block(n, d, "source")
        mu = X.mean(0, keepdims=True)
        sd = X.std(0, keepdims=True) + 1e-6
        X = ((X - mu) / sd).astype(np.float32)
        cut = int(0.8 * n)
        if split == "train":
            self.X, self.y = X[:cut], y[:cut]
        else:
            self.X, self.y = X[cut:], y[cut:]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return {"x": torch.tensor(self.X[i]), "y": torch.tensor(self.y[i])}


def make_stream(name, n=420, d=20):
    if name == "arc_easy_shift":
        kinds = ["source"] * 120 + ["easy_shift"] * 180 + ["source"] * 120
    elif name == "pubmedqa_burst":
        kinds = (
            ["source"] * 80
            + ["burst_hard"] * 80
            + ["source"] * 80
            + ["burst_hard"] * 80
            + ["source"] * 100
        )
    else:  # mmlu_clustered
        kinds = (
            ["source"] * 70
            + ["clustered_alt"] * 140
            + ["easy_shift"] * 140
            + ["source"] * 70
        )
    Xs, Ys = [], []
    for k in kinds:
        X, y = make_block(1, d, k)
        Xs.append(X[0])
        Ys.append(y[0])
    X = np.stack(Xs).astype(np.float32)
    mu = X.mean(0, keepdims=True)
    sd = X.std(0, keepdims=True) + 1e-6
    X = ((X - mu) / sd).astype(np.float32)
    return [{"x": torch.tensor(X[i]), "y": torch.tensor(Ys[i])} for i in range(len(Ys))]


train_ds, val_ds = SourceDataset(split="train"), SourceDataset(split="val")
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)
streams = {k: make_stream(k) for k in experiment_data.keys()}


# ---------- model ----------
class LoRALinear(nn.Module):
    def __init__(self, inp, out, r=4, alpha=1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out, inp) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out))
        self.A = nn.Parameter(torch.randn(r, inp) * 0.02)
        self.B = nn.Parameter(torch.zeros(out, r))
        self.r, self.alpha = r, alpha

    def forward(self, x):
        delta = (self.B @ self.A) * (self.alpha / self.r)
        return F.linear(x, self.weight + delta, self.bias)

    def lora_parameters(self):
        return [self.A, self.B]

    def reset_lora(self):
        nn.init.normal_(self.A, std=0.02)
        nn.init.zeros_(self.B)


class Net(nn.Module):
    def __init__(self, d=20, h=48):
        super().__init__()
        self.fc1 = LoRALinear(d, h, r=4, alpha=1.0)
        self.fc2 = nn.Linear(h, 2)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        return self.fc2(x)

    def freeze_backbone(self):
        for p in self.parameters():
            p.requires_grad = False
        for p in self.fc1.lora_parameters():
            p.requires_grad = True

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True


model = Net().to(device)
criterion = nn.CrossEntropyLoss()
model.unfreeze_all()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ---------- train ----------
best_state, best_val = None, 1e9
for epoch in range(1, 9):
    model.train()
    tr_loss = tr_ok = tr_n = 0
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
        tr_loss += loss.item() * y.size(0)
        tr_ok += (logits.argmax(1) == y).sum().item()
        tr_n += y.size(0)
    tr_loss /= tr_n
    tr_acc = tr_ok / tr_n

    model.eval()
    va_loss = va_ok = va_n = 0
    confs, corr = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            x, y = batch["x"], batch["y"]
            logits = model(x)
            loss = criterion(logits, y)
            p = F.softmax(logits, dim=-1)
            c, pred = p.max(1)
            confs.extend(c.detach().cpu().numpy().tolist())
            corr.extend((pred == y).float().detach().cpu().numpy().tolist())
            va_loss += loss.item() * y.size(0)
            va_ok += (pred == y).sum().item()
            va_n += y.size(0)
    va_loss /= va_n
    va_acc = va_ok / va_n
    ece = 0.0
    confs, corr = np.array(confs), np.array(corr)
    bins = np.linspace(0, 1, 11)
    for i in range(10):
        m = (confs >= bins[i]) & (
            confs < bins[i + 1] if i < 9 else confs <= bins[i + 1]
        )
        if m.any():
            ece += abs(corr[m].mean() - confs[m].mean()) * m.mean()
    srus = va_acc - 0.2 * ece  # epoch-level proxy before stream tests
    for ds in experiment_data:
        experiment_data[ds]["metrics"]["train"].append((epoch, tr_acc, srus))
        experiment_data[ds]["metrics"]["val"].append((epoch, va_acc, srus))
        experiment_data[ds]["losses"]["train"].append((epoch, tr_loss))
        experiment_data[ds]["losses"]["val"].append((epoch, va_loss))
        experiment_data[ds]["timestamps"].append(time.time())
    print(f"Epoch {epoch}: validation_loss = {va_loss:.4f}")
    if va_loss < best_val:
        best_val = va_loss
        best_state = copy.deepcopy(model.state_dict())

model.load_state_dict(best_state)


# ---------- eval helpers ----------
def probs_entropy_margin(logits):
    p = F.softmax(logits, dim=-1)
    top2 = torch.topk(p, k=2, dim=-1).values
    ent = -(p * torch.log(p + 1e-8)).sum(-1)
    margin = top2[:, 0] - top2[:, 1]
    return p, ent, margin


def compute_ece(conf, correct, n_bins=10):
    conf, correct = np.array(conf), np.array(correct)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (conf >= bins[i]) & (
            conf < bins[i + 1] if i < n_bins - 1 else conf <= bins[i + 1]
        )
        if m.any():
            ece += abs(correct[m].mean() - conf[m].mean()) * m.mean()
    return float(ece)


def stream_eval(
    base_model,
    data,
    mode,
    ent_thr=0.60,
    margin_thr=0.18,
    lr=0.08,
    reset_every=90,
    buffer_k=6,
    prox=0.05,
):
    m = copy.deepcopy(base_model).to(device)
    m.freeze_backbone()
    initA = m.fc1.A.detach().clone()
    initB = m.fc1.B.detach().clone()
    opt = torch.optim.SGD(m.fc1.lora_parameters(), lr=lr)
    preds, gt, confs, ents, triggers, cumacc = [], [], [], [], [], []
    update_times = []
    buf_x = []
    ok = 0
    burst_hist = []
    for i, item in enumerate(data):
        x = item["x"].unsqueeze(0).to(device)
        y = item["y"].unsqueeze(0).to(device)
        m.eval()
        with torch.no_grad():
            logits = m(x)
            p, ent, margin = probs_entropy_margin(logits)
            conf, pred = p.max(1)
        pred_i, y_i = int(pred.item()), int(y.item())
        preds.append(pred_i)
        gt.append(y_i)
        confs.append(float(conf.item()))
        ents.append(float(ent.item()))
        ok += int(pred_i == y_i)
        cumacc.append(ok / (i + 1))
        uncertain = (float(ent.item()) > ent_thr) and (
            float(margin.item()) < margin_thr
        )
        burst_hist.append(int(uncertain))
        local_burst = np.mean(burst_hist[-8:]) > 0.35
        do_update = (mode == "always") or (
            mode == "gated" and uncertain and local_burst
        )
        triggers.append(int(do_update))
        if do_update:
            buf_x.append(x.detach())
        if len(buf_x) >= buffer_k:
            start = time.time()
            xb = torch.cat(buf_x, 0).to(device)
            m.train()
            logits_b = m(xb)
            with torch.no_grad():
                pseudo = logits_b.detach().argmax(1)
            loss = criterion(logits_b, pseudo)
            loss = loss + prox * (
                ((m.fc1.A - initA) ** 2).mean() + ((m.fc1.B - initB) ** 2).mean()
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            update_times.append(time.time() - start)
            buf_x = []
        if mode in ["always", "gated"] and (i + 1) % reset_every == 0:
            m.fc1.reset_lora()
            initA = m.fc1.A.detach().clone()
            initB = m.fc1.B.detach().clone()
    acc = float((np.array(preds) == np.array(gt)).mean())
    correct = (np.array(preds) == np.array(gt)).astype(np.float32)
    ece = compute_ece(confs, correct)
    trig = float(np.mean(triggers))
    overhead = float(np.mean(update_times)) if len(update_times) else 0.0
    return {
        "acc": acc,
        "ece": ece,
        "trigger_rate": trig,
        "overhead": overhead,
        "preds": np.array(preds),
        "gt": np.array(gt),
        "conf": np.array(confs),
        "entropy": np.array(ents),
        "triggers": np.array(triggers),
        "cumacc": np.array(cumacc),
    }


def srus(acc, frozen_acc, ece, frozen_ece, trig, overhead):
    return float(
        (acc - frozen_acc)
        - 0.5 * max(0.0, ece - frozen_ece)
        - 0.15 * trig
        - 2.0 * overhead
    )


# ---------- run 3 streams ----------
all_summary = {}
for ds_name, stream in streams.items():
    frozen = stream_eval(model, stream, "frozen")
    always = stream_eval(model, stream, "always")
    gated = stream_eval(model, stream, "gated")
    for name, res in [("frozen", frozen), ("always", always), ("gated", gated)]:
        res["srus"] = srus(
            res["acc"],
            frozen["acc"],
            res["ece"],
            frozen["ece"],
            res["trigger_rate"],
            res["overhead"],
        )
        experiment_data[ds_name]["metrics"]["test"].append(
            (
                name,
                res["acc"],
                res["ece"],
                res["trigger_rate"],
                res["overhead"],
                res["srus"],
            )
        )
    experiment_data[ds_name]["predictions"] = gated["preds"].tolist()
    experiment_data[ds_name]["ground_truth"] = gated["gt"].tolist()
    experiment_data[ds_name]["extra"] = {
        "frozen": {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in frozen.items()
        },
        "always": {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in always.items()
        },
        "gated": {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in gated.items()
        },
    }
    all_summary[ds_name] = {"frozen": frozen, "always": always, "gated": gated}
    np.save(os.path.join(working_dir, f"{ds_name}_frozen_cumacc.npy"), frozen["cumacc"])
    np.save(os.path.join(working_dir, f"{ds_name}_always_cumacc.npy"), always["cumacc"])
    np.save(os.path.join(working_dir, f"{ds_name}_gated_cumacc.npy"), gated["cumacc"])
    np.save(
        os.path.join(working_dir, f"{ds_name}_gated_triggers.npy"), gated["triggers"]
    )
    np.save(os.path.join(working_dir, f"{ds_name}_gated_entropy.npy"), gated["entropy"])
    plt.figure(figsize=(8, 4))
    plt.plot(frozen["cumacc"], label=f"frozen {frozen['acc']:.3f}")
    plt.plot(always["cumacc"], label=f"always {always['acc']:.3f}")
    plt.plot(gated["cumacc"], label=f"gated {gated['acc']:.3f}")
    plt.xlabel("stream step")
    plt.ylabel("cumulative accuracy")
    plt.title(ds_name)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, f"{ds_name}_stream_accuracy.png"))
    plt.close()
    print(
        f"{ds_name} | frozen_acc={frozen['acc']:.4f} always_acc={always['acc']:.4f} gated_acc={gated['acc']:.4f} | frozen_srus={frozen['srus']:.4f} always_srus={always['srus']:.4f} gated_srus={gated['srus']:.4f}"
    )

# aggregate print
for ds_name, res in all_summary.items():
    print(f"{ds_name} metrics:")
    for mode in ["frozen", "always", "gated"]:
        r = res[mode]
        print(
            f"  {mode}: acc={r['acc']:.4f}, ece={r['ece']:.4f}, trigger_rate={r['trigger_rate']:.4f}, overhead={r['overhead']:.6f}, SRUS={r['srus']:.4f}"
        )

np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
