import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import copy, time, math, json
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

dataset_names = ["arc_easy_shift", "pubmedqa_burst", "mmlu_clustered"]
experiment_data = {
    name: {
        "metrics": {"train": [], "val": [], "test": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
        "extra": {},
    }
    for name in dataset_names
}


# ---------- synthetic HF-benchmark-inspired data with fixed source normalization ----------
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


source_all_X, source_all_y = make_block(2400, 20, "source")
cut = int(0.8 * len(source_all_y))
norm_mu = source_all_X[:cut].mean(0, keepdims=True)
norm_sd = source_all_X[:cut].std(0, keepdims=True) + 1e-6


class SourceDataset(Dataset):
    def __init__(self, split="train"):
        X = ((source_all_X - norm_mu) / norm_sd).astype(np.float32)
        y = source_all_y
        if split == "train":
            self.X, self.y = X[:cut], y[:cut]
        else:
            self.X, self.y = X[cut:], y[cut:]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return {
            "x": torch.tensor(self.X[i], dtype=torch.float32),
            "y": torch.tensor(self.y[i], dtype=torch.long),
        }


def make_stream(name, d=20):
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
    else:
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
    X = ((X - norm_mu) / norm_sd).astype(np.float32)
    return [
        {
            "x": torch.tensor(X[i], dtype=torch.float32),
            "y": torch.tensor(Ys[i], dtype=torch.long),
        }
        for i in range(len(Ys))
    ]


train_ds, val_ds = SourceDataset("train"), SourceDataset("val")
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)
streams = {k: make_stream(k) for k in dataset_names}


# ---------- model ----------
class LoRALinear(nn.Module):
    def __init__(self, inp, out, r=4, alpha=2.0):
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
        self.fc1 = LoRALinear(d, h, r=4, alpha=2.0)
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


# ---------- metrics ----------
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


def probs_entropy_margin(logits):
    p = F.softmax(logits, dim=-1)
    top2 = torch.topk(p, k=2, dim=-1).values
    ent = -(p * torch.log(p + 1e-8)).sum(-1)
    margin = top2[:, 0] - top2[:, 1]
    return p, ent, margin


def srus(acc, frozen_acc, ece, frozen_ece, trig, overhead):
    return float(
        (acc - frozen_acc)
        - 0.5 * max(0.0, ece - frozen_ece)
        - 0.10 * trig
        - 1.5 * overhead
    )


# ---------- train ----------
best_state, best_val = None, 1e9
val_uncertainty_cache = {"entropy": [], "margin": [], "conf": [], "correct": []}
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
    confs, corr, ents, margins = [], [], [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            x, y = batch["x"], batch["y"]
            logits = model(x)
            loss = criterion(logits, y)
            p, ent, margin = probs_entropy_margin(logits)
            c, pred = p.max(1)
            confs.extend(c.detach().cpu().numpy().tolist())
            corr.extend((pred == y).float().detach().cpu().numpy().tolist())
            ents.extend(ent.detach().cpu().numpy().tolist())
            margins.extend(margin.detach().cpu().numpy().tolist())
            va_loss += loss.item() * y.size(0)
            va_ok += (pred == y).sum().item()
            va_n += y.size(0)
    va_loss /= va_n
    va_acc = va_ok / va_n
    ece = compute_ece(confs, corr)
    val_srus = va_acc - 0.2 * ece
    val_uncertainty_cache = {
        "entropy": ents,
        "margin": margins,
        "conf": confs,
        "correct": corr,
    }

    for ds in experiment_data:
        experiment_data[ds]["metrics"]["train"].append((epoch, tr_acc, tr_acc))
        experiment_data[ds]["metrics"]["val"].append((epoch, va_acc, val_srus))
        experiment_data[ds]["losses"]["train"].append((epoch, tr_loss))
        experiment_data[ds]["losses"]["val"].append((epoch, va_loss))
        experiment_data[ds]["timestamps"].append(time.time())
    print(f"Epoch {epoch}: validation_loss = {va_loss:.4f}")
    if va_loss < best_val:
        best_val = va_loss
        best_state = copy.deepcopy(model.state_dict())

model.load_state_dict(best_state)

# calibrated uncertainty thresholds from validation
ent_thr = float(np.quantile(val_uncertainty_cache["entropy"], 0.75))
margin_thr = float(np.quantile(val_uncertainty_cache["margin"], 0.25))
conf_thr = float(np.quantile(val_uncertainty_cache["conf"], 0.25))


# ---------- stream eval ----------
def stream_eval(
    base_model,
    data,
    mode,
    ent_thr,
    margin_thr,
    conf_thr,
    lr=0.05,
    reset_every=90,
    buffer_k=4,
    prox=0.02,
    adapt_steps=2,
):
    m = copy.deepcopy(base_model).to(device)
    m.freeze_backbone()
    initA = m.fc1.A.detach().clone()
    initB = m.fc1.B.detach().clone()
    opt = torch.optim.SGD(m.fc1.lora_parameters(), lr=lr, momentum=0.0)

    preds, gt, confs, ents, margins = [], [], [], [], []
    gate_flags, update_flags, reset_flags, cumacc = [], [], [], []
    update_times, buf_x = [], []
    uncertainty_hist = []
    ok = 0

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
        margins.append(float(margin.item()))
        ok += int(pred_i == y_i)
        cumacc.append(ok / (i + 1))

        uncertain = (
            (float(ent.item()) >= ent_thr)
            or (float(margin.item()) <= margin_thr)
            or (float(conf.item()) <= conf_thr)
        )
        uncertainty_hist.append(int(uncertain))
        recent = uncertainty_hist[-8:]
        local_burst = len(recent) >= 3 and np.mean(recent) >= 0.375

        gate_fire = False
        if mode == "always":
            gate_fire = True
        elif mode == "gated":
            gate_fire = uncertain and (local_burst or len(buf_x) > 0)
        elif mode == "reset":
            gate_fire = True
        gate_flags.append(int(gate_fire))

        if gate_fire:
            buf_x.append(x.detach())

        did_update = 0
        if mode in ["always", "gated", "reset"] and len(buf_x) >= buffer_k:
            start = time.time()
            xb = torch.cat(buf_x[-buffer_k:], 0).to(device)
            for _ in range(adapt_steps):
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
                torch.nn.utils.clip_grad_norm_(m.fc1.lora_parameters(), 1.0)
                opt.step()
            update_times.append(time.time() - start)
            did_update = 1
            if mode == "gated":
                buf_x = buf_x[-(buffer_k // 2) :]
            else:
                buf_x = []
        update_flags.append(did_update)

        do_reset = 0
        if mode in ["always", "gated", "reset"] and (i + 1) % reset_every == 0:
            m.fc1.reset_lora()
            initA = m.fc1.A.detach().clone()
            initB = m.fc1.B.detach().clone()
            buf_x = []
            do_reset = 1
        reset_flags.append(do_reset)

    preds_np, gt_np = np.array(preds), np.array(gt)
    correct = (preds_np == gt_np).astype(np.float32)
    acc = float(correct.mean())
    ece = compute_ece(confs, correct)
    return {
        "acc": acc,
        "ece": ece,
        "trigger_rate": float(np.mean(gate_flags)),
        "update_rate": float(np.mean(update_flags)),
        "reset_rate": float(np.mean(reset_flags)),
        "overhead": float(np.mean(update_times)) if len(update_times) else 0.0,
        "preds": preds_np,
        "gt": gt_np,
        "conf": np.array(confs),
        "entropy": np.array(ents),
        "margin": np.array(margins),
        "gate_flags": np.array(gate_flags),
        "update_flags": np.array(update_flags),
        "reset_flags": np.array(reset_flags),
        "cumacc": np.array(cumacc),
    }


# ---------- run 3 streams x 4 baselines ----------
all_summary = {}
for ds_name, stream in streams.items():
    frozen = stream_eval(model, stream, "frozen", ent_thr, margin_thr, conf_thr)
    always = stream_eval(model, stream, "always", ent_thr, margin_thr, conf_thr)
    gated = stream_eval(model, stream, "gated", ent_thr, margin_thr, conf_thr)
    reset = stream_eval(
        model, stream, "reset", ent_thr, margin_thr, conf_thr, reset_every=45
    )

    results = {"frozen": frozen, "always": always, "gated": gated, "reset": reset}
    for name, res in results.items():
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
                res["update_rate"],
                res["overhead"],
                res["srus"],
            )
        )

    experiment_data[ds_name]["predictions"] = gated["preds"].tolist()
    experiment_data[ds_name]["ground_truth"] = gated["gt"].tolist()
    experiment_data[ds_name]["extra"] = {
        k: {
            kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv)
            for kk, vv in v.items()
        }
        for k, v in results.items()
    }
    all_summary[ds_name] = results

    for mode_name, res in results.items():
        np.save(
            os.path.join(working_dir, f"{ds_name}_{mode_name}_cumacc.npy"),
            res["cumacc"],
        )
        np.save(
            os.path.join(working_dir, f"{ds_name}_{mode_name}_conf.npy"), res["conf"]
        )
        np.save(
            os.path.join(working_dir, f"{ds_name}_{mode_name}_entropy.npy"),
            res["entropy"],
        )
        np.save(
            os.path.join(working_dir, f"{ds_name}_{mode_name}_margin.npy"),
            res["margin"],
        )
        np.save(
            os.path.join(working_dir, f"{ds_name}_{mode_name}_gate_flags.npy"),
            res["gate_flags"],
        )
        np.save(
            os.path.join(working_dir, f"{ds_name}_{mode_name}_update_flags.npy"),
            res["update_flags"],
        )
        np.save(
            os.path.join(working_dir, f"{ds_name}_{mode_name}_reset_flags.npy"),
            res["reset_flags"],
        )

    plt.figure(figsize=(8, 4))
    for mode_name, color in [
        ("frozen", "black"),
        ("always", "tab:red"),
        ("gated", "tab:blue"),
        ("reset", "tab:green"),
    ]:
        plt.plot(
            results[mode_name]["cumacc"],
            label=f"{mode_name} {results[mode_name]['acc']:.3f}",
            linewidth=1.8,
        )
    plt.xlabel("stream step")
    plt.ylabel("cumulative accuracy")
    plt.title(ds_name)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, f"{ds_name}_stream_accuracy.png"))
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(gated["entropy"], label="gated entropy", alpha=0.9)
    plt.plot(
        gated["gate_flags"] * max(1e-6, np.max(gated["entropy"])),
        label="gate fired",
        alpha=0.8,
    )
    plt.axhline(
        ent_thr, color="tab:red", linestyle="--", label=f"entropy_thr={ent_thr:.3f}"
    )
    plt.xlabel("stream step")
    plt.ylabel("value")
    plt.title(f"{ds_name} gated uncertainty")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, f"{ds_name}_gated_uncertainty.png"))
    plt.close()

    print(
        f"{ds_name} | frozen_acc={frozen['acc']:.4f} always_acc={always['acc']:.4f} gated_acc={gated['acc']:.4f} reset_acc={reset['acc']:.4f} | "
        f"frozen_srus={frozen['srus']:.4f} always_srus={always['srus']:.4f} gated_srus={gated['srus']:.4f} reset_srus={reset['srus']:.4f}"
    )

for ds_name, res in all_summary.items():
    print(f"{ds_name} metrics:")
    for mode in ["frozen", "always", "gated", "reset"]:
        r = res[mode]
        print(
            f"  {mode}: acc={r['acc']:.4f}, ece={r['ece']:.4f}, trigger_rate={r['trigger_rate']:.4f}, "
            f"update_rate={r['update_rate']:.4f}, overhead={r['overhead']:.6f}, SRUS={r['srus']:.4f}"
        )

np.save(
    os.path.join(working_dir, "thresholds.npy"),
    np.array([ent_thr, margin_thr, conf_thr], dtype=np.float32),
)
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
np.savez_compressed(
    os.path.join(working_dir, "experiment_data_compressed.npz"),
    experiment_data=np.array([experiment_data], dtype=object),
)

with open(os.path.join(working_dir, "summary.json"), "w") as f:
    json.dump(
        {
            ds: {
                mode: {
                    "acc": float(all_summary[ds][mode]["acc"]),
                    "ece": float(all_summary[ds][mode]["ece"]),
                    "trigger_rate": float(all_summary[ds][mode]["trigger_rate"]),
                    "update_rate": float(all_summary[ds][mode]["update_rate"]),
                    "overhead": float(all_summary[ds][mode]["overhead"]),
                    "srus": float(all_summary[ds][mode]["srus"]),
                }
                for mode in all_summary[ds]
            }
            for ds in all_summary
        },
        f,
        indent=2,
    )
