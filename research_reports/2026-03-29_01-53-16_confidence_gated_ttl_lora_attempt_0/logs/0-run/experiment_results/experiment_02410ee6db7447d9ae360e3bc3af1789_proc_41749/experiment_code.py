import os
import copy
import time
import json
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.manual_seed(7)
np.random.seed(7)

dataset_names = ["arc_easy_shift", "pubmedqa_burst", "mmlu_clustered"]
lora_ranks = [0, 1, 2, 4, 8]
mode_names = ["frozen", "always", "gated", "reset"]
ablation_name = "lora_capacity_ablation"

experiment_data = {
    ablation_name: {
        ds: {
            "metrics": {"train": [], "val": [], "test": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "timestamps": [],
            "thresholds": {},
            "rank_summaries": {},
            "rank_traces": {},
        }
        for ds in dataset_names
    }
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
        self.r = int(r)
        self.alpha = float(alpha)
        if self.r > 0:
            self.A = nn.Parameter(torch.randn(self.r, inp) * 0.02)
            self.B = nn.Parameter(torch.zeros(out, self.r))
        else:
            self.register_parameter("A", None)
            self.register_parameter("B", None)

    def forward(self, x):
        if self.r > 0:
            delta = (self.B @ self.A) * (self.alpha / max(1, self.r))
            w = self.weight + delta
        else:
            w = self.weight
        return F.linear(x, w, self.bias)

    def lora_parameters(self):
        return [] if self.r == 0 else [self.A, self.B]

    def reset_lora(self):
        if self.r > 0:
            nn.init.normal_(self.A, std=0.02)
            nn.init.zeros_(self.B)


class Net(nn.Module):
    def __init__(self, d=20, h=48, r=4, alpha=2.0):
        super().__init__()
        self.fc1 = LoRALinear(d, h, r=r, alpha=alpha)
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


# ---------- source training per rank ----------
def train_source_model(rank, epochs=8, lr=1e-3):
    model = Net(r=rank).to(device)
    criterion = nn.CrossEntropyLoss()
    model.unfreeze_all()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_state, best_val = None, 1e9
    train_metrics, val_metrics = [], []
    train_losses, val_losses = [], []
    val_uncertainty_cache = {"entropy": [], "margin": [], "conf": [], "correct": []}

    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss = tr_ok = tr_n = 0
        for batch in train_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
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
                x = batch["x"].to(device)
                y = batch["y"].to(device)
                logits = model(x)
                loss = criterion(logits, y)
                p, ent, margin = probs_entropy_margin(logits)
                c, pred = p.max(1)
                confs.extend(c.cpu().numpy().tolist())
                corr.extend((pred == y).float().cpu().numpy().tolist())
                ents.extend(ent.cpu().numpy().tolist())
                margins.extend(margin.cpu().numpy().tolist())
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

        train_metrics.append((epoch, tr_acc, tr_acc))
        val_metrics.append((epoch, va_acc, val_srus))
        train_losses.append((epoch, tr_loss))
        val_losses.append((epoch, va_loss))

        if va_loss < best_val:
            best_val = va_loss
            best_state = copy.deepcopy(model.state_dict())

        print(
            f"[rank={rank}] Epoch {epoch}: train_acc={tr_acc:.4f} val_acc={va_acc:.4f} val_loss={va_loss:.4f}"
        )

    model.load_state_dict(best_state)
    ent_thr = float(np.quantile(val_uncertainty_cache["entropy"], 0.75))
    margin_thr = float(np.quantile(val_uncertainty_cache["margin"], 0.25))
    conf_thr = float(np.quantile(val_uncertainty_cache["conf"], 0.25))

    return model, {
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "thresholds": {"entropy": ent_thr, "margin": margin_thr, "conf": conf_thr},
        "best_val_loss": float(best_val),
    }


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
    lora_params = m.fc1.lora_parameters()
    has_lora = len(lora_params) > 0

    initA = m.fc1.A.detach().clone() if has_lora else None
    initB = m.fc1.B.detach().clone() if has_lora else None
    opt = torch.optim.SGD(lora_params, lr=lr, momentum=0.0) if has_lora else None

    preds, gt, confs, ents, margins = [], [], [], [], []
    gate_flags, update_flags, reset_flags, cumacc = [], [], [], []
    update_times, buf_x, uncertainty_hist = [], [], []
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
        if has_lora and mode in ["always", "gated", "reset"] and len(buf_x) >= buffer_k:
            start = time.time()
            xb = torch.cat(buf_x[-buffer_k:], 0).to(device)
            for _ in range(adapt_steps):
                m.train()
                logits_b = m(xb)
                with torch.no_grad():
                    pseudo = logits_b.detach().argmax(1)
                loss = F.cross_entropy(logits_b, pseudo)
                loss = loss + prox * (
                    ((m.fc1.A - initA) ** 2).mean() + ((m.fc1.B - initB) ** 2).mean()
                )
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
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
            if has_lora:
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


# ---------- run ablation ----------
all_summary = {ds: {} for ds in dataset_names}
rank_models = {}
rank_training_info = {}

for rank in lora_ranks:
    model, train_info = train_source_model(rank)
    rank_models[rank] = model
    rank_training_info[rank] = train_info

    ent_thr = train_info["thresholds"]["entropy"]
    margin_thr = train_info["thresholds"]["margin"]
    conf_thr = train_info["thresholds"]["conf"]

    for ds_name, stream in streams.items():
        frozen = stream_eval(model, stream, "frozen", ent_thr, margin_thr, conf_thr)
        always = stream_eval(model, stream, "always", ent_thr, margin_thr, conf_thr)
        gated = stream_eval(model, stream, "gated", ent_thr, margin_thr, conf_thr)
        reset = stream_eval(
            model, stream, "reset", ent_thr, margin_thr, conf_thr, reset_every=45
        )
        results = {"frozen": frozen, "always": always, "gated": gated, "reset": reset}

        for mode, res in results.items():
            res["srus"] = srus(
                res["acc"],
                frozen["acc"],
                res["ece"],
                frozen["ece"],
                res["trigger_rate"],
                res["overhead"],
            )

        rank_key = f"r_{rank}"
        ds_entry = experiment_data[ablation_name][ds_name]

        if rank_key not in ds_entry["rank_summaries"]:
            ds_entry["rank_summaries"][rank_key] = {}
        if rank_key not in ds_entry["rank_traces"]:
            ds_entry["rank_traces"][rank_key] = {}

        ds_entry["metrics"]["train"].append(
            np.array(train_info["train_metrics"], dtype=np.float32)
        )
        ds_entry["metrics"]["val"].append(
            np.array(train_info["val_metrics"], dtype=np.float32)
        )
        ds_entry["losses"]["train"].append(
            np.array(train_info["train_losses"], dtype=np.float32)
        )
        ds_entry["losses"]["val"].append(
            np.array(train_info["val_losses"], dtype=np.float32)
        )
        ds_entry["timestamps"].append(time.time())
        ds_entry["thresholds"][rank_key] = np.array(
            [ent_thr, margin_thr, conf_thr], dtype=np.float32
        )

        test_rows = []
        for mode in mode_names:
            r = results[mode]
            test_rows.append(
                [
                    rank,
                    mode_names.index(mode),
                    r["acc"],
                    r["ece"],
                    r["trigger_rate"],
                    r["update_rate"],
                    r["reset_rate"],
                    r["overhead"],
                    r["srus"],
                ]
            )
            ds_entry["rank_summaries"][rank_key][mode] = {
                "acc": float(r["acc"]),
                "ece": float(r["ece"]),
                "trigger_rate": float(r["trigger_rate"]),
                "update_rate": float(r["update_rate"]),
                "reset_rate": float(r["reset_rate"]),
                "overhead": float(r["overhead"]),
                "srus": float(r["srus"]),
            }
            ds_entry["rank_traces"][rank_key][mode] = {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in r.items()
            }
            np.save(
                os.path.join(
                    working_dir, f"{ablation_name}_{ds_name}_r{rank}_{mode}_cumacc.npy"
                ),
                r["cumacc"],
            )
            np.save(
                os.path.join(
                    working_dir, f"{ablation_name}_{ds_name}_r{rank}_{mode}_conf.npy"
                ),
                r["conf"],
            )
            np.save(
                os.path.join(
                    working_dir, f"{ablation_name}_{ds_name}_r{rank}_{mode}_entropy.npy"
                ),
                r["entropy"],
            )
            np.save(
                os.path.join(
                    working_dir, f"{ablation_name}_{ds_name}_r{rank}_{mode}_margin.npy"
                ),
                r["margin"],
            )
            np.save(
                os.path.join(
                    working_dir,
                    f"{ablation_name}_{ds_name}_r{rank}_{mode}_gate_flags.npy",
                ),
                r["gate_flags"],
            )
            np.save(
                os.path.join(
                    working_dir,
                    f"{ablation_name}_{ds_name}_r{rank}_{mode}_update_flags.npy",
                ),
                r["update_flags"],
            )
            np.save(
                os.path.join(
                    working_dir,
                    f"{ablation_name}_{ds_name}_r{rank}_{mode}_reset_flags.npy",
                ),
                r["reset_flags"],
            )
            np.save(
                os.path.join(
                    working_dir, f"{ablation_name}_{ds_name}_r{rank}_{mode}_preds.npy"
                ),
                r["preds"],
            )
            np.save(
                os.path.join(
                    working_dir, f"{ablation_name}_{ds_name}_r{rank}_{mode}_gt.npy"
                ),
                r["gt"],
            )

        ds_entry["metrics"]["test"].append(np.array(test_rows, dtype=object))
        ds_entry["predictions"] = gated["preds"].tolist()
        ds_entry["ground_truth"] = gated["gt"].tolist()
        all_summary[ds_name][rank_key] = ds_entry["rank_summaries"][rank_key]

        print(
            f"{ds_name} | rank={rank} | "
            f"frozen_acc={frozen['acc']:.4f} always_acc={always['acc']:.4f} gated_acc={gated['acc']:.4f} reset_acc={reset['acc']:.4f} | "
            f"frozen_srus={frozen['srus']:.4f} always_srus={always['srus']:.4f} gated_srus={gated['srus']:.4f} reset_srus={reset['srus']:.4f}"
        )

# ---------- plots ----------
for ds_name in dataset_names:
    # cumulative accuracy for gated across ranks
    plt.figure(figsize=(8, 4))
    for rank in lora_ranks:
        r = experiment_data[ablation_name][ds_name]["rank_traces"][f"r_{rank}"]["gated"]
        plt.plot(
            np.array(r["cumacc"]), linewidth=1.8, label=f"r={rank} acc={r['acc']:.3f}"
        )
    plt.xlabel("stream step")
    plt.ylabel("cumulative accuracy")
    plt.title(f"{ds_name} gated cumulative accuracy by LoRA rank")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, f"{ablation_name}_{ds_name}_gated_rank_cumacc.png")
    )
    plt.close()

    # SRUS vs rank for each mode
    plt.figure(figsize=(8, 4))
    for mode, color in [
        ("frozen", "black"),
        ("always", "tab:red"),
        ("gated", "tab:blue"),
        ("reset", "tab:green"),
    ]:
        ys = [all_summary[ds_name][f"r_{rank}"][mode]["srus"] for rank in lora_ranks]
        plt.plot(lora_ranks, ys, marker="o", linewidth=1.8, color=color, label=mode)
    plt.xlabel("LoRA rank r")
    plt.ylabel("SRUS")
    plt.title(f"{ds_name} SRUS vs LoRA rank")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, f"{ablation_name}_{ds_name}_srus_vs_rank.png")
    )
    plt.close()

    # accuracy vs rank for each mode
    plt.figure(figsize=(8, 4))
    for mode, color in [
        ("frozen", "black"),
        ("always", "tab:red"),
        ("gated", "tab:blue"),
        ("reset", "tab:green"),
    ]:
        ys = [all_summary[ds_name][f"r_{rank}"][mode]["acc"] for rank in lora_ranks]
        plt.plot(lora_ranks, ys, marker="o", linewidth=1.8, color=color, label=mode)
    plt.xlabel("LoRA rank r")
    plt.ylabel("accuracy")
    plt.title(f"{ds_name} accuracy vs LoRA rank")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, f"{ablation_name}_{ds_name}_acc_vs_rank.png"))
    plt.close()

# training curves by rank
plt.figure(figsize=(8, 4))
for rank in lora_ranks:
    vals = np.array(rank_training_info[rank]["val_metrics"], dtype=np.float32)
    plt.plot(vals[:, 0], vals[:, 1], marker="o", linewidth=1.8, label=f"r={rank}")
plt.xlabel("epoch")
plt.ylabel("validation accuracy")
plt.title("Source validation accuracy by LoRA rank")
plt.legend()
plt.tight_layout()
plt.savefig(
    os.path.join(working_dir, f"{ablation_name}_source_val_accuracy_by_rank.png")
)
plt.close()

plt.figure(figsize=(8, 4))
for rank in lora_ranks:
    vals = np.array(rank_training_info[rank]["val_losses"], dtype=np.float32)
    plt.plot(vals[:, 0], vals[:, 1], marker="o", linewidth=1.8, label=f"r={rank}")
plt.xlabel("epoch")
plt.ylabel("validation loss")
plt.title("Source validation loss by LoRA rank")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(working_dir, f"{ablation_name}_source_val_loss_by_rank.png"))
plt.close()

# ---------- save consolidated arrays ----------
# convert lists to savable object arrays where useful
for ds_name in dataset_names:
    ds_entry = experiment_data[ablation_name][ds_name]
    for key1 in ["train", "val", "test"]:
        ds_entry["metrics"][key1] = np.array(ds_entry["metrics"][key1], dtype=object)
    for key1 in ["train", "val"]:
        ds_entry["losses"][key1] = np.array(ds_entry["losses"][key1], dtype=object)
    ds_entry["predictions"] = np.array(ds_entry["predictions"], dtype=np.int64)
    ds_entry["ground_truth"] = np.array(ds_entry["ground_truth"], dtype=np.int64)
    ds_entry["timestamps"] = np.array(ds_entry["timestamps"], dtype=np.float64)

np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)

summary_json = {
    "ablation": ablation_name,
    "ranks": lora_ranks,
    "datasets": {
        ds: {
            rank_key: {
                mode: {
                    "acc": float(all_summary[ds][rank_key][mode]["acc"]),
                    "ece": float(all_summary[ds][rank_key][mode]["ece"]),
                    "trigger_rate": float(
                        all_summary[ds][rank_key][mode]["trigger_rate"]
                    ),
                    "update_rate": float(
                        all_summary[ds][rank_key][mode]["update_rate"]
                    ),
                    "reset_rate": float(all_summary[ds][rank_key][mode]["reset_rate"]),
                    "overhead": float(all_summary[ds][rank_key][mode]["overhead"]),
                    "srus": float(all_summary[ds][rank_key][mode]["srus"]),
                }
                for mode in mode_names
            }
            for rank_key in all_summary[ds]
        }
        for ds in dataset_names
    },
}
with open(os.path.join(working_dir, "summary.json"), "w") as f:
    json.dump(summary_json, f, indent=2)

# also save rank-level train info
np.save(
    os.path.join(working_dir, f"{ablation_name}_rank_training_info.npy"),
    {f"r_{k}": v for k, v in rank_training_info.items()},
    allow_pickle=True,
)

print("\n=== LoRA Capacity Ablation Summary ===")
for ds_name in dataset_names:
    print(f"\n{ds_name}:")
    for rank in lora_ranks:
        rk = f"r_{rank}"
        row = all_summary[ds_name][rk]
        print(
            f"  rank={rank} | "
            f"frozen(acc={row['frozen']['acc']:.4f}, SRUS={row['frozen']['srus']:.4f}) "
            f"always(acc={row['always']['acc']:.4f}, SRUS={row['always']['srus']:.4f}) "
            f"gated(acc={row['gated']['acc']:.4f}, SRUS={row['gated']['srus']:.4f}) "
            f"reset(acc={row['reset']['acc']:.4f}, SRUS={row['reset']['srus']:.4f})"
        )

print(f"\nSaved experiment data to: {os.path.join(working_dir, 'experiment_data.npy')}")
