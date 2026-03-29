import os
import copy
import time
import json
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
ablation_types = ["with_source_norm", "no_source_norm"]

experiment_data = {
    ablation: {
        ds: {
            "metrics": {"train": [], "val": [], "test": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "timestamps": [],
            "extra": {},
        }
        for ds in dataset_names
    }
    for ablation in ablation_types
}


# ---------- synthetic data ----------
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


def apply_norm(X, use_source_norm):
    if use_source_norm:
        return ((X - norm_mu) / norm_sd).astype(np.float32)
    return X.astype(np.float32)


class SourceDataset(Dataset):
    def __init__(self, split="train", use_source_norm=True):
        X = apply_norm(source_all_X, use_source_norm)
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


def make_stream(name, use_source_norm=True, d=20):
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
    Xs, Ys, Ks = [], [], []
    for k in kinds:
        X, y = make_block(1, d, k)
        Xs.append(X[0])
        Ys.append(y[0])
        Ks.append(k)
    X = np.stack(Xs).astype(np.float32)
    X = apply_norm(X, use_source_norm)
    return [
        {
            "x": torch.tensor(X[i], dtype=torch.float32),
            "y": torch.tensor(Ys[i], dtype=torch.long),
            "kind": Ks[i],
        }
        for i in range(len(Ys))
    ]


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


criterion = nn.CrossEntropyLoss()


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


def evaluate_shift_slices(preds, gt, kinds):
    preds = np.array(preds)
    gt = np.array(gt)
    kinds = np.array(kinds)
    out = {}
    for k in sorted(set(kinds.tolist())):
        m = kinds == k
        out[k] = float((preds[m] == gt[m]).mean()) if m.any() else None
    non_source = kinds != "source"
    out["shifted_only_acc"] = (
        float((preds[non_source] == gt[non_source]).mean())
        if non_source.any()
        else None
    )
    src = kinds == "source"
    out["source_only_acc"] = (
        float((preds[src] == gt[src]).mean()) if src.any() else None
    )
    return out


# ---------- training ----------
def train_and_calibrate(use_source_norm, ablation_name):
    train_ds = SourceDataset("train", use_source_norm=use_source_norm)
    val_ds = SourceDataset("val", use_source_norm=use_source_norm)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)

    model = Net().to(device)
    model.unfreeze_all()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_state, best_val = None, 1e9
    val_uncertainty_cache = {"entropy": [], "margin": [], "conf": [], "correct": []}

    for epoch in range(1, 9):
        model.train()
        tr_loss = tr_ok = tr_n = 0
        for batch in train_loader:
            x, y = batch["x"].to(device), batch["y"].to(device)
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
                x, y = batch["x"].to(device), batch["y"].to(device)
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

        for ds in dataset_names:
            experiment_data[ablation_name][ds]["metrics"]["train"].append(
                (epoch, tr_acc, tr_acc)
            )
            experiment_data[ablation_name][ds]["metrics"]["val"].append(
                (epoch, va_acc, val_srus, ece)
            )
            experiment_data[ablation_name][ds]["losses"]["train"].append(
                (epoch, tr_loss)
            )
            experiment_data[ablation_name][ds]["losses"]["val"].append((epoch, va_loss))
            experiment_data[ablation_name][ds]["timestamps"].append(time.time())

        print(
            f"[{ablation_name}] Epoch {epoch}: train_acc={tr_acc:.4f} val_loss={va_loss:.4f} val_acc={va_acc:.4f} val_ece={ece:.4f}"
        )
        if va_loss < best_val:
            best_val = va_loss
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    ent_thr = float(np.quantile(val_uncertainty_cache["entropy"], 0.75))
    margin_thr = float(np.quantile(val_uncertainty_cache["margin"], 0.25))
    conf_thr = float(np.quantile(val_uncertainty_cache["conf"], 0.25))
    return model, (ent_thr, margin_thr, conf_thr)


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

    preds, gt, confs, ents, margins, kinds = [], [], [], [], [], []
    gate_flags, update_flags, reset_flags, cumacc = [], [], [], []
    update_times, buf_x = [], []
    uncertainty_hist = []
    ok = 0

    for i, item in enumerate(data):
        x = item["x"].unsqueeze(0).to(device)
        y = item["y"].unsqueeze(0).to(device)
        kinds.append(item["kind"])

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
    slices = evaluate_shift_slices(preds_np, gt_np, kinds)
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
        "kinds": np.array(kinds),
        "slice_acc": slices,
    }


all_summary = {}

# ---------- run ablation ----------
for ablation_name in ablation_types:
    use_source_norm = ablation_name == "with_source_norm"
    print(f"\n===== Ablation: {ablation_name} =====")
    model, (ent_thr, margin_thr, conf_thr) = train_and_calibrate(
        use_source_norm, ablation_name
    )
    streams = {
        k: make_stream(k, use_source_norm=use_source_norm) for k in dataset_names
    }
    all_summary[ablation_name] = {
        "thresholds": [ent_thr, margin_thr, conf_thr],
        "datasets": {},
    }

    np.save(
        os.path.join(working_dir, f"{ablation_name}_thresholds.npy"),
        np.array([ent_thr, margin_thr, conf_thr], dtype=np.float32),
    )

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
            experiment_data[ablation_name][ds_name]["metrics"]["test"].append(
                (
                    name,
                    res["acc"],
                    res["ece"],
                    res["trigger_rate"],
                    res["update_rate"],
                    res["reset_rate"],
                    res["overhead"],
                    res["srus"],
                    (
                        res["slice_acc"]["source_only_acc"]
                        if res["slice_acc"]["source_only_acc"] is not None
                        else -1.0
                    ),
                    (
                        res["slice_acc"]["shifted_only_acc"]
                        if res["slice_acc"]["shifted_only_acc"] is not None
                        else -1.0
                    ),
                )
            )

        experiment_data[ablation_name][ds_name]["predictions"] = gated["preds"].tolist()
        experiment_data[ablation_name][ds_name]["ground_truth"] = gated["gt"].tolist()
        experiment_data[ablation_name][ds_name]["extra"] = {
            "ablation_name": ablation_name,
            "use_source_norm": use_source_norm,
            "thresholds": {"entropy": ent_thr, "margin": margin_thr, "conf": conf_thr},
            "results": {
                k: {
                    kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv)
                    for kk, vv in v.items()
                }
                for k, v in results.items()
            },
        }
        all_summary[ablation_name]["datasets"][ds_name] = {
            mode: {
                "acc": float(results[mode]["acc"]),
                "ece": float(results[mode]["ece"]),
                "trigger_rate": float(results[mode]["trigger_rate"]),
                "update_rate": float(results[mode]["update_rate"]),
                "reset_rate": float(results[mode]["reset_rate"]),
                "overhead": float(results[mode]["overhead"]),
                "srus": float(results[mode]["srus"]),
                "source_only_acc": results[mode]["slice_acc"]["source_only_acc"],
                "shifted_only_acc": results[mode]["slice_acc"]["shifted_only_acc"],
            }
            for mode in results
        }

        for mode_name, res in results.items():
            prefix = f"{ablation_name}_{ds_name}_{mode_name}"
            np.save(os.path.join(working_dir, f"{prefix}_cumacc.npy"), res["cumacc"])
            np.save(os.path.join(working_dir, f"{prefix}_conf.npy"), res["conf"])
            np.save(os.path.join(working_dir, f"{prefix}_entropy.npy"), res["entropy"])
            np.save(os.path.join(working_dir, f"{prefix}_margin.npy"), res["margin"])
            np.save(
                os.path.join(working_dir, f"{prefix}_gate_flags.npy"), res["gate_flags"]
            )
            np.save(
                os.path.join(working_dir, f"{prefix}_update_flags.npy"),
                res["update_flags"],
            )
            np.save(
                os.path.join(working_dir, f"{prefix}_reset_flags.npy"),
                res["reset_flags"],
            )
            np.save(os.path.join(working_dir, f"{prefix}_preds.npy"), res["preds"])
            np.save(os.path.join(working_dir, f"{prefix}_gt.npy"), res["gt"])
            np.save(os.path.join(working_dir, f"{prefix}_kinds.npy"), res["kinds"])

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
                color=color,
            )
        plt.xlabel("stream step")
        plt.ylabel("cumulative accuracy")
        plt.title(f"{ablation_name} | {ds_name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, f"{ablation_name}_{ds_name}_stream_accuracy.png")
        )
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
        plt.title(f"{ablation_name} | {ds_name} gated uncertainty")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                working_dir, f"{ablation_name}_{ds_name}_gated_uncertainty.png"
            )
        )
        plt.close()

        print(
            f"[{ablation_name}] {ds_name} | "
            f"frozen_acc={frozen['acc']:.4f} always_acc={always['acc']:.4f} gated_acc={gated['acc']:.4f} reset_acc={reset['acc']:.4f} | "
            f"frozen_shift={frozen['slice_acc']['shifted_only_acc']:.4f} gated_shift={gated['slice_acc']['shifted_only_acc']:.4f}"
        )

# ---------- cross-ablation comparison ----------
comparison = {}
for ds_name in dataset_names:
    comparison[ds_name] = {}
    for mode in ["frozen", "always", "gated", "reset"]:
        a = all_summary["with_source_norm"]["datasets"][ds_name][mode]
        b = all_summary["no_source_norm"]["datasets"][ds_name][mode]
        comparison[ds_name][mode] = {
            "acc_delta_no_minus_norm": b["acc"] - a["acc"],
            "ece_delta_no_minus_norm": b["ece"] - a["ece"],
            "srus_delta_no_minus_norm": b["srus"] - a["srus"],
            "shifted_acc_delta_no_minus_norm": (
                None
                if a["shifted_only_acc"] is None or b["shifted_only_acc"] is None
                else b["shifted_only_acc"] - a["shifted_only_acc"]
            ),
            "source_acc_delta_no_minus_norm": (
                None
                if a["source_only_acc"] is None or b["source_only_acc"] is None
                else b["source_only_acc"] - a["source_only_acc"]
            ),
        }

for ablation_name in ablation_types:
    print(f"\n=== Summary: {ablation_name} ===")
    for ds_name in dataset_names:
        print(f"{ds_name} metrics:")
        for mode in ["frozen", "always", "gated", "reset"]:
            r = all_summary[ablation_name]["datasets"][ds_name][mode]
            print(
                f"  {mode}: acc={r['acc']:.4f}, ece={r['ece']:.4f}, trigger_rate={r['trigger_rate']:.4f}, "
                f"update_rate={r['update_rate']:.4f}, overhead={r['overhead']:.6f}, SRUS={r['srus']:.4f}, "
                f"source_acc={r['source_only_acc']:.4f}, shifted_acc={r['shifted_only_acc']:.4f}"
            )

print("\n=== No-normalization minus source-normalization deltas ===")
for ds_name in dataset_names:
    print(ds_name)
    for mode in ["frozen", "always", "gated", "reset"]:
        d = comparison[ds_name][mode]
        print(
            f"  {mode}: Δacc={d['acc_delta_no_minus_norm']:.4f}, Δece={d['ece_delta_no_minus_norm']:.4f}, "
            f"Δsrus={d['srus_delta_no_minus_norm']:.4f}, Δshifted_acc={d['shifted_acc_delta_no_minus_norm']:.4f}, "
            f"Δsource_acc={d['source_acc_delta_no_minus_norm']:.4f}"
        )

# ---------- save required outputs ----------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
np.save(
    os.path.join(working_dir, "ablation_comparison.npy"), comparison, allow_pickle=True
)
np.save(os.path.join(working_dir, "all_summary.npy"), all_summary, allow_pickle=True)

with open(os.path.join(working_dir, "summary.json"), "w") as f:
    json.dump(
        {
            "all_summary": all_summary,
            "comparison_no_minus_norm": comparison,
        },
        f,
        indent=2,
    )
