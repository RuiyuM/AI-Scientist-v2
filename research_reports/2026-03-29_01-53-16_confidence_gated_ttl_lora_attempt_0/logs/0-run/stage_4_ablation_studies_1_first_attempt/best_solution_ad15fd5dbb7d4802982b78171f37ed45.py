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
ablation_types = ["calibrated_quantile", "fixed_threshold", "threshold_free"]
baseline_bucket = "context_baselines"

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
    + [
        f"{ds}__{ab}"
        for ds in dataset_names
        for ab in ablation_types + [baseline_bucket]
    ]
}


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
    else:
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


def compute_ece(conf, correct, n_bins=10):
    conf, correct = np.array(conf), np.array(correct)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (conf >= bins[i]) & (
            (conf < bins[i + 1]) if i < n_bins - 1 else (conf <= bins[i + 1])
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


def adaptation_efficiency_score(acc, frozen_acc, update_rate, overhead, trigger_rate):
    cost = update_rate + 0.5 * trigger_rate + 50.0 * overhead
    return float((acc - frozen_acc) / (cost + 1e-8))


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
    val_aes = adaptation_efficiency_score(va_acc, va_acc, 0.0, 0.0, 0.0)
    val_uncertainty_cache = {
        "entropy": ents,
        "margin": margins,
        "conf": confs,
        "correct": corr,
    }

    for name in experiment_data:
        experiment_data[name]["metrics"]["train"].append((epoch, tr_acc, tr_acc, 0.0))
        experiment_data[name]["metrics"]["val"].append(
            (epoch, va_acc, val_srus, val_aes)
        )
        experiment_data[name]["losses"]["train"].append((epoch, tr_loss))
        experiment_data[name]["losses"]["val"].append((epoch, va_loss))
        experiment_data[name]["timestamps"].append(time.time())

    print(f"Epoch {epoch}: validation_loss = {va_loss:.4f}")
    if va_loss < best_val:
        best_val = va_loss
        best_state = copy.deepcopy(model.state_dict())

model.load_state_dict(best_state)

calibrated_thresholds = {
    "entropy": float(np.quantile(val_uncertainty_cache["entropy"], 0.75)),
    "margin": float(np.quantile(val_uncertainty_cache["margin"], 0.25)),
    "conf": float(np.quantile(val_uncertainty_cache["conf"], 0.25)),
}
fixed_thresholds = {"entropy": 0.62, "margin": 0.18, "conf": 0.62}
threshold_configs = {
    "calibrated_quantile": calibrated_thresholds,
    "fixed_threshold": fixed_thresholds,
    "threshold_free": {"entropy": None, "margin": None, "conf": None},
}


def stream_eval(
    base_model,
    data,
    mode,
    ent_thr=None,
    margin_thr=None,
    conf_thr=None,
    trigger_policy="calibrated_quantile",
    lr=0.05,
    reset_every=90,
    buffer_k=4,
    prox=0.02,
    adapt_steps=2,
    **kwargs,
):
    if ent_thr is None and "entropy" in kwargs:
        ent_thr = kwargs["entropy"]
    if margin_thr is None and "margin" in kwargs:
        margin_thr = kwargs["margin"]
    if conf_thr is None and "conf" in kwargs:
        conf_thr = kwargs["conf"]

    m = copy.deepcopy(base_model).to(device)
    m.freeze_backbone()
    initA = m.fc1.A.detach().clone().to(device)
    initB = m.fc1.B.detach().clone().to(device)
    opt = torch.optim.SGD(m.fc1.lora_parameters(), lr=lr, momentum=0.0)

    preds, gt, confs, ents, margins = [], [], [], [], []
    gate_flags, update_flags, reset_flags, cumacc = [], [], [], []
    update_times, buf_x = [], []
    uncertainty_hist, uncertain_flags = [], []
    ok = 0

    for i, item in enumerate(data):
        x = item["x"].unsqueeze(0).to(device)
        y = item["y"].unsqueeze(0).to(device)

        m.eval()
        with torch.no_grad():
            logits = m(x)
            p, ent, margin = probs_entropy_margin(logits)
            conf, pred = p.max(1)

        ent_v, margin_v, conf_v = (
            float(ent.item()),
            float(margin.item()),
            float(conf.item()),
        )
        pred_i, y_i = int(pred.item()), int(y.item())

        preds.append(pred_i)
        gt.append(y_i)
        confs.append(conf_v)
        ents.append(ent_v)
        margins.append(margin_v)
        ok += int(pred_i == y_i)
        cumacc.append(ok / (i + 1))

        if trigger_policy == "threshold_free":
            score_uncertain = (
                (ent_v >= np.mean(ents[-8:]) if len(ents) >= 3 else False)
                or (margin_v <= np.mean(margins[-8:]) if len(margins) >= 3 else False)
                or (conf_v <= np.mean(confs[-8:]) if len(confs) >= 3 else False)
            )
        else:
            score_uncertain = (
                (ent_thr is not None and ent_v >= ent_thr)
                or (margin_thr is not None and margin_v <= margin_thr)
                or (conf_thr is not None and conf_v <= conf_thr)
            )

        uncertainty_hist.append(int(score_uncertain))
        uncertain_flags.append(int(score_uncertain))
        recent = uncertainty_hist[-8:]
        local_burst = len(recent) >= 3 and (np.mean(recent) >= 0.375)

        if mode == "always":
            gate_fire = True
        elif mode == "gated":
            gate_fire = (
                (local_burst or len(buf_x) > 0)
                if trigger_policy == "threshold_free"
                else (score_uncertain and (local_burst or len(buf_x) > 0))
            )
        elif mode == "reset":
            gate_fire = True
        else:
            gate_fire = False
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
            buf_x = buf_x[-(buffer_k // 2) :] if mode == "gated" else []
        update_flags.append(did_update)

        do_reset = 0
        if mode in ["always", "gated", "reset"] and (i + 1) % reset_every == 0:
            m.fc1.reset_lora()
            initA = m.fc1.A.detach().clone().to(device)
            initB = m.fc1.B.detach().clone().to(device)
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
        "uncertain_flags": np.array(uncertain_flags),
        "gate_flags": np.array(gate_flags),
        "update_flags": np.array(update_flags),
        "reset_flags": np.array(reset_flags),
        "cumacc": np.array(cumacc),
    }


all_summary = {}
for ds_name, stream in streams.items():
    all_summary[ds_name] = {}

    frozen = stream_eval(
        model,
        stream,
        "frozen",
        **calibrated_thresholds,
        trigger_policy="calibrated_quantile",
    )
    always = stream_eval(
        model,
        stream,
        "always",
        **calibrated_thresholds,
        trigger_policy="calibrated_quantile",
    )
    reset = stream_eval(
        model,
        stream,
        "reset",
        **calibrated_thresholds,
        trigger_policy="calibrated_quantile",
        reset_every=45,
    )

    for name, res in {"frozen": frozen, "always": always, "reset": reset}.items():
        res["srus"] = srus(
            res["acc"],
            frozen["acc"],
            res["ece"],
            frozen["ece"],
            res["trigger_rate"],
            res["overhead"],
        )
        res["aes"] = adaptation_efficiency_score(
            res["acc"],
            frozen["acc"],
            res["update_rate"],
            res["overhead"],
            res["trigger_rate"],
        )

    key = f"{ds_name}__{baseline_bucket}"
    experiment_data[key]["metrics"]["test"] = [
        (
            "frozen",
            frozen["acc"],
            frozen["ece"],
            frozen["trigger_rate"],
            frozen["update_rate"],
            frozen["overhead"],
            frozen["srus"],
            frozen["aes"],
        ),
        (
            "always",
            always["acc"],
            always["ece"],
            always["trigger_rate"],
            always["update_rate"],
            always["overhead"],
            always["srus"],
            always["aes"],
        ),
        (
            "reset",
            reset["acc"],
            reset["ece"],
            reset["trigger_rate"],
            reset["update_rate"],
            reset["overhead"],
            reset["srus"],
            reset["aes"],
        ),
    ]
    experiment_data[key]["predictions"] = frozen["preds"].tolist()
    experiment_data[key]["ground_truth"] = frozen["gt"].tolist()
    experiment_data[key]["extra"] = {
        k: {
            kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv)
            for kk, vv in v.items()
        }
        for k, v in {"frozen": frozen, "always": always, "reset": reset}.items()
    }

    ablation_results = {}
    for ab_name, th in threshold_configs.items():
        gated = stream_eval(
            model,
            stream,
            "gated",
            ent_thr=th["entropy"],
            margin_thr=th["margin"],
            conf_thr=th["conf"],
            trigger_policy=ab_name,
        )
        gated["srus"] = srus(
            gated["acc"],
            frozen["acc"],
            gated["ece"],
            frozen["ece"],
            gated["trigger_rate"],
            gated["overhead"],
        )
        gated["aes"] = adaptation_efficiency_score(
            gated["acc"],
            frozen["acc"],
            gated["update_rate"],
            gated["overhead"],
            gated["trigger_rate"],
        )
        ablation_results[ab_name] = gated
        all_summary[ds_name][ab_name] = gated

        key = f"{ds_name}__{ab_name}"
        experiment_data[key]["metrics"]["test"] = [
            (
                "gated",
                gated["acc"],
                gated["ece"],
                gated["trigger_rate"],
                gated["update_rate"],
                gated["overhead"],
                gated["srus"],
                gated["aes"],
            )
        ]
        experiment_data[key]["predictions"] = gated["preds"].tolist()
        experiment_data[key]["ground_truth"] = gated["gt"].tolist()
        experiment_data[key]["extra"] = {
            "gated": {
                kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv)
                for kk, vv in gated.items()
            },
            "thresholds": th,
            "frozen_reference": {
                "acc": frozen["acc"],
                "ece": frozen["ece"],
                "srus": frozen["srus"],
                "aes": frozen["aes"],
            },
        }

    save_map = {
        "frozen": frozen,
        "always": always,
        "reset": reset,
        "gated_calibrated_quantile": ablation_results["calibrated_quantile"],
        "gated_fixed_threshold": ablation_results["fixed_threshold"],
        "gated_threshold_free": ablation_results["threshold_free"],
    }

    for mode_name, res in save_map.items():
        for key_arr in [
            "cumacc",
            "conf",
            "entropy",
            "margin",
            "uncertain_flags",
            "gate_flags",
            "update_flags",
            "reset_flags",
            "preds",
            "gt",
        ]:
            np.save(
                os.path.join(working_dir, f"{ds_name}_{mode_name}_{key_arr}.npy"),
                res[key_arr] if key_arr in res else np.array([]),
            )

    plt.figure(figsize=(9, 4.5))
    for mode_name, color in [
        ("frozen", "black"),
        ("always", "tab:red"),
        ("reset", "tab:green"),
        ("gated_calibrated_quantile", "tab:blue"),
        ("gated_fixed_threshold", "tab:orange"),
        ("gated_threshold_free", "tab:purple"),
    ]:
        plt.plot(
            save_map[mode_name]["cumacc"],
            label=f"{mode_name} {save_map[mode_name]['acc']:.3f}",
            linewidth=1.7,
            color=color,
        )
    plt.xlabel("stream step")
    plt.ylabel("cumulative accuracy")
    plt.title(f"{ds_name} ablation comparison")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, f"{ds_name}_ablation_cumacc.png"))
    plt.close()

    plt.figure(figsize=(9, 4.5))
    plt.plot(
        ablation_results["calibrated_quantile"]["entropy"], label="entropy", alpha=0.9
    )
    scale = max(1e-6, np.max(ablation_results["calibrated_quantile"]["entropy"]))
    plt.plot(
        ablation_results["calibrated_quantile"]["gate_flags"] * scale,
        label="calibrated gate",
        alpha=0.8,
    )
    plt.plot(
        ablation_results["fixed_threshold"]["gate_flags"] * scale * 0.9,
        label="fixed gate",
        alpha=0.7,
    )
    plt.plot(
        ablation_results["threshold_free"]["gate_flags"] * scale * 0.8,
        label="free gate",
        alpha=0.7,
    )
    plt.axhline(
        calibrated_thresholds["entropy"],
        color="tab:red",
        linestyle="--",
        label=f"cal_ent={calibrated_thresholds['entropy']:.3f}",
    )
    plt.axhline(
        fixed_thresholds["entropy"],
        color="tab:orange",
        linestyle=":",
        label=f"fixed_ent={fixed_thresholds['entropy']:.3f}",
    )
    plt.xlabel("stream step")
    plt.ylabel("value")
    plt.title(f"{ds_name} gating signals")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, f"{ds_name}_gating_signals.png"))
    plt.close()

    print(
        f"{ds_name} | frozen={frozen['acc']:.4f} always={always['acc']:.4f} reset={reset['acc']:.4f} | "
        f"cal={ablation_results['calibrated_quantile']['acc']:.4f} fixed={ablation_results['fixed_threshold']['acc']:.4f} "
        f"free={ablation_results['threshold_free']['acc']:.4f}"
    )

summary_json = {
    "thresholds": {
        "calibrated_quantile": calibrated_thresholds,
        "fixed_threshold": fixed_thresholds,
        "threshold_free": {"entropy": None, "margin": None, "conf": None},
    },
    "results": {},
}

for ds_name in dataset_names:
    summary_json["results"][ds_name] = {
        "context_baselines": {
            k: {
                "acc": float(v["acc"]),
                "ece": float(v["ece"]),
                "trigger_rate": float(v["trigger_rate"]),
                "update_rate": float(v["update_rate"]),
                "overhead": float(v["overhead"]),
                "srus": float(v["srus"]),
                "aes": float(v["aes"]),
            }
            for k, v in {
                "frozen": experiment_data[f"{ds_name}__{baseline_bucket}"]["extra"][
                    "frozen"
                ],
                "always": experiment_data[f"{ds_name}__{baseline_bucket}"]["extra"][
                    "always"
                ],
                "reset": experiment_data[f"{ds_name}__{baseline_bucket}"]["extra"][
                    "reset"
                ],
            }.items()
        },
        "ablations": {
            ab: {
                "acc": float(all_summary[ds_name][ab]["acc"]),
                "ece": float(all_summary[ds_name][ab]["ece"]),
                "trigger_rate": float(all_summary[ds_name][ab]["trigger_rate"]),
                "update_rate": float(all_summary[ds_name][ab]["update_rate"]),
                "overhead": float(all_summary[ds_name][ab]["overhead"]),
                "srus": float(all_summary[ds_name][ab]["srus"]),
                "aes": float(all_summary[ds_name][ab]["aes"]),
            }
            for ab in ablation_types
        },
    }

for ds_name in dataset_names:
    print(f"{ds_name} ablations:")
    for ab in ablation_types:
        r = all_summary[ds_name][ab]
        print(
            f"  {ab}: acc={r['acc']:.4f}, ece={r['ece']:.4f}, trigger_rate={r['trigger_rate']:.4f}, "
            f"update_rate={r['update_rate']:.4f}, overhead={r['overhead']:.6f}, SRUS={r['srus']:.4f}, AES={r['aes']:.4f}"
        )

np.save(
    os.path.join(working_dir, "thresholds_calibrated.npy"),
    np.array(
        [
            calibrated_thresholds["entropy"],
            calibrated_thresholds["margin"],
            calibrated_thresholds["conf"],
        ],
        dtype=np.float32,
    ),
)
np.save(
    os.path.join(working_dir, "thresholds_fixed.npy"),
    np.array(
        [
            fixed_thresholds["entropy"],
            fixed_thresholds["margin"],
            fixed_thresholds["conf"],
        ],
        dtype=np.float32,
    ),
)
np.save(
    os.path.join(working_dir, "val_entropy.npy"),
    np.array(val_uncertainty_cache["entropy"], dtype=np.float32),
)
np.save(
    os.path.join(working_dir, "val_margin.npy"),
    np.array(val_uncertainty_cache["margin"], dtype=np.float32),
)
np.save(
    os.path.join(working_dir, "val_conf.npy"),
    np.array(val_uncertainty_cache["conf"], dtype=np.float32),
)
np.save(
    os.path.join(working_dir, "val_correct.npy"),
    np.array(val_uncertainty_cache["correct"], dtype=np.float32),
)

for key, value in experiment_data.items():
    np.save(
        os.path.join(working_dir, f"{key}_metrics_train.npy"),
        np.array(value["metrics"]["train"], dtype=object),
        allow_pickle=True,
    )
    np.save(
        os.path.join(working_dir, f"{key}_metrics_val.npy"),
        np.array(value["metrics"]["val"], dtype=object),
        allow_pickle=True,
    )
    np.save(
        os.path.join(working_dir, f"{key}_metrics_test.npy"),
        np.array(value["metrics"]["test"], dtype=object),
        allow_pickle=True,
    )
    np.save(
        os.path.join(working_dir, f"{key}_losses_train.npy"),
        np.array(value["losses"]["train"], dtype=object),
        allow_pickle=True,
    )
    np.save(
        os.path.join(working_dir, f"{key}_losses_val.npy"),
        np.array(value["losses"]["val"], dtype=object),
        allow_pickle=True,
    )
    np.save(
        os.path.join(working_dir, f"{key}_predictions.npy"),
        np.array(value["predictions"], dtype=object),
        allow_pickle=True,
    )
    np.save(
        os.path.join(working_dir, f"{key}_ground_truth.npy"),
        np.array(value["ground_truth"], dtype=object),
        allow_pickle=True,
    )
    np.save(
        os.path.join(working_dir, f"{key}_timestamps.npy"),
        np.array(value["timestamps"], dtype=np.float64),
        allow_pickle=True,
    )

np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)

with open(os.path.join(working_dir, "summary.json"), "w") as f:
    json.dump(summary_json, f, indent=2)

print("Done. Saved experiment_data.npy and all stream traces.")
