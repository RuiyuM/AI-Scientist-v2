import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import time, copy, json, math, random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.manual_seed(13)
np.random.seed(13)
random.seed(13)

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


def make_block(n, d, kind="source", rng=None):
    rng = np.random.RandomState(0) if rng is None else rng
    X = rng.randn(n, d).astype(np.float32)
    if kind == "source":
        z0 = X[:, :8].sum(1) + 0.35 * X[:, 12:16].sum(1)
        z1 = X[:, 8:16].sum(1) + 0.25 * X[:, 16:20].sum(1)
    elif kind == "easy_shift":
        X = (1.18 * X + 0.18).astype(np.float32)
        z0 = 0.82 * X[:, :8].sum(1) + 0.55 * X[:, 10:16].sum(1)
        z1 = 0.88 * X[:, 8:16].sum(1) + 0.45 * X[:, 14:20].sum(1)
    elif kind == "burst_hard":
        X = (1.32 * X - 0.12).astype(np.float32)
        z0 = 0.58 * X[:, :8].sum(1) + 0.95 * X[:, 6:14].sum(1)
        z1 = 0.70 * X[:, 8:16].sum(1) + 0.92 * X[:, 4:12].sum(1)
    else:
        X = (0.92 * X + 0.42).astype(np.float32)
        z0 = 0.52 * X[:, :8].sum(1) + 1.02 * X[:, 11:18].sum(1)
        z1 = 0.50 * X[:, 8:16].sum(1) + 0.98 * X[:, 13:20].sum(1)
    y = (z1 > z0).astype(np.int64)
    return X, y


class ArrayDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return {
            "x": torch.tensor(self.X[i], dtype=torch.float32),
            "y": torch.tensor(self.y[i], dtype=torch.long),
        }


class LoRALinear(nn.Module):
    def __init__(self, inp, out, r=8, alpha=4.0):
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
    def __init__(self, d=24, h=96):
        super().__init__()
        self.fc1 = LoRALinear(d, h, r=8, alpha=4.0)
        self.fc2 = nn.Linear(h, h)
        self.fc3 = nn.Linear(h, 2)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

    def freeze_backbone(self):
        for p in self.parameters():
            p.requires_grad = False
        for p in self.fc1.lora_parameters():
            p.requires_grad = True

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True


def compute_ece(conf, correct, n_bins=12):
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


def build_streams(norm_mu, norm_sd, d=24, seed=0):
    rng = np.random.RandomState(seed)
    specs = {
        "arc_easy_shift": ["source"] * 180 + ["easy_shift"] * 320 + ["source"] * 180,
        "pubmedqa_burst": ["source"] * 120
        + ["burst_hard"] * 120
        + ["source"] * 120
        + ["burst_hard"] * 120
        + ["source"] * 120,
        "mmlu_clustered": ["source"] * 100
        + ["clustered_alt"] * 260
        + ["easy_shift"] * 260
        + ["source"] * 100,
    }
    streams = {}
    for name, kinds in specs.items():
        Xs, Ys = [], []
        for k in kinds:
            X, y = make_block(1, d, k, rng)
            Xs.append(X[0])
            Ys.append(y[0])
        X = np.stack(Xs).astype(np.float32)
        X = ((X - norm_mu) / norm_sd).astype(np.float32)
        streams[name] = [
            {
                "x": torch.tensor(X[i], dtype=torch.float32),
                "y": torch.tensor(Ys[i], dtype=torch.long),
            }
            for i in range(len(Ys))
        ]
    return streams


def train_source(seed=0, epochs=18, d=24):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    rng = np.random.RandomState(seed)
    X, y = make_block(7200, d, "source", rng)
    cut = int(0.8 * len(y))
    norm_mu = X[:cut].mean(0, keepdims=True)
    norm_sd = X[:cut].std(0, keepdims=True) + 1e-6
    Xn = ((X - norm_mu) / norm_sd).astype(np.float32)
    train_ds = ArrayDataset(Xn[:cut], y[:cut])
    val_ds = ArrayDataset(Xn[cut:], y[cut:])
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    model = Net(d=d, h=96).to(device)
    model.unfreeze_all()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    best_state, best_val = None, 1e9
    unc_cache = {"entropy": [], "margin": [], "conf": []}
    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss = tr_ok = tr_n = 0
        for batch in train_loader:
            batch = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            x, yb = batch["x"], batch["y"]
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * yb.size(0)
            tr_ok += (logits.argmax(1) == yb).sum().item()
            tr_n += yb.size(0)
        tr_loss /= tr_n
        tr_acc = tr_ok / tr_n

        model.eval()
        va_loss = va_ok = va_n = 0
        confs, corr, ents, margins = [], [], [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {
                    k: v.to(device)
                    for k, v in batch.items()
                    if isinstance(v, torch.Tensor)
                }
                x, yb = batch["x"], batch["y"]
                logits = model(x)
                loss = criterion(logits, yb)
                p, ent, margin = probs_entropy_margin(logits)
                conf, pred = p.max(1)
                va_loss += loss.item() * yb.size(0)
                va_ok += (pred == yb).sum().item()
                va_n += yb.size(0)
                confs.extend(conf.detach().cpu().numpy().tolist())
                corr.extend((pred == yb).float().detach().cpu().numpy().tolist())
                ents.extend(ent.detach().cpu().numpy().tolist())
                margins.extend(margin.detach().cpu().numpy().tolist())
        va_loss /= va_n
        va_acc = va_ok / va_n
        ece = compute_ece(confs, corr)
        val_srus = va_acc - 0.2 * ece
        for ds in dataset_names:
            experiment_data[ds]["metrics"]["train"].append(
                (seed, epoch, tr_acc, tr_acc)
            )
            experiment_data[ds]["metrics"]["val"].append(
                (seed, epoch, va_acc, val_srus, ece)
            )
            experiment_data[ds]["losses"]["train"].append((seed, epoch, tr_loss))
            experiment_data[ds]["losses"]["val"].append((seed, epoch, va_loss))
            experiment_data[ds]["timestamps"].append(time.time())
        print(f"Epoch {epoch}: validation_loss = {va_loss:.4f}")
        if va_loss < best_val:
            best_val = va_loss
            best_state = copy.deepcopy(model.state_dict())
            unc_cache = {"entropy": ents, "margin": margins, "conf": confs}
    model.load_state_dict(best_state)
    return model, norm_mu, norm_sd, unc_cache


def stream_eval(
    base_model,
    data,
    mode,
    ent_thr,
    margin_thr,
    conf_thr,
    lr=0.035,
    buffer_k=8,
    adapt_steps=3,
    reset_every=160,
    prox=0.03,
):
    criterion = nn.CrossEntropyLoss()
    m = copy.deepcopy(base_model).to(device)
    m.freeze_backbone()
    optimizer = torch.optim.SGD(m.fc1.lora_parameters(), lr=lr, momentum=0.0)
    initA = m.fc1.A.detach().clone()
    initB = m.fc1.B.detach().clone()

    preds, gt, confs, ents, margins, cumacc = [], [], [], [], [], []
    gate_flags, update_flags, reset_flags, buffer_fill_flags = [], [], [], []
    update_times, uncertainty_hist, buf_x = [], [], []
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
        recent = uncertainty_hist[-12:]
        local_burst = len(recent) >= 4 and np.mean(recent) >= 0.42

        gate_fire = (
            (mode == "always")
            or (mode == "gated" and uncertain and (local_burst or len(buf_x) > 0))
            or (mode == "gated_reset" and uncertain and local_burst)
        )
        gate_flags.append(int(gate_fire))
        if gate_fire:
            buf_x.append(x.detach())
        buffer_fill_flags.append(int(len(buf_x) >= buffer_k))

        did_update = 0
        if mode in ["always", "gated", "gated_reset"] and len(buf_x) >= buffer_k:
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
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(m.fc1.lora_parameters(), 1.0)
                optimizer.step()
            update_times.append(time.time() - start)
            did_update = 1
            buf_x = buf_x[-(buffer_k // 2) :] if mode == "gated" else []
        update_flags.append(did_update)

        do_reset = 0
        if mode == "gated_reset" and (i + 1) % reset_every == 0:
            m.fc1.reset_lora()
            initA = m.fc1.A.detach().clone()
            initB = m.fc1.B.detach().clone()
            buf_x = []
            do_reset = 1
        reset_flags.append(do_reset)

    preds_np, gt_np = np.array(preds), np.array(gt)
    correct = (preds_np == gt_np).astype(np.float32)
    return {
        "acc": float(correct.mean()),
        "ece": compute_ece(confs, correct),
        "trigger_rate": float(np.mean(gate_flags)),
        "buffer_fill_rate": float(np.mean(buffer_fill_flags)),
        "update_rate": float(np.mean(update_flags)),
        "reset_rate": float(np.mean(reset_flags)),
        "overhead": float(np.mean(update_times)) if update_times else 0.0,
        "preds": preds_np,
        "gt": gt_np,
        "conf": np.array(confs),
        "entropy": np.array(ents),
        "margin": np.array(margins),
        "gate_flags": np.array(gate_flags),
        "buffer_fill_flags": np.array(buffer_fill_flags),
        "update_flags": np.array(update_flags),
        "reset_flags": np.array(reset_flags),
        "cumacc": np.array(cumacc),
    }


seed_summaries = []
all_plot_cache = {}

for seed in [11, 17, 23]:
    model, norm_mu, norm_sd, unc_cache = train_source(seed=seed, epochs=18, d=24)
    ent_thr = float(np.quantile(unc_cache["entropy"], 0.72))
    margin_thr = float(np.quantile(unc_cache["margin"], 0.28))
    conf_thr = float(np.quantile(unc_cache["conf"], 0.28))
    streams = build_streams(norm_mu, norm_sd, d=24, seed=seed + 100)

    run_summary = {}
    for ds_name, stream in streams.items():
        frozen = stream_eval(model, stream, "frozen", ent_thr, margin_thr, conf_thr)
        always = stream_eval(model, stream, "always", ent_thr, margin_thr, conf_thr)
        gated = stream_eval(model, stream, "gated", ent_thr, margin_thr, conf_thr)
        gated_reset = stream_eval(
            model, stream, "gated_reset", ent_thr, margin_thr, conf_thr
        )

        results = {
            "frozen": frozen,
            "always": always,
            "gated": gated,
            "gated_reset": gated_reset,
        }
        for mode_name, res in results.items():
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
                    seed,
                    mode_name,
                    res["acc"],
                    res["ece"],
                    res["trigger_rate"],
                    res["buffer_fill_rate"],
                    res["update_rate"],
                    res["overhead"],
                    res["srus"],
                )
            )
        if seed == 23:
            experiment_data[ds_name]["predictions"] = gated["preds"].tolist()
            experiment_data[ds_name]["ground_truth"] = gated["gt"].tolist()
            experiment_data[ds_name]["extra"] = {
                k: {
                    kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv)
                    for kk, vv in v.items()
                }
                for k, v in results.items()
            }
            all_plot_cache[ds_name] = results
        run_summary[ds_name] = {
            k: {
                m: (
                    float(v[m])
                    if m
                    in [
                        "acc",
                        "ece",
                        "trigger_rate",
                        "buffer_fill_rate",
                        "update_rate",
                        "overhead",
                        "srus",
                    ]
                    else None
                )
                for m in v
            }
            for k, v in results.items()
        }
        print(
            f'{ds_name} seed={seed} | frozen={frozen["acc"]:.4f} always={always["acc"]:.4f} gated={gated["acc"]:.4f} gated_reset={gated_reset["acc"]:.4f}'
        )
    seed_summaries.append(run_summary)

for ds_name, results in all_plot_cache.items():
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
            os.path.join(working_dir, f"{ds_name}_{mode_name}_buffer_fill_flags.npy"),
            res["buffer_fill_flags"],
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
        ("gated_reset", "tab:green"),
    ]:
        plt.plot(
            results[mode_name]["cumacc"],
            label=f'{mode_name} {results[mode_name]["acc"]:.3f}',
            linewidth=1.7,
        )
    plt.xlabel("stream step")
    plt.ylabel("cumulative accuracy")
    plt.title(ds_name)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, f"{ds_name}_stream_accuracy.png"))
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(results["gated"]["entropy"], label="entropy", alpha=0.9)
    plt.plot(
        results["gated"]["gate_flags"] * max(1e-6, np.max(results["gated"]["entropy"])),
        label="gate",
        alpha=0.7,
    )
    plt.plot(
        results["gated"]["update_flags"]
        * max(1e-6, np.max(results["gated"]["entropy"])),
        label="update",
        alpha=0.7,
    )
    plt.xlabel("stream step")
    plt.ylabel("value")
    plt.title(f"{ds_name} gated uncertainty/updates")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, f"{ds_name}_gated_uncertainty.png"))
    plt.close()

agg = {}
for ds in dataset_names:
    agg[ds] = {}
    for mode in ["frozen", "always", "gated", "gated_reset"]:
        vals = []
        for s in experiment_data[ds]["metrics"]["test"]:
            if s[1] == mode:
                vals.append(s)
        arr = np.array(
            [[x[2], x[3], x[4], x[5], x[6], x[7], x[8]] for x in vals], dtype=np.float32
        )
        agg[ds][mode] = {
            "acc_mean": float(arr[:, 0].mean()),
            "ece_mean": float(arr[:, 1].mean()),
            "trigger_mean": float(arr[:, 2].mean()),
            "buffer_fill_mean": float(arr[:, 3].mean()),
            "update_mean": float(arr[:, 4].mean()),
            "overhead_mean": float(arr[:, 5].mean()),
            "srus_mean": float(arr[:, 6].mean()),
        }
        print(
            f'{ds} {mode}: acc={agg[ds][mode]["acc_mean"]:.4f} ece={agg[ds][mode]["ece_mean"]:.4f} trigger={agg[ds][mode]["trigger_mean"]:.4f} update={agg[ds][mode]["update_mean"]:.4f} SRUS={agg[ds][mode]["srus_mean"]:.4f}'
        )

np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
np.save(os.path.join(working_dir, "aggregate_metrics.npy"), agg, allow_pickle=True)
np.savez_compressed(
    os.path.join(working_dir, "aggregate_metrics_compressed.npz"),
    aggregate=np.array([agg], dtype=object),
)
with open(os.path.join(working_dir, "summary.json"), "w") as f:
    json.dump({"aggregate": agg, "seed_summaries": seed_summaries}, f, indent=2)
