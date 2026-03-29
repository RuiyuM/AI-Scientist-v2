import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import time, json, copy, math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

dataset_names = ["arc_easy_shift", "pubmedqa_burst", "mmlu_clustered"]
experiment_data = {
    k: {
        "metrics": {"train": [], "val": [], "test": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
        "other": {},
    }
    for k in dataset_names
}


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_ece(conf, correct, n_bins=12):
    conf, correct = np.asarray(conf), np.asarray(correct)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (conf >= bins[i]) & (
            conf < bins[i + 1] if i < n_bins - 1 else conf <= bins[i + 1]
        )
        if m.any():
            ece += np.abs(correct[m].mean() - conf[m].mean()) * m.mean()
    return float(ece)


def srus(acc, frozen_acc, ece, frozen_ece, trig, overhead):
    return float(
        (acc - frozen_acc)
        - 0.5 * max(0.0, ece - frozen_ece)
        - 0.08 * trig
        - 1.2 * overhead
    )


def make_block(n, d, kind="source"):
    X = np.random.randn(n, d).astype(np.float32)
    if kind == "source":
        z0 = 0.9 * X[:, :10].sum(1) + 0.25 * X[:, 20:28].sum(1)
        z1 = 0.9 * X[:, 10:20].sum(1) + 0.25 * X[:, 28:36].sum(1)
    elif kind == "easy_shift":
        X = (1.2 * X + 0.25).astype(np.float32)
        z0 = 0.7 * X[:, :10].sum(1) + 0.55 * X[:, 16:26].sum(1)
        z1 = 0.8 * X[:, 10:20].sum(1) + 0.45 * X[:, 26:36].sum(1)
    elif kind == "burst_hard":
        X = (1.45 * X - 0.15).astype(np.float32)
        z0 = 0.55 * X[:, :10].sum(1) + 0.85 * X[:, 12:24].sum(1)
        z1 = 0.6 * X[:, 10:20].sum(1) + 0.9 * X[:, 6:18].sum(1)
    else:
        X = (0.95 * X + 0.5).astype(np.float32)
        z0 = 0.45 * X[:, :10].sum(1) + 1.0 * X[:, 18:30].sum(1)
        z1 = 0.45 * X[:, 10:20].sum(1) + 1.0 * X[:, 24:36].sum(1)
    y = (z1 > z0).astype(np.int64)
    return X, y


class ArrayDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return {"x": self.X[i], "y": self.y[i]}


class LoRALinear(nn.Module):
    def __init__(self, inp, out, r=8, alpha=8.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out, inp) * 0.03)
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
    def __init__(self, d=40, h=128):
        super().__init__()
        self.fc1 = LoRALinear(d, h, r=8, alpha=8.0)
        self.fc2 = LoRALinear(h, h, r=8, alpha=8.0)
        self.out = nn.Linear(h, 2)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

    def freeze_backbone(self):
        for p in self.parameters():
            p.requires_grad = False
        for p in self.fc1.lora_parameters() + self.fc2.lora_parameters():
            p.requires_grad = True

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True


def probs_entropy_margin(logits):
    p = F.softmax(logits, dim=-1)
    top2 = torch.topk(p, k=2, dim=-1).values
    ent = -(p * torch.log(p + 1e-8)).sum(-1)
    margin = top2[:, 0] - top2[:, 1]
    conf = top2[:, 0]
    return p, ent, margin, conf


def prepare_data(seed, d=40):
    set_seed(seed)
    source_X, source_y = make_block(12000, d, "source")
    cut = 10000
    mu = source_X[:cut].mean(0, keepdims=True)
    sd = source_X[:cut].std(0, keepdims=True) + 1e-6
    Xn = ((source_X - mu) / sd).astype(np.float32)
    train_ds = ArrayDataset(Xn[:cut], source_y[:cut])
    val_ds = ArrayDataset(Xn[cut:], source_y[cut:])

    def make_stream(name):
        if name == "arc_easy_shift":
            kinds = (
                ["source"] * 250
                + ["easy_shift"] * 450
                + ["source"] * 250
                + ["easy_shift"] * 250
            )
        elif name == "pubmedqa_burst":
            kinds = (
                ["source"] * 180
                + ["burst_hard"] * 200
                + ["source"] * 180
                + ["burst_hard"] * 200
                + ["source"] * 180
            )
        else:
            kinds = (
                ["source"] * 150
                + ["clustered_alt"] * 350
                + ["easy_shift"] * 350
                + ["source"] * 150
            )
        Xs, Ys = [], []
        for k in kinds:
            x, y = make_block(1, d, k)
            Xs.append(x[0])
            Ys.append(y[0])
        X = ((np.stack(Xs) - mu) / sd).astype(np.float32)
        y = np.array(Ys, dtype=np.int64)
        return [
            {
                "x": torch.tensor(X[i], dtype=torch.float32),
                "y": torch.tensor(y[i], dtype=torch.long),
            }
            for i in range(len(y))
        ]

    streams = {k: make_stream(k) for k in dataset_names}
    return train_ds, val_ds, streams


def train_source(seed):
    train_ds, val_ds, streams = prepare_data(seed)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    model = Net().to(device)
    model.unfreeze_all()
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    best_state, best_val = None, 1e9
    cache = {"entropy": [], "margin": [], "conf": [], "correct": []}

    for epoch in range(1, 25):
        model.train()
        tr_loss = tr_ok = tr_n = 0
        for batch in train_loader:
            batch = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            x, y = batch["x"], batch["y"]
            x = torch.clamp(x, -6, 6)
            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item() * y.size(0)
            tr_ok += (logits.argmax(1) == y).sum().item()
            tr_n += y.size(0)
        tr_loss /= tr_n
        tr_acc = tr_ok / tr_n

        model.eval()
        va_loss = va_ok = va_n = 0
        confs = []
        corr = []
        ents = []
        margins = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {
                    k: v.to(device)
                    for k, v in batch.items()
                    if isinstance(v, torch.Tensor)
                }
                x, y = batch["x"], batch["y"]
                x = torch.clamp(x, -6, 6)
                logits = model(x)
                loss = crit(logits, y)
                _, ent, margin, conf = probs_entropy_margin(logits)
                pred = logits.argmax(1)
                confs.extend(conf.detach().cpu().numpy())
                corr.extend((pred == y).float().detach().cpu().numpy())
                ents.extend(ent.detach().cpu().numpy())
                margins.extend(margin.detach().cpu().numpy())
                va_loss += loss.item() * y.size(0)
                va_ok += (pred == y).sum().item()
                va_n += y.size(0)
        va_loss /= va_n
        va_acc = va_ok / va_n
        va_ece = compute_ece(confs, corr)
        va_srus = srus(va_acc, va_acc, va_ece, va_ece, 0.0, 0.0)
        cache = {
            "entropy": np.array(ents),
            "margin": np.array(margins),
            "conf": np.array(confs),
            "correct": np.array(corr),
        }

        for ds in dataset_names:
            experiment_data[ds]["metrics"]["train"].append((seed, epoch, tr_acc))
            experiment_data[ds]["metrics"]["val"].append(
                (seed, epoch, va_acc, va_ece, va_srus)
            )
            experiment_data[ds]["losses"]["train"].append((seed, epoch, tr_loss))
            experiment_data[ds]["losses"]["val"].append((seed, epoch, va_loss))
            experiment_data[ds]["timestamps"].append((seed, epoch, time.time()))
        print(f"Epoch {epoch}: validation_loss = {va_loss:.4f}")
        if va_loss < best_val:
            best_val = va_loss
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    thrs = {
        "ent": float(np.quantile(cache["entropy"], 0.70)),
        "margin": float(np.quantile(cache["margin"], 0.30)),
        "conf": float(np.quantile(cache["conf"], 0.30)),
    }
    return model, streams, thrs


def stream_eval(
    base_model,
    data,
    thrs,
    mode="frozen",
    lr=0.035,
    reset_every=180,
    buffer_k=12,
    adapt_steps=3,
    prox=0.03,
):
    m = copy.deepcopy(base_model).to(device)
    m.freeze_backbone()
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.SGD([p for p in m.parameters() if p.requires_grad], lr=lr)
    init = {n: p.detach().clone() for n, p in m.named_parameters() if p.requires_grad}

    preds = []
    gt = []
    confs = []
    ents = []
    margins = []
    gates = []
    updates = []
    resets = []
    cumacc = []
    drift = []
    step_over = []
    buf_x = []
    correct_so_far = 0
    uncertainty_vals = []
    for i, item in enumerate(data):
        x = item["x"].unsqueeze(0).to(device)
        y = item["y"].unsqueeze(0).to(device)
        x = torch.clamp(x, -6, 6)

        m.eval()
        with torch.no_grad():
            logits = m(x)
            _, ent, margin, conf = probs_entropy_margin(logits)
            pred = logits.argmax(1)
        pred_i, y_i = int(pred.item()), int(y.item())
        preds.append(pred_i)
        gt.append(y_i)
        confs.append(float(conf.item()))
        ents.append(float(ent.item()))
        margins.append(float(margin.item()))
        correct_so_far += int(pred_i == y_i)
        cumacc.append(correct_so_far / (i + 1))

        ent_score = max(
            0.0, (float(ent.item()) - thrs["ent"]) / (abs(thrs["ent"]) + 1e-6)
        )
        margin_score = max(
            0.0, (thrs["margin"] - float(margin.item())) / (abs(thrs["margin"]) + 1e-6)
        )
        conf_score = max(
            0.0, (thrs["conf"] - float(conf.item())) / (abs(thrs["conf"]) + 1e-6)
        )
        u = 0.45 * ent_score + 0.30 * margin_score + 0.25 * conf_score
        uncertainty_vals.append(u)
        recent = uncertainty_vals[-10:]
        burst = len(recent) >= 4 and np.mean(np.array(recent) > 0.15) >= 0.4

        gate = False
        if mode == "always":
            gate = True
        elif mode == "gated":
            gate = (u > 0.12) and burst
        elif mode == "gated_reset":
            gate = (u > 0.12) and burst
        gates.append(int(gate))

        if gate:
            buf_x.append(x.detach())

        did_update = 0
        t0 = time.time()
        if mode != "frozen" and len(buf_x) >= buffer_k:
            xb = torch.cat(buf_x[-buffer_k:], dim=0).to(device)
            for _ in range(adapt_steps):
                m.train()
                logits_b = m(xb)
                with torch.no_grad():
                    pseudo = logits_b.detach().argmax(1)
                loss = crit(logits_b, pseudo)
                prox_term = 0.0
                for n, p in m.named_parameters():
                    if p.requires_grad:
                        prox_term = prox_term + ((p - init[n]) ** 2).mean()
                loss = loss + prox * prox_term
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in m.parameters() if p.requires_grad], 1.0
                )
                opt.step()
            did_update = 1
            buf_x = buf_x[-(buffer_k // 3) :] if "gated" in mode else []
        updates.append(did_update)
        step_over.append(time.time() - t0 if did_update else 0.0)

        do_reset = 0
        if mode == "gated_reset" and (i + 1) % reset_every == 0:
            for n, p in m.named_parameters():
                if p.requires_grad:
                    p.data.copy_(init[n].data)
            buf_x = []
            do_reset = 1
        resets.append(do_reset)

        dn = 0.0
        for n, p in m.named_parameters():
            if p.requires_grad:
                dn += ((p.detach() - init[n]) ** 2).mean().item()
        drift.append(dn)

    preds = np.array(preds)
    gt = np.array(gt)
    corr = (preds == gt).astype(np.float32)
    acc = float(corr.mean())
    ece = compute_ece(confs, corr)
    return {
        "acc": acc,
        "ece": ece,
        "trigger_rate": float(np.mean(gates)),
        "update_rate": float(np.mean(updates)),
        "reset_rate": float(np.mean(resets)),
        "overhead": float(np.mean(step_over)),
        "preds": preds,
        "gt": gt,
        "conf": np.array(confs),
        "entropy": np.array(ents),
        "margin": np.array(margins),
        "gate_flags": np.array(gates),
        "update_flags": np.array(updates),
        "reset_flags": np.array(resets),
        "cumacc": np.array(cumacc),
        "drift": np.array(drift),
    }


all_results = {
    ds: {m: [] for m in ["frozen", "always", "gated", "gated_reset"]}
    for ds in dataset_names
}
seeds = [3, 11, 19]

for seed in seeds:
    model, streams, thrs = train_source(seed)
    np.save(
        os.path.join(working_dir, f"thresholds_seed{seed}.npy"),
        np.array([thrs["ent"], thrs["margin"], thrs["conf"]], dtype=np.float32),
    )
    for ds_name, stream in streams.items():
        res = {
            "frozen": stream_eval(model, stream, thrs, "frozen"),
            "always": stream_eval(model, stream, thrs, "always"),
            "gated": stream_eval(model, stream, thrs, "gated"),
            "gated_reset": stream_eval(model, stream, thrs, "gated_reset"),
        }
        frozen_acc, frozen_ece = res["frozen"]["acc"], res["frozen"]["ece"]
        for mode in res:
            res[mode]["srus"] = srus(
                res[mode]["acc"],
                frozen_acc,
                res[mode]["ece"],
                frozen_ece,
                res[mode]["trigger_rate"],
                res[mode]["overhead"],
            )
            all_results[ds_name][mode].append(res[mode])
            experiment_data[ds_name]["metrics"]["test"].append(
                (
                    seed,
                    mode,
                    res[mode]["acc"],
                    res[mode]["ece"],
                    res[mode]["trigger_rate"],
                    res[mode]["update_rate"],
                    res[mode]["overhead"],
                    res[mode]["srus"],
                )
            )
        experiment_data[ds_name]["predictions"].append(
            (seed, "gated", res["gated"]["preds"].tolist())
        )
        experiment_data[ds_name]["ground_truth"].append(
            (seed, res["gated"]["gt"].tolist())
        )
        experiment_data[ds_name]["other"][f"seed_{seed}"] = {
            k: {
                kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv)
                for kk, vv in v.items()
            }
            for k, v in res.items()
        }

        for mode in res:
            np.save(
                os.path.join(working_dir, f"{ds_name}_{mode}_seed{seed}_cumacc.npy"),
                res[mode]["cumacc"],
            )
            np.save(
                os.path.join(working_dir, f"{ds_name}_{mode}_seed{seed}_entropy.npy"),
                res[mode]["entropy"],
            )
            np.save(
                os.path.join(working_dir, f"{ds_name}_{mode}_seed{seed}_margin.npy"),
                res[mode]["margin"],
            )
            np.save(
                os.path.join(working_dir, f"{ds_name}_{mode}_seed{seed}_conf.npy"),
                res[mode]["conf"],
            )
            np.save(
                os.path.join(
                    working_dir, f"{ds_name}_{mode}_seed{seed}_gate_flags.npy"
                ),
                res[mode]["gate_flags"],
            )
            np.save(
                os.path.join(
                    working_dir, f"{ds_name}_{mode}_seed{seed}_update_flags.npy"
                ),
                res[mode]["update_flags"],
            )
            np.save(
                os.path.join(
                    working_dir, f"{ds_name}_{mode}_seed{seed}_reset_flags.npy"
                ),
                res[mode]["reset_flags"],
            )
            np.save(
                os.path.join(working_dir, f"{ds_name}_{mode}_seed{seed}_drift.npy"),
                res[mode]["drift"],
            )

        plt.figure(figsize=(9, 4))
        for mode, c in [
            ("frozen", "black"),
            ("always", "tab:red"),
            ("gated", "tab:blue"),
            ("gated_reset", "tab:green"),
        ]:
            plt.plot(
                res[mode]["cumacc"],
                label=f'{mode} {res[mode]["acc"]:.3f}',
                linewidth=1.6,
            )
        plt.xlabel("stream step")
        plt.ylabel("cumulative accuracy")
        plt.title(f"{ds_name} seed{seed}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, f"{ds_name}_seed{seed}_stream_accuracy.png")
        )
        plt.close()

        plt.figure(figsize=(9, 4))
        plt.plot(res["gated"]["entropy"], label="entropy", alpha=0.9)
        plt.plot(
            res["gated"]["gate_flags"] * max(1e-6, res["gated"]["entropy"].max()),
            label="gate",
            alpha=0.8,
        )
        plt.axhline(
            thrs["ent"],
            color="tab:red",
            linestyle="--",
            label=f'ent_thr={thrs["ent"]:.3f}',
        )
        plt.xlabel("stream step")
        plt.ylabel("value")
        plt.title(f"{ds_name} seed{seed} gated uncertainty")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, f"{ds_name}_seed{seed}_gated_uncertainty.png")
        )
        plt.close()

summary = {}
for ds in dataset_names:
    summary[ds] = {}
    print(f"\n{ds}")
    for mode in ["frozen", "always", "gated", "gated_reset"]:
        vals = all_results[ds][mode]
        accs = np.array([v["acc"] for v in vals])
        eces = np.array([v["ece"] for v in vals])
        srus_vals = np.array([v["srus"] for v in vals])
        trigs = np.array([v["trigger_rate"] for v in vals])
        upds = np.array([v["update_rate"] for v in vals])
        ovs = np.array([v["overhead"] for v in vals])
        summary[ds][mode] = {
            "acc_mean": float(accs.mean()),
            "acc_std": float(accs.std()),
            "ece_mean": float(eces.mean()),
            "srus_mean": float(srus_vals.mean()),
            "trigger_mean": float(trigs.mean()),
            "update_mean": float(upds.mean()),
            "overhead_mean": float(ovs.mean()),
        }
        print(
            f"{mode}: acc={accs.mean():.4f}±{accs.std():.4f} ece={eces.mean():.4f} srus={srus_vals.mean():.4f} trigger={trigs.mean():.4f} update={upds.mean():.4f} overhead={ovs.mean():.6f}"
        )

np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
np.savez_compressed(
    os.path.join(working_dir, "experiment_data_compressed.npz"),
    experiment_data=np.array([experiment_data], dtype=object),
)
with open(os.path.join(working_dir, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("\nFinal summary:")
print(json.dumps(summary, indent=2))
