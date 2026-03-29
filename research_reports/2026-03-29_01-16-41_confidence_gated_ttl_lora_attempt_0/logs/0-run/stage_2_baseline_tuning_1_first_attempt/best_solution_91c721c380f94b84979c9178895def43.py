import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.manual_seed(42)
np.random.seed(42)

experiment_data = {
    "synthetic_shift_stream": {
        "metrics": {"train": [], "val": [], "stream": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "entropies": [],
        "segment_ids": [],
        "segment_accs": [],
        "config_results": {},
        "summary": {},
    },
    "rotated_shift_stream": {
        "metrics": {"train": [], "val": [], "stream": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "entropies": [],
        "segment_ids": [],
        "segment_accs": [],
        "config_results": {},
        "summary": {},
    },
    "imbalanced_noise_stream": {
        "metrics": {"train": [], "val": [], "stream": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "entropies": [],
        "segment_ids": [],
        "segment_accs": [],
        "config_results": {},
        "summary": {},
    },
}


class NumpyDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return {"x": self.x[idx], "y": self.y[idx]}


def make_gaussian_data(n, means, cov_scale=0.8, shift=None, class_probs=None):
    xs, ys = [], []
    counts = [n] * len(means)
    if class_probs is not None:
        counts = np.random.multinomial(n * len(means), class_probs).tolist()
    for c, m in enumerate(means):
        mean = np.array(m, dtype=np.float32)
        if shift is not None:
            mean = shift(mean, c)
        cov = np.eye(len(mean), dtype=np.float32) * cov_scale
        cnt = counts[c]
        if cnt == 0:
            continue
        pts = np.random.multivariate_normal(mean, cov, size=cnt)
        xs.append(pts)
        ys.append(np.full(cnt, c))
    x = np.concatenate(xs, axis=0).astype(np.float32)
    y = np.concatenate(ys, axis=0).astype(np.int64)
    idx = np.random.permutation(len(x))
    return x[idx], y[idx]


base_means = [(-2, -2), (2, -2), (0, 2.5)]
num_classes, input_dim = 3, 2

train_x, train_y = make_gaussian_data(500, base_means, cov_scale=0.7)
val_x, val_y = make_gaussian_data(150, base_means, cov_scale=0.7)

mu = train_x.mean(axis=0, keepdims=True)
sigma = train_x.std(axis=0, keepdims=True) + 1e-6
train_x = (train_x - mu) / sigma
val_x = (val_x - mu) / sigma


class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=4, alpha=1.0):
        super().__init__()
        self.base = nn.Linear(in_features, out_features)
        self.rank = rank
        self.alpha = alpha
        self.A = nn.Parameter(torch.zeros(in_features, rank))
        self.B = nn.Parameter(torch.zeros(rank, out_features))
        nn.init.normal_(self.A, std=0.02)
        nn.init.normal_(self.B, std=0.02)  # bugfix: avoid exactly-zero LoRA path

    def forward(self, x):
        return self.base(x) + (x @ self.A @ self.B) * (self.alpha / self.rank)


class SmallNet(nn.Module):
    def __init__(self, in_dim=2, hidden=32, out_dim=3, rank=4):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.head = LoRALinear(hidden, out_dim, rank=rank, alpha=1.0)

    def forward(self, x):
        return self.head(self.feat(x))


def evaluate(model, loader):
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            x, y = batch["x"], batch["y"]
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            total_loss += loss.item() * x.size(0)
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total += x.size(0)
    return total_loss / total, total_correct / total


def freeze_except_lora(m):
    for p in m.parameters():
        p.requires_grad = False
    m.head.A.requires_grad = True
    m.head.B.requires_grad = True


def batch_entropy(logits):
    p = torch.softmax(logits, dim=-1)
    return -(p * torch.log(p + 1e-8)).sum(dim=-1)


def batch_margin(logits):
    probs = torch.softmax(logits, dim=-1)
    top2 = torch.topk(probs, k=2, dim=-1).values
    return top2[:, 0] - top2[:, 1]


def build_stream(dataset_name):
    segs = []
    if dataset_name == "synthetic_shift_stream":
        shifts = [
            lambda mean, c: mean,
            lambda mean, c: mean + np.array([1.0, -0.5], dtype=np.float32),
            lambda mean, c: mean * np.array([0.6, 1.3], dtype=np.float32),
            lambda mean, c: np.array([mean[1], mean[0]], dtype=np.float32) * 0.9,
        ]
        covs = [0.7, 0.95, 1.0, 1.05]
        probs = [None, None, None, None]
    elif dataset_name == "rotated_shift_stream":

        def rot(theta):
            ct, st = np.cos(theta), np.sin(theta)
            return np.array([[ct, -st], [st, ct]], dtype=np.float32)

        shifts = [
            lambda mean, c: mean,
            lambda mean, c: rot(np.pi / 8) @ mean,
            lambda mean, c: rot(np.pi / 5) @ mean
            + np.array([0.3, 0.2], dtype=np.float32),
            lambda mean, c: rot(-np.pi / 6) @ mean
            + np.array([-0.4, 0.3], dtype=np.float32),
        ]
        covs = [0.7, 0.9, 1.0, 1.1]
        probs = [None, None, None, None]
    elif dataset_name == "imbalanced_noise_stream":
        shifts = [
            lambda mean, c: mean,
            lambda mean, c: mean + np.array([0.8 * (c - 1), 0.2], dtype=np.float32),
            lambda mean, c: mean * np.array([1.2, 0.7], dtype=np.float32),
            lambda mean, c: mean + np.array([-0.5, 0.6], dtype=np.float32),
        ]
        covs = [0.7, 1.1, 1.15, 1.2]
        probs = [None, [0.65, 0.25, 0.10], [0.20, 0.60, 0.20], [0.10, 0.20, 0.70]]
    else:
        raise ValueError(dataset_name)

    for sid, sh in enumerate(shifts):
        x, y = make_gaussian_data(
            90, base_means, cov_scale=covs[sid], shift=sh, class_probs=probs[sid]
        )
        x = (x - mu) / sigma  # important normalization
        segs.append((x, y, sid))
    return segs


def run_stream(
    base_state,
    dataset_name,
    adapt_mode="none",
    entropy_threshold=0.70,
    margin_threshold=0.55,
    adapt_steps=3,
    adapt_lr=1e-2,
):
    stream_segments = build_stream(dataset_name)
    m = SmallNet(in_dim=input_dim, out_dim=num_classes, rank=4).to(device)
    m.load_state_dict(copy.deepcopy(base_state))
    freeze_except_lora(m)
    opt = torch.optim.Adam([m.head.A, m.head.B], lr=adapt_lr)

    all_preds, all_gt, all_seg, all_ent = [], [], [], []
    seg_accs = []
    num_adapt_batches, total_batches = 0, 0

    for x_np, y_np, sid in stream_segments:
        ds = NumpyDataset(x_np, y_np)
        loader = DataLoader(ds, batch_size=32, shuffle=False)
        seg_correct, seg_total = 0, 0
        for batch in loader:
            total_batches += 1
            batch = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            x, y = batch["x"], batch["y"]

            m.eval()
            with torch.no_grad():
                logits_pre = m(x)
                ent_vec = batch_entropy(logits_pre)
                ent_mean = ent_vec.mean().item()
                margin_mean = batch_margin(logits_pre).mean().item()
                preds = logits_pre.argmax(dim=1)

            do_adapt = False
            if adapt_mode == "always":
                do_adapt = True
            elif adapt_mode == "gated":
                do_adapt = (ent_mean > entropy_threshold) or (
                    margin_mean < margin_threshold
                )

            if adapt_mode != "none" and do_adapt:
                num_adapt_batches += 1
                m.train()
                for _ in range(adapt_steps):
                    opt.zero_grad()
                    logits_adapt = m(x)
                    p = torch.softmax(logits_adapt, dim=-1)
                    ent_loss = batch_entropy(logits_adapt).mean()
                    conf_reg = -torch.max(p, dim=-1).values.mean()
                    adapt_loss = ent_loss + 0.25 * conf_reg
                    adapt_loss.backward()
                    torch.nn.utils.clip_grad_norm_([m.head.A, m.head.B], 1.0)
                    opt.step()
                m.eval()
                with torch.no_grad():
                    logits = m(x)
                    preds = logits.argmax(dim=1)
                    ent_mean = batch_entropy(logits).mean().item()

            seg_correct += (preds == y).sum().item()
            seg_total += x.size(0)
            all_preds.extend(preds.detach().cpu().numpy().tolist())
            all_gt.extend(y.detach().cpu().numpy().tolist())
            all_seg.extend([sid] * x.size(0))
            all_ent.extend([ent_mean] * x.size(0))
        seg_accs.append(seg_correct / max(seg_total, 1))

    return {
        "shift_robust_acc": float(np.mean(seg_accs)),
        "predictions": np.array(all_preds, dtype=np.int64),
        "ground_truth": np.array(all_gt, dtype=np.int64),
        "segment_ids": np.array(all_seg, dtype=np.int64),
        "entropies": np.array(all_ent, dtype=np.float32),
        "segment_accs": np.array(seg_accs, dtype=np.float32),
        "adapt_frequency": float(num_adapt_batches / max(total_batches, 1)),
    }


def train_and_eval_config(base_lr, epochs, batch_size, dataset_name):
    torch.manual_seed(42)
    np.random.seed(42)

    train_loader = DataLoader(
        NumpyDataset(train_x, train_y), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(NumpyDataset(val_x, val_y), batch_size=128, shuffle=False)

    model = SmallNet(in_dim=input_dim, out_dim=num_classes, rank=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)

    train_metrics, val_metrics = [], []
    train_losses, val_losses = [], []
    timestamps = []
    best_state, best_val_acc, best_val_loss, best_epoch = None, -1.0, None, -1

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, running_correct, total = 0.0, 0, 0
        for batch in train_loader:
            batch = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            x, y = batch["x"], batch["y"]
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            running_correct += (logits.argmax(dim=1) == y).sum().item()
            total += x.size(0)

        train_loss = running_loss / total
        train_acc = running_correct / total
        val_loss, val_acc = evaluate(model, val_loader)

        train_metrics.append((epoch, train_acc))
        val_metrics.append((epoch, val_acc))
        train_losses.append((epoch, train_loss))
        val_losses.append((epoch, val_loss))
        timestamps.append(time.time())

        print(
            f"Config lr={base_lr:.1e}, bs={batch_size}, ep={epochs} | Epoch {epoch}: validation_loss = {val_loss:.4f}"
        )
        print(
            f"Config lr={base_lr:.1e}, bs={batch_size}, ep={epochs} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    results_none = run_stream(best_state, dataset_name, "none")
    results_always = run_stream(
        best_state, dataset_name, "always", adapt_steps=3, adapt_lr=1e-2
    )
    results_gated = run_stream(
        best_state,
        dataset_name,
        "gated",
        entropy_threshold=0.70,
        margin_threshold=0.55,
        adapt_steps=3,
        adapt_lr=1e-2,
    )

    gain_always = results_always["shift_robust_acc"] - results_none["shift_robust_acc"]
    gain_gated = results_gated["shift_robust_acc"] - results_none["shift_robust_acc"]
    srag_always = gain_always / max(results_always["adapt_frequency"], 1e-6)
    srag_gated = gain_gated / max(results_gated["adapt_frequency"], 1e-6)

    return {
        "config": {"base_lr": base_lr, "epochs": epochs, "batch_size": batch_size},
        "metrics": {
            "train": train_metrics,
            "val": val_metrics,
            "stream": [
                {
                    "no_adapt": results_none["shift_robust_acc"],
                    "always_on": results_always["shift_robust_acc"],
                    "gated": results_gated["shift_robust_acc"],
                    "adapt_frequency_always": results_always["adapt_frequency"],
                    "adapt_frequency_gated": results_gated["adapt_frequency"],
                    "shift_robust_accuracy_gain_always": gain_always,
                    "shift_robust_accuracy_gain_gated": gain_gated,
                    "srag_always": srag_always,
                    "srag_gated": srag_gated,
                }
            ],
        },
        "losses": {"train": train_losses, "val": val_losses},
        "predictions": {
            "no_adapt": results_none["predictions"],
            "always_on": results_always["predictions"],
            "gated": results_gated["predictions"],
        },
        "ground_truth": results_none["ground_truth"],
        "segment_ids": results_none["segment_ids"],
        "entropies": {
            "no_adapt": results_none["entropies"],
            "always_on": results_always["entropies"],
            "gated": results_gated["entropies"],
        },
        "segment_accs": {
            "no_adapt": results_none["segment_accs"],
            "always_on": results_always["segment_accs"],
            "gated": results_gated["segment_accs"],
        },
        "timestamps": timestamps,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss,
    }


dataset_names = [
    "synthetic_shift_stream",
    "rotated_shift_stream",
    "imbalanced_noise_stream",
]
configs = [
    {"base_lr": 3e-4, "epochs": 24, "batch_size": 32},
    {"base_lr": 1e-3, "epochs": 24, "batch_size": 32},
    {"base_lr": 3e-3, "epochs": 16, "batch_size": 64},
]

for dataset_name in dataset_names:
    store = experiment_data[dataset_name]
    print(f"\n===== Running dataset: {dataset_name} =====")
    best_idx = None
    best_score = -1e9
    for i, cfg in enumerate(configs):
        key = f"lr_{cfg['base_lr']}_ep_{cfg['epochs']}_bs_{cfg['batch_size']}".replace(
            ".", "p"
        ).replace("-", "m")
        result = train_and_eval_config(
            cfg["base_lr"], cfg["epochs"], cfg["batch_size"], dataset_name
        )
        store["config_results"][key] = result

        if result["metrics"]["stream"][0]["gated"] > best_score:
            best_score = result["metrics"]["stream"][0]["gated"]
            best_idx = key

        print(
            f"{dataset_name} | {key} | best_val_acc={result['best_val_acc']:.4f} | "
            f"stream_none={result['metrics']['stream'][0]['no_adapt']:.4f} | "
            f"stream_always={result['metrics']['stream'][0]['always_on']:.4f} | "
            f"stream_gated={result['metrics']['stream'][0]['gated']:.4f} | "
            f"srag_gated={result['metrics']['stream'][0]['srag_gated']:.4f}"
        )

    keys = list(store["config_results"].keys())
    val_scores = [store["config_results"][k]["best_val_acc"] for k in keys]
    gated_scores = [
        store["config_results"][k]["metrics"]["stream"][0]["gated"] for k in keys
    ]
    srag_scores = [
        store["config_results"][k]["metrics"]["stream"][0]["srag_gated"] for k in keys
    ]
    best_val_idx = int(np.argmax(val_scores))
    best_gated_idx = int(np.argmax(gated_scores))
    best_srag_idx = int(np.argmax(srag_scores))

    store["summary"] = {
        "best_config_by_val": store["config_results"][keys[best_val_idx]]["config"],
        "best_val_acc": float(val_scores[best_val_idx]),
        "best_config_by_gated_stream": store["config_results"][keys[best_gated_idx]][
            "config"
        ],
        "best_gated_stream_acc": float(gated_scores[best_gated_idx]),
        "best_config_by_srag": store["config_results"][keys[best_srag_idx]]["config"],
        "best_srag": float(srag_scores[best_srag_idx]),
    }

    best_entry = store["config_results"][keys[best_gated_idx]]
    store["metrics"]["train"] = best_entry["metrics"]["train"]
    store["metrics"]["val"] = best_entry["metrics"]["val"]
    store["metrics"]["stream"] = best_entry["metrics"]["stream"]
    store["losses"]["train"] = best_entry["losses"]["train"]
    store["losses"]["val"] = best_entry["losses"]["val"]
    store["predictions"] = best_entry["predictions"]["gated"]
    store["ground_truth"] = best_entry["ground_truth"]
    store["entropies"] = best_entry["entropies"]["gated"]
    store["segment_ids"] = best_entry["segment_ids"]
    store["segment_accs"] = best_entry["segment_accs"]["gated"]

    np.save(
        os.path.join(working_dir, f"{dataset_name}_predictions.npy"),
        store["predictions"],
    )
    np.save(
        os.path.join(working_dir, f"{dataset_name}_ground_truth.npy"),
        store["ground_truth"],
    )
    np.save(
        os.path.join(working_dir, f"{dataset_name}_entropies.npy"), store["entropies"]
    )
    np.save(
        os.path.join(working_dir, f"{dataset_name}_segment_ids.npy"),
        store["segment_ids"],
    )
    np.save(
        os.path.join(working_dir, f"{dataset_name}_segment_accs.npy"),
        np.array(store["segment_accs"], dtype=np.float32),
    )

    train_loss_arr = np.array(
        [x[1] for x in store["losses"]["train"]], dtype=np.float32
    )
    val_loss_arr = np.array([x[1] for x in store["losses"]["val"]], dtype=np.float32)
    val_acc_arr = np.array([x[1] for x in store["metrics"]["val"]], dtype=np.float32)
    epoch_arr = np.arange(1, len(train_loss_arr) + 1)

    np.save(
        os.path.join(working_dir, f"{dataset_name}_train_losses.npy"), train_loss_arr
    )
    np.save(os.path.join(working_dir, f"{dataset_name}_val_losses.npy"), val_loss_arr)
    np.save(os.path.join(working_dir, f"{dataset_name}_val_accuracy.npy"), val_acc_arr)

    plt.figure(figsize=(7, 4))
    plt.plot(epoch_arr, train_loss_arr, label="train_loss")
    plt.plot(epoch_arr, val_loss_arr, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss Curves - {dataset_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, f"{dataset_name}_loss_curves.png"))
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(epoch_arr, val_acc_arr, label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Validation Accuracy - {dataset_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, f"{dataset_name}_val_accuracy_plot.png"))
    plt.close()

    plt.figure(figsize=(7, 4))
    seg_ids = np.unique(store["segment_ids"])
    seg_acc = np.array(store["segment_accs"], dtype=np.float32)
    plt.bar(seg_ids, seg_acc)
    plt.xlabel("Stream Segment")
    plt.ylabel("Accuracy")
    plt.title(f"Gated Segment Accuracy - {dataset_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, f"{dataset_name}_segment_accuracy.png"))
    plt.close()

summary_rows = []
for dataset_name in dataset_names:
    keys = list(experiment_data[dataset_name]["config_results"].keys())
    for k in keys:
        e = experiment_data[dataset_name]["config_results"][k]
        s = e["metrics"]["stream"][0]
        summary_rows.append(
            [
                dataset_names.index(dataset_name),
                e["config"]["base_lr"],
                e["config"]["epochs"],
                e["config"]["batch_size"],
                e["best_epoch"],
                e["best_val_acc"],
                e["best_val_loss"],
                s["no_adapt"],
                s["always_on"],
                s["gated"],
                s["adapt_frequency_always"],
                s["adapt_frequency_gated"],
                s["shift_robust_accuracy_gain_always"],
                s["shift_robust_accuracy_gain_gated"],
                s["srag_always"],
                s["srag_gated"],
            ]
        )

summary_rows = np.array(summary_rows, dtype=np.float32)
np.save(os.path.join(working_dir, "all_dataset_config_summary.npy"), summary_rows)
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)

for dataset_name in dataset_names:
    keys = list(experiment_data[dataset_name]["config_results"].keys())
    labels = []
    none_scores, always_scores, gated_scores = [], [], []
    for k in keys:
        e = experiment_data[dataset_name]["config_results"][k]
        labels.append(
            f"lr={e['config']['base_lr']:.0e}\nbs={e['config']['batch_size']}\nep={e['config']['epochs']}"
        )
        none_scores.append(e["metrics"]["stream"][0]["no_adapt"])
        always_scores.append(e["metrics"]["stream"][0]["always_on"])
        gated_scores.append(e["metrics"]["stream"][0]["gated"])

    xpos = np.arange(len(keys))
    width = 0.25
    plt.figure(figsize=(10, 4))
    plt.bar(xpos - width, none_scores, width=width, label="no_adapt")
    plt.bar(xpos, always_scores, width=width, label="always_on")
    plt.bar(xpos + width, gated_scores, width=width, label="gated")
    plt.xticks(xpos, labels)
    plt.ylabel("Shift-Robust Accuracy")
    plt.title(f"Stream Robustness by Config - {dataset_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, f"{dataset_name}_stream_robustness_by_config.png")
    )
    plt.close()

print("\n===== Final summaries =====")
for dataset_name in dataset_names:
    s = experiment_data[dataset_name]["summary"]
    print(
        f"{dataset_name} | best_val_acc={s['best_val_acc']:.4f} | "
        f"best_gated_stream_acc={s['best_gated_stream_acc']:.4f} | "
        f"best_srag={s['best_srag']:.4f}"
    )
