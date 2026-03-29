import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import copy
import time
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

BASE_SEED = 42
random.seed(BASE_SEED)
np.random.seed(BASE_SEED)
torch.manual_seed(BASE_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(BASE_SEED)

experiment_data = {
    "synthetic_reasoning_stream": {
        "metrics": {"train": [], "val": [], "test": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
        "stream_results": {},
        "trials": {},
        "epoch_candidates": [],
        "selected_epochs": None,
        "selected_hparams": None,
        "shift_normalized_accuracy_gain": [],
    },
    "feature_permutation_stream": {
        "metrics": {"train": [], "val": [], "test": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
        "stream_results": {},
        "severity": None,
        "shift_normalized_accuracy_gain": [],
    },
    "nonlinear_boundary_stream": {
        "metrics": {"train": [], "val": [], "test": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
        "stream_results": {},
        "severity": None,
        "shift_normalized_accuracy_gain": [],
    },
}


class SyntheticReasoningDataset(Dataset):
    def __init__(self, n=2000, d=20, variant="source", seed=42):
        self.n = n
        self.d = d
        self.variant = variant
        self.seed = seed
        self.X, self.y, self.severity = self.make_data(n, d, variant, seed)

    @staticmethod
    def _normalize(X):
        mu = X.mean(axis=0, keepdims=True)
        std = X.std(axis=0, keepdims=True) + 1e-6
        return ((X - mu) / std).astype(np.float32)

    @staticmethod
    def make_data(n, d, variant="source", seed=42):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n, d), dtype=np.float32)

        if variant == "source":
            s0 = X[:, :5].sum(1)
            s1 = X[:, 5:10].sum(1)
            y = (s1 > s0).astype(np.int64)
            severity = 0.0

        elif variant == "shifted":
            X = (1.3 * X + 0.4).astype(np.float32)
            s0 = 0.6 * X[:, :5].sum(1) + 0.4 * X[:, 10:15].sum(1)
            s1 = 0.7 * X[:, 5:10].sum(1) + 0.3 * X[:, 15:20].sum(1)
            y = (s1 > s0 + 0.2).astype(np.int64)
            severity = 1.0

        elif variant == "feature_permutation":
            X = (1.1 * X - 0.2).astype(np.float32)
            perm = np.array([5, 6, 7, 8, 9, 0, 1, 2, 3, 4] + list(range(10, d)))
            X = X[:, perm]
            s0 = 0.9 * X[:, :5].sum(1) + 0.15 * X[:, 10:15].sum(1)
            s1 = 1.0 * X[:, 5:10].sum(1) + 0.1 * X[:, 15:20].sum(1)
            y = (s1 > s0 + 0.1).astype(np.int64)
            severity = 1.3

        elif variant == "nonlinear_boundary":
            X = (1.2 * X + 0.15).astype(np.float32)
            s0 = X[:, :5].sum(1)
            s1 = X[:, 5:10].sum(1)
            nonlin = 0.35 * (X[:, 10:15] ** 2).sum(1) - 0.2 * np.abs(X[:, 15:20]).sum(1)
            y = ((s1 + nonlin) > (s0 + 0.25)).astype(np.int64)
            severity = 1.6

        else:
            raise ValueError(f"Unknown variant: {variant}")

        X = SyntheticReasoningDataset._normalize(X)
        return torch.tensor(X), torch.tensor(y), float(severity)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {"x": self.X[idx], "y": self.y[idx]}


class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=4, alpha=1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.rank = rank
        self.alpha = alpha
        self.A = nn.Parameter(torch.zeros(rank, in_features))
        self.B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.normal_(self.A, std=0.02)
        nn.init.zeros_(self.B)
        self.register_buffer("A_init", self.A.detach().clone())
        self.register_buffer("B_init", self.B.detach().clone())

    def forward(self, x):
        delta = (self.B @ self.A) * (self.alpha / self.rank)
        return F.linear(x, self.weight + delta, self.bias)

    def lora_parameters(self):
        return [self.A, self.B]

    def lora_reg_loss(self):
        return ((self.A - self.A_init) ** 2).mean() + (
            (self.B - self.B_init) ** 2
        ).mean()

    def reset_lora(self):
        with torch.no_grad():
            self.A.copy_(self.A_init)
            self.B.copy_(self.B_init)


class SimpleNet(nn.Module):
    def __init__(self, d=20, h=64, num_classes=2, rank=4):
        super().__init__()
        self.fc1 = LoRALinear(d, h, rank=rank, alpha=1.0)
        self.fc2 = nn.Linear(h, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.tanh(x)
        return self.fc2(x)

    def freeze_backbone(self):
        for p in self.parameters():
            p.requires_grad = False
        for p in self.fc1.lora_parameters():
            p.requires_grad = True

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True

    def reset_lora(self):
        self.fc1.reset_lora()

    def lora_reg_loss(self):
        return self.fc1.lora_reg_loss()


def entropy_from_logits(logits):
    p = F.softmax(logits, dim=-1)
    return -(p * torch.log(p + 1e-8)).sum(dim=-1)


def margin_from_logits(logits):
    probs = F.softmax(logits, dim=-1)
    top2 = torch.topk(probs, k=min(2, probs.size(-1)), dim=-1).values
    if top2.size(-1) == 1:
        return top2[:, 0]
    return top2[:, 0] - top2[:, 1]


def make_datasets_and_loaders(seed=42, batch_size=64):
    train_ds = SyntheticReasoningDataset(n=2500, d=20, variant="source", seed=seed)
    val_ds = SyntheticReasoningDataset(n=500, d=20, variant="source", seed=seed + 1)
    test_shift_ds = SyntheticReasoningDataset(
        n=800, d=20, variant="shifted", seed=seed + 2
    )
    test_perm_ds = SyntheticReasoningDataset(
        n=800, d=20, variant="feature_permutation", seed=seed + 3
    )
    test_nl_ds = SyntheticReasoningDataset(
        n=800, d=20, variant="nonlinear_boundary", seed=seed + 4
    )

    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, generator=g
    )
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)
    return (
        train_ds,
        val_ds,
        [test_shift_ds, test_perm_ds, test_nl_ds],
        train_loader,
        val_loader,
    )


def evaluate_loader(model, loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, total_correct, total = 0.0, 0, 0
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            x, y = batch["x"], batch["y"]
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * y.size(0)
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.size(0)
    return total_loss / total, total_correct / total


def stream_evaluate(
    base_model,
    dataset,
    mode="frozen",
    entropy_thresh=0.50,
    margin_thresh=0.20,
    reset_every=200,
    lr=5e-3,
    reg_lambda=5e-3,
    grad_clip=1.0,
):
    stream_model = copy.deepcopy(base_model).to(device)
    stream_model.freeze_backbone()
    stream_model.eval()
    opt = torch.optim.Adam(stream_model.fc1.lora_parameters(), lr=lr)

    preds_all, gt_all, ent_all, margin_all, triggered, cumulative_acc = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    correct = 0

    for i in range(len(dataset)):
        batch = dataset[i]
        x = batch["x"].unsqueeze(0).to(device)
        y = batch["y"].unsqueeze(0).to(device)

        stream_model.eval()
        with torch.no_grad():
            logits = stream_model(x)
            ent = entropy_from_logits(logits).item()
            mar = margin_from_logits(logits).item()
            pred = logits.argmax(dim=1)

        preds_all.append(pred.item())
        gt_all.append(y.item())
        ent_all.append(ent)
        margin_all.append(mar)
        correct += int(pred.item() == y.item())
        cumulative_acc.append(correct / (i + 1))

        do_update = False
        if mode == "always":
            do_update = True
        elif mode == "gated":
            do_update = (ent > entropy_thresh) or (mar < margin_thresh)

        triggered.append(int(do_update))

        if do_update:
            stream_model.train()
            opt.zero_grad()
            logits_u = stream_model(x)
            p = F.softmax(logits_u, dim=-1)
            ent_loss = -(p * torch.log(p + 1e-8)).sum(dim=-1).mean()
            reg_loss = stream_model.lora_reg_loss()
            loss_u = ent_loss + reg_lambda * reg_loss
            loss_u.backward()
            torch.nn.utils.clip_grad_norm_(
                stream_model.fc1.lora_parameters(), grad_clip
            )
            opt.step()

        if (
            mode in ["always", "gated"]
            and reset_every > 0
            and (i + 1) % reset_every == 0
        ):
            stream_model.reset_lora()

    acc = float(np.mean(np.array(preds_all) == np.array(gt_all)))
    return {
        "accuracy": acc,
        "preds": np.array(preds_all),
        "gt": np.array(gt_all),
        "entropy": np.array(ent_all),
        "margin": np.array(margin_all),
        "triggered": np.array(triggered),
        "cumulative_acc": np.array(cumulative_acc),
    }


def train_source_model(
    epochs, train_loader, val_loader, seed=42, lr=1e-3, weight_decay=1e-4
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = SimpleNet().to(device)
    criterion = nn.CrossEntropyLoss()
    model.unfreeze_all()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state, best_val, best_epoch = None, float("inf"), -1
    history = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "timestamps": [],
        "best_val_loss": None,
        "best_epoch": None,
    }

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for batch in train_loader:
            batch = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            x, y = batch["x"], batch["y"]
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * y.size(0)
            train_correct += (logits.argmax(dim=1) == y).sum().item()
            train_total += y.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total
        val_loss, val_acc = evaluate_loader(model, val_loader)

        history["metrics"]["train"].append((epoch, train_acc))
        history["metrics"]["val"].append((epoch, val_acc))
        history["losses"]["train"].append((epoch, train_loss))
        history["losses"]["val"].append((epoch, val_loss))
        history["timestamps"].append(time.time())

        print(f"Epoch {epoch}: validation_loss = {val_loss:.4f}")
        print(
            f"[epochs={epochs}, lr={lr}, bs={train_loader.batch_size}] Epoch {epoch}: train_acc={train_acc:.4f} val_acc={val_acc:.4f} val_loss={val_loss:.4f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    history["best_val_loss"] = best_val
    history["best_epoch"] = best_epoch
    return model, history


def compute_shift_normalized_gain(frozen_acc, gated_acc, severity):
    denom = max(float(severity), 1e-6)
    return (gated_acc - frozen_acc) / denom


epoch_candidates = [8, 16, 30]
lr_candidates = [1e-3, 3e-4]
batch_candidates = [64, 128]
experiment_data["synthetic_reasoning_stream"]["epoch_candidates"] = epoch_candidates

best_trial_key = None
best_trial_score = -1e9
all_scores = []

trial_idx = 0
for epochs in epoch_candidates:
    for lr in lr_candidates:
        for batch_size in batch_candidates:
            trial_idx += 1
            trial_key = f"trial_{trial_idx}_epochs_{epochs}_lr_{lr}_bs_{batch_size}"
            train_ds, val_ds, test_datasets, train_loader, val_loader = (
                make_datasets_and_loaders(seed=BASE_SEED, batch_size=batch_size)
            )
            model, history = train_source_model(
                epochs,
                train_loader,
                val_loader,
                seed=BASE_SEED,
                lr=lr,
                weight_decay=1e-4,
            )

            ds_names = [
                "synthetic_reasoning_stream",
                "feature_permutation_stream",
                "nonlinear_boundary_stream",
            ]
            ds_objs = {
                "synthetic_reasoning_stream": test_datasets[0],
                "feature_permutation_stream": test_datasets[1],
                "nonlinear_boundary_stream": test_datasets[2],
            }

            per_dataset = {}
            gains = []
            for ds_name in ds_names:
                ds = ds_objs[ds_name]
                frozen_res = stream_evaluate(model, ds, mode="frozen")
                always_res = stream_evaluate(
                    model,
                    ds,
                    mode="always",
                    entropy_thresh=0.50,
                    margin_thresh=0.20,
                    reset_every=200,
                    lr=5e-3,
                )
                gated_res = stream_evaluate(
                    model,
                    ds,
                    mode="gated",
                    entropy_thresh=0.50,
                    margin_thresh=0.20,
                    reset_every=200,
                    lr=5e-3,
                )

                gain = compute_shift_normalized_gain(
                    frozen_res["accuracy"], gated_res["accuracy"], ds.severity
                )
                gains.append(gain)

                per_dataset[ds_name] = {
                    "severity": ds.severity,
                    "frozen": frozen_res,
                    "always": always_res,
                    "gated": gated_res,
                    "sna_gain": gain,
                }

                print(
                    f'[{trial_key}] {ds_name} | frozen={frozen_res["accuracy"]:.4f} | '
                    f'always={always_res["accuracy"]:.4f} | gated={gated_res["accuracy"]:.4f} | '
                    f"Shift-Normalized Accuracy Gain={gain:.6f}"
                )

            mean_gain = float(np.mean(gains))
            all_scores.append((epochs, lr, batch_size, mean_gain))

            trial = {
                "epochs": epochs,
                "lr": lr,
                "batch_size": batch_size,
                "best_epoch": history["best_epoch"],
                "best_val_loss": history["best_val_loss"],
                "metrics": history["metrics"],
                "losses": history["losses"],
                "timestamps": history["timestamps"],
                "per_dataset": {},
                "mean_shift_normalized_accuracy_gain": mean_gain,
            }

            for ds_name, results in per_dataset.items():
                trial["per_dataset"][ds_name] = {
                    "severity": results["severity"],
                    "stream_results": {
                        "frozen": {
                            "Shifted-Stream Accuracy": results["frozen"]["accuracy"],
                            "trigger_rate": float(
                                results["frozen"]["triggered"].mean()
                            ),
                        },
                        "always": {
                            "Shifted-Stream Accuracy": results["always"]["accuracy"],
                            "trigger_rate": float(
                                results["always"]["triggered"].mean()
                            ),
                        },
                        "gated": {
                            "Shifted-Stream Accuracy": results["gated"]["accuracy"],
                            "trigger_rate": float(results["gated"]["triggered"].mean()),
                        },
                    },
                    "test_metrics": {
                        "frozen_acc": results["frozen"]["accuracy"],
                        "always_acc": results["always"]["accuracy"],
                        "gated_acc": results["gated"]["accuracy"],
                        "shift_normalized_accuracy_gain": results["sna_gain"],
                    },
                    "arrays": {
                        "frozen_preds": results["frozen"]["preds"],
                        "always_preds": results["always"]["preds"],
                        "gated_preds": results["gated"]["preds"],
                        "ground_truth": results["gated"]["gt"],
                        "gated_entropy": results["gated"]["entropy"],
                        "gated_margin": results["gated"]["margin"],
                        "gated_triggered": results["gated"]["triggered"],
                        "frozen_cumacc": results["frozen"]["cumulative_acc"],
                        "always_cumacc": results["always"]["cumulative_acc"],
                        "gated_cumacc": results["gated"]["cumulative_acc"],
                    },
                }

            experiment_data["synthetic_reasoning_stream"]["trials"][trial_key] = trial

            if mean_gain > best_trial_score:
                best_trial_score = mean_gain
                best_trial_key = trial_key

selected = experiment_data["synthetic_reasoning_stream"]["trials"][best_trial_key]
experiment_data["synthetic_reasoning_stream"]["selected_epochs"] = selected["epochs"]
experiment_data["synthetic_reasoning_stream"]["selected_hparams"] = {
    "epochs": selected["epochs"],
    "lr": selected["lr"],
    "batch_size": selected["batch_size"],
}
experiment_data["synthetic_reasoning_stream"]["metrics"]["train"] = selected["metrics"][
    "train"
]
experiment_data["synthetic_reasoning_stream"]["metrics"]["val"] = selected["metrics"][
    "val"
]
experiment_data["synthetic_reasoning_stream"]["losses"]["train"] = selected["losses"][
    "train"
]
experiment_data["synthetic_reasoning_stream"]["losses"]["val"] = selected["losses"][
    "val"
]
experiment_data["synthetic_reasoning_stream"]["timestamps"] = selected["timestamps"]

for ds_name in [
    "synthetic_reasoning_stream",
    "feature_permutation_stream",
    "nonlinear_boundary_stream",
]:
    ds_res = selected["per_dataset"][ds_name]
    experiment_data[ds_name]["stream_results"] = ds_res["stream_results"]
    experiment_data[ds_name]["metrics"]["test"].append((0, ds_res["test_metrics"]))
    experiment_data[ds_name]["predictions"] = ds_res["arrays"]["gated_preds"].tolist()
    experiment_data[ds_name]["ground_truth"] = ds_res["arrays"]["ground_truth"].tolist()
    experiment_data[ds_name]["severity"] = ds_res["severity"]
    experiment_data[ds_name]["shift_normalized_accuracy_gain"].append(
        (0, ds_res["test_metrics"]["shift_normalized_accuracy_gain"])
    )
    if ds_name == "synthetic_reasoning_stream":
        experiment_data[ds_name]["stream_arrays"] = ds_res["arrays"]

print(
    f'Selected trial: {best_trial_key} | epochs={selected["epochs"]} | lr={selected["lr"]} | '
    f'bs={selected["batch_size"]} | mean Shift-Normalized Accuracy Gain={selected["mean_shift_normalized_accuracy_gain"]:.6f}'
)

np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)

for trial_key, trial in experiment_data["synthetic_reasoning_stream"]["trials"].items():
    np.save(
        os.path.join(working_dir, f"{trial_key}_train_metrics.npy"),
        np.array(trial["metrics"]["train"], dtype=object),
        allow_pickle=True,
    )
    np.save(
        os.path.join(working_dir, f"{trial_key}_val_metrics.npy"),
        np.array(trial["metrics"]["val"], dtype=object),
        allow_pickle=True,
    )
    np.save(
        os.path.join(working_dir, f"{trial_key}_train_losses.npy"),
        np.array(trial["losses"]["train"], dtype=object),
        allow_pickle=True,
    )
    np.save(
        os.path.join(working_dir, f"{trial_key}_val_losses.npy"),
        np.array(trial["losses"]["val"], dtype=object),
        allow_pickle=True,
    )
    np.save(
        os.path.join(working_dir, f"{trial_key}_timestamps.npy"),
        np.array(trial["timestamps"]),
        allow_pickle=True,
    )

    for ds_name, ds_trial in trial["per_dataset"].items():
        prefix = f"{trial_key}_{ds_name}"
        np.save(
            os.path.join(working_dir, f"{prefix}_frozen_preds.npy"),
            ds_trial["arrays"]["frozen_preds"],
        )
        np.save(
            os.path.join(working_dir, f"{prefix}_always_preds.npy"),
            ds_trial["arrays"]["always_preds"],
        )
        np.save(
            os.path.join(working_dir, f"{prefix}_gated_preds.npy"),
            ds_trial["arrays"]["gated_preds"],
        )
        np.save(
            os.path.join(working_dir, f"{prefix}_ground_truth.npy"),
            ds_trial["arrays"]["ground_truth"],
        )
        np.save(
            os.path.join(working_dir, f"{prefix}_gated_entropy.npy"),
            ds_trial["arrays"]["gated_entropy"],
        )
        np.save(
            os.path.join(working_dir, f"{prefix}_gated_margin.npy"),
            ds_trial["arrays"]["gated_margin"],
        )
        np.save(
            os.path.join(working_dir, f"{prefix}_gated_triggered.npy"),
            ds_trial["arrays"]["gated_triggered"],
        )
        np.save(
            os.path.join(working_dir, f"{prefix}_frozen_cumacc.npy"),
            ds_trial["arrays"]["frozen_cumacc"],
        )
        np.save(
            os.path.join(working_dir, f"{prefix}_always_cumacc.npy"),
            ds_trial["arrays"]["always_cumacc"],
        )
        np.save(
            os.path.join(working_dir, f"{prefix}_gated_cumacc.npy"),
            ds_trial["arrays"]["gated_cumacc"],
        )
        np.save(
            os.path.join(working_dir, f"{prefix}_sna_gain.npy"),
            np.array(
                [ds_trial["test_metrics"]["shift_normalized_accuracy_gain"]],
                dtype=np.float32,
            ),
        )

np.save(
    os.path.join(working_dir, "trial_scores.npy"),
    np.array(all_scores, dtype=object),
    allow_pickle=True,
)

plt.figure(figsize=(8, 5))
for trial_key, trial in experiment_data["synthetic_reasoning_stream"]["trials"].items():
    x = [e for e, _ in trial["losses"]["val"]]
    y = [v for _, v in trial["losses"]["val"]]
    plt.plot(x, y, label=f'{trial_key} (best={trial["best_val_loss"]:.3f})')
plt.xlabel("Epoch")
plt.ylabel("Validation loss")
plt.title("Validation Loss Across Hyperparameter Trials")
plt.legend(fontsize=7)
plt.tight_layout()
plt.savefig(os.path.join(working_dir, "baseline_tuning_val_loss.png"))
plt.close()

scores_sorted = sorted(all_scores, key=lambda z: z[3], reverse=True)
labels = [f"e{e}|lr{lr}|bs{bs}" for e, lr, bs, _ in scores_sorted]
vals = [s for _, _, _, s in scores_sorted]
plt.figure(figsize=(10, 5))
plt.bar(range(len(vals)), vals)
plt.xticks(range(len(vals)), labels, rotation=45, ha="right")
plt.ylabel("Mean Shift-Normalized Accuracy Gain")
plt.title("Hyperparameter Tuning by Target Metric")
plt.tight_layout()
plt.savefig(os.path.join(working_dir, "baseline_tuning_sna_gain.png"))
plt.close()

sel_key = best_trial_key
for ds_name in [
    "synthetic_reasoning_stream",
    "feature_permutation_stream",
    "nonlinear_boundary_stream",
]:
    arr = selected["per_dataset"][ds_name]["arrays"]
    frozen_acc = selected["per_dataset"][ds_name]["stream_results"]["frozen"][
        "Shifted-Stream Accuracy"
    ]
    always_acc = selected["per_dataset"][ds_name]["stream_results"]["always"][
        "Shifted-Stream Accuracy"
    ]
    gated_acc = selected["per_dataset"][ds_name]["stream_results"]["gated"][
        "Shifted-Stream Accuracy"
    ]

    plt.figure(figsize=(8, 5))
    plt.plot(arr["frozen_cumacc"], label=f"Frozen ({frozen_acc:.3f})")
    plt.plot(arr["always_cumacc"], label=f"Always-LoRA ({always_acc:.3f})")
    plt.plot(arr["gated_cumacc"], label=f"Gated-LoRA ({gated_acc:.3f})")
    plt.xlabel("Stream step")
    plt.ylabel("Cumulative accuracy")
    plt.title(f"Cumulative Accuracy: {ds_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, f"{ds_name}_cumulative_accuracy.png"))
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(arr["gated_entropy"], label="Entropy")
    plt.plot(
        arr["gated_triggered"] * max(arr["gated_entropy"].max(), 1e-6),
        label="Triggered (scaled)",
        alpha=0.7,
    )
    plt.xlabel("Stream step")
    plt.ylabel("Value")
    plt.title(f"Gating Signals: {ds_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, f"{ds_name}_gating_signals.png"))
    plt.close()

print("Finished. Saved metrics, arrays, and plots to:", working_dir)
