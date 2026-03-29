import os
import time
import copy
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

experiment_data = {
    "batch_size_tuning": {
        "synthetic_vlm_stream": {
            "config": {
                "batch_sizes": [32, 64, 128],
                "epochs": 15,
                "optimizer": "Adam",
                "lr": 1e-3,
                "weight_decay": 1e-4,
                "seed_torch": 7,
                "seed_numpy": 7,
            },
            "trials": {},
            "summary": {
                "ranking": [],
                "best_batch_size_by_val": None,
                "best_batch_size_by_stream_consistency": None,
            },
        }
    }
}


class SyntheticWorld:
    def __init__(self, dim=16, num_classes=4, seed=7):
        rng = np.random.RandomState(seed)
        self.dim = dim
        self.num_classes = num_classes
        self.W_img = rng.randn(num_classes, dim).astype(np.float32) * 1.2
        self.W_cap = rng.randn(num_classes, dim).astype(np.float32) * 0.8
        self.W_rat = rng.randn(num_classes, dim).astype(np.float32) * 0.8


class SyntheticVLMDataset(Dataset):
    def __init__(self, world, n=2000, shift=False, seed=0):
        self.world = world
        self.n = n
        self.shift = shift
        self.rng = np.random.RandomState(seed)
        self.data = []
        dim = world.dim
        K = world.num_classes

        for i in range(n):
            y = self.rng.randint(0, K)
            z = self.rng.randn(dim).astype(np.float32)

            img = z + world.W_img[y] + 0.35 * self.rng.randn(dim).astype(np.float32)
            cap = (
                0.7 * z + world.W_cap[y] + 0.45 * self.rng.randn(dim).astype(np.float32)
            )
            rat = (
                0.5 * z + world.W_rat[y] + 0.50 * self.rng.randn(dim).astype(np.float32)
            )

            if shift:
                img = (
                    0.60 * img
                    + 0.85 * np.roll(img, 1)
                    + 0.40 * self.rng.randn(dim).astype(np.float32)
                )
                if i % 7 == 0:
                    img = img + 1.0 * self.rng.randn(dim).astype(np.float32)

            self.data.append(
                (
                    img.astype(np.float32),
                    cap.astype(np.float32),
                    rat.astype(np.float32),
                    int(y),
                )
            )

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        img, cap, rat, y = self.data[idx]
        return {
            "image": torch.tensor(img, dtype=torch.float32),
            "caption": torch.tensor(cap, dtype=torch.float32),
            "rationale": torch.tensor(rat, dtype=torch.float32),
            "label": torch.tensor(y, dtype=torch.long),
        }


class FeatureNormalizer:
    def __init__(self, mean_dict, std_dict):
        self.mean = {
            k: torch.tensor(v, dtype=torch.float32).to(device)
            for k, v in mean_dict.items()
        }
        self.std = {
            k: torch.tensor(v, dtype=torch.float32).to(device)
            for k, v in std_dict.items()
        }

    @classmethod
    def from_dataset(cls, dataset):
        stats = {}
        for key in ["image", "caption", "rationale"]:
            arr = np.stack([sample[key].numpy() for sample in dataset], axis=0)
            stats[key] = (arr.mean(axis=0), arr.std(axis=0) + 1e-6)
        mean_dict = {k: v[0] for k, v in stats.items()}
        std_dict = {k: v[1] for k, v in stats.items()}
        return cls(mean_dict, std_dict)

    def normalize(self, batch):
        out = {}
        for k, v in batch.items():
            out[k] = v.to(device) if isinstance(v, torch.Tensor) else v
        for key in ["image", "caption", "rationale"]:
            out[key] = (out[key] - self.mean[key]) / self.std[key]
        return out


class SmallVLM(nn.Module):
    def __init__(self, dim=16, hidden=64, num_classes=4):
        super().__init__()
        self.img_backbone = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU()
        )
        self.caption_head = nn.Sequential(
            nn.Linear(dim, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, num_classes)
        )
        self.rationale_head = nn.Sequential(
            nn.Linear(dim, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, num_classes)
        )
        self.adapter = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, num_classes)
        )

    def forward(self, image, caption, rationale):
        h = self.img_backbone(image)
        logits_main = self.adapter(h)
        logits_cap = self.caption_head(caption)
        logits_rat = self.rationale_head(rationale)
        return logits_main, logits_cap, logits_rat


def supervised_loss(logits_main, logits_cap, logits_rat, y):
    return (
        F.cross_entropy(logits_main, y)
        + 0.35 * F.cross_entropy(logits_cap, y)
        + 0.35 * F.cross_entropy(logits_rat, y)
    )


def evaluate_loader(model, loader, normalizer):
    model.eval()
    total_loss, total, correct = 0.0, 0, 0
    preds_all, gt_all = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            batch = normalizer.normalize(batch)
            lm, lc, lr = model(batch["image"], batch["caption"], batch["rationale"])
            loss = supervised_loss(lm, lc, lr, batch["label"])
            pred = lm.argmax(dim=-1)
            bs = batch["label"].size(0)
            total_loss += loss.item() * bs
            total += bs
            correct += (pred == batch["label"]).sum().item()
            preds_all.extend(pred.detach().cpu().numpy().tolist())
            gt_all.extend(batch["label"].detach().cpu().numpy().tolist())
    return total_loss / total, correct / total, np.array(preds_all), np.array(gt_all)


def entropy_min_loss(logits):
    p = F.softmax(logits, dim=-1)
    return -(p * torch.log(p + 1e-8)).sum(dim=-1).mean()


def consistency_alignment_loss(logits_main, logits_cap, logits_rat, temperature=1.0):
    log_p_main = F.log_softmax(logits_main / temperature, dim=-1)
    p_cap = F.softmax(logits_cap.detach() / temperature, dim=-1)
    p_rat = F.softmax(logits_rat.detach() / temperature, dim=-1)
    target = 0.5 * (p_cap + p_rat)
    return F.kl_div(log_p_main, target, reduction="batchmean")


def stream_eval_with_tta(
    model_init, stream_loader, normalizer, mode="frozen", lr=2e-3, conf_thresh=0.70
):
    model = copy.deepcopy(model_init).to(device)
    for p in model.img_backbone.parameters():
        p.requires_grad = False
    for p in model.caption_head.parameters():
        p.requires_grad = False
    for p in model.rationale_head.parameters():
        p.requires_grad = False
    for p in model.adapter.parameters():
        p.requires_grad = True
    optimizer = torch.optim.Adam(model.adapter.parameters(), lr=lr)

    preds, gts, update_flags, confidences, losses = [], [], [], [], []
    model.eval()

    for _, batch in enumerate(stream_loader):
        batch = {
            k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
        }
        batch = normalizer.normalize(batch)

        with torch.no_grad():
            lm, lc, lrh = model(batch["image"], batch["caption"], batch["rationale"])
            p_main = F.softmax(lm, dim=-1)
            p_cap = F.softmax(lc, dim=-1)
            p_rat = F.softmax(lrh, dim=-1)
            conf, pred = p_main.max(dim=-1)
            pred_cap = p_cap.argmax(dim=-1)
            pred_rat = p_rat.argmax(dim=-1)

            js_like = 0.5 * (
                F.kl_div(torch.log(p_main + 1e-8), p_cap, reduction="batchmean")
                + F.kl_div(torch.log(p_main + 1e-8), p_rat, reduction="batchmean")
            )
            agree = (pred == pred_cap) & (pred == pred_rat)
            consistency_score = float(torch.exp(-js_like).item())

            preds.append(int(pred.item()))
            gts.append(int(batch["label"].item()))
            confidences.append(float(conf.item()))

        do_update = False
        if mode == "entropy":
            do_update = True
        elif mode == "consistency":
            do_update = bool(
                (conf > conf_thresh).item()
                and agree.item()
                and consistency_score > 0.65
            )

        if do_update:
            model.train()
            optimizer.zero_grad()
            lm2, lc2, lr2 = model(batch["image"], batch["caption"], batch["rationale"])
            if mode == "entropy":
                loss = entropy_min_loss(lm2)
            else:
                loss = consistency_alignment_loss(
                    lm2, lc2, lr2
                ) + 0.1 * entropy_min_loss(lm2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.adapter.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
            update_flags.append(1)
            model.eval()
        else:
            losses.append(0.0)
            update_flags.append(0)

    preds = np.array(preds)
    gts = np.array(gts)
    update_flags = np.array(update_flags)
    confidences = np.array(confidences)
    acc = float((preds == gts).mean())

    correctness = (preds == gts).astype(np.float32)
    w = 25
    rolling = np.array(
        [correctness[max(0, i - w + 1) : i + 1].mean() for i in range(len(correctness))]
    )
    penalties = []
    for i in np.where(update_flags == 1)[0]:
        pre = rolling[max(0, i - 1)]
        post = rolling[min(len(rolling) - 1, i + 5)]
        penalties.append(max(0.0, pre - post))
    volatility_penalty = float(np.mean(penalties)) if penalties else 0.0
    stability_adjusted_acc = acc - 0.5 * volatility_penalty

    return {
        "acc": acc,
        "stability_adjusted_acc": stability_adjusted_acc,
        "preds": preds,
        "gts": gts,
        "updates": update_flags,
        "confidences": confidences,
        "rolling_acc": rolling,
        "adapt_loss_mean": float(np.mean(losses)),
        "adapt_freq": float(update_flags.mean()),
    }


def set_all_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_trial(batch_size, train_ds, val_ds, stream_ds, normalizer, epochs=15):
    set_all_seeds(7)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)
    stream_loader = DataLoader(stream_ds, batch_size=1, shuffle=False)

    model = SmallVLM(dim=16, hidden=64, num_classes=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    trial_data = {
        "metrics": {"train": [], "val": [], "test": []},
        "losses": {"train": [], "val": []},
        "predictions": {},
        "ground_truth": [],
        "stream": {},
        "best_val_loss": None,
        "best_epoch": None,
        "batch_size": batch_size,
    }

    best_state = None
    best_val = float("inf")
    best_epoch = -1

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total, correct = 0.0, 0, 0

        for batch in train_loader:
            batch = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            batch = normalizer.normalize(batch)

            optimizer.zero_grad()
            lm, lc, lrh = model(batch["image"], batch["caption"], batch["rationale"])
            loss = supervised_loss(lm, lc, lrh, batch["label"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            pred = lm.argmax(dim=-1)
            bs = batch["label"].size(0)
            total_loss += loss.item() * bs
            total += bs
            correct += (pred == batch["label"]).sum().item()

        train_loss = total_loss / total
        train_acc = correct / total
        val_loss, val_acc, _, _ = evaluate_loader(model, val_loader, normalizer)

        print(
            f"[batch_size={batch_size}] Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

        ts = time.time()
        trial_data["losses"]["train"].append((epoch, train_loss, ts))
        trial_data["losses"]["val"].append((epoch, val_loss, ts))
        trial_data["metrics"]["train"].append((epoch, train_acc, ts))
        trial_data["metrics"]["val"].append((epoch, val_acc, ts))

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    trial_data["best_val_loss"] = float(best_val)
    trial_data["best_epoch"] = int(best_epoch)

    frozen_res = stream_eval_with_tta(model, stream_loader, normalizer, mode="frozen")
    entropy_res = stream_eval_with_tta(model, stream_loader, normalizer, mode="entropy")
    consistency_res = stream_eval_with_tta(
        model, stream_loader, normalizer, mode="consistency"
    )
    results = {
        "frozen": frozen_res,
        "entropy": entropy_res,
        "consistency": consistency_res,
    }

    for name, res in results.items():
        print(
            f"[batch_size={batch_size}] {name}: "
            f"acc={res['acc']:.4f}, "
            f"stability_adjusted_acc={res['stability_adjusted_acc']:.4f}, "
            f"adapt_freq={res['adapt_freq']:.4f}, "
            f"adapt_loss_mean={res['adapt_loss_mean']:.4f}"
        )
        trial_data["metrics"]["test"].append(
            {
                "name": name,
                "acc": res["acc"],
                "stability_adjusted_acc": res["stability_adjusted_acc"],
                "adapt_freq": res["adapt_freq"],
                "adapt_loss_mean": res["adapt_loss_mean"],
                "timestamp": time.time(),
            }
        )

    trial_data["predictions"] = {
        "frozen": frozen_res["preds"],
        "entropy": entropy_res["preds"],
        "consistency": consistency_res["preds"],
    }
    trial_data["ground_truth"] = frozen_res["gts"]
    trial_data["stream"] = {
        "frozen_updates": frozen_res["updates"],
        "entropy_updates": entropy_res["updates"],
        "consistency_updates": consistency_res["updates"],
        "frozen_rolling_acc": frozen_res["rolling_acc"],
        "entropy_rolling_acc": entropy_res["rolling_acc"],
        "consistency_rolling_acc": consistency_res["rolling_acc"],
        "frozen_confidences": frozen_res["confidences"],
        "entropy_confidences": entropy_res["confidences"],
        "consistency_confidences": consistency_res["confidences"],
    }
    return trial_data


# Data
world = SyntheticWorld(dim=16, num_classes=4, seed=7)
train_ds = SyntheticVLMDataset(world, n=2500, shift=False, seed=11)
val_ds = SyntheticVLMDataset(world, n=500, shift=False, seed=13)
test_stream_ds = SyntheticVLMDataset(world, n=700, shift=True, seed=17)
normalizer = FeatureNormalizer.from_dataset(train_ds)

batch_sizes = [32, 64, 128]
all_summaries = []

for bs in batch_sizes:
    trial_key = f"batch_size_{bs}"
    trial_res = train_one_trial(
        bs, train_ds, val_ds, test_stream_ds, normalizer, epochs=15
    )
    experiment_data["batch_size_tuning"]["synthetic_vlm_stream"]["trials"][
        trial_key
    ] = trial_res

    test_metrics = {m["name"]: m for m in trial_res["metrics"]["test"]}
    summary_row = {
        "batch_size": bs,
        "best_val_loss": trial_res["best_val_loss"],
        "best_epoch": trial_res["best_epoch"],
        "best_val_acc": float(max([x[1] for x in trial_res["metrics"]["val"]])),
        "frozen_acc": test_metrics["frozen"]["acc"],
        "frozen_stability_adjusted_acc": test_metrics["frozen"][
            "stability_adjusted_acc"
        ],
        "entropy_acc": test_metrics["entropy"]["acc"],
        "entropy_stability_adjusted_acc": test_metrics["entropy"][
            "stability_adjusted_acc"
        ],
        "consistency_acc": test_metrics["consistency"]["acc"],
        "consistency_stability_adjusted_acc": test_metrics["consistency"][
            "stability_adjusted_acc"
        ],
    }
    all_summaries.append(summary_row)

ranking = sorted(
    all_summaries,
    key=lambda x: (x["best_val_loss"], -x["consistency_stability_adjusted_acc"]),
)
best_by_val = ranking[0]["batch_size"]
best_by_stream = sorted(
    all_summaries, key=lambda x: -x["consistency_stability_adjusted_acc"]
)[0]["batch_size"]

experiment_data["batch_size_tuning"]["synthetic_vlm_stream"]["summary"][
    "ranking"
] = ranking
experiment_data["batch_size_tuning"]["synthetic_vlm_stream"]["summary"][
    "best_batch_size_by_val"
] = best_by_val
experiment_data["batch_size_tuning"]["synthetic_vlm_stream"]["summary"][
    "best_batch_size_by_stream_consistency"
] = best_by_stream

print("\nBatch size tuning summary:")
for row in ranking:
    print(
        f"batch_size={row['batch_size']}, "
        f"best_val_loss={row['best_val_loss']:.4f}, "
        f"consistency_stability_adjusted_acc={row['consistency_stability_adjusted_acc']:.4f}, "
        f"entropy_stability_adjusted_acc={row['entropy_stability_adjusted_acc']:.4f}, "
        f"frozen_stability_adjusted_acc={row['frozen_stability_adjusted_acc']:.4f}"
    )

print(f"\nSelected by validation loss: batch_size={best_by_val}")
print(
    f"Best downstream consistency stability-adjusted accuracy: batch_size={best_by_stream}"
)

# Save all plottable arrays per trial
for bs in batch_sizes:
    trial_key = f"batch_size_{bs}"
    d = experiment_data["batch_size_tuning"]["synthetic_vlm_stream"]["trials"][
        trial_key
    ]

    np.save(
        os.path.join(working_dir, f"{trial_key}_train_losses.npy"),
        np.array(d["losses"]["train"], dtype=object),
    )
    np.save(
        os.path.join(working_dir, f"{trial_key}_val_losses.npy"),
        np.array(d["losses"]["val"], dtype=object),
    )
    np.save(
        os.path.join(working_dir, f"{trial_key}_train_metrics.npy"),
        np.array(d["metrics"]["train"], dtype=object),
    )
    np.save(
        os.path.join(working_dir, f"{trial_key}_val_metrics.npy"),
        np.array(d["metrics"]["val"], dtype=object),
    )
    np.save(
        os.path.join(working_dir, f"{trial_key}_test_metrics.npy"),
        np.array(d["metrics"]["test"], dtype=object),
    )

    np.save(
        os.path.join(working_dir, f"{trial_key}_preds_frozen.npy"),
        d["predictions"]["frozen"],
    )
    np.save(
        os.path.join(working_dir, f"{trial_key}_preds_entropy.npy"),
        d["predictions"]["entropy"],
    )
    np.save(
        os.path.join(working_dir, f"{trial_key}_preds_consistency.npy"),
        d["predictions"]["consistency"],
    )
    np.save(
        os.path.join(working_dir, f"{trial_key}_ground_truth.npy"), d["ground_truth"]
    )

    np.save(
        os.path.join(working_dir, f"{trial_key}_frozen_updates.npy"),
        d["stream"]["frozen_updates"],
    )
    np.save(
        os.path.join(working_dir, f"{trial_key}_entropy_updates.npy"),
        d["stream"]["entropy_updates"],
    )
    np.save(
        os.path.join(working_dir, f"{trial_key}_consistency_updates.npy"),
        d["stream"]["consistency_updates"],
    )

    np.save(
        os.path.join(working_dir, f"{trial_key}_frozen_rolling_acc.npy"),
        d["stream"]["frozen_rolling_acc"],
    )
    np.save(
        os.path.join(working_dir, f"{trial_key}_entropy_rolling_acc.npy"),
        d["stream"]["entropy_rolling_acc"],
    )
    np.save(
        os.path.join(working_dir, f"{trial_key}_consistency_rolling_acc.npy"),
        d["stream"]["consistency_rolling_acc"],
    )

    np.save(
        os.path.join(working_dir, f"{trial_key}_frozen_confidences.npy"),
        d["stream"]["frozen_confidences"],
    )
    np.save(
        os.path.join(working_dir, f"{trial_key}_entropy_confidences.npy"),
        d["stream"]["entropy_confidences"],
    )
    np.save(
        os.path.join(working_dir, f"{trial_key}_consistency_confidences.npy"),
        d["stream"]["consistency_confidences"],
    )

# Save summary arrays
summary_arr = np.array(
    [
        [
            r["batch_size"],
            r["best_val_loss"],
            r["best_epoch"],
            r["best_val_acc"],
            r["frozen_acc"],
            r["frozen_stability_adjusted_acc"],
            r["entropy_acc"],
            r["entropy_stability_adjusted_acc"],
            r["consistency_acc"],
            r["consistency_stability_adjusted_acc"],
        ]
        for r in ranking
    ],
    dtype=float,
)
np.save(os.path.join(working_dir, "batch_size_tuning_summary.npy"), summary_arr)

# Visualization 1: validation loss across batch sizes
plt.figure(figsize=(9, 4))
for bs in batch_sizes:
    trial_key = f"batch_size_{bs}"
    val_curve = np.array(
        [
            [x[0], x[1]]
            for x in experiment_data["batch_size_tuning"]["synthetic_vlm_stream"][
                "trials"
            ][trial_key]["losses"]["val"]
        ],
        dtype=float,
    )
    plt.plot(val_curve[:, 0], val_curve[:, 1], marker="o", label=f"bs={bs}")
plt.xlabel("Epoch")
plt.ylabel("Validation loss")
plt.title("Validation loss by training batch size")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(working_dir, "batch_size_val_loss_comparison.png"), dpi=160)
plt.close()

# Visualization 2: train loss across batch sizes
plt.figure(figsize=(9, 4))
for bs in batch_sizes:
    trial_key = f"batch_size_{bs}"
    tr_curve = np.array(
        [
            [x[0], x[1]]
            for x in experiment_data["batch_size_tuning"]["synthetic_vlm_stream"][
                "trials"
            ][trial_key]["losses"]["train"]
        ],
        dtype=float,
    )
    plt.plot(tr_curve[:, 0], tr_curve[:, 1], marker="o", label=f"bs={bs}")
plt.xlabel("Epoch")
plt.ylabel("Train loss")
plt.title("Train loss by training batch size")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(working_dir, "batch_size_train_loss_comparison.png"), dpi=160)
plt.close()

# Visualization 3: summary bar chart for consistency stability-adjusted accuracy
plt.figure(figsize=(8, 4))
xs = np.arange(len(batch_sizes))
vals = []
for bs in batch_sizes:
    row = [r for r in all_summaries if r["batch_size"] == bs][0]
    vals.append(row["consistency_stability_adjusted_acc"])
plt.bar(xs, vals, tick_label=[str(bs) for bs in batch_sizes])
plt.xlabel("Training batch size")
plt.ylabel("Consistency stability-adjusted acc")
plt.title("Downstream stream performance by batch size")
plt.tight_layout()
plt.savefig(
    os.path.join(working_dir, "batch_size_consistency_stream_summary.png"), dpi=160
)
plt.close()

# Visualization 4: rolling accuracy under consistency TTA for each batch size
plt.figure(figsize=(10, 5))
for bs in batch_sizes:
    trial_key = f"batch_size_{bs}"
    rolling = experiment_data["batch_size_tuning"]["synthetic_vlm_stream"]["trials"][
        trial_key
    ]["stream"]["consistency_rolling_acc"]
    plt.plot(np.arange(len(rolling)), rolling, linewidth=2, label=f"bs={bs}")
plt.xlabel("Stream step")
plt.ylabel("Rolling accuracy")
plt.title("Consistency-gated TTA rolling accuracy by source train batch size")
plt.legend()
plt.tight_layout()
plt.savefig(
    os.path.join(working_dir, "batch_size_consistency_rolling_accuracy.png"), dpi=160
)
plt.close()

# Visualization 5: prefix predictions for best validation model
best_trial_key = f"batch_size_{best_by_val}"
best_trial = experiment_data["batch_size_tuning"]["synthetic_vlm_stream"]["trials"][
    best_trial_key
]
n_show = 60
plt.figure(figsize=(10, 4))
plt.plot(best_trial["ground_truth"][:n_show], label="Ground Truth", marker="o")
plt.plot(best_trial["predictions"]["frozen"][:n_show], label="Frozen Pred", marker="x")
plt.plot(
    best_trial["predictions"]["consistency"][:n_show],
    label="Consistency Pred",
    marker="s",
)
plt.yticks(sorted(np.unique(best_trial["ground_truth"])))
plt.xlabel("Sample index")
plt.ylabel("Class")
plt.title(f"Best-val batch size={best_by_val}: stream prefix predictions")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(working_dir, "best_batch_size_prediction_prefix.png"), dpi=160)
plt.close()

# Final required save
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f"\nSaved experiment data to: {os.path.join(working_dir, 'experiment_data.npy')}")
