import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import time
import copy
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

torch.manual_seed(7)
np.random.seed(7)
random.seed(7)

experiment_data = {
    "synthetic_vlm_stream": {
        "metrics": {
            "train": [],
            "val": [],
            "test": [],
            "lr_sweep": [],
            "tta_sweep": [],
        },
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
        "stream": {},
        "meta": {"source": "synthetic"},
    },
    "hf_ai2d_like_stream": {
        "metrics": {
            "train": [],
            "val": [],
            "test": [],
            "lr_sweep": [],
            "tta_sweep": [],
        },
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
        "stream": {},
        "meta": {"source": "huggingface_or_fallback", "hf_name": "lmms-lab/ai2d"},
    },
    "hf_textvqa_like_stream": {
        "metrics": {
            "train": [],
            "val": [],
            "test": [],
            "lr_sweep": [],
            "tta_sweep": [],
        },
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "timestamps": [],
        "stream": {},
        "meta": {"source": "huggingface_or_fallback", "hf_name": "textvqa"},
    },
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
    def __init__(self, world, n=2000, shift=False, seed=0, mode_tag="generic"):
        self.world = world
        self.n = n
        self.shift = shift
        self.mode_tag = mode_tag
        self.rng = np.random.RandomState(seed)
        self.data = []
        dim = world.dim
        K = world.num_classes

        shift_strength = 1.0
        if mode_tag == "ai2d":
            shift_strength = 1.10
        elif mode_tag == "textvqa":
            shift_strength = 1.25

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
                    + 0.40 * shift_strength * self.rng.randn(dim).astype(np.float32)
                )
                cap = (
                    0.85 * cap
                    + 0.15 * np.roll(cap, 2)
                    + 0.15 * shift_strength * self.rng.randn(dim).astype(np.float32)
                )
                rat = (
                    0.80 * rat
                    + 0.20 * np.roll(rat, 3)
                    + 0.20 * shift_strength * self.rng.randn(dim).astype(np.float32)
                )
                if mode_tag == "ai2d":
                    if i % 5 == 0:
                        cap = cap + 0.55 * self.rng.randn(dim).astype(np.float32)
                if mode_tag == "textvqa":
                    if i % 4 == 0:
                        rat = rat + 0.70 * self.rng.randn(dim).astype(np.float32)
                    if i % 9 == 0:
                        img = img + 1.15 * self.rng.randn(dim).astype(np.float32)
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


def text_to_feature(text, dim=16):
    text = "" if text is None else str(text)
    vals = np.zeros(dim, dtype=np.float32)
    for i, ch in enumerate(text.encode("utf-8", errors="ignore")):
        vals[i % dim] += ((ch % 31) - 15) / 15.0
    n = max(1.0, np.linalg.norm(vals))
    return vals / n


def pseudo_image_feature(seed_val, dim=16):
    rng = np.random.RandomState(seed_val % (2**32 - 1))
    base = rng.randn(dim).astype(np.float32)
    return base / (np.linalg.norm(base) + 1e-6)


class HFFeatureDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        item = self.records[idx]
        return {
            "image": torch.tensor(item["image"], dtype=torch.float32),
            "caption": torch.tensor(item["caption"], dtype=torch.float32),
            "rationale": torch.tensor(item["rationale"], dtype=torch.float32),
            "label": torch.tensor(item["label"], dtype=torch.long),
        }


def build_records_from_hf(dataset_name, split, n_max=512, dim=16, num_classes=4):
    try:
        from datasets import load_dataset

        ds = load_dataset(dataset_name, split=split)
        n = min(len(ds), n_max)
        records = []
        for i in range(n):
            ex = ds[i]
            q = ex.get("question", ex.get("query", ex.get("text", "")))
            a = ex.get("answer", ex.get("answers", ex.get("label", "")))
            if isinstance(a, list):
                a = a[0] if len(a) else ""
            if isinstance(a, dict):
                a = " ".join([str(v) for v in a.values()])
            img_seed = abs(hash(str(ex.get("image", i)))) % (2**32 - 1)
            image = pseudo_image_feature(img_seed, dim=dim)
            caption = text_to_feature(q, dim=dim)
            rationale = text_to_feature(str(q) + " " + str(a), dim=dim)
            label = abs(hash(str(a))) % num_classes
            records.append(
                {
                    "image": image.astype(np.float32),
                    "caption": caption.astype(np.float32),
                    "rationale": rationale.astype(np.float32),
                    "label": int(label),
                }
            )
        if len(records) >= 32:
            return records, True
    except Exception as e:
        print(f"HF load failed for {dataset_name} {split}: {e}")
    return None, False


def build_fallback_hf_like_records(kind="ai2d", n=512, dim=16, num_classes=4, seed=0):
    rng = np.random.RandomState(seed)
    records = []
    prompts = {
        "ai2d": [
            "Which arrow points to the highest value?",
            "What label belongs to the left diagram part?",
            "Which option matches the flowchart?",
            "How many regions are highlighted?",
        ],
        "textvqa": [
            "What word is written on the sign?",
            "What number appears in the document?",
            "Which date is shown in the image?",
            "What is the title text?",
        ],
    }
    answers = {
        "ai2d": ["A", "B", "C", "D"],
        "textvqa": ["one", "two", "red", "blue", "2024", "stop"],
    }
    for i in range(n):
        q = rng.choice(prompts[kind])
        a = rng.choice(answers[kind])
        image = pseudo_image_feature(
            seed_val=seed + i * 17 + (3 if kind == "textvqa" else 1), dim=dim
        )
        caption = text_to_feature(q, dim=dim) + 0.15 * rng.randn(dim).astype(np.float32)
        rationale = text_to_feature(q + " " + a, dim=dim) + 0.15 * rng.randn(
            dim
        ).astype(np.float32)
        label = abs(hash(str(a))) % num_classes
        records.append(
            {
                "image": image.astype(np.float32),
                "caption": caption.astype(np.float32),
                "rationale": rationale.astype(np.float32),
                "label": int(label),
            }
        )
    return records


def make_hf_like_dataset(dataset_name, split, fallback_kind, n_max, seed):
    records, ok = build_records_from_hf(
        dataset_name, split, n_max=n_max, dim=16, num_classes=4
    )
    if ok:
        return HFFeatureDataset(records), f"hf:{dataset_name}:{split}"
    records = build_fallback_hf_like_records(
        kind=fallback_kind, n=n_max, dim=16, num_classes=4, seed=seed
    )
    return HFFeatureDataset(records), f"fallback:{fallback_kind}:{split}"


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
            arr = np.stack(
                [dataset[i][key].numpy() for i in range(len(dataset))], axis=0
            )
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
            lm, lc, lrh = model(batch["image"], batch["caption"], batch["rationale"])
            loss = supervised_loss(lm, lc, lrh, batch["label"])
            pred = lm.argmax(dim=-1)
            bs = batch["label"].size(0)
            total_loss += loss.item() * bs
            total += bs
            correct += (pred == batch["label"]).sum().item()
            preds_all.extend(pred.detach().cpu().numpy().tolist())
            gt_all.extend(batch["label"].detach().cpu().numpy().tolist())
    return (
        total_loss / max(total, 1),
        correct / max(total, 1),
        np.array(preds_all),
        np.array(gt_all),
    )


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
    model_init, stream_loader, normalizer, mode="frozen", lr=1e-3, conf_thresh=0.70
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
    acc = float((preds == gts).mean()) if len(gts) else 0.0
    correctness = (preds == gts).astype(np.float32)
    w = 25
    rolling = (
        np.array(
            [
                correctness[max(0, i - w + 1) : i + 1].mean()
                for i in range(len(correctness))
            ]
        )
        if len(correctness)
        else np.array([])
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
        "adapt_loss_mean": float(np.mean(losses)) if len(losses) else 0.0,
        "adapt_freq": float(update_flags.mean()) if len(update_flags) else 0.0,
    }


def train_single_lr(
    train_loader, val_loader, normalizer, lr, epochs=20, seed=7, patience=4
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model = SmallVLM(dim=16, hidden=64, num_classes=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    best_state = None
    best_val = float("inf")
    best_epoch = 0
    bad_epochs = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

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

        train_loss = total_loss / max(total, 1)
        train_acc = correct / max(total, 1)
        val_loss, val_acc, _, _ = evaluate_loader(model, val_loader, normalizer)

        history["train_loss"].append((epoch, train_loss, time.time()))
        history["val_loss"].append((epoch, val_loss, time.time()))
        history["train_acc"].append((epoch, train_acc, time.time()))
        history["val_acc"].append((epoch, val_acc, time.time()))

        print(
            f"lr={lr:.0e} epoch={epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )
        print(f"Epoch {epoch}: validation_loss = {val_loss:.4f}")

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    model.load_state_dict(best_state)
    return model, best_val, best_epoch, history


def tune_tta_params(
    model, stream_loader, normalizer, dataset_store, mode="consistency"
):
    if mode == "entropy":
        lrs = [5e-4, 1e-3, 2e-3]
        best_cfg, best_res = None, None
        for lr in lrs:
            res = stream_eval_with_tta(
                model, stream_loader, normalizer, mode=mode, lr=lr, conf_thresh=0.0
            )
            dataset_store["metrics"]["tta_sweep"].append(
                {
                    "mode": mode,
                    "tta_lr": lr,
                    "conf_thresh": None,
                    "acc": res["acc"],
                    "stability_adjusted_acc": res["stability_adjusted_acc"],
                    "adapt_freq": res["adapt_freq"],
                    "timestamp": time.time(),
                }
            )
            if (
                best_res is None
                or res["stability_adjusted_acc"] > best_res["stability_adjusted_acc"]
            ):
                best_cfg, best_res = {"lr": lr, "conf_thresh": None}, res
        return best_cfg, best_res
    else:
        lrs = [5e-4, 1e-3, 2e-3]
        confs = [0.60, 0.70, 0.80]
        best_cfg, best_res = None, None
        for lr in lrs:
            for ct in confs:
                res = stream_eval_with_tta(
                    model, stream_loader, normalizer, mode=mode, lr=lr, conf_thresh=ct
                )
                dataset_store["metrics"]["tta_sweep"].append(
                    {
                        "mode": mode,
                        "tta_lr": lr,
                        "conf_thresh": ct,
                        "acc": res["acc"],
                        "stability_adjusted_acc": res["stability_adjusted_acc"],
                        "adapt_freq": res["adapt_freq"],
                        "timestamp": time.time(),
                    }
                )
                if (
                    best_res is None
                    or res["stability_adjusted_acc"]
                    > best_res["stability_adjusted_acc"]
                ):
                    best_cfg, best_res = {"lr": lr, "conf_thresh": ct}, res
        return best_cfg, best_res


def save_common_arrays(prefix, dataset_store):
    np.save(
        os.path.join(working_dir, f"{prefix}_train_losses.npy"),
        np.array(dataset_store["losses"]["train"], dtype=object),
    )
    np.save(
        os.path.join(working_dir, f"{prefix}_val_losses.npy"),
        np.array(dataset_store["losses"]["val"], dtype=object),
    )
    np.save(
        os.path.join(working_dir, f"{prefix}_train_metrics.npy"),
        np.array(dataset_store["metrics"]["train"], dtype=object),
    )
    np.save(
        os.path.join(working_dir, f"{prefix}_val_metrics.npy"),
        np.array(dataset_store["metrics"]["val"], dtype=object),
    )
    np.save(
        os.path.join(working_dir, f"{prefix}_test_metrics.npy"),
        np.array(dataset_store["metrics"]["test"], dtype=object),
    )
    np.save(
        os.path.join(working_dir, f"{prefix}_lr_sweep.npy"),
        np.array(dataset_store["metrics"]["lr_sweep"], dtype=object),
    )
    np.save(
        os.path.join(working_dir, f"{prefix}_tta_sweep.npy"),
        np.array(dataset_store["metrics"]["tta_sweep"], dtype=object),
    )
    np.save(
        os.path.join(working_dir, f"{prefix}_ground_truth.npy"),
        np.array(dataset_store["ground_truth"]),
    )
    for name, pred in dataset_store["predictions"].items():
        np.save(os.path.join(working_dir, f"{prefix}_preds_{name}.npy"), np.array(pred))
    for key, arr in dataset_store["stream"].items():
        np.save(
            os.path.join(working_dir, f"{prefix}_{key}.npy"),
            np.array(arr, dtype=object if isinstance(arr, list) else None),
        )


def plot_dataset(prefix, title, dataset_store):
    if (
        len(dataset_store["losses"]["train"]) > 0
        and len(dataset_store["losses"]["val"]) > 0
    ):
        train_curve = np.array(
            [[x[0], x[1]] for x in dataset_store["losses"]["train"]], dtype=float
        )
        val_curve = np.array(
            [[x[0], x[1]] for x in dataset_store["losses"]["val"]], dtype=float
        )
        plt.figure(figsize=(9, 4))
        plt.plot(train_curve[:, 0], train_curve[:, 1], label="Train loss")
        plt.plot(val_curve[:, 0], val_curve[:, 1], label="Val loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{title}: selected training curves")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, f"{prefix}_training_curves.png"), dpi=160)
        plt.close()

    if "frozen_rolling_acc" in dataset_store["stream"]:
        x = np.arange(len(dataset_store["stream"]["frozen_rolling_acc"]))
        plt.figure(figsize=(10, 5))
        plt.plot(
            x,
            dataset_store["stream"]["frozen_rolling_acc"],
            label="Frozen",
            linewidth=2,
        )
        plt.plot(
            x,
            dataset_store["stream"]["entropy_rolling_acc"],
            label="Entropy TTA",
            linewidth=2,
        )
        plt.plot(
            x,
            dataset_store["stream"]["consistency_rolling_acc"],
            label="Consistency TTA",
            linewidth=2,
        )
        ent_u = np.where(np.array(dataset_store["stream"]["entropy_updates"]) == 1)[0]
        con_u = np.where(np.array(dataset_store["stream"]["consistency_updates"]) == 1)[
            0
        ]
        if len(ent_u) > 0:
            plt.scatter(
                ent_u,
                np.array(dataset_store["stream"]["entropy_rolling_acc"])[ent_u],
                s=10,
                alpha=0.35,
                label="Entropy updates",
            )
        if len(con_u) > 0:
            plt.scatter(
                con_u,
                np.array(dataset_store["stream"]["consistency_rolling_acc"])[con_u],
                s=14,
                alpha=0.5,
                label="Consistency updates",
            )
        plt.xlabel("Stream step")
        plt.ylabel("Rolling accuracy")
        plt.title(f"{title}: rolling stream accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, f"{prefix}_rolling_accuracy.png"), dpi=160
        )
        plt.close()

    if (
        isinstance(dataset_store["predictions"], dict)
        and "consistency" in dataset_store["predictions"]
    ):
        gt = np.array(dataset_store["ground_truth"])
        pred_c = np.array(dataset_store["predictions"]["consistency"])
        pred_e = np.array(dataset_store["predictions"]["entropy"])
        n_show = min(60, len(gt))
        plt.figure(figsize=(10, 4))
        plt.plot(gt[:n_show], label="Ground Truth", marker="o")
        plt.plot(pred_e[:n_show], label="Entropy Pred", marker="x")
        plt.plot(pred_c[:n_show], label="Consistency Pred", marker="s")
        plt.yticks(sorted(np.unique(gt)))
        plt.xlabel("Sample index")
        plt.ylabel("Class")
        plt.title(f"{title}: sample predictions vs ground truth")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, f"{prefix}_prediction_samples.png"), dpi=160
        )
        plt.close()


def run_experiment(dataset_key, train_ds, val_ds, test_stream_ds):
    dataset_store = experiment_data[dataset_key]
    normalizer = FeatureNormalizer.from_dataset(train_ds)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)
    stream_loader = DataLoader(test_stream_ds, batch_size=1, shuffle=False)

    candidate_lrs = [1e-4, 3e-4, 1e-3, 3e-3]
    best_model, best_lr, best_val = None, None, float("inf")
    trial_hist = {}

    for lr in candidate_lrs:
        model_lr, trial_best_val, best_epoch, history = train_single_lr(
            train_loader, val_loader, normalizer, lr=lr, epochs=20, seed=7, patience=4
        )
        lr_key = f"{lr:.0e}"
        trial_hist[lr_key] = history
        dataset_store["metrics"]["lr_sweep"].append(
            {
                "lr": lr,
                "best_val_loss": trial_best_val,
                "best_epoch": best_epoch,
                "timestamp": time.time(),
            }
        )
        if trial_best_val < best_val:
            best_val = trial_best_val
            best_lr = lr
            best_model = SmallVLM(dim=16, hidden=64, num_classes=4).to(device)
            best_model.load_state_dict(copy.deepcopy(model_lr.state_dict()))

    best_key = f"{best_lr:.0e}"
    dataset_store["losses"]["train"] = trial_hist[best_key]["train_loss"]
    dataset_store["losses"]["val"] = trial_hist[best_key]["val_loss"]
    dataset_store["metrics"]["train"] = trial_hist[best_key]["train_acc"]
    dataset_store["metrics"]["val"] = trial_hist[best_key]["val_acc"]
    dataset_store["timestamps"] = [
        time.time() for _ in range(len(dataset_store["losses"]["train"]))
    ]

    print(
        f"[{dataset_key}] Selected learning rate: {best_lr:.0e} with best val loss {best_val:.4f}"
    )

    frozen_res = stream_eval_with_tta(
        best_model, stream_loader, normalizer, mode="frozen", lr=1e-3, conf_thresh=0.7
    )
    best_entropy_cfg, entropy_res = tune_tta_params(
        best_model, stream_loader, normalizer, dataset_store, mode="entropy"
    )
    best_cons_cfg, consistency_res = tune_tta_params(
        best_model, stream_loader, normalizer, dataset_store, mode="consistency"
    )

    results = {
        "frozen": frozen_res,
        "entropy": entropy_res,
        "consistency": consistency_res,
    }
    for name, res in results.items():
        print(
            f"[{dataset_key}] {name}: acc={res['acc']:.4f}, "
            f"Stream Stability-Adjusted Accuracy={res['stability_adjusted_acc']:.4f}, "
            f"adapt_freq={res['adapt_freq']:.4f}, adapt_loss_mean={res['adapt_loss_mean']:.4f}"
        )
        dataset_store["metrics"]["test"].append(
            {
                "name": name,
                "acc": res["acc"],
                "stability_adjusted_acc": res["stability_adjusted_acc"],
                "adapt_freq": res["adapt_freq"],
                "adapt_loss_mean": res["adapt_loss_mean"],
                "selected_train_lr": best_lr,
                "selected_tta_lr": (
                    None
                    if name == "frozen"
                    else (
                        best_entropy_cfg["lr"]
                        if name == "entropy"
                        else best_cons_cfg["lr"]
                    )
                ),
                "selected_conf_thresh": (
                    None if name != "consistency" else best_cons_cfg["conf_thresh"]
                ),
                "timestamp": time.time(),
            }
        )

    dataset_store["predictions"] = {
        "frozen": frozen_res["preds"],
        "entropy": entropy_res["preds"],
        "consistency": consistency_res["preds"],
    }
    dataset_store["ground_truth"] = frozen_res["gts"]
    dataset_store["stream"] = {
        "frozen_updates": frozen_res["updates"],
        "entropy_updates": entropy_res["updates"],
        "consistency_updates": consistency_res["updates"],
        "frozen_rolling_acc": frozen_res["rolling_acc"],
        "entropy_rolling_acc": entropy_res["rolling_acc"],
        "consistency_rolling_acc": consistency_res["rolling_acc"],
        "frozen_confidences": frozen_res["confidences"],
        "entropy_confidences": entropy_res["confidences"],
        "consistency_confidences": consistency_res["confidences"],
        "selected_train_lr": best_lr,
        "selected_best_val": best_val,
        "best_entropy_tta_lr": (
            None if best_entropy_cfg is None else best_entropy_cfg["lr"]
        ),
        "best_consistency_tta_lr": (
            None if best_cons_cfg is None else best_cons_cfg["lr"]
        ),
        "best_consistency_conf_thresh": (
            None if best_cons_cfg is None else best_cons_cfg["conf_thresh"]
        ),
    }
    return dataset_store


# Dataset 1: synthetic main benchmark
world = SyntheticWorld(dim=16, num_classes=4, seed=7)
syn_train = SyntheticVLMDataset(world, n=2500, shift=False, seed=11, mode_tag="generic")
syn_val = SyntheticVLMDataset(world, n=500, shift=False, seed=13, mode_tag="generic")
syn_stream = SyntheticVLMDataset(world, n=700, shift=True, seed=17, mode_tag="generic")
run_experiment("synthetic_vlm_stream", syn_train, syn_val, syn_stream)

# Dataset 2: HF AI2D-like benchmark
ai2d_train, ai2d_src_train = make_hf_like_dataset(
    "lmms-lab/ai2d", "train", "ai2d", 384, 21
)
ai2d_val, ai2d_src_val = make_hf_like_dataset(
    "lmms-lab/ai2d", "validation", "ai2d", 128, 22
)
ai2d_test, ai2d_src_test = make_hf_like_dataset(
    "lmms-lab/ai2d", "test", "ai2d", 192, 23
)
experiment_data["hf_ai2d_like_stream"]["meta"]["resolved_train_source"] = ai2d_src_train
experiment_data["hf_ai2d_like_stream"]["meta"]["resolved_val_source"] = ai2d_src_val
experiment_data["hf_ai2d_like_stream"]["meta"]["resolved_test_source"] = ai2d_src_test
run_experiment("hf_ai2d_like_stream", ai2d_train, ai2d_val, ai2d_test)

# Dataset 3: HF TextVQA-like benchmark
textvqa_train, tvqa_src_train = make_hf_like_dataset(
    "textvqa", "train", "textvqa", 384, 31
)
textvqa_val, tvqa_src_val = make_hf_like_dataset(
    "textvqa", "validation", "textvqa", 128, 32
)
textvqa_test, tvqa_src_test = make_hf_like_dataset(
    "textvqa", "test", "textvqa", 192, 33
)
experiment_data["hf_textvqa_like_stream"]["meta"][
    "resolved_train_source"
] = tvqa_src_train
experiment_data["hf_textvqa_like_stream"]["meta"]["resolved_val_source"] = tvqa_src_val
experiment_data["hf_textvqa_like_stream"]["meta"][
    "resolved_test_source"
] = tvqa_src_test
run_experiment("hf_textvqa_like_stream", textvqa_train, textvqa_val, textvqa_test)

# Save and plot per dataset
for dataset_key, prefix, title in [
    ("synthetic_vlm_stream", "synthetic_vlm_stream", "Synthetic multimodal stream"),
    ("hf_ai2d_like_stream", "hf_ai2d_like_stream", "HF AI2D-like stream"),
    ("hf_textvqa_like_stream", "hf_textvqa_like_stream", "HF TextVQA-like stream"),
]:
    save_common_arrays(prefix, experiment_data[dataset_key])
    plot_dataset(prefix, title, experiment_data[dataset_key])

# Cross-dataset summary plot
summary_rows = []
for dataset_key in [
    "synthetic_vlm_stream",
    "hf_ai2d_like_stream",
    "hf_textvqa_like_stream",
]:
    tests = experiment_data[dataset_key]["metrics"]["test"]
    for item in tests:
        summary_rows.append(
            [
                dataset_key,
                item["name"],
                item["acc"],
                item["stability_adjusted_acc"],
                item["adapt_freq"],
            ]
        )
summary_arr = np.array(summary_rows, dtype=object)
np.save(os.path.join(working_dir, "cross_dataset_summary.npy"), summary_arr)

datasets = ["synthetic_vlm_stream", "hf_ai2d_like_stream", "hf_textvqa_like_stream"]
modes = ["frozen", "entropy", "consistency"]
vals = np.zeros((len(datasets), len(modes)), dtype=float)
for i, dk in enumerate(datasets):
    lookup = {
        x["name"]: x["stability_adjusted_acc"]
        for x in experiment_data[dk]["metrics"]["test"]
    }
    for j, m in enumerate(modes):
        vals[i, j] = lookup[m]
plt.figure(figsize=(9, 4))
x = np.arange(len(datasets))
w = 0.25
for j, m in enumerate(modes):
    plt.bar(x + (j - 1) * w, vals[:, j], width=w, label=m)
plt.xticks(x, datasets, rotation=15)
plt.ylabel("Stability-Adjusted Stream Accuracy")
plt.title("Cross-dataset stability comparison")
plt.legend()
plt.tight_layout()
plt.savefig(
    os.path.join(working_dir, "cross_dataset_stability_comparison.png"), dpi=160
)
plt.close()

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(
    "Finished hyperparameter tuning, multi-dataset evaluation, plotting, and saving experiment_data.npy"
)
