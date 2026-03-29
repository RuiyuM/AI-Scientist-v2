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
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

BASE_SEED = 42
random.seed(BASE_SEED)
np.random.seed(BASE_SEED)
torch.manual_seed(BASE_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(BASE_SEED)

HF_MODEL = "distilbert-base-uncased"
MAX_LEN = 192

dataset_names = [
    "ai2_arc_easy_stream",
    "ai2_arc_challenge_stream",
    "mmlu_elementary_math_stream",
    "truthfulqa_mc1_stream",
    "openbookqa_stream",
]

experiment_data = {
    name: {
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
        "severity": None,
    }
    for name in dataset_names
}


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_text(x):
    x = "" if x is None else str(x)
    return " ".join(x.strip().split())


def mc_example_to_text(question, choices):
    question = normalize_text(question)
    choices = [normalize_text(c) for c in choices]
    parts = [f"Question: {question}", "Options:"]
    for i, c in enumerate(choices):
        parts.append(f"{chr(65+i)}. {c}")
    return " ".join(parts)


def parse_arc(split_name="ARC-Easy", n_train=512, n_val=128, n_test=256):
    ds = load_dataset("allenai/ai2_arc", split_name)
    train_raw = (
        ds["train"]
        .shuffle(seed=BASE_SEED)
        .select(range(min(n_train, len(ds["train"]))))
    )
    val_raw = (
        ds["validation"]
        .shuffle(seed=BASE_SEED)
        .select(range(min(n_val, len(ds["validation"]))))
    )
    test_raw = (
        ds["test"].shuffle(seed=BASE_SEED).select(range(min(n_test, len(ds["test"]))))
    )

    def convert(ex):
        labels = ex["choices"]["label"]
        texts = ex["choices"]["text"]
        ans = ex["answerKey"]
        if ans not in labels:
            return None
        y = labels.index(ans)
        if y >= len(texts) or len(texts) < 2:
            return None
        return {
            "text": mc_example_to_text(ex["question"], texts),
            "label": int(y),
            "num_choices": len(texts),
        }

    def filt(raw):
        out = []
        for ex in raw:
            z = convert(ex)
            if z is not None and z["num_choices"] <= 5:
                out.append(z)
        return out

    return filt(train_raw), filt(val_raw), filt(test_raw)


def parse_openbookqa(n_train=512, n_val=128, n_test=256):
    ds = load_dataset("allenai/openbookqa", "main")
    train_raw = (
        ds["train"]
        .shuffle(seed=BASE_SEED)
        .select(range(min(n_train, len(ds["train"]))))
    )
    val_raw = (
        ds["validation"]
        .shuffle(seed=BASE_SEED)
        .select(range(min(n_val, len(ds["validation"]))))
    )
    test_raw = (
        ds["test"].shuffle(seed=BASE_SEED).select(range(min(n_test, len(ds["test"]))))
    )

    def convert(ex):
        labels = ex["choices"]["label"]
        texts = ex["choices"]["text"]
        ans = ex["answerKey"]
        if ans not in labels:
            return None
        y = labels.index(ans)
        return {
            "text": mc_example_to_text(ex["question_stem"], texts),
            "label": int(y),
            "num_choices": len(texts),
        }

    def filt(raw):
        return [
            z
            for z in (convert(ex) for ex in raw)
            if z is not None and z["num_choices"] <= 5
        ]

    return filt(train_raw), filt(val_raw), filt(test_raw)


def parse_truthfulqa_mc1(n_val=128, n_test=256):
    ds = load_dataset("truthful_qa", "multiple_choice")
    val_raw = ds["validation"].shuffle(seed=BASE_SEED)
    n_total = min(n_val + n_test, len(val_raw))
    val_raw = val_raw.select(range(n_total))

    def convert(ex):
        choices = ex["mc1_targets"]["choices"]
        labels = ex["mc1_targets"]["labels"]
        if not labels or sum(labels) != 1:
            return None
        y = int(np.argmax(labels))
        return {
            "text": mc_example_to_text(ex["question"], choices),
            "label": y,
            "num_choices": len(choices),
        }

    parsed = [
        z
        for z in (convert(ex) for ex in val_raw)
        if z is not None and z["num_choices"] <= 5
    ]
    split = min(n_val, len(parsed) // 2 if len(parsed) < n_val + n_test else n_val)
    val = parsed[:split]
    test = parsed[split : split + n_test]
    return [], val, test


def parse_mmlu(subject="elementary_mathematics", n_dev=256, n_test=256):
    ds = load_dataset("cais/mmlu", subject)
    dev_raw = (
        ds["dev"].shuffle(seed=BASE_SEED).select(range(min(n_dev, len(ds["dev"]))))
    )
    val_raw = (
        ds["validation"]
        .shuffle(seed=BASE_SEED)
        .select(range(min(n_test, len(ds["validation"]))))
    )

    def convert(ex):
        choices = ex["choices"]
        y = int(ex["answer"])
        return {
            "text": mc_example_to_text(ex["question"], choices),
            "label": y,
            "num_choices": len(choices),
        }

    dev = [convert(ex) for ex in dev_raw]
    val = [convert(ex) for ex in val_raw]
    return dev, dev[: min(128, len(dev))], val


class HFMCQDataset(Dataset):
    def __init__(self, examples, tokenizer, max_len=192, severity=0.0, name="dataset"):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.severity = float(severity)
        self.name = name

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        enc = self.tokenizer(
            normalize_text(ex["text"]),
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "y": torch.tensor(ex["label"], dtype=torch.long),
        }


class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=8.0, dropout=0.05):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.rank = rank
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout)
        self.A = nn.Parameter(torch.zeros(rank, in_features))
        self.B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.normal_(self.A, std=0.02)
        nn.init.zeros_(self.B)
        self.register_buffer("A_init", self.A.detach().clone())
        self.register_buffer("B_init", self.B.detach().clone())

    def forward(self, x):
        x_d = self.dropout(x)
        delta = (self.B @ self.A) * (self.alpha / max(self.rank, 1))
        return F.linear(x_d, self.weight + delta, self.bias)

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


class MCQEncoder(nn.Module):
    def __init__(self, encoder_name=HF_MODEL, num_classes=5, rank=8):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        h = self.encoder.config.hidden_size
        self.classifier = LoRALinear(h, num_classes, rank=rank, alpha=8.0, dropout=0.05)

    def encode(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        return pooled

    def forward(self, input_ids, attention_mask):
        pooled = self.encode(input_ids, attention_mask)
        return self.classifier(pooled)

    def freeze_backbone(self):
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.classifier.weight.requires_grad = False
        self.classifier.bias.requires_grad = False
        for p in self.classifier.lora_parameters():
            p.requires_grad = True

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True

    def reset_lora(self):
        self.classifier.reset_lora()

    def lora_reg_loss(self):
        return self.classifier.lora_reg_loss()


def entropy_from_logits(logits):
    p = F.softmax(logits, dim=-1)
    return -(p * torch.log(p + 1e-8)).sum(dim=-1)


def margin_from_logits(logits):
    probs = F.softmax(logits, dim=-1)
    top2 = torch.topk(probs, k=2, dim=-1).values
    return top2[:, 0] - top2[:, 1]


def evaluate_loader(model, loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, total_correct, total = 0.0, 0, 0
    all_preds, all_gt = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = criterion(logits, batch["y"])
            total_loss += loss.item() * batch["y"].size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == batch["y"]).sum().item()
            total += batch["y"].size(0)
            all_preds.append(preds.detach().cpu().numpy())
            all_gt.append(batch["y"].detach().cpu().numpy())
    all_preds = np.concatenate(all_preds) if all_preds else np.array([])
    all_gt = np.concatenate(all_gt) if all_gt else np.array([])
    return total_loss / max(total, 1), total_correct / max(total, 1), all_preds, all_gt


def collect_confidence_stats(model, loader, max_batches=8):
    model.eval()
    ents, mars = [], []
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if bi >= max_batches:
                break
            batch = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            logits = model(batch["input_ids"], batch["attention_mask"])
            ents.append(entropy_from_logits(logits).detach().cpu())
            mars.append(margin_from_logits(logits).detach().cpu())
    ents = torch.cat(ents).numpy()
    mars = torch.cat(mars).numpy()
    return {
        "entropy_thresh": float(np.quantile(ents, 0.70)),
        "margin_thresh": float(np.quantile(mars, 0.30)),
        "entropy_mean": float(np.mean(ents)),
        "margin_mean": float(np.mean(mars)),
    }


def stream_evaluate(
    base_model,
    dataset,
    mode="frozen",
    entropy_thresh=0.8,
    margin_thresh=0.15,
    reset_every=64,
    lr=2e-4,
    reg_lambda=5e-3,
    grad_clip=0.5,
    update_steps=1,
    warmup=8,
):
    stream_model = copy.deepcopy(base_model).to(device)
    stream_model.freeze_backbone()
    stream_model.eval()
    opt = torch.optim.AdamW(
        stream_model.classifier.lora_parameters(), lr=lr, weight_decay=0.0
    )

    preds_all, gt_all, ent_all, margin_all = [], [], [], []
    triggered, cumulative_acc = [], []
    correct = 0

    for i in range(len(dataset)):
        batch = dataset[i]
        batch = {
            k: (v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        x_ids, x_mask, y = batch["input_ids"], batch["attention_mask"], batch["y"]

        stream_model.eval()
        with torch.no_grad():
            logits = stream_model(x_ids, x_mask)
            ent = entropy_from_logits(logits).item()
            mar = margin_from_logits(logits).item()
            pred = logits.argmax(dim=1)

        preds_all.append(int(pred.item()))
        gt_all.append(int(y.item()))
        ent_all.append(ent)
        margin_all.append(mar)
        correct += int(pred.item() == y.item())
        cumulative_acc.append(correct / (i + 1))

        do_update = False
        if i + 1 > warmup:
            if mode == "always":
                do_update = True
            elif mode == "gated":
                do_update = (ent > entropy_thresh) or (mar < margin_thresh)
        triggered.append(int(do_update))

        if do_update:
            stream_model.train()
            for _ in range(update_steps):
                opt.zero_grad()
                logits_u = stream_model(x_ids, x_mask)
                p = F.softmax(logits_u, dim=-1)
                ent_loss = -(p * torch.log(p + 1e-8)).sum(dim=-1).mean()
                pseudo = logits_u.detach().argmax(dim=-1)
                pseudo_loss = F.cross_entropy(logits_u, pseudo)
                reg_loss = stream_model.lora_reg_loss()
                loss_u = 0.3 * ent_loss + 0.7 * pseudo_loss + reg_lambda * reg_loss
                loss_u.backward()
                torch.nn.utils.clip_grad_norm_(
                    stream_model.classifier.lora_parameters(), grad_clip
                )
                opt.step()

        if (
            mode in ["always", "gated"]
            and reset_every > 0
            and (i + 1) % reset_every == 0
        ):
            stream_model.reset_lora()

    acc = float(np.mean(np.array(preds_all) == np.array(gt_all))) if gt_all else 0.0
    return {
        "accuracy": acc,
        "preds": np.array(preds_all),
        "gt": np.array(gt_all),
        "entropy": np.array(ent_all),
        "margin": np.array(margin_all),
        "triggered": np.array(triggered),
        "cumulative_acc": np.array(cumulative_acc),
    }


def compute_shift_normalized_gain(frozen_acc, gated_acc, severity):
    return (gated_acc - frozen_acc) / max(float(severity), 1e-6)


def build_loaders(tokenizer, batch_size=16):
    arc_easy_train, arc_easy_val, arc_easy_test = parse_arc(
        "ARC-Easy", n_train=512, n_val=128, n_test=256
    )
    arc_ch_train, arc_ch_val, arc_ch_test = parse_arc(
        "ARC-Challenge", n_train=512, n_val=128, n_test=256
    )
    mmlu_train, mmlu_val, mmlu_test = parse_mmlu(
        "elementary_mathematics", n_dev=256, n_test=256
    )
    _, tfqa_val, tfqa_test = parse_truthfulqa_mc1(n_val=128, n_test=256)
    obqa_train, obqa_val, obqa_test = parse_openbookqa(
        n_train=512, n_val=128, n_test=256
    )

    train_examples = arc_easy_train + arc_ch_train + mmlu_train + obqa_train
    val_examples = arc_easy_val + arc_ch_val + mmlu_val + tfqa_val + obqa_val

    train_ds = HFMCQDataset(
        train_examples, tokenizer, MAX_LEN, severity=0.0, name="train_mix"
    )
    val_ds = HFMCQDataset(
        val_examples, tokenizer, MAX_LEN, severity=0.0, name="val_mix"
    )

    test_datasets = {
        "ai2_arc_easy_stream": HFMCQDataset(
            arc_easy_test, tokenizer, MAX_LEN, severity=0.2, name="ai2_arc_easy_stream"
        ),
        "ai2_arc_challenge_stream": HFMCQDataset(
            arc_ch_test,
            tokenizer,
            MAX_LEN,
            severity=1.0,
            name="ai2_arc_challenge_stream",
        ),
        "mmlu_elementary_math_stream": HFMCQDataset(
            mmlu_test,
            tokenizer,
            MAX_LEN,
            severity=1.2,
            name="mmlu_elementary_math_stream",
        ),
        "truthfulqa_mc1_stream": HFMCQDataset(
            tfqa_test, tokenizer, MAX_LEN, severity=1.4, name="truthfulqa_mc1_stream"
        ),
        "openbookqa_stream": HFMCQDataset(
            obqa_test, tokenizer, MAX_LEN, severity=0.8, name="openbookqa_stream"
        ),
    }

    g = torch.Generator().manual_seed(BASE_SEED)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, generator=g
    )
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    return train_ds, val_ds, test_datasets, train_loader, val_loader


def train_source_model(
    epochs, train_loader, val_loader, seed=42, lr=2e-5, weight_decay=1e-4
):
    set_seed(seed)
    model = MCQEncoder(encoder_name=HF_MODEL, num_classes=5, rank=8).to(device)
    criterion = nn.CrossEntropyLoss()
    model.unfreeze_all()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state, best_val, best_epoch = None, float("inf"), -1
    history = {
        "metrics": {"train": [], "val": [], "test": []},
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
            optimizer.zero_grad()
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = criterion(logits, batch["y"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * batch["y"].size(0)
            train_correct += (logits.argmax(dim=1) == batch["y"]).sum().item()
            train_total += batch["y"].size(0)

        train_loss /= max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)
        val_loss, val_acc, _, _ = evaluate_loader(model, val_loader)

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


tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)

epoch_candidates = [2, 3]
lr_candidates = [2e-5, 3e-5]
batch_candidates = [16]
adapt_lr_candidates = [1e-4, 2e-4]
reset_candidates = [64, 128]
experiment_data["ai2_arc_challenge_stream"]["epoch_candidates"] = epoch_candidates

best_trial_key = None
best_trial_score = -1e9
all_scores = []

trial_idx = 0
for epochs in epoch_candidates:
    for lr in lr_candidates:
        for batch_size in batch_candidates:
            for adapt_lr in adapt_lr_candidates:
                for reset_every in reset_candidates:
                    trial_idx += 1
                    trial_key = f"trial_{trial_idx}_epochs_{epochs}_lr_{lr}_bs_{batch_size}_alr_{adapt_lr}_reset_{reset_every}"
                    train_ds, val_ds, test_datasets, train_loader, val_loader = (
                        build_loaders(tokenizer, batch_size=batch_size)
                    )
                    model, history = train_source_model(
                        epochs,
                        train_loader,
                        val_loader,
                        seed=BASE_SEED,
                        lr=lr,
                        weight_decay=1e-4,
                    )

                    calib = collect_confidence_stats(model, val_loader, max_batches=8)
                    entropy_thresh = calib["entropy_thresh"]
                    margin_thresh = calib["margin_thresh"]

                    source_name = "ai2_arc_easy_stream"
                    source_frozen = stream_evaluate(
                        model,
                        test_datasets[source_name],
                        mode="frozen",
                        entropy_thresh=entropy_thresh,
                        margin_thresh=margin_thresh,
                        reset_every=reset_every,
                        lr=adapt_lr,
                    )
                    source_gated = stream_evaluate(
                        model,
                        test_datasets[source_name],
                        mode="gated",
                        entropy_thresh=entropy_thresh,
                        margin_thresh=margin_thresh,
                        reset_every=reset_every,
                        lr=adapt_lr,
                        reg_lambda=5e-3,
                        grad_clip=0.5,
                        update_steps=1,
                        warmup=8,
                    )
                    source_penalty = (
                        source_gated["accuracy"] - source_frozen["accuracy"]
                    )

                    ds_names = [
                        "ai2_arc_challenge_stream",
                        "mmlu_elementary_math_stream",
                        "truthfulqa_mc1_stream",
                        "openbookqa_stream",
                    ]

                    per_dataset = {}
                    gains = [source_penalty]
                    history["metrics"]["test"].append(
                        (
                            epochs,
                            source_frozen["accuracy"],
                            source_gated["accuracy"],
                            source_penalty,
                        )
                    )

                    for ds_name in ds_names:
                        ds = test_datasets[ds_name]
                        frozen_res = stream_evaluate(
                            model,
                            ds,
                            mode="frozen",
                            entropy_thresh=entropy_thresh,
                            margin_thresh=margin_thresh,
                            reset_every=reset_every,
                            lr=adapt_lr,
                        )
                        always_res = stream_evaluate(
                            model,
                            ds,
                            mode="always",
                            entropy_thresh=entropy_thresh,
                            margin_thresh=margin_thresh,
                            reset_every=reset_every,
                            lr=adapt_lr,
                            reg_lambda=5e-3,
                            grad_clip=0.5,
                            update_steps=1,
                            warmup=8,
                        )
                        gated_res = stream_evaluate(
                            model,
                            ds,
                            mode="gated",
                            entropy_thresh=entropy_thresh,
                            margin_thresh=margin_thresh,
                            reset_every=reset_every,
                            lr=adapt_lr,
                            reg_lambda=5e-3,
                            grad_clip=0.5,
                            update_steps=1,
                            warmup=8,
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
                    all_scores.append(
                        (epochs, lr, batch_size, adapt_lr, reset_every, mean_gain)
                    )

                    trial = {
                        "epochs": epochs,
                        "lr": lr,
                        "batch_size": batch_size,
                        "adapt_lr": adapt_lr,
                        "reset_every": reset_every,
                        "best_epoch": history["best_epoch"],
                        "best_val_loss": history["best_val_loss"],
                        "metrics": history["metrics"],
                        "losses": history["losses"],
                        "timestamps": history["timestamps"],
                        "calibration": calib,
                        "source_stream": {
                            "name": source_name,
                            "frozen_acc": source_frozen["accuracy"],
                            "gated_acc": source_gated["accuracy"],
                            "source_penalty": source_penalty,
                        },
                        "per_dataset": {},
                        "mean_shift_normalized_accuracy_gain": mean_gain,
                    }

                    trial["per_dataset"][source_name] = {
                        "severity": test_datasets[source_name].severity,
                        "stream_results": {
                            "frozen": {
                                "Shifted-Stream Accuracy": source_frozen["accuracy"],
                                "trigger_rate": float(
                                    source_frozen["triggered"].mean()
                                ),
                            },
                            "always": {
                                "Shifted-Stream Accuracy": 0.0,
                                "trigger_rate": 0.0,
                            },
                            "gated": {
                                "Shifted-Stream Accuracy": source_gated["accuracy"],
                                "trigger_rate": float(source_gated["triggered"].mean()),
                            },
                        },
                        "test_metrics": {
                            "frozen_acc": source_frozen["accuracy"],
                            "always_acc": 0.0,
                            "gated_acc": source_gated["accuracy"],
                            "shift_normalized_accuracy_gain": source_penalty,
                        },
                        "arrays": {
                            "frozen_preds": source_frozen["preds"],
                            "always_preds": np.array([]),
                            "gated_preds": source_gated["preds"],
                            "ground_truth": source_gated["gt"],
                            "gated_entropy": source_gated["entropy"],
                            "gated_margin": source_gated["margin"],
                            "gated_triggered": source_gated["triggered"],
                            "frozen_cumacc": source_frozen["cumulative_acc"],
                            "always_cumacc": np.array([]),
                            "gated_cumacc": source_gated["cumulative_acc"],
                        },
                    }

                    for ds_name, results in per_dataset.items():
                        trial["per_dataset"][ds_name] = {
                            "severity": results["severity"],
                            "stream_results": {
                                "frozen": {
                                    "Shifted-Stream Accuracy": results["frozen"][
                                        "accuracy"
                                    ],
                                    "trigger_rate": float(
                                        results["frozen"]["triggered"].mean()
                                    ),
                                },
                                "always": {
                                    "Shifted-Stream Accuracy": results["always"][
                                        "accuracy"
                                    ],
                                    "trigger_rate": float(
                                        results["always"]["triggered"].mean()
                                    ),
                                },
                                "gated": {
                                    "Shifted-Stream Accuracy": results["gated"][
                                        "accuracy"
                                    ],
                                    "trigger_rate": float(
                                        results["gated"]["triggered"].mean()
                                    ),
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

                    experiment_data["ai2_arc_challenge_stream"]["trials"][
                        trial_key
                    ] = trial

                    if mean_gain > best_trial_score:
                        best_trial_score = mean_gain
                        best_trial_key = trial_key

selected = experiment_data["ai2_arc_challenge_stream"]["trials"][best_trial_key]
experiment_data["ai2_arc_challenge_stream"]["selected_epochs"] = selected["epochs"]
experiment_data["ai2_arc_challenge_stream"]["selected_hparams"] = {
    "epochs": selected["epochs"],
    "lr": selected["lr"],
    "batch_size": selected["batch_size"],
    "adapt_lr": selected["adapt_lr"],
    "reset_every": selected["reset_every"],
    "entropy_thresh": selected["calibration"]["entropy_thresh"],
    "margin_thresh": selected["calibration"]["margin_thresh"],
}
experiment_data["ai2_arc_challenge_stream"]["metrics"]["train"] = selected["metrics"][
    "train"
]
experiment_data["ai2_arc_challenge_stream"]["metrics"]["val"] = selected["metrics"][
    "val"
]
experiment_data["ai2_arc_challenge_stream"]["losses"]["train"] = selected["losses"][
    "train"
]
experiment_data["ai2_arc_challenge_stream"]["losses"]["val"] = selected["losses"]["val"]
experiment_data["ai2_arc_challenge_stream"]["timestamps"] = selected["timestamps"]

for ds_name in dataset_names:
    if ds_name not in selected["per_dataset"]:
        continue
    ds_res = selected["per_dataset"][ds_name]
    experiment_data[ds_name]["stream_results"] = ds_res["stream_results"]
    experiment_data[ds_name]["metrics"]["test"].append((0, ds_res["test_metrics"]))
    experiment_data[ds_name]["predictions"] = ds_res["arrays"]["gated_preds"].tolist()
    experiment_data[ds_name]["ground_truth"] = ds_res["arrays"]["ground_truth"].tolist()
    experiment_data[ds_name]["severity"] = ds_res["severity"]
    experiment_data[ds_name]["shift_normalized_accuracy_gain"].append(
        (0, ds_res["test_metrics"]["shift_normalized_accuracy_gain"])
    )
    experiment_data[ds_name]["timestamps"] = selected["timestamps"]
    if ds_name == "ai2_arc_challenge_stream":
        experiment_data[ds_name]["stream_arrays"] = ds_res["arrays"]

print(
    f'Selected trial: {best_trial_key} | epochs={selected["epochs"]} | lr={selected["lr"]} | '
    f'bs={selected["batch_size"]} | adapt_lr={selected["adapt_lr"]} | reset={selected["reset_every"]} | '
    f'mean Shift-Normalized Accuracy Gain={selected["mean_shift_normalized_accuracy_gain"]:.6f}'
)

np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)

for trial_key, trial in experiment_data["ai2_arc_challenge_stream"]["trials"].items():
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
        os.path.join(working_dir, f"{trial_key}_test_metrics.npy"),
        np.array(trial["metrics"]["test"], dtype=object),
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
    np.save(
        os.path.join(working_dir, f"{trial_key}_calibration.npy"),
        np.array(
            [
                trial["calibration"]["entropy_thresh"],
                trial["calibration"]["margin_thresh"],
                trial["source_stream"]["frozen_acc"],
                trial["source_stream"]["gated_acc"],
                trial["source_stream"]["source_penalty"],
            ],
            dtype=np.float32,
        ),
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
for trial_key, trial in experiment_data["ai2_arc_challenge_stream"]["trials"].items():
    x = [e for e, _ in trial["losses"]["val"]]
    y = [v for _, v in trial["losses"]["val"]]
    plt.plot(x, y, label=f'{trial_key} (best={trial["best_val_loss"]:.3f})')
plt.xlabel("Epoch")
plt.ylabel("Validation loss")
plt.title("Validation Loss Across Hyperparameter Trials")
plt.legend(fontsize=6)
plt.tight_layout()
plt.savefig(os.path.join(working_dir, "baseline_tuning_val_loss.png"))
plt.close()

scores_sorted = sorted(all_scores, key=lambda z: z[-1], reverse=True)
labels = [
    f"e{e}|lr{lr}|bs{bs}|alr{alr}|r{re}" for e, lr, bs, alr, re, _ in scores_sorted
]
vals = [s for *_, s in scores_sorted]
plt.figure(figsize=(12, 5))
plt.bar(range(len(vals)), vals)
plt.xticks(range(len(vals)), labels, rotation=60, ha="right")
plt.ylabel("Mean Shift-Normalized Accuracy Gain")
plt.title("Hyperparameter Tuning by Target Metric")
plt.tight_layout()
plt.savefig(os.path.join(working_dir, "baseline_tuning_sna_gain.png"))
plt.close()

for ds_name in dataset_names:
    if ds_name not in selected["per_dataset"]:
        continue
    arr = selected["per_dataset"][ds_name]["arrays"]
    frozen_acc = selected["per_dataset"][ds_name]["stream_results"]["frozen"][
        "Shifted-Stream Accuracy"
    ]
    gated_acc = selected["per_dataset"][ds_name]["stream_results"]["gated"][
        "Shifted-Stream Accuracy"
    ]
    always_acc = selected["per_dataset"][ds_name]["stream_results"]["always"][
        "Shifted-Stream Accuracy"
    ]

    plt.figure(figsize=(8, 5))
    if len(arr["frozen_cumacc"]) > 0:
        plt.plot(arr["frozen_cumacc"], label=f"Frozen ({frozen_acc:.3f})")
    if len(arr["always_cumacc"]) > 0:
        plt.plot(arr["always_cumacc"], label=f"Always-LoRA ({always_acc:.3f})")
    if len(arr["gated_cumacc"]) > 0:
        plt.plot(arr["gated_cumacc"], label=f"Gated-LoRA ({gated_acc:.3f})")
    plt.xlabel("Stream step")
    plt.ylabel("Cumulative accuracy")
    plt.title(f"Cumulative Accuracy: {ds_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, f"{ds_name}_cumulative_accuracy.png"))
    plt.close()

    if len(arr["gated_entropy"]) > 0:
        plt.figure(figsize=(8, 4))
        scale = max(float(arr["gated_entropy"].max()), 1e-6)
        plt.plot(arr["gated_entropy"], label="Entropy")
        plt.plot(arr["gated_margin"], label="Margin")
        plt.plot(arr["gated_triggered"] * scale, label="Triggered (scaled)", alpha=0.7)
        plt.xlabel("Stream step")
        plt.ylabel("Value")
        plt.title(f"Gating Signals: {ds_name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, f"{ds_name}_gating_signals.png"))
        plt.close()

print("Finished. Saved metrics, arrays, and plots to:", working_dir)
