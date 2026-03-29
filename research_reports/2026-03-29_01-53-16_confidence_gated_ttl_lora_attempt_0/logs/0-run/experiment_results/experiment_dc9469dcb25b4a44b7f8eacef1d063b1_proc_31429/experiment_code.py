import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import sys
import time
import copy
import math
import random
import json
import types
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


# stdout/stderr compatibility patch for environments where RedirectQueue lacks isatty()
def _patch_stream_isatty(stream_name):
    stream = getattr(sys, stream_name, None)
    if stream is not None and not hasattr(stream, "isatty"):
        try:
            setattr(stream, "isatty", lambda: False)
        except Exception:

            class _StreamWrapper:
                def __init__(self, wrapped):
                    self._wrapped = wrapped

                def write(self, *args, **kwargs):
                    return self._wrapped.write(*args, **kwargs)

                def flush(self, *args, **kwargs):
                    if hasattr(self._wrapped, "flush"):
                        return self._wrapped.flush(*args, **kwargs)

                def isatty(self):
                    return False

                def __getattr__(self, name):
                    return getattr(self._wrapped, name)

            setattr(sys, stream_name, _StreamWrapper(stream))


_patch_stream_isatty("stdout")
_patch_stream_isatty("stderr")

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

experiment_data = {
    "allenai_ai2_arc": {
        "metrics": {"train": [], "val": [], "test": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "confidences": [],
        "triggers": [],
    },
    "qiaojin_PubMedQA": {
        "metrics": {"train": [], "val": [], "test": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "confidences": [],
        "triggers": [],
    },
    "cais_mmlu": {
        "metrics": {"train": [], "val": [], "test": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "confidences": [],
        "triggers": [],
    },
}

model_name = "sshleifer/tiny-gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

choices_letters = ["A", "B", "C", "D"]


def normalize_text(x):
    return " ".join(str(x).strip().split())


def build_arc(limit=30):
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=f"train[:{limit}]")
    items = []
    for ex in ds:
        labels = ex["choices"]["label"]
        texts = ex["choices"]["text"]
        pairs = [(l, t) for l, t in zip(labels, texts) if l in choices_letters]
        if len(pairs) < 2:
            continue
        pairs = pairs[:4]
        opts = "\n".join([f"{l}. {normalize_text(t)}" for l, t in pairs])
        ans = ex["answerKey"]
        if ans not in [p[0] for p in pairs]:
            continue
        prompt = f"Question: {normalize_text(ex['question'])}\n{opts}\nAnswer:"
        items.append({"dataset": "allenai_ai2_arc", "prompt": prompt, "label": ans})
    return items


def build_pubmed(limit=30):
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split=f"train[:{limit}]")
    items = []
    mapping = {"yes": "A", "no": "B", "maybe": "C"}
    for ex in ds:
        q = normalize_text(ex["question"])
        ctx = normalize_text(" ".join(ex["context"]["contexts"][:2]))
        prompt = (
            f"Question: {q}\nContext: {ctx}\nOptions:\nA. yes\nB. no\nC. maybe\nAnswer:"
        )
        label = mapping.get(str(ex["final_decision"]).lower(), None)
        if label is not None:
            items.append(
                {"dataset": "qiaojin_PubMedQA", "prompt": prompt, "label": label}
            )
    return items


def build_mmlu(limit=30):
    ds = load_dataset("cais/mmlu", "abstract_algebra", split=f"test[:{limit}]")
    items = []
    for ex in ds:
        ch = ex["choices"][:4]
        if len(ch) < 4:
            continue
        opts = "\n".join(
            [f"{choices_letters[i]}. {normalize_text(ch[i])}" for i in range(4)]
        )
        label = choices_letters[int(ex["answer"])]
        prompt = f"Question: {normalize_text(ex['question'])}\n{opts}\nAnswer:"
        items.append({"dataset": "cais_mmlu", "prompt": prompt, "label": label})
    return items


arc = build_arc()
pub = build_pubmed()
mmlu = build_mmlu()


def split_items(arr):
    n = len(arr)
    n_train = max(6, n // 3)
    n_val = max(5, n // 5)
    tr = arr[:n_train]
    va = arr[n_train : n_train + n_val]
    te = arr[n_train + n_val :]
    return tr, va, te


arc_tr, arc_va, arc_te = split_items(arc)
pub_tr, pub_va, pub_te = split_items(pub)
mmlu_tr, mmlu_va, mmlu_te = split_items(mmlu)

for name, tr, va, te in [
    ("allenai_ai2_arc", arc_tr, arc_va, arc_te),
    ("qiaojin_PubMedQA", pub_tr, pub_va, pub_te),
    ("cais_mmlu", mmlu_tr, mmlu_va, mmlu_te),
]:
    experiment_data[name]["split_sizes"] = {
        "train": len(tr),
        "val": len(va),
        "test": len(te),
    }

train_items = arc_tr + pub_tr + mmlu_tr
val_items = arc_va + pub_va + mmlu_va
test_items = arc_te + pub_te + mmlu_te


class AdapterWrappedLM(nn.Module):
    def __init__(self, model_name, device):
        super().__init__()
        self.base = AutoModelForCausalLM.from_pretrained(model_name)
        self.base = self.base.to(device)
        hidden = self.base.config.hidden_size
        vocab = self.base.config.vocab_size
        bottleneck = max(16, hidden // 2)
        self.adapter = nn.Sequential(
            nn.Linear(hidden, bottleneck),
            nn.Tanh(),
            nn.Linear(bottleneck, vocab),
        ).to(device)
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, input_ids, attention_mask=None, labels=None):
        out = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden = out.hidden_states[-1]
        delta_logits = self.adapter(hidden)
        logits = out.logits + delta_logits
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
        return types.SimpleNamespace(logits=logits, loss=loss)

    def reset_adapter(self, state_dict):
        self.adapter.load_state_dict(copy.deepcopy(state_dict))

    def adapter_state(self):
        return copy.deepcopy(self.adapter.state_dict())


model = AdapterWrappedLM(model_name, device).to(device)
optimizer = torch.optim.AdamW(model.adapter.parameters(), lr=1e-3)


def collate_prompt_answer(prompt, label, max_length=192):
    prompt = normalize_text(prompt.replace("\t", " "))
    full = prompt + " " + str(label).strip()
    enc = tokenizer(
        full,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)
    labels = input_ids.clone()
    p_ids = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=max_length
    )["input_ids"]
    plen = min(p_ids.shape[1], labels.shape[1] - 1)
    labels[:, :plen] = -100
    return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}


def valid_choices_for_item(ex):
    return (
        ["A", "B", "C"] if ex["dataset"] == "qiaojin_PubMedQA" else ["A", "B", "C", "D"]
    )


def answer_scores(cur_model, prompt, valid_letters=None):
    if valid_letters is None:
        valid_letters = choices_letters
    enc = tokenizer(
        normalize_text(prompt), return_tensors="pt", truncation=True, max_length=192
    )
    enc = {k: v.to(device) for k, v in enc.items() if isinstance(v, torch.Tensor)}
    with torch.no_grad():
        out = cur_model(**enc)
        logits = out.logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
    ids, usable_letters = [], []
    for c in valid_letters:
        tok = tokenizer.encode(" " + c, add_special_tokens=False)
        if len(tok) > 0:
            ids.append(tok[0])
            usable_letters.append(c)
    if not ids:
        scores = torch.ones(len(valid_letters), device=device) / len(valid_letters)
        pred = valid_letters[0]
        entropy = float((-(scores * torch.log(scores + 1e-8)).sum()).item())
        return scores.detach(), pred, entropy, 0.0
    scores = probs[0, ids]
    scores = scores / scores.sum().clamp_min(1e-8)
    pred_idx = int(torch.argmax(scores).item())
    sorted_scores = torch.sort(scores, descending=True).values
    entropy = float((-(scores * torch.log(scores + 1e-8)).sum()).item())
    margin = (
        float((sorted_scores[0] - sorted_scores[1]).item())
        if len(sorted_scores) > 1
        else 0.0
    )
    return scores.detach(), usable_letters[pred_idx], entropy, margin


epochs = 2
best_state = model.adapter_state()
best_val = 1e9

for epoch in range(1, epochs + 1):
    model.train()
    losses = []
    random.shuffle(train_items)
    for ex in train_items:
        batch = collate_prompt_answer(ex["prompt"], ex["label"])
        batch = {
            k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
        }
        optimizer.zero_grad()
        out = model(**batch)
        loss = out.loss
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    train_loss = float(np.mean(losses)) if losses else 0.0

    model.eval()
    val_losses, val_correct = [], 0
    per_ds_val = {k: {"correct": 0, "n": 0} for k in experiment_data}
    for ex in val_items:
        batch = collate_prompt_answer(ex["prompt"], ex["label"])
        batch = {
            k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
        }
        with torch.no_grad():
            out = model(**batch)
            val_losses.append(float(out.loss.item()))
            _, pred, _, _ = answer_scores(
                model, ex["prompt"], valid_choices_for_item(ex)
            )
            ok = int(pred == ex["label"])
            val_correct += ok
            per_ds_val[ex["dataset"]]["correct"] += ok
            per_ds_val[ex["dataset"]]["n"] += 1
    val_loss = float(np.mean(val_losses)) if val_losses else 0.0
    val_acc = val_correct / max(1, len(val_items))
    print(f"Epoch {epoch}: validation_loss = {val_loss:.4f}")

    for name in experiment_data:
        ds_val_acc = per_ds_val[name]["correct"] / max(1, per_ds_val[name]["n"])
        experiment_data[name]["metrics"]["train"].append((epoch, train_loss))
        experiment_data[name]["metrics"]["val"].append((epoch, ds_val_acc))
        experiment_data[name]["losses"]["train"].append((epoch, train_loss))
        experiment_data[name]["losses"]["val"].append((epoch, val_loss))
    if val_loss < best_val:
        best_val = val_loss
        best_state = model.adapter_state()

model.reset_adapter(best_state)


def make_stream(items, mode="clustered"):
    arr = items.copy()
    if mode == "iid":
        random.shuffle(arr)
        return arr
    if mode == "clustered":
        return sorted(arr, key=lambda z: z["dataset"])
    if mode == "hard_burst":
        scored = []
        for ex in arr:
            _, _, ent, margin = answer_scores(
                model, ex["prompt"], valid_choices_for_item(ex)
            )
            scored.append((ent - margin, ex))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [x[1] for x in scored]
    return arr


def ece_score(conf, corr, bins=10):
    conf = np.asarray(conf)
    corr = np.asarray(corr)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for i in range(bins):
        m = (conf >= edges[i]) & (
            conf < edges[i + 1] if i < bins - 1 else conf <= edges[i + 1]
        )
        if m.any():
            ece += abs(corr[m].mean() - conf[m].mean()) * m.mean()
    return float(ece)


def shift_robust_utility(acc, frozen_acc, ece, frozen_ece, trigger_rate, avg_ms):
    return float(
        (acc - frozen_acc)
        - 0.5 * max(0.0, ece - frozen_ece)
        - 0.1 * trigger_rate
        - 0.001 * avg_ms
    )


def clone_model_with_best_adapter():
    cur = AdapterWrappedLM(model_name, device).to(device)
    cur.reset_adapter(best_state)
    return cur


def stream_eval(
    stream,
    policy="frozen",
    reset_every=12,
    adapt_lr=8e-4,
    entropy_th=1.05,
    margin_th=0.22,
):
    cur = clone_model_with_best_adapter()
    opt = torch.optim.AdamW(cur.adapter.parameters(), lr=adapt_lr)

    preds, gts, confs, corrs, triggers, times_ms, entropies, margins = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    dataset_wise = {}
    buffer = []

    cur.eval()
    for i, ex in enumerate(stream):
        t0 = time.time()
        valid_letters = valid_choices_for_item(ex)
        scores, pred, ent, margin = answer_scores(cur, ex["prompt"], valid_letters)
        conf = float(scores.max().item())
        correct = int(pred == ex["label"])

        preds.append(pred)
        gts.append(ex["label"])
        confs.append(conf)
        corrs.append(correct)
        entropies.append(ent)
        margins.append(margin)

        do_update = False
        if policy == "always":
            do_update = True
        elif policy == "entropy":
            do_update = ent > entropy_th
        elif policy == "margin":
            do_update = margin < margin_th
        elif policy == "hybrid":
            do_update = (ent > entropy_th) and (margin < margin_th)

        triggers.append(int(do_update))

        if do_update:
            buffer.append({"prompt": ex["prompt"], "pseudo": pred})
            if len(buffer) >= 2:
                cur.train()
                opt.zero_grad()
                loss_sum = 0.0
                for b in buffer:
                    batch = collate_prompt_answer(b["prompt"], b["pseudo"])
                    batch = {
                        k: v.to(device)
                        for k, v in batch.items()
                        if isinstance(v, torch.Tensor)
                    }
                    out = cur(**batch)
                    loss_sum = loss_sum + out.loss
                loss_sum = loss_sum / len(buffer)
                loss_sum.backward()
                opt.step()
                buffer = []
                cur.eval()

        if policy != "frozen" and reset_every > 0 and (i + 1) % reset_every == 0:
            cur.reset_adapter(best_state)

        times_ms.append((time.time() - t0) * 1000.0)
        dataset_wise.setdefault(
            ex["dataset"], {"preds": [], "gts": [], "confs": [], "triggers": []}
        )
        dataset_wise[ex["dataset"]]["preds"].append(pred)
        dataset_wise[ex["dataset"]]["gts"].append(ex["label"])
        dataset_wise[ex["dataset"]]["confs"].append(conf)
        dataset_wise[ex["dataset"]]["triggers"].append(int(do_update))

    acc = float(np.mean([p == y for p, y in zip(preds, gts)])) if preds else 0.0
    ece = ece_score(confs, corrs) if preds else 0.0
    return {
        "acc": acc,
        "ece": ece,
        "trigger_rate": float(np.mean(triggers)) if triggers else 0.0,
        "avg_ms": float(np.mean(times_ms)) if times_ms else 0.0,
        "preds": np.array(preds),
        "gts": np.array(gts),
        "confs": np.array(confs),
        "corrs": np.array(corrs),
        "triggers": np.array(triggers),
        "entropies": np.array(entropies),
        "margins": np.array(margins),
        "dataset_wise": dataset_wise,
    }


stream_mode = "clustered"
stream = make_stream(test_items, mode=stream_mode)

frozen = stream_eval(stream, policy="frozen")
always = stream_eval(stream, policy="always")
hybrid = stream_eval(stream, policy="hybrid")

for res in [frozen, always, hybrid]:
    res["srus"] = shift_robust_utility(
        res["acc"],
        frozen["acc"],
        res["ece"],
        frozen["ece"],
        res["trigger_rate"],
        res["avg_ms"],
    )

for ds_name in experiment_data:
    idxs = [i for i, ex in enumerate(stream) if ex["dataset"] == ds_name]
    ds_gt = [stream[i]["label"] for i in idxs]
    metrics = {
        "frozen_acc": (
            float(np.mean([frozen["preds"][j] == ds_gt[k] for k, j in enumerate(idxs)]))
            if idxs
            else 0.0
        ),
        "always_acc": (
            float(np.mean([always["preds"][j] == ds_gt[k] for k, j in enumerate(idxs)]))
            if idxs
            else 0.0
        ),
        "hybrid_acc": (
            float(np.mean([hybrid["preds"][j] == ds_gt[k] for k, j in enumerate(idxs)]))
            if idxs
            else 0.0
        ),
        "frozen_ece": (
            ece_score(
                [float(frozen["confs"][j]) for j in idxs],
                [int(frozen["preds"][j] == ds_gt[k]) for k, j in enumerate(idxs)],
            )
            if idxs
            else 0.0
        ),
        "always_ece": (
            ece_score(
                [float(always["confs"][j]) for j in idxs],
                [int(always["preds"][j] == ds_gt[k]) for k, j in enumerate(idxs)],
            )
            if idxs
            else 0.0
        ),
        "hybrid_ece": (
            ece_score(
                [float(hybrid["confs"][j]) for j in idxs],
                [int(hybrid["preds"][j] == ds_gt[k]) for k, j in enumerate(idxs)],
            )
            if idxs
            else 0.0
        ),
    }
    for epoch in range(1, epochs + 1):
        experiment_data[ds_name]["metrics"]["test"].append((epoch, metrics))
    experiment_data[ds_name]["predictions"] = (
        hybrid["preds"][idxs].tolist() if idxs else []
    )
    experiment_data[ds_name]["ground_truth"] = np.array(ds_gt).tolist()
    experiment_data[ds_name]["confidences"] = (
        hybrid["confs"][idxs].tolist() if idxs else []
    )
    experiment_data[ds_name]["triggers"] = (
        hybrid["triggers"][idxs].tolist() if idxs else []
    )


def save_plot_predictions(ds_name, preds, gts):
    if len(gts) == 0:
        return
    uniq = sorted(list(set(list(gts) + list(preds))))
    mapping = {k: i for i, k in enumerate(uniq)}
    x = np.arange(len(gts))
    plt.figure(figsize=(8, 3))
    plt.plot(x, [mapping[g] for g in gts], marker="o", label="ground_truth")
    plt.plot(x, [mapping[p] for p in preds], marker="x", label="prediction")
    plt.yticks(list(mapping.values()), list(mapping.keys()))
    plt.xlabel("sample")
    plt.ylabel("label")
    plt.title(f"Predictions vs Ground Truth: {ds_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, f"{ds_name}_pred_vs_gt.png"))
    plt.close()


for ds_name in experiment_data:
    save_plot_predictions(
        ds_name,
        experiment_data[ds_name]["predictions"],
        experiment_data[ds_name]["ground_truth"],
    )


def save_stream_plot(name, values, ylabel, filename):
    plt.figure(figsize=(8, 3))
    plt.plot(np.arange(len(values)), values)
    plt.xlabel("stream_index")
    plt.ylabel(ylabel)
    plt.title(name)
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, filename))
    plt.close()


save_stream_plot(
    "Hybrid Entropy", hybrid["entropies"], "entropy", "hybrid_entropy_stream.png"
)
save_stream_plot(
    "Hybrid Margin", hybrid["margins"], "margin", "hybrid_margin_stream.png"
)
save_stream_plot(
    "Hybrid Trigger", hybrid["triggers"], "trigger", "hybrid_trigger_stream.png"
)
save_stream_plot(
    "Hybrid Confidence", hybrid["confs"], "confidence", "hybrid_confidence_stream.png"
)

sample_outputs = {}
gen_model = clone_model_with_best_adapter()
gen_model.eval()
for ds_name in ["allenai_ai2_arc", "qiaojin_PubMedQA", "cais_mmlu"]:
    subset = [ex for ex in stream if ex["dataset"] == ds_name][:3]
    sample_outputs[ds_name] = []
    for ex in subset:
        enc = tokenizer(
            normalize_text(ex["prompt"]),
            return_tensors="pt",
            truncation=True,
            max_length=128,
        )
        enc = {k: v.to(device) for k, v in enc.items() if isinstance(v, torch.Tensor)}
        with torch.no_grad():
            out = gen_model.base.generate(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                max_new_tokens=4,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        sample_outputs[ds_name].append(
            {"prompt": ex["prompt"], "generated": text, "label": ex["label"]}
        )

with open(os.path.join(working_dir, "sample_generations.json"), "w") as f:
    json.dump(sample_outputs, f, indent=2)

np.save(os.path.join(working_dir, "stream_preds_frozen.npy"), frozen["preds"])
np.save(os.path.join(working_dir, "stream_preds_always.npy"), always["preds"])
np.save(os.path.join(working_dir, "stream_preds_hybrid.npy"), hybrid["preds"])
np.save(os.path.join(working_dir, "stream_gt.npy"), hybrid["gts"])
np.save(os.path.join(working_dir, "stream_conf_frozen.npy"), frozen["confs"])
np.save(os.path.join(working_dir, "stream_conf_always.npy"), always["confs"])
np.save(os.path.join(working_dir, "stream_conf_hybrid.npy"), hybrid["confs"])
np.save(os.path.join(working_dir, "stream_entropy.npy"), hybrid["entropies"])
np.save(os.path.join(working_dir, "stream_margin.npy"), hybrid["margins"])
np.save(os.path.join(working_dir, "stream_triggers.npy"), hybrid["triggers"])
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
np.save(
    os.path.join(working_dir, "summary_metrics.npy"),
    {
        "frozen": {
            k: frozen[k] for k in ["acc", "ece", "trigger_rate", "avg_ms", "srus"]
        },
        "always": {
            k: always[k] for k in ["acc", "ece", "trigger_rate", "avg_ms", "srus"]
        },
        "hybrid": {
            k: hybrid[k] for k in ["acc", "ece", "trigger_rate", "avg_ms", "srus"]
        },
    },
    allow_pickle=True,
)

print(
    f"Frozen | acc={frozen['acc']:.4f} ece={frozen['ece']:.4f} trigger={frozen['trigger_rate']:.4f} ms={frozen['avg_ms']:.2f} SRUS={frozen['srus']:.4f}"
)
print(
    f"Always | acc={always['acc']:.4f} ece={always['ece']:.4f} trigger={always['trigger_rate']:.4f} ms={always['avg_ms']:.2f} SRUS={always['srus']:.4f}"
)
print(
    f"Hybrid | acc={hybrid['acc']:.4f} ece={hybrid['ece']:.4f} trigger={hybrid['trigger_rate']:.4f} ms={hybrid['avg_ms']:.2f} SRUS={hybrid['srus']:.4f}"
)
