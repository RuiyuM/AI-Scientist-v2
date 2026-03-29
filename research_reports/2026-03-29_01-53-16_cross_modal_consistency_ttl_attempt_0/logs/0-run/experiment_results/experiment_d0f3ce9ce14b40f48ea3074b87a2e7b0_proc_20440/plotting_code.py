import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

exp_path = os.path.join(working_dir, "experiment_data.npy")
try:
    experiment_data = np.load(exp_path, allow_pickle=True).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

ds_name = "synthetic_multimodal"
ds = experiment_data.get(ds_name, {})
metrics = ds.get("metrics", {})
losses = ds.get("losses", {})
stream_details = ds.get("stream_details", {})
preds = np.array(ds.get("predictions", []))
gts = np.array(ds.get("ground_truth", []))


def moving_avg(x, k=25):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    if len(x) < k:
        return x
    return np.convolve(x, np.ones(k) / k, mode="valid")


try:
    tr = losses.get("train", [])
    va = losses.get("val", [])
    if tr or va:
        plt.figure(figsize=(7, 4))
        if tr:
            ep, val = zip(*tr)
            plt.plot(ep, val, marker="o", label="Train Loss")
        if va:
            ep, val = zip(*va)
            plt.plot(ep, val, marker="o", label="Validation Loss")
        plt.title(
            "Synthetic Multimodal Dataset — Training/Validation Loss\nSubtitle: Source train vs validation epochs"
        )
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, f"{ds_name}_loss_curves.png"))
        plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

try:
    tr = metrics.get("train", [])
    va = metrics.get("val", [])
    if tr or va:
        plt.figure(figsize=(7, 4))
        if tr:
            ep, val = zip(*tr)
            plt.plot(ep, val, marker="o", label="Train Accuracy")
        if va:
            ep, val = zip(*va)
            plt.plot(ep, val, marker="o", label="Validation Accuracy")
        plt.title(
            "Synthetic Multimodal Dataset — Training/Validation Accuracy\nSubtitle: Source train vs validation epochs"
        )
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1.05)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, f"{ds_name}_accuracy_curves.png"))
        plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

try:
    stream = metrics.get("stream", [])
    if stream:
        last = stream[-1]
        labels = [
            "Frozen Acc",
            "Tent Acc",
            "Cons Acc",
            "Frozen SSAA",
            "Tent SSAA",
            "Cons SSAA",
            "Tent Adapt Freq",
            "Cons Adapt Freq",
        ]
        values = [
            last.get("frozen_accuracy", np.nan),
            last.get("tent_accuracy", np.nan),
            last.get("cons_accuracy", np.nan),
            last.get("frozen_ssaa", np.nan),
            last.get("tent_ssaa", np.nan),
            last.get("cons_ssaa", np.nan),
            last.get("tent_adapt_freq", np.nan),
            last.get("cons_adapt_freq", np.nan),
        ]
        plt.figure(figsize=(10, 4))
        plt.bar(np.arange(len(labels)), values)
        plt.xticks(np.arange(len(labels)), labels, rotation=30, ha="right")
        plt.title(
            "Synthetic Multimodal Dataset — Final Stream Metrics Summary\nSubtitle: Accuracy, SSAA, and adaptation frequency by method"
        )
        plt.ylabel("Value")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, f"{ds_name}_stream_metric_summary.png"))
        plt.close()
except Exception as e:
    print(f"Error creating stream summary plot: {e}")
    plt.close()

try:
    fa = np.asarray(stream_details.get("frozen_acc_stream", []))
    ta = np.asarray(stream_details.get("tent_acc_stream", []))
    ca = np.asarray(stream_details.get("cons_acc_stream", []))
    if fa.size or ta.size or ca.size:
        plt.figure(figsize=(8, 4))
        if fa.size:
            plt.plot(moving_avg(fa), label="Frozen")
        if ta.size:
            plt.plot(moving_avg(ta), label="Entropy Adaptation")
        if ca.size:
            plt.plot(moving_avg(ca), label="Consistency Adaptation")
        plt.title(
            "Synthetic Multimodal Dataset — Stream Accuracy Over Time\nSubtitle: Moving-average target-stream accuracy by adaptation mode"
        )
        plt.xlabel("Stream Step")
        plt.ylabel("Moving Accuracy")
        plt.ylim(0, 1.05)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, f"{ds_name}_stream_accuracy.png"))
        plt.close()
except Exception as e:
    print(f"Error creating stream accuracy plot: {e}")
    plt.close()

try:
    tu = np.asarray(stream_details.get("tent_update_stream", []))
    cu = np.asarray(stream_details.get("cons_update_stream", []))
    if tu.size or cu.size:
        plt.figure(figsize=(8, 4))
        if tu.size:
            plt.plot(tu, label="Entropy Updates", alpha=0.8)
        if cu.size:
            plt.plot(cu, label="Consistency Updates", alpha=0.8)
        plt.title(
            "Synthetic Multimodal Dataset — Adaptation Update Events\nSubtitle: Target-stream update triggers for entropy and consistency modes"
        )
        plt.xlabel("Stream Step")
        plt.ylabel("Update Event")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, f"{ds_name}_adaptation_updates.png"))
        plt.close()
except Exception as e:
    print(f"Error creating update plot: {e}")
    plt.close()

try:
    te = np.asarray(stream_details.get("tent_entropy_stream", []))
    ce = np.asarray(stream_details.get("cons_entropy_stream", []))
    if te.size or ce.size:
        plt.figure(figsize=(8, 4))
        if te.size:
            plt.plot(te, label="Entropy Mode")
        if ce.size:
            plt.plot(ce, label="Consistency Mode")
        plt.title(
            "Synthetic Multimodal Dataset — Prediction Entropy Across Stream\nSubtitle: Per-sample answer entropy on target stream"
        )
        plt.xlabel("Stream Step")
        plt.ylabel("Entropy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, f"{ds_name}_entropy_stream.png"))
        plt.close()
except Exception as e:
    print(f"Error creating entropy plot: {e}")
    plt.close()

try:
    if preds.size and gts.size and preds.shape == gts.shape:
        acc = float(np.mean(preds == gts))
        cm = np.zeros((2, 2), dtype=int)
        for gt, pr in zip(gts, preds):
            if gt in [0, 1] and pr in [0, 1]:
                cm[int(gt), int(pr)] += 1
        plt.figure(figsize=(4.5, 4))
        plt.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                plt.text(j, i, str(cm[i, j]), ha="center", va="center")
        plt.xticks([0, 1], ["Pred 0", "Pred 1"])
        plt.yticks([0, 1], ["True 0", "True 1"])
        plt.title(
            "Synthetic Multimodal Dataset — Consistency-Mode Confusion Matrix\nSubtitle: Left axis ground truth, bottom axis predictions"
        )
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, f"{ds_name}_confusion_matrix_consistency.png")
        )
        plt.close()
        print(f"consistency_prediction_accuracy={acc:.4f}")
except Exception as e:
    print(f"Error creating confusion matrix plot: {e}")
    plt.close()

stream = metrics.get("stream", [])
if stream:
    last = stream[-1]
    print(
        f"frozen_accuracy={last.get('frozen_accuracy', np.nan):.4f}, "
        f"frozen_ssaa={last.get('frozen_ssaa', np.nan):.4f}, "
        f"tent_accuracy={last.get('tent_accuracy', np.nan):.4f}, "
        f"tent_ssaa={last.get('tent_ssaa', np.nan):.4f}, "
        f"tent_adapt_freq={last.get('tent_adapt_freq', np.nan):.4f}, "
        f"cons_accuracy={last.get('cons_accuracy', np.nan):.4f}, "
        f"cons_ssaa={last.get('cons_ssaa', np.nan):.4f}, "
        f"cons_adapt_freq={last.get('cons_adapt_freq', np.nan):.4f}"
    )
