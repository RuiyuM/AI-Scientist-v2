import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

exp = {}
try:
    exp = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

ds_name = "synthetic_reasoning_stream"
ds = exp.get(ds_name, {})
metrics = ds.get("metrics", {})
losses = ds.get("losses", {})
stream_results = ds.get("stream_results", {})


def load_optional(name):
    path = os.path.join(working_dir, name)
    return np.load(path, allow_pickle=True) if os.path.exists(path) else None


frozen_cumacc = load_optional("frozen_cumacc.npy")
always_cumacc = load_optional("always_cumacc.npy")
gated_cumacc = load_optional("gated_cumacc.npy")
gated_entropy = load_optional("gated_entropy.npy")
gated_triggered = load_optional("gated_triggered.npy")
ground_truth = load_optional("ground_truth.npy")
gated_preds = load_optional("gated_preds.npy")

try:
    tr = metrics.get("train", [])
    va = metrics.get("val", [])
    if tr or va:
        plt.figure(figsize=(8, 5))
        if tr:
            e, y = zip(*tr)
            plt.plot(e, y, marker="o", label="Train Accuracy")
        if va:
            e, y = zip(*va)
            plt.plot(e, y, marker="o", label="Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Synthetic Reasoning Dataset\nTraining and Validation Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, f"{ds_name}_train_val_accuracy.png"))
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

try:
    tr = losses.get("train", [])
    va = losses.get("val", [])
    if tr or va:
        plt.figure(figsize=(8, 5))
        if tr:
            e, y = zip(*tr)
            plt.plot(e, y, marker="o", label="Train Loss")
        if va:
            e, y = zip(*va)
            plt.plot(e, y, marker="o", label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("Synthetic Reasoning Dataset\nTraining and Validation Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, f"{ds_name}_train_val_loss.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

try:
    if stream_results:
        modes = [m for m in ["frozen", "always", "gated"] if m in stream_results]
        accs = [stream_results[m].get("Shifted-Stream Accuracy", np.nan) for m in modes]
        trigs = [stream_results[m].get("trigger_rate", np.nan) for m in modes]
        x = np.arange(len(modes))
        plt.figure(figsize=(9, 4))
        plt.subplot(1, 2, 1)
        plt.bar(x, accs)
        plt.xticks(x, modes)
        plt.ylabel("Accuracy")
        plt.title("Synthetic Shifted Dataset\nShifted-Stream Accuracy by Mode")
        plt.subplot(1, 2, 2)
        plt.bar(x, trigs)
        plt.xticks(x, modes)
        plt.ylabel("Trigger Rate")
        plt.title("Synthetic Shifted Dataset\nAdaptation Trigger Rate by Mode")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, f"{ds_name}_stream_results_bar.png"))
    plt.close()
except Exception as e:
    print(f"Error creating stream results bar plot: {e}")
    plt.close()

try:
    if any(v is not None for v in [frozen_cumacc, always_cumacc, gated_cumacc]):
        plt.figure(figsize=(8, 5))
        if frozen_cumacc is not None:
            plt.plot(frozen_cumacc, label="Frozen")
        if always_cumacc is not None:
            plt.plot(always_cumacc, label="Always-LoRA")
        if gated_cumacc is not None:
            plt.plot(gated_cumacc, label="Gated-LoRA")
        plt.xlabel("Shifted Stream Step")
        plt.ylabel("Cumulative Accuracy")
        plt.title(
            "Synthetic Shifted Dataset\nCumulative Stream Accuracy by Adaptation Mode"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, f"{ds_name}_cumulative_stream_accuracy.png")
        )
    plt.close()
except Exception as e:
    print(f"Error creating cumulative accuracy plot: {e}")
    plt.close()

try:
    if ground_truth is not None and gated_preds is not None:
        n = min(200, len(ground_truth))
        plt.figure(figsize=(10, 4))
        plt.plot(np.arange(n), ground_truth[:n], label="Ground Truth", linewidth=1.5)
        plt.plot(
            np.arange(n),
            gated_preds[:n],
            label="Gated Predictions",
            linewidth=1.0,
            alpha=0.8,
        )
        plt.xlabel("Shifted Stream Step")
        plt.ylabel("Class Label")
        plt.title(
            "Synthetic Shifted Dataset\nLeft: Ground Truth, Right: Generated Samples (Predicted Labels Trace)"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                working_dir, f"{ds_name}_gated_predictions_vs_ground_truth_prefix.png"
            )
        )
    plt.close()
except Exception as e:
    print(f"Error creating prediction trace plot: {e}")
    plt.close()

try:
    if ground_truth is not None and gated_preds is not None:
        cm = np.zeros((2, 2), dtype=int)
        for g, p in zip(ground_truth.astype(int), gated_preds.astype(int)):
            if g in [0, 1] and p in [0, 1]:
                cm[g, p] += 1
        plt.figure(figsize=(5, 4))
        plt.imshow(cm, cmap="Blues")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, str(cm[i, j]), ha="center", va="center")
        plt.xticks([0, 1], ["Pred 0", "Pred 1"])
        plt.yticks([0, 1], ["True 0", "True 1"])
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Synthetic Shifted Dataset\nGated Mode Confusion Matrix")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, f"{ds_name}_gated_confusion_matrix.png"))
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix plot: {e}")
    plt.close()

try:
    if gated_entropy is not None and gated_triggered is not None:
        plt.figure(figsize=(10, 4))
        x = np.arange(len(gated_entropy))
        plt.plot(x, gated_entropy, label="Entropy")
        idx = np.where(gated_triggered > 0)[0]
        if len(idx) > 0:
            plt.scatter(idx, gated_entropy[idx], s=10, label="Triggered Updates")
        plt.xlabel("Shifted Stream Step")
        plt.ylabel("Entropy")
        plt.title(
            "Synthetic Shifted Dataset\nGated Adaptation Entropy and Trigger Events"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, f"{ds_name}_gated_entropy_triggers.png"))
    plt.close()
except Exception as e:
    print(f"Error creating entropy/trigger plot: {e}")
    plt.close()

test_metrics = metrics.get("test", [])
if test_metrics:
    print("Test metrics:", test_metrics[-1][1])
if stream_results:
    print("Stream results:")
    for k, v in stream_results.items():
        print(k, v)
