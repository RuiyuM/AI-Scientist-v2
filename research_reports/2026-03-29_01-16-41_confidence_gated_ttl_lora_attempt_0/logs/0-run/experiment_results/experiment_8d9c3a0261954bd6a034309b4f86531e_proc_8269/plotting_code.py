import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

ds_name = "synthetic_shift_stream"
ds = experiment_data.get(ds_name, {})
metrics = ds.get("metrics", {})
losses = ds.get("losses", {})
preds = ds.get("predictions", {})
gt = np.array(ds.get("ground_truth", []))
segment_ids = np.array(ds.get("segment_ids", []))
entropies = ds.get("entropies", {})
segment_accs = ds.get("segment_accs", {})

stream_metrics = metrics.get("stream", [])
if len(stream_metrics) > 0 and isinstance(stream_metrics[0], dict):
    overall = stream_metrics[0]
    for k, v in overall.items():
        print(f"{k}_shift_robust_acc: {float(v):.4f}")
for k, v in segment_accs.items():
    arr = np.array(v)
    if arr.size:
        print(f"{k}_mean_segment_acc: {arr.mean():.4f}")

try:
    tr = metrics.get("train", [])
    va = metrics.get("val", [])
    if len(tr) and len(va):
        tr = np.array(tr, dtype=object)
        va = np.array(va, dtype=object)
        plt.figure(figsize=(6, 4))
        plt.plot(tr[:, 0].astype(float), tr[:, 1].astype(float), label="Train Accuracy")
        plt.plot(
            va[:, 0].astype(float), va[:, 1].astype(float), label="Validation Accuracy"
        )
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Synthetic Shift Stream Dataset\nTraining and Validation Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, f"{ds_name}_train_val_accuracy.png"))
        plt.close()
except Exception as e:
    print(f"Error creating train/val accuracy plot: {e}")
    plt.close()

try:
    tr = losses.get("train", [])
    va = losses.get("val", [])
    if len(tr) and len(va):
        tr = np.array(tr, dtype=object)
        va = np.array(va, dtype=object)
        plt.figure(figsize=(6, 4))
        plt.plot(tr[:, 0].astype(float), tr[:, 1].astype(float), label="Train Loss")
        plt.plot(
            va[:, 0].astype(float), va[:, 1].astype(float), label="Validation Loss"
        )
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Synthetic Shift Stream Dataset\nTraining and Validation Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, f"{ds_name}_train_val_loss.png"))
        plt.close()
except Exception as e:
    print(f"Error creating train/val loss plot: {e}")
    plt.close()

try:
    if isinstance(overall, dict) and len(overall):
        keys = list(overall.keys())
        vals = [float(overall[k]) for k in keys]
        plt.figure(figsize=(6, 4))
        plt.bar(keys, vals)
        plt.ylim(0, 1)
        plt.ylabel("Shift-Robust Accuracy")
        plt.title(
            "Synthetic Shift Stream Dataset\nOverall Stream Accuracy by Adaptation Mode"
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, f"{ds_name}_overall_stream_accuracy_bar.png")
        )
        plt.close()
except Exception as e:
    print(f"Error creating overall stream accuracy plot: {e}")
    plt.close()

try:
    if isinstance(segment_accs, dict) and len(segment_accs):
        plt.figure(figsize=(7, 4))
        for k, v in segment_accs.items():
            arr = np.array(v, dtype=float)
            if arr.size:
                plt.plot(np.arange(arr.size), arr, marker="o", label=k)
        plt.xlabel("Stream Segment")
        plt.ylabel("Accuracy")
        plt.title(
            "Synthetic Shift Stream Dataset\nSegment-wise Accuracy Across Adaptation Modes"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, f"{ds_name}_segment_wise_accuracy.png"))
        plt.close()
except Exception as e:
    print(f"Error creating segment accuracy plot: {e}")
    plt.close()

try:
    if isinstance(entropies, dict) and len(entropies):
        plt.figure(figsize=(8, 4))
        for k, v in entropies.items():
            arr = np.array(v, dtype=float)
            if arr.size:
                plt.plot(arr, label=k, alpha=0.8)
        plt.xlabel("Stream Sample Index")
        plt.ylabel("Predictive Entropy")
        plt.title(
            "Synthetic Shift Stream Dataset\nEntropy over Evaluation Stream by Adaptation Mode"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, f"{ds_name}_entropy_over_stream.png"))
        plt.close()
except Exception as e:
    print(f"Error creating entropy plot: {e}")
    plt.close()

try:
    if gt.size and segment_ids.size:
        uniq = np.unique(segment_ids)
        counts = [np.sum(segment_ids == s) for s in uniq]
        plt.figure(figsize=(6, 4))
        plt.bar([str(int(u)) for u in uniq], counts)
        plt.xlabel("Segment ID")
        plt.ylabel("Number of Samples")
        plt.title(
            "Synthetic Shift Stream Dataset\nEvaluation Stream Samples per Segment"
        )
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, f"{ds_name}_samples_per_segment.png"))
        plt.close()
except Exception as e:
    print(f"Error creating samples-per-segment plot: {e}")
    plt.close()


def confusion_matrix(y_true, y_pred, n_classes=None):
    y_true, y_pred = np.asarray(y_true, int), np.asarray(y_pred, int)
    if n_classes is None:
        n_classes = int(
            max(y_true.max() if y_true.size else 0, y_pred.max() if y_pred.size else 0)
            + 1
        )
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


for mode in ["no_adapt", "always_on", "gated"]:
    try:
        if gt.size and isinstance(preds, dict) and mode in preds:
            yp = np.array(preds[mode])
            if yp.size == gt.size:
                cm = confusion_matrix(gt, yp)
                plt.figure(figsize=(5, 4))
                plt.imshow(cm, interpolation="nearest")
                plt.colorbar()
                plt.xlabel("Predicted Label")
                plt.ylabel("True Label")
                plt.title(
                    f"Synthetic Shift Stream Dataset\nConfusion Matrix for {mode}"
                )
                plt.tight_layout()
                plt.savefig(
                    os.path.join(working_dir, f"{ds_name}_{mode}_confusion_matrix.png")
                )
                plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix for {mode}: {e}")
        plt.close()
