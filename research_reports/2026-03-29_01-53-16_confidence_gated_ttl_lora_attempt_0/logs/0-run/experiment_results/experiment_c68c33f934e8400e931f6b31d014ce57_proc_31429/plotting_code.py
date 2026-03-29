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

dataset_names = list(exp.keys())
print("Datasets found:", dataset_names)


def load_optional(name):
    path = os.path.join(working_dir, name)
    return np.load(path, allow_pickle=True) if os.path.exists(path) else None


def mean_by_epoch(rows, val_idx):
    out = {}
    for r in rows:
        if len(r) > val_idx:
            ep = int(r[1])
            out.setdefault(ep, []).append(float(r[val_idx]))
    xs = sorted(out.keys())
    ys = [np.mean(out[x]) for x in xs]
    return xs, ys


mode_order = ["frozen", "always", "gated", "gated_reset"]
agg_acc = {m: [] for m in mode_order}
agg_srus = {m: [] for m in mode_order}

for ds_name in dataset_names:
    ds = exp.get(ds_name, {})
    metrics = ds.get("metrics", {})
    losses = ds.get("losses", {})
    preds = np.array(ds.get("predictions", []))
    gt = np.array(ds.get("ground_truth", []))
    test_rows = metrics.get("test", [])

    frozen_cumacc = load_optional(f"{ds_name}_frozen_cumacc.npy")
    always_cumacc = load_optional(f"{ds_name}_always_cumacc.npy")
    gated_cumacc = load_optional(f"{ds_name}_gated_cumacc.npy")
    gated_reset_cumacc = load_optional(f"{ds_name}_gated_reset_cumacc.npy")
    gated_entropy = load_optional(f"{ds_name}_gated_entropy.npy")
    gated_gate = load_optional(f"{ds_name}_gated_gate_flags.npy")
    gated_update = load_optional(f"{ds_name}_gated_update_flags.npy")
    gated_reset = load_optional(f"{ds_name}_gated_reset_flags.npy")

    try:
        tr = metrics.get("train", [])
        va = metrics.get("val", [])
        if tr or va:
            plt.figure(figsize=(8, 5))
            if tr:
                e, y = mean_by_epoch(tr, 2)
                plt.plot(e, y, marker="o", label="Train Accuracy")
            if va:
                e, y = mean_by_epoch(va, 2)
                plt.plot(e, y, marker="o", label="Validation Accuracy")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title(f"{ds_name} Dataset\nTraining and Validation Accuracy")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(working_dir, f"{ds_name}_train_val_accuracy.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating {ds_name} accuracy plot: {e}")
        plt.close()

    try:
        tr = losses.get("train", [])
        va = losses.get("val", [])
        if tr or va:
            plt.figure(figsize=(8, 5))
            if tr:
                e, y = mean_by_epoch(tr, 2)
                plt.plot(e, y, marker="o", label="Train Loss")
            if va:
                e, y = mean_by_epoch(va, 2)
                plt.plot(e, y, marker="o", label="Validation Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.title(f"{ds_name} Dataset\nTraining and Validation Loss")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(working_dir, f"{ds_name}_train_val_loss.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating {ds_name} loss plot: {e}")
        plt.close()

    try:
        if test_rows:
            by_mode = {}
            for r in test_rows:
                mode = r[1]
                by_mode.setdefault(mode, []).append(r)
            modes = [m for m in mode_order if m in by_mode]
            acc = [np.mean([x[2] for x in by_mode[m]]) for m in modes]
            ece = [np.mean([x[3] for x in by_mode[m]]) for m in modes]
            trig = [np.mean([x[4] for x in by_mode[m]]) for m in modes]
            upd = [np.mean([x[6] for x in by_mode[m]]) for m in modes]
            srus = [np.mean([x[8] for x in by_mode[m]]) for m in modes]
            for m, a, s in zip(modes, acc, srus):
                agg_acc[m].append(a)
                agg_srus[m].append(s)
            x = np.arange(len(modes))
            plt.figure(figsize=(12, 8))
            plt.subplot(2, 3, 1)
            plt.bar(x, acc)
            plt.xticks(x, modes, rotation=20)
            plt.ylabel("Accuracy")
            plt.title(f"{ds_name} Dataset\nTest Accuracy by Mode")
            plt.subplot(2, 3, 2)
            plt.bar(x, ece)
            plt.xticks(x, modes, rotation=20)
            plt.ylabel("ECE")
            plt.title(f"{ds_name} Dataset\nTest ECE by Mode")
            plt.subplot(2, 3, 3)
            plt.bar(x, trig)
            plt.xticks(x, modes, rotation=20)
            plt.ylabel("Trigger Rate")
            plt.title(f"{ds_name} Dataset\nTrigger Rate by Mode")
            plt.subplot(2, 3, 4)
            plt.bar(x, upd)
            plt.xticks(x, modes, rotation=20)
            plt.ylabel("Update Rate")
            plt.title(f"{ds_name} Dataset\nUpdate Rate by Mode")
            plt.subplot(2, 3, 5)
            plt.bar(x, srus)
            plt.xticks(x, modes, rotation=20)
            plt.ylabel("SRUS")
            plt.title(f"{ds_name} Dataset\nSRUS by Mode")
            plt.tight_layout()
            plt.savefig(os.path.join(working_dir, f"{ds_name}_test_mode_metrics.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating {ds_name} test metrics plot: {e}")
        plt.close()

    try:
        if any(
            v is not None
            for v in [frozen_cumacc, always_cumacc, gated_cumacc, gated_reset_cumacc]
        ):
            plt.figure(figsize=(8, 5))
            if frozen_cumacc is not None:
                plt.plot(frozen_cumacc, label="Frozen")
            if always_cumacc is not None:
                plt.plot(always_cumacc, label="Always")
            if gated_cumacc is not None:
                plt.plot(gated_cumacc, label="Gated")
            if gated_reset_cumacc is not None:
                plt.plot(gated_reset_cumacc, label="Gated Reset")
            plt.xlabel("Stream Step")
            plt.ylabel("Cumulative Accuracy")
            plt.title(
                f"{ds_name} Dataset\nCumulative Stream Accuracy by Adaptation Mode"
            )
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                os.path.join(working_dir, f"{ds_name}_cumulative_stream_accuracy.png")
            )
        plt.close()
    except Exception as e:
        print(f"Error creating {ds_name} cumulative accuracy plot: {e}")
        plt.close()

    try:
        if gated_entropy is not None:
            plt.figure(figsize=(10, 4))
            x = np.arange(len(gated_entropy))
            plt.plot(x, gated_entropy, label="Entropy")
            if gated_gate is not None:
                idx = np.where(gated_gate > 0)[0]
                if len(idx):
                    plt.scatter(idx, gated_entropy[idx], s=10, label="Gate Fired")
            if gated_update is not None:
                idx = np.where(gated_update > 0)[0]
                if len(idx):
                    plt.scatter(idx, gated_entropy[idx], s=10, label="Updated")
            if gated_reset is not None:
                idx = np.where(gated_reset > 0)[0]
                if len(idx):
                    plt.scatter(idx, gated_entropy[idx], s=16, label="Reset")
            plt.xlabel("Stream Step")
            plt.ylabel("Entropy")
            plt.title(
                f"{ds_name} Dataset\nGated Entropy with Gate, Update, and Reset Events"
            )
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                os.path.join(working_dir, f"{ds_name}_gated_entropy_events.png")
            )
        plt.close()
    except Exception as e:
        print(f"Error creating {ds_name} entropy/events plot: {e}")
        plt.close()

    try:
        if gt.size > 0 and preds.size > 0:
            n = min(200, len(gt))
            plt.figure(figsize=(10, 4))
            plt.plot(np.arange(n), gt[:n], label="Ground Truth", linewidth=1.5)
            plt.plot(
                np.arange(n),
                preds[:n],
                label="Gated Predictions",
                linewidth=1.0,
                alpha=0.8,
            )
            plt.xlabel("Stream Step")
            plt.ylabel("Class Label")
            plt.title(
                f"{ds_name} Dataset\nLeft: Ground Truth, Right: Generated Samples (Predicted Labels Trace)"
            )
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    working_dir,
                    f"{ds_name}_gated_predictions_vs_ground_truth_prefix.png",
                )
            )
        plt.close()
    except Exception as e:
        print(f"Error creating {ds_name} prediction trace plot: {e}")
        plt.close()

    try:
        if gt.size > 0 and preds.size > 0:
            cm = np.zeros((2, 2), dtype=int)
            for g, p in zip(gt.astype(int), preds.astype(int)):
                if g in [0, 1] and p in [0, 1]:
                    cm[g, p] += 1
            plt.figure(figsize=(5, 4))
            plt.imshow(cm, cmap="Blues")
            for i in range(2):
                for j in range(2):
                    plt.text(j, i, str(cm[i, j]), ha="center", va="center")
            plt.xticks([0, 1], ["Pred 0", "Pred 1"])
            plt.yticks([0, 1], ["True 0", "True 1"])
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            plt.title(f"{ds_name} Dataset\nGated Mode Confusion Matrix")
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(
                os.path.join(working_dir, f"{ds_name}_gated_confusion_matrix.png")
            )
        plt.close()
    except Exception as e:
        print(f"Error creating {ds_name} confusion matrix plot: {e}")
        plt.close()

try:
    valid_modes = [m for m in mode_order if len(agg_acc[m]) > 0]
    if dataset_names and valid_modes:
        x = np.arange(len(dataset_names))
        width = 0.2
        plt.figure(figsize=(10, 5))
        for i, m in enumerate(valid_modes):
            vals = [
                (
                    np.mean([r[2] for r in exp[ds]["metrics"]["test"] if r[1] == m])
                    if exp[ds]["metrics"]["test"]
                    else np.nan
                )
                for ds in dataset_names
            ]
            plt.bar(
                x + (i - (len(valid_modes) - 1) / 2) * width, vals, width=width, label=m
            )
        plt.xticks(x, dataset_names, rotation=20)
        plt.ylabel("Mean Test Accuracy")
        plt.title("All Datasets\nComparison of Mean Test Accuracy by Adaptation Mode")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "all_datasets_mean_test_accuracy_by_mode.png")
        )
    plt.close()
except Exception as e:
    print(f"Error creating cross-dataset accuracy plot: {e}")
    plt.close()

try:
    valid_modes = [m for m in mode_order if len(agg_srus[m]) > 0]
    if dataset_names and valid_modes:
        x = np.arange(len(dataset_names))
        width = 0.2
        plt.figure(figsize=(10, 5))
        for i, m in enumerate(valid_modes):
            vals = [
                (
                    np.mean([r[8] for r in exp[ds]["metrics"]["test"] if r[1] == m])
                    if exp[ds]["metrics"]["test"]
                    else np.nan
                )
                for ds in dataset_names
            ]
            plt.bar(
                x + (i - (len(valid_modes) - 1) / 2) * width, vals, width=width, label=m
            )
        plt.xticks(x, dataset_names, rotation=20)
        plt.ylabel("Mean SRUS")
        plt.title("All Datasets\nComparison of Mean SRUS by Adaptation Mode")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "all_datasets_mean_srus_by_mode.png"))
    plt.close()
except Exception as e:
    print(f"Error creating cross-dataset SRUS plot: {e}")
    plt.close()

for ds_name in dataset_names:
    rows = exp[ds_name].get("metrics", {}).get("test", [])
    if rows:
        modes = sorted(set(r[1] for r in rows))
        print(ds_name)
        for m in modes:
            sub = [r for r in rows if r[1] == m]
            print(
                m,
                {
                    "acc_mean": float(np.mean([r[2] for r in sub])),
                    "ece_mean": float(np.mean([r[3] for r in sub])),
                    "trigger_mean": float(np.mean([r[4] for r in sub])),
                    "update_mean": float(np.mean([r[6] for r in sub])),
                    "srus_mean": float(np.mean([r[8] for r in sub])),
                },
            )
