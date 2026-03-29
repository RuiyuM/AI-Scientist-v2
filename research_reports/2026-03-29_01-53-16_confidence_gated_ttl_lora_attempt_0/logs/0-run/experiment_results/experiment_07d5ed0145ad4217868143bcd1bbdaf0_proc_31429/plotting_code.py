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


def load_optional(name):
    path = os.path.join(working_dir, name)
    return np.load(path, allow_pickle=True) if os.path.exists(path) else None


def unpack_metric_curve(curve, idx_val=1):
    xs, ys = [], []
    for t in curve:
        if isinstance(t, (list, tuple)) and len(t) > idx_val:
            xs.append(t[0])
            ys.append(t[idx_val])
    return xs, ys


for ds_name in dataset_names:
    ds = exp.get(ds_name, {})
    metrics = ds.get("metrics", {})
    losses = ds.get("losses", {})
    extra = ds.get("extra", {})
    gt = np.array(ds.get("ground_truth", []))
    preds = np.array(ds.get("predictions", []))

    try:
        tr = metrics.get("train", [])
        va = metrics.get("val", [])
        plt.figure(figsize=(8, 5))
        ok = False
        if tr:
            e, y = unpack_metric_curve(tr, 1)
            if e:
                plt.plot(e, y, marker="o", label="Train Accuracy")
                ok = True
        if va:
            e, y = unpack_metric_curve(va, 1)
            if e:
                plt.plot(e, y, marker="o", label="Validation Accuracy")
                ok = True
        if ok:
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
        plt.figure(figsize=(8, 5))
        ok = False
        if tr:
            e, y = unpack_metric_curve(tr, 1)
            if e:
                plt.plot(e, y, marker="o", label="Train Loss")
                ok = True
        if va:
            e, y = unpack_metric_curve(va, 1)
            if e:
                plt.plot(e, y, marker="o", label="Validation Loss")
                ok = True
        if ok:
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
        test = metrics.get("test", [])
        rows = [r for r in test if isinstance(r, (list, tuple)) and len(r) >= 7]
        if rows:
            modes = [r[0] for r in rows]
            accs = [r[1] for r in rows]
            eces = [r[2] for r in rows]
            srus = [r[6] for r in rows]
            x = np.arange(len(modes))
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 3, 1)
            plt.bar(x, accs)
            plt.xticks(x, modes, rotation=20)
            plt.ylabel("Accuracy")
            plt.title(f"{ds_name} Dataset\nTest Accuracy by Mode")
            plt.subplot(1, 3, 2)
            plt.bar(x, eces)
            plt.xticks(x, modes, rotation=20)
            plt.ylabel("ECE")
            plt.title(f"{ds_name} Dataset\nTest ECE by Mode")
            plt.subplot(1, 3, 3)
            plt.bar(x, srus)
            plt.xticks(x, modes, rotation=20)
            plt.ylabel("SRUS")
            plt.title(f"{ds_name} Dataset\nTest SRUS by Mode")
            plt.tight_layout()
            plt.savefig(os.path.join(working_dir, f"{ds_name}_test_metrics_bar.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating {ds_name} test metrics bar plot: {e}")
        plt.close()

    try:
        plt.figure(figsize=(8, 5))
        ok = False
        for mode in ["frozen", "always", "gated", "reset"]:
            arr = load_optional(f"{ds_name}_{mode}_cumacc.npy")
            if arr is not None:
                plt.plot(arr, label=mode)
                ok = True
        if ok:
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
        ent = load_optional(f"{ds_name}_gated_entropy.npy")
        gate = load_optional(f"{ds_name}_gated_gate_flags.npy")
        upd = load_optional(f"{ds_name}_gated_update_flags.npy")
        rst = load_optional(f"{ds_name}_gated_reset_flags.npy")
        if ent is not None:
            plt.figure(figsize=(10, 4))
            x = np.arange(len(ent))
            plt.plot(x, ent, label="Gated Entropy")
            if gate is not None:
                idx = np.where(np.array(gate) > 0)[0]
                if len(idx):
                    plt.scatter(idx, ent[idx], s=10, label="Gate Fired")
            if upd is not None:
                idx = np.where(np.array(upd) > 0)[0]
                if len(idx):
                    plt.scatter(idx, ent[idx], s=12, marker="x", label="Updated")
            if rst is not None:
                idx = np.where(np.array(rst) > 0)[0]
                if len(idx):
                    plt.scatter(idx, ent[idx], s=20, marker="|", label="Reset")
            plt.xlabel("Stream Step")
            plt.ylabel("Entropy")
            plt.title(
                f"{ds_name} Dataset\nGated Uncertainty, Trigger, Update, and Reset Events"
            )
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                os.path.join(working_dir, f"{ds_name}_gated_uncertainty_events.png")
            )
        plt.close()
    except Exception as e:
        print(f"Error creating {ds_name} gated uncertainty plot: {e}")
        plt.close()

    try:
        conf = load_optional(f"{ds_name}_gated_conf.npy")
        margin = load_optional(f"{ds_name}_gated_margin.npy")
        if conf is not None and margin is not None:
            plt.figure(figsize=(10, 4))
            x = np.arange(len(conf))
            plt.subplot(1, 2, 1)
            plt.plot(x, conf)
            plt.xlabel("Stream Step")
            plt.ylabel("Confidence")
            plt.title(f"{ds_name} Dataset\nGated Confidence over Stream")
            plt.subplot(1, 2, 2)
            plt.plot(x, margin)
            plt.xlabel("Stream Step")
            plt.ylabel("Margin")
            plt.title(f"{ds_name} Dataset\nGated Margin over Stream")
            plt.tight_layout()
            plt.savefig(
                os.path.join(working_dir, f"{ds_name}_gated_confidence_margin.png")
            )
        plt.close()
    except Exception as e:
        print(f"Error creating {ds_name} confidence/margin plot: {e}")
        plt.close()

    try:
        if gt.size and preds.size:
            n = min(200, len(gt), len(preds))
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
        if gt.size and preds.size:
            labels = sorted(set(np.unique(gt).tolist() + np.unique(preds).tolist()))
            if labels:
                m = {lab: i for i, lab in enumerate(labels)}
                cm = np.zeros((len(labels), len(labels)), dtype=int)
                for g, p in zip(gt.astype(int), preds.astype(int)):
                    if g in m and p in m:
                        cm[m[g], m[p]] += 1
                plt.figure(figsize=(5, 4))
                plt.imshow(cm, cmap="Blues")
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        plt.text(j, i, str(cm[i, j]), ha="center", va="center")
                plt.xticks(range(len(labels)), [f"Pred {l}" for l in labels])
                plt.yticks(range(len(labels)), [f"True {l}" for l in labels])
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
    plt.figure(figsize=(8, 5))
    ok = False
    for ds_name in dataset_names:
        va = exp.get(ds_name, {}).get("metrics", {}).get("val", [])
        e, y = unpack_metric_curve(va, 1)
        if e:
            plt.plot(e, y, marker="o", label=ds_name)
            ok = True
    if ok:
        plt.xlabel("Epoch")
        plt.ylabel("Validation Accuracy")
        plt.title("All Datasets Comparison\nValidation Accuracy Across Datasets")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "all_datasets_validation_accuracy_comparison.png")
        )
    plt.close()
except Exception as e:
    print(f"Error creating cross-dataset validation accuracy plot: {e}")
    plt.close()

try:
    plt.figure(figsize=(8, 5))
    ok = False
    for ds_name in dataset_names:
        va = exp.get(ds_name, {}).get("losses", {}).get("val", [])
        e, y = unpack_metric_curve(va, 1)
        if e:
            plt.plot(e, y, marker="o", label=ds_name)
            ok = True
    if ok:
        plt.xlabel("Epoch")
        plt.ylabel("Validation Loss")
        plt.title("All Datasets Comparison\nValidation Loss Across Datasets")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "all_datasets_validation_loss_comparison.png")
        )
    plt.close()
except Exception as e:
    print(f"Error creating cross-dataset validation loss plot: {e}")
    plt.close()

try:
    modes = ["frozen", "always", "gated", "reset"]
    width = 0.2
    x = np.arange(len(dataset_names))
    plt.figure(figsize=(10, 5))
    ok = False
    for i, mode in enumerate(modes):
        vals = []
        for ds_name in dataset_names:
            rows = exp.get(ds_name, {}).get("metrics", {}).get("test", [])
            val = np.nan
            for r in rows:
                if isinstance(r, (list, tuple)) and len(r) >= 7 and r[0] == mode:
                    val = r[6]
            vals.append(val)
        if np.isfinite(np.array(vals, dtype=float)).any():
            plt.bar(x + (i - 1.5) * width, vals, width=width, label=mode)
            ok = True
    if ok:
        plt.xticks(x, dataset_names, rotation=15)
        plt.ylabel("SRUS")
        plt.title("All Datasets Comparison\nFinal Test SRUS by Adaptation Mode")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "all_datasets_test_srus_by_mode.png"))
    plt.close()
except Exception as e:
    print(f"Error creating cross-dataset SRUS plot: {e}")
    plt.close()

try:
    modes = ["frozen", "always", "gated", "reset"]
    width = 0.2
    x = np.arange(len(dataset_names))
    plt.figure(figsize=(10, 5))
    ok = False
    for i, mode in enumerate(modes):
        vals = []
        for ds_name in dataset_names:
            rows = exp.get(ds_name, {}).get("metrics", {}).get("test", [])
            val = np.nan
            for r in rows:
                if isinstance(r, (list, tuple)) and len(r) >= 2 and r[0] == mode:
                    val = r[1]
            vals.append(val)
        if np.isfinite(np.array(vals, dtype=float)).any():
            plt.bar(x + (i - 1.5) * width, vals, width=width, label=mode)
            ok = True
    if ok:
        plt.xticks(x, dataset_names, rotation=15)
        plt.ylabel("Accuracy")
        plt.title("All Datasets Comparison\nFinal Test Accuracy by Adaptation Mode")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "all_datasets_test_accuracy_by_mode.png"))
    plt.close()
except Exception as e:
    print(f"Error creating cross-dataset accuracy plot: {e}")
    plt.close()

for ds_name in dataset_names:
    test = exp.get(ds_name, {}).get("metrics", {}).get("test", [])
    if test:
        print(ds_name, "test metrics:")
        for row in test:
            print(row)
