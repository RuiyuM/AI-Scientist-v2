import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

exp = {}
try:
    p1 = os.path.join(working_dir, "experiment_data.npy")
    p2 = os.path.join(os.getcwd(), "experiment_data.npy")
    path = p1 if os.path.exists(p1) else p2
    exp = np.load(path, allow_pickle=True).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

dataset_names = list(exp.keys())
modes = ["frozen", "always", "gated", "gated_reset"]


def mean_by_epoch(rows, val_idx):
    out = {}
    for r in rows:
        if len(r) > val_idx:
            epoch = int(r[1])
            val = float(r[val_idx])
            out.setdefault(epoch, []).append(val)
    xs = sorted(out)
    ys = [np.mean(out[x]) for x in xs]
    return xs, ys


def aggregate_test(rows):
    res = {
        m: {
            "acc": [],
            "ece": [],
            "trigger": [],
            "update": [],
            "overhead": [],
            "srus": [],
        }
        for m in modes
    }
    for r in rows:
        if len(r) >= 8:
            _, mode, acc, ece, trig, upd, ov, srus = r
            if mode in res:
                res[mode]["acc"].append(acc)
                res[mode]["ece"].append(ece)
                res[mode]["trigger"].append(trig)
                res[mode]["update"].append(upd)
                res[mode]["overhead"].append(ov)
                res[mode]["srus"].append(srus)
    return {
        m: {k: (float(np.mean(v)) if len(v) else np.nan) for k, v in d.items()}
        for m, d in res.items()
    }


summary_print = {}

for ds_name in dataset_names:
    ds = exp.get(ds_name, {})
    metrics = ds.get("metrics", {})
    losses = ds.get("losses", {})
    other = ds.get("other", {})
    preds_store = ds.get("predictions", [])
    gt_store = ds.get("ground_truth", [])
    summary_print[ds_name] = aggregate_test(metrics.get("test", []))

    try:
        tr = metrics.get("train", [])
        va = metrics.get("val", [])
        x1, y1 = mean_by_epoch(tr, 2)
        x2, y2 = mean_by_epoch(va, 2)
        if x1 or x2:
            plt.figure(figsize=(8, 5))
            if x1:
                plt.plot(x1, y1, marker="o", label="Train Accuracy")
            if x2:
                plt.plot(x2, y2, marker="o", label="Validation Accuracy")
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
        x1, y1 = mean_by_epoch(tr, 2)
        x2, y2 = mean_by_epoch(va, 2)
        if x1 or x2:
            plt.figure(figsize=(8, 5))
            if x1:
                plt.plot(x1, y1, marker="o", label="Train Loss")
            if x2:
                plt.plot(x2, y2, marker="o", label="Validation Loss")
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
        agg = aggregate_test(metrics.get("test", []))
        xs = np.arange(len(modes))
        w = 0.38
        accs = [agg[m]["acc"] for m in modes]
        srus = [agg[m]["srus"] for m in modes]
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.bar(xs, accs)
        plt.xticks(xs, modes, rotation=20)
        plt.ylabel("Accuracy")
        plt.title(f"{ds_name} Dataset\nMean Test Accuracy by Adaptation Mode")
        plt.subplot(1, 2, 2)
        plt.bar(xs, srus)
        plt.xticks(xs, modes, rotation=20)
        plt.ylabel("SRUS")
        plt.title(f"{ds_name} Dataset\nMean Test SRUS by Adaptation Mode")
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, f"{ds_name}_test_accuracy_srus_by_mode.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating {ds_name} mode bar plot: {e}")
        plt.close()

    try:
        agg = aggregate_test(metrics.get("test", []))
        xs = np.arange(len(modes))
        eces = [agg[m]["ece"] for m in modes]
        trigs = [agg[m]["trigger"] for m in modes]
        ovs = [agg[m]["overhead"] for m in modes]
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.bar(xs, eces)
        plt.xticks(xs, modes, rotation=20)
        plt.ylabel("ECE")
        plt.title(f"{ds_name} Dataset\nMean Test ECE by Adaptation Mode")
        plt.subplot(1, 3, 2)
        plt.bar(xs, trigs)
        plt.xticks(xs, modes, rotation=20)
        plt.ylabel("Trigger Rate")
        plt.title(f"{ds_name} Dataset\nMean Trigger Rate by Adaptation Mode")
        plt.subplot(1, 3, 3)
        plt.bar(xs, ovs)
        plt.xticks(xs, modes, rotation=20)
        plt.ylabel("Overhead")
        plt.title(f"{ds_name} Dataset\nMean Overhead by Adaptation Mode")
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                working_dir, f"{ds_name}_test_ece_trigger_overhead_by_mode.png"
            )
        )
        plt.close()
    except Exception as e:
        print(f"Error creating {ds_name} auxiliary mode plot: {e}")
        plt.close()

    try:
        seed_keys = sorted(other.keys())
        if seed_keys:
            rep = other[seed_keys[0]]
            plt.figure(figsize=(9, 4))
            for mode, c in [
                ("frozen", "black"),
                ("always", "tab:red"),
                ("gated", "tab:blue"),
                ("gated_reset", "tab:green"),
            ]:
                if mode in rep and "cumacc" in rep[mode]:
                    arr = np.array(rep[mode]["cumacc"])
                    if arr.size:
                        plt.plot(arr, label=mode, linewidth=1.5, color=c)
            plt.xlabel("Stream Step")
            plt.ylabel("Cumulative Accuracy")
            plt.title(
                f"{ds_name} Dataset\nRepresentative Seed Stream Cumulative Accuracy by Mode"
            )
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    working_dir,
                    f"{ds_name}_representative_seed_cumulative_accuracy.png",
                )
            )
        plt.close()
    except Exception as e:
        print(f"Error creating {ds_name} cumulative accuracy plot: {e}")
        plt.close()

    try:
        seed_keys = sorted(other.keys())
        if seed_keys:
            rep = other[seed_keys[0]]
            g = rep.get("gated", {})
            ent = np.array(g.get("entropy", []))
            gates = np.array(g.get("gate_flags", []))
            updates = np.array(g.get("update_flags", []))
            if ent.size:
                plt.figure(figsize=(10, 4))
                x = np.arange(len(ent))
                plt.plot(x, ent, label="Entropy")
                idx1 = np.where(gates > 0)[0]
                idx2 = np.where(updates > 0)[0]
                if len(idx1):
                    plt.scatter(idx1, ent[idx1], s=10, label="Gate Flags")
                if len(idx2):
                    plt.scatter(idx2, ent[idx2], s=10, label="Update Flags")
                plt.xlabel("Stream Step")
                plt.ylabel("Entropy")
                plt.title(
                    f"{ds_name} Dataset\nGated Entropy with Gate and Update Events"
                )
                plt.legend()
                plt.tight_layout()
                plt.savefig(
                    os.path.join(
                        working_dir, f"{ds_name}_gated_entropy_gate_update_events.png"
                    )
                )
            plt.close()
    except Exception as e:
        print(f"Error creating {ds_name} entropy/events plot: {e}")
        plt.close()

    try:
        seed_keys = sorted(other.keys())
        if seed_keys:
            rep = other[seed_keys[0]]
            plt.figure(figsize=(9, 4))
            for mode in modes:
                if mode in rep and "drift" in rep[mode]:
                    arr = np.array(rep[mode]["drift"])
                    if arr.size:
                        plt.plot(arr, label=mode)
            plt.xlabel("Stream Step")
            plt.ylabel("Parameter Drift")
            plt.title(
                f"{ds_name} Dataset\nRepresentative Seed Parameter Drift by Adaptation Mode"
            )
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    working_dir, f"{ds_name}_representative_seed_parameter_drift.png"
                )
            )
        plt.close()
    except Exception as e:
        print(f"Error creating {ds_name} drift plot: {e}")
        plt.close()

    try:
        if preds_store and gt_store:
            pred_map = {
                int(s): np.array(p) for s, mode, p in preds_store if mode == "gated"
            }
            gt_map = {int(s): np.array(g) for s, g in gt_store}
            common = sorted(set(pred_map) & set(gt_map))
            if common:
                seed = common[0]
                p = pred_map[seed]
                g = gt_map[seed]
                n = min(200, len(g), len(p))
                plt.figure(figsize=(10, 4))
                plt.plot(np.arange(n), g[:n], label="Ground Truth", linewidth=1.5)
                plt.plot(
                    np.arange(n),
                    p[:n],
                    label="Generated Samples / Gated Predictions",
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
        if preds_store and gt_store:
            pred_map = {
                int(s): np.array(p) for s, mode, p in preds_store if mode == "gated"
            }
            gt_map = {int(s): np.array(g) for s, g in gt_store}
            common = sorted(set(pred_map) & set(gt_map))
            if common:
                seed = common[0]
                p = pred_map[seed].astype(int)
                g = gt_map[seed].astype(int)
                cm = np.zeros((2, 2), dtype=int)
                for yy, pp in zip(g, p):
                    if yy in [0, 1] and pp in [0, 1]:
                        cm[yy, pp] += 1
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
    if dataset_names:
        for metric, ylabel, fname in [
            ("acc", "Accuracy", "cross_dataset_accuracy_by_mode.png"),
            ("ece", "ECE", "cross_dataset_ece_by_mode.png"),
            ("srus", "SRUS", "cross_dataset_srus_by_mode.png"),
            ("trigger", "Trigger Rate", "cross_dataset_trigger_rate_by_mode.png"),
            ("overhead", "Overhead", "cross_dataset_overhead_by_mode.png"),
        ]:
            plt.figure(figsize=(10, 5))
            x = np.arange(len(dataset_names))
            width = 0.18
            for i, mode in enumerate(modes):
                vals = [summary_print[ds][mode][metric] for ds in dataset_names]
                plt.bar(x + (i - 1.5) * width, vals, width=width, label=mode)
            plt.xticks(x, dataset_names, rotation=20)
            plt.ylabel(ylabel)
            plt.title(f"All Datasets\nCross-Dataset Mean {ylabel} by Adaptation Mode")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
except Exception as e:
    print(f"Error creating cross-dataset comparison plots: {e}")
    plt.close()

for ds_name, agg in summary_print.items():
    print(ds_name)
    for mode in modes:
        d = agg[mode]
        print(
            f"  {mode}: acc={d['acc']:.4f}, ece={d['ece']:.4f}, srus={d['srus']:.4f}, trigger={d['trigger']:.4f}, overhead={d['overhead']:.6f}"
        )
