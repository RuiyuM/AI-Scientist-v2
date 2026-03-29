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


def load_optional(name):
    p = os.path.join(working_dir, name)
    return np.load(p, allow_pickle=True) if os.path.exists(p) else None


summary = load_optional("summary_metrics.npy")
if summary is not None:
    try:
        summary = summary.item()
    except Exception:
        pass

stream_preds_frozen = load_optional("stream_preds_frozen.npy")
stream_preds_always = load_optional("stream_preds_always.npy")
stream_preds_hybrid = load_optional("stream_preds_hybrid.npy")
stream_gt = load_optional("stream_gt.npy")
stream_conf_frozen = load_optional("stream_conf_frozen.npy")
stream_conf_always = load_optional("stream_conf_always.npy")
stream_conf_hybrid = load_optional("stream_conf_hybrid.npy")
stream_entropy = load_optional("stream_entropy.npy")
stream_margin = load_optional("stream_margin.npy")
stream_triggers = load_optional("stream_triggers.npy")


def extract_xy(seq, prefer_key=None):
    xs, ys = [], []
    for item in seq or []:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            x, y = item[0], item[1]
            if isinstance(y, dict):
                if prefer_key is not None and prefer_key in y:
                    xs.append(x)
                    ys.append(y[prefer_key])
            else:
                try:
                    ys.append(float(y))
                    xs.append(x)
                except Exception:
                    pass
    return xs, ys


datasets = [k for k, v in exp.items() if isinstance(v, dict)]
print("Detected datasets:", datasets)

final_val_acc = {}
final_train_loss = {}
policy_test_acc = {p: {} for p in ["frozen_acc", "always_acc", "hybrid_acc"]}
policy_test_ece = {p: {} for p in ["frozen_ece", "always_ece", "hybrid_ece"]}

for ds_name in datasets:
    ds = exp.get(ds_name, {})
    metrics = ds.get("metrics", {})
    losses = ds.get("losses", {})
    preds = ds.get("predictions", [])
    gts = ds.get("ground_truth", [])
    confs = ds.get("confidences", [])
    trigs = ds.get("triggers", [])

    xv, yv = extract_xy(metrics.get("val", []))
    if yv:
        final_val_acc[ds_name] = yv[-1]
    xtl, ytl = extract_xy(losses.get("train", []))
    if ytl:
        final_train_loss[ds_name] = ytl[-1]

    test_seq = metrics.get("test", [])
    if (
        test_seq
        and isinstance(test_seq[-1], (list, tuple))
        and len(test_seq[-1]) >= 2
        and isinstance(test_seq[-1][1], dict)
    ):
        last = test_seq[-1][1]
        for k in policy_test_acc:
            if k in last:
                policy_test_acc[k][ds_name] = last[k]
        for k in policy_test_ece:
            if k in last:
                policy_test_ece[k][ds_name] = last[k]

    try:
        xt, yt = extract_xy(metrics.get("train", []))
        xv, yv = extract_xy(metrics.get("val", []))
        if xt or xv:
            plt.figure(figsize=(8, 5))
            if xt:
                plt.plot(xt, yt, marker="o", label="Train Metric")
            if xv:
                plt.plot(xv, yv, marker="o", label="Validation Metric")
            plt.xlabel("Epoch")
            plt.ylabel("Metric")
            plt.title(f"{ds_name} Dataset\nTraining and Validation Metric Curves")
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                os.path.join(working_dir, f"{ds_name}_train_val_metric_curves.png")
            )
        plt.close()
    except Exception as e:
        print(f"Error creating train/val metric plot for {ds_name}: {e}")
        plt.close()

    try:
        xt, yt = extract_xy(losses.get("train", []))
        xv, yv = extract_xy(losses.get("val", []))
        if xt or xv:
            plt.figure(figsize=(8, 5))
            if xt:
                plt.plot(xt, yt, marker="o", label="Train Loss")
            if xv:
                plt.plot(xv, yv, marker="o", label="Validation Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.title(f"{ds_name} Dataset\nTraining and Validation Loss Curves")
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                os.path.join(working_dir, f"{ds_name}_train_val_loss_curves.png")
            )
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {ds_name}: {e}")
        plt.close()

    try:
        if preds and gts:
            labels = sorted(list(set(list(map(str, gts)) + list(map(str, preds)))))
            m = {k: i for i, k in enumerate(labels)}
            n = min(len(gts), len(preds))
            x = np.arange(n)
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            plt.plot(x, [m[str(v)] for v in gts[:n]], marker="o", label="Ground Truth")
            plt.yticks(list(m.values()), list(m.keys()))
            plt.xlabel("Sample")
            plt.ylabel("Label")
            plt.title(f"{ds_name} Dataset\nLeft: Ground Truth")
            plt.subplot(1, 2, 2)
            plt.plot(
                x,
                [m[str(v)] for v in preds[:n]],
                marker="x",
                color="orange",
                label="Generated Samples",
            )
            plt.yticks(list(m.values()), list(m.keys()))
            plt.xlabel("Sample")
            plt.ylabel("Label")
            plt.title(f"{ds_name} Dataset\nRight: Generated Samples")
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    working_dir, f"{ds_name}_ground_truth_vs_generated_samples.png"
                )
            )
        plt.close()
    except Exception as e:
        print(f"Error creating prediction comparison plot for {ds_name}: {e}")
        plt.close()

    try:
        if confs:
            plt.figure(figsize=(7, 4))
            plt.hist(
                np.asarray(confs, dtype=float),
                bins=min(10, max(3, len(confs))),
                edgecolor="black",
            )
            plt.xlabel("Confidence")
            plt.ylabel("Count")
            plt.title(f"{ds_name} Dataset\nHybrid Confidence Distribution")
            plt.tight_layout()
            plt.savefig(
                os.path.join(working_dir, f"{ds_name}_hybrid_confidence_histogram.png")
            )
        plt.close()
    except Exception as e:
        print(f"Error creating confidence histogram for {ds_name}: {e}")
        plt.close()

    try:
        if trigs:
            arr = np.asarray(trigs, dtype=float)
            plt.figure(figsize=(8, 4))
            plt.plot(np.arange(len(arr)), arr, marker="o")
            plt.xlabel("Sample")
            plt.ylabel("Trigger")
            plt.title(f"{ds_name} Dataset\nHybrid Adaptation Trigger Trace")
            plt.tight_layout()
            plt.savefig(
                os.path.join(working_dir, f"{ds_name}_hybrid_trigger_trace.png")
            )
        plt.close()
    except Exception as e:
        print(f"Error creating trigger plot for {ds_name}: {e}")
        plt.close()

    try:
        test_seq = metrics.get("test", [])
        if (
            test_seq
            and isinstance(test_seq[-1], (list, tuple))
            and len(test_seq[-1]) >= 2
            and isinstance(test_seq[-1][1], dict)
        ):
            last = test_seq[-1][1]
            names = ["frozen_acc", "always_acc", "hybrid_acc"]
            vals = [last.get(k, np.nan) for k in names]
            plt.figure(figsize=(7, 4))
            plt.bar(np.arange(len(names)), vals)
            plt.xticks(np.arange(len(names)), ["Frozen", "Always", "Hybrid"])
            plt.ylabel("Accuracy")
            plt.title(f"{ds_name} Dataset\nTest Accuracy by Adaptation Policy")
            plt.tight_layout()
            plt.savefig(
                os.path.join(working_dir, f"{ds_name}_test_accuracy_by_policy.png")
            )
        plt.close()
    except Exception as e:
        print(f"Error creating test accuracy bar plot for {ds_name}: {e}")
        plt.close()

    try:
        test_seq = metrics.get("test", [])
        if (
            test_seq
            and isinstance(test_seq[-1], (list, tuple))
            and len(test_seq[-1]) >= 2
            and isinstance(test_seq[-1][1], dict)
        ):
            last = test_seq[-1][1]
            names = ["frozen_ece", "always_ece", "hybrid_ece"]
            vals = [last.get(k, np.nan) for k in names]
            plt.figure(figsize=(7, 4))
            plt.bar(np.arange(len(names)), vals)
            plt.xticks(np.arange(len(names)), ["Frozen", "Always", "Hybrid"])
            plt.ylabel("ECE")
            plt.title(f"{ds_name} Dataset\nTest Calibration Error by Adaptation Policy")
            plt.tight_layout()
            plt.savefig(os.path.join(working_dir, f"{ds_name}_test_ece_by_policy.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating test ECE bar plot for {ds_name}: {e}")
        plt.close()

try:
    if final_val_acc:
        names = list(final_val_acc.keys())
        vals = [final_val_acc[k] for k in names]
        plt.figure(figsize=(9, 4))
        plt.bar(np.arange(len(names)), vals)
        plt.xticks(np.arange(len(names)), names, rotation=20)
        plt.ylabel("Final Validation Metric")
        plt.title("All Datasets\nComparison of Final Validation Metric")
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                working_dir, "all_datasets_final_validation_metric_comparison.png"
            )
        )
    plt.close()
except Exception as e:
    print(f"Error creating cross-dataset validation comparison: {e}")
    plt.close()

try:
    if final_train_loss:
        names = list(final_train_loss.keys())
        vals = [final_train_loss[k] for k in names]
        plt.figure(figsize=(9, 4))
        plt.bar(np.arange(len(names)), vals)
        plt.xticks(np.arange(len(names)), names, rotation=20)
        plt.ylabel("Final Train Loss")
        plt.title("All Datasets\nComparison of Final Training Loss")
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "all_datasets_final_train_loss_comparison.png")
        )
    plt.close()
except Exception as e:
    print(f"Error creating cross-dataset train loss comparison: {e}")
    plt.close()

try:
    if any(policy_test_acc[k] for k in policy_test_acc):
        ds_names = sorted(
            set().union(*[set(v.keys()) for v in policy_test_acc.values()])
        )
        x = np.arange(len(ds_names))
        w = 0.25
        plt.figure(figsize=(10, 5))
        for i, (k, lab) in enumerate(
            zip(
                ["frozen_acc", "always_acc", "hybrid_acc"],
                ["Frozen", "Always", "Hybrid"],
            )
        ):
            vals = [policy_test_acc[k].get(ds, np.nan) for ds in ds_names]
            plt.bar(x + (i - 1) * w, vals, width=w, label=lab)
        plt.xticks(x, ds_names, rotation=20)
        plt.ylabel("Test Accuracy")
        plt.title("All Datasets\nTest Accuracy Comparison Across Adaptation Policies")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                working_dir, "all_datasets_test_accuracy_policy_comparison.png"
            )
        )
    plt.close()
except Exception as e:
    print(f"Error creating cross-dataset test accuracy comparison: {e}")
    plt.close()

try:
    if any(policy_test_ece[k] for k in policy_test_ece):
        ds_names = sorted(
            set().union(*[set(v.keys()) for v in policy_test_ece.values()])
        )
        x = np.arange(len(ds_names))
        w = 0.25
        plt.figure(figsize=(10, 5))
        for i, (k, lab) in enumerate(
            zip(
                ["frozen_ece", "always_ece", "hybrid_ece"],
                ["Frozen", "Always", "Hybrid"],
            )
        ):
            vals = [policy_test_ece[k].get(ds, np.nan) for ds in ds_names]
            plt.bar(x + (i - 1) * w, vals, width=w, label=lab)
        plt.xticks(x, ds_names, rotation=20)
        plt.ylabel("ECE")
        plt.title(
            "All Datasets\nTest Calibration Error Comparison Across Adaptation Policies"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "all_datasets_test_ece_policy_comparison.png")
        )
    plt.close()
except Exception as e:
    print(f"Error creating cross-dataset test ECE comparison: {e}")
    plt.close()

try:
    if stream_gt is not None and stream_preds_hybrid is not None:
        n = min(200, len(stream_gt), len(stream_preds_hybrid))
        labels = sorted(
            list(
                set(
                    list(map(str, stream_gt[:n]))
                    + list(map(str, stream_preds_hybrid[:n]))
                )
            )
        )
        m = {k: i for i, k in enumerate(labels)}
        x = np.arange(n)
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(x, [m[str(v)] for v in stream_gt[:n]], linewidth=1.5)
        plt.yticks(list(m.values()), list(m.keys()))
        plt.xlabel("Stream Step")
        plt.ylabel("Label")
        plt.title("Mixed Stream Dataset\nLeft: Ground Truth")
        plt.subplot(1, 2, 2)
        plt.plot(
            x,
            [m[str(v)] for v in stream_preds_hybrid[:n]],
            linewidth=1.2,
            color="orange",
        )
        plt.yticks(list(m.values()), list(m.keys()))
        plt.xlabel("Stream Step")
        plt.ylabel("Label")
        plt.title("Mixed Stream Dataset\nRight: Generated Samples")
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                working_dir, "mixed_stream_ground_truth_vs_generated_samples_prefix.png"
            )
        )
    plt.close()
except Exception as e:
    print(f"Error creating stream prediction trace plot: {e}")
    plt.close()

try:
    if any(
        v is not None
        for v in [stream_conf_frozen, stream_conf_always, stream_conf_hybrid]
    ):
        plt.figure(figsize=(9, 5))
        if stream_conf_frozen is not None:
            plt.plot(stream_conf_frozen, label="Frozen")
        if stream_conf_always is not None:
            plt.plot(stream_conf_always, label="Always")
        if stream_conf_hybrid is not None:
            plt.plot(stream_conf_hybrid, label="Hybrid")
        plt.xlabel("Stream Step")
        plt.ylabel("Confidence")
        plt.title("Mixed Stream Dataset\nConfidence Trace by Adaptation Policy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "mixed_stream_confidence_trace_by_policy.png")
        )
    plt.close()
except Exception as e:
    print(f"Error creating stream confidence comparison: {e}")
    plt.close()

try:
    if stream_entropy is not None and stream_triggers is not None:
        x = np.arange(len(stream_entropy))
        idx = np.where(np.asarray(stream_triggers) > 0)[0]
        plt.figure(figsize=(10, 4))
        plt.plot(x, stream_entropy, label="Entropy")
        if len(idx) > 0:
            plt.scatter(
                idx, np.asarray(stream_entropy)[idx], s=12, label="Triggered Updates"
            )
        plt.xlabel("Stream Step")
        plt.ylabel("Entropy")
        plt.title("Mixed Stream Dataset\nHybrid Entropy and Trigger Events")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "mixed_stream_hybrid_entropy_and_triggers.png")
        )
    plt.close()
except Exception as e:
    print(f"Error creating entropy/trigger stream plot: {e}")
    plt.close()

try:
    if stream_margin is not None:
        plt.figure(figsize=(9, 4))
        plt.plot(np.arange(len(stream_margin)), stream_margin)
        plt.xlabel("Stream Step")
        plt.ylabel("Margin")
        plt.title("Mixed Stream Dataset\nHybrid Prediction Margin Trace")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "mixed_stream_hybrid_margin_trace.png"))
    plt.close()
except Exception as e:
    print(f"Error creating margin stream plot: {e}")
    plt.close()

try:
    if summary is not None and isinstance(summary, dict):
        modes = [m for m in ["frozen", "always", "hybrid"] if m in summary]
        accs = [summary[m].get("acc", np.nan) for m in modes]
        eces = [summary[m].get("ece", np.nan) for m in modes]
        srus = [summary[m].get("srus", np.nan) for m in modes]
        x = np.arange(len(modes))
        plt.figure(figsize=(11, 4))
        plt.subplot(1, 3, 1)
        plt.bar(x, accs)
        plt.xticks(x, modes)
        plt.ylabel("Accuracy")
        plt.title("Mixed Stream Dataset\nAccuracy by Policy")
        plt.subplot(1, 3, 2)
        plt.bar(x, eces)
        plt.xticks(x, modes)
        plt.ylabel("ECE")
        plt.title("Mixed Stream Dataset\nCalibration Error by Policy")
        plt.subplot(1, 3, 3)
        plt.bar(x, srus)
        plt.xticks(x, modes)
        plt.ylabel("SRUS")
        plt.title("Mixed Stream Dataset\nShift-Robust Utility by Policy")
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "mixed_stream_summary_metrics_by_policy.png")
        )
    plt.close()
except Exception as e:
    print(f"Error creating summary metrics plot: {e}")
    plt.close()

if isinstance(summary, dict):
    print("Summary metrics:")
    for k, v in summary.items():
        print(k, v)
for ds in datasets:
    out = {}
    if ds in final_val_acc:
        out["final_val_metric"] = final_val_acc[ds]
    if ds in final_train_loss:
        out["final_train_loss"] = final_train_loss[ds]
    if any(ds in policy_test_acc[p] for p in policy_test_acc):
        out["test_acc"] = {p: policy_test_acc[p].get(ds, None) for p in policy_test_acc}
    if any(ds in policy_test_ece[p] for p in policy_test_ece):
        out["test_ece"] = {p: policy_test_ece[p].get(ds, None) for p in policy_test_ece}
    print(ds, out)
