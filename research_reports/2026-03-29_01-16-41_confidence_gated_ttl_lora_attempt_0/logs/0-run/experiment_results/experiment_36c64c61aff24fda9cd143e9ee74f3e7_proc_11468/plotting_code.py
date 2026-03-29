import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

exp_path = os.path.join(working_dir, "experiment_data.npy")
if not os.path.exists(exp_path):
    alt = os.path.join(os.getcwd(), "experiment_data.npy")
    exp_path = alt if os.path.exists(alt) else exp_path

try:
    experiment_data = np.load(exp_path, allow_pickle=True).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

plot_count = 0


def confusion_matrix(y_true, y_pred, n_classes=None):
    y_true, y_pred = np.asarray(y_true, int), np.asarray(y_pred, int)
    if y_true.size == 0 or y_pred.size == 0:
        return np.zeros((0, 0), dtype=int)
    if n_classes is None:
        n_classes = int(max(y_true.max(), y_pred.max()) + 1)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


datasets = list(experiment_data.keys())
final_val_accs, stream_none, stream_always, stream_gated, trigger_rates = (
    {},
    {},
    {},
    {},
    {},
)

for ds_name in datasets:
    ds = experiment_data.get(ds_name, {})
    metrics = ds.get("metrics", {})
    losses = ds.get("losses", {})
    preds = ds.get("predictions", {})
    gt = np.array(ds.get("ground_truth", []))
    extra = ds.get("extra", {})
    stream_metrics = metrics.get("stream", [])
    overall = (
        stream_metrics[0]
        if len(stream_metrics) and isinstance(stream_metrics[0], dict)
        else {}
    )

    tr_m = (
        np.array(metrics.get("train", []), dtype=object)
        if len(metrics.get("train", []))
        else np.array([])
    )
    va_m = (
        np.array(metrics.get("val", []), dtype=object)
        if len(metrics.get("val", []))
        else np.array([])
    )
    tr_l = (
        np.array(losses.get("train", []), dtype=object)
        if len(losses.get("train", []))
        else np.array([])
    )
    va_l = (
        np.array(losses.get("val", []), dtype=object)
        if len(losses.get("val", []))
        else np.array([])
    )

    if va_m.size:
        final_val_accs[ds_name] = float(va_m[-1, 1])
    if "none_acc" in overall:
        stream_none[ds_name] = float(overall["none_acc"])
    if "always_acc" in overall:
        stream_always[ds_name] = float(overall["always_acc"])
    if "gated_acc" in overall:
        stream_gated[ds_name] = float(overall["gated_acc"])
    if "gated_trigger_rate" in overall:
        trigger_rates[ds_name] = float(overall["gated_trigger_rate"])

    print(f"\nDataset: {ds_name}")
    if va_m.size:
        print(f"final_val_accuracy: {float(va_m[-1,1]):.4f}")
    if va_l.size:
        print(f"final_val_loss: {float(va_l[-1,1]):.4f}")
    for k in [
        "none_acc",
        "always_acc",
        "gated_acc",
        "gated_trigger_rate",
        "gated_overhead_sec",
        "gated_ent_err_corr",
    ]:
        if k in overall:
            print(f"{k}: {float(overall[k]):.4f}")

    try:
        if tr_m.size and va_m.size:
            plt.figure(figsize=(6, 4))
            plt.plot(
                tr_m[:, 0].astype(float),
                tr_m[:, 1].astype(float),
                marker="o",
                label="Train Accuracy",
            )
            plt.plot(
                va_m[:, 0].astype(float),
                va_m[:, 1].astype(float),
                marker="o",
                label="Validation Accuracy",
            )
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title(f"{ds_name} Dataset\nTraining and Validation Accuracy")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(working_dir, f"{ds_name}_train_val_accuracy.png"))
            plt.close()
            plot_count += 1
    except Exception as e:
        print(f"Error creating train/val accuracy plot for {ds_name}: {e}")
        plt.close()

    try:
        if tr_l.size and va_l.size:
            plt.figure(figsize=(6, 4))
            plt.plot(
                tr_l[:, 0].astype(float),
                tr_l[:, 1].astype(float),
                marker="o",
                label="Train Loss",
            )
            plt.plot(
                va_l[:, 0].astype(float),
                va_l[:, 1].astype(float),
                marker="o",
                label="Validation Loss",
            )
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"{ds_name} Dataset\nTraining and Validation Loss")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(working_dir, f"{ds_name}_train_val_loss.png"))
            plt.close()
            plot_count += 1
    except Exception as e:
        print(f"Error creating train/val loss plot for {ds_name}: {e}")
        plt.close()

    try:
        keys = [
            k
            for k in [
                "none_acc",
                "always_acc",
                "gated_acc",
                "gated_trigger_rate",
                "gated_ent_err_corr",
            ]
            if k in overall
        ]
        if keys:
            vals = [float(overall[k]) for k in keys]
            plt.figure(figsize=(7, 4))
            plt.bar(keys, vals)
            plt.ylabel("Value")
            plt.title(
                f"{ds_name} Dataset\nOverall Stream Metrics by Adaptation Setting"
            )
            plt.xticks(rotation=20, ha="right")
            plt.tight_layout()
            plt.savefig(
                os.path.join(working_dir, f"{ds_name}_overall_stream_metrics_bar.png")
            )
            plt.close()
            plot_count += 1
    except Exception as e:
        print(f"Error creating stream metrics bar plot for {ds_name}: {e}")
        plt.close()

    try:
        segs = {
            k.replace("segment_accs_", ""): np.array(v, dtype=float)
            for k, v in extra.items()
            if k.startswith("segment_accs_")
        }
        if len(segs):
            plt.figure(figsize=(7, 4))
            for k, arr in segs.items():
                if arr.size:
                    plt.plot(np.arange(arr.size), arr, marker="o", label=k)
            plt.xlabel("Stream Segment")
            plt.ylabel("Accuracy")
            plt.title(
                f"{ds_name} Dataset\nSegment-wise Accuracy Across Adaptation Modes"
            )
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                os.path.join(working_dir, f"{ds_name}_segment_accuracy_comparison.png")
            )
            plt.close()
            plot_count += 1
    except Exception as e:
        print(f"Error creating segment accuracy plot for {ds_name}: {e}")
        plt.close()

    try:
        ent_none = np.array(extra.get("entropies_none", []), dtype=float)
        ent_gated = np.array(extra.get("entropies_gated", []), dtype=float)
        if ent_none.size or ent_gated.size:
            plt.figure(figsize=(8, 4))
            if ent_none.size:
                plt.plot(ent_none, label="none entropy", alpha=0.8)
            if ent_gated.size:
                plt.plot(ent_gated, label="gated entropy", alpha=0.8)
            plt.xlabel("Stream Sample Index")
            plt.ylabel("Predictive Entropy")
            plt.title(
                f"{ds_name} Dataset\nEntropy over Evaluation Stream by Adaptation Mode"
            )
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(working_dir, f"{ds_name}_entropy_over_stream.png"))
            plt.close()
            plot_count += 1
    except Exception as e:
        print(f"Error creating entropy plot for {ds_name}: {e}")
        plt.close()

    try:
        margins = np.array(extra.get("margins_gated", []), dtype=float)
        if margins.size:
            plt.figure(figsize=(8, 4))
            plt.plot(margins, label="gated margin")
            plt.xlabel("Stream Sample Index")
            plt.ylabel("Prediction Margin")
            plt.title(
                f"{ds_name} Dataset\nGated Adaptation Prediction Margins over Stream"
            )
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                os.path.join(working_dir, f"{ds_name}_gated_margin_over_stream.png")
            )
            plt.close()
            plot_count += 1
    except Exception as e:
        print(f"Error creating margin plot for {ds_name}: {e}")
        plt.close()

    for mode in ["none", "always", "gated"]:
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
                        f"{ds_name} Dataset\nConfusion Matrix for {mode} Adaptation"
                    )
                    plt.tight_layout()
                    plt.savefig(
                        os.path.join(
                            working_dir, f"{ds_name}_{mode}_confusion_matrix.png"
                        )
                    )
                    plt.close()
                    plot_count += 1
        except Exception as e:
            print(f"Error creating confusion matrix for {ds_name} {mode}: {e}")
            plt.close()

try:
    names = [d for d in datasets if d in final_val_accs]
    if names:
        plt.figure(figsize=(7, 4))
        plt.bar(names, [final_val_accs[d] for d in names])
        plt.ylabel("Final Validation Accuracy")
        plt.title("All Datasets Comparison\nFinal Validation Accuracy by Dataset")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "all_datasets_final_val_accuracy_comparison.png")
        )
        plt.close()
        plot_count += 1
except Exception as e:
    print(f"Error creating cross-dataset final validation accuracy plot: {e}")
    plt.close()

try:
    names = [
        d
        for d in datasets
        if d in stream_none or d in stream_always or d in stream_gated
    ]
    if names:
        x = np.arange(len(names))
        w = 0.25
        plt.figure(figsize=(8, 4))
        plt.bar(
            x - w, [stream_none.get(d, np.nan) for d in names], width=w, label="none"
        )
        plt.bar(
            x, [stream_always.get(d, np.nan) for d in names], width=w, label="always"
        )
        plt.bar(
            x + w, [stream_gated.get(d, np.nan) for d in names], width=w, label="gated"
        )
        plt.ylabel("Shift-Robust Accuracy")
        plt.title("All Datasets Comparison\nStream Accuracy by Adaptation Mode")
        plt.xticks(x, names, rotation=20, ha="right")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "all_datasets_stream_accuracy_comparison.png")
        )
        plt.close()
        plot_count += 1
except Exception as e:
    print(f"Error creating cross-dataset stream accuracy plot: {e}")
    plt.close()

try:
    names = [d for d in datasets if d in trigger_rates]
    if names:
        plt.figure(figsize=(7, 4))
        plt.bar(names, [trigger_rates[d] for d in names])
        plt.ylabel("Trigger Rate")
        plt.title("All Datasets Comparison\nGated Adaptation Trigger Rate by Dataset")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "all_datasets_gated_trigger_rate_comparison.png")
        )
        plt.close()
        plot_count += 1
except Exception as e:
    print(f"Error creating cross-dataset trigger rate plot: {e}")
    plt.close()

try:
    sm_path = os.path.join(working_dir, "summary_metrics.npz")
    if os.path.exists(sm_path):
        sm = np.load(sm_path)
        if "snus" in sm:
            print(f"\nSNUS: {float(np.array(sm['snus']).ravel()[0]):.4f}")
except Exception as e:
    print(f"Error loading summary metrics: {e}")

print(f"\nplots_generated: {plot_count}")
