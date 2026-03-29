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

dataset_name = "synthetic_vlm_stream"
root = experiment_data.get("batch_size_tuning", {}).get(dataset_name, {})
trials = root.get("trials", {})
summary = root.get("summary", {})
ranking = summary.get("ranking", [])
best_by_val = summary.get("best_batch_size_by_val", None)
best_by_stream = summary.get("best_batch_size_by_stream_consistency", None)


def curve_xy(seq):
    arr = np.array(seq, dtype=object)
    xs, ys = [], []
    for row in arr:
        if isinstance(row, (list, tuple, np.ndarray)) and len(row) >= 2:
            xs.append(float(row[0]))
            ys.append(float(row[1]))
    return np.array(xs), np.array(ys)


def get_test_metric(trial, name, field):
    for m in trial.get("metrics", {}).get("test", []):
        if isinstance(m, dict) and m.get("name") == name and field in m:
            return float(m[field])
    return np.nan


batch_sizes = sorted(
    [int(k.split("_")[-1]) for k in trials.keys() if k.startswith("batch_size_")]
)

print(f"Dataset: {dataset_name}")
print(f"Best batch size by validation loss: {best_by_val}")
print(
    f"Best batch size by stream consistency stability-adjusted accuracy: {best_by_stream}"
)
if ranking:
    for row in ranking:
        if isinstance(row, dict):
            print(
                f"bs={row.get('batch_size')}, best_val_loss={row.get('best_val_loss'):.4f}, "
                f"best_val_acc={row.get('best_val_acc'):.4f}, "
                f"frozen_saa={row.get('frozen_stability_adjusted_acc'):.4f}, "
                f"entropy_saa={row.get('entropy_stability_adjusted_acc'):.4f}, "
                f"consistency_saa={row.get('consistency_stability_adjusted_acc'):.4f}"
            )

try:
    plt.figure(figsize=(9, 4))
    for bs in batch_sizes:
        x, y = curve_xy(trials[f"batch_size_{bs}"].get("losses", {}).get("train", []))
        if len(x):
            plt.plot(x, y, marker="o", label=f"bs={bs}")
    plt.xlabel("Epoch")
    plt.ylabel("Train loss")
    plt.title("Synthetic VLM Stream Dataset\nTraining loss curves by batch size")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, f"{dataset_name}_train_loss_curves.png"), dpi=160
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot1: {e}")
    plt.close()

try:
    plt.figure(figsize=(9, 4))
    for bs in batch_sizes:
        x, y = curve_xy(trials[f"batch_size_{bs}"].get("metrics", {}).get("val", []))
        if len(x):
            plt.plot(x, y, marker="o", label=f"bs={bs}")
    plt.xlabel("Epoch")
    plt.ylabel("Validation accuracy")
    plt.title("Synthetic VLM Stream Dataset\nValidation accuracy curves by batch size")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, f"{dataset_name}_validation_accuracy_curves.png"),
        dpi=160,
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot2: {e}")
    plt.close()

try:
    plt.figure(figsize=(8, 4))
    vals = []
    for bs in batch_sizes:
        trial = trials[f"batch_size_{bs}"]
        vals.append(float(trial.get("best_val_loss", np.nan)))
    plt.bar(
        np.arange(len(batch_sizes)), vals, tick_label=[str(bs) for bs in batch_sizes]
    )
    plt.xlabel("Training batch size")
    plt.ylabel("Best validation loss")
    plt.title(
        "Synthetic VLM Stream Dataset\nBest validation loss summary by batch size"
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, f"{dataset_name}_best_validation_loss_summary.png"),
        dpi=160,
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot3: {e}")
    plt.close()

try:
    plt.figure(figsize=(9, 4))
    xs = np.arange(len(batch_sizes))
    w = 0.25
    frozen = [
        get_test_metric(trials[f"batch_size_{bs}"], "frozen", "stability_adjusted_acc")
        for bs in batch_sizes
    ]
    entropy = [
        get_test_metric(trials[f"batch_size_{bs}"], "entropy", "stability_adjusted_acc")
        for bs in batch_sizes
    ]
    consistency = [
        get_test_metric(
            trials[f"batch_size_{bs}"], "consistency", "stability_adjusted_acc"
        )
        for bs in batch_sizes
    ]
    plt.bar(xs - w, frozen, width=w, label="Frozen")
    plt.bar(xs, entropy, width=w, label="Entropy TTA")
    plt.bar(xs + w, consistency, width=w, label="Consistency TTA")
    plt.xticks(xs, [str(bs) for bs in batch_sizes])
    plt.xlabel("Training batch size")
    plt.ylabel("Stability-adjusted accuracy")
    plt.title(
        "Synthetic VLM Stream Dataset\nStream test performance by batch size and adaptation mode"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            working_dir,
            f"{dataset_name}_stream_stability_adjusted_accuracy_summary.png",
        ),
        dpi=160,
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot4: {e}")
    plt.close()

try:
    if best_by_stream is not None and f"batch_size_{best_by_stream}" in trials:
        trial = trials[f"batch_size_{best_by_stream}"]
        plt.figure(figsize=(10, 5))
        for mode, label in [
            ("frozen", "Frozen"),
            ("entropy", "Entropy TTA"),
            ("consistency", "Consistency TTA"),
        ]:
            y = np.array(
                trial.get("stream", {}).get(f"{mode}_rolling_acc", []), dtype=float
            )
            if y.size:
                plt.plot(np.arange(len(y)), y, label=label)
        plt.xlabel("Stream step")
        plt.ylabel("Rolling accuracy")
        plt.title(
            f"Synthetic VLM Stream Dataset\nRolling stream accuracy for best-stream batch size={best_by_stream}"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                working_dir,
                f"{dataset_name}_best_stream_batchsize_rolling_accuracy.png",
            ),
            dpi=160,
        )
        plt.close()
except Exception as e:
    print(f"Error creating plot5: {e}")
    plt.close()

try:
    if best_by_val is not None and f"batch_size_{best_by_val}" in trials:
        trial = trials[f"batch_size_{best_by_val}"]
        gt = np.array(trial.get("ground_truth", []))
        n = min(80, len(gt))
        plt.figure(figsize=(10, 4))
        if n:
            plt.plot(gt[:n], label="Ground Truth", marker="o")
            for mode, mk, label in [
                ("frozen", "x", "Frozen Pred"),
                ("consistency", "s", "Consistency Pred"),
            ]:
                pred = np.array(trial.get("predictions", {}).get(mode, []))
                if len(pred) >= n:
                    plt.plot(pred[:n], label=label, marker=mk)
        plt.xlabel("Sample index")
        plt.ylabel("Class")
        plt.title(
            f"Synthetic VLM Stream Dataset\nLeft: Ground Truth, Right: Generated Samples not available; stream prefix class predictions for best-val batch size={best_by_val}"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                working_dir, f"{dataset_name}_best_val_batchsize_prediction_prefix.png"
            ),
            dpi=160,
        )
        plt.close()
except Exception as e:
    print(f"Error creating plot6: {e}")
    plt.close()
