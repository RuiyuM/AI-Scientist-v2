import os
import numpy as np

working_dir = os.path.join(os.getcwd(), "working")
file_path = os.path.join(working_dir, "experiment_data.npy")

experiment_data = np.load(file_path, allow_pickle=True).item()

ablation_name = next(iter(experiment_data.keys()))
ablation_data = experiment_data[ablation_name]

mode_names = ["frozen", "always", "gated", "reset"]


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return x


def summarize_training_metrics(ds_entry):
    train_metrics_all = ds_entry["metrics"]["train"]
    val_metrics_all = ds_entry["metrics"]["val"]
    train_losses_all = ds_entry["losses"]["train"]
    val_losses_all = ds_entry["losses"]["val"]

    final_train_accuracies = []
    final_validation_accuracies = []
    best_validation_accuracies = []
    final_train_losses = []
    final_validation_losses = []
    best_validation_losses = []

    for arr in train_metrics_all:
        arr = np.array(arr, dtype=np.float32)
        if arr.size > 0:
            final_train_accuracies.append(float(arr[-1, 1]))

    for arr in val_metrics_all:
        arr = np.array(arr, dtype=np.float32)
        if arr.size > 0:
            final_validation_accuracies.append(float(arr[-1, 1]))
            best_validation_accuracies.append(float(np.max(arr[:, 1])))

    for arr in train_losses_all:
        arr = np.array(arr, dtype=np.float32)
        if arr.size > 0:
            final_train_losses.append(float(arr[-1, 1]))

    for arr in val_losses_all:
        arr = np.array(arr, dtype=np.float32)
        if arr.size > 0:
            final_validation_losses.append(float(arr[-1, 1]))
            best_validation_losses.append(float(np.min(arr[:, 1])))

    summary = {
        "final train accuracy": (
            max(final_train_accuracies) if final_train_accuracies else None
        ),
        "final validation accuracy": (
            max(final_validation_accuracies) if final_validation_accuracies else None
        ),
        "best validation accuracy": (
            max(best_validation_accuracies) if best_validation_accuracies else None
        ),
        "final train loss": min(final_train_losses) if final_train_losses else None,
        "final validation loss": (
            min(final_validation_losses) if final_validation_losses else None
        ),
        "best validation loss": (
            min(best_validation_losses) if best_validation_losses else None
        ),
    }
    return summary


def summarize_test_metrics(ds_entry):
    rank_summaries = ds_entry.get("rank_summaries", {})

    best_per_mode = {}
    overall_best_srus = None
    overall_best_accuracy = None

    for mode in mode_names:
        candidates = []
        for rank_key, mode_dict in rank_summaries.items():
            if mode in mode_dict:
                metrics = mode_dict[mode]
                candidates.append((rank_key, metrics))

                srus_val = float(metrics["srus"])
                acc_val = float(metrics["acc"])

                if (
                    overall_best_srus is None
                    or srus_val > overall_best_srus["test SRUS"]
                ):
                    overall_best_srus = {
                        "rank": rank_key,
                        "mode": mode,
                        "test accuracy": acc_val,
                        "test expected calibration error": float(metrics["ece"]),
                        "test trigger rate": float(metrics["trigger_rate"]),
                        "test update rate": float(metrics["update_rate"]),
                        "test reset rate": float(metrics["reset_rate"]),
                        "test overhead": float(metrics["overhead"]),
                        "test SRUS": srus_val,
                    }

                if (
                    overall_best_accuracy is None
                    or acc_val > overall_best_accuracy["test accuracy"]
                ):
                    overall_best_accuracy = {
                        "rank": rank_key,
                        "mode": mode,
                        "test accuracy": acc_val,
                        "test expected calibration error": float(metrics["ece"]),
                        "test trigger rate": float(metrics["trigger_rate"]),
                        "test update rate": float(metrics["update_rate"]),
                        "test reset rate": float(metrics["reset_rate"]),
                        "test overhead": float(metrics["overhead"]),
                        "test SRUS": srus_val,
                    }

        if candidates:
            best_rank_key, best_metrics = max(
                candidates, key=lambda x: float(x[1]["srus"])
            )
            best_per_mode[mode] = {
                "rank": best_rank_key,
                "test accuracy": float(best_metrics["acc"]),
                "test expected calibration error": float(best_metrics["ece"]),
                "test trigger rate": float(best_metrics["trigger_rate"]),
                "test update rate": float(best_metrics["update_rate"]),
                "test reset rate": float(best_metrics["reset_rate"]),
                "test overhead": float(best_metrics["overhead"]),
                "test SRUS": float(best_metrics["srus"]),
            }

    return best_per_mode, overall_best_srus, overall_best_accuracy


for dataset_name, ds_entry in ablation_data.items():
    print(f"Dataset: {dataset_name}")

    training_summary = summarize_training_metrics(ds_entry)
    print(
        f"Metric: final train accuracy = {safe_float(training_summary['final train accuracy']):.6f}"
    )
    print(
        f"Metric: final validation accuracy = {safe_float(training_summary['final validation accuracy']):.6f}"
    )
    print(
        f"Metric: best validation accuracy = {safe_float(training_summary['best validation accuracy']):.6f}"
    )
    print(
        f"Metric: final train loss = {safe_float(training_summary['final train loss']):.6f}"
    )
    print(
        f"Metric: final validation loss = {safe_float(training_summary['final validation loss']):.6f}"
    )
    print(
        f"Metric: best validation loss = {safe_float(training_summary['best validation loss']):.6f}"
    )

    best_per_mode, overall_best_srus, overall_best_accuracy = summarize_test_metrics(
        ds_entry
    )

    for mode in mode_names:
        if mode in best_per_mode:
            mode_summary = best_per_mode[mode]
            print(f"Metric: best {mode} test rank = {mode_summary['rank']}")
            print(
                f"Metric: best {mode} test accuracy = {mode_summary['test accuracy']:.6f}"
            )
            print(
                f"Metric: best {mode} test expected calibration error = {mode_summary['test expected calibration error']:.6f}"
            )
            print(
                f"Metric: best {mode} test trigger rate = {mode_summary['test trigger rate']:.6f}"
            )
            print(
                f"Metric: best {mode} test update rate = {mode_summary['test update rate']:.6f}"
            )
            print(
                f"Metric: best {mode} test reset rate = {mode_summary['test reset rate']:.6f}"
            )
            print(
                f"Metric: best {mode} test overhead = {mode_summary['test overhead']:.6f}"
            )
            print(f"Metric: best {mode} test SRUS = {mode_summary['test SRUS']:.6f}")

    if overall_best_srus is not None:
        print(f"Metric: overall best test SRUS rank = {overall_best_srus['rank']}")
        print(f"Metric: overall best test SRUS mode = {overall_best_srus['mode']}")
        print(f"Metric: overall best test SRUS = {overall_best_srus['test SRUS']:.6f}")
        print(
            f"Metric: overall best test SRUS accuracy = {overall_best_srus['test accuracy']:.6f}"
        )
        print(
            f"Metric: overall best test SRUS expected calibration error = {overall_best_srus['test expected calibration error']:.6f}"
        )

    if overall_best_accuracy is not None:
        print(
            f"Metric: overall best test accuracy rank = {overall_best_accuracy['rank']}"
        )
        print(
            f"Metric: overall best test accuracy mode = {overall_best_accuracy['mode']}"
        )
        print(
            f"Metric: overall best test accuracy = {overall_best_accuracy['test accuracy']:.6f}"
        )
        print(
            f"Metric: overall best test accuracy expected calibration error = {overall_best_accuracy['test expected calibration error']:.6f}"
        )
        print(
            f"Metric: overall best test accuracy SRUS = {overall_best_accuracy['test SRUS']:.6f}"
        )

    print("-" * 60)
