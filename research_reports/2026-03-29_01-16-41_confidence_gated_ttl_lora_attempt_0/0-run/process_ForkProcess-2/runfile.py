import os
import numpy as np

working_dir = os.path.join(os.getcwd(), "working")
experiment_data_path = os.path.join(working_dir, "experiment_data.npy")

experiment_data = np.load(experiment_data_path, allow_pickle=True).item()


def to_python_scalar(x):
    if isinstance(x, np.generic):
        return x.item()
    return x


def format_value(x):
    x = to_python_scalar(x)
    if isinstance(x, float):
        return f"{x:.6f}"
    if isinstance(x, np.ndarray):
        return np.array2string(x, precision=6, separator=", ")
    return str(x)


def get_final_epoch_metric(metric_list):
    if metric_list is None or len(metric_list) == 0:
        return None, None
    last_item = metric_list[-1]
    if isinstance(last_item, (list, tuple)) and len(last_item) >= 2:
        epoch = to_python_scalar(last_item[0])
        value = to_python_scalar(last_item[1])
        return epoch, value
    return None, to_python_scalar(last_item)


def get_stream_best_gated(gated_sweep):
    if gated_sweep is None or len(gated_sweep) == 0:
        return None
    best_item = max(gated_sweep, key=lambda d: d.get("shift_robust_acc", float("-inf")))
    return best_item


def print_metric(metric_name, value):
    print(f"{metric_name}: {format_value(value)}")


for experiment_name, datasets in experiment_data.items():
    for dataset_name, dataset_store in datasets.items():
        print(f"Dataset: {dataset_name}")

        train_epoch, final_train_accuracy = get_final_epoch_metric(
            dataset_store.get("metrics", {}).get("train", [])
        )
        val_epoch, final_validation_accuracy = get_final_epoch_metric(
            dataset_store.get("metrics", {}).get("val", [])
        )
        _, final_train_loss = get_final_epoch_metric(
            dataset_store.get("losses", {}).get("train", [])
        )
        _, final_validation_loss = get_final_epoch_metric(
            dataset_store.get("losses", {}).get("val", [])
        )

        stream_metrics = dataset_store.get("metrics", {}).get("stream", {})
        no_adapt_shift_robust_accuracy = stream_metrics.get("none", None)
        always_on_shift_robust_accuracy = stream_metrics.get("always", None)

        tuning_summary = dataset_store.get("tuning_summary", {})
        best_threshold = tuning_summary.get("best_threshold", None)
        best_gated_shift_robust_accuracy = tuning_summary.get(
            "best_shift_robust_acc", None
        )
        best_vs_always_delta = tuning_summary.get("best_vs_always_delta", None)
        best_vs_none_delta = tuning_summary.get("best_vs_none_delta", None)

        gated_sweep = stream_metrics.get("gated_sweep", [])
        best_gated_item = get_stream_best_gated(gated_sweep)
        best_gated_trigger_rate = (
            best_gated_item.get("trigger_rate", None)
            if best_gated_item is not None
            else None
        )

        print_metric("experiment name", experiment_name)
        if train_epoch is not None:
            print_metric("final train epoch", train_epoch)
        print_metric("final train accuracy", final_train_accuracy)
        print_metric("final train loss", final_train_loss)

        if val_epoch is not None:
            print_metric("final validation epoch", val_epoch)
        print_metric("final validation accuracy", final_validation_accuracy)
        print_metric("final validation loss", final_validation_loss)

        print_metric(
            "no-adaptation shift-robust accuracy", no_adapt_shift_robust_accuracy
        )
        print_metric(
            "always-on adaptation shift-robust accuracy",
            always_on_shift_robust_accuracy,
        )
        print_metric("best gated entropy threshold", best_threshold)
        print_metric(
            "best gated shift-robust accuracy", best_gated_shift_robust_accuracy
        )
        print_metric("best gated trigger rate", best_gated_trigger_rate)
        print_metric(
            "best gated shift-robust accuracy delta versus always-on adaptation",
            best_vs_always_delta,
        )
        print_metric(
            "best gated shift-robust accuracy delta versus no adaptation",
            best_vs_none_delta,
        )

        segment_accs_none = dataset_store.get("segment_accs", {}).get("none", None)
        segment_accs_always = dataset_store.get("segment_accs", {}).get("always", None)
        if segment_accs_none is not None and len(segment_accs_none) > 0:
            print_metric(
                "final stream segment accuracy with no adaptation",
                segment_accs_none[-1],
            )
        if segment_accs_always is not None and len(segment_accs_always) > 0:
            print_metric(
                "final stream segment accuracy with always-on adaptation",
                segment_accs_always[-1],
            )

        if best_threshold is not None:
            best_threshold_key = str(best_threshold)
            gated_segment_accs = (
                dataset_store.get("segment_accs", {})
                .get("gated", {})
                .get(best_threshold_key, None)
            )
            if gated_segment_accs is not None and len(gated_segment_accs) > 0:
                print_metric(
                    "final stream segment accuracy with best gated adaptation",
                    gated_segment_accs[-1],
                )

        print()
