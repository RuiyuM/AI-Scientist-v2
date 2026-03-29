import os
import numpy as np

working_dir = os.path.join(os.getcwd(), "working")
experiment_path = os.path.join(working_dir, "experiment_data.npy")

experiment_data = np.load(experiment_path, allow_pickle=True).item()


def to_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, np.ndarray):
        return x.tolist()
    return list(x)


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return x


def format_value(x):
    if x is None:
        return "None"
    if isinstance(x, (float, np.floating)):
        return f"{float(x):.6f}"
    return str(x)


def extract_final_metric(history):
    history = to_list(history)
    if not history:
        return None
    last = history[-1]
    if isinstance(last, (list, tuple)) and len(last) >= 2:
        return safe_float(last[1])
    return None


def extract_best_metric(history, mode="max"):
    history = to_list(history)
    values = []
    for item in history:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            values.append(safe_float(item[1]))
    values = [v for v in values if isinstance(v, (int, float, np.floating))]
    if not values:
        return None
    return max(values) if mode == "max" else min(values)


def extract_best_lr_sweep_entry(lr_sweep):
    lr_sweep = to_list(lr_sweep)
    if not lr_sweep:
        return None
    valid = [x for x in lr_sweep if isinstance(x, dict) and "best_val_loss" in x]
    if not valid:
        return None
    return min(valid, key=lambda x: x["best_val_loss"])


def extract_test_metric_entry(test_metrics, name):
    test_metrics = to_list(test_metrics)
    for item in test_metrics:
        if isinstance(item, dict) and item.get("name") == name:
            return item
    return None


def print_metric(label, value):
    print(f"{label}: {format_value(value)}")


for dataset_name, dataset_store in experiment_data.items():
    print(f"Dataset name: {dataset_name}")

    metrics = dataset_store.get("metrics", {})
    losses = dataset_store.get("losses", {})
    meta = dataset_store.get("meta", {})

    final_train_loss = extract_final_metric(losses.get("train", []))
    final_validation_loss = extract_final_metric(losses.get("val", []))
    best_validation_loss_from_curve = extract_best_metric(
        losses.get("val", []), mode="min"
    )

    final_train_accuracy = extract_final_metric(metrics.get("train", []))
    final_validation_accuracy = extract_final_metric(metrics.get("val", []))
    best_validation_accuracy = extract_best_metric(metrics.get("val", []), mode="max")

    best_lr_entry = extract_best_lr_sweep_entry(metrics.get("lr_sweep", []))

    print_metric("final train loss", final_train_loss)
    print_metric("final validation loss", final_validation_loss)
    print_metric("best validation loss", best_validation_loss_from_curve)
    print_metric("final train accuracy", final_train_accuracy)
    print_metric("final validation accuracy", final_validation_accuracy)
    print_metric("best validation accuracy", best_validation_accuracy)

    if best_lr_entry is not None:
        print_metric("selected training learning rate", best_lr_entry.get("lr"))
        print_metric(
            "best learning-rate-sweep validation loss",
            best_lr_entry.get("best_val_loss"),
        )
        print_metric("best learning-rate-sweep epoch", best_lr_entry.get("best_epoch"))

    if meta:
        for k, v in meta.items():
            print_metric(f"metadata {k}", v)

    for mode_name in ["frozen", "entropy", "consistency"]:
        test_entry = extract_test_metric_entry(metrics.get("test", []), mode_name)
        if test_entry is not None:
            print_metric(f"test mode", mode_name)
            print_metric(f"{mode_name} test accuracy", test_entry.get("acc"))
            print_metric(
                f"{mode_name} test stability-adjusted accuracy",
                test_entry.get("stability_adjusted_acc"),
            )
            print_metric(
                f"{mode_name} test adaptation frequency", test_entry.get("adapt_freq")
            )
            print_metric(
                f"{mode_name} test adaptation loss mean",
                test_entry.get("adapt_loss_mean"),
            )
            print_metric(
                f"{mode_name} selected test-time adaptation learning rate",
                test_entry.get("selected_tta_lr"),
            )
            print_metric(
                f"{mode_name} selected confidence threshold",
                test_entry.get("selected_conf_thresh"),
            )

    print("-" * 60)
