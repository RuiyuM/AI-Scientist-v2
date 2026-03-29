import os
import numpy as np


def load_experiment_data():
    working_dir = os.path.join(os.getcwd(), "working")
    file_path = os.path.join(working_dir, "experiment_data.npy")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find experiment data file: {file_path}")
    return np.load(file_path, allow_pickle=True).item()


def safe_last_pair_value(seq):
    if seq is None or len(seq) == 0:
        return None
    last_item = seq[-1]
    if isinstance(last_item, (list, tuple)) and len(last_item) >= 2:
        return last_item[1]
    return last_item


def safe_best_min_pair_value(seq):
    if seq is None or len(seq) == 0:
        return None
    valid = []
    for item in seq:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            valid.append(item[1])
    if len(valid) == 0:
        return None
    return min(valid)


def safe_best_max_pair_value(seq):
    if seq is None or len(seq) == 0:
        return None
    valid = []
    for item in seq:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            valid.append(item[1])
    if len(valid) == 0:
        return None
    return max(valid)


def format_value(value):
    if value is None:
        return "N/A"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6f}"
    return str(value)


def print_metric(metric_name, value):
    print(f"{metric_name}: {format_value(value)}")


def print_source_training_metrics(dataset_name, dataset_payload):
    print(f"Dataset: {dataset_name}")

    train_acc_final = safe_last_pair_value(
        dataset_payload.get("metrics", {}).get("train", [])
    )
    val_acc_final = safe_last_pair_value(
        dataset_payload.get("metrics", {}).get("val", [])
    )
    train_loss_final = safe_last_pair_value(
        dataset_payload.get("losses", {}).get("train", [])
    )
    val_loss_final = safe_last_pair_value(
        dataset_payload.get("losses", {}).get("val", [])
    )

    train_acc_best = safe_best_max_pair_value(
        dataset_payload.get("metrics", {}).get("train", [])
    )
    val_acc_best = safe_best_max_pair_value(
        dataset_payload.get("metrics", {}).get("val", [])
    )
    train_loss_best = safe_best_min_pair_value(
        dataset_payload.get("losses", {}).get("train", [])
    )
    val_loss_best = safe_best_min_pair_value(
        dataset_payload.get("losses", {}).get("val", [])
    )

    print_metric("final train accuracy", train_acc_final)
    print_metric("best train accuracy", train_acc_best)
    print_metric("final validation accuracy", val_acc_final)
    print_metric("best validation accuracy", val_acc_best)
    print_metric("final train loss", train_loss_final)
    print_metric("best train loss", train_loss_best)
    print_metric("final validation loss", val_loss_final)
    print_metric("best validation loss", val_loss_best)

    selected_epochs = dataset_payload.get("selected_epochs", None)
    selected_hparams = dataset_payload.get("selected_hparams", None)
    if selected_epochs is not None:
        print_metric("selected training epochs", selected_epochs)
    if selected_hparams is not None:
        print_metric("selected source learning rate", selected_hparams.get("lr"))
        print_metric("selected batch size", selected_hparams.get("batch_size"))
        print_metric(
            "selected adaptation learning rate", selected_hparams.get("adapt_lr")
        )
        print_metric("selected reset interval", selected_hparams.get("reset_every"))
        print_metric(
            "selected entropy threshold", selected_hparams.get("entropy_thresh")
        )
        print_metric("selected margin threshold", selected_hparams.get("margin_thresh"))

    print()


def print_stream_dataset_metrics(dataset_name, dataset_payload):
    print(f"Dataset: {dataset_name}")

    severity = dataset_payload.get("severity", None)
    print_metric("dataset severity", severity)

    stream_results = dataset_payload.get("stream_results", {})
    frozen = stream_results.get("frozen", {})
    always = stream_results.get("always", {})
    gated = stream_results.get("gated", {})

    print_metric("frozen-stream test accuracy", frozen.get("Shifted-Stream Accuracy"))
    print_metric("frozen-stream trigger rate", frozen.get("trigger_rate"))
    print_metric(
        "always-adapted-stream test accuracy", always.get("Shifted-Stream Accuracy")
    )
    print_metric("always-adapted-stream trigger rate", always.get("trigger_rate"))
    print_metric(
        "gated-adapted-stream test accuracy", gated.get("Shifted-Stream Accuracy")
    )
    print_metric("gated-adapted-stream trigger rate", gated.get("trigger_rate"))

    sna_gain = safe_last_pair_value(
        dataset_payload.get("shift_normalized_accuracy_gain", [])
    )
    print_metric("shift-normalized accuracy gain", sna_gain)

    test_metrics_entries = dataset_payload.get("metrics", {}).get("test", [])
    if len(test_metrics_entries) > 0:
        test_metrics = safe_last_pair_value(test_metrics_entries)
        if isinstance(test_metrics, dict):
            print_metric(
                "frozen-stream test accuracy (detailed)", test_metrics.get("frozen_acc")
            )
            print_metric(
                "always-adapted-stream test accuracy (detailed)",
                test_metrics.get("always_acc"),
            )
            print_metric(
                "gated-adapted-stream test accuracy (detailed)",
                test_metrics.get("gated_acc"),
            )
            print_metric(
                "shift-normalized accuracy gain (detailed)",
                test_metrics.get("shift_normalized_accuracy_gain"),
            )

    predictions = dataset_payload.get("predictions", [])
    ground_truth = dataset_payload.get("ground_truth", [])
    print_metric("number of gated predictions", len(predictions))
    print_metric("number of ground-truth labels", len(ground_truth))

    print()


experiment_data = load_experiment_data()

dataset_order = [
    "synthetic_reasoning_stream",
    "feature_permutation_stream",
    "nonlinear_boundary_stream",
    "label_flip_stream",
    "correlated_noise_stream",
]

if "synthetic_reasoning_stream" in experiment_data:
    print_source_training_metrics(
        "synthetic_reasoning_stream", experiment_data["synthetic_reasoning_stream"]
    )

for dataset_name in dataset_order:
    if dataset_name in experiment_data:
        print_stream_dataset_metrics(dataset_name, experiment_data[dataset_name])
