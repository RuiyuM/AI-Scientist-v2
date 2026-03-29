import os
import numpy as np

working_dir = os.path.join(os.getcwd(), "working")
file_path = os.path.join(working_dir, "experiment_data.npy")

experiment_data = np.load(file_path, allow_pickle=True).item()


def get_last_metric_value(metric_list):
    if not metric_list:
        return None
    last_item = metric_list[-1]
    if isinstance(last_item, (list, tuple)) and len(last_item) >= 2:
        return last_item[1]
    return last_item


def format_value(value):
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def print_metric(dataset_name, metric_name, value):
    print(f"Dataset: {dataset_name}")
    print(f"{metric_name}: {format_value(value)}")


def print_stream_results(dataset_name, stream_results):
    if not stream_results:
        return

    mode_to_label = {
        "frozen": "frozen stream accuracy",
        "always": "always-update stream accuracy",
        "gated": "gated stream accuracy",
    }
    trigger_to_label = {
        "frozen": "frozen trigger rate",
        "always": "always-update trigger rate",
        "gated": "gated trigger rate",
    }

    for mode in ["frozen", "always", "gated"]:
        if mode in stream_results:
            mode_results = stream_results[mode]
            if "Shifted-Stream Accuracy" in mode_results:
                print_metric(
                    dataset_name,
                    mode_to_label[mode],
                    mode_results["Shifted-Stream Accuracy"],
                )
            if "trigger_rate" in mode_results:
                print_metric(
                    dataset_name,
                    trigger_to_label[mode],
                    mode_results["trigger_rate"],
                )


for dataset_name, dataset_info in experiment_data.items():
    print(f"\n{'=' * 60}")
    print(f"Dataset: {dataset_name}")
    print(f"{'=' * 60}")

    if dataset_name == "synthetic_reasoning_stream":
        selected_epochs = dataset_info.get("selected_epochs", None)
        selected_hparams = dataset_info.get("selected_hparams", None)

        print_metric(
            dataset_name,
            "selected number of training epochs",
            selected_epochs,
        )

        if selected_hparams is not None:
            print_metric(
                dataset_name,
                "selected learning rate",
                selected_hparams.get("lr", None),
            )
            print_metric(
                dataset_name,
                "selected batch size",
                selected_hparams.get("batch_size", None),
            )

    metrics = dataset_info.get("metrics", {})
    losses = dataset_info.get("losses", {})

    train_accuracy = get_last_metric_value(metrics.get("train", []))
    validation_accuracy = get_last_metric_value(metrics.get("val", []))
    test_metrics_entry = get_last_metric_value(metrics.get("test", []))

    train_loss = get_last_metric_value(losses.get("train", []))
    validation_loss = get_last_metric_value(losses.get("val", []))

    if train_accuracy is not None:
        print_metric(dataset_name, "final train accuracy", train_accuracy)
    if validation_accuracy is not None:
        print_metric(dataset_name, "final validation accuracy", validation_accuracy)
    if train_loss is not None:
        print_metric(dataset_name, "final train loss", train_loss)
    if validation_loss is not None:
        print_metric(dataset_name, "final validation loss", validation_loss)

    if isinstance(test_metrics_entry, dict):
        for key, value in test_metrics_entry.items():
            pretty_name = key.replace("_", " ")
            print_metric(dataset_name, f"final {pretty_name}", value)

    severity = dataset_info.get("severity", None)
    if severity is not None:
        print_metric(dataset_name, "dataset severity", severity)

    sna_gain = get_last_metric_value(
        dataset_info.get("shift_normalized_accuracy_gain", [])
    )
    if sna_gain is not None:
        print_metric(
            dataset_name,
            "final shift normalized accuracy gain",
            sna_gain,
        )

    stream_results = dataset_info.get("stream_results", {})
    print_stream_results(dataset_name, stream_results)
