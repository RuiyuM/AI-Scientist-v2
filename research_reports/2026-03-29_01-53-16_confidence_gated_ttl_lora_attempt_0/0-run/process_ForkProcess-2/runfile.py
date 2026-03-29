import os
import numpy as np

working_dir = os.path.join(os.getcwd(), "working")
experiment_path = os.path.join(working_dir, "experiment_data.npy")

experiment_data = np.load(experiment_path, allow_pickle=True).item()


def safe_last_metric(metric_list):
    if metric_list is None or len(metric_list) == 0:
        return None
    return metric_list[-1]


def format_value(value):
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6f}"
    return str(value)


def print_metric(metric_name, value):
    print(f"{metric_name}: {format_value(value)}")


for dataset_name, dataset_info in experiment_data.items():
    print(f"Dataset: {dataset_name}")

    metrics = dataset_info.get("metrics", {})
    losses = dataset_info.get("losses", {})
    stream_results = dataset_info.get("stream_results", {})
    selected_hparams = dataset_info.get("selected_hparams", None)
    selected_epochs = dataset_info.get("selected_epochs", None)
    severity = dataset_info.get("severity", None)
    sna_gain_list = dataset_info.get("shift_normalized_accuracy_gain", [])

    train_acc_entry = safe_last_metric(metrics.get("train", []))
    if train_acc_entry is not None:
        if isinstance(train_acc_entry, (list, tuple)) and len(train_acc_entry) >= 2:
            print_metric("final train accuracy", train_acc_entry[1])

    val_acc_entry = safe_last_metric(metrics.get("val", []))
    if val_acc_entry is not None:
        if isinstance(val_acc_entry, (list, tuple)) and len(val_acc_entry) >= 2:
            print_metric("final validation accuracy", val_acc_entry[1])

    train_loss_entry = safe_last_metric(losses.get("train", []))
    if train_loss_entry is not None:
        if isinstance(train_loss_entry, (list, tuple)) and len(train_loss_entry) >= 2:
            print_metric("final train loss", train_loss_entry[1])

    val_loss_entry = safe_last_metric(losses.get("val", []))
    if val_loss_entry is not None:
        if isinstance(val_loss_entry, (list, tuple)) and len(val_loss_entry) >= 2:
            print_metric("final validation loss", val_loss_entry[1])

    test_entry = safe_last_metric(metrics.get("test", []))
    if test_entry is not None:
        if isinstance(test_entry, (list, tuple)) and len(test_entry) >= 2:
            test_payload = test_entry[1]
            if isinstance(test_payload, dict):
                for key, value in test_payload.items():
                    pretty_name = key.replace("_", " ")
                    print_metric(f"final {pretty_name}", value)
            else:
                print_metric("final test metric", test_payload)

    if severity is not None:
        print_metric("dataset severity", severity)

    sna_gain_entry = safe_last_metric(sna_gain_list)
    if sna_gain_entry is not None:
        if isinstance(sna_gain_entry, (list, tuple)) and len(sna_gain_entry) >= 2:
            print_metric("final shift normalized accuracy gain", sna_gain_entry[1])
        else:
            print_metric("final shift normalized accuracy gain", sna_gain_entry)

    if isinstance(stream_results, dict) and len(stream_results) > 0:
        frozen = stream_results.get("frozen", {})
        always = stream_results.get("always", {})
        gated = stream_results.get("gated", {})

        if "Shifted-Stream Accuracy" in frozen:
            print_metric(
                "final frozen shifted stream accuracy",
                frozen["Shifted-Stream Accuracy"],
            )
        if "trigger_rate" in frozen:
            print_metric("final frozen trigger rate", frozen["trigger_rate"])

        if "Shifted-Stream Accuracy" in always:
            print_metric(
                "final always adaptation shifted stream accuracy",
                always["Shifted-Stream Accuracy"],
            )
        if "trigger_rate" in always:
            print_metric("final always adaptation trigger rate", always["trigger_rate"])

        if "Shifted-Stream Accuracy" in gated:
            print_metric(
                "final gated adaptation shifted stream accuracy",
                gated["Shifted-Stream Accuracy"],
            )
        if "trigger_rate" in gated:
            print_metric("final gated adaptation trigger rate", gated["trigger_rate"])

    if selected_epochs is not None:
        print_metric("selected number of training epochs", selected_epochs)

    if isinstance(selected_hparams, dict) and len(selected_hparams) > 0:
        for key, value in selected_hparams.items():
            pretty_name = key.replace("_", " ")
            print_metric(f"selected hyperparameter {pretty_name}", value)

    print()
