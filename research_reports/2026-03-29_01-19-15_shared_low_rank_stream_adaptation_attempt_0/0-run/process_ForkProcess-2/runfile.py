import os
import numpy as np

working_dir = os.path.join(os.getcwd(), "working")
file_path = os.path.join(working_dir, "experiment_data.npy")

experiment_data = np.load(file_path, allow_pickle=True).item()


def to_python_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def format_value(value):
    value = to_python_scalar(value)
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def get_final_from_pairs(pairs):
    if not pairs:
        return None
    return pairs[-1][1]


def get_best_from_pairs(pairs, mode="max"):
    if not pairs:
        return None
    values = [to_python_scalar(v) for _, v in pairs]
    if mode == "min":
        return min(values)
    return max(values)


def print_metric(label, value):
    if value is not None:
        print(f"{label}: {format_value(value)}")


for dataset_name, dataset_data in experiment_data.items():
    print(f"Dataset: {dataset_name}")

    metrics = dataset_data.get("metrics", {})
    losses = dataset_data.get("losses", {})

    train_accuracy_final = get_final_from_pairs(metrics.get("train", []))
    train_accuracy_best = get_best_from_pairs(metrics.get("train", []), mode="max")
    validation_accuracy_final = get_final_from_pairs(metrics.get("val", []))
    validation_accuracy_best = get_best_from_pairs(metrics.get("val", []), mode="max")

    train_loss_final = get_final_from_pairs(losses.get("train", []))
    train_loss_best = get_best_from_pairs(losses.get("train", []), mode="min")
    validation_loss_final = get_final_from_pairs(losses.get("val", []))
    validation_loss_best = get_best_from_pairs(losses.get("val", []), mode="min")

    stream_summary_metrics = metrics.get("stream", [])
    stream_summary_losses = losses.get("stream", [])

    if train_accuracy_final is not None:
        print_metric("Final train accuracy", train_accuracy_final)
    if train_accuracy_best is not None:
        print_metric("Best train accuracy", train_accuracy_best)
    if validation_accuracy_final is not None:
        print_metric("Final validation accuracy", validation_accuracy_final)
    if validation_accuracy_best is not None:
        print_metric("Best validation accuracy", validation_accuracy_best)

    if train_loss_final is not None:
        print_metric("Final train loss", train_loss_final)
    if train_loss_best is not None:
        print_metric("Best train loss", train_loss_best)
    if validation_loss_final is not None:
        print_metric("Final validation loss", validation_loss_final)
    if validation_loss_best is not None:
        print_metric("Best validation loss", validation_loss_best)

    if stream_summary_metrics:
        for method_name, metric_value in stream_summary_metrics:
            print_metric(f"Final stream mean accuracy ({method_name})", metric_value)

    if stream_summary_losses:
        for method_name, loss_value in stream_summary_losses:
            print_metric(f"Final stream mean loss ({method_name})", loss_value)

    methods = dataset_data.get("methods", {})
    for method_name, method_data in methods.items():
        print(f"Dataset: {dataset_name} | Method: {method_name}")

        method_metrics = method_data.get("metrics", {})
        method_losses = method_data.get("losses", {})

        stream_accuracy_final = get_final_from_pairs(method_metrics.get("stream", []))
        stream_accuracy_best = get_best_from_pairs(
            method_metrics.get("stream", []), mode="max"
        )

        stream_loss_final = get_final_from_pairs(method_losses.get("stream", []))
        stream_loss_best = get_best_from_pairs(
            method_losses.get("stream", []), mode="min"
        )

        adapter_drift = method_data.get("adapter_drift", [])
        final_adapter_drift = adapter_drift[-1] if adapter_drift else None
        best_adapter_drift = min(adapter_drift) if adapter_drift else None

        print_metric("Final online stream accuracy", stream_accuracy_final)
        print_metric("Best online stream accuracy", stream_accuracy_best)
        print_metric("Final online stream loss", stream_loss_final)
        print_metric("Best online stream loss", stream_loss_best)
        print_metric("Final adapter drift", final_adapter_drift)
        print_metric("Best adapter drift", best_adapter_drift)
