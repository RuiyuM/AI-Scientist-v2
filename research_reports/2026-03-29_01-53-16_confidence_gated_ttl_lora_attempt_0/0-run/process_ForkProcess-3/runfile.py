import os
import numpy as np

working_dir = os.path.join(os.getcwd(), "working")
experiment_data_path = os.path.join(working_dir, "experiment_data.npy")

experiment_data = np.load(experiment_data_path, allow_pickle=True).item()


def safe_last(sequence, default=None):
    if sequence is None or len(sequence) == 0:
        return default
    return sequence[-1]


def safe_best_min(sequence, default=None):
    if sequence is None or len(sequence) == 0:
        return default
    values = []
    for item in sequence:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            values.append(item[1])
    return min(values) if values else default


def safe_best_max(sequence, default=None):
    if sequence is None or len(sequence) == 0:
        return default
    values = []
    for item in sequence:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            values.append(item[1])
    return max(values) if values else default


def format_value(value):
    if value is None:
        return "N/A"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6f}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)


for dataset_name, dataset_data in experiment_data.items():
    print(f"Dataset: {dataset_name}")

    metrics = dataset_data.get("metrics", {})
    losses = dataset_data.get("losses", {})

    train_metric_history = metrics.get("train", [])
    validation_metric_history = metrics.get("val", [])
    test_metric_history = metrics.get("test", [])

    train_loss_history = losses.get("train", [])
    validation_loss_history = losses.get("val", [])

    final_train_metric = safe_last(train_metric_history)
    final_validation_metric = safe_last(validation_metric_history)
    final_train_loss = safe_last(train_loss_history)
    final_validation_loss = safe_last(validation_loss_history)
    final_test_metric = safe_last(test_metric_history)

    if final_train_metric is not None:
        print(f"Final training objective value: {format_value(final_train_metric[1])}")
    else:
        print("Final training objective value: N/A")

    if final_validation_metric is not None:
        print(f"Final validation accuracy: {format_value(final_validation_metric[1])}")
    else:
        print("Final validation accuracy: N/A")

    if final_train_loss is not None:
        print(f"Final training loss: {format_value(final_train_loss[1])}")
    else:
        print("Final training loss: N/A")

    if final_validation_loss is not None:
        print(f"Final validation loss: {format_value(final_validation_loss[1])}")
    else:
        print("Final validation loss: N/A")

    best_training_objective = safe_best_min(train_metric_history)
    best_validation_accuracy = safe_best_max(validation_metric_history)
    best_training_loss = safe_best_min(train_loss_history)
    best_validation_loss = safe_best_min(validation_loss_history)

    print(f"Best training objective value: {format_value(best_training_objective)}")
    print(f"Best validation accuracy: {format_value(best_validation_accuracy)}")
    print(f"Best training loss: {format_value(best_training_loss)}")
    print(f"Best validation loss: {format_value(best_validation_loss)}")

    if (
        final_test_metric is not None
        and isinstance(final_test_metric, (list, tuple))
        and len(final_test_metric) >= 2
    ):
        test_metrics_dict = final_test_metric[1]
        print(
            f"Final test accuracy with frozen policy: {format_value(test_metrics_dict.get('frozen_acc'))}"
        )
        print(
            f"Final test expected calibration error with frozen policy: {format_value(test_metrics_dict.get('frozen_ece'))}"
        )
        print(
            f"Final test accuracy with always-adapt policy: {format_value(test_metrics_dict.get('always_acc'))}"
        )
        print(
            f"Final test expected calibration error with always-adapt policy: {format_value(test_metrics_dict.get('always_ece'))}"
        )
        print(
            f"Final test accuracy with hybrid policy: {format_value(test_metrics_dict.get('hybrid_acc'))}"
        )
        print(
            f"Final test expected calibration error with hybrid policy: {format_value(test_metrics_dict.get('hybrid_ece'))}"
        )
    else:
        print("Final test accuracy with frozen policy: N/A")
        print("Final test expected calibration error with frozen policy: N/A")
        print("Final test accuracy with always-adapt policy: N/A")
        print("Final test expected calibration error with always-adapt policy: N/A")
        print("Final test accuracy with hybrid policy: N/A")
        print("Final test expected calibration error with hybrid policy: N/A")

    predictions = dataset_data.get("predictions", [])
    ground_truth = dataset_data.get("ground_truth", [])
    confidences = dataset_data.get("confidences", [])
    triggers = dataset_data.get("triggers", [])

    if (
        len(predictions) > 0
        and len(ground_truth) > 0
        and len(predictions) == len(ground_truth)
    ):
        final_hybrid_test_accuracy_from_predictions = float(
            np.mean([p == g for p, g in zip(predictions, ground_truth)])
        )
        print(
            f"Final hybrid test accuracy from stored predictions: {format_value(final_hybrid_test_accuracy_from_predictions)}"
        )
    else:
        print("Final hybrid test accuracy from stored predictions: N/A")

    if len(confidences) > 0:
        print(
            f"Final mean confidence under hybrid policy: {format_value(float(np.mean(confidences)))}"
        )
    else:
        print("Final mean confidence under hybrid policy: N/A")

    if len(triggers) > 0:
        print(
            f"Final trigger rate under hybrid policy: {format_value(float(np.mean(triggers)))}"
        )
    else:
        print("Final trigger rate under hybrid policy: N/A")

    split_sizes = dataset_data.get("split_sizes", {})
    if split_sizes:
        print(f"Training sample count: {format_value(split_sizes.get('train'))}")
        print(f"Validation sample count: {format_value(split_sizes.get('val'))}")
        print(f"Test sample count: {format_value(split_sizes.get('test'))}")
    else:
        print("Training sample count: N/A")
        print("Validation sample count: N/A")
        print("Test sample count: N/A")

    print()
