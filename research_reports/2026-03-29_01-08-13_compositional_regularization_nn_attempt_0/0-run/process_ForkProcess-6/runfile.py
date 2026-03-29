import os
import numpy as np

working_dir = os.path.join(os.getcwd(), "working")
experiment_data_path = os.path.join(working_dir, "experiment_data.npy")

experiment_data = np.load(experiment_data_path, allow_pickle=True).item()


def safe_final_from_epoch_metric(arr):
    if arr is None:
        return None
    arr = np.asarray(arr)
    if arr.size == 0:
        return None
    if arr.ndim >= 2 and arr.shape[1] >= 2:
        return float(arr[-1, 1])
    return float(arr[-1])


def print_run_metrics(dataset_name, run_name, payload):
    print(f"Dataset: {dataset_name}")
    print(f"Configuration: {run_name}")

    metrics = payload.get("metrics", {})
    losses = payload.get("losses", {})

    final_train_accuracy = safe_final_from_epoch_metric(metrics.get("train"))
    final_validation_accuracy_from_curve = safe_final_from_epoch_metric(
        metrics.get("val")
    )
    final_train_loss = safe_final_from_epoch_metric(losses.get("train"))
    final_validation_loss_from_curve = safe_final_from_epoch_metric(losses.get("val"))
    final_regularization_value = safe_final_from_epoch_metric(
        payload.get("regularization")
    )

    if "lr" in payload:
        print(f"learning rate: {payload['lr']}")
    if "reg_lambda" in payload:
        print(f"regularization lambda: {payload['reg_lambda']}")
    if "best_epoch" in payload:
        print(f"best epoch: {payload['best_epoch']}")
    if "best_val_acc" in payload:
        print(f"best validation accuracy: {payload['best_val_acc']}")
    if final_train_accuracy is not None:
        print(f"final train accuracy: {final_train_accuracy}")
    if "final_val_acc" in payload:
        print(f"final validation accuracy: {payload['final_val_acc']}")
    elif final_validation_accuracy_from_curve is not None:
        print(f"final validation accuracy: {final_validation_accuracy_from_curve}")
    if final_train_loss is not None:
        print(f"final train loss: {final_train_loss}")
    if "final_val_loss" in payload:
        print(f"final validation loss: {payload['final_val_loss']}")
    elif final_validation_loss_from_curve is not None:
        print(f"final validation loss: {final_validation_loss_from_curve}")
    if final_regularization_value is not None:
        print(f"final regularization value: {final_regularization_value}")

    predictions = payload.get("predictions")
    ground_truth = payload.get("ground_truth")
    if predictions is not None and ground_truth is not None:
        predictions = np.asarray(predictions)
        ground_truth = np.asarray(ground_truth)
        if predictions.shape == ground_truth.shape and predictions.size > 0:
            best_checkpoint_validation_accuracy = float(
                (predictions == ground_truth).mean()
            )
            print(
                f"best-checkpoint validation accuracy from saved predictions: {best_checkpoint_validation_accuracy}"
            )

    print("-" * 60)


for experiment_group_name, experiment_group_payload in experiment_data.items():
    for dataset_name, dataset_payload in experiment_group_payload.items():
        for run_name, run_payload in dataset_payload.items():
            print_run_metrics(dataset_name, run_name, run_payload)
