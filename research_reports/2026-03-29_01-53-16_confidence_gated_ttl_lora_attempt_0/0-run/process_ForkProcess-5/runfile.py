import os
import numpy as np

working_dir = os.path.join(os.getcwd(), "working")
file_path = os.path.join(working_dir, "experiment_data.npy")

experiment_data = np.load(file_path, allow_pickle=True).item()

if not isinstance(experiment_data, dict) or len(experiment_data) == 0:
    raise ValueError("Loaded experiment_data is empty or not a dictionary.")

ablation_name = next(iter(experiment_data.keys()))
ablation_data = experiment_data[ablation_name]

test_metric_names = [
    "test accuracy",
    "test expected calibration error",
    "test trigger rate",
    "test update rate",
    "test reset rate",
    "test reset count",
    "test overhead",
    "test SRUS",
]

for dataset_name, dataset_data in ablation_data.items():
    print(f"Dataset: {dataset_name}")

    metrics = dataset_data.get("metrics", {})
    losses = dataset_data.get("losses", {})

    train_metric_entries = metrics.get("train", [])
    val_metric_entries = metrics.get("val", [])
    test_metric_entries = metrics.get("test", [])

    train_loss_entries = losses.get("train", [])
    val_loss_entries = losses.get("val", [])

    if train_metric_entries:
        final_train_epoch, final_train_accuracy, final_train_srus_like = (
            train_metric_entries[-1]
        )
        print(f"Final train accuracy: {final_train_accuracy}")
    else:
        print("Final train accuracy: N/A")

    if val_metric_entries:
        final_val_epoch, final_validation_accuracy, final_validation_srus = (
            val_metric_entries[-1]
        )
        print(f"Final validation accuracy: {final_validation_accuracy}")
        print(f"Final validation SRUS: {final_validation_srus}")
    else:
        print("Final validation accuracy: N/A")
        print("Final validation SRUS: N/A")

    if train_loss_entries:
        final_train_loss_epoch, final_train_loss = train_loss_entries[-1]
        print(f"Final train loss: {final_train_loss}")
    else:
        print("Final train loss: N/A")

    if val_loss_entries:
        final_val_loss_epoch, final_validation_loss = val_loss_entries[-1]
        print(f"Final validation loss: {final_validation_loss}")
    else:
        print("Final validation loss: N/A")

    if test_metric_entries:
        # Each entry format:
        # (mode_name, acc, ece, trigger_rate, update_rate, reset_rate, reset_count, overhead, srus)
        best_test_entry = max(test_metric_entries, key=lambda x: x[-1])
        best_mode_name = best_test_entry[0]
        print(f"Best test evaluation mode: {best_mode_name}")

        for metric_name, metric_value in zip(test_metric_names, best_test_entry[1:]):
            print(f"Best {metric_name}: {metric_value}")
    else:
        print("Best test evaluation mode: N/A")
        for metric_name in test_metric_names:
            print(f"Best {metric_name}: N/A")

    print()
