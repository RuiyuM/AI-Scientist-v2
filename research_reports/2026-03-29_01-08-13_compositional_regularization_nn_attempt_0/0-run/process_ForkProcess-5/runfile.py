import os
import numpy as np

working_dir = os.path.join(os.getcwd(), "working")
file_path = os.path.join(working_dir, "experiment_data.npy")

experiment_data = np.load(file_path, allow_pickle=True).item()


def to_scalar(value):
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return value.item()
        return value.tolist()
    return value


def print_dataset_metrics(dataset_name, dataset_data):
    print(f"Dataset: {dataset_name}")

    epoch_budgets = [to_scalar(x) for x in dataset_data.get("epoch_budgets", [])]
    best_val_acc_per_budget = [
        to_scalar(x) for x in dataset_data.get("best_val_acc_per_budget", [])
    ]
    final_val_acc_per_budget = [
        to_scalar(x) for x in dataset_data.get("final_val_acc_per_budget", [])
    ]
    best_epoch_per_budget = [
        to_scalar(x) for x in dataset_data.get("best_epoch_per_budget", [])
    ]
    best_val_loss_per_budget = [
        to_scalar(x) for x in dataset_data.get("best_val_loss_per_budget", [])
    ]

    for i, epoch_budget in enumerate(epoch_budgets):
        print(f"  Epoch budget: {epoch_budget}")

        if i < len(best_val_acc_per_budget):
            print(f"  Best validation accuracy: {best_val_acc_per_budget[i]:.6f}")
        if i < len(final_val_acc_per_budget):
            print(f"  Final validation accuracy: {final_val_acc_per_budget[i]:.6f}")
        if i < len(best_val_loss_per_budget):
            print(f"  Best validation loss: {best_val_loss_per_budget[i]:.6f}")
        if i < len(best_epoch_per_budget):
            print(f"  Best epoch: {best_epoch_per_budget[i]}")

        print()

    summaries = dataset_data.get("summaries", [])
    if summaries:
        best_summary = max(
            summaries,
            key=lambda s: (
                to_scalar(s.get("best_val_acc", float("-inf"))),
                -to_scalar(s.get("best_val_loss", float("inf"))),
            ),
        )
        print("  Overall best summary:")
        print(f"  Best epoch budget: {to_scalar(best_summary.get('epochs'))}")
        print(
            f"  Best validation accuracy: {to_scalar(best_summary.get('best_val_acc')):.6f}"
        )
        print(
            f"  Best validation loss: {to_scalar(best_summary.get('best_val_loss')):.6f}"
        )
        print(f"  Best epoch: {to_scalar(best_summary.get('best_epoch'))}")
        print(
            f"  Final validation accuracy: {to_scalar(best_summary.get('final_val_acc')):.6f}"
        )
        print(
            f"  Final validation loss: {to_scalar(best_summary.get('final_val_loss')):.6f}"
        )

    predictions = dataset_data.get("predictions", [])
    ground_truth = dataset_data.get("ground_truth", [])
    if len(predictions) > 0:
        print(f"  Number of stored best-validation predictions: {len(predictions)}")
    if len(ground_truth) > 0:
        print(f"  Number of stored ground-truth labels: {len(ground_truth)}")

    print("-" * 60)


epoch_tuning_data = experiment_data.get("epoch_tuning", {})

for dataset_name, dataset_data in epoch_tuning_data.items():
    print_dataset_metrics(dataset_name, dataset_data)
