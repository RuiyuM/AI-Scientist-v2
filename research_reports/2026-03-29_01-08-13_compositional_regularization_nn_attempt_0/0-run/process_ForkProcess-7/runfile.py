import os
import numpy as np

working_dir = os.path.join(os.getcwd(), "working")
file_path = os.path.join(working_dir, "experiment_data.npy")

experiment_data = np.load(file_path, allow_pickle=True).item()

batch_size_tuning = experiment_data.get("batch_size_tuning", {})

dataset_keys = [key for key in batch_size_tuning.keys() if key != "summary"]

for dataset_name in dataset_keys:
    print(f"Dataset: {dataset_name}")

    dataset_runs = batch_size_tuning.get(dataset_name, {})
    run_keys = sorted(
        [k for k in dataset_runs.keys() if isinstance(dataset_runs[k], dict)],
        key=lambda x: int(x.split("_")[-1]) if x.split("_")[-1].isdigit() else x,
    )

    for run_key in run_keys:
        run_data = dataset_runs[run_key]
        print(f"Run: {run_key}")

        train_metrics = run_data.get("metrics", {}).get("train")
        val_metrics = run_data.get("metrics", {}).get("val")
        train_losses = run_data.get("losses", {}).get("train")
        val_losses = run_data.get("losses", {}).get("val")

        if train_metrics is not None and len(train_metrics) > 0:
            final_train_accuracy = float(train_metrics[-1, 1])
            best_train_accuracy = float(np.max(train_metrics[:, 1]))
            best_train_accuracy_epoch = int(
                train_metrics[np.argmax(train_metrics[:, 1]), 0]
            )
            print(f"train accuracy (final): {final_train_accuracy:.6f}")
            print(f"train accuracy (best): {best_train_accuracy:.6f}")
            print(f"train accuracy best epoch: {best_train_accuracy_epoch}")

        if val_metrics is not None and len(val_metrics) > 0:
            final_validation_accuracy_from_curve = float(val_metrics[-1, 1])
            best_validation_accuracy_from_curve = float(np.max(val_metrics[:, 1]))
            best_validation_accuracy_epoch_from_curve = int(
                val_metrics[np.argmax(val_metrics[:, 1]), 0]
            )
            print(
                f"validation accuracy (final from history): {final_validation_accuracy_from_curve:.6f}"
            )
            print(
                f"validation accuracy (best from history): {best_validation_accuracy_from_curve:.6f}"
            )
            print(
                f"validation accuracy best epoch (from history): {best_validation_accuracy_epoch_from_curve}"
            )

        if train_losses is not None and len(train_losses) > 0:
            final_train_loss = float(train_losses[-1, 1])
            best_train_loss = float(np.min(train_losses[:, 1]))
            best_train_loss_epoch = int(train_losses[np.argmin(train_losses[:, 1]), 0])
            print(f"train loss (final): {final_train_loss:.6f}")
            print(f"train loss (best minimum): {best_train_loss:.6f}")
            print(f"train loss best epoch: {best_train_loss_epoch}")

        if val_losses is not None and len(val_losses) > 0:
            final_validation_loss = float(val_losses[-1, 1])
            best_validation_loss = float(np.min(val_losses[:, 1]))
            best_validation_loss_epoch = int(val_losses[np.argmin(val_losses[:, 1]), 0])
            print(f"validation loss (final): {final_validation_loss:.6f}")
            print(f"validation loss (best minimum): {best_validation_loss:.6f}")
            print(f"validation loss best epoch: {best_validation_loss_epoch}")

        if "final_val_acc" in run_data:
            print(
                f"validation accuracy (final stored scalar): {float(run_data['final_val_acc']):.6f}"
            )
        if "best_val_acc" in run_data:
            print(
                f"validation accuracy (best stored scalar): {float(run_data['best_val_acc']):.6f}"
            )
        if "best_epoch" in run_data:
            print(
                f"validation accuracy best epoch (stored scalar): {int(run_data['best_epoch'])}"
            )
        if "batch_size" in run_data:
            print(f"batch size: {int(run_data['batch_size'])}")
        if "reg_lambda" in run_data:
            print(f"regularization lambda: {float(run_data['reg_lambda']):.6f}")

    summary = batch_size_tuning.get("summary", {}).get(dataset_name, None)
    if summary is not None:
        print(f"Dataset summary for: {dataset_name}")

        batch_sizes = summary.get("batch_sizes")
        final_val_accs = summary.get("final_val_accs")
        best_val_accs = summary.get("best_val_accs")
        best_epochs = summary.get("best_epochs")

        if batch_sizes is not None:
            print(f"batch sizes evaluated: {batch_sizes.tolist()}")
        if final_val_accs is not None:
            print(
                f"final validation accuracies by batch size: {final_val_accs.tolist()}"
            )
        if best_val_accs is not None:
            print(f"best validation accuracies by batch size: {best_val_accs.tolist()}")
        if best_epochs is not None:
            print(
                f"best validation accuracy epochs by batch size: {best_epochs.tolist()}"
            )

        if (
            batch_sizes is not None
            and best_val_accs is not None
            and len(batch_sizes) == len(best_val_accs)
            and len(batch_sizes) > 0
        ):
            best_idx = int(np.argmax(best_val_accs))
            print(
                f"best batch size by validation accuracy: {int(batch_sizes[best_idx])}"
            )
            print(
                f"best validation accuracy across batch sizes: {float(best_val_accs[best_idx]):.6f}"
            )
            if best_epochs is not None and len(best_epochs) > best_idx:
                print(
                    f"best validation accuracy epoch for best batch size: {int(best_epochs[best_idx])}"
                )
