import os
import numpy as np

working_dir = os.path.join(os.getcwd(), "working")
experiment_path = os.path.join(working_dir, "experiment_data.npy")

experiment_data = np.load(experiment_path, allow_pickle=True).item()

ablation_key = "lora_capacity_ablation"
stream_modes = ["frozen", "always", "gated", "reset"]


def safe_last(seq):
    return seq[-1] if seq else None


def print_rank_dataset_metrics(rank_key, dataset_name, dataset_blob):
    print(f"Dataset: {dataset_name} | Rank: {rank_key}")

    metrics = dataset_blob.get("metrics", {})
    losses = dataset_blob.get("losses", {})
    thresholds = dataset_blob.get("thresholds", {})
    summary = dataset_blob.get("summary", {})

    train_metric_final = safe_last(metrics.get("train", []))
    val_metric_final = safe_last(metrics.get("val", []))
    train_loss_final = safe_last(losses.get("train", []))
    val_loss_final = safe_last(losses.get("val", []))

    if train_metric_final is not None:
        epoch, train_accuracy, train_score = train_metric_final
        print(f"  final train epoch: {epoch}")
        print(f"  final train accuracy: {train_accuracy}")
        print(f"  final train score: {train_score}")
    else:
        print("  final train epoch: N/A")
        print("  final train accuracy: N/A")
        print("  final train score: N/A")

    if val_metric_final is not None:
        epoch, validation_accuracy, validation_srus_like_score = val_metric_final
        print(f"  final validation epoch: {epoch}")
        print(f"  final validation accuracy: {validation_accuracy}")
        print(f"  final validation score: {validation_srus_like_score}")
    else:
        print("  final validation epoch: N/A")
        print("  final validation accuracy: N/A")
        print("  final validation score: N/A")

    if train_loss_final is not None:
        epoch, train_loss = train_loss_final
        print(f"  final train loss: {train_loss}")
    else:
        print("  final train loss: N/A")

    if val_loss_final is not None:
        epoch, validation_loss = val_loss_final
        print(f"  final validation loss: {validation_loss}")
    else:
        print("  final validation loss: N/A")

    if thresholds:
        print(f"  entropy threshold: {thresholds.get('entropy', 'N/A')}")
        print(f"  margin threshold: {thresholds.get('margin', 'N/A')}")
        print(f"  confidence threshold: {thresholds.get('conf', 'N/A')}")
    else:
        print("  entropy threshold: N/A")
        print("  margin threshold: N/A")
        print("  confidence threshold: N/A")

    if summary:
        best_accuracy_mode = max(
            summary.items(), key=lambda kv: kv[1].get("acc", float("-inf"))
        )
        best_ece_mode = min(
            summary.items(), key=lambda kv: kv[1].get("ece", float("inf"))
        )
        best_srus_mode = max(
            summary.items(), key=lambda kv: kv[1].get("srus", float("-inf"))
        )
        best_trigger_mode = max(
            summary.items(), key=lambda kv: kv[1].get("trigger_rate", float("-inf"))
        )
        best_update_mode = max(
            summary.items(), key=lambda kv: kv[1].get("update_rate", float("-inf"))
        )
        best_reset_mode = max(
            summary.items(), key=lambda kv: kv[1].get("reset_rate", float("-inf"))
        )
        best_overhead_mode = min(
            summary.items(), key=lambda kv: kv[1].get("overhead", float("inf"))
        )

        print(
            f"  best test accuracy: {best_accuracy_mode[1].get('acc')} (mode: {best_accuracy_mode[0]})"
        )
        print(
            f"  best test expected calibration error: {best_ece_mode[1].get('ece')} (mode: {best_ece_mode[0]})"
        )
        print(
            f"  best test SRUS: {best_srus_mode[1].get('srus')} (mode: {best_srus_mode[0]})"
        )
        print(
            f"  best test trigger rate: {best_trigger_mode[1].get('trigger_rate')} (mode: {best_trigger_mode[0]})"
        )
        print(
            f"  best test update rate: {best_update_mode[1].get('update_rate')} (mode: {best_update_mode[0]})"
        )
        print(
            f"  best test reset rate: {best_reset_mode[1].get('reset_rate')} (mode: {best_reset_mode[0]})"
        )
        print(
            f"  best test overhead: {best_overhead_mode[1].get('overhead')} (mode: {best_overhead_mode[0]})"
        )

        for mode in stream_modes:
            mode_blob = summary.get(mode)
            if mode_blob is None:
                continue
            print(f"  final test accuracy [{mode}]: {mode_blob.get('acc', 'N/A')}")
            print(
                f"  final test expected calibration error [{mode}]: {mode_blob.get('ece', 'N/A')}"
            )
            print(
                f"  final test trigger rate [{mode}]: {mode_blob.get('trigger_rate', 'N/A')}"
            )
            print(
                f"  final test update rate [{mode}]: {mode_blob.get('update_rate', 'N/A')}"
            )
            print(
                f"  final test reset rate [{mode}]: {mode_blob.get('reset_rate', 'N/A')}"
            )
            print(f"  final test overhead [{mode}]: {mode_blob.get('overhead', 'N/A')}")
            print(f"  final test SRUS [{mode}]: {mode_blob.get('srus', 'N/A')}")
            print(
                f"  final adapter rank [{mode}]: {mode_blob.get('adapter_rank', 'N/A')}"
            )
            print(
                f"  final adapter parameter count [{mode}]: {mode_blob.get('adapter_params', 'N/A')}"
            )
    else:
        print("  best test accuracy: N/A")
        print("  best test expected calibration error: N/A")
        print("  best test SRUS: N/A")
        print("  best test trigger rate: N/A")
        print("  best test update rate: N/A")
        print("  best test reset rate: N/A")
        print("  best test overhead: N/A")

    print()


ablation_data = experiment_data.get(ablation_key, {})

for rank_key, rank_blob in ablation_data.items():
    for dataset_name, dataset_blob in rank_blob.items():
        print_rank_dataset_metrics(rank_key, dataset_name, dataset_blob)
