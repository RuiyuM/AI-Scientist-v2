import os
import numpy as np

working_dir = os.path.join(os.getcwd(), "working")
file_path = os.path.join(working_dir, "experiment_data.npy")

experiment_data = np.load(file_path, allow_pickle=True).item()

dataset_names = ["arc_easy_shift", "pubmedqa_burst", "mmlu_clustered"]
ablation_types = ["pseudo_label_self_training", "entropy_minimization"]


def safe_last(seq):
    return seq[-1] if seq else None


def safe_best(seq, value_index, mode="max"):
    if not seq:
        return None
    if mode == "max":
        return max(seq, key=lambda x: x[value_index])
    if mode == "min":
        return min(seq, key=lambda x: x[value_index])
    raise ValueError("mode must be 'max' or 'min'")


def parse_test_metrics(test_entries):
    parsed = {}
    for entry in test_entries:
        if len(entry) != 7:
            continue
        mode_name, acc, ece, trigger_rate, update_rate, overhead, srus = entry
        parsed[mode_name] = {
            "test accuracy": acc,
            "test expected calibration error": ece,
            "test trigger rate": trigger_rate,
            "test update rate": update_rate,
            "test overhead": overhead,
            "test SRUS": srus,
        }
    return parsed


for ablation in ablation_types:
    print(f"\nAblation: {ablation}")
    for dataset_name in dataset_names:
        print(f"\nDataset: {dataset_name}")
        ds_block = experiment_data.get(ablation, {}).get(dataset_name, {})

        train_metrics = ds_block.get("metrics", {}).get("train", [])
        val_metrics = ds_block.get("metrics", {}).get("val", [])
        test_metrics = ds_block.get("metrics", {}).get("test", [])

        train_losses = ds_block.get("losses", {}).get("train", [])
        val_losses = ds_block.get("losses", {}).get("val", [])

        final_train = safe_last(train_metrics)
        best_train_acc = safe_best(train_metrics, 1, mode="max")
        final_train_loss = safe_last(train_losses)
        best_train_loss = safe_best(train_losses, 1, mode="min")

        final_val = safe_last(val_metrics)
        best_val_acc = safe_best(val_metrics, 1, mode="max")
        best_val_srus = safe_best(val_metrics, 2, mode="max")
        final_val_loss = safe_last(val_losses)
        best_val_loss = safe_best(val_losses, 1, mode="min")

        if final_train is not None:
            epoch, train_acc, train_srus_like = final_train
            print(f"final train epoch: {epoch}")
            print(f"final train accuracy: {train_acc:.6f}")
            print(f"final stored train score: {train_srus_like:.6f}")
        if best_train_acc is not None:
            epoch, train_acc, _ = best_train_acc
            print(f"best train accuracy epoch: {epoch}")
            print(f"best train accuracy: {train_acc:.6f}")
        if final_train_loss is not None:
            epoch, train_loss = final_train_loss
            print(f"final train loss epoch: {epoch}")
            print(f"final train loss: {train_loss:.6f}")
        if best_train_loss is not None:
            epoch, train_loss = best_train_loss
            print(f"best train loss epoch: {epoch}")
            print(f"best train loss: {train_loss:.6f}")

        if final_val is not None:
            epoch, val_acc, val_srus = final_val
            print(f"final validation epoch: {epoch}")
            print(f"final validation accuracy: {val_acc:.6f}")
            print(f"final validation SRUS: {val_srus:.6f}")
        if best_val_acc is not None:
            epoch, val_acc, _ = best_val_acc
            print(f"best validation accuracy epoch: {epoch}")
            print(f"best validation accuracy: {val_acc:.6f}")
        if best_val_srus is not None:
            epoch, _, val_srus = best_val_srus
            print(f"best validation SRUS epoch: {epoch}")
            print(f"best validation SRUS: {val_srus:.6f}")
        if final_val_loss is not None:
            epoch, val_loss = final_val_loss
            print(f"final validation loss epoch: {epoch}")
            print(f"final validation loss: {val_loss:.6f}")
        if best_val_loss is not None:
            epoch, val_loss = best_val_loss
            print(f"best validation loss epoch: {epoch}")
            print(f"best validation loss: {val_loss:.6f}")

        parsed_test = parse_test_metrics(test_metrics)
        for mode_name in ["frozen", "always", "gated", "reset"]:
            if mode_name not in parsed_test:
                continue
            print(f"test evaluation mode: {mode_name}")
            mode_metrics = parsed_test[mode_name]
            for metric_name, metric_value in mode_metrics.items():
                print(f"{mode_name} {metric_name}: {metric_value:.6f}")
