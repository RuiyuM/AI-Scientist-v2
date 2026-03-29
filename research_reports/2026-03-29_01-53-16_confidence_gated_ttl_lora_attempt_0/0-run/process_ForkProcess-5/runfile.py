import os
import numpy as np

working_dir = os.path.join(os.getcwd(), "working")
experiment_path = os.path.join(working_dir, "experiment_data.npy")

experiment_data = np.load(experiment_path, allow_pickle=True).item()

TEST_METRIC_INDEX = {
    "test accuracy": 1,
    "test ECE": 2,
    "test trigger rate": 3,
    "test update rate": 4,
    "test reset rate": 5,
    "test overhead": 6,
    "test SRUS": 7,
}

TEST_METRIC_BEST_RULE = {
    "test accuracy": "max",
    "test ECE": "min",
    "test trigger rate": "min",
    "test update rate": "min",
    "test reset rate": "min",
    "test overhead": "min",
    "test SRUS": "max",
}


def safe_last(seq):
    return seq[-1] if seq else None


def extract_final_train_metrics(ds_blob):
    out = {}
    train_metrics = ds_blob.get("metrics", {}).get("train", [])
    train_losses = ds_blob.get("losses", {}).get("train", [])

    last_train_metric = safe_last(train_metrics)
    last_train_loss = safe_last(train_losses)

    if last_train_metric is not None:
        # original structure: (epoch, acc, acc)
        out["train accuracy"] = float(last_train_metric[1])

    if last_train_loss is not None:
        # original structure: (epoch, loss)
        out["train loss"] = float(last_train_loss[1])

    return out


def extract_best_validation_metrics(ds_blob):
    out = {}
    val_metrics = ds_blob.get("metrics", {}).get("val", [])
    val_losses = ds_blob.get("losses", {}).get("val", [])

    if val_metrics:
        # original structure: (epoch, acc, srus)
        best_val_acc_row = max(val_metrics, key=lambda x: x[1])
        best_val_srus_row = max(val_metrics, key=lambda x: x[2])
        out["validation accuracy"] = float(best_val_acc_row[1])
        out["validation SRUS"] = float(best_val_srus_row[2])

    if val_losses:
        # original structure: (epoch, loss)
        best_val_loss_row = min(val_losses, key=lambda x: x[1])
        out["validation loss"] = float(best_val_loss_row[1])

    return out


def extract_best_test_metrics(ds_blob):
    out = {}
    test_metrics = ds_blob.get("metrics", {}).get("test", [])

    # original structure per entry:
    # (mode_name, acc, ece, trigger_rate, update_rate, reset_rate, overhead, srus)
    for metric_name, idx in TEST_METRIC_INDEX.items():
        if not test_metrics:
            continue

        rule = TEST_METRIC_BEST_RULE[metric_name]
        if rule == "max":
            best_row = max(test_metrics, key=lambda x: x[idx])
        else:
            best_row = min(test_metrics, key=lambda x: x[idx])

        out[metric_name] = float(best_row[idx])
        out[f"{metric_name} mode"] = str(best_row[0])

    return out


def collect_dataset_names(exp_data):
    dataset_names = set()
    for ablation_name, ablation_blob in exp_data.items():
        if isinstance(ablation_blob, dict):
            dataset_names.update(ablation_blob.keys())
    return sorted(dataset_names)


def collect_ablation_names(exp_data):
    return sorted(exp_data.keys())


dataset_names = collect_dataset_names(experiment_data)
ablation_names = collect_ablation_names(experiment_data)

for dataset_name in dataset_names:
    print(f"Dataset: {dataset_name}")

    for ablation_name in ablation_names:
        ds_blob = experiment_data.get(ablation_name, {}).get(dataset_name)
        if ds_blob is None:
            continue

        print(f"Ablation: {ablation_name}")

        train_summary = extract_final_train_metrics(ds_blob)
        val_summary = extract_best_validation_metrics(ds_blob)
        test_summary = extract_best_test_metrics(ds_blob)

        ordered_metric_names = [
            "train accuracy",
            "train loss",
            "validation accuracy",
            "validation loss",
            "validation SRUS",
            "test accuracy",
            "test ECE",
            "test trigger rate",
            "test update rate",
            "test reset rate",
            "test overhead",
            "test SRUS",
        ]

        for metric_name in ordered_metric_names:
            if metric_name in train_summary:
                print(f"{metric_name}: {train_summary[metric_name]:.6f}")
            elif metric_name in val_summary:
                print(f"{metric_name}: {val_summary[metric_name]:.6f}")
            elif metric_name in test_summary:
                print(f"{metric_name}: {test_summary[metric_name]:.6f}")
                mode_key = f"{metric_name} mode"
                if mode_key in test_summary:
                    print(f"{metric_name} mode: {test_summary[mode_key]}")

        print()

    print("-" * 60)
