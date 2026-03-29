import os
import numpy as np

working_dir = os.path.join(os.getcwd(), "working")
experiment_data_path = os.path.join(working_dir, "experiment_data.npy")

experiment_data = np.load(experiment_data_path, allow_pickle=True).item()


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return x


def format_value(x):
    if isinstance(x, (float, np.floating)):
        return f"{float(x):.6f}"
    return str(x)


def print_metric(label, value):
    print(f"{label}: {format_value(value)}")


def summarize_losses(dataset_name, dataset_dict):
    losses = dataset_dict.get("losses", {})
    split_name_map = {
        "train": "train loss",
        "val": "validation loss",
        "test_unseen": "unseen test loss",
        "test": "test loss",
    }

    for split_key, values in losses.items():
        if values is None or len(values) == 0:
            continue
        arr = np.array(values, dtype=float)
        label_base = split_name_map.get(split_key, f"{split_key} loss")
        print_metric(f"Final {label_base}", arr[-1])
        print_metric(f"Best {label_base}", arr.min())


def summarize_accuracy_metrics(dataset_name, dataset_dict):
    metrics = dataset_dict.get("metrics", {})
    split_name_map = {
        "train": "train",
        "val": "validation",
        "test_unseen": "unseen test",
        "test": "test",
    }

    for split_key, metric_list in metrics.items():
        if metric_list is None or len(metric_list) == 0:
            continue

        split_label = split_name_map.get(split_key, split_key)
        metric_names = sorted(
            {
                k
                for entry in metric_list
                if isinstance(entry, dict)
                for k in entry.keys()
            }
        )

        for metric_name in metric_names:
            values = [
                entry[metric_name]
                for entry in metric_list
                if isinstance(entry, dict) and metric_name in entry
            ]
            if len(values) == 0:
                continue
            arr = np.array(values, dtype=float)

            readable_metric_name = metric_name.replace("_", " ")
            print_metric(f"Final {split_label} {readable_metric_name}", arr[-1])
            print_metric(f"Best {split_label} {readable_metric_name}", arr.max())


def summarize_latent_stats(dataset_name, dataset_dict):
    latent_stats = dataset_dict.get("latent_stats", [])
    if latent_stats is None or len(latent_stats) == 0:
        return

    stat_names = sorted(
        {
            k
            for entry in latent_stats
            if isinstance(entry, dict)
            for k in entry.keys()
            if k != "epoch"
        }
    )
    final_entry = latent_stats[-1]

    for stat_name in stat_names:
        values = [
            entry[stat_name]
            for entry in latent_stats
            if isinstance(entry, dict) and stat_name in entry
        ]
        if len(values) == 0:
            continue
        arr = np.array(values, dtype=float)
        readable_name = stat_name.replace("_", " ")
        if stat_name in final_entry:
            print_metric(f"Final {readable_name}", final_entry[stat_name])
        print_metric(f"Best {readable_name}", arr.max())


def summarize_final_predictions(dataset_name, dataset_dict):
    preds = dataset_dict.get("predictions", None)
    gts = dataset_dict.get("ground_truth", None)

    if preds is None or gts is None:
        return

    preds = np.array(preds)
    gts = np.array(gts)

    if preds.size == 0 or gts.size == 0 or preds.shape != gts.shape:
        return

    if preds.ndim == 2 and preds.shape[1] >= 2:
        color_acc = (preds[:, 0] == gts[:, 0]).mean()
        shape_acc = (preds[:, 1] == gts[:, 1]).mean()
        joint_acc = ((preds[:, 0] == gts[:, 0]) & (preds[:, 1] == gts[:, 1])).mean()

        print_metric("Final saved prediction color accuracy", color_acc)
        print_metric("Final saved prediction shape accuracy", shape_acc)
        print_metric("Final saved prediction joint accuracy", joint_acc)
    else:
        exact_acc = (preds == gts).mean()
        print_metric("Final saved prediction accuracy", exact_acc)


for dataset_name, dataset_dict in experiment_data.items():
    print(f"Dataset: {dataset_name}")
    summarize_losses(dataset_name, dataset_dict)
    summarize_accuracy_metrics(dataset_name, dataset_dict)
    summarize_latent_stats(dataset_name, dataset_dict)
    summarize_final_predictions(dataset_name, dataset_dict)
    print("-" * 60)
