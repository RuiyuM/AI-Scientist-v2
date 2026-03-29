import os
import numpy as np


def safe_float(x):
    try:
        if isinstance(x, np.ndarray):
            if x.size == 0:
                return None
            return float(np.ravel(x)[-1])
        return float(x)
    except Exception:
        return None


def print_metric(metric_name, value):
    if value is None:
        print(f"{metric_name}: N/A")
    else:
        if isinstance(value, float):
            print(f"{metric_name}: {value:.6f}")
        else:
            print(f"{metric_name}: {value}")


def extract_final_acc(metric_list):
    if not metric_list:
        return None
    last_item = metric_list[-1]
    if isinstance(last_item, dict) and "acc" in last_item:
        return safe_float(last_item["acc"])
    return None


def extract_final_loss(loss_list):
    if not loss_list:
        return None
    last_item = loss_list[-1]
    if isinstance(last_item, dict) and "loss" in last_item:
        return safe_float(last_item["loss"])
    return None


def compute_accuracy_from_preds(preds, gts):
    try:
        preds = np.asarray(preds)
        gts = np.asarray(gts)
        if preds.size == 0 or gts.size == 0 or preds.shape != gts.shape:
            return None
        return float(np.mean(preds == gts))
    except Exception:
        return None


def extract_best_tuning_metrics(tuning_results):
    if not tuning_results:
        return None

    best_row = None
    best_anchor = -np.inf

    for row in tuning_results:
        if not isinstance(row, dict):
            continue
        anchored_acc = row.get("anchored_stream_acc", None)
        anchored_acc = safe_float(anchored_acc)
        if anchored_acc is not None and anchored_acc > best_anchor:
            best_anchor = anchored_acc
            best_row = row

    return best_row


def analyze_dataset(dataset_name, dataset_data):
    print(f"Dataset: {dataset_name}")

    metrics = dataset_data.get("metrics", {})
    losses = dataset_data.get("losses", {})
    tuning_results = dataset_data.get("tuning_results", [])
    selected_config = dataset_data.get("selected_config", {})
    per_strategy = dataset_data.get("per_strategy", {})
    predictions = dataset_data.get("predictions", [])
    ground_truth = dataset_data.get("ground_truth", [])

    final_train_accuracy = extract_final_acc(metrics.get("train", []))
    final_validation_accuracy = extract_final_acc(metrics.get("val", []))
    final_train_loss = extract_final_loss(losses.get("train", []))
    final_validation_loss = extract_final_loss(losses.get("val", []))

    print_metric("final train accuracy", final_train_accuracy)
    print_metric("final validation accuracy", final_validation_accuracy)
    print_metric("final train loss", final_train_loss)
    print_metric("final validation loss", final_validation_loss)

    selected_base_training_epochs = selected_config.get("base_training_epochs", None)
    selected_best_validation_loss = safe_float(
        selected_config.get("best_val_loss", None)
    )
    selected_anchored_stream_accuracy = safe_float(
        selected_config.get("anchored_stream_acc", None)
    )

    print_metric("selected base training epochs", selected_base_training_epochs)
    print_metric(
        "selected configuration best validation loss", selected_best_validation_loss
    )
    print_metric(
        "selected configuration anchored stream accuracy",
        selected_anchored_stream_accuracy,
    )

    stream_metrics = metrics.get("stream", [])
    if stream_metrics:
        stream_item = stream_metrics[-1]
        if isinstance(stream_item, dict):
            print_metric(
                "selected configuration reset stream accuracy",
                safe_float(stream_item.get("reset_acc", None)),
            )
            print_metric(
                "selected configuration carry stream accuracy",
                safe_float(stream_item.get("carry_acc", None)),
            )
            print_metric(
                "selected configuration anchored stream accuracy",
                safe_float(stream_item.get("anchored_acc", None)),
            )

    selected_prediction_accuracy = compute_accuracy_from_preds(
        predictions, ground_truth
    )
    print_metric(
        "selected anchored prediction accuracy from stored predictions",
        selected_prediction_accuracy,
    )

    best_tuning = extract_best_tuning_metrics(tuning_results)
    if best_tuning is not None:
        print_metric(
            "best tuned base training epochs",
            best_tuning.get("base_training_epochs", None),
        )
        print_metric(
            "best tuned validation loss",
            safe_float(best_tuning.get("best_val_loss", None)),
        )
        print_metric(
            "best tuned reset stream accuracy",
            safe_float(best_tuning.get("reset_stream_acc", None)),
        )
        print_metric(
            "best tuned carry stream accuracy",
            safe_float(best_tuning.get("carry_stream_acc", None)),
        )
        print_metric(
            "best tuned anchored stream accuracy",
            safe_float(best_tuning.get("anchored_stream_acc", None)),
        )

    if selected_base_training_epochs is not None:
        strategy_key = f"epochs_{selected_base_training_epochs}"
        strategy_block = per_strategy.get(strategy_key, {})
        for strategy_name in ["reset", "carry", "anchored"]:
            strategy_data = strategy_block.get(strategy_name, {})
            strategy_metrics = strategy_data.get("metrics", {})
            strategy_stream_acc = safe_float(strategy_metrics.get("stream_acc", None))

            cumulative_acc = strategy_data.get("cumulative_acc", [])
            final_cumulative_acc = None
            if isinstance(cumulative_acc, np.ndarray):
                if cumulative_acc.size > 0:
                    final_cumulative_acc = safe_float(cumulative_acc[-1])
            elif isinstance(cumulative_acc, list) and len(cumulative_acc) > 0:
                final_cumulative_acc = safe_float(cumulative_acc[-1])

            strategy_preds = strategy_data.get("predictions", [])
            strategy_gts = strategy_data.get("ground_truth", [])
            recomputed_acc = compute_accuracy_from_preds(strategy_preds, strategy_gts)

            print_metric(
                f"selected {strategy_name} stream accuracy", strategy_stream_acc
            )
            print_metric(
                f"selected {strategy_name} final cumulative accuracy",
                final_cumulative_acc,
            )
            print_metric(
                f"selected {strategy_name} prediction accuracy from stored predictions",
                recomputed_acc,
            )

    print("")


working_dir = os.path.join(os.getcwd(), "working")
file_path = os.path.join(working_dir, "experiment_data.npy")

experiment_data = np.load(file_path, allow_pickle=True).item()

for experiment_name, datasets in experiment_data.items():
    if not isinstance(datasets, dict):
        continue
    for dataset_name, dataset_data in datasets.items():
        analyze_dataset(dataset_name, dataset_data)
