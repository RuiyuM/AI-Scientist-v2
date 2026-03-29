import os
import numpy as np

working_dir = os.path.join(os.getcwd(), "working")
file_path = os.path.join(working_dir, "experiment_data.npy")

experiment_data = np.load(file_path, allow_pickle=True).item()


def get_last_value(metric_list):
    if metric_list is None or len(metric_list) == 0:
        return None
    last_item = metric_list[-1]
    if isinstance(last_item, (list, tuple)) and len(last_item) >= 2:
        return last_item[1]
    return last_item


def safe_print_metric(metric_name, value):
    if value is None:
        print(f"{metric_name}: N/A")
    else:
        print(f"{metric_name}: {value}")


for dataset_name, dataset_payload in experiment_data.items():
    print(f"Dataset: {dataset_name}")

    top_metrics = dataset_payload.get("metrics", {})
    top_losses = dataset_payload.get("losses", {})
    best_lr_per_method = dataset_payload.get("best_lr_per_method", {})
    task_catalog = dataset_payload.get("task_catalog", [])
    hf_proxy_datasets = dataset_payload.get("hf_proxy_datasets", [])

    final_train_accuracy = get_last_value(top_metrics.get("train", []))
    final_validation_accuracy = get_last_value(top_metrics.get("val", []))
    final_stream_summary = top_metrics.get("stream", [])
    final_train_loss = get_last_value(top_losses.get("train", []))
    final_validation_loss = get_last_value(top_losses.get("val", []))
    final_stream_loss_summary = top_losses.get("stream", [])

    safe_print_metric("final train accuracy", final_train_accuracy)
    safe_print_metric("final validation accuracy", final_validation_accuracy)
    safe_print_metric("final train loss", final_train_loss)
    safe_print_metric("final validation loss", final_validation_loss)
    safe_print_metric("number of catalog tasks", len(task_catalog))
    safe_print_metric("number of HF proxy datasets", len(hf_proxy_datasets))

    if len(task_catalog) > 0:
        print(f"task catalog: {task_catalog}")
    if len(hf_proxy_datasets) > 0:
        print(f"HF proxy datasets: {hf_proxy_datasets}")

    if final_stream_summary:
        print("best mean pre-adaptation stream accuracy by method:")
        for method_name, value in final_stream_summary:
            print(
                f"  method={method_name} | mean pre-adaptation stream accuracy: {value}"
            )

    if final_stream_loss_summary:
        print("best mean stream loss by method:")
        for method_name, value in final_stream_loss_summary:
            print(f"  method={method_name} | mean stream loss: {value}")

    methods_payload = dataset_payload.get("methods", {})
    for method_name, method_payload in methods_payload.items():
        print(f"Dataset: {dataset_name} | Method: {method_name}")

        method_metrics = method_payload.get("metrics", {})
        method_losses = method_payload.get("losses", {})

        best_learning_rate = best_lr_per_method.get(method_name)
        best_stream_avg_retained_accuracy = method_payload.get(
            "stream_avg_retained_accuracy"
        )

        final_best_pre_adaptation_stream_accuracy = get_last_value(
            method_metrics.get("stream", [])
        )
        final_best_retained_accuracy = get_last_value(
            method_metrics.get("retained", [])
        )
        final_best_stream_loss = get_last_value(method_losses.get("stream", []))

        pre_acc_curve = method_payload.get("pre_acc_curve", [])
        post_acc_curve = method_payload.get("post_acc_curve", [])
        retained_curve = method_payload.get("retained_curve", [])
        adapter_drift = method_payload.get("adapter_drift", [])

        final_pre_curve_accuracy = pre_acc_curve[-1] if len(pre_acc_curve) > 0 else None
        final_post_curve_accuracy = (
            post_acc_curve[-1] if len(post_acc_curve) > 0 else None
        )
        final_retained_curve_accuracy = (
            retained_curve[-1] if len(retained_curve) > 0 else None
        )
        final_adapter_drift = adapter_drift[-1] if len(adapter_drift) > 0 else None

        safe_print_metric("best adaptation learning rate", best_learning_rate)
        safe_print_metric(
            "best mean pre-adaptation stream accuracy",
            final_best_pre_adaptation_stream_accuracy,
        )
        safe_print_metric("best final retained accuracy", final_best_retained_accuracy)
        safe_print_metric("best mean stream loss", final_best_stream_loss)
        safe_print_metric(
            "best stream-averaged retained accuracy", best_stream_avg_retained_accuracy
        )
        safe_print_metric(
            "final pre-adaptation stream accuracy curve value", final_pre_curve_accuracy
        )
        safe_print_metric(
            "final post-adaptation stream accuracy curve value",
            final_post_curve_accuracy,
        )
        safe_print_metric(
            "final retained accuracy curve value", final_retained_curve_accuracy
        )
        safe_print_metric("final adapter drift", final_adapter_drift)

        lr_sweep = method_payload.get("lr_sweep", {})
        if lr_sweep:
            print("best values available from learning-rate sweep entries:")
            best_entry_name = None
            best_entry_score = None
            for lr_key, lr_entry in lr_sweep.items():
                mean_pre = lr_entry.get("mean_acc_pre")
                retained = lr_entry.get("stream_avg_retained_accuracy")
                if mean_pre is None or retained is None:
                    continue
                score = 0.7 * mean_pre + 0.3 * retained
                if best_entry_score is None or score > best_entry_score:
                    best_entry_score = score
                    best_entry_name = lr_key

            if best_entry_name is not None:
                best_entry = lr_sweep[best_entry_name]
                safe_print_metric("best learning-rate sweep key", best_entry_name)
                safe_print_metric(
                    "best learning-rate sweep mean pre-adaptation stream accuracy",
                    best_entry.get("mean_acc_pre"),
                )
                safe_print_metric(
                    "best learning-rate sweep mean post-adaptation stream accuracy",
                    best_entry.get("mean_acc_post"),
                )
                safe_print_metric(
                    "best learning-rate sweep stream-averaged retained accuracy",
                    best_entry.get("stream_avg_retained_accuracy"),
                )
                safe_print_metric(
                    "best learning-rate sweep mean stream loss",
                    best_entry.get("mean_loss"),
                )

    print("-" * 80)
