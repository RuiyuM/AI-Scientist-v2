import os
import numpy as np

working_dir = os.path.join(os.getcwd(), "working")
experiment_data_path = os.path.join(working_dir, "experiment_data.npy")

experiment_data = np.load(experiment_data_path, allow_pickle=True).item()


def is_trial_entry(value):
    return isinstance(value, dict) and (
        "best_val_acc" in value
        or "best_val_loss" in value
        or "methods" in value
        or ("metrics" in value and "losses" in value)
    )


def format_value(value):
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6f}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)


def print_dataset_level_selection(dataset_name, dataset_blob):
    selection = dataset_blob.get("selection", {})
    if not selection:
        return
    print(f"Dataset: {dataset_name}")
    if "best_trial_key" in selection:
        print(f"selected best trial: {selection['best_trial_key']}")
    if "best_epochs" in selection:
        print(f"selected best pretraining epochs: {selection['best_epochs']}")
    if "best_pretrain_lr" in selection:
        print(
            f"selected best pretraining learning rate: {format_value(selection['best_pretrain_lr'])}"
        )
    if "best_val_acc" in selection:
        print(
            f"selected best validation accuracy: {format_value(selection['best_val_acc'])}"
        )
    if "best_val_loss" in selection:
        print(
            f"selected best validation loss: {format_value(selection['best_val_loss'])}"
        )
    if "criterion" in selection:
        print(f"selection criterion: {selection['criterion']}")
    print()


def print_dataset_level_summary(dataset_name, dataset_blob):
    metrics = dataset_blob.get("metrics", {})
    losses = dataset_blob.get("losses", {})

    print(f"Dataset: {dataset_name}")

    train_metric_summary = metrics.get("train", [])
    if train_metric_summary:
        final_trial_key, final_best_val_acc = train_metric_summary[-1]
        print(
            f"final summary trial key for best validation accuracy listing: {final_trial_key}"
        )
        print(
            f"final summary best validation accuracy: {format_value(final_best_val_acc)}"
        )

    validation_loss_summary = losses.get("val", [])
    if validation_loss_summary:
        final_trial_key, final_best_val_loss = validation_loss_summary[-1]
        print(f"final summary trial key for validation loss listing: {final_trial_key}")
        print(f"final summary validation loss: {format_value(final_best_val_loss)}")

    stream_summary = metrics.get("stream", [])
    if stream_summary:
        final_trial_key, final_stream_metrics = stream_summary[-1]
        print(f"final summary trial key for stream accuracy listing: {final_trial_key}")
        if isinstance(final_stream_metrics, dict):
            for method_name, method_value in final_stream_metrics.items():
                print(
                    f"final stream-averaged online accuracy ({method_name}): {format_value(method_value)}"
                )

    retained_summary = metrics.get("retained", [])
    if retained_summary:
        final_trial_key, final_retained_metrics = retained_summary[-1]
        print(
            f"final summary trial key for retained accuracy listing: {final_trial_key}"
        )
        if isinstance(final_retained_metrics, dict):
            for method_name, method_value in final_retained_metrics.items():
                print(
                    f"final stream-averaged retained accuracy SARA ({method_name}): {format_value(method_value)}"
                )

    print()


def print_trial_metrics(dataset_name, trial_key, trial_blob):
    print(f"Dataset: {dataset_name}")
    print(f"trial name: {trial_key}")

    if "best_val_acc" in trial_blob:
        print(f"best validation accuracy: {format_value(trial_blob['best_val_acc'])}")
    if "best_val_loss" in trial_blob:
        print(f"best validation loss: {format_value(trial_blob['best_val_loss'])}")

    metrics = trial_blob.get("metrics", {})
    losses = trial_blob.get("losses", {})

    train_acc_history = metrics.get("train", [])
    if train_acc_history:
        final_epoch, final_train_acc = train_acc_history[-1]
        print(
            f"final pretraining epoch for train accuracy: {format_value(final_epoch)}"
        )
        print(f"final train accuracy: {format_value(final_train_acc)}")

    val_acc_history = metrics.get("val", [])
    if val_acc_history:
        final_epoch, final_val_acc = val_acc_history[-1]
        print(
            f"final pretraining epoch for validation accuracy: {format_value(final_epoch)}"
        )
        print(f"final validation accuracy: {format_value(final_val_acc)}")

    train_loss_history = losses.get("train", [])
    if train_loss_history:
        final_epoch, final_train_loss = train_loss_history[-1]
        print(f"final pretraining epoch for train loss: {format_value(final_epoch)}")
        print(f"final train loss: {format_value(final_train_loss)}")

    val_loss_history = losses.get("val", [])
    if val_loss_history:
        final_epoch, final_val_loss = val_loss_history[-1]
        print(
            f"final pretraining epoch for validation loss: {format_value(final_epoch)}"
        )
        print(f"final validation loss from history: {format_value(final_val_loss)}")

    stream_metrics = metrics.get("stream", {})
    if isinstance(stream_metrics, dict):
        for method_name, method_value in stream_metrics.items():
            print(
                f"best stream-averaged online accuracy ({method_name}): {format_value(method_value)}"
            )

    stream_losses = losses.get("stream", {})
    if isinstance(stream_losses, dict):
        for method_name, method_value in stream_losses.items():
            print(
                f"best stream-averaged online loss ({method_name}): {format_value(method_value)}"
            )

    retained_metrics = metrics.get("retained", {})
    if isinstance(retained_metrics, dict):
        for method_name, method_value in retained_metrics.items():
            print(
                f"best stream-averaged retained accuracy SARA ({method_name}): {format_value(method_value)}"
            )

    methods_blob = trial_blob.get("methods", {})
    if isinstance(methods_blob, dict):
        for method_name, method_data in methods_blob.items():
            method_metrics = method_data.get("metrics", {})
            method_losses = method_data.get("losses", {})

            stream_curve = method_metrics.get("stream", [])
            if stream_curve:
                final_step, final_stream_acc = stream_curve[-1]
                print(
                    f"final stream batch index ({method_name}) for online accuracy: {format_value(final_step)}"
                )
                print(
                    f"final online stream batch accuracy ({method_name}): {format_value(final_stream_acc)}"
                )

            retained_curve = method_metrics.get("retained", [])
            if retained_curve:
                final_step, final_retained_acc = retained_curve[-1]
                print(
                    f"final stream batch index ({method_name}) for retained accuracy: {format_value(final_step)}"
                )
                print(
                    f"final retained accuracy over seen tasks ({method_name}): {format_value(final_retained_acc)}"
                )

            stream_loss_curve = method_losses.get("stream", [])
            if stream_loss_curve:
                final_step, final_stream_loss = stream_loss_curve[-1]
                print(
                    f"final stream batch index ({method_name}) for online loss: {format_value(final_step)}"
                )
                print(
                    f"final online stream batch loss ({method_name}): {format_value(final_stream_loss)}"
                )

            drift_curve = method_data.get("adapter_drift", [])
            if len(drift_curve) > 0:
                print(
                    f"final adapter drift ({method_name}): {format_value(drift_curve[-1])}"
                )

            gate_signal = method_data.get("gate_signal", [])
            if gate_signal:
                final_gate = gate_signal[-1]
                if len(final_gate) >= 4:
                    step_idx, entropy_before, margin_before, update_flag = final_gate[
                        :4
                    ]
                    print(
                        f"final gate evaluation stream batch index ({method_name}): {format_value(step_idx)}"
                    )
                    print(
                        f"final gate entropy before update ({method_name}): {format_value(entropy_before)}"
                    )
                    print(
                        f"final gate confidence margin before update ({method_name}): {format_value(margin_before)}"
                    )
                    print(
                        f"final gate update decision ({method_name}): {format_value(update_flag)}"
                    )

    print()


for dataset_name, dataset_blob in experiment_data.items():
    print_dataset_level_selection(dataset_name, dataset_blob)
    print_dataset_level_summary(dataset_name, dataset_blob)

    trial_keys = []
    for key, value in dataset_blob.items():
        if key in {
            "metrics",
            "losses",
            "predictions",
            "ground_truth",
            "selection",
            "plots",
            "config",
        }:
            continue
        if is_trial_entry(value):
            trial_keys.append(key)

    trial_keys = sorted(trial_keys)
    for trial_key in trial_keys:
        print_trial_metrics(dataset_name, trial_key, dataset_blob[trial_key])
