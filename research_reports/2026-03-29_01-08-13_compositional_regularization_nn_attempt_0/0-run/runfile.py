import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

os.makedirs(working_dir, exist_ok=True)

experiment_data_path_list = [
    "experiments/2026-03-29_01-08-13_compositional_regularization_nn_attempt_0/logs/0-run/experiment_results/experiment_61440bcf545a43b9829590fe6157b447_proc_6579/experiment_data.npy",
    "experiments/2026-03-29_01-08-13_compositional_regularization_nn_attempt_0/logs/0-run/experiment_results/experiment_1f16a5b52c304e0a801ce1e49cd4557a_proc_6580/experiment_data.npy",
    "experiments/2026-03-29_01-08-13_compositional_regularization_nn_attempt_0/logs/0-run/experiment_results/experiment_be2b94c260d442dea4d38fd817f0eeef_proc_6581/experiment_data.npy",
]

all_experiment_data = []
try:
    for experiment_data_path in experiment_data_path_list:
        experiment_data = np.load(
            os.path.join(os.getenv("AI_SCIENTIST_ROOT"), experiment_data_path),
            allow_pickle=True,
        ).item()
        all_experiment_data.append(experiment_data)
    print(f"Loaded {len(all_experiment_data)} experiment files")
    for i, experiment_data in enumerate(all_experiment_data):
        print(f"Run {i}: datasets = {list(experiment_data.keys())}")
except Exception as e:
    print(f"Error loading experiment data: {e}")


def compute_sem(values):
    values = np.array(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) <= 1:
        return 0.0
    return float(np.std(values, ddof=1) / np.sqrt(len(values)))


def extract_xy(arr):
    arr = np.array(arr)
    if arr.size == 0:
        return None, None
    if arr.ndim == 1:
        x = np.arange(len(arr))
        y = arr.astype(float)
        return x, y
    if arr.ndim == 2 and arr.shape[1] >= 2:
        x = arr[:, 0].astype(float)
        y = arr[:, 1].astype(float)
        return x, y
    return None, None


def aggregate_curves(run_curves):
    epoch_to_vals = {}
    for x, y in run_curves:
        if x is None or y is None:
            continue
        for xi, yi in zip(x, y):
            if np.isfinite(xi) and np.isfinite(yi):
                epoch_to_vals.setdefault(float(xi), []).append(float(yi))
    if not epoch_to_vals:
        return None, None, None
    epochs = np.array(sorted(epoch_to_vals.keys()))
    means = np.array([np.mean(epoch_to_vals[e]) for e in epochs], dtype=float)
    sems = np.array([compute_sem(epoch_to_vals[e]) for e in epochs], dtype=float)
    return epochs, means, sems


def sanitize_name(name):
    return str(name).replace(" ", "_").replace("/", "_")


# Print per-run and aggregate evaluation metrics
dataset_names = (
    sorted(set().union(*[set(d.keys()) for d in all_experiment_data]))
    if all_experiment_data
    else []
)
for dataset_name in dataset_names:
    accs = []
    cms = []
    for run_idx, experiment_data in enumerate(all_experiment_data):
        try:
            data = experiment_data.get(dataset_name, {})
            preds = np.array(data.get("predictions", []))
            gt = np.array(data.get("ground_truth", []))
            if preds.size and gt.size and len(preds) == len(gt):
                acc = float((preds == gt).mean())
                accs.append(acc)
                print(f"{dataset_name} run_{run_idx}: val_accuracy={acc:.4f}")

                unique_vals = np.unique(np.concatenate([gt.flatten(), preds.flatten()]))
                if np.all(np.isin(unique_vals, [0, 1])):
                    cm = np.zeros((2, 2), dtype=int)
                    for p, y in zip(preds.astype(int), gt.astype(int)):
                        if 0 <= y < 2 and 0 <= p < 2:
                            cm[y, p] += 1
                    cms.append(cm)
                    print(f"{dataset_name} run_{run_idx}: confusion_matrix=\n{cm}")
            else:
                print(
                    f"{dataset_name} run_{run_idx}: predictions/ground_truth unavailable or mismatched"
                )
        except Exception as e:
            print(f"Error computing metrics for {dataset_name} run_{run_idx}: {e}")
    if accs:
        print(
            f"{dataset_name}: mean_val_accuracy={np.mean(accs):.4f}, "
            f"sem_val_accuracy={compute_sem(accs):.4f}, n_runs={len(accs)}"
        )
    if cms:
        cm_sum = np.sum(cms, axis=0)
        print(f"{dataset_name}: aggregated_confusion_matrix=\n{cm_sum}")

# Aggregated accuracy curves
try:
    for dataset_name in dataset_names:
        plt.figure(figsize=(9, 5))
        plotted = False

        train_curves = []
        val_curves = []
        for experiment_data in all_experiment_data:
            data = experiment_data.get(dataset_name, {})
            tr = data.get("metrics", {}).get("train", [])
            va = data.get("metrics", {}).get("val", [])
            train_curves.append(extract_xy(tr))
            val_curves.append(extract_xy(va))

        tr_epochs, tr_mean, tr_sem = aggregate_curves(train_curves)
        va_epochs, va_mean, va_sem = aggregate_curves(val_curves)

        if tr_epochs is not None:
            plt.plot(tr_epochs, tr_mean, label="Train Mean Accuracy")
            plt.fill_between(
                tr_epochs,
                tr_mean - tr_sem,
                tr_mean + tr_sem,
                alpha=0.2,
                label="Train Std. Error",
            )
            plotted = True
        if va_epochs is not None:
            plt.plot(va_epochs, va_mean, label="Val Mean Accuracy")
            plt.fill_between(
                va_epochs,
                va_mean - va_sem,
                va_mean + va_sem,
                alpha=0.2,
                label="Val Std. Error",
            )
            plotted = True

        if plotted:
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title(
                f"{dataset_name}\nTraining and Validation Accuracy Curves (Mean ± Std. Error)"
            )
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    working_dir,
                    f"{sanitize_name(dataset_name)}_aggregated_accuracy_curves.png",
                )
            )
        plt.close()
except Exception as e:
    print(f"Error creating plot1: {e}")
    plt.close()

# Aggregated loss curves
try:
    for dataset_name in dataset_names:
        plt.figure(figsize=(9, 5))
        plotted = False

        train_curves = []
        val_curves = []
        for experiment_data in all_experiment_data:
            data = experiment_data.get(dataset_name, {})
            tr = data.get("losses", {}).get("train", [])
            va = data.get("losses", {}).get("val", [])
            train_curves.append(extract_xy(tr))
            val_curves.append(extract_xy(va))

        tr_epochs, tr_mean, tr_sem = aggregate_curves(train_curves)
        va_epochs, va_mean, va_sem = aggregate_curves(val_curves)

        if tr_epochs is not None:
            plt.plot(tr_epochs, tr_mean, label="Train Mean Loss")
            plt.fill_between(
                tr_epochs,
                tr_mean - tr_sem,
                tr_mean + tr_sem,
                alpha=0.2,
                label="Train Std. Error",
            )
            plotted = True
        if va_epochs is not None:
            plt.plot(va_epochs, va_mean, label="Val Mean Loss")
            plt.fill_between(
                va_epochs,
                va_mean - va_sem,
                va_mean + va_sem,
                alpha=0.2,
                label="Val Std. Error",
            )
            plotted = True

        if plotted:
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(
                f"{dataset_name}\nTraining and Validation Loss Curves (Mean ± Std. Error)"
            )
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    working_dir,
                    f"{sanitize_name(dataset_name)}_aggregated_loss_curves.png",
                )
            )
        plt.close()
except Exception as e:
    print(f"Error creating plot2: {e}")
    plt.close()

# Per-run validation accuracy with aggregate mean ± std. error
try:
    for dataset_name in dataset_names:
        run_labels = []
        accs = []
        for run_idx, experiment_data in enumerate(all_experiment_data):
            data = experiment_data.get(dataset_name, {})
            preds = np.array(data.get("predictions", []))
            gt = np.array(data.get("ground_truth", []))
            if preds.size and gt.size and len(preds) == len(gt):
                run_labels.append(f"Run {run_idx}")
                accs.append(float((preds == gt).mean()))

        if accs:
            plt.figure(figsize=(8, 5))
            x = np.arange(len(accs))
            mean_acc = np.mean(accs)
            sem_acc = compute_sem(accs)

            plt.bar(x, accs, label="Per-run Val Accuracy")
            plt.axhline(
                mean_acc, color="red", linestyle="--", label="Mean Val Accuracy"
            )
            plt.fill_between(
                [-0.5, len(accs) - 0.5],
                [mean_acc - sem_acc, mean_acc - sem_acc],
                [mean_acc + sem_acc, mean_acc + sem_acc],
                color="red",
                alpha=0.2,
                label="Std. Error",
            )
            plt.xticks(x, run_labels)
            plt.ylabel("Accuracy")
            plt.xlabel("Run")
            plt.title(
                f"{dataset_name}\nValidation Accuracy Across Runs (Mean ± Std. Error)"
            )
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    working_dir,
                    f"{sanitize_name(dataset_name)}_val_accuracy_across_runs.png",
                )
            )
            plt.close()
except Exception as e:
    print(f"Error creating plot3: {e}")
    plt.close()

# Aggregated confusion matrices
try:
    for dataset_name in dataset_names:
        cms = []
        for experiment_data in all_experiment_data:
            data = experiment_data.get(dataset_name, {})
            preds = np.array(data.get("predictions", []))
            gt = np.array(data.get("ground_truth", []))
            if preds.size and gt.size and len(preds) == len(gt):
                unique_vals = np.unique(np.concatenate([gt.flatten(), preds.flatten()]))
                if np.all(np.isin(unique_vals, [0, 1])):
                    cm = np.zeros((2, 2), dtype=int)
                    for p, y in zip(preds.astype(int), gt.astype(int)):
                        if 0 <= y < 2 and 0 <= p < 2:
                            cm[y, p] += 1
                    cms.append(cm)

        if cms:
            cm_sum = np.sum(cms, axis=0)
            plt.figure(figsize=(5, 4))
            plt.imshow(cm_sum, cmap="Blues")
            plt.colorbar()
            for i in range(cm_sum.shape[0]):
                for j in range(cm_sum.shape[1]):
                    plt.text(
                        j, i, str(cm_sum[i, j]), ha="center", va="center", color="black"
                    )
            plt.xticks([0, 1], ["Pred 0", "Pred 1"])
            plt.yticks([0, 1], ["True 0", "True 1"])
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            plt.title(
                f"{dataset_name}\nAggregated Validation Confusion Matrix Across Runs"
            )
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    working_dir,
                    f"{sanitize_name(dataset_name)}_aggregated_confusion_matrix.png",
                )
            )
            plt.close()
except Exception as e:
    print(f"Error creating plot4: {e}")
    plt.close()
