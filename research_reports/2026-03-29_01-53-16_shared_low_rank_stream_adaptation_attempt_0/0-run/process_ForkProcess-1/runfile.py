import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

exp = None
try:
    exp = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    print("Loaded experiment_data.npy")
except Exception as e:
    print(f"Error loading experiment data: {e}")

dataset = "synthetic_stream"
data = exp.get(dataset, {}) if isinstance(exp, dict) else {}
per_strategy = data.get("per_strategy", {})

# Print metrics if available
try:
    stream_metrics = data.get("metrics", {}).get("stream", [])
    if stream_metrics:
        m = stream_metrics[-1]
        for k, v in m.items():
            print(f"{k}: {v:.4f}")
except Exception as e:
    print(f"Error printing stream metrics: {e}")

try:
    tr = data.get("metrics", {}).get("train", [])
    va = data.get("metrics", {}).get("val", [])
    if tr or va:
        plt.figure(figsize=(8, 4))
        if tr:
            plt.plot(
                [x["epoch"] for x in tr], [x["acc"] for x in tr], label="Train Accuracy"
            )
        if va:
            plt.plot(
                [x["epoch"] for x in va],
                [x["acc"] for x in va],
                label="Validation Accuracy",
            )
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Synthetic Stream Dataset - Training and Validation Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, f"{dataset}_training_validation_accuracy.png"),
            dpi=150,
        )
        plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

try:
    trl = data.get("losses", {}).get("train", [])
    val = data.get("losses", {}).get("val", [])
    if trl or val:
        plt.figure(figsize=(8, 4))
        if trl:
            plt.plot(
                [x["epoch"] for x in trl], [x["loss"] for x in trl], label="Train Loss"
            )
        if val:
            plt.plot(
                [x["epoch"] for x in val],
                [x["loss"] for x in val],
                label="Validation Loss",
            )
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Synthetic Stream Dataset - Training and Validation Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, f"{dataset}_training_validation_loss.png"),
            dpi=150,
        )
        plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

try:
    if per_strategy:
        plt.figure(figsize=(8, 4))
        for s in ["reset", "carry", "anchored"]:
            if s in per_strategy and "cumulative_acc" in per_strategy[s]:
                y = np.asarray(per_strategy[s]["cumulative_acc"])
                plt.plot(np.arange(1, len(y) + 1), y, label=s)
        plt.xlabel("Stream Batch")
        plt.ylabel("Cumulative Accuracy")
        plt.title(
            "Synthetic Stream Dataset - Cumulative Accuracy by Adaptation Strategy"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, f"{dataset}_cumulative_accuracy_by_strategy.png"),
            dpi=150,
        )
        plt.close()
except Exception as e:
    print(f"Error creating cumulative accuracy plot: {e}")
    plt.close()

try:
    vals, labels = [], []
    for s in ["reset", "carry", "anchored"]:
        if (
            s in per_strategy
            and "metrics" in per_strategy[s]
            and "stream_acc" in per_strategy[s]["metrics"]
        ):
            labels.append(s)
            vals.append(per_strategy[s]["metrics"]["stream_acc"])
    if vals:
        plt.figure(figsize=(6, 4))
        plt.bar(labels, vals)
        plt.ylim(0, 1)
        plt.xlabel("Strategy")
        plt.ylabel("Final Stream Accuracy")
        plt.title("Synthetic Stream Dataset - Final Stream Accuracy by Strategy")
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, f"{dataset}_final_stream_accuracy_bar.png"),
            dpi=150,
        )
        plt.close()
except Exception as e:
    print(f"Error creating final accuracy bar plot: {e}")
    plt.close()

try:
    if per_strategy:
        plt.figure(figsize=(8, 4))
        for s in ["reset", "carry", "anchored"]:
            if s in per_strategy and "losses" in per_strategy[s]:
                y = np.asarray(per_strategy[s]["losses"])
                plt.plot(np.arange(1, len(y) + 1), y, label=s)
        plt.xlabel("Stream Batch")
        plt.ylabel("Batch Adaptation Loss")
        plt.title("Synthetic Stream Dataset - Batch Adaptation Loss by Strategy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                working_dir, f"{dataset}_batch_adaptation_loss_by_strategy.png"
            ),
            dpi=150,
        )
        plt.close()
except Exception as e:
    print(f"Error creating batch loss plot: {e}")
    plt.close()

try:
    gt = np.asarray(data.get("ground_truth", []))
    pred = np.asarray(data.get("predictions", []))
    if gt.size and pred.size and gt.size == pred.size:
        n = min(len(gt), 200)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        axes[0].plot(np.arange(n), gt[:n], marker="o", linestyle="-", markersize=2)
        axes[0].set_title("Synthetic Stream Dataset - Left: Ground Truth")
        axes[0].set_xlabel("Sample Index")
        axes[0].set_ylabel("Class Label")
        axes[1].plot(
            np.arange(n),
            pred[:n],
            marker="o",
            linestyle="-",
            markersize=2,
            color="tab:orange",
        )
        axes[1].set_title(
            "Synthetic Stream Dataset - Right: Generated Samples / Predicted Labels"
        )
        axes[1].set_xlabel("Sample Index")
        fig.suptitle(
            "Synthetic Stream Dataset - Left: Ground Truth, Right: Generated Samples / Predicted Labels"
        )
        fig.tight_layout()
        fig.savefig(
            os.path.join(working_dir, f"{dataset}_ground_truth_vs_predictions.png"),
            dpi=150,
        )
        plt.close(fig)
except Exception as e:
    print(f"Error creating ground truth vs predictions plot: {e}")
    plt.close("all")
