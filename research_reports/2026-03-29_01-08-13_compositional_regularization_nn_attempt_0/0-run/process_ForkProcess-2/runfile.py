import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    print("Loaded datasets:", list(experiment_data.keys()))
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# Print evaluation metrics
for name, data in experiment_data.items():
    try:
        preds = np.array(data.get("predictions", []))
        gt = np.array(data.get("ground_truth", []))
        if preds.size and gt.size and len(preds) == len(gt):
            acc = float((preds == gt).mean())
            cm = np.zeros((2, 2), dtype=int)
            for p, y in zip(preds.astype(int), gt.astype(int)):
                if 0 <= y < 2 and 0 <= p < 2:
                    cm[y, p] += 1
            print(f"{name}: val_accuracy={acc:.4f}")
            print(f"{name}: confusion_matrix=\n{cm}")
        else:
            print(f"{name}: predictions/ground_truth unavailable or mismatched")
    except Exception as e:
        print(f"Error computing metrics for {name}: {e}")

try:
    plt.figure(figsize=(8, 5))
    plotted = False
    for name, data in experiment_data.items():
        tr = data.get("metrics", {}).get("train", [])
        va = data.get("metrics", {}).get("val", [])
        if len(tr):
            tr = np.array(tr)
            plt.plot(tr[:, 0], tr[:, 1], label=f"{name} Train Acc")
            plotted = True
        if len(va):
            va = np.array(va)
            plt.plot(va[:, 0], va[:, 1], label=f"{name} Val Acc")
            plotted = True
    if plotted:
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(
            "Synthetic Compositional Dataset\nTraining and Validation Accuracy Curves"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "synthetic_compositional_accuracy_curves.png")
        )
    plt.close()
except Exception as e:
    print(f"Error creating plot1: {e}")
    plt.close()

try:
    plt.figure(figsize=(8, 5))
    plotted = False
    for name, data in experiment_data.items():
        tr = data.get("losses", {}).get("train", [])
        va = data.get("losses", {}).get("val", [])
        if len(tr):
            tr = np.array(tr)
            plt.plot(tr[:, 0], tr[:, 1], label=f"{name} Train Loss")
            plotted = True
        if len(va):
            va = np.array(va)
            plt.plot(va[:, 0], va[:, 1], label=f"{name} Val Loss")
            plotted = True
    if plotted:
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(
            "Synthetic Compositional Dataset\nTraining and Validation Loss Curves"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "synthetic_compositional_loss_curves.png")
        )
    plt.close()
except Exception as e:
    print(f"Error creating plot2: {e}")
    plt.close()

for name, data in experiment_data.items():
    try:
        preds = np.array(data.get("predictions", []))
        gt = np.array(data.get("ground_truth", []))
        if preds.size and gt.size and len(preds) == len(gt):
            cm = np.zeros((2, 2), dtype=int)
            for p, y in zip(preds.astype(int), gt.astype(int)):
                if 0 <= y < 2 and 0 <= p < 2:
                    cm[y, p] += 1
            plt.figure(figsize=(4, 4))
            plt.imshow(cm, cmap="Blues")
            plt.colorbar()
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(
                        j, i, str(cm[i, j]), ha="center", va="center", color="black"
                    )
            plt.xticks([0, 1], ["Pred 0", "Pred 1"])
            plt.yticks([0, 1], ["True 0", "True 1"])
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            plt.title(
                f"Synthetic Compositional Dataset\n{name} Validation Confusion Matrix"
            )
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    working_dir, f"synthetic_compositional_{name}_confusion_matrix.png"
                )
            )
            plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix for {name}: {e}")
        plt.close()
