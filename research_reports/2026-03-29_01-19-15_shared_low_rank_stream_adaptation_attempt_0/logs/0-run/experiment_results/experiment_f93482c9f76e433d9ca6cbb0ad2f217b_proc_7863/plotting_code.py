import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

ds_name = "synthetic_stream"
ds = experiment_data.get(ds_name, {})
methods = ds.get("methods", {})


def pairs_to_xy(pairs):
    if not pairs:
        return np.array([]), np.array([])
    x, y = zip(*pairs)
    return np.array(x), np.array(y)


train_mx, train_my = pairs_to_xy(ds.get("metrics", {}).get("train", []))
val_mx, val_my = pairs_to_xy(ds.get("metrics", {}).get("val", []))
train_lx, train_ly = pairs_to_xy(ds.get("losses", {}).get("train", []))
val_lx, val_ly = pairs_to_xy(ds.get("losses", {}).get("val", []))
stream_summary = ds.get("metrics", {}).get("stream", [])
loss_summary = ds.get("losses", {}).get("stream", [])

if len(val_my):
    print(f"Best validation accuracy: {val_my.max():.4f}")
if len(val_ly):
    print(f"Final validation loss: {val_ly[-1]:.4f}")
for m in methods:
    sx, sy = pairs_to_xy(methods[m].get("metrics", {}).get("stream", []))
    lx, ly = pairs_to_xy(methods[m].get("losses", {}).get("stream", []))
    if len(sy):
        print(f"{m} mean stream accuracy: {sy.mean():.4f}")
    if len(ly):
        print(f"{m} mean stream loss: {ly.mean():.4f}")

task_boundaries = []
task_labels = []
try:
    ref_method = next(iter(methods)) if methods else None
    if ref_method:
        names = methods[ref_method].get("task_name", [])
        prev = None
        for i, n in enumerate(names):
            if n != prev:
                task_boundaries.append(i)
                task_labels.append(n)
                prev = n
except Exception:
    pass

try:
    plt.figure(figsize=(7, 4))
    if len(train_mx):
        plt.plot(train_mx, train_my, marker="o", label="Train")
    if len(val_mx):
        plt.plot(val_mx, val_my, marker="o", label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(
        "Synthetic Stream Dataset - Pretraining Accuracy Curves\nTrain vs Validation"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, f"{ds_name}_pretraining_accuracy_curves.png"), dpi=150
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot1: {e}")
    plt.close()

try:
    plt.figure(figsize=(7, 4))
    if len(train_lx):
        plt.plot(train_lx, train_ly, marker="o", label="Train")
    if len(val_lx):
        plt.plot(val_lx, val_ly, marker="o", label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Synthetic Stream Dataset - Pretraining Loss Curves\nTrain vs Validation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, f"{ds_name}_pretraining_loss_curves.png"), dpi=150
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot2: {e}")
    plt.close()

try:
    plt.figure(figsize=(7, 4))
    if stream_summary:
        labels = [k for k, _ in stream_summary]
        vals = [v for _, v in stream_summary]
        plt.bar(labels, vals)
    plt.xlabel("Method")
    plt.ylabel("Mean Online Accuracy")
    plt.title(
        "Synthetic Stream Dataset - Stream-Averaged Online Accuracy\nMethod Comparison"
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, f"{ds_name}_method_stream_accuracy_bar.png"), dpi=150
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot3: {e}")
    plt.close()

try:
    plt.figure(figsize=(8, 4))
    for m, md in methods.items():
        x, y = pairs_to_xy(md.get("metrics", {}).get("stream", []))
        if len(x):
            plt.plot(x, y, label=m)
    for b in task_boundaries[1:]:
        plt.axvline(b, color="gray", linestyle="--", linewidth=0.8)
    plt.xlabel("Stream Position")
    plt.ylabel("Online Accuracy")
    plt.title(
        "Synthetic Stream Dataset - Online Accuracy Over Stream\nTask Segments Marked by Vertical Lines"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, f"{ds_name}_online_accuracy_over_stream.png"), dpi=150
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot4: {e}")
    plt.close()

try:
    plt.figure(figsize=(8, 4))
    for m, md in methods.items():
        x, y = pairs_to_xy(md.get("losses", {}).get("stream", []))
        if len(x):
            plt.plot(x, y, label=m)
    for b in task_boundaries[1:]:
        plt.axvline(b, color="gray", linestyle="--", linewidth=0.8)
    plt.xlabel("Stream Position")
    plt.ylabel("Online Supervised Loss")
    plt.title(
        "Synthetic Stream Dataset - Online Loss Over Stream\nTask Segments Marked by Vertical Lines"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, f"{ds_name}_online_loss_over_stream.png"), dpi=150
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot5: {e}")
    plt.close()

try:
    plt.figure(figsize=(8, 4))
    for m, md in methods.items():
        drift = np.array(md.get("adapter_drift", []))
        if drift.size:
            plt.plot(np.arange(len(drift)), drift, label=m)
    for b in task_boundaries[1:]:
        plt.axvline(b, color="gray", linestyle="--", linewidth=0.8)
    plt.xlabel("Stream Position")
    plt.ylabel("Adapter Drift")
    plt.title(
        "Synthetic Stream Dataset - Adapter Drift Over Stream\nTask Segments Marked by Vertical Lines"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, f"{ds_name}_adapter_drift_over_stream.png"), dpi=150
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot6: {e}")
    plt.close()

try:
    plt.figure(figsize=(7, 4))
    if loss_summary:
        labels = [k for k, _ in loss_summary]
        vals = [v for _, v in loss_summary]
        plt.bar(labels, vals)
    plt.xlabel("Method")
    plt.ylabel("Mean Stream Loss")
    plt.title(
        "Synthetic Stream Dataset - Stream-Averaged Online Loss\nMethod Comparison"
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, f"{ds_name}_method_stream_loss_bar.png"), dpi=150
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot7: {e}")
    plt.close()
