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

dataset_name = next(iter(experiment_data.keys()), None)
ds = experiment_data.get(dataset_name, {}) if dataset_name else {}
losses = ds.get("losses", {})
metrics = ds.get("metrics", {})
stream = ds.get("stream", {})
preds = ds.get("predictions", {})
gts = np.array(ds.get("ground_truth", []))


def to_epoch_value(arr):
    out = []
    for x in arr:
        try:
            out.append((float(x[0]), float(x[1])))
        except Exception:
            pass
    return np.array(out) if out else np.empty((0, 2))


train_loss = to_epoch_value(losses.get("train", []))
val_loss = to_epoch_value(losses.get("val", []))
train_acc = to_epoch_value(metrics.get("train", []))
val_acc = to_epoch_value(metrics.get("val", []))
test_metrics = metrics.get("test", [])

print(f"dataset={dataset_name}")
if len(train_loss):
    print(f"final_train_loss={train_loss[-1,1]:.4f}")
if len(val_loss):
    print(f"final_val_loss={val_loss[-1,1]:.4f}")
if len(train_acc):
    print(f"final_train_acc={train_acc[-1,1]:.4f}")
if len(val_acc):
    print(f"final_val_acc={val_acc[-1,1]:.4f}")
for m in test_metrics:
    if isinstance(m, dict):
        print(
            f"{m.get('name','unknown')}: "
            f"acc={m.get('acc', np.nan):.4f}, "
            f"stability_adjusted_acc={m.get('stability_adjusted_acc', np.nan):.4f}, "
            f"adapt_freq={m.get('adapt_freq', np.nan):.4f}, "
            f"adapt_loss_mean={m.get('adapt_loss_mean', np.nan):.4f}"
        )

try:
    if len(train_loss) and len(val_loss):
        plt.figure(figsize=(8, 4))
        plt.plot(train_loss[:, 0], train_loss[:, 1], label="Train loss")
        plt.plot(val_loss[:, 0], val_loss[:, 1], label="Validation loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{dataset_name} dataset: Training curves\nTrain vs Validation Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, f"{dataset_name}_training_validation_loss.png"),
            dpi=160,
        )
    plt.close()
except Exception as e:
    print(f"Error creating plot1: {e}")
    plt.close()

try:
    if len(train_acc) and len(val_acc):
        plt.figure(figsize=(8, 4))
        plt.plot(train_acc[:, 0], train_acc[:, 1], label="Train accuracy")
        plt.plot(val_acc[:, 0], val_acc[:, 1], label="Validation accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(
            f"{dataset_name} dataset: Training curves\nTrain vs Validation Accuracy"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                working_dir, f"{dataset_name}_training_validation_accuracy.png"
            ),
            dpi=160,
        )
    plt.close()
except Exception as e:
    print(f"Error creating plot2: {e}")
    plt.close()

try:
    fr = np.array(stream.get("frozen_rolling_acc", []))
    en = np.array(stream.get("entropy_rolling_acc", []))
    co = np.array(stream.get("consistency_rolling_acc", []))
    eu = np.array(stream.get("entropy_updates", []))
    cu = np.array(stream.get("consistency_updates", []))
    if len(fr) or len(en) or len(co):
        plt.figure(figsize=(10, 5))
        if len(fr):
            plt.plot(np.arange(len(fr)), fr, label="Frozen", linewidth=2)
        if len(en):
            plt.plot(np.arange(len(en)), en, label="Entropy TTA", linewidth=2)
        if len(co):
            plt.plot(np.arange(len(co)), co, label="Consistency-Gated TTA", linewidth=2)
        if len(en) and len(eu):
            idx = np.where(eu == 1)[0]
            if len(idx):
                plt.scatter(idx, en[idx], s=10, alpha=0.4, label="Entropy updates")
        if len(co) and len(cu):
            idx = np.where(cu == 1)[0]
            if len(idx):
                plt.scatter(idx, co[idx], s=12, alpha=0.5, label="Consistency updates")
        plt.xlabel("Stream step")
        plt.ylabel("Rolling accuracy")
        plt.title(
            f"{dataset_name} dataset: Stream performance\nRolling Accuracy with Adaptation Events"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                working_dir, f"{dataset_name}_stream_rolling_accuracy_updates.png"
            ),
            dpi=160,
        )
    plt.close()
except Exception as e:
    print(f"Error creating plot3: {e}")
    plt.close()

try:
    n_show = (
        min(60, len(gts), *(len(np.array(v)) for v in preds.values()))
        if len(preds)
        else min(60, len(gts))
    )
    if n_show > 0:
        plt.figure(figsize=(10, 4))
        plt.plot(np.arange(n_show), gts[:n_show], label="Ground Truth", marker="o")
        for name in ["frozen", "entropy", "consistency"]:
            arr = np.array(preds.get(name, []))
            if len(arr) >= n_show:
                plt.plot(
                    np.arange(n_show),
                    arr[:n_show],
                    label=f"{name.title()} Pred",
                    marker="x" if name == "entropy" else None,
                )
        plt.xlabel("Sample index")
        plt.ylabel("Class")
        plt.title(
            f"{dataset_name} dataset: Prediction comparison\nLeft: Ground Truth, Right: Generated Samples over Stream Prefix"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                working_dir, f"{dataset_name}_prediction_vs_ground_truth_prefix.png"
            ),
            dpi=160,
        )
    plt.close()
except Exception as e:
    print(f"Error creating plot4: {e}")
    plt.close()

try:
    fc = np.array(stream.get("frozen_confidences", []))
    ec = np.array(stream.get("entropy_confidences", []))
    cc = np.array(stream.get("consistency_confidences", []))
    if len(fc) or len(ec) or len(cc):
        plt.figure(figsize=(10, 4))
        if len(fc):
            plt.plot(np.arange(len(fc)), fc, label="Frozen confidence")
        if len(ec):
            plt.plot(np.arange(len(ec)), ec, label="Entropy TTA confidence")
        if len(cc):
            plt.plot(np.arange(len(cc)), cc, label="Consistency-Gated TTA confidence")
        plt.xlabel("Stream step")
        plt.ylabel("Confidence")
        plt.title(
            f"{dataset_name} dataset: Stream confidence\nPrediction Confidence Across Stream Steps"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, f"{dataset_name}_stream_confidence_curves.png"),
            dpi=160,
        )
    plt.close()
except Exception as e:
    print(f"Error creating plot5: {e}")
    plt.close()

try:
    if test_metrics:
        names = [m.get("name", f"model_{i}") for i, m in enumerate(test_metrics)]
        accs = [m.get("acc", np.nan) for m in test_metrics]
        stabs = [m.get("stability_adjusted_acc", np.nan) for m in test_metrics]
        x = np.arange(len(names))
        plt.figure(figsize=(8, 4))
        w = 0.35
        plt.bar(x - w / 2, accs, width=w, label="Accuracy")
        plt.bar(x + w / 2, stabs, width=w, label="Stability-adjusted accuracy")
        plt.xticks(x, names)
        plt.ylabel("Metric value")
        plt.title(
            f"{dataset_name} dataset: Final test metrics\nComparison of Stream Evaluation Modes"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, f"{dataset_name}_final_test_metrics_bar.png"),
            dpi=160,
        )
    plt.close()
except Exception as e:
    print(f"Error creating plot6: {e}")
    plt.close()
