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
    raise SystemExit

ablation_name = list(experiment_data.keys())[0]
ablation_data = experiment_data[ablation_name]
dataset_names = list(ablation_data.keys())


def rank_num(rk):
    try:
        return int(str(rk).split("_")[-1])
    except:
        return 0


mode_names = ["frozen", "always", "gated", "reset"]

for ds in dataset_names:
    try:
        rs = ablation_data[ds].get("rank_summaries", {})
        best_srus = None
        best_acc = None
        for rk, modes in rs.items():
            for mode, vals in modes.items():
                tup_s = (vals.get("srus", -1e9), rk, mode)
                tup_a = (vals.get("acc", -1e9), rk, mode)
                if best_srus is None or tup_s[0] > best_srus[0]:
                    best_srus = tup_s
                if best_acc is None or tup_a[0] > best_acc[0]:
                    best_acc = tup_a
        if best_srus and best_acc:
            print(
                f"{ds}: best_srus={best_srus[0]:.4f} ({best_srus[1]}, {best_srus[2]}), best_acc={best_acc[0]:.4f} ({best_acc[1]}, {best_acc[2]})"
            )
    except Exception as e:
        print(f"Error summarizing {ds}: {e}")

try:
    plt.figure(figsize=(8, 4))
    any_plot = False
    ref_ds = dataset_names[0]
    rank_keys = sorted(
        ablation_data[ref_ds].get("rank_summaries", {}).keys(), key=rank_num
    )
    val_list = ablation_data[ref_ds].get("metrics", {}).get("val", [])
    for i, rk in enumerate(rank_keys):
        if i < len(val_list):
            arr = np.array(val_list[i], dtype=np.float32)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                plt.plot(arr[:, 0], arr[:, 1], marker="o", label=f"{rk}")
                any_plot = True
    if any_plot:
        plt.xlabel("Epoch")
        plt.ylabel("Validation Accuracy")
        plt.title(
            f"Dataset: source training\nSubtitle: Validation accuracy by epoch across LoRA ranks"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                working_dir,
                f"{ablation_name}_source_training_validation_accuracy_by_rank.png",
            )
        )
    plt.close()
except Exception as e:
    print(f"Error creating source validation accuracy plot: {e}")
    plt.close()

try:
    plt.figure(figsize=(8, 4))
    any_plot = False
    ref_ds = dataset_names[0]
    rank_keys = sorted(
        ablation_data[ref_ds].get("rank_summaries", {}).keys(), key=rank_num
    )
    val_losses = ablation_data[ref_ds].get("losses", {}).get("val", [])
    for i, rk in enumerate(rank_keys):
        if i < len(val_losses):
            arr = np.array(val_losses[i], dtype=np.float32)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                plt.plot(arr[:, 0], arr[:, 1], marker="o", label=f"{rk}")
                any_plot = True
    if any_plot:
        plt.xlabel("Epoch")
        plt.ylabel("Validation Loss")
        plt.title(
            f"Dataset: source training\nSubtitle: Validation loss by epoch across LoRA ranks"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                working_dir,
                f"{ablation_name}_source_training_validation_loss_by_rank.png",
            )
        )
    plt.close()
except Exception as e:
    print(f"Error creating source validation loss plot: {e}")
    plt.close()

for ds in dataset_names:
    try:
        plt.figure(figsize=(8, 4))
        rs = ablation_data[ds].get("rank_summaries", {})
        rank_keys = sorted(rs.keys(), key=rank_num)
        ranks = [rank_num(rk) for rk in rank_keys]
        any_plot = False
        for mode in mode_names:
            ys = [rs[rk][mode]["srus"] for rk in rank_keys if mode in rs[rk]]
            if len(ys) == len(ranks) and len(ys) > 0:
                plt.plot(ranks, ys, marker="o", label=mode)
                any_plot = True
        if any_plot:
            plt.xlabel("LoRA Rank")
            plt.ylabel("SRUS")
            plt.title(
                f"Dataset: {ds}\nSubtitle: SRUS versus LoRA rank for adaptation modes"
            )
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(working_dir, f"{ds}_srus_vs_rank_plot.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating SRUS plot for {ds}: {e}")
        plt.close()

    try:
        plt.figure(figsize=(8, 4))
        rs = ablation_data[ds].get("rank_summaries", {})
        rank_keys = sorted(rs.keys(), key=rank_num)
        ranks = [rank_num(rk) for rk in rank_keys]
        any_plot = False
        for mode in mode_names:
            ys = [rs[rk][mode]["acc"] for rk in rank_keys if mode in rs[rk]]
            if len(ys) == len(ranks) and len(ys) > 0:
                plt.plot(ranks, ys, marker="o", label=mode)
                any_plot = True
        if any_plot:
            plt.xlabel("LoRA Rank")
            plt.ylabel("Accuracy")
            plt.title(
                f"Dataset: {ds}\nSubtitle: Accuracy versus LoRA rank for adaptation modes"
            )
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(working_dir, f"{ds}_accuracy_vs_rank_plot.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot for {ds}: {e}")
        plt.close()

    try:
        plt.figure(figsize=(8, 4))
        rt = ablation_data[ds].get("rank_traces", {})
        rank_keys = sorted(rt.keys(), key=rank_num)
        any_plot = False
        for rk in rank_keys:
            gated = rt[rk].get("gated", {})
            cumacc = np.array(gated.get("cumacc", []), dtype=np.float32)
            if cumacc.size > 0:
                plt.plot(np.arange(len(cumacc)), cumacc, label=f"{rk}")
                any_plot = True
        if any_plot:
            plt.xlabel("Stream Step")
            plt.ylabel("Cumulative Accuracy")
            plt.title(
                f"Dataset: {ds}\nSubtitle: Gated-mode cumulative accuracy across LoRA ranks"
            )
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    working_dir, f"{ds}_gated_cumulative_accuracy_by_rank_plot.png"
                )
            )
        plt.close()
    except Exception as e:
        print(f"Error creating gated cumulative accuracy plot for {ds}: {e}")
        plt.close()
