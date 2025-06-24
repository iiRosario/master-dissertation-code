import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from env import *
import ast

def plot_avg_accuracy_per_cycle_by_lr(base_dir: str, dataset: str, save_path: str, filename: str):
    learning_rates = ["0.0001", "0.002", "0.01"]
    color_map = {"0.0001": "tab:red", "0.002": "tab:blue", "0.01": "tab:green"}
    accuracy_mean_by_lr = {}
    accuracy_std_by_lr = {}

    for lr in learning_rates:
        path = os.path.join(base_dir, dataset, lr, "uncertainty_sampling", "ground_truth")
        results_dirs = glob.glob(os.path.join(path, "results_*"))
        all_runs = []

        for result_dir in results_dirs:
            csv_files = glob.glob(os.path.join(result_dir, "results_*.csv"))
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file)
                    if "accuracy_per_class" not in df.columns:
                        continue

                    run_accuracies = []
                    for _, row in df.iterrows():
                        acc_list_str = row["accuracy_per_class"]
                        acc_list = ast.literal_eval(acc_list_str)
                        acc_values = [float(a) for a in acc_list]
                        avg_cycle_acc = np.mean(acc_values)
                        run_accuracies.append(avg_cycle_acc)

                    all_runs.append(run_accuracies)
                except Exception as e:
                    print(f"[ERRO] ao processar {csv_file}: {e}")

        if not all_runs:
            print(f"[AVISO] Sem dados para learning rate {lr}")
            continue

        min_len = min(len(run) for run in all_runs)
        all_runs = [run[:min_len] for run in all_runs]
        all_runs = np.array(all_runs)

        accuracy_mean_by_lr[lr] = np.mean(all_runs, axis=0)
        accuracy_std_by_lr[lr] = np.std(all_runs, axis=0)

    # Plot
    plt.figure(figsize=(8, 5))
    for lr in learning_rates:
        if lr in accuracy_mean_by_lr:
            mean = accuracy_mean_by_lr[lr]
            std = accuracy_std_by_lr[lr]
            x = np.arange(len(mean))
            plt.plot(x, mean, label=f"LR = {lr}", color=color_map[lr])
            plt.fill_between(x, mean - std, mean + std, color=color_map[lr], alpha=0.3)

    plt.xlabel("Cycle")
    plt.ylabel("Average Accuracy")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    os.makedirs(save_path, exist_ok=True)
    plot_path = os.path.join(save_path, filename)
    plt.savefig(plot_path)
    print(f"[OK] Plot salvo em: {plot_path}")
    plt.close()

plot_avg_accuracy_per_cycle_by_lr(
    base_dir=RESULTS_PATH,
    dataset=DATASET_CIFAR_10,
    save_path="plots/",
    filename="accuracy_by_lr.png"
)