import os
import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from env import *

def plot_accuracy_with_std(base_path: str,
                                  results_prefix: str = "results_",
                                  confusion_column: str = "confusion_matrix",
                                  save_path: str = None):
    all_accuracies = {}

    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)
        if os.path.isdir(folder_path) and folder.startswith(results_prefix):
            for file in os.listdir(folder_path):
                if file.endswith(".csv"):
                    file_path = os.path.join(folder_path, file)

                    try:
                        df = pd.read_csv(file_path, sep=None, engine='python')
                        df.columns = [col.strip().replace('\ufeff', '') for col in df.columns]

                        for _, row in df.iterrows():
                            cycle = int(row['cycle'])

                            # Extrai a matriz de confusão
                            confusion_str = row[confusion_column]
                            confusion = np.array(ast.literal_eval(confusion_str))

                            correct = np.trace(confusion)
                            total = np.sum(confusion)
                            accuracy = correct / total if total > 0 else 0.0

                            if cycle not in all_accuracies:
                                all_accuracies[cycle] = []
                            all_accuracies[cycle].append(accuracy)

                    except Exception as e:
                        print(f"Erro no arquivo {file_path}, linha {row.name}: {e}")

    # Organiza os dados por ciclo
    if not all_accuracies:
        print("Nenhuma matriz de confusão encontrada.")
        return

    cycles = sorted(all_accuracies.keys())
    mean_accuracies = [np.mean(all_accuracies[c]) for c in cycles]
    std_accuracies = [np.std(all_accuracies[c]) for c in cycles]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(cycles, mean_accuracies, linestyle='-', linewidth=2, label='Avg. Accuracy')
    plt.fill_between(
        cycles,
        np.array(mean_accuracies) - np.array(std_accuracies),
        np.array(mean_accuracies) + np.array(std_accuracies),
        color='blue',
        alpha=0.2,
        label='± std'
    )
    plt.xlabel('Cycle')
    plt.ylabel('Accuracy (Avg. ± std)')
    plt.title('Avg. Accuracy per Cycle')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Gráfico salvo em {save_path}")
        plt.close()
    else:
        plt.show()

def plot_precision_with_std(base_path: str,
                             results_prefix: str = "results_",
                             confusion_column: str = "confusion_matrix",
                             save_path: str = None):
    all_precisions = {}

    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)
        if os.path.isdir(folder_path) and folder.startswith(results_prefix):
            for file in os.listdir(folder_path):
                if file.endswith(".csv"):
                    file_path = os.path.join(folder_path, file)

                    try:
                        df = pd.read_csv(file_path, sep=None, engine='python')
                        df.columns = [col.strip().replace('\ufeff', '') for col in df.columns]

                        for _, row in df.iterrows():
                            cycle = int(row['cycle'])

                            # Extrai e interpreta a matriz de confusão
                            confusion_str = row[confusion_column]
                            confusion = np.array(ast.literal_eval(confusion_str))

                            # Calcula precision por classe
                            precisions = []
                            for i in range(confusion.shape[0]):
                                tp = confusion[i, i]
                                fp = np.sum(confusion[:, i]) - tp
                                denom = tp + fp
                                if denom > 0:
                                    precisions.append(tp / denom)

                            avg_precision = np.mean(precisions) if precisions else 0.0

                            if cycle not in all_precisions:
                                all_precisions[cycle] = []
                            all_precisions[cycle].append(avg_precision)

                    except Exception as e:
                        print(f"Erro no arquivo {file_path}, linha {row.name}: {e}")

    if not all_precisions:
        print("Nenhuma matriz de confusão encontrada.")
        return

    cycles = sorted(all_precisions.keys())
    mean_precisions = [np.mean(all_precisions[c]) for c in cycles]
    std_precisions = [np.std(all_precisions[c]) for c in cycles]

    plt.figure(figsize=(10, 6))
    plt.plot(cycles, mean_precisions, linestyle='-', linewidth=2, color='darkgreen', label='Avg. Precision')
    plt.fill_between(
        cycles,
        np.array(mean_precisions) - np.array(std_precisions),
        np.array(mean_precisions) + np.array(std_precisions),
        color='green',
        alpha=0.2,
        label='± std'
    )
    plt.xlabel('Cycle')
    plt.ylabel('Precision (Avg. ± std)')
    plt.title('Avg. Precision per Cycle (from confusion matrix)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Gráfico salvo em {save_path}")
        plt.close()
    else:
        plt.show()

def plot_f1_score_with_std(base_path: str,
                            results_prefix: str = "results_",
                            confusion_column: str = "confusion_matrix",
                            save_path: str = None):
    all_f1_scores = {}

    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)
        if os.path.isdir(folder_path) and folder.startswith(results_prefix):
            for file in os.listdir(folder_path):
                if file.endswith(".csv"):
                    file_path = os.path.join(folder_path, file)

                    try:
                        df = pd.read_csv(file_path, sep=None, engine='python')
                        df.columns = [col.strip().replace('\ufeff', '') for col in df.columns]

                        for _, row in df.iterrows():
                            cycle = int(row['cycle'])

                            # Extrai e interpreta a matriz de confusão
                            confusion_str = row[confusion_column]
                            confusion = np.array(ast.literal_eval(confusion_str))

                            f1_scores = []
                            for i in range(confusion.shape[0]):
                                tp = confusion[i, i]
                                fp = np.sum(confusion[:, i]) - tp
                                fn = np.sum(confusion[i, :]) - tp

                                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                                recall = tp / (tp + fn) if (tp + fn) > 0 else 0

                                denom = precision + recall
                                if denom > 0:
                                    f1 = 2 * precision * recall / denom
                                    f1_scores.append(f1)

                            avg_f1 = np.mean(f1_scores) if f1_scores else 0.0

                            if cycle not in all_f1_scores:
                                all_f1_scores[cycle] = []
                            all_f1_scores[cycle].append(avg_f1)

                    except Exception as e:
                        print(f"Erro no arquivo {file_path}, linha {row.name}: {e}")

    if not all_f1_scores:
        print("Nenhuma matriz de confusão encontrada.")
        return

    cycles = sorted(all_f1_scores.keys())
    mean_f1s = [np.mean(all_f1_scores[c]) for c in cycles]
    std_f1s = [np.std(all_f1_scores[c]) for c in cycles]

    plt.figure(figsize=(10, 6))
    plt.plot(cycles, mean_f1s, linestyle='-', linewidth=2, color='purple', label='Avg. F1-score')
    plt.fill_between(
        cycles,
        np.array(mean_f1s) - np.array(std_f1s),
        np.array(mean_f1s) + np.array(std_f1s),
        color='purple',
        alpha=0.2,
        label='± std'
    )
    plt.xlabel('Cycle')
    plt.ylabel('F1-score (Avg. ± std)')
    plt.title('Avg. F1-score per Cycle (from confusion matrix)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Gráfico salvo em {save_path}")
        plt.close()
    else:
        plt.show()


# =================================================================================

def plot_accuracy_comparative_reputation_based_E(dataset: str,
                                                 query_strategy: str = "uncertainty_sampling",
                                                 confusion_column: str = "confusion_matrix",
                                                 results_prefix: str = "results_",
                                                 save_path: str = None):

    num_annotators = ["5", "15", "30"]
    color_map = {"5": "red", "15": "orange", "30": "green"}

    all_results = {}

    for n in num_annotators:
        path = os.path.join(BASE_DIR, "runs", dataset, query_strategy, "reputation_based", n, "E")
        all_accuracies = {}

        if not os.path.isdir(path):
            print(f"[IGNORADO] Pasta não encontrada: {path}")
            continue

        for folder in os.listdir(path):
            folder_path = os.path.join(path, folder)
            if os.path.isdir(folder_path) and folder.startswith(results_prefix):
                for file in os.listdir(folder_path):
                    if file.endswith(".csv"):
                        file_path = os.path.join(folder_path, file)

                        try:
                            df = pd.read_csv(file_path, sep=None, engine='python')
                            df.columns = [col.strip().replace('\ufeff', '') for col in df.columns]

                            for _, row in df.iterrows():
                                cycle = int(row['cycle'])
                                confusion_str = row[confusion_column]
                                confusion = np.array(ast.literal_eval(confusion_str))
                                correct = np.trace(confusion)
                                total = np.sum(confusion)
                                accuracy = correct / total if total > 0 else 0.0

                                if cycle not in all_accuracies:
                                    all_accuracies[cycle] = []
                                all_accuracies[cycle].append(accuracy)

                        except Exception as e:
                            print(f"Erro no arquivo {file_path}, linha {row.name}: {e}")

        if all_accuracies:
            cycles = sorted(all_accuracies.keys())
            mean_accuracies = [np.mean(all_accuracies[c]) for c in cycles]
            std_accuracies = [np.std(all_accuracies[c]) for c in cycles]
            all_results[n] = (cycles, mean_accuracies, std_accuracies)

    # Plotagem comparativa
    plt.figure(figsize=(10, 6))
    for n, (cycles, means, stds) in all_results.items():
        means_array = np.array(means)
        stds_array = np.array(stds)
        upper = np.clip(means_array + stds_array, 0, 1)
        lower = np.clip(means_array - stds_array, 0, 1)

        plt.plot(cycles, means, label=f"{n} annotators", linewidth=2, color=color_map[n])
        plt.fill_between(cycles, lower, upper, alpha=0.2, color=color_map[n])

    plt.xlabel('Cycle')
    plt.ylabel('Accuracy (Avg. ± std)')
    plt.title(f'Accuracy - Reputation Based - {dataset}')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Gráfico salvo em {save_path}")
        plt.close()
    else:
        plt.show()


def plot_precision_comparative_reputation_based_E(dataset: str,
                                                   query_strategy: str = "uncertainty_sampling",
                                                   confusion_column: str = "confusion_matrix",
                                                   results_prefix: str = "results_",
                                                   save_path: str = None):

    num_annotators = ["5", "15", "30"]
    color_map = {"5": "red", "15": "orange", "30": "green"}

    all_results = {}

    for n in num_annotators:
        path = os.path.join("runs", dataset, query_strategy, "reputation_based", n, "E")
        all_precisions = {}

        if not os.path.isdir(path):
            print(f"[IGNORADO] Pasta não encontrada: {path}")
            continue

        for folder in os.listdir(path):
            folder_path = os.path.join(path, folder)
            if os.path.isdir(folder_path) and folder.startswith(results_prefix):
                for file in os.listdir(folder_path):
                    if file.endswith(".csv"):
                        file_path = os.path.join(folder_path, file)

                        try:
                            df = pd.read_csv(file_path, sep=None, engine='python')
                            df.columns = [col.strip().replace('\ufeff', '') for col in df.columns]

                            for _, row in df.iterrows():
                                cycle = int(row['cycle'])
                                confusion_str = row[confusion_column]
                                confusion = np.array(ast.literal_eval(confusion_str))

                                # Evita divisão por zero e calcula precision macro
                                with np.errstate(divide='ignore', invalid='ignore'):
                                    precision_per_class = np.diag(confusion) / np.sum(confusion, axis=0)
                                    precision_per_class = np.nan_to_num(precision_per_class)  # substitui NaNs por 0
                                    precision = np.mean(precision_per_class)

                                if cycle not in all_precisions:
                                    all_precisions[cycle] = []
                                all_precisions[cycle].append(precision)

                        except Exception as e:
                            print(f"Erro no arquivo {file_path}, linha {row.name}: {e}")

        if all_precisions:
            cycles = sorted(all_precisions.keys())
            mean_precisions = [np.mean(all_precisions[c]) for c in cycles]
            std_precisions = [np.std(all_precisions[c]) for c in cycles]
            all_results[n] = (cycles, mean_precisions, std_precisions)

    # Plotagem comparativa
    plt.figure(figsize=(10, 6))
    for n, (cycles, means, stds) in all_results.items():
        plt.plot(cycles, means, label=f"{n} annotators", linewidth=2, color=color_map[n])
        upper = np.clip(np.array(means) + np.array(stds), 0, 1)
        lower = np.clip(np.array(means) - np.array(stds), 0, 1)
        plt.fill_between(cycles, lower, upper, alpha=0.2, color=color_map[n])

    plt.xlabel('Cycle')
    plt.ylabel('Precision (Avg. ± std)')
    plt.title(f'Precision - Reputation Based - {dataset}')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Gráfico salvo em {save_path}")
        plt.close()
    else:
        plt.show()

def plot_f1_comparative_reputation_based_E(dataset: str,
                                           query_strategy: str = "uncertainty_sampling",
                                           confusion_column: str = "confusion_matrix",
                                           results_prefix: str = "results_",
                                           save_path: str = None):

    num_annotators = ["5", "15", "30"]
    color_map = {"5": "red", "15": "orange", "30": "green"}

    all_results = {}

    for n in num_annotators:
        path = os.path.join("runs", dataset, query_strategy, "reputation_based", n, "E")
        all_f1_scores = {}

        if not os.path.isdir(path):
            print(f"[IGNORADO] Pasta não encontrada: {path}")
            continue

        for folder in os.listdir(path):
            folder_path = os.path.join(path, folder)
            if os.path.isdir(folder_path) and folder.startswith(results_prefix):
                for file in os.listdir(folder_path):
                    if file.endswith(".csv"):
                        file_path = os.path.join(folder_path, file)

                        try:
                            df = pd.read_csv(file_path, sep=None, engine='python')
                            df.columns = [col.strip().replace('\ufeff', '') for col in df.columns]

                            for _, row in df.iterrows():
                                cycle = int(row['cycle'])
                                confusion_str = row[confusion_column]
                                confusion = np.array(ast.literal_eval(confusion_str))

                                # Calcula precision e recall por classe
                                with np.errstate(divide='ignore', invalid='ignore'):
                                    tp = np.diag(confusion)
                                    fp = np.sum(confusion, axis=0) - tp
                                    fn = np.sum(confusion, axis=1) - tp

                                    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) != 0)
                                    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) != 0)
                                    f1 = np.divide(2 * precision * recall, precision + recall,
                                                   out=np.zeros_like(precision, dtype=float),
                                                   where=(precision + recall) != 0)

                                    f1_score_macro = np.mean(f1)

                                if cycle not in all_f1_scores:
                                    all_f1_scores[cycle] = []
                                all_f1_scores[cycle].append(f1_score_macro)

                        except Exception as e:
                            print(f"Erro no arquivo {file_path}, linha {row.name}: {e}")

        if all_f1_scores:
            cycles = sorted(all_f1_scores.keys())
            mean_f1 = [np.mean(all_f1_scores[c]) for c in cycles]
            std_f1 = [np.std(all_f1_scores[c]) for c in cycles]
            all_results[n] = (cycles, mean_f1, std_f1)

    # Plotagem comparativa
    plt.figure(figsize=(10, 6))
    for n, (cycles, means, stds) in all_results.items():
        plt.plot(cycles, means, label=f"{n} annotators", linewidth=2, color=color_map[n])
        upper = np.clip(np.array(means) + np.array(stds), 0, 1)
        lower = np.clip(np.array(means) - np.array(stds), 0, 1)
        plt.fill_between(cycles, lower, upper, alpha=0.2, color=color_map[n])

    plt.xlabel('Cycle')
    plt.ylabel('F1-score (Avg. ± std)')
    plt.title(f'F1-score - Reputation Based - {dataset}')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Gráfico salvo em {save_path}")
        plt.close()
    else:
        plt.show()

def plot_recall_comparative_reputation_based_E(dataset: str,
                                               query_strategy: str = "uncertainty_sampling",
                                               confusion_column: str = "confusion_matrix",
                                               results_prefix: str = "results_",
                                               save_path: str = None):

    num_annotators = ["5", "15", "30"]
    color_map = {"5": "red", "15": "orange", "30": "green"}

    all_results = {}

    for n in num_annotators:
        path = os.path.join("runs", dataset, query_strategy, "reputation_based", n, "E")
        all_recalls = {}

        if not os.path.isdir(path):
            print(f"[IGNORADO] Pasta não encontrada: {path}")
            continue

        for folder in os.listdir(path):
            folder_path = os.path.join(path, folder)
            if os.path.isdir(folder_path) and folder.startswith(results_prefix):
                for file in os.listdir(folder_path):
                    if file.endswith(".csv"):
                        file_path = os.path.join(folder_path, file)

                        try:
                            df = pd.read_csv(file_path, sep=None, engine='python')
                            df.columns = [col.strip().replace('\ufeff', '') for col in df.columns]

                            for _, row in df.iterrows():
                                cycle = int(row['cycle'])
                                confusion_str = row[confusion_column]
                                confusion = np.array(ast.literal_eval(confusion_str))

                                # Cálculo do recall por classe
                                with np.errstate(divide='ignore', invalid='ignore'):
                                    tp = np.diag(confusion)
                                    fn = np.sum(confusion, axis=1) - tp

                                    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) != 0)
                                    recall_macro = np.mean(recall)

                                if cycle not in all_recalls:
                                    all_recalls[cycle] = []
                                all_recalls[cycle].append(recall_macro)

                        except Exception as e:
                            print(f"Erro no arquivo {file_path}, linha {row.name}: {e}")

        if all_recalls:
            cycles = sorted(all_recalls.keys())
            mean_recall = [np.mean(all_recalls[c]) for c in cycles]
            std_recall = [np.std(all_recalls[c]) for c in cycles]
            all_results[n] = (cycles, mean_recall, std_recall)

    # Plotagem comparativa
    plt.figure(figsize=(10, 6))
    for n, (cycles, means, stds) in all_results.items():
        plt.plot(cycles, means, label=f"{n} annotators", linewidth=2, color=color_map[n])
        upper = np.clip(np.array(means) + np.array(stds), 0, 1)
        lower = np.clip(np.array(means) - np.array(stds), 0, 1)
        plt.fill_between(cycles, lower, upper, alpha=0.2, color=color_map[n])

    plt.xlabel('Cycle')
    plt.ylabel('Recall (Avg. ± std)')
    plt.title(f"Recall - Reputation Based - {dataset}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Gráfico salvo em {save_path}")
        plt.close()
    else:
        plt.show()


# =================================================================================
def plot_accuracy_comparison_strategies(dataset: str,
                                        query_strategy: str = "uncertainty_sampling",
                                        confusion_column: str = "confusion_matrix",
                                        results_prefix: str = "results_",
                                        save_path: str = None):
    

    strategies = {
        "Ground Truth": os.path.join(BASE_DIR, "runs", dataset, query_strategy, "ground_truth"),
        "5 Annotators": os.path.join(BASE_DIR, "runs", dataset, query_strategy, "reputation_based", "5", "E"),
        "15 Annotators": os.path.join(BASE_DIR, "runs", dataset, query_strategy, "reputation_based", "15", "E"),
        "30 Annotators": os.path.join(BASE_DIR, "runs", dataset, query_strategy, "reputation_based", "30", "E"),
        "Random": os.path.join(BASE_DIR, "runs", dataset, query_strategy, "random")
    }

    color_map = {
        "Ground Truth": "blue",
        "5 Annotators": "red",
        "15 Annotators": "green",
        "30 Annotators": "orange",
        "Random": "purple"
    }

    all_results = {}

    for label, path in strategies.items():
        all_accuracies = {}

        if not os.path.isdir(path):
            print(f"[IGNORADO] Pasta não encontrada: {path}")
            continue

        for folder in os.listdir(path):
            folder_path = os.path.join(path, folder)
            if os.path.isdir(folder_path) and folder.startswith(results_prefix):
                for file in os.listdir(folder_path):
                    if file.endswith(".csv"):
                        file_path = os.path.join(folder_path, file)

                        try:
                            df = pd.read_csv(file_path, sep=None, engine='python')
                            df.columns = [col.strip().replace('\ufeff', '') for col in df.columns]

                            for _, row in df.iterrows():
                                cycle = int(row['cycle'])
                                confusion_str = row[confusion_column]
                                confusion = np.array(ast.literal_eval(confusion_str))
                                correct = np.trace(confusion)
                                total = np.sum(confusion)
                                accuracy = correct / total if total > 0 else 0.0

                                if cycle not in all_accuracies:
                                    all_accuracies[cycle] = []
                                all_accuracies[cycle].append(accuracy)

                        except Exception as e:
                            print(f"Erro no arquivo {file_path}, linha {row.name}: {e}")

        if all_accuracies:
            cycles = sorted(all_accuracies.keys())
            mean_accuracies = [np.mean(all_accuracies[c]) for c in cycles]
            std_accuracies = [np.std(all_accuracies[c]) for c in cycles]
            all_results[label] = (cycles, mean_accuracies, std_accuracies)

    # Plot
    plt.figure(figsize=(12, 6))
    for label, (cycles, means, stds) in all_results.items():
        means_array = np.array(means)
        stds_array = np.array(stds)
        upper = np.clip(means_array + stds_array, 0, 1)
        lower = np.clip(means_array - stds_array, 0, 1)

        plt.plot(cycles, means, label=label, linewidth=2, color=color_map[label])
        plt.fill_between(cycles, lower, upper, alpha=0.2, color=color_map[label])

    plt.xlabel('Cycle')
    plt.ylabel('Accuracy (Avg. ± std)')
    plt.title(f'Accuracy Comparison - {dataset}')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Gráfico salvo em {save_path}")
        plt.close()
    else:
        plt.show()
# ==============================================================================



def plot_all():

    datasets = [DATASET_MNIST_FASHION]
    query_strategies = ["uncertainty_sampling", "margin_sampling", "entropy_sampling"]
    folders = ["random", "reputation_based", "ground_truth", "majority_voting"]
    sub_folders = ["5", "15", "30"]
    sub_sub_folders = ["E", "VH", "H", "M", "L"]

    for dataset in datasets:
        for query in query_strategies:
            for strategy in folders:
                # Permutação total só para "reputation_based"
                if strategy == "reputation_based":
                    for sub in sub_folders:
                        for level in sub_sub_folders:
                            path = os.path.join(BASE_DIR, "runs", dataset, query, strategy, sub, level)

                            if os.path.isdir(path):
                                print(f"Gerando plots para: {path}")

                                acc_path = os.path.join(path, "accuracy_plot.png")
                                prec_path = os.path.join(path, "precision_plot.png")
                                f1_path = os.path.join(path, "f1_score_plot.png")

                                plot_accuracy_with_std(path, save_path=acc_path)
                                plot_precision_with_std(path, save_path=prec_path)
                                plot_f1_score_with_std(path, save_path=f1_path)
                            else:
                                print(f"[IGNORADO] Pasta não encontrada: {path}")
                else:
                    # Sem subpastas
                    path = os.path.join(BASE_DIR, "runs", dataset, query, strategy)

                    if os.path.isdir(path):
                        print(f"Gerando plots para: {path}")

                        acc_path = os.path.join(path, "accuracy_plot.png")
                        prec_path = os.path.join(path, "precision_plot.png")
                        f1_path = os.path.join(path, "f1_score_plot.png")

                        plot_accuracy_with_std(path, save_path=acc_path)
                        plot_precision_with_std(path, save_path=prec_path)
                        plot_f1_score_with_std(path, save_path=f1_path)
                    else:
                        print(f"[IGNORADO] Pasta não encontrada: {path}")


# =================================================================================

#plot_accuracy_comparative_reputation_based_E(DATASET_MNIST_FASHION, "uncertainty_sampling", save_path=os.path.join(BASE_DIR, "runs", DATASET_MNIST_FASHION, "uncertainty_sampling", "reputation_based", "comparative_accuracy_plot.png"))

#plot_precision_comparative_reputation_based_E(DATASET_MNIST_FASHION, "uncertainty_sampling", save_path=os.path.join(BASE_DIR, "runs", DATASET_MNIST_FASHION, "uncertainty_sampling", "reputation_based", "comparative_precision_plot.png"))

#plot_f1_comparative_reputation_based_E(DATASET_MNIST_FASHION, "uncertainty_sampling", save_path=os.path.join(BASE_DIR, "runs", DATASET_MNIST_FASHION, "uncertainty_sampling", "reputation_based", "comparative_f1_plot.png"))

#plot_recall_comparative_reputation_based_E(DATASET_MNIST_FASHION, "uncertainty_sampling", save_path=os.path.join(BASE_DIR, "runs", DATASET_MNIST_FASHION, "uncertainty_sampling", "reputation_based", "comparative_recall_plot.png"))

# ===================

plot_accuracy_comparison_strategies(DATASET_MNIST_FASHION, "uncertainty_sampling", save_path=os.path.join(BASE_DIR, "runs", DATASET_MNIST_FASHION, "uncertainty_sampling", "comparative_accuracy_plot.png"))



# =================================================================================

#plot_accuracy_comparative_reputation_based_E(DATASET_MNIST, "uncertainty_sampling", save_path=os.path.join(BASE_DIR, "runs", DATASET_MNIST, "uncertainty_sampling", "reputation_based", "comparative_accuracy_plot.png"))

#plot_precision_comparative_reputation_based_E(DATASET_MNIST, "uncertainty_sampling", save_path=os.path.join(BASE_DIR, "runs", DATASET_MNIST, "uncertainty_sampling", "reputation_based", "comparative_precision_plot.png"))

#plot_f1_comparative_reputation_based_E(DATASET_MNIST, "uncertainty_sampling", save_path=os.path.join(BASE_DIR, "runs", DATASET_MNIST, "uncertainty_sampling", "reputation_based", "comparative_f1_plot.png"))

#plot_recall_comparative_reputation_based_E(DATASET_MNIST, "uncertainty_sampling", save_path=os.path.join(BASE_DIR, "runs", DATASET_MNIST, "uncertainty_sampling", "reputation_based", "comparative_recall_plot.png"))


# =================================================================================


#base_path = os.path.join(BASE_DIR, "runs", DATASET_MNIST_FASHION, "uncertainty_sampling", "reputation_based")

#sub_folders = ["5", "15", "30"]
#sub_sub_folders = ["E", "VH", "H", "M", "L"]

#plot_accuracy_with_std(base_path, save_path=os.path.join(base_path, "accuracy_plot.png"))
#plot_precision_with_std(base_path, save_path=os.path.join(base_path, "precision_plot.png"))
#plot_f1_score_with_std(base_path, save_path=os.path.join(base_path, "f1_score_plot.png"))