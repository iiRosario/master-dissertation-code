import os
import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from env import *
from matplotlib.cm import get_cmap
from matplotlib import colormaps


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

def compare_accuracy_with_without_rating(base_dir: str,
                                         dataset: str,
                                         query_strategy: str,
                                         strategy: str,
                                         oracle_size: str,
                                         expertise: str,
                                         confusion_column: str = "confusion_matrix",
                                         results_prefix: str = "results_",
                                         save_path: str = None):

    conditions = {
        "without_rating": os.path.join(base_dir, dataset, query_strategy, strategy, "without_rating", oracle_size, expertise),
        "with_rating": os.path.join(base_dir, dataset, query_strategy, strategy, "with_rating", oracle_size, expertise)
    }

    color_map = {
        "without_rating": "blue",
        "with_rating": "green"
    }

    all_results = {}

    for condition, path in conditions.items():
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
            all_results[condition] = (cycles, mean_accuracies, std_accuracies)

    # Plotagem comparativa
    plt.figure(figsize=(10, 6))
    for condition, (cycles, means, stds) in all_results.items():
        means_array = np.array(means)
        stds_array = np.array(stds)
        upper = np.clip(means_array + stds_array, 0, 1)
        lower = np.clip(means_array - stds_array, 0, 1)

        plt.plot(cycles, means, label=condition.replace("_", " ").title(), linewidth=2, color=color_map[condition])
        plt.fill_between(cycles, lower, upper, alpha=0.2, color=color_map[condition])

    plt.xlabel('Cycle')
    plt.ylabel('Accuracy (Avg. ± std)')
    plt.title(f'Accuracy Comparison - {strategy.replace("_", " ").title()} - {dataset}')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Gráfico salvo em {save_path}")
        plt.close()
    else:
        plt.show()



CLASSES_MNIST = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
CLASSES_MNIST_FASHION = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
CLASSES_EMNIST_LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
CLASSES_CIFAR_10 = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def plot_bar_experts_by_class(base_dir: str,
                              dataset: str,
                              query_strategy: str,
                              strategy: str,
                              oracle_size: str,
                              save_path: str,
                              filename: str):

    class_map = {
        DATASET_MNIST: CLASSES_MNIST,
        DATASET_MNIST_FASHION: CLASSES_MNIST_FASHION,
        DATASET_EMNIST_LETTERS: CLASSES_EMNIST_LETTERS,
        DATASET_CIFAR_10: CLASSES_CIFAR_10
    }

    if dataset not in class_map:
        raise ValueError(f"Unknown dataset '{dataset}'. Use one of: {list(class_map.keys())}")

    classes = class_map[dataset]
    num_classes = len(classes)

    expertise_levels = ["H", "M", "L"]
    rating_flags = ["with_rating", "without_rating"]
    color_map = {
        "H": "tab:red",
        "M": "tab:blue",
        "L": "tab:green"
    }

    # Estrutura: expertise -> classe -> lista de valores agregados across ratings
    data_by_expertise = {level: [[] for _ in range(num_classes)] for level in expertise_levels}

    for rating_flag in rating_flags:
        for expertise in expertise_levels:
            path = os.path.join(base_dir, dataset, query_strategy, strategy, rating_flag, oracle_size, expertise)
            if not os.path.isdir(path):
                print(f"[IGNORADO] Pasta não encontrada: {path}")
                continue

            for folder in os.listdir(path):
                folder_path = os.path.join(path, folder)
                if os.path.isdir(folder_path) and folder.startswith("results_"):
                    txt_file = os.path.join(folder_path, "expert_classes.txt")
                    if os.path.isfile(txt_file):
                        try:
                            with open(txt_file, "r") as f:
                                content = f.read().strip()
                                expert_counts = ast.literal_eval(content)
                                if len(expert_counts) != num_classes:
                                    continue
                                for i, count in enumerate(expert_counts):
                                    data_by_expertise[expertise][i].append(count)
                        except Exception as e:
                            print(f"[ERRO] ao ler {txt_file}: {e}")

    # --- Plotagem ---
    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(num_classes)
    width = 0.2

    for i, level in enumerate(expertise_levels):
        means = []
        lower_err = []
        upper_err = []

        for j in range(num_classes):
            values = data_by_expertise[level][j]
            if values:
                mean = np.mean(values)
                std = np.std(values)
                means.append(mean)

                # Clamp inferior para não descer abaixo de 0
                lo = std if mean - std >= 0 else mean
                lower_err.append(lo)
                upper_err.append(std)
            else:
                means.append(0)
                lower_err.append(0)
                upper_err.append(0)

        offset = (i - 1) * width
        positions = x + offset
        ax.bar(positions, means, width=width, color=color_map[level], label=level)

        # Barras de erro assimétricas reais (sem valores negativos)
        for pos, mean, lo, up in zip(positions, means, lower_err, upper_err):
            ax.errorbar(pos, mean, yerr=[[lo], [up]], fmt='none',
                        ecolor='black', capsize=5, linewidth=1.5)

    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of Expert Annotators")
    ax.set_title(f"Avg. Annotators per Class (With + Without Rating) - {dataset.upper()}, Oracle {oracle_size}")
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.legend(title="Expertise", loc="upper right")

    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    plot_path = os.path.join(save_path, filename)
    plt.savefig(plot_path)
    print(f"[OK] Bar plot with asymmetric error saved to {plot_path}")
    plt.close()



def plot_reputation_contributions(base_dir: str, 
                                   dataset: str,
                                   query_strategy: str,
                                   oracle_size: str,
                                   save_path: str,
                                   filename: str):
    strategy = "reputation_based"
    expertise_levels = ["H", "M", "L"]
    bar_colors = {"acc": "tab:red", "s": "tab:blue"}

    for expertise in expertise_levels:
        path = os.path.join(base_dir, dataset, query_strategy, strategy, "with_rating", oracle_size, expertise)
        if not os.path.isdir(path):
            print(f"[IGNORADO] Pasta não encontrada: {path}")
            continue

        acc_vals_final = []
        s_vals_final = []
        reputations_final = []
        annotator_ids = []

        for folder in os.listdir(path):
            folder_path = os.path.join(path, folder)
            annotator_dir = os.path.join(folder_path, "Annotators")
            if os.path.isdir(annotator_dir):
                for file in os.listdir(annotator_dir):
                    if file.startswith("Annotator_") and file.endswith(".csv"):
                        file_path = os.path.join(annotator_dir, file)
                        try:
                            df = pd.read_csv(file_path)
                            if "accuracies" in df.columns and "reputations" in df.columns:
                                acc_series = df["accuracies"].apply(ast.literal_eval)
                                rep_series = df["reputations"].apply(ast.literal_eval)

                                if not acc_series.empty and not rep_series.empty:
                                    last_acc = np.array(acc_series.iloc[-1])
                                    last_rep = np.array(rep_series.iloc[-1])
                                    last_s = last_rep - last_acc

                                    acc_mean = np.mean(last_acc)
                                    s_mean = np.mean(last_s)
                                    rep_total = np.mean(last_rep)

                                    acc_vals_final.append(acc_mean)
                                    s_vals_final.append(s_mean)
                                    reputations_final.append(rep_total)
                                    annotator_ids.append(file.replace(".csv", ""))
                        except Exception as e:
                            print(f"[ERRO] ao ler {file_path}: {e}")

        if not annotator_ids:
            print(f"[AVISO] Nenhum anotador encontrado para expertise {expertise}.")
            continue

        # Ordenar por reputação total (opcional)
        zipped = list(zip(annotator_ids, acc_vals_final, s_vals_final, reputations_final))
        zipped.sort(key=lambda x: x[3], reverse=True)  # ordena por reputação total
        annotator_ids, acc_vals_final, s_vals_final, reputations_final = zip(*zipped)

        # Limitar ao número de oracle_size (e.g., 30)
        oracle_size_int = int(oracle_size)
        annotator_ids = annotator_ids[:oracle_size_int]
        acc_vals_final = acc_vals_final[:oracle_size_int]
        s_vals_final = s_vals_final[:oracle_size_int]
        reputations_final = reputations_final[:oracle_size_int]

        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(annotator_ids))
        width = 0.6

        ax.bar(x, acc_vals_final, width, label="Accuracy", color=bar_colors["acc"])
        ax.bar(x, s_vals_final, width, bottom=acc_vals_final, label=r"$\mathcal{S}$", color=bar_colors["s"])
        ax.set_title(f"Reputation Contributions - {dataset.upper()}, Expertise: {expertise}, Oracle Size: {oracle_size}")
        ax.set_xticks(x)
        ax.set_xticklabels(annotator_ids, rotation=90)
        ax.set_xlabel(r"Annotator ($a_i$)")
        ax.set_ylabel(r"Final Reputation ($\mathcal{R}$)")
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        ax.legend()

        std_rep = np.std(reputations_final)
        print(f"[{expertise}] Std(Reputation) = {std_rep:.4f}")

        plt.tight_layout()
        os.makedirs(save_path, exist_ok=True)
        plot_path = os.path.join(save_path, f"{filename}_{expertise}.png")
        plt.savefig(plot_path)
        print(f"[OK] Plot salvo em: {plot_path}")
        plt.close()

# =================================================================================




def compare_rating_effect_across_expertise(base_dir: str,
                                           dataset: str,
                                           query_strategy: str,
                                           strategy: str,
                                           oracle_size: str,
                                           confusion_column: str = "confusion_matrix",
                                           results_prefix: str = "results_",
                                           save_dir: str = None,
                                           filename: str = None):

    expertise_levels = ["H", "M", "L"]  # Ordem controlada
    rating_conditions = ["with_rating", "without_rating"]

    color_map = {
        "with_rating": {
            "H": "red",
            "M": "green",
            "L": "brown"
        },
        "without_rating": {
            "H": "blue",
            "M": "orange",
            "L": "cyan"
        }
    }

    line_styles = {"H": "-", "M": "-", "L": "-"}

    all_results = {}

    for rating in rating_conditions:
        for expertise in expertise_levels:
            key = f"{rating}_{expertise}"
            path = os.path.join(base_dir, dataset, query_strategy, strategy, rating, oracle_size, expertise)
            all_accuracies = {}

            if not os.path.isdir(path):
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
                                    confusion = np.array(ast.literal_eval(row[confusion_column]))
                                    correct = np.trace(confusion)
                                    total = np.sum(confusion)
                                    accuracy = correct / total if total > 0 else 0.0
                                    all_accuracies.setdefault(cycle, []).append(accuracy)
                            except Exception:
                                continue

            if all_accuracies:
                cycles = sorted(all_accuracies.keys())
                mean_accuracies = [np.mean(all_accuracies[c]) for c in cycles]
                std_accuracies = [np.std(all_accuracies[c]) for c in cycles]
                all_results[key] = (cycles, mean_accuracies, std_accuracies)

    # --- Baseline ---
    baseline_path = os.path.join(base_dir, dataset, "uncertainty_sampling", "random")
    baseline_accuracies = {}

    if os.path.isdir(baseline_path):
        for folder in os.listdir(baseline_path):
            folder_path = os.path.join(baseline_path, folder)
            if os.path.isdir(folder_path) and folder.startswith(results_prefix):
                for file in os.listdir(folder_path):
                    if file.endswith(".csv"):
                        file_path = os.path.join(folder_path, file)
                        try:
                            df = pd.read_csv(file_path, sep=None, engine='python')
                            df.columns = [col.strip().replace('\ufeff', '') for col in df.columns]
                            for _, row in df.iterrows():
                                cycle = int(row['cycle'])
                                confusion = np.array(ast.literal_eval(row[confusion_column]))
                                correct = np.trace(confusion)
                                total = np.sum(confusion)
                                accuracy = correct / total if total > 0 else 0.0
                                baseline_accuracies.setdefault(cycle, []).append(accuracy)
                        except Exception:
                            continue

    # --- Plot ---
    plt.figure(figsize=(12, 7))
    plot_entries = []

    # Ordena a legenda: H, M, L (with) → H, M, L (without)
    for rating in rating_conditions:
        for expertise in expertise_levels:
            key = f"{rating}_{expertise}"
            if key not in all_results:
                continue
            cycles, means, stds = all_results[key]
            color = color_map[rating][expertise]
            linestyle = line_styles[expertise]
            label = f"{expertise} ({'W/ ' if rating == 'with_rating' else 'N/ '} Rating)"
            plot_entries.append((cycles, means, stds, color, linestyle, label))

    for cycles, means, stds, color, linestyle, label in plot_entries:
        means_array = np.array(means)
        stds_array = np.array(stds)
        upper = np.clip(means_array + stds_array, 0, 1)
        lower = np.clip(means_array - stds_array, 0, 1)
        plt.plot(cycles, means, label=label, linewidth=2, color=color, linestyle=linestyle)
        plt.fill_between(cycles, lower, upper, alpha=0.15, color=color)

    # Baseline
    if baseline_accuracies:
        baseline_cycles = sorted(baseline_accuracies.keys())
        baseline_means = [np.mean(baseline_accuracies[c]) for c in baseline_cycles]
        baseline_stds = [np.std(baseline_accuracies[c]) for c in baseline_cycles]

        means_array = np.array(baseline_means)
        stds_array = np.array(baseline_stds)
        upper = np.clip(means_array + stds_array, 0, 1)
        lower = np.clip(means_array - stds_array, 0, 1)

        plt.plot(baseline_cycles, means_array, label="Baseline", color="purple", linewidth=2)
        plt.fill_between(baseline_cycles, lower, upper, alpha=0.15, color="purple")

    plt.xlabel("Cycle")
    plt.ylabel("Accuracy (Avg. ± std)")
    plt.title(f"Accuracy Comparison by Expertise - {dataset}, {strategy}, oracle={oracle_size}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        if filename:
            save_path = os.path.join(save_dir, filename)
        else:
            save_path = os.path.join(save_dir, f"accuracy_comparison_{dataset}_{strategy}_{oracle_size}.png")
        plt.savefig(save_path)
        print(f"Plot saved at {save_path}")
        plt.close()
    else:
        plt.show()



def compare_all_rating_effect_by_oracle_size(base_dir: str,
                                         dataset: str,
                                         query_strategy: str,
                                         strategy: str,
                                         expertise: str,
                                         confusion_column: str = "confusion_matrix",
                                         results_prefix: str = "results_",
                                         save_dir: str = None,
                                         filename: str = None):

    oracle_sizes = ["5", "15", "30"]
    rating_conditions = ["with_rating", "without_rating"]

    # Cores distintas fixas por configuração
    color_map = {
        "with_rating": {
            "5": "red",
            "15": "green",
            "30": "blue"
        },
        "without_rating": {
            "5": "orange",
            "15": "brown",
            "30": "cyan"
        }
    }

    all_results = {}

    # Coleta os resultados por rating e oracle_size
    for rating in rating_conditions:
        for oracle_size in oracle_sizes:
            key = f"{rating}_{oracle_size}"
            path = os.path.join(base_dir, dataset, query_strategy, strategy, rating, oracle_size, expertise)
            all_accuracies = {}

            if not os.path.isdir(path):
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
                                    confusion = np.array(ast.literal_eval(row[confusion_column]))
                                    correct = np.trace(confusion)
                                    total = np.sum(confusion)
                                    accuracy = correct / total if total > 0 else 0.0
                                    all_accuracies.setdefault(cycle, []).append(accuracy)
                            except Exception:
                                continue

            if all_accuracies:
                cycles = sorted(all_accuracies.keys())
                mean_accuracies = [np.mean(all_accuracies[c]) for c in cycles]
                std_accuracies = [np.std(all_accuracies[c]) for c in cycles]
                all_results[key] = {
                    "cycles": cycles,
                    "means": mean_accuracies,
                    "stds": std_accuracies,
                    "color": color_map[rating][oracle_size]
                }

    # --- Coleta da baseline ---
    baseline_accuracies = {}
    baseline_path = os.path.join(base_dir, dataset, "uncertainty_sampling", "random")

    if os.path.isdir(baseline_path):
        for folder in os.listdir(baseline_path):
            folder_path = os.path.join(baseline_path, folder)
            if os.path.isdir(folder_path) and folder.startswith(results_prefix):
                for file in os.listdir(folder_path):
                    if file.endswith(".csv"):
                        file_path = os.path.join(folder_path, file)
                        try:
                            df = pd.read_csv(file_path, sep=None, engine='python')
                            df.columns = [col.strip().replace('\ufeff', '') for col in df.columns]

                            for _, row in df.iterrows():
                                cycle = int(row['cycle'])
                                confusion = np.array(ast.literal_eval(row[confusion_column]))
                                correct = np.trace(confusion)
                                total = np.sum(confusion)
                                accuracy = correct / total if total > 0 else 0.0
                                baseline_accuracies.setdefault(cycle, []).append(accuracy)
                        except Exception:
                            continue

    # --- Plotagem ---
    plt.figure(figsize=(12, 7))
    plot_entries = []

    # Adiciona as curvas com ordem: with_rating primeiro, depois without_rating
    for rating in ["with_rating", "without_rating"]:
        for oracle_size in oracle_sizes:
            key = f"{rating}_{oracle_size}"
            if key in all_results:
                data = all_results[key]
                label = f"oracle={oracle_size} ({'W/ ' if rating == 'with_rating' else 'N/ '} Rating)"
                plot_entries.append((data["cycles"], data["means"], data["stds"], data["color"], label))

    for cycles, means, stds, color, label in plot_entries:
        means = np.array(means)
        stds = np.array(stds)
        upper = np.clip(means + stds, 0, 1)
        lower = np.clip(means - stds, 0, 1)
        plt.plot(cycles, means, label=label, linewidth=2, color=color)
        plt.fill_between(cycles, lower, upper, alpha=0.15, color=color)

    # Adiciona a baseline (última)
    if baseline_accuracies:
        baseline_cycles = sorted(baseline_accuracies.keys())
        baseline_means = [np.mean(baseline_accuracies[c]) for c in baseline_cycles]
        baseline_stds = [np.std(baseline_accuracies[c]) for c in baseline_cycles]

        means_array = np.array(baseline_means)
        stds_array = np.array(baseline_stds)
        upper = np.clip(means_array + stds_array, 0, 1)
        lower = np.clip(means_array - stds_array, 0, 1)

        plt.plot(baseline_cycles, means_array, label="Baseline", color="purple", linewidth=2)
        plt.fill_between(baseline_cycles, lower, upper, alpha=0.15, color="purple")

    # Layout
    plt.xlabel("Cycle")
    plt.ylabel("Accuracy (Avg. ± std)")
    plt.title(f"Accuracy by Oracle Size - Expertise {expertise}, {dataset}, {strategy}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Salvar ou mostrar
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        if filename:
            save_path = os.path.join(save_dir, filename)
        else:
            save_path = os.path.join(save_dir, f"accuracy_oracle_comparison_{dataset}_{strategy}_{expertise}.png")
        plt.savefig(save_path)
        print(f"Plot saved at {save_path}")
        plt.close()
    else:
        plt.show()

def compare_rating_effect_by_oracle_size(base_dir: str,
                                         dataset: str,
                                         query_strategy: str,
                                         strategy: str,
                                         expertise: str,
                                         confusion_column: str = "confusion_matrix",
                                         results_prefix: str = "results_",
                                         save_dir: str = None,
                                         filename: str = None):

    oracle_sizes = ["5", "15", "30"]
    rating_conditions = ["with_rating", "without_rating"]

    # Cores distintas por configuração
    color_map = {
        "with_rating": {
            "5": "red",
            "15": "green",
            "30": "blue"
        },
        "without_rating": {
            "5": "orange",
            "15": "brown",
            "30": "cyan"
        }
    }

    all_results = {}

    for rating in rating_conditions:
        for oracle_size in oracle_sizes:
            key = f"{rating}_{oracle_size}"
            path = os.path.join(base_dir, dataset, query_strategy, strategy, rating, oracle_size, expertise)
            all_accuracies = {}

            if not os.path.isdir(path):
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
                                    confusion = np.array(ast.literal_eval(row[confusion_column]))
                                    correct = np.trace(confusion)
                                    total = np.sum(confusion)
                                    accuracy = correct / total if total > 0 else 0.0
                                    all_accuracies.setdefault(cycle, []).append(accuracy)
                            except Exception:
                                continue

            if all_accuracies:
                cycles = sorted(all_accuracies.keys())
                mean_accuracies = [np.mean(all_accuracies[c]) for c in cycles]
                std_accuracies = [np.std(all_accuracies[c]) for c in cycles]
                all_results[key] = {
                    "cycles": cycles,
                    "means": mean_accuracies,
                    "stds": std_accuracies,
                    "color": color_map[rating][oracle_size]
                }

    # --- Coleta da baseline (random) ---
    baseline_accuracies = {}
    baseline_path = os.path.join(base_dir, dataset, "uncertainty_sampling", "random")

    if os.path.isdir(baseline_path):
        for folder in os.listdir(baseline_path):
            folder_path = os.path.join(baseline_path, folder)
            if os.path.isdir(folder_path) and folder.startswith(results_prefix):
                for file in os.listdir(folder_path):
                    if file.endswith(".csv"):
                        file_path = os.path.join(folder_path, file)
                        try:
                            df = pd.read_csv(file_path, sep=None, engine='python')
                            df.columns = [col.strip().replace('\ufeff', '') for col in df.columns]

                            for _, row in df.iterrows():
                                cycle = int(row['cycle'])
                                confusion = np.array(ast.literal_eval(row[confusion_column]))
                                correct = np.trace(confusion)
                                total = np.sum(confusion)
                                accuracy = correct / total if total > 0 else 0.0
                                baseline_accuracies.setdefault(cycle, []).append(accuracy)
                        except Exception:
                            continue

    # --- Plotagem ---
    plt.figure(figsize=(12, 7))
    plot_entries = []

    # Curvas por oracle_size e rating
    for rating in rating_conditions:
        for oracle_size in oracle_sizes:
            key = f"{rating}_{oracle_size}"
            if key in all_results:
                data = all_results[key]
                label = f"oracle={oracle_size} ({'W/ ' if rating == 'with_rating' else 'N/ '} Rating)"
                plot_entries.append((data["cycles"], data["means"], data["stds"], data["color"], label))

    # Plot ordenado
    for cycles, means, stds, color, label in plot_entries:
        means = np.array(means)
        stds = np.array(stds)
        upper = np.clip(means + stds, 0, 1)
        lower = np.clip(means - stds, 0, 1)
        plt.plot(cycles, means, label=label, linewidth=2, color=color)
        plt.fill_between(cycles, lower, upper, alpha=0.15, color=color)

    # Plot baseline
    if baseline_accuracies:
        baseline_cycles = sorted(baseline_accuracies.keys())
        baseline_means = [np.mean(baseline_accuracies[c]) for c in baseline_cycles]
        baseline_stds = [np.std(baseline_accuracies[c]) for c in baseline_cycles]

        means_array = np.array(baseline_means)
        stds_array = np.array(baseline_stds)
        upper = np.clip(means_array + stds_array, 0, 1)
        lower = np.clip(means_array - stds_array, 0, 1)

        plt.plot(baseline_cycles, means_array, label="Baseline", color="purple", linewidth=2)
        plt.fill_between(baseline_cycles, lower, upper, alpha=0.15, color="purple")

    plt.xlabel("Cycle")
    plt.ylabel("Accuracy (Avg. ± std)")
    plt.title(f"Accuracy by Oracle Size - Expertise {expertise}, {dataset}, {strategy}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Salvar ou exibir
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        if filename:
            save_path = os.path.join(save_dir, filename)
        else:
            save_path = os.path.join(save_dir, f"accuracy_oracle_comparison_{dataset}_{strategy}_{expertise}.png")
        plt.savefig(save_path)
        print(f"Plot saved at {save_path}")
        plt.close()
    else:
        plt.show()




def compare_expertise_within_rating(base_dir, dataset, query_strategy, strategy, oracle_size, rating, confusion_column="confusion_matrix", results_prefix="results_", save_dir=None, filename=None):

    assert rating in ["with_rating", "without_rating"], "Parameter 'rating' must be 'with_rating' or 'without_rating'."

    expertise_levels = ["H", "L", "M"]
    color_map = {"H": "red", "M": "blue", "L": "green"}
    baseline_color = "purple"
    line_styles = {"H": "-", "L": "-", "M": "-"}
    all_results = {}

    # --- Processa cada expertise ---
    for expertise in expertise_levels:
        path = os.path.join(base_dir, dataset, query_strategy, strategy, rating, oracle_size, expertise)
        all_accuracies = {}

        if not os.path.isdir(path):
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
                                confusion = np.array(ast.literal_eval(row[confusion_column]))
                                correct = np.trace(confusion)
                                total = np.sum(confusion)
                                accuracy = correct / total if total > 0 else 0.0
                                all_accuracies.setdefault(cycle, []).append(accuracy)
                        except Exception:
                            continue

        if all_accuracies:
            cycles = sorted(all_accuracies.keys())
            means = [np.mean(all_accuracies[c]) for c in cycles]
            stds = [np.std(all_accuracies[c]) for c in cycles]
            all_results[expertise] = (cycles, means, stds)

    # --- Processa baseline (random) ---
    baseline_path = os.path.join(base_dir, dataset, "uncertainty_sampling", "random")
    baseline_accuracies = {}

    if os.path.isdir(baseline_path):
        for folder in os.listdir(baseline_path):
            folder_path = os.path.join(baseline_path, folder)
            if os.path.isdir(folder_path) and folder.startswith(results_prefix):
                for file in os.listdir(folder_path):
                    if file.endswith(".csv"):
                        file_path = os.path.join(folder_path, file)
                        try:
                            df = pd.read_csv(file_path, sep=None, engine='python')
                            df.columns = [col.strip().replace('\ufeff', '') for col in df.columns]

                            for _, row in df.iterrows():
                                cycle = int(row['cycle'])
                                confusion = np.array(ast.literal_eval(row[confusion_column]))
                                correct = np.trace(confusion)
                                total = np.sum(confusion)
                                accuracy = correct / total if total > 0 else 0.0
                                baseline_accuracies.setdefault(cycle, []).append(accuracy)
                        except Exception:
                            continue

    # --- Plotagem ---
    plt.figure(figsize=(12, 7))

    # Curvas por expertise
    for expertise, (cycles, means, stds) in all_results.items():
        color = color_map[expertise]
        linestyle = line_styles[expertise]
        means_array = np.array(means)
        stds_array = np.array(stds)
        upper = np.clip(means_array + stds_array, 0, 1)
        lower = np.clip(means_array - stds_array, 0, 1)
        plt.plot(cycles, means, label=expertise, linewidth=2, color=color, linestyle=linestyle)
        plt.fill_between(cycles, lower, upper, alpha=0.15, color=color)

    # Curva baseline
    if baseline_accuracies:
        baseline_cycles = sorted(baseline_accuracies.keys())
        baseline_means = [np.mean(baseline_accuracies[c]) for c in baseline_cycles]
        baseline_stds = [np.std(baseline_accuracies[c]) for c in baseline_cycles]
        means_array = np.array(baseline_means)
        stds_array = np.array(baseline_stds)
        upper = np.clip(means_array + stds_array, 0, 1)
        lower = np.clip(means_array - stds_array, 0, 1)
        plt.plot(baseline_cycles, baseline_means, label="Baseline", color=baseline_color, linewidth=2, linestyle="-")
        plt.fill_between(baseline_cycles, lower, upper, alpha=0.15, color=baseline_color)

    # Layout do gráfico
    plt.xlabel("Cycle")
    plt.ylabel("Accuracy (Avg. ± std)")
    plt.title(f"Expertise Comparison ({'With' if rating == 'with_rating' else 'Without'} Rating) - {dataset}, {strategy}, oracle={oracle_size}")
    plt.grid(True)
    plt.legend(title="Expertise")
    plt.tight_layout()

    # Salvamento ou exibição
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        if not filename:
            filename = f"expertise_comparison_{rating}_{dataset}_{strategy}_{oracle_size}.png"
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()



def compare_rating_for_expertise(base_dir,  dataset,
                                 query_strategy: str,
                                 strategy: str,
                                 oracle_size: str,
                                 expertise: str,  # "H", "M", "L"
                                 confusion_column: str = "confusion_matrix",
                                 results_prefix: str = "results_",
                                 save_dir: str = None,
                                 filename: str = None):

    assert expertise in ["H", "M", "L"], "Expertise must be one of: 'H', 'M', 'L'."

    ratings = ["with_rating", "without_rating"]
    rating_colors = {
        "with_rating": "red",
        "without_rating": "blue"
    }
    line_styles = {
        "with_rating": "-",
        "without_rating": "-"
    }

    all_results = {}

    for rating in ratings:
        path = os.path.join(base_dir, dataset, query_strategy, strategy, rating, oracle_size, expertise)
        all_accuracies = {}

        if not os.path.isdir(path):
            print(f"[IGNORED] Folder not found: {path}")
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
                            print(f"Error in file {file_path}, row {row.name}: {e}")

        if all_accuracies:
            cycles = sorted(all_accuracies.keys())
            mean_accuracies = [np.mean(all_accuracies[c]) for c in cycles]
            std_accuracies = [np.std(all_accuracies[c]) for c in cycles]
            all_results[rating] = (cycles, mean_accuracies, std_accuracies)

    # --- Baseline ---
    baseline_path = os.path.join(base_dir, dataset, "uncertainty_sampling", "random")
    baseline_accuracies = {}

    if os.path.isdir(baseline_path):
        for folder in os.listdir(baseline_path):
            folder_path = os.path.join(baseline_path, folder)
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

                                if cycle not in baseline_accuracies:
                                    baseline_accuracies[cycle] = []
                                baseline_accuracies[cycle].append(accuracy)

                        except Exception as e:
                            print(f"Error in baseline file {file_path}, row {row.name}: {e}")

    # --- Plot ---
    plt.figure(figsize=(12, 7))

    for rating, (cycles, means, stds) in all_results.items():
        color = rating_colors[rating]
        linestyle = line_styles[rating]
        if(rating == WITH_RATING):
            label = f"{expertise} - (W/ Rating)"
        else:
            label = f"{expertise} - (N/ Rating)"

        means_array = np.array(means)
        stds_array = np.array(stds)
        upper = np.clip(means_array + stds_array, 0, 1)
        lower = np.clip(means_array - stds_array, 0, 1)

        plt.plot(cycles, means, label=label, linewidth=2, color=color, linestyle=linestyle)
        plt.fill_between(cycles, lower, upper, alpha=0.15, color=color)

    # Baseline
    if baseline_accuracies:
        baseline_cycles = sorted(baseline_accuracies.keys())
        baseline_means = [np.mean(baseline_accuracies[c]) for c in baseline_cycles]
        baseline_stds = [np.std(baseline_accuracies[c]) for c in baseline_cycles]

        means_array = np.array(baseline_means)
        stds_array = np.array(baseline_stds)
        upper = np.clip(means_array + stds_array, 0, 1)
        lower = np.clip(means_array - stds_array, 0, 1)

        plt.plot(baseline_cycles, baseline_means, label="Baseline", color="purple", linewidth=2, linestyle="-")
        plt.fill_between(baseline_cycles, lower, upper, alpha=0.15, color="purple")

    plt.xlabel("Cycle")
    plt.ylabel("Accuracy (Avg. ± std)")
    plt.title(f"Expertise {expertise} - With vs Without Rating - {dataset}, {strategy}, oracle={oracle_size}")
    plt.grid(True)
    plt.legend(title="Setting")
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        if filename:
            save_path = os.path.join(save_dir, filename)
        else:
            save_path = os.path.join(save_dir, f"{expertise}_rating_comparison_{dataset}_{strategy}_{oracle_size}.png")
        plt.savefig(save_path)
        print(f"Plot saved at {save_path}")
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
        "Baseline": os.path.join(BASE_DIR, "runs", dataset, query_strategy, "random")
    }

    color_map = {
        "Ground Truth": "blue",
        "5 Annotators": "red",
        "15 Annotators": "green",
        "30 Annotators": "orange",
        "Baseline": "purple"
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

def plot_precision_comparison_strategies(dataset: str,
                                         query_strategy: str = "uncertainty_sampling",
                                         confusion_column: str = "confusion_matrix",
                                         results_prefix: str = "results_",
                                         save_path: str = None):

    strategies = {
        "Ground Truth": os.path.join(BASE_DIR, "runs", dataset, query_strategy, "ground_truth"),
        "5 Annotators": os.path.join(BASE_DIR, "runs", dataset, query_strategy, "reputation_based", "5", "E"),
        "15 Annotators": os.path.join(BASE_DIR, "runs", dataset, query_strategy, "reputation_based", "15", "E"),
        "30 Annotators": os.path.join(BASE_DIR, "runs", dataset, query_strategy, "reputation_based", "30", "E"),
        "Baseline": os.path.join(BASE_DIR, "runs", dataset, query_strategy, "random")
    }

    color_map = {
        "Ground Truth": "blue",
        "5 Annotators": "red",
        "15 Annotators": "green",
        "30 Annotators": "orange",
        "Baseline": "purple"
    }

    all_results = {}

    for label, path in strategies.items():
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

                                with np.errstate(divide='ignore', invalid='ignore'):
                                    tp = np.diag(confusion)
                                    fp = np.sum(confusion, axis=0) - tp
                                    precision_per_class = tp / (tp + fp)
                                    precision_per_class = np.nan_to_num(precision_per_class)
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
            all_results[label] = (cycles, mean_precisions, std_precisions)

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
    plt.ylabel('Precision (Avg. ± std)')
    plt.title(f'Precision Comparison - {dataset}')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Gráfico salvo em {save_path}")
        plt.close()
    else:
        plt.show()

def plot_f1_score_comparison_strategies(dataset: str,
                                        query_strategy: str = "uncertainty_sampling",
                                        confusion_column: str = "confusion_matrix",
                                        results_prefix: str = "results_",
                                        save_path: str = None):

    strategies = {
        "Ground Truth": os.path.join(BASE_DIR, "runs", dataset, query_strategy, "ground_truth"),
        "5 Annotators": os.path.join(BASE_DIR, "runs", dataset, query_strategy, "reputation_based", "5", "E"),
        "15 Annotators": os.path.join(BASE_DIR, "runs", dataset, query_strategy, "reputation_based", "15", "E"),
        "30 Annotators": os.path.join(BASE_DIR, "runs", dataset, query_strategy, "reputation_based", "30", "E"),
        "Baseline": os.path.join(BASE_DIR, "runs", dataset, query_strategy, "random")
    }

    color_map = {
        "Ground Truth": "blue",
        "5 Annotators": "red",
        "15 Annotators": "green",
        "30 Annotators": "orange",
        "Baseline": "purple"
    }

    all_results = {}

    for label, path in strategies.items():
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

                                with np.errstate(divide='ignore', invalid='ignore'):
                                    tp = np.diag(confusion)
                                    fp = np.sum(confusion, axis=0) - tp
                                    fn = np.sum(confusion, axis=1) - tp

                                    precision = tp / (tp + fp)
                                    recall = tp / (tp + fn)

                                    precision = np.nan_to_num(precision)
                                    recall = np.nan_to_num(recall)

                                    f1 = 2 * precision * recall / (precision + recall)
                                    f1 = np.nan_to_num(f1)

                                    f1_macro = np.mean(f1)

                                if cycle not in all_f1_scores:
                                    all_f1_scores[cycle] = []
                                all_f1_scores[cycle].append(f1_macro)

                        except Exception as e:
                            print(f"Erro no arquivo {file_path}, linha {row.name}: {e}")

        if all_f1_scores:
            cycles = sorted(all_f1_scores.keys())
            mean_f1s = [np.mean(all_f1_scores[c]) for c in cycles]
            std_f1s = [np.std(all_f1_scores[c]) for c in cycles]
            all_results[label] = (cycles, mean_f1s, std_f1s)

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
    plt.ylabel('F1-score (Avg. ± std)')
    plt.title(f'F1-score Comparison - {dataset}')
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
def plot_reputation_per_class(base_path: str, save_path_prefix: str = None):
    class_reputations = {i: {} for i in range(10)}  # Dict de class_id -> iteration -> lista de reputações

    # Caminha por todas as subpastas "results_X/Annotators"
    for results_folder in os.listdir(base_path):
        results_path = os.path.join(base_path, results_folder, "Annotators")
        if not os.path.isdir(results_path):
            continue

        for annotator_file in os.listdir(results_path):
            if not annotator_file.startswith("Annotator_") or not annotator_file.endswith(".csv"):
                continue

            file_path = os.path.join(results_path, annotator_file)
            try:
                df = pd.read_csv(file_path, sep=None, engine='python')
                df.columns = [col.strip().replace('\ufeff', '') for col in df.columns]

                for _, row in df.iterrows():
                    iteration = int(row['iteration'])
                    reputations = ast.literal_eval(row['reputations'])

                    for cls_id, rep in enumerate(reputations):
                        if iteration not in class_reputations[cls_id]:
                            class_reputations[cls_id][iteration] = []
                        class_reputations[cls_id][iteration].append(rep)

            except Exception as e:
                print(f"Erro ao processar {file_path}: {e}")

    # Função auxiliar para plotar um grupo de classes
    def plot_group(class_ids, title_suffix, file_suffix):
        plt.figure(figsize=(12, 7))
        for cls_id in class_ids:
            iterations = sorted(class_reputations[cls_id].keys())
            means = [np.mean(class_reputations[cls_id][it]) for it in iterations]
            stds = [np.std(class_reputations[cls_id][it]) for it in iterations]
            upper = np.array(means) + np.array(stds)
            lower = np.array(means) - np.array(stds)

            plt.plot(iterations, means, label=f'Class {cls_id}')
            plt.fill_between(iterations, lower, upper, alpha=0.2)

        plt.xlabel('Iteration')
        plt.ylabel('Reputation (Avg ± std)')
        plt.title(f'Class Reputation over Iterations (Classes {title_suffix})')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        if save_path_prefix:
            out_path = f"{save_path_prefix}_classes_{file_suffix}.png"
            plt.savefig(out_path)
            print(f"Gráfico salvo em {out_path}")
            plt.close()
        else:
            plt.show()

    # Plot para classes 0 a 4
    plot_group(class_ids=range(5), title_suffix="0–4", file_suffix="0_4")

    # Plot para classes 5 a 9
    plot_group(class_ids=range(5, 10), title_suffix="5–9", file_suffix="5_9")




# =================================================================================
def save_metrics_summary_csv(base_dir: str, 
                              dataset: str,
                              query_strategy: str,
                              results_prefix: str = "results_",
                              save_path: str = None,
                              filename_prefix: str = None):
    strategy = "reputation_based"
    expertise_levels = ["H", "M", "L"]
    rating_conditions = ["with_rating", "without_rating", WITH_ONLY_RATING]
    metric_columns = {
        "accuracy": "accuracy_per_class",
        "precision": "precision_per_class",
        "f1_score": "f1_score_per_class"
    }

    discovered_oracle_sizes = set()

    # Detectar oracle_sizes
    for rating in rating_conditions:
        rating_path = os.path.join(base_dir, dataset, query_strategy, strategy, rating)
        if not os.path.isdir(rating_path):
            continue
        for oracle_size in os.listdir(rating_path):
            oracle_path = os.path.join(rating_path, oracle_size)
            if os.path.isdir(oracle_path):
                discovered_oracle_sizes.add(oracle_size)

    oracle_sizes = sorted(discovered_oracle_sizes, key=lambda x: int(x))
    col_labels = [f"{sz} ({'W/' if r == 'with_rating' else 'N/'} Rating)"
                  for r in rating_conditions for sz in oracle_sizes]
    row_labels = [f"{e} (W/ Rating)" for e in expertise_levels] + \
                 [f"{e} (N/ Rating)" for e in expertise_levels]

    os.makedirs(save_path, exist_ok=True)

    # Um CSV por métrica
    for metric_name, column_name in metric_columns.items():
        df_out = pd.DataFrame(index=row_labels, columns=col_labels)

        for rating in rating_conditions:
            for expertise in expertise_levels:
                row_label = f"{expertise} ({'W/ Rating' if rating == 'with_rating' else 'N/ Rating'})"

                for oracle_size in oracle_sizes:
                    col_label = f"{oracle_size} ({'W/' if rating == 'with_rating' else 'N/'} Rating)"
                    path = os.path.join(base_dir, dataset, query_strategy, strategy, rating, oracle_size, expertise)
                    values = []

                    if not os.path.isdir(path):
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

                                        if column_name in df.columns:
                                            for val in df[column_name].dropna():
                                                parsed = [float(x) for x in ast.literal_eval(val)]
                                                values.append(np.mean(parsed))
                                    except Exception:
                                        continue

                    if values:
                        mean_val = np.mean(values)
                        std_val = np.std(values)
                        df_out.loc[row_label, col_label] = f"{mean_val:.4f} ± {std_val:.4f}"
                    else:
                        df_out.loc[row_label, col_label] = "-"

        # Salvar CSV final
        csv_filename = f"{filename_prefix}_{metric_name}_table_{dataset}_{strategy}.csv" if filename_prefix else f"{metric_name}_table_{dataset}_{strategy}.csv"
        full_csv_path = os.path.join(save_path, csv_filename)
        df_out.to_csv(full_csv_path)
        print(f"[OK] CSV salvo: {full_csv_path}")


def export_classwise_annotator_stats_to_csv(base_dir: str,
                                            dataset: str,
                                            query_strategy: str,
                                            strategy: str,
                                            oracle_size: str,
                                            save_path: str,
                                            filename: str):
    # Mapear classes
    if dataset == "MNIST":
        classes = CLASSES_MNIST
    elif dataset == "MNIST_FASHION":
        classes = CLASSES_MNIST_FASHION
    elif dataset == "EMNIST_LETTERS":
        classes = CLASSES_EMNIST_LETTERS
    elif dataset == "CIFAR-10":
        classes = CLASSES_CIFAR_10
    else:
        raise ValueError(f"Unknown dataset '{dataset}'")

    num_classes = len(classes)
    expertise_levels = ["H", "M", "L"]
    rating_flags = [WITH_RATING, WITHOUT_RATING, WITH_ONLY_RATING]

    # Estrutura: (expertise, class_idx) -> lista de contagens
    data_by_class = {}

    for expertise in expertise_levels:
        for class_idx in range(num_classes):
            data_by_class[(expertise, class_idx)] = []

        for rating_flag in rating_flags:
            path = os.path.join(base_dir, dataset, query_strategy, strategy, rating_flag, oracle_size, expertise)
            if not os.path.isdir(path):
                print(f"[IGNORADO] Pasta não encontrada: {path}")
                continue

            for folder in os.listdir(path):
                folder_path = os.path.join(path, folder)
                if os.path.isdir(folder_path) and folder.startswith("results_"):
                    txt_file = os.path.join(folder_path, "expert_classes.txt")
                    if os.path.isfile(txt_file):
                        try:
                            with open(txt_file, "r") as f:
                                content = f.read().strip()
                                expert_counts = ast.literal_eval(content)
                                if len(expert_counts) != num_classes:
                                    continue
                                for i, count in enumerate(expert_counts):
                                    data_by_class[(expertise, i)].append(count)
                        except Exception as e:
                            print(f"[ERRO] ao ler {txt_file}: {e}")

    # Gerar CSV com colunas: Class, Expertise, Mean, Std
    os.makedirs(save_path, exist_ok=True)
    csv_path = os.path.join(save_path, filename)
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Class", "Expertise", "Mean #Annotators", "Std"])

        for (expertise, class_idx), values in data_by_class.items():
            class_name = classes[class_idx]
            if values:
                mean_val = np.mean(values)
                std_val = np.std(values)
            else:
                mean_val = "-"
                std_val = "-"
            writer.writerow([class_name, expertise, mean_val, std_val])

    print(f"[OK] Estatísticas por classe salvas em {csv_path}")


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


def plot_all_results(dataset):
    #plot_accuracy_comparative_reputation_based_E(dataset, "uncertainty_sampling", save_path=os.path.join(BASE_DIR, "runs", dataset, "uncertainty_sampling", "reputation_based", "comparative_accuracy_plot.png"))
    #plot_precision_comparative_reputation_based_E(dataset, "uncertainty_sampling", save_path=os.path.join(BASE_DIR, "runs", dataset, "uncertainty_sampling", "reputation_based", "comparative_precision_plot.png"))
    #plot_f1_comparative_reputation_based_E(dataset, "uncertainty_sampling", save_path=os.path.join(BASE_DIR, "runs", dataset, "uncertainty_sampling", "reputation_based", "comparative_f1_plot.png"))


    #plot_accuracy_comparison_strategies(dataset, "uncertainty_sampling", save_path=os.path.join(BASE_DIR, "runs", dataset, "uncertainty_sampling", "comparative_accuracy_plot.png"))
    #plot_precision_comparison_strategies(dataset, "uncertainty_sampling", save_path=os.path.join(BASE_DIR, "runs", dataset, "uncertainty_sampling", "comparative_precision_plot.png"))
    #plot_f1_score_comparison_strategies(dataset, "uncertainty_sampling", save_path=os.path.join(BASE_DIR, "runs", dataset, "uncertainty_sampling", "comparative_f1_plot.png"))


    plot_reputation_per_class(base_path=os.path.join(BASE_DIR, "runs", dataset, "uncertainty_sampling", "reputation_based", "5", "E"),
                              save_path_prefix=os.path.join(BASE_DIR, "runs", dataset, "uncertainty_sampling", "reputation_based", "5", "E", "reputation_per"))
    plot_reputation_per_class(base_path=os.path.join(BASE_DIR, "runs", dataset, "uncertainty_sampling", "reputation_based", "15", "E"),
                              save_path_prefix=os.path.join(BASE_DIR, "runs", dataset, "uncertainty_sampling", "reputation_based", "15", "E", "reputation_per"))
    plot_reputation_per_class(base_path=os.path.join(BASE_DIR, "runs", dataset, "uncertainty_sampling", "reputation_based", "30", "E"),
                              save_path_prefix=os.path.join(BASE_DIR, "runs", dataset, "uncertainty_sampling", "reputation_based", "30", "E", "reputation_per"))








def main():
    #plot_all_results(dataset=DATASET_MNIST_FASHION)
    #plot_all_results(dataset=DATASET_MNIST)
    #plot_all_results(dataset=DATASET_CIFAR_10)
    #plot_all_results(dataset=DATASET_EMNIST_DIGITS)


    for dataset in DATASETS:
        for expertise in EXPERTISES:
            filename_1 = f"{dataset}_{expertise}_rating_comparison.png"
            save_dir = os.path.join(RESULTS_PATH, dataset, "plots", "expertise_comparison")
            compare_rating_effect_by_oracle_size(base_dir=RESULTS_PATH, dataset=dataset, query_strategy="uncertainty_sampling", strategy="reputation_based", expertise=expertise, save_dir=save_dir, filename=filename_1) 

    
    """ for dataset in DATASETS:
        for expertise in EXPERTISES:
            filename_1 = f"{dataset}_all_{expertise}_rating_comparison.png"
            save_dir = os.path.join(RESULTS_PATH, dataset, "plots", "all_expertise_comparison")
            compare_all_rating_effect_by_oracle_size(base_dir=RESULTS_PATH, dataset=dataset, query_strategy="uncertainty_sampling", strategy="reputation_based", expertise=expertise, save_dir=save_dir, filename=filename_1) """ 

    """ for dataset in DATASETS:
        for oracle_size in ORACLE_SIZES:
            filename_1 = f"{dataset}_{str(oracle_size)}_rating_comparison_all.png"
            save_dir = os.path.join(RESULTS_PATH, dataset, "plots", "all_rating_comparison")
            compare_rating_effect_across_expertise(base_dir=RESULTS_PATH, dataset=dataset, query_strategy="uncertainty_sampling", strategy="reputation_based", oracle_size=str(oracle_size), save_dir=save_dir, filename=filename_1)  """



    """ for dataset in DATASETS:
        for oracle_size in ORACLE_SIZES:
            for expertise in EXPERTISES:
                filename_1 = f"{dataset}_{str(oracle_size)}_{expertise}_rating_comparison.png"
                save_dir_rating_comparison = os.path.join(RESULTS_PATH, dataset, "plots", "rating_comparison")
                compare_rating_for_expertise(base_dir=RESULTS_PATH, dataset=dataset, query_strategy="uncertainty_sampling",strategy="reputation_based", oracle_size=str(oracle_size), expertise=expertise, save_dir=save_dir_rating_comparison, filename=filename_1)  """
    
    """ for dataset in DATASETS:
        for oracle_size in ORACLE_SIZES:            
            for within_rating in RATINGS_PERMUTATIONS:
                filename_2 = f"{dataset}_expertise_by_{within_rating}.png"
                save_dir_compare_expertise = os.path.join(RESULTS_PATH, dataset, "plots", "expertise_comparison")
                compare_expertise_within_rating(base_dir=RESULTS_PATH, dataset=dataset, query_strategy="uncertainty_sampling", strategy="reputation_based", oracle_size=str(oracle_size), rating=within_rating,save_dir=save_dir_compare_expertise, filename=filename_2)  """
                
    """ for dataset in DATASETS:
        for oracle_size in ORACLE_SIZES:            
            filename_2 = f"{dataset}_annotators_in_{str(oracle_size)}.png"
            save_dir = os.path.join(RESULTS_PATH, dataset, "plots", "expertise_comparison")
            plot_bar_experts_by_class(base_dir=RESULTS_PATH, dataset=dataset, query_strategy="uncertainty_sampling", strategy="reputation_based", oracle_size=str(oracle_size), save_path=save_dir, filename=filename_2) """
    
    
    """ for dataset in DATASETS:
        save_dir = os.path.join(RESULTS_PATH, dataset, "plots", "tables")
        filename_1 = f"{dataset}_table"
        save_metrics_summary_csv(base_dir=RESULTS_PATH, dataset=dataset, query_strategy="uncertainty_sampling", save_path=save_dir, filename_prefix=filename_1)
    """
    """ for dataset in DATASETS:
        for oracle_size in ORACLE_SIZES:  
            save_dir = os.path.join(RESULTS_PATH, dataset, "plots", "reputation_contributions")
            filename_1 = f"{dataset}_{str(oracle_size)}_reputation.png"    
            plot_reputation_contributions(base_dir=RESULTS_PATH, dataset=dataset, query_strategy="uncertainty_sampling", oracle_size=str(oracle_size), save_path=save_dir, filename=filename_1) """
    
    """  for dataset in DATASETS:
        for oracle_size in ORACLE_SIZES:  
            filename_2 = f"{dataset}_annotators_in_{str(oracle_size)}.csv"
            save_dir = os.path.join(RESULTS_PATH, dataset, "plots", "annotators")
            export_classwise_annotator_stats_to_csv(
                base_dir=RESULTS_PATH,
                dataset=dataset,
                query_strategy="uncertainty_sampling",
                strategy="reputation_based",
                oracle_size=str(oracle_size),
                save_path=save_dir,
                filename=filename_2)
    """


    
if __name__ == "__main__":
    main()