import os
import ast
import numpy as np
from env import *
import matplotlib.pyplot as plt
import ast
import glob
from sklearn.metrics import precision_score, f1_score

CLASSES_MNIST = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
CLASSES_MNIST_FASHION = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
CLASSES_EMNIST_LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
CLASSES_CIFAR_10 = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

DATASET_CLASS_MAP = {
    DATASET_MNIST: CLASSES_MNIST,
    DATASET_MNIST_FASHION: CLASSES_MNIST_FASHION,
    DATASET_EMNIST_LETTERS: CLASSES_EMNIST_LETTERS,
    DATASET_CIFAR_10: CLASSES_CIFAR_10
}

def generate_annotators_table_latex(base_dir: str,
                                     dataset: str,
                                     query_strategy: str,
                                     strategy: str,
                                     oracle_sizes: list,
                                     caption: str,
                                     label: str,
                                     save_path: str,
                                     filename: str):
    class_names = DATASET_CLASS_MAP[dataset]
    num_classes = len(class_names)
    expertise_levels = ["L", "M", "H"]
    rating_flags = ["with_rating", "without_rating"]
    
    stats = {size: {i: {e: [] for e in expertise_levels} for i in range(num_classes)} for size in oracle_sizes}

    for oracle_size in oracle_sizes:
        for rating_flag in rating_flags:
            for expertise in expertise_levels:
                path = os.path.join(base_dir, dataset, query_strategy, strategy, rating_flag, str(oracle_size), expertise)
                if not os.path.isdir(path):
                    continue
                for folder in os.listdir(path):
                    if folder.startswith("results_"):
                        txt_path = os.path.join(path, folder, "expert_classes.txt")
                        if os.path.isfile(txt_path):
                            try:
                                with open(txt_path, "r") as f:
                                    counts = ast.literal_eval(f.read().strip())
                                    for i, val in enumerate(counts):
                                        stats[oracle_size][i][expertise].append(val)
                            except Exception as e:
                                print(f"[ERRO] lendo {txt_path}: {e}")

    header = (
        "\\begin{table}[h!]\n"
        "    \\centering\n"
        "    \\footnotesize\n"
        "    \\renewcommand{\\arraystretch}{1.2}\n"
        "    \\setlength{\\tabcolsep}{8pt}\n"
        "    \\begin{tabular}{l" + "c" * len(oracle_sizes) + "}\n"
        "        \\toprule\n"
        "        & " + " & ".join([f"$\\bm{{|A| = {size}}}$" for size in oracle_sizes]) + " \\\\\n"
        "        \\midrule\n"
        "        \\textbf{Class} & " + " & ".join([""] * len(oracle_sizes)) + " \\\\\n"
    )

    rows = ""
    for i, cls_name in enumerate(class_names):
        row_vals = []
        for oracle_size in oracle_sizes:
            values = []
            for expertise in expertise_levels:
                data = stats[oracle_size][i][expertise]
                if data:
                    values.extend(data)
            if values:
                mean = np.mean(values)
                std = np.std(values)
                formatted = f"{mean:.2f} $\\pm$ {std:.2f}"
            else:
                formatted = "--"
            row_vals.append(formatted)
        rows += f"        \\quad \\textbf{{{cls_name}}} & " + " & ".join(row_vals) + " \\\\\n"

    footer = (
        "        \\bottomrule\n"
        "    \\end{tabular}\n"
        f"    \\caption{{{caption}}}\n"
        f"    \\label{{{label}}}\n"
        "\\end{table}\n"
    )

    table = header + rows + footer

    # Salvar no caminho especificado
    os.makedirs(save_path, exist_ok=True)
    file_path = os.path.join(save_path, filename)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(table)

    print(f"[OK] Tabela LaTeX salva em: {file_path}")



def compute_metrics_from_confusion(conf_matrix):
    if conf_matrix.sum() == 0:
        return 0.0, 0.0, 0.0
    correct = np.trace(conf_matrix)
    total = np.sum(conf_matrix)
    accuracy = correct / total

    true_labels = []
    pred_labels = []
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            true_labels.extend([i] * conf_matrix[i, j])
            pred_labels.extend([j] * conf_matrix[i, j])

    precision = precision_score(true_labels, pred_labels, average='macro', zero_division=0)
    f1 = f1_score(true_labels, pred_labels, average='macro', zero_division=0)
    return accuracy, precision, f1


#===========================================================================
#===========================================================================
#===========================================================================

def compare_rating_conditions_f1(base_dir: str,
                                 dataset: str,
                                 query_strategy: str,
                                 strategy: str,
                                 oracle_size: str,
                                 expertise: str,
                                 confusion_column: str = "confusion_matrix",
                                 results_prefix: str = "results_",
                                 save_dir: str = None,
                                 filename: str = None):
    
    rating_configs = {
        "without_rating": "Only Acc",
        "with_rating": "Both $\mathcal{S}$ and Acc",
        "with_only_rating": "Only $\mathcal{S}$"
    }

    colors = {
        "Only Acc": "blue",
        "Both $\mathcal{S}$ and Acc": "red",
        "Only $\mathcal{S}$": "green",
        "Baseline": "purple"
    }

    def compute_f1(conf_matrix):
        with np.errstate(divide='ignore', invalid='ignore'):
            precision = np.diag(conf_matrix) / np.sum(conf_matrix, axis=0)
            recall = np.diag(conf_matrix) / np.sum(conf_matrix, axis=1)
            f1 = 2 * precision * recall / (precision + recall)
            return np.nanmean(f1)

    def collect_scores(path):
        scores = {}
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
                                score = compute_f1(confusion)
                                scores.setdefault(cycle, []).append(score)
                        except Exception:
                            continue
        return scores

    all_results = {}
    for rating_key, label in rating_configs.items():
        path = os.path.join(base_dir, dataset, query_strategy, strategy, rating_key, oracle_size, expertise)
        if os.path.isdir(path):
            scores = collect_scores(path)
            if scores:
                cycles = sorted(scores.keys())
                means = [np.mean(scores[c]) for c in cycles]
                stds = [np.std(scores[c]) for c in cycles]
                all_results[label] = (cycles, means, stds)

    # Baseline
    baseline_path = os.path.join(base_dir, dataset, "uncertainty_sampling", "random")
    baseline_scores = collect_scores(baseline_path) if os.path.isdir(baseline_path) else {}

    # Plot
    plt.figure(figsize=(8, 5))
    for label, (cycles, means, stds) in all_results.items():
        upper = np.clip(np.array(means) + np.array(stds), 0, 1)
        lower = np.clip(np.array(means) - np.array(stds), 0, 1)
        plt.plot(cycles, means, label=label, linewidth=2, color=colors[label])
        plt.fill_between(cycles, lower, upper, alpha=0.15, color=colors[label])

    if baseline_scores:
        baseline_cycles = sorted(baseline_scores.keys())
        baseline_means = [np.mean(baseline_scores[c]) for c in baseline_cycles]
        baseline_stds = [np.std(baseline_scores[c]) for c in baseline_cycles]
        upper = np.clip(np.array(baseline_means) + np.array(baseline_stds), 0, 1)
        lower = np.clip(np.array(baseline_means) - np.array(baseline_stds), 0, 1)
        plt.plot(baseline_cycles, baseline_means, label="Baseline", linewidth=2, color=colors["Baseline"])
        plt.fill_between(baseline_cycles, lower, upper, alpha=0.15, color=colors["Baseline"])

    plt.xlabel("Cycle")
    plt.ylabel("F1-Score (Avg. ± std)")
    plt.title(f"F1-Score by Rating Condition - Expertise {expertise}, Oracle |A| = {oracle_size}, {dataset}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename or f"f1_comparison_{dataset}_{expertise}_{oracle_size}.png")
        plt.savefig(save_path)
        print(f"[OK] Plot saved at {save_path}")
        plt.close()
    else:
        plt.show()

def compare_rating_conditions_precision(base_dir: str,
                                        dataset: str,
                                        query_strategy: str,
                                        strategy: str,
                                        oracle_size: str,
                                        expertise: str,
                                        confusion_column: str = "confusion_matrix",
                                        results_prefix: str = "results_",
                                        save_dir: str = None,
                                        filename: str = None):
    rating_configs = {
        "without_rating": "Only Acc",
        "with_rating": "Both $\mathcal{S}$ and Acc",
        "with_only_rating": "Only $\mathcal{S}$"
    }

    colors = {
        "Only Acc": "blue",
        "Both $\mathcal{S}$ and Acc": "red",
        "Only $\mathcal{S}$": "green",
        "Baseline": "purple"
    }

    def compute_precision(conf_matrix):
        with np.errstate(divide='ignore', invalid='ignore'):
            precision_per_class = np.diag(conf_matrix) / np.sum(conf_matrix, axis=0)
            return np.nanmean(precision_per_class)

    def collect_scores(path):
        scores = {}
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
                                score = compute_precision(confusion)
                                scores.setdefault(cycle, []).append(score)
                        except Exception:
                            continue
        return scores

    all_results = {}
    for rating_key, label in rating_configs.items():
        path = os.path.join(base_dir, dataset, query_strategy, strategy, rating_key, oracle_size, expertise)
        if os.path.isdir(path):
            scores = collect_scores(path)
            if scores:
                cycles = sorted(scores.keys())
                means = [np.mean(scores[c]) for c in cycles]
                stds = [np.std(scores[c]) for c in cycles]
                all_results[label] = (cycles, means, stds)

    # Baseline
    baseline_path = os.path.join(base_dir, dataset, "uncertainty_sampling", "random")
    baseline_scores = collect_scores(baseline_path) if os.path.isdir(baseline_path) else {}

    # Plot
    plt.figure(figsize=(8, 5))
    for label, (cycles, means, stds) in all_results.items():
        upper = np.clip(np.array(means) + np.array(stds), 0, 1)
        lower = np.clip(np.array(means) - np.array(stds), 0, 1)
        plt.plot(cycles, means, label=label, linewidth=2, color=colors[label])
        plt.fill_between(cycles, lower, upper, alpha=0.15, color=colors[label])

    if baseline_scores:
        baseline_cycles = sorted(baseline_scores.keys())
        baseline_means = [np.mean(baseline_scores[c]) for c in baseline_cycles]
        baseline_stds = [np.std(baseline_scores[c]) for c in baseline_cycles]
        upper = np.clip(np.array(baseline_means) + np.array(baseline_stds), 0, 1)
        lower = np.clip(np.array(baseline_means) - np.array(baseline_stds), 0, 1)
        plt.plot(baseline_cycles, baseline_means, label="Baseline", linewidth=2, color=colors["Baseline"])
        plt.fill_between(baseline_cycles, lower, upper, alpha=0.15, color=colors["Baseline"])

    plt.xlabel("Cycle")
    plt.ylabel("Precision (Avg. ± std)")
    plt.title(f"Precision by Rating Condition - Expertise {expertise}, Oracle |A| = {oracle_size}, {dataset}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename or f"precision_comparison_{dataset}_{expertise}_{oracle_size}.png")
        plt.savefig(save_path)
        print(f"[OK] Plot saved at {save_path}")
        plt.close()
    else:
        plt.show()

def compare_rating_conditions_acc(base_dir: str,
                                         dataset: str,
                                         query_strategy: str,
                                         strategy: str,
                                         oracle_size: str,
                                         expertise: str,
                                         confusion_column: str = "confusion_matrix",
                                         results_prefix: str = "results_",
                                         save_dir: str = None,
                                         filename: str = None):


    rating_configs = {
        "without_rating": "Only Acc",
        "with_rating": "Both $\mathcal{S}$ and Acc",
        "with_only_rating": "Only $\mathcal{S}$"
    }

    colors = {
        "Only Acc": "blue",
        "Both $\mathcal{S}$ and Acc": "red",
        "Only $\mathcal{S}$": "green",
        "Baseline": "purple"
    }

    all_results = {}

    for rating_key, label in rating_configs.items():
        path = os.path.join(base_dir, dataset, query_strategy, strategy, rating_key, oracle_size, expertise)
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
            all_results[label] = (cycles, mean_accuracies, std_accuracies)

    # --- Baseline ---
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

    # --- Plot ---
    plt.figure(figsize=(8, 5))
    for label, (cycles, means, stds) in all_results.items():
        means_array = np.array(means)
        stds_array = np.array(stds)
        upper = np.clip(means_array + stds_array, 0, 1)
        lower = np.clip(means_array - stds_array, 0, 1)
        plt.plot(cycles, means_array, label=label, linewidth=2, color=colors[label])
        plt.fill_between(cycles, lower, upper, alpha=0.15, color=colors[label])

    # Baseline
    if baseline_accuracies:
        baseline_cycles = sorted(baseline_accuracies.keys())
        baseline_means = [np.mean(baseline_accuracies[c]) for c in baseline_cycles]
        baseline_stds = [np.std(baseline_accuracies[c]) for c in baseline_cycles]
        upper = np.clip(np.array(baseline_means) + np.array(baseline_stds), 0, 1)
        lower = np.clip(np.array(baseline_means) - np.array(baseline_stds), 0, 1)
        plt.plot(baseline_cycles, baseline_means, label="Baseline", color=colors["Baseline"], linewidth=2)
        plt.fill_between(baseline_cycles, lower, upper, alpha=0.15, color=colors["Baseline"])

    # Títulos
    plt.xlabel("Cycle")
    plt.ylabel("Accuracy (Avg. ± std)")
    plt.title(f"Accuracy by Rating Condition - Expertise {expertise}, Oracle |A| = {oracle_size}, {dataset}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Salvar ou mostrar
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename or f"rating_comparison_{dataset}_{expertise}_{oracle_size}.png")
        plt.savefig(save_path)
        print(f"[OK] Plot saved at {save_path}")
        plt.close()
    else:
        plt.show()




#===========================================================================
#===========================================================================
#===========================================================================
def compare_avg_reputation_by_oracle_size_cycles(base_dir: str,
                                                          dataset: str,
                                                          query_strategy: str,
                                                          strategy: str,
                                                          rating_condition: str,
                                                          expertise: str,
                                                          save_dir: str = None,
                                                          filename: str = None):
    """
    Plota a média da reputação dos anotadores ao longo dos ciclos (1 ciclo = 16 iterações) para diferentes oracle_sizes,
    mantendo fixos dataset, estratégia, rating_condition e expertise.
    Garante que os valores inferiores não sejam negativos (clipping em 0).
    """

    oracle_sizes = ["5", "15", "30"]
    color_map = {"5": "red", "15": "green", "30": "blue"}

    plt.figure(figsize=(5, 4))

    for oracle_size in oracle_sizes:
        path = os.path.join(base_dir, dataset, query_strategy, strategy, rating_condition, oracle_size, expertise)
        if not os.path.isdir(path):
            print(f"[IGNORADO] Caminho não encontrado: {path}")
            continue

        reputations_by_cycle = {}

        for folder in os.listdir(path):
            folder_path = os.path.join(path, folder)
            annotator_dir = os.path.join(folder_path, "Annotators")
            if os.path.isdir(annotator_dir):
                for file in os.listdir(annotator_dir):
                    if file.startswith("Annotator_") and file.endswith(".csv"):
                        file_path = os.path.join(annotator_dir, file)
                        try:
                            df = pd.read_csv(file_path)
                            if "reputations" in df.columns:
                                rep_series = df["reputations"].apply(eval)
                                for i, rep in enumerate(rep_series):
                                    cycle_index = i // 16
                                    reputations_by_cycle.setdefault(cycle_index, []).append(np.mean(rep))
                        except Exception:
                            continue

        if not reputations_by_cycle:
            print(f"[AVISO] Nenhuma reputação encontrada para oracle_size={oracle_size}")
            continue

        cycles = sorted(reputations_by_cycle.keys())
        avg_rep = [np.mean(reputations_by_cycle[c]) for c in cycles]
        std_rep = [np.std(reputations_by_cycle[c]) for c in cycles]

        avg_array = np.array(avg_rep)
        std_array = np.array(std_rep)

        lower = np.clip(avg_array - std_array, 0, None)
        upper = avg_array + std_array

        plt.plot(cycles, avg_array, label=f"$\\mathbf{{|A| = {oracle_size}}}$",
                 color=color_map[oracle_size], linewidth=2)
        plt.fill_between(cycles, lower, upper, color=color_map[oracle_size], alpha=0.2)

    plt.xlabel("Cycle")
    plt.ylabel("Avg. Reputation (± std)")
    plt.title(f"Avg. Annotator Reputation over Cycles\n{dataset}, Expertise {expertise}, {rating_condition}")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename or f"avg_reputation_cycles_{dataset}_{expertise}.png")
        plt.savefig(save_path)
        print(f"[OK] Plot saved at {save_path}")
        plt.close()
    else:
        plt.show()


def compare_avg_reputation_by_expertise_cycles(base_dir: str,
                                                dataset: str,
                                                query_strategy: str,
                                                strategy: str,
                                                rating_condition: str,
                                                save_dir: str = None,
                                                filename: str = None):
    """
    Plota a média da reputação dos anotadores ao longo dos ciclos (1 ciclo = 16 iterações) para diferentes níveis de expertise (H, M, L),
    fazendo a média dos valores de reputação considerando todos os oracle_sizes.
    Garante que os valores inferiores não sejam negativos (clipping em 0).
    """

    expertises = ["H", "M", "L"]
    oracle_sizes = ["5", "15", "30"]
    color_map = {"H": "red", "M": "blue", "L": "green"}

    plt.figure(figsize=(5, 4))

    for expertise in expertises:
        reputations_by_cycle = {}

        for oracle_size in oracle_sizes:
            path = os.path.join(base_dir, dataset, query_strategy, strategy, rating_condition, oracle_size, expertise)
            if not os.path.isdir(path):
                print(f"[IGNORADO] Caminho não encontrado: {path}")
                continue

            for folder in os.listdir(path):
                folder_path = os.path.join(path, folder)
                annotator_dir = os.path.join(folder_path, "Annotators")
                if os.path.isdir(annotator_dir):
                    for file in os.listdir(annotator_dir):
                        if file.startswith("Annotator_") and file.endswith(".csv"):
                            file_path = os.path.join(annotator_dir, file)
                            try:
                                df = pd.read_csv(file_path)
                                if "reputations" in df.columns:
                                    rep_series = df["reputations"].apply(eval)
                                    for i, rep in enumerate(rep_series):
                                        cycle_index = i // 16
                                        reputations_by_cycle.setdefault(cycle_index, []).append(np.mean(rep))
                            except Exception:
                                continue

        if not reputations_by_cycle:
            print(f"[AVISO] Nenhuma reputação encontrada para expertise={expertise}")
            continue

        cycles = sorted(reputations_by_cycle.keys())
        avg_rep = [np.mean(reputations_by_cycle[c]) for c in cycles]
        std_rep = [np.std(reputations_by_cycle[c]) for c in cycles]

        avg_array = np.array(avg_rep)
        std_array = np.array(std_rep)

        lower = np.clip(avg_array - std_array, 0, None)
        upper = avg_array + std_array

        plt.plot(cycles, avg_array, label=f"$\\mathbf{{{expertise}}}$",
                 color=color_map[expertise], linewidth=2)
        plt.fill_between(cycles, lower, upper, color=color_map[expertise], alpha=0.2)

    plt.xlabel("Cycle")
    plt.ylabel("Avg. Reputation (± std)")
    plt.title(f"Avg. Annotator Reputation over Cycles\n{dataset}, {rating_condition}")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(title="Expertise")
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename or f"avg_reputation_cycles_{dataset}_{rating_condition}.png")
        plt.savefig(save_path)
        print(f"[OK] Plot saved at {save_path}")
        plt.close()
    else:
        plt.show()



WITHOUT_RATING = "without_rating"
WITH_RATING = "with_rating"
WITH_ONLY_RATING = "with_only_rating"

RATINGS_PERMUTATIONS = [WITH_ONLY_RATING, WITH_RATING, WITHOUT_RATING]

LATEX_LABELS = {
    WITH_ONLY_RATING: "Only $\\mathcal{S}$",
    WITH_RATING: "Both $\\mathcal{S}$ and \\textit{Acc}",
    WITHOUT_RATING: "Only \\textit{Acc}"
}

def generate_latex_reputation_table_by_configuration(base_dir: str,
                                                      dataset: str,
                                                      query_strategy: str,
                                                      strategy: str,
                                                      save_path: str,
                                                      filename: str):
    """
    Gera e salva uma tabela LaTeX com reputações finais médias (± std) para cada configuração de reputação,
    por nível de expertise (L, M, H) e tamanho do oráculo (|A| = 5, 15, 30).
    Lê apenas o último valor válido do campo 'reputations' em cada ficheiro.
    """
    oracle_sizes = ["5", "15", "30"]
    expertises = ["L", "M", "H"]

    def extract_final_reputation(file_path):
        try:
            df = pd.read_csv(file_path)
            if "reputations" in df.columns:
                # Lê de trás para frente até achar um valor não nulo
                for val in reversed(df["reputations"].dropna()):
                    try:
                        rep = eval(val)
                        return np.mean(rep)
                    except:
                        continue
        except:
            pass
        return None

    def compute_mean_std(path):
        if not os.path.isdir(path):
            return None, None
        final_reputations = []
        for folder in os.listdir(path):
            folder_path = os.path.join(path, folder)
            annotator_dir = os.path.join(folder_path, "Annotators")
            if os.path.isdir(annotator_dir):
                for file in os.listdir(annotator_dir):
                    if file.startswith("Annotator_") and file.endswith(".csv"):
                        rep = extract_final_reputation(os.path.join(annotator_dir, file))
                        if rep is not None:
                            final_reputations.append(rep)
        if final_reputations:
            return np.mean(final_reputations), np.std(final_reputations)
        return None, None

    lines = [
        "\\begin{table}[h!]",
        "    \\centering",
        "    \\footnotesize",
        "    \\renewcommand{\\arraystretch}{1.2}",
        "    \\setlength{\\tabcolsep}{8pt}",
        "    \\begin{tabular}{lccc}",
        "        \\toprule",
        "        & $\\bm{|A| = 5}$ & $\\bm{|A| = 15}$ & $\\bm{|A| = 30}$ \\\\",
        "        \\midrule"
    ]

    for rating in RATINGS_PERMUTATIONS:
        lines.append(f"        \\textbf{{{LATEX_LABELS[rating]}}} &  &  &  \\\\")
        for expertise in expertises:
            row_vals = []
            for oracle_size in oracle_sizes:
                path = os.path.join(base_dir, dataset, query_strategy, strategy, rating, oracle_size, expertise)
                mean, std = compute_mean_std(path)
                if mean is not None:
                    row_vals.append(f"{mean:.4f} ± {std:.4f}")
                else:
                    row_vals.append("-")
            # Negrito no maior valor da linha
            try:
                numeric_vals = [float(val.split(" ± ")[0]) for val in row_vals if "±" in val]
                if len(numeric_vals) == 3:
                    max_idx = int(np.argmax(numeric_vals))
                    row_vals[max_idx] = f"\\textbf{{{row_vals[max_idx]}}}"
            except:
                pass
            lines.append(f"        \\quad {expertise} & " + " & ".join(row_vals) + " \\\\")
        lines.append("        \\addlinespace")

    lines.append("        \\bottomrule")
    lines.append("    \\end{tabular}")
    lines.append(f"    \\caption{{Reputation metrics for reputation configurations on \\textit{{{dataset}}}.}}")
    lines.append(f"    \\label{{tab:{dataset.lower()}_reputation}}")
    lines.append("\\end{table}")

    os.makedirs(save_path, exist_ok=True)
    full_path = os.path.join(save_path, filename)
    with open(full_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[OK] Tabela LaTeX salva em: {full_path}")

#===========================================================================
#===========================================================================
#===========================================================================
def plot_violin_avg_annotator_accuracy(base_dir: str,
                                       dataset: str,
                                       query_strategy: str,
                                       strategy: str,
                                       oracle_size: str,
                                       save_path: str,
                                       filename: str):
    expertise_levels = ["H", "M", "L"]
    rating_flags = ["with_rating", "without_rating", "with_only_rating"]
    color_map = {"H": "tab:red", "M": "tab:blue", "L": "tab:green"}

    accuracy_data = {level: [] for level in expertise_levels}

    for rating_flag in rating_flags:
        for expertise in expertise_levels:
            path = os.path.join(base_dir, dataset, query_strategy, strategy, rating_flag, oracle_size, expertise)
            if not os.path.isdir(path):
                print(f"[IGNORADO] Pasta não encontrada: {path}")
                continue

            results_dirs = glob.glob(os.path.join(path, "results_*"))
            for result_dir in results_dirs:
                annotator_files = glob.glob(os.path.join(result_dir, "Annotators", "Annotator_*.csv"))
                for annot_file in annotator_files:
                    try:
                        df = pd.read_csv(annot_file)
                        if df.empty or "accuracies" not in df.columns:
                            continue
                        last_row = df.iloc[-1]
                        accuracies = ast.literal_eval(last_row["accuracies"])
                        if not accuracies or not isinstance(accuracies, list):
                            continue
                        avg_acc = np.mean(accuracies)
                        accuracy_data[expertise].append(avg_acc)
                    except Exception as e:
                        print(f"[ERRO] ao processar {annot_file}: {e}")

    # --- Violin Plot ---
    plt.figure(figsize=(4, 4))
    data = [accuracy_data[level] for level in expertise_levels]
    parts = plt.violinplot(data, showmeans=True, showextrema=False, showmedians=False)

    for pc, level in zip(parts['bodies'], expertise_levels):
        pc.set_facecolor(color_map[level])
        pc.set_edgecolor('black')
        pc.set_alpha(0.6)

    # Pontos individuais com jitter
    for i, level in enumerate(expertise_levels):
        x_jitter = np.random.normal(i + 1, 0.05, size=len(accuracy_data[level]))
        plt.scatter(x_jitter, accuracy_data[level], alpha=0.6, color=color_map[level], s=20)

    plt.xticks(ticks=[1, 2, 3], labels=expertise_levels)
    plt.ylabel("Average Accuracy")
    plt.title(f"Annotator Avg. Accuracy - {dataset.upper()}, Oracle {oracle_size}")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()

    os.makedirs(save_path, exist_ok=True)
    plot_path = os.path.join(save_path, filename)
    plt.savefig(plot_path)
    print(f"[OK] Violin plot saved to {plot_path}")
    plt.close()


#===========================================================================
#===========================================================================
#===========================================================================
def compare_oracle_sizes_precision_averaging_ratings(
    base_dir: str,
    dataset: str,
    query_strategy: str,
    strategy: str,
    expertise: str,
    confusion_column: str = "confusion_matrix",
    results_prefix: str = "results_",
    save_dir: str = None,
    filename: str = None
):
    """
    Compara diferentes tamanhos de oráculo (|A|) para uma expertise fixa,
    calculando a média da precision entre todas as condições de rating.

    :param base_dir: Caminho base dos resultados
    :param dataset: Nome do dataset
    :param query_strategy: Estratégia de query
    :param strategy: Estratégia usada (ex: reputation)
    :param expertise: Expertise fixa ("L", "M", "H")
    :param confusion_column: Nome da coluna com matriz de confusão
    :param results_prefix: Prefixo das pastas
    :param save_dir: Diretório de saída
    :param filename: Nome do arquivo de saída
    """

    oracle_sizes = ["5", "15", "30"]
    rating_conditions = ["with_rating", "without_rating", "with_only_rating"]
    color_map = {"5": "red", "15": "green", "30": "blue"}

    all_results = {}

    for oracle_size in oracle_sizes:
        combined_precisions = {}

        for rating in rating_conditions:
            path = os.path.join(base_dir, dataset, query_strategy, strategy, rating, oracle_size, expertise)
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
                                    TP = np.diag(confusion)
                                    FP = np.sum(confusion, axis=0) - TP
                                    with np.errstate(divide='ignore', invalid='ignore'):
                                        precision = np.where((TP + FP) > 0, TP / (TP + FP), 0.0)
                                    mean_precision = np.nanmean(precision)
                                    combined_precisions.setdefault(cycle, []).append(mean_precision)
                            except Exception:
                                continue

        if combined_precisions:
            cycles = sorted(combined_precisions.keys())
            mean_values = [np.mean(combined_precisions[c]) for c in cycles]
            std_values = [np.std(combined_precisions[c]) for c in cycles]
            all_results[oracle_size] = {
                "cycles": cycles,
                "means": mean_values,
                "stds": std_values
            }

    # --- Baseline ---
    baseline_precisions = {}
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
                                TP = np.diag(confusion)
                                FP = np.sum(confusion, axis=0) - TP
                                with np.errstate(divide='ignore', invalid='ignore'):
                                    precision = np.where((TP + FP) > 0, TP / (TP + FP), 0.0)
                                mean_precision = np.nanmean(precision)
                                baseline_precisions.setdefault(cycle, []).append(mean_precision)
                        except Exception:
                            continue

    # --- Plotagem ---
    plt.figure(figsize=(8, 5))

    for oracle_size, data in all_results.items():
        means = np.array(data["means"])
        stds = np.array(data["stds"])
        upper = np.clip(means + stds, 0, 1)
        lower = np.clip(means - stds, 0, 1)
        label = f"$\\mathbf{{|A| = {oracle_size}}}$"
        color = color_map[oracle_size]

        plt.plot(data["cycles"], means, label=label, linewidth=2, color=color)
        plt.fill_between(data["cycles"], lower, upper, alpha=0.15, color=color)

    if baseline_precisions:
        baseline_cycles = sorted(baseline_precisions.keys())
        means = [np.mean(baseline_precisions[c]) for c in baseline_cycles]
        stds = [np.std(baseline_precisions[c]) for c in baseline_cycles]
        upper = np.clip(np.array(means) + np.array(stds), 0, 1)
        lower = np.clip(np.array(means) - np.array(stds), 0, 1)
        plt.plot(baseline_cycles, means, label="Baseline", color="purple", linewidth=2)
        plt.fill_between(baseline_cycles, lower, upper, alpha=0.15, color="purple")

    plt.xlabel("Cycle")
    plt.ylabel("Precision (Avg. ± std)")
    plt.title(f"Precision - Oracle Size Comparison (Averaged Ratings) - Expertise {expertise}, {dataset}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename or f"precision_oracle_size_avg_rating_{dataset}_{expertise}.png")
        plt.savefig(save_path)
        print(f"[OK] Plot saved at {save_path}")
        plt.close()
    else:
        plt.show()


def compare_oracle_sizes_f1_averaging_ratings(
    base_dir: str,
    dataset: str,
    query_strategy: str,
    strategy: str,
    expertise: str,
    confusion_column: str = "confusion_matrix",
    results_prefix: str = "results_",
    save_dir: str = None,
    filename: str = None
):
    """
    Compara diferentes tamanhos de oráculo (|A|) para uma expertise fixa,
    calculando a média do F1-score (macro) entre todas as condições de rating.

    :param base_dir: Caminho base dos resultados
    :param dataset: Nome do dataset
    :param query_strategy: Estratégia de query
    :param strategy: Estratégia usada (ex: reputation)
    :param expertise: Expertise fixa ("L", "M", "H")
    :param confusion_column: Nome da coluna com matriz de confusão
    :param results_prefix: Prefixo das pastas
    :param save_dir: Diretório de saída
    :param filename: Nome do arquivo de saída
    """

    oracle_sizes = ["5", "15", "30"]
    rating_conditions = ["with_rating", "without_rating", "with_only_rating"]
    color_map = {"5": "red", "15": "green", "30": "blue"}

    all_results = {}

    for oracle_size in oracle_sizes:
        combined_f1 = {}

        for rating in rating_conditions:
            path = os.path.join(base_dir, dataset, query_strategy, strategy, rating, oracle_size, expertise)
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
                                    TP = np.diag(confusion)
                                    FP = np.sum(confusion, axis=0) - TP
                                    FN = np.sum(confusion, axis=1) - TP

                                    with np.errstate(divide='ignore', invalid='ignore'):
                                        precision = np.where((TP + FP) > 0, TP / (TP + FP), 0.0)
                                        recall = np.where((TP + FN) > 0, TP / (TP + FN), 0.0)
                                        f1 = np.where((precision + recall) > 0, 2 * precision * recall / (precision + recall), 0.0)

                                    mean_f1 = np.nanmean(f1)
                                    combined_f1.setdefault(cycle, []).append(mean_f1)
                            except Exception:
                                continue

        if combined_f1:
            cycles = sorted(combined_f1.keys())
            mean_values = [np.mean(combined_f1[c]) for c in cycles]
            std_values = [np.std(combined_f1[c]) for c in cycles]
            all_results[oracle_size] = {
                "cycles": cycles,
                "means": mean_values,
                "stds": std_values
            }

    # --- Baseline ---
    baseline_f1 = {}
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
                                TP = np.diag(confusion)
                                FP = np.sum(confusion, axis=0) - TP
                                FN = np.sum(confusion, axis=1) - TP

                                with np.errstate(divide='ignore', invalid='ignore'):
                                    precision = np.where((TP + FP) > 0, TP / (TP + FP), 0.0)
                                    recall = np.where((TP + FN) > 0, TP / (TP + FN), 0.0)
                                    f1 = np.where((precision + recall) > 0, 2 * precision * recall / (precision + recall), 0.0)

                                mean_f1 = np.nanmean(f1)
                                baseline_f1.setdefault(cycle, []).append(mean_f1)
                        except Exception:
                            continue

    # --- Plotagem ---
    plt.figure(figsize=(8, 5))

    for oracle_size, data in all_results.items():
        means = np.array(data["means"])
        stds = np.array(data["stds"])
        upper = np.clip(means + stds, 0, 1)
        lower = np.clip(means - stds, 0, 1)
        label = f"$\\mathbf{{|A| = {oracle_size}}}$"
        color = color_map[oracle_size]

        plt.plot(data["cycles"], means, label=label, linewidth=2, color=color)
        plt.fill_between(data["cycles"], lower, upper, alpha=0.15, color=color)

    if baseline_f1:
        baseline_cycles = sorted(baseline_f1.keys())
        means = [np.mean(baseline_f1[c]) for c in baseline_cycles]
        stds = [np.std(baseline_f1[c]) for c in baseline_cycles]
        upper = np.clip(np.array(means) + np.array(stds), 0, 1)
        lower = np.clip(np.array(means) - np.array(stds), 0, 1)
        plt.plot(baseline_cycles, means, label="Baseline", color="purple", linewidth=2)
        plt.fill_between(baseline_cycles, lower, upper, alpha=0.15, color="purple")

    plt.xlabel("Cycle")
    plt.ylabel("F1-score (Avg. ± std)")
    plt.title(f"F1-score - Oracle Size Comparison (Averaged Ratings) - Expertise {expertise}, {dataset}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename or f"f1_oracle_size_avg_rating_{dataset}_{expertise}.png")
        plt.savefig(save_path)
        print(f"[OK] Plot saved at {save_path}")
        plt.close()
    else:
        plt.show()










def plot_expertise_comparison():
    for dataset in DATASETS:
        for oracle_size in ORACLE_SIZES:
            for expertise in EXPERTISES:
                save_dir = os.path.join(RESULTS_PATH, dataset, "plots", "expertise_comparison")
                compare_rating_conditions_acc(
                    base_dir=RESULTS_PATH,
                    dataset=dataset,
                    query_strategy="uncertainty_sampling",
                    strategy="reputation_based",
                    oracle_size=str(oracle_size),
                    expertise=expertise,
                    save_dir=save_dir,
                    filename=f"acc_compare_ratings_{expertise}_{str(oracle_size)}.png"
                )
                compare_rating_conditions_precision(
                    base_dir=RESULTS_PATH,
                    dataset=dataset,
                    query_strategy="uncertainty_sampling",
                    strategy="reputation_based",
                    oracle_size=str(oracle_size),
                    expertise=expertise,
                    save_dir=save_dir,
                    filename=f"pre_compare_ratings_{expertise}_{str(oracle_size)}.png"
                )
                compare_rating_conditions_f1(
                    base_dir=RESULTS_PATH,
                    dataset=dataset,
                    query_strategy="uncertainty_sampling",
                    strategy="reputation_based",
                    oracle_size=str(oracle_size),
                    expertise=expertise,
                    save_dir=save_dir,
                    filename=f"f1_compare_ratings_{expertise}_{str(oracle_size)}.png"
                )

def table_number_annotators():
    for dataset in DATASETS:
        save_dir = os.path.join(RESULTS_PATH, dataset, "plots", "tables")
        latex = generate_annotators_table_latex(
            base_dir=RESULTS_PATH,
            dataset=dataset,  # ou "fashion", "emnist", "cifar10"
            query_strategy="uncertainty_sampling",
            strategy="reputation_based",
            oracle_sizes=ORACLE_SIZES,
            caption=f"Average number of annotators per class on {dataset}.",
            label=f"tab:annotator_counts_{dataset}",
            save_path=save_dir,
            filename=f"avg_n_annotators_per_class_{dataset}.tex"
        )

        print(latex)

def reputation_comparison_per_oracle():
   
    for dataset in DATASETS: 
        for rating_flag in RATINGS_PERMUTATIONS:
            save_dir = os.path.join(RESULTS_PATH, dataset, "plots", "reputation_comparison") 
            compare_avg_reputation_by_expertise_cycles(base_dir=RESULTS_PATH,
                                                    dataset=dataset,
                                                    query_strategy="uncertainty_sampling",
                                                    strategy="reputation_based",
                                                    rating_condition=rating_flag,
                                                    save_dir=save_dir,
                                                    filename=f"avg_reputation_{rating_flag}.png")

def table_reputation_comparison_per_oracle():
    for dataset in DATASETS:
        save_dir = os.path.join(RESULTS_PATH, dataset, "plots", "tables")
        filename = f"reputation_table_{dataset.lower()}.tex"
        generate_latex_reputation_table_by_configuration(
            base_dir=RESULTS_PATH,
            dataset=dataset,
            query_strategy="uncertainty_sampling",
            strategy="reputation_based",
            save_path=save_dir,
            filename=filename
        )


def violin_metrics_annotators():
    for dataset in DATASETS:
        for oracle_size in ORACLE_SIZES:
                save_dir = os.path.join(RESULTS_PATH, dataset, "plots", "all_metrics_boxplots")
                plot_violin_avg_annotator_accuracy(
                    base_dir=RESULTS_PATH,
                    dataset=dataset,
                    query_strategy="uncertainty_sampling",
                    strategy="reputation_based",
                    oracle_size=str(oracle_size),
                    save_path=save_dir,
                    filename=f"{dataset}_{oracle_size}_annotator_accuracy_.png"
                )



def oracle_effect():
    for dataset in DATASETS:
        for expertise in EXPERTISES:
            save_dir = os.path.join(RESULTS_PATH, dataset, "plots", "expertise_comparison")
            compare_oracle_sizes_precision_averaging_ratings(
                base_dir=RESULTS_PATH,
                dataset=dataset,
                query_strategy="uncertainty_sampling",
                strategy="reputation_based",
                expertise=expertise,
                save_dir=save_dir,
                filename=f"pre_oracle_size_effect_{expertise}.png"
            )

            compare_oracle_sizes_f1_averaging_ratings(
                base_dir=RESULTS_PATH,
                dataset=dataset,
                query_strategy="uncertainty_sampling",
                strategy="reputation_based",
                expertise=expertise,
                save_dir=save_dir,
                filename=f"f1_oracle_size_effect_{expertise}.png"
            )



if __name__ == "__main__":
    #plot_expertise_comparison()
    #table_reputation_comparison_per_oracle()
    #violin_metrics_annotators()
    reputation_comparison_per_oracle()
    #oracle_effect()
