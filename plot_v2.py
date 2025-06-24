import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from env import *
import ast
import seaborn as sns
from entities.Committee import Committee

def parse_metric_column(column):
    """Converte a string de lista para uma lista de floats e retorna a média"""
    try:
        values = eval(column)
        return np.mean([float(x) for x in values])
    except Exception:
        return np.nan

def load_results(base_path_pattern, metric):
    all_runs = glob.glob(base_path_pattern)
    results_by_cycle = {}

    for run_file in all_runs:
        df = pd.read_csv(run_file, sep='\t' if '\t' in open(run_file).readline() else ',')
        for _, row in df.iterrows():
            cycle = int(row['cycle'])
            if cycle not in results_by_cycle:
                results_by_cycle[cycle] = []
            col_name = {
                'accuracy': 'accuracy_per_class',
                'precision': 'precision_per_class',
                'f1-score': 'f1_score_per_class'
            }[metric]
            results_by_cycle[cycle].append(parse_metric_column(row[col_name]))

    cycles = sorted(results_by_cycle.keys())
    means = [np.nanmean(results_by_cycle[c]) for c in cycles]
    stds = [np.nanstd(results_by_cycle[c]) for c in cycles]
    return cycles, means, stds

def load_metric_across_runs(path_pattern, metric):
    """
    Lê todos os arquivos CSV que combinam com o path, extrai a métrica média por classe,
    e retorna a média e desvio padrão final (entre execuções).
    """
    all_runs = glob.glob(path_pattern)
    per_run_means = []

    for run_file in all_runs:
        df = pd.read_csv(run_file, sep='\t' if '\t' in open(run_file).readline() else ',')
        last_row = df.iloc[-1]
        col_name = {
            'accuracy': 'accuracy_per_class',
            'precision': 'precision_per_class',
            'f1-score': 'f1_score_per_class'
        }[metric]

        try:
            class_values = eval(last_row[col_name])  # lista de strings
            class_floats = [float(x) for x in class_values]
            mean_per_class = np.mean(class_floats)
            per_run_means.append(mean_per_class)
        except Exception:
            continue

    if not per_run_means:
        return np.nan, np.nan

    return np.mean(per_run_means), np.std(per_run_means)



def format_entry(mean, std, best=False):
    value = f"{mean:.4f} ± {std:.4f}"
    return f"\\textbf{{{value}}}" if best else value


CLASS_NAMES = {
    "MNIST": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
    "MNIST_FASHION": ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                      "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"],
    "EMNIST_LETTERS": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
    "CIFAR-10": ["airplane", "automobile", "bird", "cat", "deer",
                "dog", "frog", "horse", "ship", "truck"]
}

def plot_expected_accuracy_by_expert_class(annotators, dataset_name, expertise, save_path=None, filename=None):

    expertise_levels = {"L": 0.40, "M": 0.70, "H": 0.98}
    if expertise not in expertise_levels:
        raise ValueError(f"Invalid expertise level '{expertise}'. Choose from {list(expertise_levels.keys())}.")
    target_acc = expertise_levels[expertise]

    dataset_key = dataset_name.upper()
    if dataset_key not in CLASS_NAMES:
        raise ValueError(f"Dataset '{dataset_name}' não reconhecido. Use um de: {list(CLASS_NAMES.keys())}")

    class_labels = CLASS_NAMES[dataset_key]
    num_classes = len(class_labels)

    accuracy_by_class = {i: [] for i in range(num_classes)}

    for ann in annotators:
        expert_class = ann.expert_class
        if 0 <= expert_class < num_classes:
            expected_accuracy = ann.cm_prob[expert_class][expert_class]
            accuracy_by_class[expert_class].append(expected_accuracy)
        else:
            print(f"[IGNORADO] expert_class inválida: {expert_class}")

    # Reestruturar dados
    data = []
    for i in range(num_classes):
        for acc in accuracy_by_class[i]:
            data.append((class_labels[i], acc))

    if not data:
        print("[AVISO] Nenhum dado válido para plotar.")
        return

    df = pd.DataFrame(data, columns=["Expert Class", "Expected Accuracy"])

    # Plot
    plt.figure(figsize=(10, 5))
    sns.violinplot(data=df, x="Expert Class", y="Expected Accuracy",
                   palette="pastel", inner=None, cut=0)
    sns.stripplot(data=df, x="Expert Class", y="Expected Accuracy",
                  color='black', size=3, jitter=True, alpha=0.6)

    # Linha do target acc
    plt.axhline(y=target_acc, color="red", linestyle="--", linewidth=1.2, label=f"Target acc$_t$ = {target_acc}")
    plt.legend()

    plt.title(f"Expected Accuracy per Expert Class - {dataset_name.upper()} ({expertise}-level)")
    plt.xlabel("Expert Class")
    plt.ylabel("Expected Accuracy")
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()

    if save_path and filename:
        os.makedirs(save_path, exist_ok=True)
        plot_path = os.path.join(save_path, filename)
        plt.savefig(plot_path)
        print(f"[OK] Plot saved to {plot_path}")
    else:
        plt.show()

    plt.close()



def generate_latex_table(dataset, query_strategy, rating_type, oracle_sizes, expertises, caption, label):
    metrics = ['accuracy', 'precision', 'f1-score']
    latex = []
    latex.append("\\begin{table}[h!]")
    latex.append("    \\centering")
    latex.append("    \\footnotesize")
    latex.append("    \\renewcommand{\\arraystretch}{1.2}")
    latex.append("    \\setlength{\\tabcolsep}{8pt}")
    latex.append("    \\begin{tabular}{l" + "c" * len(oracle_sizes) + "}")
    latex.append("        \\toprule")
    header = "        & " + " & ".join([f"$\\bm{{|A| = {a}}}$" for a in oracle_sizes]) + " \\\\"
    latex.append(header)
    latex.append("        \\midrule")

    for metric in metrics:
        latex.append(f"        \\textbf{{{metric.capitalize()}}} & " + " & ".join([""] * len(oracle_sizes)) + " \\\\")
        
        # Baseline
        path_pattern_baseline = f"runs/{dataset}/{query_strategy}/random/results_*/results_*.csv"
        mean, std = load_metric_across_runs(path_pattern_baseline, metric)
        if np.isnan(mean):
            baseline_row = ["--"] * len(oracle_sizes)
        else:
            entry = f"{mean:.4f} ± {std:.4f}"
            baseline_row = [entry] * len(oracle_sizes)
        latex.append("        \\quad Baseline & " + " & ".join(baseline_row) + " \\\\")

        # Expertises
        for exp in expertises:
            row = [f"        \\quad {exp}"]
            for size in oracle_sizes:
                path_pattern = (
                    f"runs/{dataset}/{query_strategy}/reputation_based/{rating_type}/"
                    f"{size}/{exp}/results_*/results_*.csv"
                )
                mean, std = load_metric_across_runs(path_pattern, metric)
                row.append((mean, std))

            values = [m for m, s in row[1:]]
            if all(np.isnan(v) for v in values):
                formatted = ["--"] * len(values)
            else:
                best_idx = int(np.nanargmax(values))
                formatted = [
                    format_entry(m, s, best=(i == best_idx)) if not np.isnan(m) else "--"
                    for i, (m, s) in enumerate(row[1:])
                ]
            latex.append("        " + row[0] + " & " + " & ".join(formatted) + " \\\\")

        latex.append("        \\addlinespace")

    latex.append("        \\bottomrule")
    latex.append("    \\end{tabular}")
    latex.append(f"    \\caption{{{caption}}}")
    latex.append(f"    \\label{{{label}}}")
    latex.append("\\end{table}")
    return "\n".join(latex)





def plot_metric_over_cycles(dataset, query_strategy, oracle_size, expertise, metric='accuracy', save_dir=None, filename=None):
    configs = {
        "Only $\mathcal{S}$": ("with_only_rating", 'red'),
        "Both $\mathcal{S}$ and Acc": ("with_rating", None),
        "Only Acc": ("without_rating", None),
        "Baseline": (None, 'purple')
    }

    plt.figure(figsize=(10, 6))

    for label, (rating, forced_color) in configs.items():
        if rating is None:
            path_pattern = f"runs/{dataset}/{query_strategy}/random/results_*/results_*.csv"
        else:
            path_pattern = (
                f"runs/{dataset}/{query_strategy}/reputation_based/{rating}/"
                f"{oracle_size}/{expertise}/results_*/results_*.csv"
            )

        cycles, means, stds = load_results(path_pattern, metric)

        # Define cor
        color = forced_color

        # Plot linha contínua sem pontos
        line, = plt.plot(cycles, means, label=label, linestyle='-', linewidth=2, color=color)

        # Usa a mesma cor da linha para o sombreamento
        fill_color = line.get_color()
        lower = np.array(means) - np.array(stds)
        upper = np.array(means) + np.array(stds)
        plt.fill_between(cycles, lower, upper, alpha=0.2, color=fill_color)

    plt.title(f"{metric.capitalize()} over AL Cycles")
    plt.xlabel("Cycles")
    plt.ylabel(metric.capitalize())
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_dir and filename:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, filename)
        plt.savefig(path, dpi=300)
        print(f"Plot saved to: {path}")
    else:
        plt.show()


def plot_boxplot_annotator_accuracy_top_class(base_dir: str,
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
                        top_class_acc = max(accuracies)
                        accuracy_data[expertise].append(top_class_acc)
                    except Exception as e:
                        print(f"[ERRO] ao processar {annot_file}: {e}")

    # --- Boxplot ---
    plt.figure(figsize=(6, 4))  # menor largura para menos espaçamento
    data = [accuracy_data[level] for level in expertise_levels]
    plt.boxplot(data, labels=expertise_levels, patch_artist=True,
                boxprops=dict(facecolor='lightgray'),
                medianprops=dict(color='black'),
                whiskerprops=dict(color='black'),
                capprops=dict(color='black'),
                flierprops=dict(markerfacecolor='gray', marker='o', markersize=4, linestyle='none'))

    for i, level in enumerate(expertise_levels):
        plt.scatter(np.random.normal(i + 1, 0.02, size=len(accuracy_data[level])),  # jitter menor
                    accuracy_data[level], alpha=0.6, color=color_map[level], label=f"{level} annotators")   

    plt.ylabel("Avg. Accuracy")
    plt.title(f"Expert Accuracy on Best Class - {dataset.upper()}, Oracle {oracle_size}")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    plot_path = os.path.join(save_path, filename)
    plt.savefig(plot_path)
    print(f"[OK] Boxplot saved to {plot_path}")
    plt.close()




def compare_oracle_sizes_averaging_ratings(base_dir: str,
                                           dataset: str,
                                           query_strategy: str,
                                           strategy: str,
                                           expertise: str,
                                           confusion_column: str = "confusion_matrix",
                                           results_prefix: str = "results_",
                                           save_dir: str = None,
                                           filename: str = None):
    """
    Compara diferentes tamanhos de oráculo (|A|) para uma expertise fixa,
    fazendo a média entre todas as condições de rating.

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
        combined_accuracies = {}

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
                                    correct = np.trace(confusion)
                                    total = np.sum(confusion)
                                    accuracy = correct / total if total > 0 else 0.0
                                    combined_accuracies.setdefault(cycle, []).append(accuracy)
                            except Exception:
                                continue

        if combined_accuracies:
            cycles = sorted(combined_accuracies.keys())
            mean_accuracies = [np.mean(combined_accuracies[c]) for c in cycles]
            std_accuracies = [np.std(combined_accuracies[c]) for c in cycles]
            all_results[oracle_size] = {
                "cycles": cycles,
                "means": mean_accuracies,
                "stds": std_accuracies
            }

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

    # Baseline
    if baseline_accuracies:
        baseline_cycles = sorted(baseline_accuracies.keys())
        baseline_means = [np.mean(baseline_accuracies[c]) for c in baseline_cycles]
        baseline_stds = [np.std(baseline_accuracies[c]) for c in baseline_cycles]
        upper = np.clip(np.array(baseline_means) + np.array(baseline_stds), 0, 1)
        lower = np.clip(np.array(baseline_means) - np.array(baseline_stds), 0, 1)

        plt.plot(baseline_cycles, baseline_means, label="Baseline", color="purple", linewidth=2)
        plt.fill_between(baseline_cycles, lower, upper, alpha=0.15, color="purple")

    # Finalização
    plt.xlabel("Cycle")
    plt.ylabel("Accuracy (Avg. ± std)")
    plt.title(f"Oracle Size Comparison (Averaged Ratings) - Expertise {expertise}, {dataset}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename or f"oracle_size_avg_rating_{dataset}_{expertise}.png")
        plt.savefig(save_path)
        print(f"[OK] Plot saved at {save_path}")
        plt.close()
    else:
        plt.show()


def compare_rating_conditions_for_oracle(base_dir: str,
                                         dataset: str,
                                         query_strategy: str,
                                         strategy: str,
                                         oracle_size: str,
                                         expertise: str,
                                         confusion_column: str = "confusion_matrix",
                                         results_prefix: str = "results_",
                                         save_dir: str = None,
                                         filename: str = None):
    """
    Compara diferentes condições de rating (Only Acc, Both S and Acc, Only S)
    para um oracle_size e nível de expertise fixos.

    :param base_dir: Diretório base com os resultados
    :param dataset: Nome do dataset
    :param query_strategy: Estratégia de query
    :param strategy: Nome da estratégia (ex: reputation)
    :param oracle_size: Tamanho do oráculo (ex: "5")
    :param expertise: Nível de expertise (ex: "L", "M", "H")
    :param confusion_column: Nome da coluna da matriz de confusão
    :param results_prefix: Prefixo das pastas de resultados
    :param save_dir: Caminho para salvar o gráfico
    :param filename: Nome do arquivo
    """

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




def main():
    for dataset in DATASETS:
        for oracle_size in ORACLE_SIZES: 
            for expertise in EXPERTISES: 
                for metric in ['accuracy', 'precision', 'f1-score']:
                    save_dir = os.path.join(RESULTS_PATH, dataset, "plots", "all_metrics")
                    os.makedirs(save_dir, exist_ok=True)
                
                    filename = f"{metric}_{str(oracle_size)}_{expertise}.png"
                    """ plot_metric_over_cycles(
                        dataset=dataset,
                        query_strategy="uncertainty_sampling",
                        oracle_size=str(oracle_size),
                        expertise=expertise,
                        metric=metric,
                        save_dir=save_dir,
                        filename=filename
                    ) """




    """  oracle_sizes = ORACLE_SIZES
    expertises = EXPERTISES
    for dataset in DATASETS:
        latex_s = generate_latex_table(
                    dataset=dataset,
                    query_strategy="uncertainty_sampling",
                    rating_type="with_only_rating",
                    oracle_sizes=oracle_sizes,
                    expertises=expertises,
                    caption="Performance metrics for Only $\\mathcal{S}$ on \\textit{MNIST}.",
                    label="tab:mnist_metrics_only_s"
                )

        latex_both = generate_latex_table(
            dataset=dataset,
            query_strategy="uncertainty_sampling",
            rating_type="with_rating",
            oracle_sizes=oracle_sizes,
            expertises=expertises,
            caption="Performance metrics for Both $\\mathcal{S}$ and Acc on \\textit{MNIST}.",
            label="tab:mnist_metrics_both"
        )

        latex_acc = generate_latex_table(
            dataset=dataset,
            query_strategy="uncertainty_sampling",
            rating_type="without_rating",
            oracle_sizes=oracle_sizes,
            expertises=expertises,
            caption="Performance metrics for Only Acc on \\textit{MNIST}.",
            label="tab:mnist_metrics_only_acc"
        )

        filename_tex_1 = save_dir = os.path.join(RESULTS_PATH, dataset, "plots", "all_metrics", "tables_only_S.tex") 
        filename_tex_2 = save_dir = os.path.join(RESULTS_PATH, dataset, "plots", "all_metrics", "tables_both.tex") 
        filename_tex_3 = save_dir = os.path.join(RESULTS_PATH, dataset, "plots", "all_metrics", "tables_only_acc.tex") 
        
        with open(filename_tex_1, "w") as f:
            f.write(latex_s)

        with open(filename_tex_2, "w") as f:
            f.write(latex_both)

        with open(filename_tex_3, "w") as f:
            f.write(latex_acc)
     """


    for dataset in DATASETS:
        for oracle_size in ORACLE_SIZES:
                save_dir = os.path.join(RESULTS_PATH, dataset, "plots", "all_metrics_boxplots")
                plot_boxplot_annotator_accuracy_top_class(
                        base_dir=RESULTS_PATH,
                        dataset=dataset,
                        query_strategy="uncertainty_sampling",
                        strategy="reputation_based",
                        oracle_size=str(oracle_size),
                        save_path=save_dir,
                        filename=f"{dataset}_{oracle_size}_annotator_accuracy_.png"
                    )




def test_oracle():

    for dataset in DATASETS:
        for expertise in EXPERTISES:
            all_annotators = []
            for oracle_size in ORACLE_SIZES:
                for ratting_flag in RATINGS_PERMUTATIONS:
                    for i in range(30):
                        committee = Committee(size=oracle_size, seed=i, expertise=expertise, results_path=None)
                        all_annotators.extend(committee.annotators)

            print(len(all_annotators))
            save_path= os.path.join(RESULTS_PATH, dataset, "plots", "all_metrics_exp_accuracy")
            plot_expected_accuracy_by_expert_class(
                annotators=all_annotators,
                dataset_name=dataset,
                expertise=expertise,
                save_path=save_path,
                filename=f"exp_accuracy_{expertise}.png"
            )
                            



# FIXADO O EXPERTISE E COMPARAÇAO POR ORACLE SIZE 
def plot_oracle_comparison():
    for dataset in DATASETS:
        for expertise in EXPERTISES:
            save_dir = os.path.join(RESULTS_PATH, dataset, "plots", "expertise_comparison")
            compare_oracle_sizes_averaging_ratings(
                base_dir=RESULTS_PATH,
                dataset=dataset,
                query_strategy="uncertainty_sampling",
                strategy="reputation_based",
                expertise=expertise,
                save_dir=save_dir,
                filename=f"oracle_size_effect_{expertise}.png"
            )



def plot_expertise_comparison():
    for dataset in DATASETS:
        for oracle_size in ORACLE_SIZES:
            for expertise in EXPERTISES:
                save_dir = os.path.join(RESULTS_PATH, dataset, "plots", "expertise_comparison")
                compare_rating_conditions_for_oracle(
                base_dir=RESULTS_PATH,
                dataset=dataset,
                query_strategy="uncertainty_sampling",
                strategy="reputation_based",
                oracle_size=str(oracle_size),
                expertise=expertise,
                save_dir=save_dir,
                filename=f"compare_ratings_{expertise}_{str(oracle_size)}.png"
            )




def plot_reputation_comparison_per_oracle():
    for dataset in DATASETS:
        for rating_flag in RATINGS_PERMUTATIONS:
            for expertise in EXPERTISES:
                save_dir = os.path.join(RESULTS_PATH, dataset, "plots", "reputation_comparison")
                compare_avg_reputation_by_oracle_size_cycles(
                    base_dir=RESULTS_PATH,
                    dataset=dataset,
                    query_strategy="uncertainty_sampling",
                    strategy="reputation_based",
                    rating_condition=rating_flag,
                    expertise=expertise,
                    save_dir=save_dir,
                    filename=f"avg_reputation_{rating_flag}_{expertise}.png"
                )



if __name__ == "__main__":
    #main()
    #test_oracle()
    #plot_oracle_comparison()
    #plot_expertise_comparison()
    plot_reputation_comparison_per_oracle()
