"""
Trabalho de IA/ML - Pipeline de Machine Learning
Autores: Gabriel Stiegemeier e Guilherme Einloft
Cadeira: Inteligência Artificial
Dataset: Pima Indians Diabetes (UCI / OpenML)
Descrição: Etapa 4b - Classificação Multiclasse com MLP.
           Target derivado dos limiares clínicos de glicose (ADA):
           Classe 0 = Normal (<100 mg/dL), Classe 1 = Pré-diabético (100-140),
           Classe 2 = Diabético (≥140 mg/dL).
"""
import sys
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')
import warnings 
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # backend não-interativo
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.datasets import fetch_openml

def run_etapa_4b(state):
    globals().update(state)
    df_pre_scaling = state.get("df_pre_scaling")
    TARGET_REG = state.get("TARGET_REG")
    df = state.get("df")
    OUTPUT_DIR = state.get("OUTPUT_DIR")
    # ================= ETAPA 4b — Classificação Multiclasse com MLP =================
    print("\n" + "="*60)
    print("[INFO] Iniciando ETAPA 4b — Classificação Multiclasse com MLP...")
    print("="*60)
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, confusion_matrix, ConfusionMatrixDisplay)
    # -------------------------------------------------------
    # 4b.1  Criação do target multiclasse
    # -------------------------------------------------------
    # Critérios da American Diabetes Association (ADA) para glicose plasmática em jejum:
    #   - Normal:         glicose < 100 mg/dL
    #   - Pré-diabético:  100 ≤ glicose < 140 mg/dL
    #   - Diabético:      glicose ≥ 140 mg/dL
    #
    # Usamos os valores originais (não escalonados) de 'plas' para aplicar os limiares.
    df_train = state.get("df_train")
    df_test = state.get("df_test")
    df_train_pre_scaling = state.get("df_train_pre_scaling")
    df_test_pre_scaling = state.get("df_test_pre_scaling")
    # Criar targets multiclasse a partir dos valores pré-escalonamento
    glucose_train = df_train_pre_scaling[TARGET_REG].copy()
    glucose_test = df_test_pre_scaling[TARGET_REG].copy()
    y_multi_train = pd.cut(
        glucose_train,
        bins=[-np.inf, 100, 140, np.inf],
        labels=[0, 1, 2]
    ).astype(int)
    y_multi_test = pd.cut(
        glucose_test,
        bins=[-np.inf, 100, 140, np.inf],
        labels=[0, 1, 2]
    ).astype(int)
    # Para distribuição geral (visualização), combinar ambos
    y_multi = pd.concat([y_multi_train, y_multi_test], ignore_index=True)
    class_names = ["Normal", "Pré-diabético", "Diabético"]
    dist_multi = y_multi.value_counts().sort_index()
    # -------------------------------------------------------
    # 4b.2  Preparação dos dados
    # -------------------------------------------------------
    # Features: todas as colunas do df escalonado, EXCETO 'plas' (que define o target).
    # Nota: 'Outcome' é incluída como feature — ela reflete um diagnóstico independente
    # (baseado em OGTT — Oral Glucose Tolerance Test), diferente da glicose de jejum.
    features_multi = [c for c in df_train.columns if c != TARGET_REG]
    # Usar os splits já existentes (sem train_test_split interno)
    X_train_m = df_train[features_multi].copy()
    X_test_m = df_test[features_multi].copy()
    y_train_m = y_multi_train
    y_test_m = y_multi_test
    # -------------------------------------------------------
    # 4b.3  Definição e treinamento da MLP Multiclasse
    # -------------------------------------------------------
    # Justificativa da arquitetura:
    #   - 3 camadas ocultas (64→32→16): mesma pirâmide decrescente da binária,
    #     adequada ao volume de dados (768 amostras) e à dimensionalidade (8 features).
    #   - A camada de saída terá 3 neurônios (um por classe), com softmax implícito
    #     no MLPClassifier do scikit-learn.
    #   - Adam + ReLU + Early Stopping: mesmas justificativas da Etapa 4.
    MLP_MULTI_PARAMS = {
        "hidden_layer_sizes": (64, 32, 16),
        "activation": "relu",
        "solver": "adam",
        "learning_rate_init": 0.001,
        "max_iter": 500,
        "batch_size": 32,
        "early_stopping": True,
        "validation_fraction": 0.15,
        "n_iter_no_change": 30,
        "random_state": 42,
        "verbose": False
    }
    mlp_multi = MLPClassifier(**MLP_MULTI_PARAMS)
    mlp_multi.fit(X_train_m, y_train_m)
    # -------------------------------------------------------
    # 4b.4  Avaliação — Métricas Multiclasse
    # -------------------------------------------------------
    y_pred_m = mlp_multi.predict(X_test_m)
    acc_m = accuracy_score(y_test_m, y_pred_m)
    prec_m = precision_score(y_test_m, y_pred_m, average='macro', zero_division=0)
    rec_m = recall_score(y_test_m, y_pred_m, average='macro', zero_division=0)
    f1_m = f1_score(y_test_m, y_pred_m, average='macro', zero_division=0)
    cm_m = confusion_matrix(y_test_m, y_pred_m)
    # Salvar métricas em arquivo para o relatório
    with open(f"{OUTPUT_DIR}/metricas_multiclasse.md", "w", encoding="utf-8") as f:
        f.write("# Métricas de Classificação Multiclasse — MLP (Etapa 4b)\n\n")
        f.write("**Target:** Nível de glicose (critérios ADA)\n\n")
        f.write("| Classe | Rótulo | Critério |\n|---|---|---|\n")
        f.write("| 0 | Normal | glicose < 100 mg/dL |\n")
        f.write("| 1 | Pré-diabético | 100 ≤ glicose < 140 mg/dL |\n")
        f.write("| 2 | Diabético | glicose ≥ 140 mg/dL |\n\n")
        f.write("**Distribuição das classes:**\n\n")
        f.write("| Classe | Amostras |\n|---|---|\n")
        for cls_idx, cls_name in enumerate(class_names):
            f.write(f"| {cls_name} | {dist_multi.get(cls_idx, 0)} |\n")
        f.write(f"\n**Métricas (conjunto de teste):**\n\n")
        f.write("| Métrica | Valor |\n|---|---|\n")
        f.write(f"| Accuracy | {acc_m:.4f} |\n")
        f.write(f"| Precision Macro | {prec_m:.4f} |\n")
        f.write(f"| Recall Macro | {rec_m:.4f} |\n")
        f.write(f"| F1 Macro | {f1_m:.4f} |\n")
    # -------------------------------------------------------
    # 4b.5  Visualizações
    # -------------------------------------------------------
    # --- 4b.5.1  Distribuição das classes multiclasse ---
    fig, ax = plt.subplots(figsize=(7, 5))
    colors_multi = ["#4CAF50", "#FF9800", "#F44336"]
    dist_multi.plot(kind="bar", ax=ax, color=colors_multi, edgecolor="white", width=0.5)
    ax.set_title("Distribuição das Classes (Glicose — ADA)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Classe")
    ax.set_ylabel("Contagem")
    ax.set_xticklabels(class_names, rotation=0)
    for i, v in enumerate(dist_multi.values):
        ax.text(i, v + 5, str(v), ha="center", fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/23_multiclasse_distribuicao.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    # --- 4b.5.2  Matriz de confusão 3×3 ---
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(cm_m, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title("Matriz de Confusão — Classificação Multiclasse (MLP)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/24_multiclasse_confusao.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    # --- 4b.5.3  Curvas de aprendizado ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # Loss curve
    epochs_m = range(1, len(mlp_multi.loss_curve_) + 1)
    axes[0].plot(epochs_m, mlp_multi.loss_curve_, color="#1565C0", linewidth=2, label="Loss (treino)")
    if hasattr(mlp_multi, "validation_scores_") and mlp_multi.validation_scores_ is not None:
        val_epochs_m = range(1, len(mlp_multi.validation_scores_) + 1)
        axes[0].plot(val_epochs_m, [1 - s for s in mlp_multi.validation_scores_],
                     color="#D32F2F", linewidth=2, linestyle="--", label="1 − Accuracy (validação)")
    axes[0].set_xlabel("Época", fontsize=11)
    axes[0].set_ylabel("Loss", fontsize=11)
    axes[0].set_title("Curva de Perda (Loss) por Época", fontsize=13, fontweight="bold")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    # Validation accuracy curve
    if hasattr(mlp_multi, "validation_scores_") and mlp_multi.validation_scores_ is not None:
        axes[1].plot(val_epochs_m, mlp_multi.validation_scores_,
                     color="#2E7D32", linewidth=2, label="Accuracy (validação)")
        axes[1].axhline(y=max(mlp_multi.validation_scores_), color="#D32F2F", linestyle=":",
                        alpha=0.7, label=f"Melhor = {max(mlp_multi.validation_scores_):.4f}")
        axes[1].set_xlabel("Época", fontsize=11)
        axes[1].set_ylabel("Accuracy", fontsize=11)
        axes[1].set_title("Evolução da Accuracy (Validação)", fontsize=13, fontweight="bold")
        axes[1].legend()
        axes[1].grid(alpha=0.3)
    fig.suptitle("MLP Multiclasse — Curvas de Aprendizado", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/25_multiclasse_curvas_aprendizado.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    # --- 4b.5.4  Barras de métricas ---
    fig, ax = plt.subplots(figsize=(8, 5))
    metrics_names_m = ["Accuracy", "Precision\nMacro", "Recall\nMacro", "F1\nMacro"]
    metrics_vals_m = [acc_m, prec_m, rec_m, f1_m]
    x_pos_m = np.arange(len(metrics_names_m))
    bars_m = ax.bar(x_pos_m, metrics_vals_m, width=0.5, color=["#1565C0", "#2E7D32", "#FF8F00", "#D32F2F"],
                    edgecolor="white", alpha=0.85)
    ax.set_ylabel("Score")
    ax.set_title("Métricas — Classificação Multiclasse (MLP)", fontsize=13, fontweight="bold")
    ax.set_xticks(x_pos_m)
    ax.set_xticklabels(metrics_names_m)
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    for bar in bars_m:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/26_multiclasse_metricas.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    state.update(locals())
    return state
