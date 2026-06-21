"""
Pipeline Completo de Machine Learning — Trabalho de IA/ML
Dataset: Pima Indians Diabetes (UCI / OpenML)
Autor: Cientista de Dados Sênior
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

def run_etapa_4(state):
    globals().update(state)
    # ================= ETAPA 4 — Classificação com MLP =================
    print("\n" + "="*60)
    print("[INFO] Iniciando ETAPA 4 — Classificação com MLP...")
    print("="*60)
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (classification_report, confusion_matrix,
                                 roc_curve, auc, ConfusionMatrixDisplay)
    from sklearn.svm import SVC
    from xgboost import XGBClassifier
    # -------------------------------------------------------
    # 4.1  Preparação dos dados (features selecionadas)
    # -------------------------------------------------------
    # Usar apenas as features selecionadas na Etapa 3
    X_clf_sel = df[selected_features].copy()
    y_clf_full = df[TARGET_CLF].copy()
    # Split treino/teste estratificado (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X_clf_sel, y_clf_full, test_size=0.2, random_state=42, stratify=y_clf_full
    )
    # -------------------------------------------------------
    # 4.2  Definição e treinamento da MLP
    # -------------------------------------------------------
    MLP_PARAMS = {
        "hidden_layer_sizes": (64, 32, 16),    # 3 camadas ocultas: 64→32→16 neurônios
        "activation": "relu",                   # ReLU: f(x) = max(0, x) — evita vanishing gradient
        "solver": "adam",                        # Adam: combina momentum + RMSProp
        "learning_rate_init": 0.001,             # Taxa de aprendizado inicial
        "max_iter": 500,                         # Número máximo de épocas
        "batch_size": 32,                        # Mini-batch de 32 amostras
        "early_stopping": True,                  # Parar se validação não melhorar
        "validation_fraction": 0.15,             # 15% do treino para validação interna
        "n_iter_no_change": 30,                  # Paciência: 30 épocas sem melhora
        "random_state": 42,
        "verbose": False
    }
    # Calcular número de parâmetros
    n_features = len(selected_features)
    layers = [n_features] + list(MLP_PARAMS["hidden_layer_sizes"]) + [1]
    n_params = sum(layers[i] * layers[i+1] + layers[i+1] for i in range(len(layers)-1))
    # Treinar MLP
    t0 = time.time()
    mlp = MLPClassifier(**MLP_PARAMS)
    mlp.fit(X_train, y_train)
    time_mlp = time.time() - t0
    # -------------------------------------------------------
    # 4.3  Avaliação da MLP
    # -------------------------------------------------------
    # Predições
    y_pred_mlp = mlp.predict(X_test)
    y_prob_mlp = mlp.predict_proba(X_test)[:, 1]
    # Métricas detalhadas
    report_mlp = classification_report(y_test, y_pred_mlp, target_names=["Negativo", "Positivo"],
                                        output_dict=True)
    # Métricas individuais
    acc_mlp = accuracy_score(y_test, y_pred_mlp)
    prec_mlp = precision_score(y_test, y_pred_mlp)
    rec_mlp = recall_score(y_test, y_pred_mlp)
    f1_mlp = f1_score(y_test, y_pred_mlp)
    fpr_mlp, tpr_mlp, _ = roc_curve(y_test, y_prob_mlp)
    auc_mlp = auc(fpr_mlp, tpr_mlp)
    # Matriz de confusão
    cm_mlp = confusion_matrix(y_test, y_pred_mlp)
    # -------------------------------------------------------
    # 4.4  Modelos clássicos para comparação
    # -------------------------------------------------------
    # --- XGBoost ---
    xgb_params = {
        "n_estimators": 100,
        "max_depth": 5,
        "learning_rate": 0.1,
        "random_state": 42,
        "eval_metric": "logloss",
        "use_label_encoder": False
    }
    t0 = time.time()
    xgb = XGBClassifier(**xgb_params)
    xgb.fit(X_train, y_train, verbose=False)
    time_xgb = time.time() - t0
    y_pred_xgb = xgb.predict(X_test)
    y_prob_xgb = xgb.predict_proba(X_test)[:, 1]
    acc_xgb = accuracy_score(y_test, y_pred_xgb)
    prec_xgb = precision_score(y_test, y_pred_xgb)
    rec_xgb = recall_score(y_test, y_pred_xgb)
    f1_xgb = f1_score(y_test, y_pred_xgb)
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_prob_xgb)
    auc_xgb = auc(fpr_xgb, tpr_xgb)
    cm_xgb = confusion_matrix(y_test, y_pred_xgb)
    # --- SVM (RBF kernel) ---
    t0 = time.time()
    svm = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42,
              class_weight="balanced")
    svm.fit(X_train, y_train)
    time_svm = time.time() - t0
    y_pred_svm = svm.predict(X_test)
    y_prob_svm = svm.predict_proba(X_test)[:, 1]
    acc_svm = accuracy_score(y_test, y_pred_svm)
    prec_svm = precision_score(y_test, y_pred_svm)
    rec_svm = recall_score(y_test, y_pred_svm)
    f1_svm = f1_score(y_test, y_pred_svm)
    fpr_svm, tpr_svm, _ = roc_curve(y_test, y_prob_svm)
    auc_svm = auc(fpr_svm, tpr_svm)
    cm_svm = confusion_matrix(y_test, y_pred_svm)
    # -------------------------------------------------------
    # 4.5  Tabela comparativa final
    # -------------------------------------------------------
    # -------------------------------------------------------
    # 4.6  Visualizações
    # -------------------------------------------------------
    # --- 4.6.1  Curva de aprendizado (Loss por época) ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # Loss curve
    epochs = range(1, len(mlp.loss_curve_) + 1)
    axes[0].plot(epochs, mlp.loss_curve_, color="#1565C0", linewidth=2, label="Loss (treino)")
    if hasattr(mlp, "validation_scores_") and mlp.validation_scores_ is not None:
        val_epochs = range(1, len(mlp.validation_scores_) + 1)
        axes[0].plot(val_epochs, [1 - s for s in mlp.validation_scores_],
                     color="#D32F2F", linewidth=2, linestyle="--", label="1 − Accuracy (validação)")
    axes[0].set_xlabel("Época", fontsize=11)
    axes[0].set_ylabel("Loss", fontsize=11)
    axes[0].set_title("Curva de Perda (Loss) por Época", fontsize=13, fontweight="bold")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    # Validation accuracy curve
    if hasattr(mlp, "validation_scores_") and mlp.validation_scores_ is not None:
        axes[1].plot(val_epochs, mlp.validation_scores_,
                     color="#2E7D32", linewidth=2, label="Accuracy (validação)")
        axes[1].axhline(y=max(mlp.validation_scores_), color="#D32F2F", linestyle=":",
                        alpha=0.7, label=f"Melhor = {max(mlp.validation_scores_):.4f}")
        axes[1].set_xlabel("Época", fontsize=11)
        axes[1].set_ylabel("Accuracy", fontsize=11)
        axes[1].set_title("Evolução da Accuracy (Validação)", fontsize=13, fontweight="bold")
        axes[1].legend()
        axes[1].grid(alpha=0.3)
    fig.suptitle("MLP — Curvas de Aprendizado", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/15_mlp_curvas_aprendizado.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    # --- 4.6.2  Matrizes de confusão (3 modelos lado a lado) ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    models_cm = [
        ("MLP", cm_mlp),
        ("XGBoost", cm_xgb),
        ("SVM (RBF)", cm_svm)
    ]
    for ax, (name, cm) in zip(axes, models_cm):
        disp = ConfusionMatrixDisplay(cm, display_labels=["Neg", "Pos"])
        disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
        ax.set_title(name, fontsize=13, fontweight="bold")
    fig.suptitle("Matrizes de Confusão — Comparação de Modelos",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/16_matrizes_confusao.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    # --- 4.6.3  Curvas ROC (3 modelos sobrepostos) ---
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(fpr_mlp, tpr_mlp, color="#1565C0", linewidth=2.5,
            label=f"MLP (AUC = {auc_mlp:.3f})")
    ax.plot(fpr_xgb, tpr_xgb, color="#2E7D32", linewidth=2.5,
            label=f"XGBoost (AUC = {auc_xgb:.3f})")
    ax.plot(fpr_svm, tpr_svm, color="#D32F2F", linewidth=2.5,
            label=f"SVM RBF (AUC = {auc_svm:.3f})")
    ax.plot([0, 1], [0, 1], color="#9E9E9E", linewidth=1.5, linestyle="--",
            label="Aleatório (AUC = 0.500)")
    ax.set_xlabel("Taxa de Falsos Positivos (FPR)", fontsize=12)
    ax.set_ylabel("Taxa de Verdadeiros Positivos (TPR)", fontsize=12)
    ax.set_title("Curvas ROC — Comparação de Modelos", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(alpha=0.3)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/17_curvas_roc.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    # --- 4.6.4  Barras comparativas de métricas ---
    fig, ax = plt.subplots(figsize=(12, 6))
    metrics_names_4 = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    mlp_vals = [acc_mlp, prec_mlp, rec_mlp, f1_mlp, auc_mlp]
    xgb_vals = [acc_xgb, prec_xgb, rec_xgb, f1_xgb, auc_xgb]
    svm_vals = [acc_svm, prec_svm, rec_svm, f1_svm, auc_svm]
    x_pos = np.arange(len(metrics_names_4))
    width = 0.25
    bars1 = ax.bar(x_pos - width, mlp_vals, width, label="MLP",
                   color="#1565C0", edgecolor="white", alpha=0.85)
    bars2 = ax.bar(x_pos, xgb_vals, width, label="XGBoost",
                   color="#2E7D32", edgecolor="white", alpha=0.85)
    bars3 = ax.bar(x_pos + width, svm_vals, width, label="SVM (RBF)",
                   color="#D32F2F", edgecolor="white", alpha=0.85)
    ax.set_ylabel("Score")
    ax.set_title("Comparação de Métricas — MLP vs XGBoost vs SVM",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metrics_names_4)
    ax.set_ylim(0.5, 1.0)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/18_comparacao_metricas_modelos.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    # -------------------------------------------------------
    # 4.7  Validação cruzada para comparação robusta
    # -------------------------------------------------------
    cv_10 = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    scoring_simple = {"accuracy": "accuracy", "f1": "f1", "roc_auc": "roc_auc"}
    # MLP CV
    t0 = time.time()
    cv_mlp = cross_validate(
        MLPClassifier(**{k: v for k, v in MLP_PARAMS.items()}),
        X_clf_sel, y_clf_full, cv=cv_10, scoring=scoring_simple, n_jobs=-1
    )
    cv_time_mlp = time.time() - t0
    # XGBoost CV
    t0 = time.time()
    cv_xgb = cross_validate(
        XGBClassifier(**xgb_params),
        X_clf_sel, y_clf_full, cv=cv_10, scoring=scoring_simple, n_jobs=-1
    )
    cv_time_xgb = time.time() - t0
    # SVM CV
    t0 = time.time()
    cv_svm = cross_validate(
        SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42,
            class_weight="balanced"),
        X_clf_sel, y_clf_full, cv=cv_10, scoring=scoring_simple, n_jobs=-1
    )
    cv_time_svm = time.time() - t0
    for name, cv_res, cv_t in [("MLP", cv_mlp, cv_time_mlp),
                                 ("XGBoost", cv_xgb, cv_time_xgb),
                                 ("SVM (RBF)", cv_svm, cv_time_svm)]:
        acc = f"{cv_res['test_accuracy'].mean():.4f}±{cv_res['test_accuracy'].std():.4f}"
        f1 = f"{cv_res['test_f1'].mean():.4f}±{cv_res['test_f1'].std():.4f}"
        rauc = f"{cv_res['test_roc_auc'].mean():.4f}±{cv_res['test_roc_auc'].std():.4f}"
    # Guardar para o relatório
    cv_results_e4 = {
        "MLP": cv_mlp, "XGBoost": cv_xgb, "SVM (RBF)": cv_svm
    }
    cv_times_e4 = {
        "MLP": cv_time_mlp, "XGBoost": cv_time_xgb, "SVM (RBF)": cv_time_svm
    }
    state.update(locals())
    return state
