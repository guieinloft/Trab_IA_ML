"""
Trabalho de IA/ML - Pipeline de Machine Learning
Autores: Gabriel Stiegemeier e Guilherme Einloft
Cadeira: Inteligência Artificial
Dataset: Pima Indians Diabetes (UCI / OpenML)
Descrição: Etapa 7 - Análise e controle de overfitting através de regularização L2 na MLP.
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

def run_etapa_7(state):
    globals().update(state)
    # ================= ETAPA 7 — Regularização e Análise de Overfitting =================
    print("\n" + "="*60)
    print("[INFO] Iniciando ETAPA 7 — Regularização e Análise de Overfitting...")
    print("="*60)
    # -------------------------------------------------------
    # 7.1 Forçando Overfitting
    print("  -> Executando passo: 7.1 Forçando Overfitting")
    # -------------------------------------------------------
    # Para forçar o overfitting na MLP do scikit-learn:
    # Usaremos uma rede muito densa, muitas épocas, alpha=0 (sem L2) e uma "artimanha"
    # com early_stopping=True mas paciência extrema (n_iter_no_change=500), 
    # apenas para que o sklearn seja obrigado a capturar e preencher a variável
    # `validation_scores_` ao longo das épocas (útil para plotarmos o gráfico do descolamento).
    mlp_overfit = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation='relu',
        solver='adam',
        alpha=0.0,             # Sem regularização
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=500,  # "Desliga" o early stopping na prática
        random_state=42
    )
    mlp_overfit.fit(X_train, y_train)
    # -------------------------------------------------------
    # 7.2 Aplicando Regularização
    print("  -> Executando passo: 7.2 Aplicando Regularização")
    # -------------------------------------------------------
    mlp_reg = MLPClassifier(
        hidden_layer_sizes=(64, 32), # Regularização estrutural (rede menor)
        activation='relu',
        solver='adam',
        alpha=0.01,            # Forte regularização L2
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=15,   # Early Stopping ativado e agressivo
        random_state=42
    )
    mlp_reg.fit(X_train, y_train)
    # -------------------------------------------------------
    # 7.2.1 Métricas comparativas no conjunto de teste
    # -------------------------------------------------------
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    # Modelo SEM regularização (overfitting)
    y_pred_overfit = mlp_overfit.predict(X_test)
    y_prob_overfit = mlp_overfit.predict_proba(X_test)[:, 1]
    acc_overfit = accuracy_score(y_test, y_pred_overfit)
    f1_overfit = f1_score(y_test, y_pred_overfit)
    auc_overfit = roc_auc_score(y_test, y_prob_overfit)
    # Modelo COM regularização
    y_pred_regularizado = mlp_reg.predict(X_test)
    y_prob_regularizado = mlp_reg.predict_proba(X_test)[:, 1]
    acc_regularizado = accuracy_score(y_test, y_pred_regularizado)
    f1_regularizado = f1_score(y_test, y_pred_regularizado)
    auc_regularizado = roc_auc_score(y_test, y_prob_regularizado)
    # Salvar comparação em arquivo para o relatório
    with open(f"{OUTPUT_DIR}/metricas_overfitting.md", "w", encoding="utf-8") as f:
        f.write("# Comparação — Sem Regularização vs Com Regularização (Etapa 7)\n\n")
        f.write("| Métrica | Sem Regularização | Com Regularização | Delta |\n")
        f.write("|---|---|---|---|\n")
        f.write(f"| Accuracy | {acc_overfit:.4f} | {acc_regularizado:.4f} | {acc_regularizado - acc_overfit:+.4f} |\n")
        f.write(f"| F1-Score | {f1_overfit:.4f} | {f1_regularizado:.4f} | {f1_regularizado - f1_overfit:+.4f} |\n")
        f.write(f"| ROC-AUC | {auc_overfit:.4f} | {auc_regularizado:.4f} | {auc_regularizado - auc_overfit:+.4f} |\n")
    # -------------------------------------------------------
    # 7.3 Visualização Comparativa
    print("  -> Executando passo: 7.3 Visualização Comparativa")
    # -------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # Plot Overfitting
    # Convertemos Loss (que tende a 0) em (1 - Loss) só para ficar na mesma
    # escala visual (ascendente) do Validation Score (Acurácia que tende a 1)
    loss_overfit_inv = 1 - np.array(mlp_overfit.loss_curve_)
    axes[0].plot(mlp_overfit.validation_scores_, label='Acurácia (Validação)', color='red', linewidth=2)
    axes[0].plot(loss_overfit_inv, label='1 - Loss (Treino)', color='blue', linestyle='--', linewidth=2)
    axes[0].set_title('Modelo com Overfitting', fontsize=14)
    axes[0].set_xlabel('Épocas')
    axes[0].set_ylabel('Performance (escala ajustada)')
    axes[0].legend()
    axes[0].grid(True, linestyle=':', alpha=0.6)
    # Plot Regularizado
    loss_reg_inv = 1 - np.array(mlp_reg.loss_curve_)
    axes[1].plot(mlp_reg.validation_scores_, label='Acurácia (Validação)', color='green', linewidth=2)
    axes[1].plot(loss_reg_inv, label='1 - Loss (Treino)', color='blue', linestyle='--', linewidth=2)
    axes[1].set_title('Modelo Regularizado', fontsize=14)
    axes[1].set_xlabel('Épocas')
    axes[1].legend()
    axes[1].grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/22_overfitting_comparacao.png", dpi=150, bbox_inches="tight")
    plt.close()
    # A discussão teórica foi movida para o arquivo discussao.md
    state.update(locals())
    return state
