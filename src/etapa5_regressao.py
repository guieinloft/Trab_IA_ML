"""
Trabalho de IA/ML - Pipeline de Machine Learning
Autores: Gabriel Stiegemeier e Guilherme Einloft
Cadeira: Inteligência Artificial
Dataset: Pima Indians Diabetes (UCI / OpenML)
Descrição: Etapa 5 - Treinamento e avaliação do modelo de regressão com MLP.
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

def run_etapa_5(state):
    globals().update(state)
    # ================= ETAPA 5 — Regressão com MLP =================
    print("\n" + "="*60)
    print("[INFO] Iniciando ETAPA 5 — Regressão com MLP...")
    print("="*60)
    from sklearn.neural_network import MLPRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    # -------------------------------------------------------
    # 5.1 Preparação dos dados para regressão
    print("  -> Executando passo: 5.1 Preparação dos dados para regressão")
    # -------------------------------------------------------
    # Para regressão, nosso target é TARGET_REG ('plas' - Glicose).
    # Usaremos todas as outras features, incluindo TARGET_CLF ('Outcome').
    features_reg = [c for c in df.columns if c != TARGET_REG]
    X_reg = df[features_reg].copy()
    # A variável alvo no 'df' está escalonada (pelo RobustScaler na Etapa 2).
    # Para interpretar os resultados (MAE, RMSE) na unidade original (mg/dL),
    # usaremos a variável do 'df_pre_scaling' como nosso target.
    y_reg = df_pre_scaling[TARGET_REG].copy()
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )
    # -------------------------------------------------------
    # 5.2 Definição e treinamento da MLP de Regressão
    print("  -> Executando passo: 5.2 Definição e treinamento da MLP de Regressão")
    # -------------------------------------------------------
    MLP_REG_PARAMS = {
        "hidden_layer_sizes": (64, 32),
        "activation": "relu",
        "solver": "adam",
        "learning_rate_init": 0.001,
        "max_iter": 1000,
        "batch_size": 32,
        "early_stopping": True,
        "validation_fraction": 0.15,
        "n_iter_no_change": 20,
        "random_state": 42,
        "verbose": False
    }
    t0_reg = time.time()
    mlp_reg = MLPRegressor(**MLP_REG_PARAMS)
    mlp_reg.fit(X_train_reg, y_train_reg)
    time_mlp_reg = time.time() - t0_reg
    # -------------------------------------------------------
    # 5.3 Avaliação do Modelo de Regressão
    print("  -> Executando passo: 5.3 Avaliação do Modelo de Regressão")
    # -------------------------------------------------------
    y_pred_reg = mlp_reg.predict(X_test_reg)
    mae = mean_absolute_error(y_test_reg, y_pred_reg)
    mse = mean_squared_error(y_test_reg, y_pred_reg)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_reg, y_pred_reg)
    # Salvar métricas em arquivo para o relatório
    with open(f"{OUTPUT_DIR}/metricas_regressao.md", "w", encoding="utf-8") as f:
        f.write("# Métricas de Regressão — MLP (Etapa 5)\n\n")
        f.write(f"| Métrica | Valor |\n|---|---|\n")
        f.write(f"| MAE | {mae:.4f} |\n")
        f.write(f"| MSE | {mse:.4f} |\n")
        f.write(f"| RMSE | {rmse:.4f} |\n")
        f.write(f"| R² | {r2:.4f} |\n")
    # -------------------------------------------------------
    # 5.4 Visualizações
    print("  -> Executando passo: 5.4 Visualizações")
    # -------------------------------------------------------
    # --- 5.4.1 Valores Reais vs Preditos ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].scatter(y_test_reg, y_pred_reg, alpha=0.6, color="#1565C0", edgecolor="white")
    # Linha ideal (x=y)
    min_val = min(y_test_reg.min(), y_pred_reg.min())
    max_val = max(y_test_reg.max(), y_pred_reg.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], color="#D32F2F", linestyle="--", linewidth=2, label="Ideal")
    axes[0].set_title("Valores Reais vs Preditos", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Valor Real (Glicose mg/dL)")
    axes[0].set_ylabel("Valor Predito")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    # --- 5.4.2 Resíduos ---
    residuos = y_test_reg - y_pred_reg
    axes[1].scatter(y_pred_reg, residuos, alpha=0.6, color="#2E7D32", edgecolor="white")
    axes[1].axhline(y=0, color="#D32F2F", linestyle="--", linewidth=2)
    axes[1].set_title("Análise de Resíduos", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Valor Predito")
    axes[1].set_ylabel("Resíduo (Real - Predito)")
    axes[1].grid(alpha=0.3)
    # --- 5.4.3 Curva de Aprendizado (Loss) ---
    epochs_reg = range(1, len(mlp_reg.loss_curve_) + 1)
    axes[2].plot(epochs_reg, mlp_reg.loss_curve_, color="#8E24AA", linewidth=2, label="Loss (treino)")
    if hasattr(mlp_reg, "validation_scores_") and mlp_reg.validation_scores_ is not None:
        val_epochs_reg = range(1, len(mlp_reg.validation_scores_) + 1)
        ax2_twin = axes[2].twinx()
        ax2_twin.plot(val_epochs_reg, mlp_reg.validation_scores_, color="#FF8F00", linewidth=2, linestyle="--", label="R² (validação)")
        ax2_twin.set_ylabel("R² score", color="#FF8F00")
        ax2_twin.tick_params(axis='y', labelcolor="#FF8F00")
        
        lines_1, labels_1 = axes[2].get_legend_handles_labels()
        lines_2, labels_2 = ax2_twin.get_legend_handles_labels()
        axes[2].legend(lines_1 + lines_2, labels_1 + labels_2, loc="center right")
    else:
        axes[2].legend(loc="center right")
    axes[2].set_title("Curva de Aprendizado", fontsize=13, fontweight="bold")
    axes[2].set_xlabel("Época")
    axes[2].set_ylabel("Loss")
    axes[2].grid(alpha=0.3)
    fig.suptitle("Regressão da Glicose ('plas') — Avaliação do Modelo MLP", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/19_mlp_regressao_avaliacao.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    state.update(locals())
    return state
