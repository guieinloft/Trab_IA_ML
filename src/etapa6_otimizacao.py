"""
Trabalho de IA/ML - Pipeline de Machine Learning
Autores: Gabriel Stiegemeier e Guilherme Einloft
Cadeira: Inteligência Artificial
Dataset: Pima Indians Diabetes (UCI / OpenML)
Descrição: Etapa 6 - Otimização de hiperparâmetros utilizando Optuna.
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

def run_etapa_6(state):
    globals().update(state)
    df_train = state.get("df_train")
    df_test = state.get("df_test")
    selected_features = state.get("selected_features")
    TARGET_CLF = state.get("TARGET_CLF")
    X_train = df_train[selected_features].copy()
    X_test = df_test[selected_features].copy()
    y_train = df_train[TARGET_CLF].copy()
    y_test = df_test[TARGET_CLF].copy()
    auc_mlp = state.get("auc_mlp")
    OUTPUT_DIR = state.get("OUTPUT_DIR")
    import time
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import roc_auc_score
    # ================= ETAPA 6 — Otimização de Hiperparâmetros com Optuna =================
    print("\n" + "="*60)
    print("[INFO] Iniciando ETAPA 6 — Otimização de Hiperparâmetros com Optuna...")
    print("="*60)
    import optuna
    from sklearn.model_selection import cross_val_score
    from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances
    # -------------------------------------------------------
    # 6.1 Definição do Espaço de Busca e Função Objetivo
    print("  -> Executando passo: 6.1 Definição do Espaço de Busca e Função Objetivo")
    # -------------------------------------------------------
    # Otimizaremos a MLP de Classificação (Etapa 4)
    # Usaremos as features selecionadas (X_clf_sel) e o target (y_clf_full)
    def objective(trial):
        # Espaço de busca para arquitetura e hiperparâmetros
        n_layers = trial.suggest_int('n_layers', 1, 3)
        layers = []
        for i in range(n_layers):
            layers.append(trial.suggest_int(f'n_units_l{i}', 16, 128, step=16))
        
        learning_rate_init = trial.suggest_float('learning_rate_init', 1e-4, 1e-2, log=True)
        activation = trial.suggest_categorical('activation', ['relu', 'tanh'])
        alpha = trial.suggest_float('alpha', 1e-5, 1e-2, log=True) # Penalização L2
        
        model = MLPClassifier(
            hidden_layer_sizes=tuple(layers),
            activation=activation,
            learning_rate_init=learning_rate_init,
            alpha=alpha,
            solver='adam',
            max_iter=300,
            early_stopping=True,
            random_state=42
        )
        
        # O objetivo é maximizar o ROC-AUC via cross-validation 5-fold
        # IMPORTANTE: CV feito apenas com dados de TREINO
        score = cross_val_score(model, X_train, y_train, n_jobs=-1, cv=5, scoring='roc_auc')
        return score.mean()
    # -------------------------------------------------------
    # 6.2 Execução da Otimização
    print("  -> Executando passo: 6.2 Execução da Otimização")
    # -------------------------------------------------------
    N_TRIALS = 30
    # Suprimir logs padrão do optuna para um terminal mais limpo
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    t0_opt = time.time()
    study = optuna.create_study(direction='maximize', study_name="MLP_Optimization")
    study.optimize(objective, n_trials=N_TRIALS)
    time_opt = time.time() - t0_opt
    # -------------------------------------------------------
    # 6.3 Resultados e Comparação
    print("  -> Executando passo: 6.3 Resultados e Comparação")
    # -------------------------------------------------------
    best_params = study.best_params
    best_score = study.best_value
    # Reconstruir melhor modelo e testar
    layers_best = [best_params[f'n_units_l{i}'] for i in range(best_params['n_layers'])]
    best_mlp = MLPClassifier(
        hidden_layer_sizes=tuple(layers_best),
        activation=best_params['activation'],
        learning_rate_init=best_params['learning_rate_init'],
        alpha=best_params['alpha'],
        solver='adam',
        max_iter=500,
        early_stopping=True,
        random_state=42
    )
    t0_best = time.time()
    best_mlp.fit(X_train, y_train)
    time_best = time.time() - t0_best
    # Avaliar no conjunto de teste final
    y_prob_best = best_mlp.predict_proba(X_test)[:, 1]
    auc_best = roc_auc_score(y_test, y_prob_best)
    delta_auc = auc_best - auc_mlp
    # Salvar resultados do Optuna em arquivo para o relatório
    with open(f"{OUTPUT_DIR}/resultados_optuna.md", "w", encoding="utf-8") as f:
        f.write("# Resultados da Otimização — Optuna (Etapa 6)\n\n")
        f.write(f"- **Nº de trials:** {N_TRIALS}\n")
        f.write(f"- **Tempo de otimização:** {time_opt:.1f}s\n")
        f.write(f"- **Melhor ROC-AUC (CV):** {best_score:.4f}\n\n")
        f.write("## Melhores Hiperparâmetros\n\n")
        f.write("| Hiperparâmetro | Valor |\n|---|---|\n")
        for k, v in best_params.items():
            f.write(f"| {k} | {v} |\n")
        f.write(f"\n## Comparação com Modelo Original\n\n")
        f.write(f"| Modelo | ROC-AUC (teste) |\n|---|---|\n")
        f.write(f"| MLP Original | {auc_mlp:.4f} |\n")
        f.write(f"| MLP Otimizada | {auc_best:.4f} |\n")
        f.write(f"| **Delta** | **{delta_auc:+.4f}** |\n")
    # -------------------------------------------------------
    # 6.4 Visualizações do Optuna
    print("  -> Executando passo: 6.4 Visualizações do Optuna")
    # -------------------------------------------------------
    # Histórico
    try:
        ax_hist = plot_optimization_history(study)
        ax_hist.figure.savefig(f"{OUTPUT_DIR}/20_optuna_history.png", dpi=150, bbox_inches="tight")
        plt.close(ax_hist.figure)
    except Exception as e:
        pass
    # Importância dos Hiperparâmetros
    try:
        ax_param = plot_param_importances(study)
        ax_param.figure.tight_layout()
        ax_param.figure.savefig(f"{OUTPUT_DIR}/21_optuna_param_importances.png", dpi=150, bbox_inches="tight")
        plt.close(ax_param.figure)
    except Exception as e:
        pass
    state.update(locals())
    return state
