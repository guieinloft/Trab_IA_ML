"""
Trabalho de IA/ML - Pipeline de Machine Learning
Autores: Gabriel Stiegemeier e Guilherme Einloft
Cadeira: Inteligência Artificial
Dataset: Pima Indians Diabetes (UCI / OpenML)
Descrição: Etapa 3 - Seleção de features utilizando Informação Mútua e SHAP.
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

def run_etapa_3(state):
    globals().update(state)
    # ================= ETAPA 3 — Seleção de Features (Classificação) =================
    print("\n" + "="*60)
    print("[INFO] Iniciando ETAPA 3 — Seleção de Features (Classificação)...")
    print("="*60)
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, cross_validate
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, roc_auc_score, make_scorer)
    import shap
    import time
    # -------------------------------------------------------
    # 3.1  Preparação dos dados para classificação (apenas TREINO)
    # -------------------------------------------------------
    # IMPORTANTE: seleção de features é feita apenas com dados de TREINO
    # para evitar data leakage.
    df_train = state.get("df_train")
    X_clf = df_train[feature_cols].copy()
    y_clf = df_train[TARGET_CLF].copy()
    # -------------------------------------------------------
    # 3.2  Seleção de Features via Informação Mútua (Método Filtro)
    # -------------------------------------------------------
    n_runs = 50
    mi_scores_all = np.zeros((n_runs, X_clf.shape[1]))
    for i in range(n_runs):
        mi_scores_all[i, :] = mutual_info_classif(
            X_clf, y_clf, discrete_features=False, random_state=i
        )
    # Média e desvio padrão das MI scores
    mi_mean = mi_scores_all.mean(axis=0)
    mi_std = mi_scores_all.std(axis=0)
    # Criar ranking
    mi_ranking = pd.DataFrame({
        "Feature": X_clf.columns,
        "MI_mean": mi_mean,
        "MI_std": mi_std
    }).sort_values("MI_mean", ascending=False).reset_index(drop=True)
    mi_ranking.index += 1  # ranking começa em 1
    for idx, row in mi_ranking.iterrows():
        bar = "█" * int(row["MI_mean"] * 100)
    # Critério de seleção: features com MI > limiar
    # Usamos a "regra do cotovelo" adaptada: selecionar features com MI > MI_média
    mi_threshold = mi_mean.mean()
    selected_features = mi_ranking[mi_ranking["MI_mean"] > mi_threshold]["Feature"].tolist()
    dropped_features = mi_ranking[mi_ranking["MI_mean"] <= mi_threshold]["Feature"].tolist()
    # (X_clf_selected é usada apenas para referência visual; a CV abaixo
    #  recalcula a seleção dentro de cada fold para evitar data leakage.)
    X_clf_selected = X_clf[selected_features]
    # -------------------------------------------------------
    # 3.3  Avaliação comparativa: todas features vs selecionadas
    #      (seleção MI feita DENTRO de cada fold para evitar data leakage)
    # -------------------------------------------------------
    # Modelo base: Random Forest com hiperparâmetros padrão
    # Usamos random_state fixo para reprodutibilidade
    rf_params = {
        "n_estimators": 100,
        "max_depth": None,
        "random_state": 42,
        "class_weight": "balanced",  # Compensa desbalanceamento moderado
        "n_jobs": -1
    }
    # Métricas de avaliação
    scoring = {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "f1": "f1",
        "roc_auc": "roc_auc"
    }
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    # --- Cenário 1: Todas as features (sem seleção — cross_validate padrão) ---
    t0 = time.time()
    rf_all = RandomForestClassifier(**rf_params)
    cv_all = cross_validate(rf_all, X_clf, y_clf, cv=cv, scoring=scoring,
                            return_train_score=True, n_jobs=-1)
    time_all = time.time() - t0
    # --- Cenário 2: Features selecionadas POR FOLD (MI dentro da CV) ---
    # Cada fold calcula MI apenas no treino do fold, seleciona features
    # com MI > média, treina e avalia. Isso elimina o data leakage.
    t0 = time.time()
    metric_names = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    cv_sel_test = {m: [] for m in metric_names}
    cv_sel_train = {m: [] for m in metric_names}
    features_per_fold = []  # para inspeção: quais features cada fold escolheu
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_clf, y_clf)):
        X_fold_train, X_fold_val = X_clf.iloc[train_idx], X_clf.iloc[val_idx]
        y_fold_train, y_fold_val = y_clf.iloc[train_idx], y_clf.iloc[val_idx]
        # Calcular MI apenas no treino do fold
        mi_fold = np.zeros((n_runs, X_fold_train.shape[1]))
        for i in range(n_runs):
            mi_fold[i, :] = mutual_info_classif(
                X_fold_train, y_fold_train,
                discrete_features=False, random_state=i
            )
        mi_fold_mean = mi_fold.mean(axis=0)
        mi_fold_threshold = mi_fold_mean.mean()
        fold_selected = [c for c, m in zip(X_clf.columns, mi_fold_mean)
                         if m > mi_fold_threshold]
        # Fallback: se nenhuma feature passar o limiar, usar todas
        if len(fold_selected) == 0:
            fold_selected = list(X_clf.columns)
        features_per_fold.append(fold_selected)
        # Treinar e avaliar com as features selecionadas neste fold
        rf_fold = RandomForestClassifier(**rf_params)
        rf_fold.fit(X_fold_train[fold_selected], y_fold_train)
        y_pred_val = rf_fold.predict(X_fold_val[fold_selected])
        y_proba_val = rf_fold.predict_proba(X_fold_val[fold_selected])[:, 1]
        y_pred_train = rf_fold.predict(X_fold_train[fold_selected])
        y_proba_train = rf_fold.predict_proba(X_fold_train[fold_selected])[:, 1]
        # Métricas de teste
        cv_sel_test["accuracy"].append(accuracy_score(y_fold_val, y_pred_val))
        cv_sel_test["precision"].append(precision_score(y_fold_val, y_pred_val, zero_division=0))
        cv_sel_test["recall"].append(recall_score(y_fold_val, y_pred_val, zero_division=0))
        cv_sel_test["f1"].append(f1_score(y_fold_val, y_pred_val, zero_division=0))
        cv_sel_test["roc_auc"].append(roc_auc_score(y_fold_val, y_proba_val))
        # Métricas de treino (para análise de overfitting)
        cv_sel_train["accuracy"].append(accuracy_score(y_fold_train, y_pred_train))
        cv_sel_train["precision"].append(precision_score(y_fold_train, y_pred_train, zero_division=0))
        cv_sel_train["recall"].append(recall_score(y_fold_train, y_pred_train, zero_division=0))
        cv_sel_train["f1"].append(f1_score(y_fold_train, y_pred_train, zero_division=0))
        cv_sel_train["roc_auc"].append(roc_auc_score(y_fold_train, y_proba_train))
    time_sel = time.time() - t0
    # Converter listas em arrays para interface compatível com cross_validate
    cv_sel = {}
    for m in metric_names:
        cv_sel[f"test_{m}"] = np.array(cv_sel_test[m])
        cv_sel[f"train_{m}"] = np.array(cv_sel_train[m])
    # Log: consistência de features entre folds
    from collections import Counter
    feat_counts = Counter(f for fold in features_per_fold for f in fold)
    n_folds = len(features_per_fold)
    print(f"  [MI inside CV] Features selecionadas em todos os {n_folds} folds:")
    for feat, cnt in feat_counts.most_common():
        print(f"    {feat}: {cnt}/{n_folds} folds")
    # Exibir resultados comparativos
    results_comparison = {}
    for metric in metric_names:
        mean_all = cv_all[f"test_{metric}"].mean()
        std_all = cv_all[f"test_{metric}"].std()
        mean_sel = cv_sel[f"test_{metric}"].mean()
        std_sel = cv_sel[f"test_{metric}"].std()
        delta = mean_sel - mean_all
        sign = "+" if delta >= 0 else ""
        results_comparison[metric] = {
            "all_mean": mean_all, "all_std": std_all,
            "sel_mean": mean_sel, "sel_std": std_sel,
            "delta": delta
        }
    # Tempo e overfitting
    # Análise de overfitting
    train_acc_all = cv_all["train_accuracy"].mean()
    test_acc_all = cv_all["test_accuracy"].mean()
    train_acc_sel = cv_sel["train_accuracy"].mean()
    test_acc_sel = cv_sel["test_accuracy"].mean()
    overfit_all = train_acc_all - test_acc_all
    overfit_sel = train_acc_sel - test_acc_sel
    # -------------------------------------------------------
    # 3.4  Análise SHAP
    # -------------------------------------------------------
    # Treinar modelo final com todas as features para SHAP
    rf_shap = RandomForestClassifier(**rf_params)
    rf_shap.fit(X_clf, y_clf)
    # Criar explicador SHAP (TreeExplainer é exato para árvores)
    explainer = shap.TreeExplainer(rf_shap)
    shap_values = explainer(X_clf)
    # Para classificação binária, usar os SHAP values da classe positiva (1)
    if isinstance(shap_values.values, list):
        shap_vals_pos = shap_values.values[1]
    else:
        # shap >= 0.40 retorna shape (n_samples, n_features, n_classes)
        if shap_values.values.ndim == 3:
            shap_vals_pos = shap_values.values[:, :, 1]
        else:
            shap_vals_pos = shap_values.values
    # Importância global SHAP (mean |SHAP|)
    shap_importance = pd.DataFrame({
        "Feature": X_clf.columns,
        "mean_abs_SHAP": np.abs(shap_vals_pos).mean(axis=0)
    }).sort_values("mean_abs_SHAP", ascending=False).reset_index(drop=True)
    shap_importance.index += 1
    for idx, row in shap_importance.iterrows():
        bar = "█" * int(row["mean_abs_SHAP"] * 100)
    # -------------------------------------------------------
    # 3.5  Comparação MI vs SHAP
    # -------------------------------------------------------
    # Combinar rankings
    comparison_df = mi_ranking[["Feature", "MI_mean"]].merge(
        shap_importance[["Feature", "mean_abs_SHAP"]], on="Feature"
    )
    comparison_df["Rank_MI"] = range(1, len(comparison_df) + 1)
    comparison_df = comparison_df.sort_values("mean_abs_SHAP", ascending=False)
    comparison_df["Rank_SHAP"] = range(1, len(comparison_df) + 1)
    comparison_df = comparison_df.sort_values("Rank_MI")
    for _, row in comparison_df.iterrows():
        sel = "✓" if row["Feature"] in selected_features else "✗"
    # Concordância dos top-K
    top_mi = set(mi_ranking.head(len(selected_features))["Feature"])
    top_shap = set(shap_importance.head(len(selected_features))["Feature"])
    concordance = top_mi & top_shap
    # Correlação de Spearman entre rankings
    from scipy.stats import spearmanr
    comparison_sorted = comparison_df.sort_values("Rank_MI")
    rho, p_value = spearmanr(comparison_sorted["Rank_MI"], comparison_sorted["Rank_SHAP"])
    # -------------------------------------------------------
    # 3.6  Visualizações SHAP e seleção de features
    # -------------------------------------------------------
    # --- 3.6.1  SHAP Summary Plot (Beeswarm — importância global) ---
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_vals_pos, X_clf, feature_names=list(X_clf.columns),
                      show=False, plot_size=None)
    plt.title("SHAP Summary Plot — Importância Global (Classe Positiva)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/09_shap_summary_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    # --- 3.6.2  SHAP Bar Plot (importância média) ---
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.bar(shap_values[:, :, 1] if shap_values.values.ndim == 3 else shap_values,
                   show=False)
    plt.title("SHAP — Importância Média dos Atributos", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/10_shap_bar_importance.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    # --- 3.6.3  Explicação local: predição individual ---
    # Escolher uma amostra da classe positiva (diabética) e uma da negativa
    idx_pos = y_clf[y_clf == 1].index[0]  # primeira amostra positiva
    idx_neg = y_clf[y_clf == 0].index[0]  # primeira amostra negativa
    pred_pos = rf_shap.predict_proba(X_clf.iloc[[idx_pos]])[0]
    pred_neg = rf_shap.predict_proba(X_clf.iloc[[idx_neg]])[0]
    # SHAP waterfall para amostra positiva
    fig, ax = plt.subplots(figsize=(10, 6))
    if shap_values.values.ndim == 3:
        sv_local_pos = shap.Explanation(
            values=shap_values.values[idx_pos, :, 1],
            base_values=shap_values.base_values[idx_pos, 1] if shap_values.base_values.ndim > 1 else shap_values.base_values[idx_pos],
            data=shap_values.data[idx_pos],
            feature_names=list(X_clf.columns)
        )
    else:
        sv_local_pos = shap_values[idx_pos]
    shap.plots.waterfall(sv_local_pos, show=False)
    plt.title(f"SHAP Waterfall — Amostra {idx_pos} (Classe Real: Positivo)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/11_shap_waterfall_positivo.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    # SHAP waterfall para amostra negativa
    fig, ax = plt.subplots(figsize=(10, 6))
    if shap_values.values.ndim == 3:
        sv_local_neg = shap.Explanation(
            values=shap_values.values[idx_neg, :, 1],
            base_values=shap_values.base_values[idx_neg, 1] if shap_values.base_values.ndim > 1 else shap_values.base_values[idx_neg],
            data=shap_values.data[idx_neg],
            feature_names=list(X_clf.columns)
        )
    else:
        sv_local_neg = shap_values[idx_neg]
    shap.plots.waterfall(sv_local_neg, show=False)
    plt.title(f"SHAP Waterfall — Amostra {idx_neg} (Classe Real: Negativo)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/12_shap_waterfall_negativo.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    # --- 3.6.4  Gráfico comparativo MI vs SHAP (ranking) ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    # MI ranking
    mi_sorted = mi_ranking.sort_values("MI_mean", ascending=True)
    colors_mi = ["#4CAF50" if f in selected_features else "#BDBDBD"
                 for f in mi_sorted["Feature"]]
    axes[0].barh(mi_sorted["Feature"], mi_sorted["MI_mean"],
                 color=colors_mi, edgecolor="white", height=0.6)
    axes[0].errorbar(mi_sorted["MI_mean"], mi_sorted["Feature"],
                     xerr=mi_sorted["MI_std"], fmt="none", color="#333", capsize=3)
    axes[0].axvline(mi_threshold, color="#D32F2F", linestyle="--", linewidth=1.5,
                    label=f"Limiar = {mi_threshold:.4f}")
    axes[0].set_title("Informação Mútua (Filtro)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("MI Score")
    axes[0].legend()
    # SHAP ranking
    shap_sorted = shap_importance.sort_values("mean_abs_SHAP", ascending=True)
    colors_shap = ["#2196F3" if f in selected_features else "#BDBDBD"
                   for f in shap_sorted["Feature"]]
    axes[1].barh(shap_sorted["Feature"], shap_sorted["mean_abs_SHAP"],
                 color=colors_shap, edgecolor="white", height=0.6)
    axes[1].set_title("SHAP (|Shapley Value| médio)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Mean |SHAP Value|")
    fig.suptitle("Comparação: Informação Mútua vs SHAP — Ranking de Features",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/13_mi_vs_shap_ranking.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    # --- 3.6.5  Gráfico de desempenho comparativo ---
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics_names = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    metrics_keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    x_pos = np.arange(len(metrics_names))
    width = 0.35
    vals_all = [results_comparison[m]["all_mean"] for m in metrics_keys]
    vals_sel = [results_comparison[m]["sel_mean"] for m in metrics_keys]
    errs_all = [results_comparison[m]["all_std"] for m in metrics_keys]
    errs_sel = [results_comparison[m]["sel_std"] for m in metrics_keys]
    bars1 = ax.bar(x_pos - width/2, vals_all, width, yerr=errs_all,
                   label=f"Todas ({X_clf.shape[1]} features)", color="#FF7043",
                   edgecolor="white", capsize=4, alpha=0.85)
    bars2 = ax.bar(x_pos + width/2, vals_sel, width, yerr=errs_sel,
                   label=f"Selecionadas ({len(selected_features)} features)", color="#42A5F5",
                   edgecolor="white", capsize=4, alpha=0.85)
    ax.set_ylabel("Score")
    ax.set_title("Desempenho: Todas Features vs Features Selecionadas (RF, 10-fold CV)",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metrics_names)
    ax.set_ylim(0.5, 1.0)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    # Adicionar valores nas barras
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/14_desempenho_comparativo.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    # -------------------------------------------------------
    # 3.7  Sumário da Etapa 3
    # -------------------------------------------------------
    state.update(locals())
    return state
