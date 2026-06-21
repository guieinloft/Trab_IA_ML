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

def run_etapa_1(state):
    globals().update(state)
    # ================= ETAPA 1 — Carga e Análise Exploratória Inicial =================
    print("\n" + "="*60)
    print("[INFO] Iniciando ETAPA 1 — Carga e Análise Exploratória Inicial...")
    print("="*60)
    # -------------------------------------------------------
    # 1.1  Carga do dataset via OpenML (Pima Indians Diabetes)
    # -------------------------------------------------------
    # O dataset "diabetes" no OpenML (id=37) corresponde ao Pima Indians Diabetes
    dataset = fetch_openml(data_id=37, as_frame=True, parser="auto")
    df = dataset.data.copy()
    # A variável alvo original é categórica: "tested_positive" / "tested_negative"
    # Codificamos como 1 e 0 respectivamente
    target_map = {"tested_positive": 1, "tested_negative": 0}
    df["Outcome"] = dataset.target.map(target_map).astype(int)
    # -------------------------------------------------------
    # 1.2  Visão geral dos dados
    # -------------------------------------------------------
    # Garantir tipos numéricos
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # -------------------------------------------------------
    # 1.3  Definição das variáveis-alvo
    # -------------------------------------------------------
    TARGET_CLF = "Outcome"       # Classificação binária (0 / 1)
    TARGET_REG = "plas"          # Regressão: concentração de glicose plasmática (contínua)
    # Distribuição da classe de classificação
    dist_clf = df[TARGET_CLF].value_counts()
    # Estatísticas da variável de regressão
    # -------------------------------------------------------
    # 1.4  Diagnóstico de valores ausentes
    # -------------------------------------------------------
    # No Pima Indians, zeros em certas colunas são biologicamente impossíveis
    # e representam valores ausentes codificados como 0.
    zero_invalid_cols = ["plas", "pres", "skin", "insu", "mass"]
    missing_report = {}
    for col in zero_invalid_cols:
        n_zeros = (df[col] == 0).sum()
        pct = n_zeros / len(df) * 100
        missing_report[col] = {"zeros": n_zeros, "pct": f"{pct:.1f}%"}
    # Valores NaN reais
    nan_total = df.isna().sum()
    # -------------------------------------------------------
    # 1.5  Diagnóstico de classes desbalanceadas
    # -------------------------------------------------------
    ratio = dist_clf.min() / dist_clf.max()
    # -------------------------------------------------------
    # 1.6  Detecção de outliers (Z-score e IQR)
    # -------------------------------------------------------
    features = [c for c in df.columns if c != TARGET_CLF]
    outlier_report = {}
    for col in features:
        col_data = df[col].dropna()
        # Método IQR
        Q1 = col_data.quantile(0.25)
        Q3 = col_data.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        n_out_iqr = ((col_data < lower) | (col_data > upper)).sum()
        # Método Z-score
        z = np.abs(stats.zscore(col_data))
        n_out_z = (z > 3).sum()
        outlier_report[col] = {"IQR": n_out_iqr, "Z>3": n_out_z}
    # -------------------------------------------------------
    # 1.7  Análise de correlação
    # -------------------------------------------------------
    corr_matrix = df[features].corr()
    # Pares com alta correlação (|r| > 0.7)
    high_corr = []
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            r = corr_matrix.iloc[i, j]
            if abs(r) > 0.7:
                high_corr.append((features[i], features[j], round(r, 3)))
    # Correlação de cada feature com os dois targets
    corr_clf = df[features].corrwith(df[TARGET_CLF]).sort_values(ascending=False)
    corr_reg = df[[c for c in features if c != TARGET_REG]].corrwith(df[TARGET_REG]).sort_values(ascending=False)
    # -------------------------------------------------------
    # 1.8  Visualizações exploratórias (salvas em disco)
    # -------------------------------------------------------
    OUTPUT_DIR = "output"
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # --- 1.8.1  Histograma de distribuição de todas as features ---
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    axes = axes.ravel()
    all_cols = features + [TARGET_CLF]
    for idx, col in enumerate(all_cols):
        ax = axes[idx]
        df[col].hist(bins=30, ax=ax, color="#4C72B0", edgecolor="white", alpha=0.85)
        ax.set_title(col, fontsize=11, fontweight="bold")
        ax.set_ylabel("Frequência")
    # Esconder eixo extra se houver
    for idx in range(len(all_cols), len(axes)):
        axes[idx].set_visible(False)
    fig.suptitle("Distribuição dos Atributos", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/01_distribuicoes.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    # --- 1.8.2  Boxplots por classe (Outcome) ---
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()
    feat_no_target = [c for c in features if c != TARGET_REG]
    for idx, col in enumerate(feat_no_target[:8]):
        ax = axes[idx]
        df.boxplot(column=col, by=TARGET_CLF, ax=ax,
                   patch_artist=True,
                   boxprops=dict(facecolor="#64B5F6", color="#1565C0"),
                   medianprops=dict(color="#D32F2F", linewidth=2))
        ax.set_title(col, fontsize=11, fontweight="bold")
        ax.set_xlabel("Outcome")
        ax.set_ylabel("")
    fig.suptitle("Boxplots por Classe (Outcome)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/02_boxplots_por_classe.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    # --- 1.8.3  Mapa de calor de correlação ---
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(df.corr(), dtype=bool))
    sns.heatmap(df.corr().round(2), annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, mask=mask, linewidths=0.5, ax=ax,
                cbar_kws={"shrink": 0.8})
    ax.set_title("Matriz de Correlação (Pearson)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/03_correlacao.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    # --- 1.8.4  Distribuição da variável de regressão (glicose) ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(df[TARGET_REG].dropna(), bins=30, color="#66BB6A", edgecolor="white", alpha=0.85)
    axes[0].set_title(f"Distribuição de '{TARGET_REG}'", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Glicose (mg/dL)")
    axes[0].set_ylabel("Frequência")
    axes[0].axvline(df[TARGET_REG].mean(), color="#D32F2F", linestyle="--", label=f"Média={df[TARGET_REG].mean():.1f}")
    axes[0].axvline(df[TARGET_REG].median(), color="#1565C0", linestyle="--", label=f"Mediana={df[TARGET_REG].median():.1f}")
    axes[0].legend()
    stats.probplot(df[TARGET_REG].dropna(), dist="norm", plot=axes[1])
    axes[1].set_title("Q-Q Plot (Normalidade)", fontsize=12, fontweight="bold")
    fig.suptitle(f"Análise da Variável de Regressão: '{TARGET_REG}'", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/04_regressao_target.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    # --- 1.8.5  Distribuição da classe de classificação ---
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = ["#64B5F6", "#EF5350"]
    dist_clf.plot(kind="bar", ax=ax, color=colors, edgecolor="white", width=0.5)
    ax.set_title(f"Distribuição da Classe '{TARGET_CLF}'", fontsize=13, fontweight="bold")
    ax.set_xlabel("Classe")
    ax.set_ylabel("Contagem")
    ax.set_xticklabels(["Negativo (0)", "Positivo (1)"], rotation=0)
    for i, v in enumerate(dist_clf.values):
        ax.text(i, v + 5, str(v), ha="center", fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/05_distribuicao_classes.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    state.update(locals())
    return state
