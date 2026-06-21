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

def run_etapa_2(state):
    globals().update(state)
    # ================= ETAPA 2 — Pré-processamento =================
    print("\n" + "="*60)
    print("[INFO] Iniciando ETAPA 2 — Pré-processamento...")
    print("="*60)
    from sklearn.impute import KNNImputer
    from sklearn.preprocessing import RobustScaler, StandardScaler
    # Preservar cópia do DataFrame original para comparações
    df_original = df.copy()
    # -------------------------------------------------------
    # 2.1  Tratamento de valores ausentes (zeros inválidos → NaN → imputação)
    # -------------------------------------------------------
    # Passo 1: Substituir zeros biologicamente impossíveis por NaN
    zero_invalid_cols = ["plas", "pres", "skin", "insu", "mass"]
    for col in zero_invalid_cols:
        n_before = (df[col] == 0).sum()
        df[col] = df[col].replace(0, np.nan)
    total_nan = df.isna().sum()
    # Passo 2: Imputação via KNN Imputer (k=5)
    #
    # - Para as colunas com alta taxa de ausência (insu 48.7%, skin 29.6%), o KNN
    #   Imputer utiliza as features disponíveis para inferir valores plausíveis.
    # Separar features e targets antes da imputação
    features_for_imputation = [c for c in df.columns if c != TARGET_CLF]
    imputer = KNNImputer(n_neighbors=5, weights="uniform")
    df[features_for_imputation] = imputer.fit_transform(df[features_for_imputation])
    # Verificar que não restam NaN
    nan_after = df.isna().sum().sum()
    assert nan_after == 0, "Erro: ainda existem valores ausentes!"
    # Comparar estatísticas antes e depois da imputação
    for col in zero_invalid_cols:
        mean_before = df_original[col].mean()
        mean_after = df[col].mean()
        med_before = df_original[col].median()
        med_after = df[col].median()
    # -------------------------------------------------------
    # 2.2  Codificação de variáveis categóricas
    # -------------------------------------------------------
    # Verificar os tipos de dados de cada coluna
    # ANÁLISE: Todos os 8 atributos preditivos são numéricos contínuos ou discretos.
    # A variável alvo de classificação (Outcome) já está codificada como 0/1 (int).
    # Portanto, NÃO há necessidade de aplicar One-Hot Encoding ou Label Encoding.
    #
    # Caso houvesse variáveis categóricas nominais (sem ordem), seria aplicado
    # One-Hot Encoding para evitar a imposição de uma ordem artificial.
    # Para variáveis ordinais, Label Encoding ou Ordinal Encoding seriam adequados.
    # -------------------------------------------------------
    # 2.3  Escalonamento numérico
    # -------------------------------------------------------
    #   - insu: 34 outliers (IQR), com distribuição fortemente assimétrica
    #   - pedi: 29 outliers (IQR), cauda longa à direita
    #   - mass: 19 outliers (IQR)
    # O RobustScaler é a escolha mais adequada para preservar a estrutura
    # dos dados sem que os outliers distorçam a escala das features.
    # Definir colunas de features (excluindo os dois targets)
    feature_cols = [c for c in df.columns if c not in [TARGET_CLF]]
    # Salvar DataFrame pré-escalonamento para visualização
    df_pre_scaling = df.copy()
    # Aplicar RobustScaler apenas nas features (não nos targets)
    scaler = RobustScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    # Estatísticas pós-escalonamento
    # Demonstrar efeito comparativo: RobustScaler vs StandardScaler
    std_scaler = StandardScaler()
    df_std = pd.DataFrame(
        std_scaler.fit_transform(df_pre_scaling[feature_cols]),
        columns=feature_cols
    )
    for col in feature_cols:
        r_med = df[col].median()
        r_iqr = df[col].quantile(0.75) - df[col].quantile(0.25)
        s_med = df_std[col].median()
        s_iqr = df_std[col].quantile(0.75) - df_std[col].quantile(0.25)
    # -------------------------------------------------------
    # 2.4  Visualizações do pré-processamento
    # -------------------------------------------------------
    # --- 2.4.1  Antes vs Depois da imputação (distribuições) ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.ravel()
    for idx, col in enumerate(zero_invalid_cols):
        ax = axes[idx]
        # Antes (com zeros)
        ax.hist(df_original[col], bins=30, alpha=0.5, color="#EF5350",
                edgecolor="white", label="Antes (com zeros)", density=True)
        # Depois (imputado)
        ax.hist(df_pre_scaling[col], bins=30, alpha=0.5, color="#42A5F5",
                edgecolor="white", label="Depois (KNN imputado)", density=True)
        ax.set_title(col, fontsize=12, fontweight="bold")
        ax.set_ylabel("Densidade")
        ax.legend(fontsize=8)
    axes[-1].set_visible(False)
    fig.suptitle("Distribuições Antes vs Depois da Imputação KNN",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/06_imputacao_antes_depois.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    # --- 2.4.2  Boxplots antes vs depois do escalonamento ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    # Antes
    bp1 = axes[0].boxplot([df_pre_scaling[c].dropna().values for c in feature_cols],
                           tick_labels=feature_cols, patch_artist=True, vert=True)
    for patch in bp1["boxes"]:
        patch.set_facecolor("#FFCC80")
        patch.set_edgecolor("#E65100")
    axes[0].set_title("Antes do Escalonamento", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Valor")
    axes[0].tick_params(axis="x", rotation=45)
    # Depois (RobustScaler)
    bp2 = axes[1].boxplot([df[c].values for c in feature_cols],
                           tick_labels=feature_cols, patch_artist=True, vert=True)
    for patch in bp2["boxes"]:
        patch.set_facecolor("#81C784")
        patch.set_edgecolor("#2E7D32")
    axes[1].set_title("Depois do Escalonamento (RobustScaler)", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Valor escalonado")
    axes[1].tick_params(axis="x", rotation=45)
    fig.suptitle("Efeito do Escalonamento nas Features",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/07_escalonamento_antes_depois.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    # --- 2.4.3  Mapa de correlação pós-processamento ---
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(df.corr(), dtype=bool))
    sns.heatmap(df.corr().round(2), annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, mask=mask, linewidths=0.5, ax=ax,
                cbar_kws={"shrink": 0.8})
    ax.set_title("Matriz de Correlação Pós-Processamento", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/08_correlacao_pos_processamento.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    # -------------------------------------------------------
    # 2.5  Sumário do pré-processamento
    # -------------------------------------------------------
    state.update(locals())
    return state
