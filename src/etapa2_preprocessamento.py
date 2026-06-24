"""
Trabalho de IA/ML - Pipeline de Machine Learning
Autores: Gabriel Stiegemeier e Guilherme Einloft
Cadeira: Inteligência Artificial
Dataset: Pima Indians Diabetes (UCI / OpenML)
Descrição: Etapa 2 - Pré-processamento, imputação (KNN) e escalonamento numérico.
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
    # Receber splits da Etapa 1
    df_train = state.get("df_train").copy()
    df_test = state.get("df_test").copy()
    # Preservar cópias originais para comparações visuais
    df_train_original = df_train.copy()
    # -------------------------------------------------------
    # 2.1  Tratamento de valores ausentes (zeros inválidos → NaN → imputação)
    # -------------------------------------------------------
    # Passo 1: Substituir zeros biologicamente impossíveis por NaN
    zero_invalid_cols = ["plas", "pres", "skin", "insu", "mass"]
    for col in zero_invalid_cols:
        df_train[col] = df_train[col].replace(0, np.nan)
        df_test[col] = df_test[col].replace(0, np.nan)
    total_nan_train = df_train.isna().sum()
    # Passo 2: Imputação via KNN Imputer (k=5)
    #
    # - IMPORTANTE: o imputer é fit apenas no conjunto de TREINO para evitar
    #   data leakage. O conjunto de teste é transformado com o imputer já treinado.
    # Separar features e targets antes da imputação
    features_for_imputation = [c for c in df_train.columns if c != TARGET_CLF]
    imputer = KNNImputer(n_neighbors=5, weights="uniform")
    df_train[features_for_imputation] = imputer.fit_transform(df_train[features_for_imputation])
    df_test[features_for_imputation] = imputer.transform(df_test[features_for_imputation])
    # Verificar que não restam NaN
    nan_after_train = df_train.isna().sum().sum()
    nan_after_test = df_test.isna().sum().sum()
    assert nan_after_train == 0, "Erro: ainda existem valores ausentes no treino!"
    assert nan_after_test == 0, "Erro: ainda existem valores ausentes no teste!"
    # Comparar estatísticas antes e depois da imputação (apenas treino)
    for col in zero_invalid_cols:
        mean_before = df_train_original[col].mean()
        mean_after = df_train[col].mean()
        med_before = df_train_original[col].median()
        med_after = df_train[col].median()
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
    #   - IMPORTANTE: o scaler é fit apenas no conjunto de TREINO.
    #     O conjunto de teste é transformado com o scaler já treinado.
    # Definir colunas de features (excluindo o target de classificação)
    feature_cols = [c for c in df_train.columns if c not in [TARGET_CLF]]
    # Salvar DataFrames pré-escalonamento (imputados mas não escalonados)
    df_train_pre_scaling = df_train.copy()
    df_test_pre_scaling = df_test.copy()
    # Aplicar RobustScaler: fit no TREINO, transform em ambos
    scaler = RobustScaler()
    df_train[feature_cols] = scaler.fit_transform(df_train[feature_cols])
    df_test[feature_cols] = scaler.transform(df_test[feature_cols])
    # Estatísticas pós-escalonamento
    # Demonstrar efeito comparativo: RobustScaler vs StandardScaler (apenas no treino)
    std_scaler = StandardScaler()
    df_std = pd.DataFrame(
        std_scaler.fit_transform(df_train_pre_scaling[feature_cols]),
        columns=feature_cols
    )
    for col in feature_cols:
        r_med = df_train[col].median()
        r_iqr = df_train[col].quantile(0.75) - df_train[col].quantile(0.25)
        s_med = df_std[col].median()
        s_iqr = df_std[col].quantile(0.75) - df_std[col].quantile(0.25)
    print(f"  -> Pré-processamento: treino {len(df_train)}, teste {len(df_test)}")
    # -------------------------------------------------------
    # 2.4  Visualizações do pré-processamento (usando dados de TREINO)
    # -------------------------------------------------------
    # --- 2.4.1  Antes vs Depois da imputação (distribuições) ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.ravel()
    for idx, col in enumerate(zero_invalid_cols):
        ax = axes[idx]
        # Antes (com zeros)
        ax.hist(df_train_original[col], bins=30, alpha=0.5, color="#EF5350",
                edgecolor="white", label="Antes (com zeros)", density=True)
        # Depois (imputado)
        ax.hist(df_train_pre_scaling[col], bins=30, alpha=0.5, color="#42A5F5",
                edgecolor="white", label="Depois (KNN imputado)", density=True)
        ax.set_title(col, fontsize=12, fontweight="bold")
        ax.set_ylabel("Densidade")
        ax.legend(fontsize=8)
    axes[-1].set_visible(False)
    fig.suptitle("Distribuições Antes vs Depois da Imputação KNN (Treino)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/06_imputacao_antes_depois.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    # --- 2.4.2  Boxplots antes vs depois do escalonamento ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    # Antes
    bp1 = axes[0].boxplot([df_train_pre_scaling[c].dropna().values for c in feature_cols],
                           tick_labels=feature_cols, patch_artist=True, vert=True)
    for patch in bp1["boxes"]:
        patch.set_facecolor("#FFCC80")
        patch.set_edgecolor("#E65100")
    axes[0].set_title("Antes do Escalonamento", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Valor")
    axes[0].tick_params(axis="x", rotation=45)
    # Depois (RobustScaler)
    bp2 = axes[1].boxplot([df_train[c].values for c in feature_cols],
                           tick_labels=feature_cols, patch_artist=True, vert=True)
    for patch in bp2["boxes"]:
        patch.set_facecolor("#81C784")
        patch.set_edgecolor("#2E7D32")
    axes[1].set_title("Depois do Escalonamento (RobustScaler)", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Valor escalonado")
    axes[1].tick_params(axis="x", rotation=45)
    fig.suptitle("Efeito do Escalonamento nas Features (Treino)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/07_escalonamento_antes_depois.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    # --- 2.4.3  Mapa de correlação pós-processamento ---
    fig, ax = plt.subplots(figsize=(10, 8))
    corr_train = df_train.corr()
    mask = np.triu(np.ones_like(corr_train, dtype=bool))
    sns.heatmap(corr_train.round(2), annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, mask=mask, linewidths=0.5, ax=ax,
                cbar_kws={"shrink": 0.8})
    ax.set_title("Matriz de Correlação Pós-Processamento (Treino)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/08_correlacao_pos_processamento.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    # -------------------------------------------------------
    # 2.5  Sumário do pré-processamento
    # -------------------------------------------------------
    state.update(locals())
    return state
