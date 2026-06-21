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
# JUSTIFICATIVA:
# - A imputação por mediana (univariada) ignora as correlações entre atributos.
#   Por exemplo, `skin` e `insu` possuem r=0.44, e `mass` e `skin` possuem r=0.39.
#   Um valor imputado para `skin` deveria considerar os valores de `insu` e `mass`.
# - O KNN Imputer (k=5) utiliza os k vizinhos mais próximos (distância euclidiana)
#   no espaço das features não-ausentes para estimar o valor faltante, preservando
#   a estrutura multivariada dos dados.
# - k=5 é um valor padrão robusto: suficientemente grande para suavizar ruído,
#   mas pequeno o bastante para capturar padrões locais.
# - Para as colunas com baixa taxa de ausência (plas 0.7%, mass 1.4%, pres 4.6%),
#   o KNN Imputer produz resultados muito próximos da mediana condicional.
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
# JUSTIFICATIVA DA ESCOLHA DO ROBUSTSCALER:
#
# Três opções foram consideradas:
#
# 1. StandardScaler (Z-score): x' = (x - μ) / σ
#    - Assume distribuição aproximadamente normal.
#    - Sensível a outliers, pois μ e σ são afetados por valores extremos.
#
# 2. MinMaxScaler: x' = (x - x_min) / (x_max - x_min)
#    - Mapeia para [0, 1]. Extremamente sensível a outliers,
#      pois x_min e x_max podem ser valores atípicos.
#
# 3. RobustScaler: x' = (x - mediana) / IQR
#    - Utiliza mediana e intervalo interquartil (Q3 - Q1).
#    - ROBUSTO a outliers: mediana e IQR são estatísticas resistentes.
#    - A mediana tem ponto de ruptura (breakdown point) de 50%, enquanto
#      a média (usada pelo StandardScaler) tem breakdown point de 0%.
#
# DECISÃO: RobustScaler.
# Na Etapa 1, diagnosticamos outliers em múltiplos atributos:
#   - pres: 45 outliers (IQR), incluindo zeros inválidos residuais
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
# 3.1  Preparação dos dados para classificação
# -------------------------------------------------------
# Features (excluindo target de classificação)
# Nota: 'plas' é target de regressão, mas aqui é usada como feature para classificação
X_clf = df[feature_cols].copy()
y_clf = df[TARGET_CLF].copy()
# -------------------------------------------------------
# 3.2  Seleção de Features via Informação Mútua (Método Filtro)
# -------------------------------------------------------
# JUSTIFICATIVA DA TÉCNICA:
#
# A Informação Mútua (MI) mede a dependência estatística entre duas variáveis:
#
#   MI(X; Y) = ∑∑ p(x,y) · log[ p(x,y) / (p(x)·p(y)) ]
#
# Propriedades que justificam a escolha:
#
# 1. CAPTURA DEPENDÊNCIAS NÃO-LINEARES: Ao contrário da correlação de Pearson
#    (que mede apenas relações lineares), a MI detecta qualquer tipo de
#    dependência estatística — crucial em dados clínicos onde relações
#    fisiológicas são frequentemente não-lineares.
#
# 2. MODEL-AGNOSTIC: Sendo um método filtro, a MI é independente do modelo
#    downstream. Isto evita viés circular (usar o mesmo modelo para selecionar
#    features e depois avaliar com ele) e permite que as features selecionadas
#    beneficiem qualquer algoritmo.
#
# 3. MI ≥ 0 sempre, e MI = 0 se e somente se X e Y são independentes.
#    Valores maiores indicam maior dependência (relevância) da feature.
#
# 4. EFICIÊNCIA COMPUTACIONAL: O(n·d·k), onde n = amostras, d = features,
#    k = vizinhos (internamente usa estimador KNN de Kraskov et al., 2004).
#    Muito mais rápido que métodos wrapper (ex: RFE com CV).
# Calcular MI com múltiplas execuções para estabilidade
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
X_clf_selected = X_clf[selected_features]
# -------------------------------------------------------
# 3.3  Avaliação comparativa: todas features vs selecionadas
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
# --- Cenário 1: Todas as features ---
t0 = time.time()
rf_all = RandomForestClassifier(**rf_params)
cv_all = cross_validate(rf_all, X_clf, y_clf, cv=cv, scoring=scoring,
                        return_train_score=True, n_jobs=-1)
time_all = time.time() - t0
# --- Cenário 2: Features selecionadas ---
t0 = time.time()
rf_sel = RandomForestClassifier(**rf_params)
cv_sel = cross_validate(rf_sel, X_clf_selected, y_clf, cv=cv, scoring=scoring,
                        return_train_score=True, n_jobs=-1)
time_sel = time.time() - t0
# Exibir resultados comparativos
results_comparison = {}
for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
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
# JUSTIFICATIVA DA ARQUITETURA:
#
# A arquitetura foi escolhida considerando:
# 1. TAMANHO DO DATASET: 768 amostras é relativamente pequeno para deep learning.
#    Redes muito profundas/largas causariam overfitting severo.
# 2. DIMENSIONALIDADE: Com apenas 4 features selecionadas, a capacidade necessária
#    é limitada. Uma rede com 3 camadas ocultas (64→32→16) oferece capacidade
#    suficiente para capturar interações não-lineares sem excesso de parâmetros.
# 3. PIRÂMIDE DECRESCENTE: A redução progressiva de neurônios (64→32→16) é uma
#    heurística clássica que força a rede a aprender representações cada vez mais
#    compactas, atuando como regularização implícita.
# 4. TOTAL DE PARÂMETROS: ~3.500 (muito menor que o nº de amostras de treino ≈614),
#    mantendo a relação amostras/parâmetros > 1.
# Hiperparâmetros explícitos
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
# JUSTIFICATIVA DA ARQUITETURA E HIPERPARÂMETROS:
# 1. Arquitetura (64, 32): Regressão muitas vezes precisa de menos camadas
#    que classificação para evitar decoreba excessiva, mantendo um funil clássico.
# 2. Ativação (relu): Padrão ótimo para não-linearidades em redes profundas.
# 3. Otimizador (adam): Adaptação automática da taxa de aprendizado.
# 4. Early Stopping: Habilitado para evitar overfitting, parando quando o 
#    erro de validação não melhorar por 20 épocas.
# 5. Para melhorar a convergência com target não escalonado, mantemos um 
#    learning rate conservador (0.001) e mais épocas (1000).
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
    score = cross_val_score(model, X_clf_sel, y_clf_full, n_jobs=-1, cv=5, scoring='roc_auc')
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