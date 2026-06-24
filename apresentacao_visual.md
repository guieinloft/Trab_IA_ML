# Trabalho Final — Inteligência Artificial
## Aprendizado Supervisionado com Seleção de Features e Otimização de Hiperparâmetros

**Autores:** Gabriel Stiegemeier · Guilherme Einloft  
**Dataset:** Pima Indians Diabetes (UCI / OpenML)

---

## 📋 Dataset — Pima Indians Diabetes

| Característica | Valor |
|---|---|
| Fonte | UCI / OpenML (id=37) |
| Amostras | 768 |
| Atributos | 8 numéricos |
| População | Mulheres indígenas Pima, ≥ 21 anos |
| Alvo (classif.) | `Outcome` → diabetes (sim/não) |
| Alvo (regressão) | `plas` → glicose plasmática (mg/dL) |
| Classes | 500 negativas (65%) · 268 positivas (35%) |

**Atributos:** `preg` · `plas` · `pres` · `skin` · `insu` · `mass` · `pedi` · `age`

**Problemas identificados:**
- Zeros inválidos em 5 colunas (`insu` = 48,7% de zeros)
- Outliers (`insu`: 34 · `pedi`: 29 · `mass`: 19)
- Desbalanceamento leve (65/35)

---

## 🗂️ Estrutura do Código

```
Trab_IA_ML/
├── src/
│   ├── main.py                        ← Orquestrador
│   ├── etapa1_exploracao.py           ← Análise exploratória
│   ├── etapa2_preprocessamento.py     ← Pré-processamento
│   ├── etapa3_selecao.py              ← Seleção de features + SHAP
│   ├── etapa4_classificacao.py        ← Classificação binária (MLP)
│   ├── etapa4b_classificacao_multiclasse.py
│   ├── etapa5_regressao.py            ← Regressão (MLP)
│   ├── etapa6_otimizacao.py           ← Optuna
│   └── etapa7_overfitting.py          ← Regularização
├── output/                            ← Gráficos gerados
├── relatorio.md
└── requirements.txt
```

---

## 🔄 Pipeline de Execução (`main.py`)

```
state = {}
  │
  ▼
┌────────────────────────────────────┐
│  Etapa 1 — Análise Exploratória   │  Carrega dataset, distribuições,
│                                    │  boxplots, correlação, outliers
└──────────────┬─────────────────────┘
               ▼
┌────────────────────────────────────┐
│  Etapa 2 — Pré-processamento      │  Split 80/20 → KNN Imputer (k=5)
│                                    │  → RobustScaler
└──────────────┬─────────────────────┘
               ▼
┌────────────────────────────────────┐
│  Etapa 3 — Seleção de Features    │  Informação Mútua (50 rounds)
│                                    │  8 → 3 features + SHAP
└──────────────┬─────────────────────┘
               ▼
┌────────────────────────────────────┐
│  Etapa 4 — Classif. Binária (MLP) │  64→32→16, Adam, Early Stop
│          + XGBoost e SVM          │  ROC-AUC = 0.79
└──────────────┬─────────────────────┘
               ▼
┌────────────────────────────────────┐
│  Etapa 4b — Classif. Multiclasse  │  3 classes (ADA)
│                                    │  F1 Macro = 0.64
└──────────────┬─────────────────────┘
               ▼
┌────────────────────────────────────┐
│  Etapa 5 — Regressão (MLP)        │  64→32, prediz glicose
│                                    │  R² = 0.42
└──────────────┬─────────────────────┘
               ▼
┌────────────────────────────────────┐
│  Etapa 6 — Optuna                 │  30 trials → ROC-AUC CV = 0.83
└──────────────┬─────────────────────┘
               ▼
┌────────────────────────────────────┐
│  Etapa 7 — Regularização          │  Sem reg. vs Com reg.
│                                    │  L2 + Early Stop + Simplif.
└────────────────────────────────────┘
               ▼
           ✅ Concluído
```

---

## ⚙️ Pré-processamento

| Etapa | Técnica | Por quê? |
|---|---|---|
| Split | 80% treino / 20% teste (estratificado) | Evitar data leakage |
| Imputação | KNN Imputer (k=5) | Preserva correlações entre variáveis |
| Escalonamento | RobustScaler (mediana + IQR) | Robusto a outliers |

---

## 🎯 Seleção de Features — Informação Mútua

| Rank | Feature | MI médio |
|---|---|---|
| 1 | `plas` | 0.1329 |
| 2 | `insu` | 0.1293 |
| 3 | `mass` | 0.0950 |
| 4 | `age` | 0.0555 |
| 5 | `skin` | 0.0420 |
| 6 | `preg` | 0.0228 |
| 7 | `pres` | 0.0201 |
| 8 | `pedi` | 0.0124 |

**Resultado:** 8 → 3 features (`plas`, `insu`, `mass`)  
**SHAP confirma:** Spearman ρ = 0.93 (p < 0.001)  
**Impacto:** −4% ROC-AUC com 62,5% menos features

---

## 🧠 Modelos e Resultados

### Classificação Binária — MLP (64→32→16)

| Hiperparâmetro | Valor |
|---|---|
| Camadas ocultas | (64, 32, 16) |
| Ativação | ReLU |
| Otimizador | Adam (lr=0.001) |
| Early Stopping | 30 épocas paciência |
| Batch size | 32 |

### Resultados Comparativos (Teste)

| Tarefa | Modelo | Métrica | Valor |
|---|---|---|---|
| Classif. Binária | MLP (64→32→16) | ROC-AUC | **0.79** |
| Classif. Multiclasse | MLP (64→32→16) | F1 Macro | **0.64** |
| Regressão | MLP (64→32) | R² | **0.42** |
|  |  | MAE | **18.6 mg/dL** |

---

## 🔧 Optuna — Otimização de Hiperparâmetros

| | Valor |
|---|---|
| Trials | 30 |
| Função objetivo | Maximizar ROC-AUC (5-fold CV) |
| Melhor ROC-AUC (CV) | **0.8344** |
| ROC-AUC no teste | 0.7811 |
| Delta vs. original | −0.009 |

**Conclusão:** modelo original já estava em boa região → limitação é dos dados (768 amostras)

---

## 🛡️ Regularização vs. Overfitting

| | Sem Regularização | Com Regularização |
|---|---|---|
| Arquitetura | (256, 128, 64) | (64, 32) |
| Alpha (L2) | 0.0 | 0.01 |
| Early Stopping | Desabilitado | 15 épocas |
| **Accuracy** | 0.708 | **0.727** |
| **F1-Score** | 0.536 | **0.544** |

**3 técnicas:** Simplificação estrutural + L2 + Early Stopping

---

## ✅ Conclusão

- Pipeline **completo e funcional** — `python main.py` executa tudo
- Cuidado com **data leakage** (MI dentro de cada fold da CV)
- **SHAP** corroborou a seleção de features (ρ = 0.93)
- Desempenho limitado pelo **volume dos dados**, não pela modelagem
- Código **modular e reprodutível**
