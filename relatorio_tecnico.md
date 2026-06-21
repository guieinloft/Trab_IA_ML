# Relatório Técnico — Pipeline Completo de Machine Learning

**Projeto:** Predição de Diabetes com Aprendizado de Máquina  
**Dataset:** Pima Indians Diabetes Database  
**Autor:** Cientista de Dados Sênior  
**Data de início:** 21/06/2026  

---

## Etapa 1 — Carga do Dataset e Análise Exploratória Inicial

### 1.1 Descrição Formal do Problema

O presente projeto tem como objetivo construir um pipeline completo de Machine Learning para a análise e predição de diabetes mellitus tipo 2 em mulheres de herança indígena Pima (Arizona, EUA). A partir de um único conjunto de dados, formulam-se **dois problemas complementares**:

1. **Classificação binária:** Predizer se uma paciente possui ou não diagnóstico de diabetes (variável `Outcome`).
2. **Regressão:** Predizer a concentração de glicose plasmática em jejum (variável `plas`), um indicador clínico contínuo diretamente associado ao diagnóstico de diabetes.

### 1.2 Procedência e Justificativa da Escolha do Dataset

| Item | Detalhe |
|---|---|
| **Nome** | Pima Indians Diabetes Database |
| **Fonte original** | National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK) |
| **Repositório** | OpenML (ID = 37); também disponível no UCI Machine Learning Repository e Kaggle |
| **Referência** | Smith, J.W., Everhart, J.E., Dickson, W.C., Knowler, W.C., & Johannes, R.S. (1988). *Using the ADAP learning algorithm to forecast the onset of diabetes mellitus.* |
| **Licença** | Domínio público (CC0) |

**Justificativas para a escolha:**

1. **Relevância clínica:** Diabetes mellitus tipo 2 é uma das doenças crônicas mais prevalentes mundialmente, afetando mais de 500 milhões de pessoas (IDF, 2021). Modelos preditivos nesta área têm impacto direto em saúde pública.
2. **Dupla formulação:** O dataset contém naturalmente uma variável categórica binária (`Outcome`) e diversas variáveis contínuas clinicamente relevantes (e.g., `plas` — glicose plasmática), permitindo a formulação simultânea de problemas de classificação e regressão.
3. **Confiabilidade e reprodutibilidade:** Os dados foram coletados pelo NIDDK sob protocolo clínico rigoroso. O dataset é amplamente utilizado na literatura acadêmica, garantindo comparabilidade dos resultados.
4. **Tamanho adequado:** Com 768 amostras e 8 atributos preditivos, o dataset apresenta dimensionalidade compatível com a aplicação de algoritmos clássicos e permite análises estatísticas robustas.

### 1.3 Descrição dos Dados

#### 1.3.1 Dimensões

- **Amostras:** 768
- **Atributos preditivos:** 8 (todos numéricos)
- **Variáveis-alvo:** 2 (uma categórica e uma contínua)

#### 1.3.2 Descrição dos Atributos

| Atributo | Descrição | Tipo | Unidade |
|---|---|---|---|
| `preg` | Número de gestações | Discreto (inteiro) | contagem |
| `plas` | Concentração de glicose plasmática (teste de tolerância oral, 2h) | Contínuo | mg/dL |
| `pres` | Pressão arterial diastólica | Contínuo | mmHg |
| `skin` | Espessura da dobra cutânea do tríceps | Contínuo | mm |
| `insu` | Insulina sérica (2h) | Contínuo | µU/mL |
| `mass` | Índice de massa corporal (IMC) | Contínuo | kg/m² |
| `pedi` | Função de pedigree de diabetes | Contínuo | — |
| `age` | Idade | Discreto (inteiro) | anos |
| `Outcome` | Diagnóstico de diabetes | Binário (0/1) | — |

#### 1.3.3 Definição das Variáveis-Alvo

| Problema | Variável | Tipo | Descrição |
|---|---|---|---|
| **Classificação** | `Outcome` | Binária (0 ou 1) | 0 = teste negativo para diabetes; 1 = teste positivo |
| **Regressão** | `plas` | Contínua | Concentração de glicose plasmática em mg/dL |

**Justificativa para `plas` como alvo de regressão:** A glicose plasmática é o principal marcador bioquímico utilizado pela OMS e pela ADA (American Diabetes Association) para o diagnóstico de diabetes. Valores ≥ 200 mg/dL no teste de tolerância oral à glicose (2h) são diagnósticos. Prever esta variável contínua é clinicamente útil para triagem e monitoramento, complementando o diagnóstico binário.

### 1.4 Estatísticas Descritivas

| Atributo | Média | Desvio Padrão | Mínimo | Q1 | Mediana | Q3 | Máximo |
|---|---|---|---|---|---|---|---|
| `preg` | 3.85 | 3.37 | 0.00 | 1.00 | 3.00 | 6.00 | 17.00 |
| `plas` | 120.89 | 31.97 | 0.00 | 99.00 | 117.00 | 140.25 | 199.00 |
| `pres` | 69.11 | 19.36 | 0.00 | 62.00 | 72.00 | 80.00 | 122.00 |
| `skin` | 20.54 | 15.95 | 0.00 | 0.00 | 23.00 | 32.00 | 99.00 |
| `insu` | 79.80 | 115.24 | 0.00 | 0.00 | 30.50 | 127.25 | 846.00 |
| `mass` | 31.99 | 7.88 | 0.00 | 27.30 | 32.00 | 36.60 | 67.10 |
| `pedi` | 0.47 | 0.33 | 0.08 | 0.24 | 0.37 | 0.63 | 2.42 |
| `age` | 33.24 | 11.76 | 21.00 | 24.00 | 29.00 | 41.00 | 81.00 |

### 1.5 Distribuição das Classes (Classificação)

| Classe | Contagem | Proporção |
|---|---|---|
| 0 — Negativo | 500 | 65.10% |
| 1 — Positivo | 268 | 34.90% |

**Razão minoria/maioria:** 0.536 (268/500).

**Análise:** O dataset apresenta um **desbalanceamento moderado**, com a classe positiva representando ~35% das amostras. Embora não seja severo (razão > 0.5), este desbalanceamento pode introduzir viés em modelos que otimizam acurácia global. Estratégias como SMOTE, ajuste de pesos de classe (`class_weight='balanced'`), ou métricas adequadas (F1-score, AUC-ROC) deverão ser consideradas nas etapas de modelagem.

### 1.6 Diagnóstico de Valores Ausentes

O dataset não contém valores `NaN` explícitos. Entretanto, para cinco atributos, **o valor 0 é biologicamente impossível** e constitui um proxy para dados ausentes — uma prática comum em datasets clínicos mais antigos.

| Atributo | Zeros (ausentes implícitos) | Proporção |
|---|---|---|
| `plas` (glicose) | 5 | 0.7% |
| `pres` (pressão) | 35 | 4.6% |
| `skin` (espessura cutânea) | 227 | 29.6% |
| `insu` (insulina) | 374 | 48.7% |
| `mass` (IMC) | 11 | 1.4% |

**Observações críticas:**
- `insu` e `skin` apresentam taxas de ausência muito elevadas (48.7% e 29.6%, respectivamente). Técnicas de imputação (e.g., KNN imputer, imputação por mediana condicionada à classe) ou a remoção dessas features devem ser avaliadas na etapa de pré-processamento.
- `plas`, `pres` e `mass` possuem poucos valores ausentes e podem ser imputados com menor risco de viés.
- Os atributos `preg`, `pedi` e `age` não apresentam este problema (0 é um valor válido para gestações, e as demais variáveis têm faixas naturais que não incluem 0 como anomalia).

### 1.7 Diagnóstico de Outliers

Aplicaram-se dois métodos complementares para detecção de outliers:

1. **Método IQR (Interquartile Range):** Valores abaixo de Q1 − 1.5×IQR ou acima de Q3 + 1.5×IQR.
2. **Z-score:** Valores com |Z| > 3 (mais de 3 desvios padrão da média).

| Atributo | Outliers (IQR) | Outliers (Z > 3) |
|---|---|---|
| `preg` | 4 | 4 |
| `plas` | 5 | 5 |
| `pres` | 45 | 35 |
| `skin` | 1 | 1 |
| `insu` | 34 | 18 |
| `mass` | 19 | 14 |
| `pedi` | 29 | 11 |
| `age` | 9 | 5 |

**Observações:**
- `pres` (pressão arterial) apresenta o maior número de outliers por IQR (45), em parte inflado pelos 35 valores nulos (0 mmHg) que são fisicamente impossíveis e são detectados como outliers inferiores. Após tratamento dos valores ausentes, espera-se redução significativa.
- `insu` (insulina) também apresenta muitos outliers, com distribuição fortemente assimétrica à direita (máximo = 846 µU/mL contra mediana = 30.5 µU/mL), o que é coerente com a fisiologia — pacientes diabéticos podem apresentar hiperinsulinemia compensatória.
- `pedi` possui outliers com cauda longa à direita, natural para uma função de risco genético.
- A estratégia de tratamento de outliers deverá ser definida na etapa de pré-processamento, considerando: (a) winsorização, (b) transformação logarítmica, ou (c) uso de modelos robustos a outliers.

### 1.8 Análise de Correlação

#### 1.8.1 Correlação entre Atributos Preditivos

Nenhum par de atributos apresentou correlação de Pearson com |r| > 0.7, indicando **ausência de multicolinearidade severa**. Os pares com correlação mais notável são:

| Par | r |
|---|---|
| `skin` ↔ `insu` | 0.44 |
| `skin` ↔ `mass` | 0.39 |
| `insu` ↔ `plas` | 0.33 |
| `mass` ↔ `pres` | 0.28 |
| `preg` ↔ `age` | 0.54 |

A correlação `preg ↔ age` (r = 0.54) é esperada biologicamente (mulheres mais velhas acumulam mais gestações). Não constitui problema para a maioria dos modelos, mas pode afetar modelos lineares se não tratada.

#### 1.8.2 Correlação com a Variável de Classificação (`Outcome`)

| Atributo | Correlação com `Outcome` |
|---|---|
| `plas` | **0.467** |
| `mass` | 0.293 |
| `age` | 0.238 |
| `preg` | 0.222 |
| `pedi` | 0.174 |
| `insu` | 0.131 |
| `skin` | 0.075 |
| `pres` | 0.065 |

A **glicose plasmática (`plas`)** é, como esperado, o atributo com maior poder discriminante para o diagnóstico de diabetes, seguida pelo IMC (`mass`) e idade (`age`).

#### 1.8.3 Correlação com a Variável de Regressão (`plas`)

| Atributo | Correlação com `plas` |
|---|---|
| `insu` | 0.331 |
| `age` | 0.264 |
| `mass` | 0.221 |
| `pres` | 0.153 |
| `pedi` | 0.137 |
| `preg` | 0.129 |
| `skin` | 0.057 |

A insulina sérica (`insu`) apresenta a maior correlação com a glicose plasmática (r = 0.331), coerente com o mecanismo fisiológico da regulação glicêmica.

### 1.9 Visualizações Geradas

As seguintes visualizações foram geradas e salvas no diretório `src/output/`:

| Arquivo | Descrição |
|---|---|
| `01_distribuicoes.png` | Histogramas de distribuição de todos os atributos |
| `02_boxplots_por_classe.png` | Boxplots de cada atributo estratificados por `Outcome` |
| `03_correlacao.png` | Mapa de calor da matriz de correlação de Pearson |
| `04_regressao_target.png` | Distribuição e Q-Q plot da variável de regressão (`plas`) |
| `05_distribuicao_classes.png` | Distribuição da variável de classificação (`Outcome`) |

### 1.10 Sumário e Próximos Passos

**Principais achados da Etapa 1:**

1. O dataset Pima Indians Diabetes é adequado para a formulação simultânea de classificação binária e regressão.
2. Há **valores ausentes implícitos** (codificados como 0) em 5 atributos, com destaque para `insu` (48.7%) e `skin` (29.6%).
3. O **desbalanceamento de classes** é moderado (razão ≈ 0.54) e deverá ser tratado.
4. **Outliers** estão presentes em múltiplos atributos, especialmente `pres` e `insu`.
5. Não há **multicolinearidade severa** entre os atributos preditivos.
6. A variável `plas` é o preditor mais forte para o `Outcome` (r = 0.467) e será excluída como feature quando usada como alvo de regressão.

**Próxima etapa:** Pré-processamento dos dados — tratamento de valores ausentes, tratamento de outliers, normalização/padronização e engenharia de atributos.

---

## Etapa 2 — Pré-processamento

### 2.1 Tratamento de Valores Ausentes

#### 2.1.1 Identificação dos Valores Ausentes

Conforme diagnosticado na Etapa 1, o dataset não contém valores `NaN` explícitos, mas utiliza **zeros como proxy para dados ausentes** em cinco atributos onde o valor 0 é biologicamente impossível. O primeiro passo do pré-processamento consistiu em substituir esses zeros por `NaN`:

| Atributo | Zeros → NaN | Proporção |
|---|---|---|
| `plas` (glicose) | 5 | 0.7% |
| `pres` (pressão) | 35 | 4.6% |
| `skin` (espessura cutânea) | 227 | 29.6% |
| `insu` (insulina) | 374 | 48.7% |
| `mass` (IMC) | 11 | 1.4% |
| **Total** | **652** | **9.4% do dataset** |

#### 2.1.2 Escolha do Método de Imputação: KNN Imputer (k=5)

**Método escolhido:** `KNNImputer` do scikit-learn com `n_neighbors=5` e `weights='uniform'`.

**Alternativas avaliadas e descartadas:**

| Método | Descrição | Limitação |
|---|---|---|
| Remoção de linhas | Excluir amostras com NaN | Eliminaria 48.7% do dataset (linhas com `insu` = 0), reduzindo severamente o tamanho amostral |
| Remoção de colunas | Excluir `insu` e `skin` | Perderia informação preditiva clinicamente relevante (insulina correlaciona-se com glicose, r = 0.33) |
| Imputação por média | Substituir por µ da coluna | Sensível a outliers; ignora relações entre features |
| Imputação por mediana | Substituir pela mediana da coluna | Robusta a outliers, mas **univariada** — ignora correlações entre atributos |
| Imputação por moda | Substituir pela moda | Inadequada para variáveis contínuas |

**Justificativa formal para o KNN Imputer:**

1. **Preservação da estrutura multivariada:** As correlações entre atributos são clinicamente significativas (e.g., `skin ↔ insu`: r = 0.44; `skin ↔ mass`: r = 0.39; `insu ↔ plas`: r = 0.33). O KNN Imputer estima cada valor ausente como a média ponderada dos k vizinhos mais próximos no espaço das features não-ausentes, preservando essas relações.

2. **Formulação matemática:** Para uma amostra $x_i$ com valor ausente na feature $j$, o KNN Imputer computa:

   $$\hat{x}_{ij} = \frac{1}{k} \sum_{m \in \mathcal{N}_k(i)} x_{mj}$$

   onde $\mathcal{N}_k(i)$ é o conjunto dos k vizinhos mais próximos de $x_i$ (usando distância euclidiana nas features disponíveis). Isto é equivalente a uma interpolação local não-paramétrica.

3. **Robustez ao percentual de ausência:** Mesmo com 48.7% de ausência em `insu`, o KNN Imputer pode operar usando as 7 features restantes (que possuem < 5% de ausência cada) para encontrar vizinhos similares.

4. **Escolha de k = 5:** Valor canônico na literatura que equilibra viés-variância. Valores menores (k=1,2) seriam sensíveis a ruído; valores maiores (k=10+) supersuavizariam as estimativas.

#### 2.1.3 Resultados da Imputação

Após a imputação, **zero valores NaN restaram** no dataset. A comparação das estatísticas antes e depois confirma que a imputação produziu valores biologicamente plausíveis:

| Atributo | Média (antes) | Média (depois) | Mediana (antes) | Mediana (depois) |
|---|---|---|---|---|
| `plas` | 120.89 | 121.60 | 117.00 | 117.00 |
| `pres` | 69.11 | 72.38 | 72.00 | 72.00 |
| `skin` | 20.54 | 29.10 | 23.00 | 29.00 |
| `insu` | 79.80 | 153.06 | 30.50 | 132.90 |
| `mass` | 31.99 | 32.42 | 32.00 | 32.09 |

**Observações:**
- Para `plas`, `pres` e `mass` (< 5% de ausência), as estatísticas mudaram minimamente, indicando impacto negligível da imputação.
- Para `skin` e `insu` (alta ausência), as médias e medianas aumentaram significativamente após a imputação. Isto é **esperado e correto**: os zeros que representavam ausência deflacionavam artificialmente as medidas de tendência central. Os valores imputados refletem estimativas fisiologicamente mais realistas.
- A mediana de `insu` passou de 30.5 para 132.9 µU/mL, o que é clinicamente coerente — o valor de 30.5 era enviesado pelos 374 zeros (48.7%), e o novo valor está dentro da faixa normal de insulina em jejum/pós-carga.

### 2.2 Codificação de Variáveis Categóricas

#### 2.2.1 Análise de Tipos de Dados

Após inspeção dos tipos de dados:

| Tipo | Colunas | Contagem |
|---|---|---|
| `float64` | `preg`, `plas`, `pres`, `skin`, `insu`, `mass`, `pedi`, `age` | 8 |
| `int64` | `Outcome` | 1 |

**Conclusão:** Todos os 8 atributos preditivos são **numéricos** (contínuos ou discretos). A variável-alvo de classificação (`Outcome`) já está codificada como 0/1 (binário inteiro). **Nenhuma codificação categórica é necessária.**

#### 2.2.2 Discussão Teórica

Embora não aplicável a este dataset, documentamos as estratégias que seriam utilizadas caso houvesse variáveis categóricas:

| Tipo de Variável | Técnica Recomendada | Justificativa |
|---|---|---|
| **Nominal** (sem ordem natural) | One-Hot Encoding | Evita a imposição de uma relação ordinal inexistente entre categorias. Cria variáveis dummy binárias (k-1 para evitar multicolinearidade) |
| **Ordinal** (com ordem natural) | Ordinal Encoding / Label Encoding | Preserva a relação de ordem entre categorias (e.g., "baixo" < "médio" < "alto") |
| **Binária** (2 categorias) | Label Encoding (0/1) | Suficiente e eficiente para duas classes. One-Hot seria redundante |
| **Alta cardinalidade** (> 20 categorias) | Target Encoding ou Frequency Encoding | One-Hot geraria dimensionalidade excessiva |

### 2.3 Escalonamento Numérico

#### 2.3.1 Análise Comparativa dos Métodos de Escalonamento

Três métodos de escalonamento foram avaliados:

| Método | Fórmula | Vantagens | Desvantagens |
|---|---|---|---|
| **StandardScaler** | $x' = \frac{x - \mu}{\sigma}$ | Centraliza em 0 com σ=1; ideal para distribuições normais | **Sensível a outliers** — µ e σ são distorcidos por valores extremos |
| **MinMaxScaler** | $x' = \frac{x - x_{min}}{x_{max} - x_{min}}$ | Mapeia para [0,1]; preserva a forma da distribuição | **Muito sensível a outliers** — um único valor extremo comprime toda a distribuição |
| **RobustScaler** | $x' = \frac{x - \tilde{x}}{IQR}$ | Usa mediana ($\tilde{x}$) e IQR; **robusto a outliers** | Não garante escala unitária de variância |

onde $\tilde{x}$ é a mediana e $IQR = Q3 - Q1$.

#### 2.3.2 Decisão: RobustScaler

**Escolha:** `RobustScaler` do scikit-learn.

**Justificativa baseada nos dados:**

Na Etapa 1, foram diagnosticados outliers significativos em múltiplos atributos:

| Atributo | Outliers (IQR) | Outliers (Z > 3) |
|---|---|---|
| `pres` | 45 | 35 |
| `insu` | 34 | 18 |
| `pedi` | 29 | 11 |
| `mass` | 19 | 14 |

A presença generalizada de outliers torna o StandardScaler inadequado, pois a média e o desvio padrão seriam inflados por valores extremos, comprimindo a maioria dos dados centrais. O MinMaxScaler seria ainda pior, pois depende diretamente de $x_{min}$ e $x_{max}$.

**Propriedade estatística chave:** A mediana possui **ponto de ruptura** (*breakdown point*) de 50%, significando que até metade das observações pode ser arbitrariamente extrema sem afetar o estimador. A média, em contraste, tem breakdown point de 0% — um único outlier pode distorcê-la arbitrariamente. Analogamente, o IQR é robusto a outliers nas caudas da distribuição.

#### 2.3.3 Resultados do Escalonamento

Após aplicar o RobustScaler, todas as features foram transformadas de forma que a **mediana = 0** e o **IQR = 1**:

| Feature | Mediana | IQR | Min | Max |
|---|---|---|---|---|
| `preg` | 0.000 | 1.000 | -0.600 | 2.800 |
| `plas` | 0.000 | 1.000 | -1.770 | 1.988 |
| `pres` | 0.000 | 1.000 | -3.000 | 3.125 |
| `skin` | 0.000 | 1.000 | -1.833 | 5.833 |
| `insu` | 0.000 | 1.000 | -1.163 | 6.974 |
| `mass` | 0.000 | 1.000 | -1.526 | 3.847 |
| `pedi` | 0.000 | 1.000 | -0.770 | 5.353 |
| `age` | 0.000 | 1.000 | -0.471 | 3.059 |

**Comparação RobustScaler vs StandardScaler:**

| Feature | Robust (mediana) | Robust (IQR) | Standard (mediana) | Standard (IQR) |
|---|---|---|---|---|
| `preg` | 0.000 | 1.000 | -0.251 | 1.485 |
| `plas` | 0.000 | 1.000 | -0.151 | 1.353 |
| `pres` | 0.000 | 1.000 | -0.031 | 1.312 |
| `skin` | 0.000 | 1.000 | -0.011 | 1.274 |
| `insu` | 0.000 | 1.000 | -0.205 | 1.041 |
| `mass` | 0.000 | 1.000 | -0.049 | 1.322 |
| `pedi` | 0.000 | 1.000 | -0.300 | 1.155 |
| `age` | 0.000 | 1.000 | -0.361 | 1.446 |

O StandardScaler apresenta medianas deslocadas de zero (especialmente `age` = -0.361 e `pedi` = -0.300), refletindo a assimetria das distribuições que afeta a média mas não a mediana. O RobustScaler, por definição, centraliza perfeitamente a mediana em zero.

### 2.4 Visualizações do Pré-processamento

| Arquivo | Descrição |
|---|---|
| `06_imputacao_antes_depois.png` | Histogramas sobrepostos antes (com zeros) e depois (KNN imputado) para as 5 colunas com valores ausentes |
| `07_escalonamento_antes_depois.png` | Boxplots lado a lado das features antes e depois do RobustScaler |
| `08_correlacao_pos_processamento.png` | Matriz de correlação de Pearson após todo o pré-processamento |

### 2.5 Sumário do Pipeline de Pré-processamento

O pipeline de pré-processamento aplicado pode ser resumido como:

```
Dados Brutos (768 × 9)
    │
    ├─ 1. Substituição de zeros inválidos → NaN (652 células, 9.4%)
    │
    ├─ 2. Imputação multivariada KNN (k=5) → 0 NaN restantes
    │
    ├─ 3. Verificação de tipos (todos numéricos → sem encoding necessário)
    │
    └─ 4. Escalonamento RobustScaler (mediana=0, IQR=1)
           │
           └─ Dados Processados (768 × 9)
```

**Objetos preservados para etapas futuras:**
- `df` — DataFrame processado completo
- `df_original` — DataFrame original (para referência)
- `df_pre_scaling` — DataFrame após imputação, antes do escalonamento
- `scaler` — RobustScaler ajustado (para transformada inversa se necessário)
- `imputer` — KNNImputer ajustado
- `feature_cols` — lista das 8 features
- `TARGET_CLF` = `'Outcome'`, `TARGET_REG` = `'plas'`

**Próxima etapa:** Modelagem — treino e avaliação de modelos de classificação e regressão, com validação cruzada e otimização de hiperparâmetros.

---

## Etapa 3 — Seleção de Features (Classificação)

### 3.1 Técnica Escolhida: Informação Mútua (Método Filtro)

#### 3.1.1 Descrição da Técnica

A **Informação Mútua** (MI) é uma medida de dependência estatística oriunda da Teoria da Informação (Shannon, 1948). Para uma feature $X$ e a variável alvo $Y$, a MI é definida como:

$$MI(X; Y) = \sum_{x \in X} \sum_{y \in Y} p(x,y) \cdot \log\left[\frac{p(x,y)}{p(x) \cdot p(y)}\right]$$

A MI quantifica a **redução de incerteza** (em bits) sobre $Y$ ao observar $X$. Na implementação do scikit-learn (`mutual_info_classif`), a estimação é feita via o estimador KNN de Kraskov et al. (2004), que não assume qualquer distribuição paramétrica.

#### 3.1.2 Justificativa da Escolha

| Critério | Informação Mútua (Filtro) | Alternativas |
|---|---|---|
| **Tipo de dependência** | Captura relações **não-lineares** | Correlação de Pearson: apenas lineares |
| **Viés de modelo** | Model-agnostic (sem viés circular) | RFE/Embedded: viés do modelo usado na seleção |
| **Eficiência** | O(n·d·k) — rápido | RFE com CV: O(n·d²) — custoso |
| **Interpretabilidade** | MI ≥ 0; MI = 0 ⟺ independência | Feature importance: depende do modelo |
| **Estabilidade** | Controlada via múltiplas execuções | Varia com random seed do modelo |

A escolha de um método **filtro** evita o viés circular que ocorreria ao usar a importância de um Random Forest para selecionar features e depois avaliar o desempenho com o mesmo tipo de modelo.

#### 3.1.3 Estabilização dos Resultados

O estimador KNN de Kraskov possui componente estocástica (adição de ruído para evitar empates). Para obter estimativas estáveis, calculamos a MI em **50 execuções** com seeds distintas (0 a 49) e reportamos média ± desvio padrão.

### 3.2 Ranking de Importância e Critério de Seleção

| Rank | Feature | MI (média ± desvio) | Status |
|---|---|---|---|
| 1 | `plas` | 0.1284 ± 0.0080 | ✓ Selecionada |
| 2 | `mass` | 0.0836 ± 0.0087 | ✓ Selecionada |
| 3 | `insu` | 0.0818 ± 0.0049 | ✓ Selecionada |
| 4 | `age` | 0.0569 ± 0.0124 | ✓ Selecionada |
| 5 | `skin` | 0.0360 ± 0.0118 | ✗ Descartada |
| 6 | `preg` | 0.0344 ± 0.0169 | ✗ Descartada |
| 7 | `pedi` | 0.0126 ± 0.0037 | ✗ Descartada |
| 8 | `pres` | 0.0125 ± 0.0127 | ✗ Descartada |

**Critério de corte:** MI > MI_médio (0.0558). Este limiar adaptativo seleciona features cuja informação mútua com o alvo é superior à média global, descartando atributos com contribuição informacional abaixo da média.

**Resultado:** 8 features → **4 features selecionadas** (`plas`, `mass`, `insu`, `age`), **4 descartadas** (`skin`, `preg`, `pedi`, `pres`).

### 3.3 Avaliação Comparativa de Desempenho

#### 3.3.1 Configuração Experimental

- **Modelo base:** Random Forest Classifier (100 árvores, `class_weight='balanced'`, `random_state=42`)
- **Validação:** StratifiedKFold com k=10 (preserva proporção das classes em cada fold)
- **Métricas:** Accuracy, Precision, Recall, F1-Score, ROC-AUC

#### 3.3.2 Resultados

| Métrica | Todas (8 feat.) | Selecionadas (4 feat.) | Δ |
|---|---|---|---|
| **Accuracy** | 0.7551 ± 0.0540 | 0.7394 ± 0.0494 | −0.0157 |
| **Precision** | 0.6374 ± 0.0819 | 0.6171 ± 0.0777 | −0.0202 |
| **Recall** | 0.7048 ± 0.0838 | 0.7091 ± 0.0789 | **+0.0043** |
| **F1-Score** | 0.6676 ± 0.0742 | 0.6559 ± 0.0561 | −0.0117 |
| **ROC-AUC** | 0.8285 ± 0.0432 | 0.8112 ± 0.0538 | −0.0173 |
| **Tempo (s)** | 4.398 | 1.384 | **−68.5%** |

#### 3.3.3 Análise dos Resultados

**1. Desempenho:** A redução de 8 para 4 features causou uma queda **marginal** nas métricas (Accuracy: −1.6pp, F1: −1.2pp, AUC: −1.7pp). Estas diferenças estão **dentro do intervalo de confiança** dos desvios padrão (σ ≈ 0.05), indicando que não são estatisticamente significativas. O Recall melhorou ligeiramente (+0.4pp), sugerindo que o modelo com menos features captura um pouco melhor os casos positivos.

**2. Redução de tempo:** O tempo de treino/validação foi reduzido em **68.5%** (de 4.4s para 1.4s). Para datasets maiores ou pipelines com tuning extensivo, esta redução seria substancialmente mais significativa.

**3. Overfitting:** A diferença train−test accuracy permaneceu estável (0.244 vs 0.259), indicando que a seleção de features não agravou significativamente o overfitting. O Random Forest com `max_depth=None` naturalmente memoriza o treino (~100% train accuracy); a regularização ocorrerá na etapa de modelagem com tuning de hiperparâmetros.

**4. Interpretabilidade:** Reduzir de 8 para 4 features melhora substancialmente a interpretabilidade clínica do modelo. As 4 features selecionadas (`plas`, `mass`, `insu`, `age`) são todas variáveis com significado clínico direto e mensurável.

### 3.4 Análise SHAP (SHapley Additive exPlanations)

#### 3.4.1 Fundamentação Teórica

Os valores SHAP são baseados nos **valores de Shapley** da teoria cooperativa dos jogos (Shapley, 1953). Para um modelo $f$ e uma instância $x$, o valor SHAP da feature $j$ é:

$$\phi_j = \sum_{S \subseteq F \setminus \{j\}} \frac{|S|! \cdot (|F| - |S| - 1)!}{|F|!} \left[ f(S \cup \{j\}) - f(S) \right]$$

onde $F$ é o conjunto completo de features e $S$ é um subconjunto que não contém $j$. Isto garante:
- **Eficiência:** A soma dos valores SHAP iguala a diferença entre a predição e o valor base.
- **Simetria:** Features com contribuições iguais recebem valores iguais.
- **Nulidade:** Features irrelevantes recebem SHAP = 0.

Para Random Forest, o `TreeExplainer` do SHAP calcula os valores **exatos** (não aproximados) em tempo polinomial, explorando a estrutura das árvores.

#### 3.4.2 Ranking SHAP (Importância Global)

| Rank | Feature | Mean |SHAP| |
|---|---|---|
| 1 | `plas` | **0.1323** |
| 2 | `mass` | 0.0737 |
| 3 | `insu` | 0.0716 |
| 4 | `age` | 0.0662 |
| 5 | `pedi` | 0.0407 |
| 6 | `skin` | 0.0391 |
| 7 | `preg` | 0.0350 |
| 8 | `pres` | 0.0180 |

O SHAP confirma que `plas` (glicose plasmática) é **de longe** o atributo mais influente, com importância quase 2× superior ao segundo colocado (`mass`).

#### 3.4.3 Interpretação do SHAP Summary Plot

O SHAP Summary Plot (beeswarm) revela a **direção e magnitude** do impacto de cada feature:

- **`plas` (glicose):** Valores altos (rosa) empurram fortemente a predição para diabetes positiva (SHAP > 0); valores baixos (azul) empurram para negativa. Padrão clinicamente correto — hiperglicemia é o principal marcador de diabetes.
- **`mass` (IMC):** Valores altos de IMC aumentam a probabilidade de diabetes. Coerente com a literatura — obesidade é fator de risco para diabetes tipo 2.
- **`insu` (insulina):** Padrão bidirecional — tanto hiperinsulinemia quanto hipoinsulinemia afetam a predição, refletindo a fisiologia complexa (resistência insulínica vs. insuficiência pancreática).
- **`age` (idade):** Idade avançada aumenta risco. Coerente com epidemiologia — prevalência de DM2 cresce com a idade.

#### 3.4.4 Explicações Locais (Waterfall Plots)

**Amostra positiva (classe real: diabética):**
- P(diabetes) = 0.960 — modelo altamente confiante.
- Principais contribuições positivas: `plas` elevado e `mass` elevado empurraram a predição para cima.

**Amostra negativa (classe real: não-diabética):**
- P(diabetes) = 0.040 — modelo altamente confiante.
- Principais contribuições negativas: `plas` baixo e `mass` baixo empurraram a predição para baixo.

### 3.5 Comparação: Informação Mútua vs SHAP

#### 3.5.1 Concordância dos Rankings

| Feature | Rank MI | Rank SHAP | Concordância |
|---|---|---|---|
| `plas` | 1 | 1 | ✓ |
| `mass` | 2 | 2 | ✓ |
| `insu` | 3 | 3 | ✓ |
| `age` | 4 | 4 | ✓ |
| `skin` | 5 | 6 | ~ |
| `preg` | 6 | 7 | ~ |
| `pedi` | 7 | 5 | ~ |
| `pres` | 8 | 8 | ✓ |

- **Top-4 concordância:** 4/4 = **100%** — MI e SHAP concordam perfeitamente sobre as 4 features mais importantes.
- **Correlação de Spearman:** ρ = **0.929** (p = 0.0009) — correlação muito forte e estatisticamente significativa.

#### 3.5.2 Significado da Concordância

A concordância de 100% no top-4 entre dois métodos fundamentalmente diferentes valida fortemente a seleção realizada:

1. **MI** é um método **filtro**, model-agnostic, baseado em teoria da informação — mede dependência estatística sem treinar modelos.
2. **SHAP** é um método **post-hoc**, model-specific, baseado em teoria dos jogos — mede a contribuição marginal de cada feature nas predições de um modelo treinado.

O fato de ambos os métodos, com fundamentos teóricos completamente distintos, convergirem para o **mesmo ranking** das 4 features mais relevantes fornece evidência robusta de que:
- As features selecionadas (`plas`, `mass`, `insu`, `age`) capturam a informação genuinamente relevante para o diagnóstico de diabetes.
- A seleção não é um artefato de um método particular.
- O modelo (Random Forest) está aprendendo relações que refletem a dependência estatística real dos dados.

### 3.6 Visualizações Geradas

| Arquivo | Descrição |
|---|---|
| `09_shap_summary_beeswarm.png` | SHAP Summary Plot — beeswarm com direção e magnitude do impacto de cada feature |
| `10_shap_bar_importance.png` | SHAP Bar Plot — importância média absoluta dos atributos |
| `11_shap_waterfall_positivo.png` | Explicação local — amostra da classe positiva (diabética) |
| `12_shap_waterfall_negativo.png` | Explicação local — amostra da classe negativa (não-diabética) |
| `13_mi_vs_shap_ranking.png` | Comparação lado a lado dos rankings MI vs SHAP |
| `14_desempenho_comparativo.png` | Barras de desempenho — todas features vs selecionadas |

### 3.7 Sumário e Conclusões da Etapa 3

| Item | Valor |
|---|---|
| Atributos iniciais | 8 |
| Atributos finais | 4 (`plas`, `mass`, `insu`, `age`) |
| Técnica | Informação Mútua (filtro) |
| Perda de F1-Score | −1.2pp (não significativa) |
| Redução de tempo | −68.5% |
| Concordância MI vs SHAP | 100% (top-4) |
| Spearman ρ | 0.929 (p < 0.001) |

**Conclusão:** A seleção de features via Informação Mútua reduziu o espaço de atributos pela metade (8 → 4) com perda de desempenho negligível e ganho significativo em tempo computacional e interpretabilidade. A concordância de 100% com o SHAP — um método de natureza completamente distinta — valida tanto a técnica de seleção quanto a qualidade do modelo.

**Próxima etapa:** Modelagem completa — treino e avaliação de múltiplos modelos de classificação e regressão, com validação cruzada e otimização de hiperparâmetros.

---

## Etapa 4 — Classificação com MLP

### 4.1 Configuração Experimental

#### 4.1.1 Dados

- **Features utilizadas:** 4 (selecionadas na Etapa 3): `plas`, `mass`, `insu`, `age`
- **Variável alvo:** `Outcome` (binária: 0 = negativo, 1 = positivo)
- **Split:** 80% treino (614 amostras) / 20% teste (154 amostras), estratificado
- **Distribuição preservada:** Treino {0: 400, 1: 214} | Teste {0: 100, 1: 54}

### 4.2 Arquitetura da MLP

#### 4.2.1 Hiperparâmetros Definidos

| Hiperparâmetro | Valor | Justificativa |
|---|---|---|
| **Camadas ocultas** | (64, 32, 16) | Pirâmide decrescente — regularização implícita |
| **Ativação** | ReLU | Evita vanishing gradient; computacionalmente eficiente |
| **Otimizador** | Adam | Combina momentum + RMSProp; adaptativo por parâmetro |
| **Taxa de aprendizado** | 0.001 | Valor padrão do Adam (Kingma & Ba, 2015) |
| **Épocas máximas** | 500 | Limite superior; early stopping controla a parada real |
| **Batch size** | 32 | Compromisso entre estabilidade do gradiente e velocidade |
| **Early stopping** | Sim (paciência=30) | Previne overfitting; monitora accuracy de validação |
| **Validação interna** | 15% do treino | Reserva 92 amostras para monitoramento |

#### 4.2.2 Justificativa da Arquitetura

```
Entrada (4 features) → [64 neurônios, ReLU] → [32 neurônios, ReLU] → [16 neurônios, ReLU] → Saída (1, sigmoid)
```

**Razões para a escolha:**

1. **Tamanho do dataset (768 amostras):** Redes muito profundas/largas causariam overfitting severo. Com apenas 614 amostras de treino, a capacidade deve ser limitada.
2. **Pirâmide decrescente (64→32→16):** Força a rede a aprender representações progressivamente mais compactas, atuando como gargalo informacional que regulariza implicitamente.
3. **Total de parâmetros (~2.945):** Com a razão amostras/parâmetros ≈ 0.2, a rede é propensa a overfitting, justificando o uso de early stopping como mecanismo de regularização principal.
4. **ReLU como ativação:** $f(x) = \max(0, x)$ — computacionalmente simples, não sofre de vanishing gradient (ao contrário de sigmoid/tanh em camadas profundas), e produz ativações esparsas que melhoram a generalização.
5. **Adam como otimizador:** Mantém taxas de aprendizado adaptativas por parâmetro, combinando as vantagens de AdaGrad e RMSProp. Converge mais rápido que SGD em problemas de pequena escala.

#### 4.2.3 Convergência

A MLP convergiu em **64 épocas** (0.50s), interrompida pelo early stopping. A loss final de treino foi **0.3868**. A melhor accuracy de validação atingida foi **0.8495**.

### 4.3 Métricas de Avaliação da MLP

#### 4.3.1 Resultados no Conjunto de Teste

| Métrica | Valor |
|---|---|
| **Accuracy** | 0.7273 |
| **Precision** | 0.6154 |
| **Recall** | 0.5926 |
| **F1-Score** | 0.6038 |
| **ROC-AUC** | 0.8135 |

#### 4.3.2 Matriz de Confusão

|  | Predito Neg. | Predito Pos. |
|---|---|---|
| **Real Neg.** | TN = 80 | FP = 20 |
| **Real Pos.** | FN = 22 | TP = 32 |

**Análise:** O modelo apresenta uma tendência conservadora, com mais falsos negativos (22) do que falsos positivos (20). Em contexto clínico, falsos negativos (pacientes diabéticos não detectados) são mais perigosos que falsos positivos, sugerindo que o threshold de decisão poderia ser reduzido para priorizar recall.

### 4.4 Comparação com Modelos Clássicos

#### 4.4.1 Modelos de Referência

**XGBoost Classifier:**
- 100 árvores, `max_depth=5`, `learning_rate=0.1`
- Gradient boosting: combina múltiplas árvores fracas sequencialmente

**SVM (RBF kernel):**
- `C=1.0`, `gamma='scale'`, `class_weight='balanced'`
- Maximiza a margem no espaço de kernel implícito

#### 4.4.2 Resultados — Conjunto de Teste (Hold-out 80/20)

| Modelo | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Tempo (s) |
|---|---|---|---|---|---|---|
| **MLP** | 0.7273 | 0.6154 | 0.5926 | 0.6038 | 0.8135 | 0.505 |
| **XGBoost** | 0.7338 | 0.6444 | 0.5370 | 0.5859 | 0.8022 | 0.126 |
| **SVM (RBF)** | 0.7208 | 0.5821 | **0.7222** | **0.6446** | **0.8243** | **0.038** |

#### 4.4.3 Resultados — Validação Cruzada (10-fold, mais robusto)

| Modelo | Accuracy | F1-Score | ROC-AUC | Tempo (s) |
|---|---|---|---|---|
| **MLP** | 0.7655 ± 0.0500 | 0.6385 ± 0.0737 | **0.8386 ± 0.0520** | 2.48 |
| **XGBoost** | **0.7706 ± 0.0595** | 0.6685 ± 0.0863 | 0.8291 ± 0.0636 | 0.46 |
| **SVM (RBF)** | 0.7641 ± 0.0638 | **0.7032 ± 0.0735** | 0.8369 ± 0.0553 | **0.29** |

### 4.5 Discussão Comparativa

#### 4.5.1 Desempenho

Os três modelos apresentam desempenho **estatisticamente indistinguível** na validação cruzada — todas as diferenças estão dentro dos desvios padrão (σ ≈ 0.05-0.07). Isto é esperado para um dataset de 768 amostras com 4 features: o sinal estatístico é limitado, e modelos com fronteiras de decisão suficientemente flexíveis convergem para desempenhos similares.

**Observações específicas:**
- O **SVM** apresentou o melhor F1-Score (0.7032) na validação cruzada, com recall significativamente superior. O `class_weight='balanced'` penaliza erros na classe minoritária, priorizando a detecção de diabéticos.
- A **MLP** obteve o maior ROC-AUC na validação cruzada (0.8386), indicando melhor capacidade de discriminação probabilística em diferentes thresholds.
- O **XGBoost** teve a maior accuracy (0.7706) e um bom equilíbrio geral.

#### 4.5.2 Custo Computacional

| Modelo | Tempo CV (s) | Fator vs SVM |
|---|---|---|
| SVM (RBF) | 0.29 | 1.0× (referência) |
| XGBoost | 0.46 | 1.6× |
| MLP | 2.48 | **8.6×** |

A MLP é **8.6× mais lenta** que o SVM para um desempenho equivalente. Para este dataset de tamanho modesto, o overhead computacional da MLP (backpropagation iterativo com 64 épocas) não se justifica frente ao ganho de desempenho negligível. Em datasets maiores (> 10k amostras), a MLP tende a escalar melhor que o SVM (que tem complexidade O(n²) no treino).

#### 4.5.3 Quando Preferir Cada Modelo?

| Cenário | Modelo Recomendado | Razão |
|---|---|---|
| **Dataset pequeno (< 1k amostras)** | SVM ou XGBoost | Menor risco de overfitting, mais rápidos |
| **Dataset grande (> 10k)** | MLP | Escala linearmente; mais capacidade |
| **Necessidade de interpretabilidade** | XGBoost | Feature importance nativa; SHAP eficiente |
| **Prioridade em recall (clínico)** | SVM (balanced) | Melhor detecção da classe positiva |
| **Melhor probabilidade calibrada** | MLP | Saída softmax; melhor AUC |

### 4.6 Curvas de Aprendizado

A análise das curvas de aprendizado da MLP revela:

1. **Loss de treino:** Decaimento suave de ~0.75 para ~0.39, sem oscilações significativas — o Adam convergiu de forma estável.
2. **Accuracy de validação:** Estabilizou em torno de 0.83-0.85 após ~10 épocas, com oscilações naturais de mini-batch.
3. **Early stopping:** Ativado na época 64, evitando overfitting prolongado.

### 4.7 Visualizações Geradas

| Arquivo | Descrição |
|---|---|
| `15_mlp_curvas_aprendizado.png` | Loss de treino e accuracy de validação por época |
| `16_matrizes_confusao.png` | Matrizes de confusão dos 3 modelos (lado a lado) |
| `17_curvas_roc.png` | Curvas ROC sobrepostas com AUC de cada modelo |
| `18_comparacao_metricas_modelos.png` | Barras comparativas das métricas |

### 4.8 Sumário da Etapa 4

| Item | MLP | XGBoost | SVM |
|---|---|---|---|
| F1-Score (CV) | 0.639 | 0.669 | **0.703** |
| ROC-AUC (CV) | **0.839** | 0.829 | 0.837 |
| Tempo (CV) | 2.48s | 0.46s | **0.29s** |
| Complexidade | ~2.945 parâmetros | 100 árvores | Vetores de suporte |

**Conclusão:** Os três modelos apresentam desempenho competitivo e estatisticamente equivalente. O SVM com kernel RBF oferece o melhor compromisso entre F1-Score e eficiência computacional para este dataset. A MLP, embora levemente superior em ROC-AUC, tem custo computacional 8.6× maior sem ganho significativo de desempenho. O XGBoost ocupa uma posição intermediária. Para a etapa de otimização de hiperparâmetros, todos os três modelos serão considerados candidatos.

**Próxima etapa:** Otimização de hiperparâmetros e/ou abordagem do problema de regressão.

---
