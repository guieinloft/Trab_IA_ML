# Relatório Técnico: Etapa 2 — Pré-processamento

Este relatório detalha as transformações aplicadas aos dados brutos do dataset *Pima Indians Diabetes*, em conformidade com as exigências da especificação do trabalho. Todas as decisões tomadas visam melhorar a qualidade dos dados antes de alimentarem os modelos preditivos, tratando problemas comuns identificados na Etapa 1.

## 1. Tratamento de Valores Ausentes

A análise exploratória (Etapa 1) revelou que diversas variáveis apresentavam o valor `0` em contextos onde isso é biologicamente impossível, mascarando dados ausentes. 

**Variáveis afetadas:** Glicose (`plas`), Pressão Sanguínea (`pres`), Espessura da Pele (`skin`), Insulina (`insu`) e IMC (`mass`).

### Decisão Técnica: Imputação via K-Nearest Neighbors (KNNImputer)
Primeiramente, todos os zeros inválidos nestas 5 colunas foram substituídos por `NaN`. Em seguida, aplicou-se o `KNNImputer` (com $k=5$).

**Justificativa:** 
A substituição pela média ou mediana global seria inadequada, especialmente para variáveis com alta taxa de ausência como Insulina (48,7%) e Espessura da pele (29,6%). A imputação pela média distorceria gravemente a variância e a distribuição original da variável. O `KNNImputer` foi escolhido pois infere o valor faltante baseando-se nas "vizinhanças" multivariadas daquela amostra (as outras features que estão presentes), resultando em estimativas muito mais realistas e preservando as relações não-lineares entre os atributos.

## 2. Codificação de Variáveis Categóricas

**Ação:** Nenhuma transformação de codificação (como One-Hot Encoding ou Label Encoding) foi aplicada.

**Justificativa:** 
O dataset Pima Indians Diabetes é composto integralmente por variáveis preditivas contínuas ou numéricas discretas (como o número de gestações). Não há presença de strings, categorias nominais ou ordinais brutas que exijam tais transformações. A variável alvo de classificação (`Outcome`) já se encontra no formato numérico binário correto (0 e 1).

## 3. Tratamento de Outliers e Escalonamento Numérico

A maioria dos algoritmos de Machine Learning (como Redes Neurais baseadas em gradiente descendente e SVM) é altamente sensível à escala dos dados. Se as features estiverem em escalas diferentes, os pesos convergirão muito lentamente ou para sub-ótimos.

**Problema identificado:**
A análise de distribuição detectou forte assimetria e presença massiva de *outliers* em certas variáveis, notavelmente em `insu` (34 outliers via IQR, cauda muito longa) e `pedi` (29 outliers).

### Decisão Técnica: `RobustScaler`
Optou-se por escalonar todas as variáveis preditivas utilizando o `RobustScaler` do scikit-learn.

**Justificativa:**
Ao contrário do `StandardScaler` ou `MinMaxScaler`, que utilizam a média, a variância e os valores extremos para calcular o escalonamento (sendo assim gravemente influenciados e distorcidos por valores atípicos severos), o `RobustScaler` remove a mediana e escala os dados de acordo com o Intervalo Interquartil (IQR). 
Isso garante que os *outliers* na insulina e na função de pedigree não achatem o restante dos dados normais das outras variáveis, garantindo que o algoritmo de *Machine Learning* foque no padrão central da distribuição.

## 4. Considerações sobre Outros Problemas (Desbalanceamento e Correlação)

A especificação menciona o tratamento de classes desbalanceadas e correlação elevada:
- **Classes Desbalanceadas:** Não houve aplicação de técnicas intrusivas no dataset como SMOTE ou sub-amostragem na etapa de pré-processamento. A estratégia adotada foi mitigar esse efeito na separação dos conjuntos de treino e teste (utilizando amostragem estratificada `stratify=y`) e utilizando métricas imunes ao desbalanceamento nas Etapas de Classificação (como F1-Score e matrizes de confusão detalhadas).
- **Correlação Elevada:** O tratamento da maldição da dimensionalidade e das redundâncias entre atributos foi delegado integralmente para a **Etapa 3** (Seleção de Features baseada em *Mutual Information*).
