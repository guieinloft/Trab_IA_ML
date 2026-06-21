# Relatório Técnico: Etapa 4 — Classificação com MLP

Este relatório detalha as arquiteturas, decisões de hiperparâmetros e os resultados obtidos nas tarefas de classificação (binária e multiclasse) utilizando Redes Neurais Multicamadas (MLP). Em atendimento à especificação, a implementação utilizou o `MLPClassifier` do `scikit-learn` e incluiu um cenário extra de comparação com modelos clássicos.

## 1. Arquitetura da Rede e Hiperparâmetros (Justificativas)

Para ambas as tarefas (binária e multiclasse), definiu-se a seguinte arquitetura base para o `MLPClassifier`:

- **Número de camadas ocultas e neurônios:** `(64, 32, 16)` — Três camadas ocultas no formato de "pirâmide invertida".  
  *Justificativa:* Como o dataset é tabular, relativamente pequeno (768 amostras) e possui poucas features (reduzido ainda mais na Etapa 3), uma rede excessivamente larga ou muito profunda causaria *overfitting* imediato. O formato piramidal permite que a rede extraia representações progressivamente mais abstratas e de menor dimensão até a camada de saída.
- **Função de ativação:** `ReLU` (Rectified Linear Unit).  
  *Justificativa:* A função $f(x) = \max(0, x)$ é o estado da arte para camadas ocultas em MLPs pois mitiga o problema do gradiente evanescente (*vanishing gradient*), garantindo que os pesos das primeiras camadas sejam atualizados de forma eficiente, diferentemente das funções Sigmoide ou Tanh.
- **Otimizador escolhido:** `Adam` (Adaptive Moment Estimation).  
  *Justificativa:* O Adam combina os benefícios do *Momentum* e do *RMSProp*, adaptando a taxa de aprendizado individualmente para cada peso. É altamente convergente e robusto a ruídos na função de custo.
- **Taxa de aprendizado inicial:** `0.001`.  
  *Justificativa:* Valor padrão empiricamente comprovado como estável para o otimizador Adam iniciar a busca pelos mínimos locais/globais.
- **Número máximo de épocas:** `500`.  
  *Justificativa:* Fornece espaço de sobra para a rede convergir. Contudo, o limite raramente é atingido devido ao uso obrigatório de *Early Stopping*.
- **Tamanho do batch:** `32`.  
  *Justificativa:* O treinamento via *Mini-batch* (ao invés de processar o dataset inteiro de uma vez) introduz um nível saudável de estocasticidade que atua como regularizador, ajudando a rede a escapar de mínimos locais. O tamanho 32 é ideal para aproveitar cálculos matriciais mantendo a robustez do gradiente.

### Controle de Overfitting Intrínseco
- Ativou-se o **Early Stopping** (`early_stopping=True`) com uma fração de validação interna de 15% (`validation_fraction=0.15`) e paciência de 30 épocas (`n_iter_no_change=30`). O treinamento cessa automaticamente caso a métrica de validação não melhore após 30 ciclos completos.

---

## 2. Abordagem 1: Classificação Binária (Etapa 4a)

**Target:** `Outcome` (0 = Negativo, 1 = Positivo).
Utilizaram-se estritamente as features selecionadas pela Informação Mútua na Etapa 3.

**Métricas extraídas e salvas:**
- Acurácia (*Accuracy*)
- Precisão (*Precision*)
- Revocação (*Recall*)
- F1-Score
- Área Sob a Curva ROC (ROC-AUC)
- Matriz de Confusão

**Visualizações geradas (salvas em disco):**
- Curvas ROC comparativas.
- Matrizes de confusão.
- Barras de métricas comparativas.
- Curvas de aprendizado (Evolução da função de custo/Loss por época e Acurácia no conjunto de validação).

---

## 3. Abordagem 2: Classificação Multiclasse (Etapa 4b)

Conforme a especificação estipulou, implementou-se o modelo multiclasse utilizando o mesmo dataset base.

**Target Multiclasse:** A coluna contínua de glicose (`plas`) foi removida das *features* e transformada no alvo categórico de 3 classes, respeitando os **Limiares Clínicos Oficiais da ADA** (Associação Americana de Diabetes):
1. **Normal (Classe 0):** Glicose < 100 mg/dL
2. **Pré-diabético (Classe 1):** 100 ≤ Glicose < 140 mg/dL
3. **Diabético (Classe 2):** Glicose ≥ 140 mg/dL

A camada de saída da MLP passou a utilizar implicitamente a ativação *Softmax* para produzir 3 probabilidades que somam 1.

**Métricas extraídas (focadas em cenários multirótulo):**
- Accuracy
- Precision Macro (Média aritmética das precisões individuais de cada classe, para avaliar imparcialmente as classes minoritárias).
- Recall Macro
- F1-Score Macro
- Matriz de Confusão 3x3.

**Visualizações geradas:** Curvas de erro por época, barras de distribuição e matriz de confusão correspondente.

---

## 4. Comparação (Opcional Extra): Modelos Clássicos vs. MLP

A especificação ofereceu um adicional opcional para comparar o desempenho e o custo computacional da MLP contra modelos clássicos de Machine Learning.

O *pipeline* treinou e testou três modelos sobre o mesmo conjunto de features:
1. **Rede Neural (MLPClassifier)**
2. **XGBoost Classifier** (Ensemble de árvores otimizado via Gradient Boosting)
3. **Support Vector Machine (SVC)** (Hyperplanos com kernel RBF)

**Discussão de Resultados (Desempenho e Custo):**
- **Custo Computacional:** A MLP exige cálculo computacional muito superior ao SVM para tabular-data pequenos. O *XGBoost* geralmente treina mais rápido que a MLP devido à paralelização eficiente da construção das árvores.
- **Desempenho:** Modelos baseados em *Ensembles* de árvores (como o XGBoost e Random Forest da Etapa 3) costumam empatar ou até superar Redes Neurais Tradicionais em dados tabulares puros com menos de 1000 amostras (como é o caso do Pima). Redes neurais costumam ter grande folga e brilhar em dados complexos e não-estruturados (imagem, áudio, NLP) ou datasets massivos. Contudo, as taxas de ROC-AUC da MLP permaneceram altamente competitivas frente aos outros dois algoritmos.
