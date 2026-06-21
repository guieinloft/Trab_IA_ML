Trabalho Prático - Aprendizado Supervisionado com Seleção de Features e Otimização de Hiperparâmetros 

Objetivo 

Desenvolver um pipeline completo de Machine Learning utilizando um dataset real de livre escolha, aplicando técnicas de pré-processamento, seleção de atributos, treinamento de modelos, otimização de hiperparâmetros e análise de resultados. 

**O trabalho deve contemplar:** 

* Classificação Binária 
* Classificação Multiclasse e Regressão 
* Utilizando o mesmo dataset quando possível, ou datasets distintos devidamente justificados. 

---

Etapa 1 - Escolha e Análise do Dataset 

Selecione um dataset público relacionado a doença cardíaca, hipertensão ou diabetes, contendo quantidade suficiente de amostras e atributos para justificar a aplicação de técnicas de seleção de features.  A procedência do dataset selecionado é fundamental. Datasets reais e/ou usados como base em artigos relevantes devem ser usados para justificar a escolha do dataset. 

**Fontes sugeridas:** 

* UCI Machine Learning Repository 
* Kaggle 
* OpenML 
* Scikit-Learn Datasets 

**Apresente:** 

* Descrição do problema. 
* Quantidade de amostras. 
* Quantidade de atributos. 
* Variável alvo. 
* Distribuição das classes (quando aplicável). 

**Possíveis problemas encontrados:** 

* Valores ausentes. 
* Classes desbalanceadas. 
* Outliers. 
* Correlação elevada entre atributos. 

---

Etapa 2 - Pré-processamento 

Realize o tratamento necessário para os dados. Justifique todas as escolhas realizadas. 

**Inclua:** 

* Tratamento de valores ausentes (Exemplos : Média , Mediana , Moda )
* Remoção de registros 
* Codificação de variáveis categóricas (Exemplos : One-Hot Encoding , Label Encoding )
* Normalização ou Padronização (Exemplos : StandardScaler , MinMaxScaler , RobustScaler )

---

Etapa 3 - Seleção de Features 

Aplicar uma única técnica de seleção de atributos, escolhida pelo grupo. O grupo deverá justificar a escolha da técnica utilizada. 

**Sugestões:** 
* **Métodos Filtro:** Correlação , Mutual Information , ANOVA F-Test , Chi-Square 
* **Métodos Wrapper:** Recursive Feature Elimination (RFE) , Sequential Feature Selection 
* **Métodos Embedded:** Lasso , Random Forest Feature Importance , XGBoost Feature Importance 

**Apresente:** 

* Quantidade inicial de atributos. 
* Quantidade final de atributos selecionados. 
* Ranking ou importância das features (quando disponível). 
* Critério utilizado para definir as features escolhidas. 

**Além disso, compare os resultados dos modelos utilizando:** 

1. Todas as features disponíveis. 
2. Apenas as features selecionadas. 

**Discuta se a seleção de atributos contribuiu para:** 
* Melhor desempenho. 
* Redução de overfitting. 
* Redução do tempo de treinamento. 
* Melhor interpretabilidade do modelo. 

Interpretabilidade dos Modelos (Opcional - Extra) 

O grupo poderá utilizar a biblioteca SHAP (SHapley Additive Explanations) para interpretar as decisões dos modelos desenvolvidos. 

**Apresentar:** 

* Importância global das features (SHAP Summary Plot). 
* Explicação local de pelo menos uma predição realizada pelo modelo. 
* Comparação entre as features selecionadas na Etapa 3 e as features mais relevantes identificadas pelo SHAP. 
**Discutindo:** 
* Quais atributos tiveram maior influência nas predições. 
* Se os resultados obtidos pelo SHAP corroboram a técnica de seleção de features utilizada. 
* Como a interpretabilidade auxilia na compreensão e validação do modelo desenvolvido. 

---

Etapa 4 - Classificação com MLP 

Desenvolver uma Rede Neural Multicamadas (MLP) para resolver um problema de classificação (binária ou multiclasse). 

**A implementação pode ser realizada utilizando:** 
* Scikit-Learn (MLPClassifier) 
* TensorFlow/Keras 
* PyTorch 

**O grupo deverá definir e justificar:** 

* Número de camadas ocultas. 
* Número de neurônios por camada. 
* Funções de ativação utilizadas. 
* Otimizador escolhido. 
* Taxa de aprendizado. 
* Número de épocas. 
* Tamanho do batch. 

**Avaliação:** 
* **Para classificação binária - Apresentar:** Accuracy , Precision , Recall , F1-Score , ROC-AUC , Matriz de Confusão 
* **Para classificação multiclasse - Apresentar:** Accuracy , Precision Macro , Recall Macro , F1 Macro , Matriz de Confusão 
* **Além disso, apresentar:** Curva de aprendizado (loss por época) e Evolução da métrica principal durante o treinamento. 

Comparação (Opcional - Extra) 

O grupo poderá comparar os resultados da MLP com um modelo clássico de Machine Learning, discutindo diferenças de desempenho e custo computacional. 

* Random Forest 
* SVM 
* XGBoost 
* Logistic Regression 

---

Etapa 5 - Regressão com MLP 

Desenvolver uma Rede Neural Multicamadas (MLP) para resolver um problema de regressão. 

**A implementação pode ser realizada utilizando:** 
* Scikit-Learn (MLPRegressor) e TensorFlow/Keras 
* PyTorch 

**O grupo deverá definir e justificar:** 

* Número de camadas ocultas. 
* Número de neurônios. 
* Funções de ativação. 
* Otimizador. 
* Taxa de aprendizado. 
* Número de épocas. 
* Batch size. 



**Avaliação - Apresentar:** 

* MAE 
* MSE 
* RMSE 
* R² 

**Além dos seguintes gráficos:** 

* Valores reais valores preditos. 
* Resíduos. 
* Curva de aprendizado (loss por época). 

Comparação (Opcional - Extra) 

Opcionalmente, comparar a MLP com um modelo tradicional de regressão, discutindo vantagens e desvantagens de cada abordagem. 

* Regressão Linear 
* Ridge 
* Lasso 
* Random Forest Regressor 
* XGBoost Regressor 

---

Etapa 6 - Otimização de Hiperparâmetros com Optuna 

Utilize o Optuna para otimizar os hiperparâmetros da Rede Neural Multicamadas (MLP) desenvolvida nas etapas anteriores. O grupo deverá definir um espaço de busca para os hiperparâmetros e justificar as escolhas realizadas. 

**Exemplos de hiperparâmetros que podem ser otimizados:** 

* **Arquitetura da Rede:** Número de camadas ocultas , Número de neurônios por camada , Funções de ativação 
* **Treinamento:** Learning Rate , Batch Size , Número de épocas 
* **Otimização:** Tipo de otimizador (Adam, SGD, RMSprop, entre outros) , Momentum (quando aplicável) 
* **Regularização:** Taxa de Dropout , Weight Decay (L2) 

**Resultados Esperados - Apresente:** 

* Espaço de busca utilizado. 
* Quantidade de testes (trials) executados. 
* Melhor conjunto de hiperparâmetros encontrado. 
* Valor da função objetivo obtido pelo Optuna. 
* Comparação entre o Modelo original e o Modelo otimizado . 

**Discuta:** 

* Ganhos de desempenho obtidos. 
* Impacto da otimização no tempo de treinamento. 
* Principais hiperparâmetros encontrados pelo Optuna. 

Comparação Opcional (Extra) 

Caso o grupo implemente outro modelo além da MLP, poderá realizar também a otimização de seus hiperparâmetros e comparar os resultados com a rede neural. 

---

Etapa 7 - Regularização e Análise de Overfitting 

Investigue o comportamento da MLP em relação ao overfitting e underfitting. O grupo deverá aplicar pelo menos uma técnica de regularização e analisar seus efeitos. 

**Técnicas sugeridas:** 

*  **Dropout:** Inserção de camadas de Dropout para reduzir a dependência entre neurônios. 
* **Weight Decay (Regularização L2):** Penalização dos pesos da rede durante o treinamento. 
* **Early Stopping:** Interrupção automática do treinamento quando não houver melhora no conjunto de validação. 
* **Redução da Complexidade da Rede:** Menor número de camadas , Menor quantidade de neurônios 

**Avaliação - Apresente:** 

* Curvas de treinamento e validação. 
* Comparação entre a Rede sem regularização e a Rede com regularização . 

**Analise:** 
* Existência de overfitting. 
* Existência de underfitting. 
* Impacto da regularização nas métricas finais. 
* Impacto da regularização na capacidade de generalização da rede. 

**Discussão - Responder:** 

1. A rede apresentou sinais de overfitting? 
2. Qual técnica de regularização foi utilizada? 
3. Houve melhoria no desempenho em dados não vistos? 
4. Qual estratégia apresentou melhor equilíbrio entre desempenho e generalização? 

Comparação Opcional (Extra) 

Caso outros modelos tenham sido implementados, discutir como esses modelos lidam com overfitting em comparação à MLP. 

---
