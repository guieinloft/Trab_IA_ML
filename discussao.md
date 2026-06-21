# Discussão Teórica sobre Overfitting (Etapa 7)

Q1: Como identificar o overfitting nas curvas de aprendizado?
R: O overfitting é identificado no gráfico quando a curva de desempenho no conjunto 
   de treino continua a subir desenfreadamente (ou a perda cai para quase zero), 
   enquanto a curva de validação estagna e, em seguida, começa a cair. É esse 
   "descolamento" crescente e a divergência entre treino e teste que evidenciam que 
   o modelo parou de aprender padrões e passou a decorar os dados.

Q2: Quais os principais impactos do overfitting na capacidade de generalização?
R: Quando ocorre o overfitting, o modelo se torna superespecializado nos ruídos, 
   outliers e minúcias exatas do dataset de treinamento. O impacto direto é a perda 
   da capacidade de generalização: o modelo fracassará em produzir previsões precisas 
   quando exposto a novos dados do mundo real que não faziam parte do seu treino original.

Q3: Como as técnicas de regularização aplicadas ajudaram a mitigar o problema?
R: Aplicamos três estratégias conjuntas no modelo regularizado para mitigar o overfitting:
   1. Simplificação Estrutural: Reduzimos as camadas de (256, 128, 64) para (64, 32). Menos
      neurônios significam menor capacidade de memorização "forçada" da rede.
   2. Regularização L2 (alpha=0.01): Penalizou pesos muito elevados durante o treinamento. 
      Isso força a rede a usar pesos menores e mais distribuídos, suavizando a fronteira 
      de decisão matemática.
   3. Early Stopping: Monitorou a acurácia de validação. Assim que a rede percebeu que não
      houve melhora durante 15 épocas (paciência), ela congelou o treinamento e "salvou" os
      pesos, impedindo-a de continuar treinando até decorar os dados residuais.
