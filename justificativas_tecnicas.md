# Justificativas Teóricas e Decisões de Arquitetura

Este documento compila as justificativas para as escolhas técnicas adotadas no projeto de Machine Learning.

## 1. Imputação de Valores Ausentes (KNN Imputer)

**Justificativa:**
- A imputação simples por mediana (univariada) ignora as correlações entre os atributos. Por exemplo, no Pima Indians Diabetes, `skin` e `insu` possuem correlação positiva de r=0.44, e `mass` e `skin` possuem r=0.39. Um valor imputado para `skin` deveria considerar as informações contidas em `insu` e `mass`.
- O KNN Imputer (com k=5) utiliza os k vizinhos mais próximos (medidos pela distância euclidiana) no espaço das features não-ausentes para estimar o valor faltante. Isso permite preservar a estrutura multivariada dos dados.
- O valor `k=5` é um padrão robusto: suficientemente grande para suavizar o ruído, mas pequeno o bastante para capturar padrões locais nos dados.

## 2. Escalonamento Numérico (RobustScaler)

**Justificativa da Escolha:**
Três opções principais foram consideradas:
1. **StandardScaler (Z-score):** Assume que a distribuição é aproximadamente normal. É altamente sensível a outliers, pois a média e o desvio padrão são severamente afetados por valores extremos.
2. **MinMaxScaler:** Mapeia os dados para o intervalo [0, 1]. É extremamente sensível a outliers, pois os limites máximo e mínimo podem ser valores atípicos isolados.
3. **RobustScaler:** Utiliza a mediana e o intervalo interquartil (IQR, Q3 - Q1).
   - **Decisão:** Optou-se pelo RobustScaler por ser ROBUSTO a outliers. A mediana e o IQR são estatísticas resistentes (a mediana tem ponto de ruptura de 50%, contra 0% da média).

## 3. Seleção de Features (Informação Mútua - MI)

**Justificativa da Técnica:**
A Informação Mútua mede a dependência estatística entre duas variáveis. Propriedades que justificam a escolha:
1. **Captura Dependências Não-lineares:** Ao contrário da correlação de Pearson (que mede apenas relações lineares), a MI detecta qualquer tipo de dependência estatística — algo crucial em dados clínicos onde as relações fisiológicas são frequentemente complexas e não-lineares.
2. **Agnóstico ao Modelo (Model-Agnostic):** Sendo um método filtro, a MI é independente do modelo downstream. Isto evita viés circular (usar o mesmo modelo para selecionar features e depois avaliá-las) e garante que as features selecionadas beneficiem qualquer algoritmo futuro.
3. **Eficiência Computacional:** A complexidade é muito menor e o processamento é consideravelmente mais rápido quando comparado a métodos wrapper (como Recursive Feature Elimination com Cross-Validation).

## 4. Arquitetura da MLP (Classificação)

**Justificativa da Arquitetura:**
A arquitetura foi definida considerando:
1. **Tamanho do Dataset:** 768 amostras é um volume relativamente pequeno para deep learning. Redes muito profundas ou largas causariam overfitting severo rapidamente.
2. **Dimensionalidade:** Com apenas 4 features selecionadas, a capacidade necessária da rede é limitada. Uma rede com 3 camadas ocultas (64 → 32 → 16) oferece flexibilidade suficiente para capturar interações não-lineares sem excesso de parâmetros.
3. **Pirâmide Decrescente:** A redução progressiva de neurônios (64 → 32 → 16) é uma heurística clássica que força a rede a aprender representações cada vez mais compactas, o que atua como uma forma de regularização implícita.
4. **Total de Parâmetros:** A rede possui cerca de 3.500 parâmetros, mantendo um número muito menor que o total de amostras de treino para evitar decoreba.

## 5. Arquitetura da MLP (Regressão)

**Justificativa da Arquitetura e Hiperparâmetros:**
1. **Arquitetura (64, 32):** Modelos de regressão muitas vezes precisam de menos camadas do que modelos de classificação para evitar decoreba excessiva, mantendo um funil clássico e contínuo.
2. **Ativação (ReLU):** É o padrão ótimo para manter não-linearidades ativas em redes profundas e evitar o desaparecimento de gradiente (vanishing gradient).
3. **Otimizador (Adam):** Promove a adaptação automática da taxa de aprendizado para diferentes parâmetros.
4. **Early Stopping:** Habilitado com paciência de 20 épocas, ele interrompe o treinamento assim que o erro de validação para de melhorar, prevenindo o overfitting e salvando o modelo com a melhor capacidade de generalização.
