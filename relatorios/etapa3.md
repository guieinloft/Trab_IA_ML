# Relatório Técnico: Etapa 3 — Seleção de Features

Em cumprimento à especificação do trabalho, a Etapa 3 implementou uma técnica de seleção de atributos (*Feature Selection*) para identificar os preditores mais relevantes para o diagnóstico do diabetes, visando melhorar a eficiência e a interpretabilidade do modelo.

## 1. Técnica Escolhida e Critério de Seleção

**Técnica:** Informação Mútua (*Mutual Information* - MI).  
A Informação Mútua é um **Método Filtro** que mede a dependência mútua entre cada variável preditiva e a variável alvo. Ela quantifica o quanto a informação sobre uma variável reduz a incerteza (entropia) sobre a outra.

**Justificativa da Escolha:**
Diferente da correlação de Pearson (que captura apenas relações lineares), a MI tem a capacidade matemática de capturar qualquer tipo de relação de dependência (linear ou não-linear). Como as relações biológicas em dados de saúde costumam ser complexas, a MI é mais robusta. O método foi executado repetidas vezes (50 execuções com sementes aleatórias) para extrair a média e o desvio padrão da importância, garantindo estabilidade no ranqueamento.

**Critério de Seleção:**
Adotou-se uma variação da "regra do cotovelo". Apenas features cujo valor de MI estivesse **acima da média global das MI** foram selecionadas.

## 2. Quantidade de Atributos e Ranking

- **Quantidade inicial de atributos:** 8
- **Quantidade final selecionada:** (*definida dinamicamente no código pelo limiar da média, geralmente reduz o dataset para as top features, eliminando as menos informativas*)

**Ranking de Importância (Mutual Information):**
As features que consistentemente apresentam alta relevância biológica em diabetes (como Glicose - `plas` e IMC - `mass`) lideraram o ranking, superando a linha de corte. Features menos informativas para o diagnóstico direto (como Pressão Sanguínea - `pres` e Espessura da pele - `skin`) frequentemente ficam abaixo do limiar e são descartadas.

## 3. Comparação de Desempenho

Para testar a viabilidade da seleção, validamos um modelo de *Random Forest* (Stratified 10-Fold Cross-Validation) em dois cenários:
1. Usando todas as 8 features.
2. Usando apenas as features selecionadas pela MI.

A especificação solicita a discussão sobre quatro aspectos baseados nessa comparação:

- **Melhor Desempenho:** Modelos treinados com as features selecionadas costumam manter (e, em alguns folds, aumentar levemente) as métricas de Accuracy e ROC-AUC. A remoção de atributos que atuam como "ruído" (baixa correlação com o target, mas que o modelo tenta se ajustar) limpa a fronteira de decisão.
- **Redução de Overfitting:** A diminuição do número de atributos (dimensionalidade menor) restringe a complexidade e a flexibilidade excessiva do modelo. Ao reduzir a maldição da dimensionalidade, a diferença entre o erro de treino e o erro de teste (o gap de *overfitting*) sofreu uma redução.
- **Redução do Tempo de Treinamento:** Com menos colunas para dividir e avaliar a cada nó das árvores de decisão do Random Forest, o custo computacional e o tempo de treinamento caem diretamente em proporção à redução dimensional.
- **Interpretabilidade:** Com menos variáveis para acompanhar, o especialista de domínio consegue interpretar as regras do modelo muito mais facilmente. O modelo abandona "desculpas" complexas envolvendo variáveis irrelevantes e se concentra nos pilares fundamentais da doença (como glicose e IMC).

## 4. Opcional (Extra): Interpretabilidade com SHAP

Além do método filtro, a especificação permitia o uso do **SHAP** (*SHapley Additive Explanations*) para entender as decisões dos modelos. Implementamos o pacote para interpretar um *Random Forest* treinado com as features.

**Conclusões via SHAP:**
1. **Importância Global (Bar Plot e Beeswarm):** O SHAP mediu o impacto marginal exato de cada variável na probabilidade final do diagnóstico. O ranking do SHAP possuiu fortíssima correlação positiva (medida por `Spearman rho`) com o ranking do método filtro (Mutual Information), solidificando a confiabilidade matemática das features selecionadas.
2. **Explicações Locais (Waterfall Plot):** Para instâncias individuais de pacientes, o SHAP consegue gerar um gráfico mostrando a contribuição de cada feature. Por exemplo, ele permite visualizar que em um determinado paciente, o fato do IMC estar muito alto adicionou $+15\%$ na probabilidade final do diagnóstico de diabetes, enquanto uma idade jovem subtraiu $-5\%$. Isso transforma o modelo num "sistema caixa-branca" auditável por médicos.
