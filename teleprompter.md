# 🎙️ TELEPROMPTER — Falas do Vídeo (~7 min)

> Leia este arquivo na tela do lado enquanto mostra o `apresentacao_visual.md` na tela principal.
> As marcações [VISUAL: ...] indicam o que deve estar aparecendo na tela naquele momento.
> Os tempos são estimados — não precisa cronometrar exatamente.

---

## BLOCO 1 — Abertura (~30s)

[VISUAL: Título do arquivo visual — cabeçalho com nomes e dataset]

> Fala, pessoal. Eu sou o Gabriel, esse trabalho foi feito em dupla com o Guilherme Einloft, e esse é o nosso trabalho final da cadeira de Inteligência Artificial.

> O objetivo do trabalho foi desenvolver um pipeline completo de Machine Learning — passando por pré-processamento, seleção de features, classificação binária e multiclasse, regressão, otimização de hiperparâmetros com Optuna e análise de overfitting.

> Vamos usar como base o dataset Pima Indians Diabetes, do repositório UCI, disponível também no OpenML.

---

## BLOCO 2 — Dataset (~1 min)

[VISUAL: Seção "Dataset — Pima Indians Diabetes" com a tabela de características]

> O dataset Pima Indians Diabetes é bem conhecido na literatura. Ele contém 768 registros clínicos de mulheres da etnia Pima, com 21 anos ou mais, coletados pelo Instituto Nacional de Diabetes dos Estados Unidos.

> São 8 atributos numéricos — como número de gestações, glicose plasmática, pressão arterial, insulina, IMC, idade, entre outros.

> Pra classificação binária, a variável alvo é o campo Outcome, que indica se a pessoa tem diabetes ou não. Pra regressão, usamos a própria glicose plasmática como variável contínua.

> Quanto às classes, o dataset é levemente desbalanceado: 65% das amostras são negativas e 35% positivas. Nada extremo, mas usamos class weight balanceado nos modelos pra mitigar isso.

[VISUAL: Parte "Problemas identificados"]

> Na análise exploratória, identificamos alguns problemas importantes. Cinco colunas tinham zeros biologicamente impossíveis — por exemplo, insulina com 48,7% de zeros, ou IMC igual a zero. Esses zeros foram tratados como valores ausentes. Também identificamos outliers significativos, principalmente em insulina e na função pedigree.

---

## BLOCO 3 — Organização do Código (~45s)

[VISUAL: Seção "Estrutura do Código" com a árvore de arquivos]

> Vamos olhar como o código está organizado. Todo o código fica na pasta src.

> O ponto de entrada é o main.py — ele é o orquestrador do pipeline. Cada etapa do trabalho é um arquivo Python separado: etapa1 é exploração, etapa2 é pré-processamento, etapa3 é seleção de features, e assim por diante.

[VISUAL: Seção "Pipeline de Execução"]

> O main.py cria um dicionário chamado state, que começa vazio, e vai passando ele de etapa em etapa. Cada função recebe o state, faz o processamento dela, adiciona os resultados e retorna o state atualizado pra próxima etapa.

> Os gráficos de cada etapa são salvos automaticamente na pasta output. Então pra reproduzir tudo, basta rodar python main.py e ele executa o pipeline inteiro de ponta a ponta.

---

## BLOCO 4 — Pré-processamento (~1 min)

[VISUAL: Seção "Pré-processamento" com a tabela de técnicas]

> No pré-processamento, a primeira coisa que fizemos foi o split dos dados em 80% treino e 20% teste, estratificado pela variável alvo. Isso é feito antes de qualquer outra operação pra evitar data leakage — ou seja, garantir que informação do teste nunca influencie o treino.

> Pra tratar os zeros inválidos, usamos o KNN Imputer com k igual a 5. A ideia é que, ao invés de simplesmente substituir por uma mediana global, o KNN Imputer olha os 5 vizinhos mais próximos de cada amostra e estima o valor faltante com base neles. Isso preserva as correlações entre as variáveis — por exemplo, espessura da pele e insulina têm correlação de 0,44.

> Pro escalonamento, escolhemos o RobustScaler, que usa a mediana e o intervalo interquartil. O motivo é que ele é robusto a outliers. O StandardScaler, por exemplo, usa média e desvio padrão — que são muito sensíveis a valores extremos. Como temos variáveis com bastante outlier, como insulina, o RobustScaler é a escolha mais adequada.

---

## BLOCO 5 — Seleção de Features (~1 min)

[VISUAL: Seção "Seleção de Features — Informação Mútua" com a tabela de ranking]

> Pra seleção de atributos, usamos Informação Mútua — que é um método filtro. A vantagem dele sobre a correlação de Pearson é que ele captura dependências não lineares, não só lineares. E por ser um método filtro, ele é independente do modelo — ou seja, não tem viés circular.

> Pra estabilizar os scores, rodamos 50 vezes com sementes diferentes e tiramos a média. O critério de corte foi: features com MI acima da média geral dos scores.

> O resultado foi uma redução de 8 pra 3 features: glicose, insulina e IMC. Essas três foram selecionadas em todos os 10 folds da validação cruzada, mostrando que são robustamente relevantes.

> Como extra, usamos o SHAP pra validar. O ranking do SHAP teve uma correlação de Spearman de 0,93 com o ranking da Informação Mútua — o que significa que os dois métodos concordam fortemente. A perda de ROC-AUC com as 3 features selecionadas foi menor que 4%.

> Um ponto importante: a seleção de features foi feita dentro de cada fold da validação cruzada, nunca usando dados de validação. Isso evita data leakage na avaliação.

---

## BLOCO 6 — Classificação e Regressão (~1 min 15s)

[VISUAL: Seção "Modelos e Resultados" com as tabelas de hiperparâmetros e resultados]

> Agora os modelos. Pra classificação binária, treinamos uma MLP com 3 camadas ocultas — 64, 32 e 16 neurônios, numa pirâmide decrescente. Usamos ReLU como ativação, Adam como otimizador, e Early Stopping com paciência de 30 épocas. A ideia da pirâmide decrescente é forçar a rede a aprender representações cada vez mais compactas, o que funciona como regularização implícita.

> No teste, a MLP obteve ROC-AUC de 0,79. Como comparação, treinamos também XGBoost e SVM — os três ficaram com desempenho similar, mostrando que a MLP é competitiva mesmo com um dataset pequeno.

> Pra classificação multiclasse, criamos 3 classes baseadas nos limiares clínicos da American Diabetes Association: Normal, Pré-diabético e Diabético, com base nos valores de glicose. O F1 Macro ficou em 0,64, que é um resultado satisfatório considerando que são classes com fronteiras limítrofes.

> Na regressão, treinamos uma MLP pra predizer a concentração de glicose plasmática. O modelo obteve R quadrado de 0,42 e MAE de 18,6 mg/dL. O R quadrado indica que o modelo explica 42% da variância — o desempenho é limitado pelo volume dos dados e pela natureza do problema.

---

## BLOCO 7 — Optuna e Regularização (~1 min)

[VISUAL: Seção "Optuna" e depois seção "Regularização vs. Overfitting"]

> Pra otimização de hiperparâmetros, usamos o Optuna com 30 trials, buscando maximizar o ROC-AUC com validação cruzada de 5 folds. O espaço de busca incluía número de camadas, neurônios por camada, learning rate, ativação e alpha do L2.

> O melhor ROC-AUC de validação cruzada foi 0,8344, mas no conjunto de teste o resultado ficou em 0,78 — praticamente igual ao modelo original. Isso mostra que o modelo que definimos manualmente já estava numa boa região do espaço de hiperparâmetros, e que o gargalo de desempenho é o volume dos dados, não a configuração da rede.

[VISUAL: Tabela de "Regularização vs. Overfitting"]

> Na análise de overfitting, treinamos dois modelos contrastantes. O primeiro, sem regularização: 256, 128, 64 neurônios, sem L2, early stopping desabilitado — propositalmente propenso a overfitting. O segundo, com regularização: 64 e 32 neurônios, L2 com alpha 0,01, e early stopping com paciência de 15 épocas.

> O modelo sem regularização mostrou o padrão clássico de overfitting nas curvas: o treino continuava melhorando enquanto a validação estagnava. O modelo regularizado manteve as curvas mais próximas. No teste, a accuracy subiu de 0,708 pra 0,727 com regularização.

> As três técnicas usadas foram: simplificação estrutural, penalização L2 e early stopping.

---

## BLOCO 8 — Conclusão (~30s)

[VISUAL: Seção "Conclusão"]

> Pra concluir: desenvolvemos um pipeline completo e funcional, que cobre todas as etapas do trabalho. Tivemos cuidado especial com data leakage — tanto no pré-processamento quanto na seleção de features. Usamos SHAP pra validar a seleção, o que fortalece a interpretabilidade. O desempenho é limitado pelo volume dos dados, mas os modelos são competitivos e os resultados são consistentes.

> O código é modular — cada etapa é um arquivo separado — e reprodutível: basta rodar python main.py.

> É isso, obrigado pela atenção!

---

> **Tempo total estimado: ~6min 30s a 7min**
