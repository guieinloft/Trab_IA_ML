# Trabalho Prático - Machine Learning

Este repositório contém um pipeline completo de Machine Learning (Classificação e Regressão) aplicado ao dataset *Pima Indians Diabetes*, conforme a especificação do trabalho da disciplina de IA/ML.

O projeto passa por carga e análise de dados, pré-processamento, seleção de atributos, otimização de hiperparâmetros e avaliações de modelos como Multilayer Perceptron (MLP), XGBoost e SVM.

## Pré-requisitos

Certifique-se de ter o [Python 3.8+](https://www.python.org/downloads/) instalado no seu sistema. 

As bibliotecas utilizadas no projeto estão listadas no arquivo `requirements.txt`.

## Passo a Passo para Execução

### 1. Acessar o Repositório
Navegue até a pasta do projeto no seu terminal:
```bash
cd caminho/para/a/pasta/Trab_IA_ML
```

### 2. Criar um Ambiente Virtual (Recomendado)
Para evitar conflito com outras bibliotecas da sua máquina, crie e ative um ambiente virtual (`venv`):

**No Windows:**
```bash
python -m venv venv
.\venv\Scripts\activate
```

**No Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar as Dependências
Com o ambiente ativado, instale as bibliotecas requeridas rodando:
```bash
pip install -r requirements.txt
```

### 4. Executar o Pipeline
Rode o script principal localizado na pasta `src/`:
```bash
python src/main.py
```
> **Aviso:** Durante a execução, o terminal imprimirá logs detalhados de cada etapa. Os modelos treinarão em sequência, o que pode levar alguns minutos (em particular a etapa de otimização Optuna e as explicações do SHAP).

### 5. Visualizar os Resultados
Todas as imagens e visualizações geradas (histogramas, boxplots, matrizes de confusão, curvas de aprendizado, curvas ROC, waterfall plots, etc.) serão automaticamente salvas na pasta `output/`. Acesse o diretório para visualizar todos os artefatos de avaliação.

## Estrutura do Projeto
- `src/main.py`: O código principal contendo todas as etapas.
- `definicao.md`: As instruções e requisitos originais do trabalho.
- `requirements.txt`: Lista de pacotes Python necessários.
- `output/`: (Gerada em tempo de execução) Diretório contendo os gráficos gerados pelo script.