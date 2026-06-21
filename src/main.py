"""
Trabalho de IA/ML - Pipeline de Machine Learning
Autores: Gabriel Stiegemeier e Guilherme Einloft
Cadeira: Inteligência Artificial
Dataset: Pima Indians Diabetes (UCI / OpenML)
Descrição: Orquestrador principal que executa todas as etapas do pipeline em sequência.
"""
import sys
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')
import warnings 
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # backend não-interativo
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.datasets import fetch_openml

from etapa1_exploracao import run_etapa_1
from etapa2_preprocessamento import run_etapa_2
from etapa3_selecao import run_etapa_3
from etapa4_classificacao import run_etapa_4
from etapa4b_classificacao_multiclasse import run_etapa_4b
from etapa5_regressao import run_etapa_5
from etapa6_otimizacao import run_etapa_6
from etapa7_overfitting import run_etapa_7

if __name__ == "__main__":
    print("\n INICIANDO PIPELINE DE MACHINE LEARNING...\n")
    state = {}
    state = run_etapa_1(state)
    state = run_etapa_2(state)
    state = run_etapa_3(state)
    state = run_etapa_4(state)
    state = run_etapa_4b(state)
    state = run_etapa_5(state)
    state = run_etapa_6(state)
    state = run_etapa_7(state)
    print("\n PIPELINE CONCLUÍDO COM SUCESSO! \n")
