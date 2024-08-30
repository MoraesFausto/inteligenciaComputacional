import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

def carregar_dados(train_csv_path, val_csv_path):
    # Carrega os dados de treinamento e validação dos arquivos CSV
    train_data = pd.read_csv(train_csv_path)
    val_data = pd.read_csv(val_csv_path)

    # Separa as características e as classes
    X_train = train_data.drop('class', axis=1).values
    y_train = train_data['class'].values
    X_val = val_data.drop('class', axis=1).values
    y_val = val_data['class'].values

    return X_train, y_train, X_val, y_val


def normaliza_com_z_score(X):
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_std[X_std == 0] = 1
    return (X - X_mean) / X_std


def salvar_matriz_confusao(conf_matrix, classes, nome_arquivo='matriz_confusao.png'):
    """
    Salva a matriz de confusão como uma imagem.

    Parameters:
    - conf_matrix: A matriz de confusão a ser plotada.
    - classes: Lista com os nomes das classes.
    - nome_arquivo: Nome do arquivo para salvar a imagem.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Matriz de Confusão')
    plt.savefig(nome_arquivo)
    plt.close()