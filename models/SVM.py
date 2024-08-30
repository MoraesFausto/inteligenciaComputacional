import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from models.utils import carregar_dados


def svm(X_train, y_train, X_valid, y_valid):
    # Melhor combinacao
    # pipe = Pipeline([
    #     ("scaler", StandardScaler()),  # Normalizar os dados
    #     ("svm", SVC(C=10, class_weight='balanced', coef0=0, gamma='scale', kernel='rbf', probability=True))
    # ])

    svm = SVC(kernel='rbf', probability=True)
    params = {
        "svm__C": [10],
        "svm__gamma": [2e-5, 2e-3, 2e-1, "auto", "scale"],
        "svm__coef0": [0, 0.1, 0.5, 1, 5, 10]
    }
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", svm)
    ])
    modelo = GridSearchCV(pipe, params, n_jobs=-1)
    modelo.fit(X_train, y_train)
    cm = ConfusionMatrixDisplay.from_estimator(modelo, X_valid, y_valid)
    # classification_report(y_train, modelo.predict(X_train))
    print(classification_report(y_valid, modelo.predict(X_valid)))
    print(cm.confusion_matrix)
    plt.savefig('matrizes_de_confusao/matriz_de_confusao_svm.png')
    plt.show()
    svm_model_info = modelo.best_estimator_.steps[1][1]
    n_support_per_class = svm_model_info.n_support_
    total_support_vectors = svm_model_info.support_.shape[0]
    print("Número de vetores de suporte por classe:", n_support_per_class)
    print("Número total de vetores de suporte:", total_support_vectors)


def executar_svm(caminho_csv_treinamento, caminho_csv_validacao):
    # Carregar os dados
    X_train, y_train, X_val, y_val = carregar_dados(caminho_csv_treinamento, caminho_csv_validacao)
    svm(X_train, y_train, X_val, y_val)

