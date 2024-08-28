import numpy as np
import csv
from scipy.spatial.distance import euclidean
# Função para carregar os dados de um arquivo CSV
# Função para carregar os dados de um arquivo CSV
def carregar_dados(caminho_arquivo):
    features = []
    labels = []
    with open(caminho_arquivo, 'r') as csvfile:
        leitor = csv.reader(csvfile)
        for linha in leitor:
            dados = list(map(float, linha))
            features.append(dados[1:])  # Assume que a primeira coluna é a label e o resto são as features
            labels.append(int(dados[0]))
    return np.array(features), np.array(labels)


# Função para implementar o KNN
def knn(train_features, train_labels, test_features, k):
    predicoes = []
    for test_point in test_features:
        # Calcular distâncias entre o ponto de teste e todos os pontos de treino
        distancias = [euclidean(test_point, train_point) for train_point in train_features]

        # Encontrar os índices dos k vizinhos mais próximos
        indices_vizinhos = np.argsort(distancias)[:k]
        vizinhos_labels = [train_labels[i] for i in indices_vizinhos]

        if len(vizinhos_labels) > 0:
            # Calcular a moda dos rótulos dos vizinhos
            predicao = np.bincount(vizinhos_labels).argmax()
        else:
            # Caso não haja vizinhos (não deveria acontecer com k > 0 e dados corretos)
            predicao = -1  # Valor padrão ou ajuste conforme necessário

        predicoes.append(predicao)
    return np.array(predicoes)

# Função para calcular a acurácia
def calcular_acuracia(predicoes, verdadeiros_labels):
    acuracia = np.mean(predicoes == verdadeiros_labels)
    return acuracia

# Caminhos para os arquivos CSV
caminho_treinamento = 'simpsonTreino.csv'
caminho_validacao = 'simpsonTeste.csv'

# Carregar dados
train_features, train_labels = carregar_dados(caminho_treinamento)
test_features, test_labels = carregar_dados(caminho_validacao)

# Definir o valor de k
k = 17

# Prever as labels para os dados de validação
predicoes = knn(train_features, train_labels, test_features, k)
# Calcular a acurácia
acuracia = calcular_acuracia(predicoes, test_labels)

print(f'Acurácia do modelo KNN: {acuracia:.2f}')