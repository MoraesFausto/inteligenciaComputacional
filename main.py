import os

from data_augmentation import executar_data_augmentation
from models.SVM import executar_svm
from models.extrator_de_caracteristicas import executar_extracao_de_caracteristicas
from models.knn import executar_knn


def main():
    base_dir = os.path.dirname(__file__)
    train_dir = os.path.join('data', 'Train')
    valid_dir = os.path.join('data', 'Valid')
    train_features_path = os.path.join(base_dir, 'data', 'train_features.csv')
    val_features_path = os.path.join(base_dir, 'data', 'val_features.csv')
    executar_data_augmentation()
    executar_extracao_de_caracteristicas(train_dir, valid_dir, train_features_path, val_features_path)
    executar_knn(train_features_path, val_features_path,
                 [1, 3, 5, 7, 9, 11, 13, 15, 17])
    executar_svm(train_features_path, val_features_path)


if __name__ == '__main__':
    main()
