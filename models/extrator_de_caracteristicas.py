import numpy as np
import cv2
import pandas as pd
import mahotas
import os
from skimage.feature import local_binary_pattern
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from models.descritores.lpq import lpq


def dividir_imagem(image, num_blocks=2):
    """Divide a imagem em blocos."""
    h, w = image.shape[:2]
    block_h, block_w = h // num_blocks, w // num_blocks
    blocks = []
    for i in range(num_blocks):
        for j in range(num_blocks):
            block = image[i * block_h:(i + 1) * block_h, j * block_w:(j + 1) * block_w]
            blocks.append(block)
    return blocks


def extrair_caracteristica_com_lbp(image, num_points=24, radius=8):
    """Extrai características usando Local Binary Pattern (LBP)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, num_points, radius, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, num_points + 3), range=(0, num_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist


def extrair_caracteristica_com_glcm(image):
    """Extrai características usando Gray-Level Co-occurrence Matrix (GLCM)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    glcm = mahotas.features.haralick(gray).mean(axis=0)
    return glcm


def extrair_caracteristica_com_lpq(image, radius=3):
    """Extrai características usando Local Phase Quantization (LPQ)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lpq_carac = lpq(gray)
    return lpq_carac


def extrair_caracteristica_de_imagem(image_path, num_blocks=2):
    """Extrai características de uma imagem combinando LBP, GLCM e LPQ para cada bloco."""
    image = cv2.imread(image_path)
    image = cv2.resize(image, (256, 256))  # Redimensiona para um tamanho padrão

    # Divide a imagem em blocos
    blocks = dividir_imagem(image, num_blocks)

    # Extrai características para cada bloco
    features = []
    for block in blocks:
        lbp_features = extrair_caracteristica_com_lbp(block)
        glcm_features = extrair_caracteristica_com_glcm(block)
        lpq_features = extrair_caracteristica_com_lpq(block)
        block_features = np.hstack([lbp_features, glcm_features, lpq_features])
        features.extend(block_features)

    return np.array(features)


def identificar_classe(nome_arquivo):
    """Identifica a classe da imagem com base no nome do arquivo."""
    if 'bart' in nome_arquivo:
        return 0
    elif 'homer' in nome_arquivo:
        return 1
    elif 'lisa' in nome_arquivo:
        return 2
    elif 'maggie' in nome_arquivo:
        return 3
    elif 'marge' in nome_arquivo:
        return 4
    else:
        return -1  # Caso padrão, se nenhuma classe for identificada


def extrair_caracteristicas_e_rotulos(image_paths, num_blocks=2):
    """Extrai características e rótulos para todas as imagens dadas."""
    features = []
    labels = []

    for image_path in image_paths:
        features.append(extrair_caracteristica_de_imagem(image_path, num_blocks))
        labels.append(identificar_classe(image_path))

    return np.array(features), np.array(labels)


def listar_arquivos_bmp(pasta):
    """Função para listar arquivos de imagem (bmp, png, jpg, jpeg) em uma pasta."""
    arquivos_bmp = []
    for root, dirs, files in os.walk(pasta):
        for file in files:
            if file.endswith((".png", ".bmp")):
                arquivos_bmp.append(os.path.join(root, file))
    return arquivos_bmp


def executar_extracao_de_caracteristicas(train_dir, valid_dir, train_features_path, val_features_path):
    # Caminhos para imagens de treinamento e validação
    train_image_paths = listar_arquivos_bmp(train_dir)
    val_image_paths = listar_arquivos_bmp(valid_dir)

    # Extração de características e classes
    print("Iniciando extração de características para treinamento e validação...")
    train_features, train_labels = extrair_caracteristicas_e_rotulos(train_image_paths, num_blocks=4)
    val_features, val_labels = extrair_caracteristicas_e_rotulos(val_image_paths, num_blocks=4)

    # Normalização das características
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    val_features = scaler.transform(val_features)

    # Redução de dimensionalidade (opcional, mas recomendada)
    pca = PCA(n_components=0.98)  # Mantém 98% da variância
    train_features = pca.fit_transform(train_features)
    val_features = pca.transform(val_features)

    # Salvar características e classes em arquivos CSV
    train_features_df = pd.DataFrame(train_features)
    val_features_df = pd.DataFrame(val_features)

    # Adicionando as classes aos DataFrames
    train_features_df['class'] = train_labels
    val_features_df['class'] = val_labels

    train_features_df.to_csv(train_features_path, index=False)
    val_features_df.to_csv(val_features_path, index=False)
    print("Características de treinamento e validação, junto com as classes, foram salvas em arquivos CSV.")
