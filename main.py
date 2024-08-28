import os
import numpy as np
import csv
from skimage import io, util, img_as_ubyte
from lbp import lbp
from glcm import glcm
from lpq import lpq


def identifica_classe(nome_arquivo):
    """
    Função para identificar a classe da imagem com base no nome do arquivo.
    """
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


def extrai_caracteristica_imagem(imagem):
    """
    Função para extrair características de uma imagem utilizando LBP, GLCM, e LPQ.
    """
    try:
        img = io.imread(imagem, as_gray=True)
        img = img_as_ubyte(img)  # Converter para uint8
    except Exception as e:
        print(f"Erro ao carregar a imagem {imagem}: {e}")
        return None

    try:
        lbp_carac = lbp(img)
        glcm_carac = glcm(img)
        lpq_carac = lpq(img)

        # Concatenar todas as características em um único vetor
        caracteristicas = np.concatenate((lbp_carac, glcm_carac, lpq_carac))
        caracteristicas = np.insert(caracteristicas, 0, identifica_classe(imagem))

        return caracteristicas
    except Exception as e:
        print(f"Erro ao extrair características da imagem {imagem}: {e}")
        return None


def listar_arquivos_bmp(pasta):
    """
    Função para listar arquivos de imagem (bmp, png, jpg, jpeg) em uma pasta.
    """
    arquivos_bmp = []
    for root, dirs, files in os.walk(pasta):
        for file in files:
            if file.endswith((".png",)):
                arquivos_bmp.append(os.path.join(root, file))
    return arquivos_bmp


def salvar_caracteristicas_csv(arquivos, nome_arquivo):
    """
    Função para salvar as características extraídas das imagens em um arquivo CSV.
    """
    with open(nome_arquivo, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for i, arquivo in enumerate(arquivos):
            caracteristicas = extrai_caracteristica_imagem(arquivo)
            if caracteristicas is not None:
                linha_formatada = [int(caracteristicas[0])] + [float(x) for x in caracteristicas[1:]]
                writer.writerow(linha_formatada)

            # Exibir progresso a cada 100 imagens
            if (i + 1) % 100 == 0:
                print(f"{i + 1} imagens processadas.")


# Caminho base e diretórios de treinamento e validação
base_dir = os.path.dirname(__file__)
train_dir = os.path.join(base_dir, 'data', 'Augmentation_train')
valid_dir = os.path.join(base_dir, 'data', 'Valid')

# Listar arquivos de imagem
arquivos_train = listar_arquivos_bmp(train_dir)
arquivos_valid = listar_arquivos_bmp(valid_dir)

# Salvar características em arquivos CSV
salvar_caracteristicas_csv(arquivos_train, "treinamento.csv")
salvar_caracteristicas_csv(arquivos_valid, "validacao.csv")
print("Carregando arquivos")