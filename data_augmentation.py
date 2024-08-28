from typing import List

import numpy as np
import matplotlib.pyplot as plt
from numpy.array_api._array_object import Array
from skimage import transform, exposure, util, io
import os


class AugmentedImage:
    def __init__(self, image, method):
        self.image = image
        self.method = method


def ajustar_contraste(img):
    """
    Ajusta o contraste da imagem usando equalização de histograma.

    Args:
        img (ndarray): Imagem a ser ajustada.

    Returns:
        ndarray: Imagem com contraste ajustado.
    """
    img_equ = exposure.equalize_hist(img)
    return np.clip(img_equ * 255, 0, 255).astype(np.uint8)
def aplica_data_augmentation(img) -> List[AugmentedImage]:
    """
    Aplica data augmentation na imagem.

    Args:
        img (ndarray): A imagem original.

    Returns:
        list: Lista com imagens aumentadas.
    """
    imagens_aumentadas = []
    img_normalized = img / 255.0  # Normalizar para [0, 1] uma vez

    # 1. Rotação em múltiplos ângulos
    angles = [-15, 15, -25, 30]
    # for angle in angles:
    #     img_rotacionada = transform.rotate(img, angle=angle, mode='reflect')  # Ajuste do modo de preenchimento
    #     img_rotacionada = (img_rotacionada - img_rotacionada.min()) / (
    #                 img_rotacionada.max() - img_rotacionada.min())  # Normaliza para [0, 1]
    #     img_rotacionada = np.clip(img_rotacionada * 255, 0, 255).astype(np.uint8)
    #     imagens_aumentadas.append(AugmentedImage(img_rotacionada, 'rotate'))
    #
    # 2. Flip Horizontal e Vertical
    # Verifique o intervalo antes do flip
    # Normaliza img_normalized para [0, 1] se necessário
    # Flip horizontal e vertical
    img_flipped_h = np.fliplr(img_normalized)
    img_flipped_v = np.flipud(img_normalized)


    # Garantir que os valores estão na faixa correta antes de converter
    img_flipped_h = np.clip(img_flipped_h, 0, 1)
    img_flipped_v = np.clip(img_flipped_v, 0, 1)

    # # Converter para o formato uint8
    # img_flipped_h = (img_flipped_h * 255).astype(np.uint8)
    # img_flipped_v = (img_flipped_v * 255).astype(np.uint8)

    # Adiciona as imagens aumentadas à lista
    imagens_aumentadas.append(AugmentedImage(ajustar_contraste(img_flipped_h), 'flip'))
    imagens_aumentadas.append(AugmentedImage(ajustar_contraste(img_flipped_v), 'flip'))
    #
    # # 3. Ajuste de Brilho e Contraste
    # gammas = [0.8, 1.2]  # Ajustar para valores menos extremos
    # for gamma in gammas:
    #     img_brilho = exposure.adjust_gamma(img_normalized, gamma=gamma)
    #     img_brilho = np.clip(img_brilho, 0, 1)
    #     imagens_aumentadas.append(AugmentedImage((img_brilho * 255).astype(np.uint8), 'contrast'))
    #
    # Aumenta o contraste usando sigmoid com ajuste
    # img_contraste = exposure.adjust_sigmoid(img_normalized, cutoff=0.1, gain=5)
    # img_contraste = np.clip(img_contraste, 0, 1)
    # imagens_aumentadas.append(AugmentedImage((img_contraste * 255).astype(np.uint8), 'contrast_sigmoid'))
    #
    # # 4. Zoom In e Zoom Out
    # zoom_scales = [1.1, 0.9]  # Menos zoom para evitar perda de detalhes
    # for scale in zoom_scales:
    #     img_zoom = transform.rescale(img_normalized, scale=scale, mode='reflect', anti_aliasing=True)
    #     if scale > 1.0:
    #         start_x = (img_zoom.shape[0] - img.shape[0]) // 2
    #         start_y = (img_zoom.shape[1] - img.shape[1]) // 2
    #         img_zoom_cropped = img_zoom[start_x:start_x + img.shape[0], start_y:start_y + img.shape[1]]
    #         img_zoom_cropped = np.clip(img_zoom_cropped, 0, 1)
    #         imagens_aumentadas.append(AugmentedImage((img_zoom_cropped * 255).astype(np.uint8), 'zoom'))
    #     else:
    #         img_zoom_resized = transform.resize(img_zoom, img.shape, mode='reflect', anti_aliasing=True)
    #         img_zoom_resized = np.clip(img_zoom_resized, 0, 1)
    #         imagens_aumentadas.append(AugmentedImage((img_zoom_resized * 255).astype(np.uint8), 'zoom_09'))

    # 5. Ruído Gaussiano - reduzido
    # img_ruido = util.random_noise(img_normalized, mode='gaussian', var=0.000001)
    # img_ruido = np.clip(img_ruido, 0, 1)
    # imagens_aumentadas.append(AugmentedImage((img_ruido * 255).astype(np.uint8), 'gaussian'))

    # 6. Transformação Elástica - ajustada ou removida
    # try:
    #     img_elastica = transform.warp(img_normalized, inverse_map=transform.AffineTransform(shear=0.1), mode='reflect')
    #     img_elastica = np.clip(img_elastica, 0, 1)
    #     imagens_aumentadas.append(AugmentedImage((img_elastica * 255).astype(np.uint8), 'elastic'))
    # except Exception as e:
    #     print(f"Erro na transformação elástica: {e}")
    #
    # # 7. Crop Aleatório - menos agressivo
    # for _ in range(2):
    #     start_x = np.random.randint(0, img.shape[0] // 4)
    #     start_y = np.random.randint(0, img.shape[1] // 4)
    #     end_x = start_x + img.shape[0] * 3 // 4
    #     end_y = start_y + img.shape[1] * 3 // 4
    #     img_crop = img_normalized[start_x:end_x, start_y:end_y]
    #     img_crop_resized = transform.resize(img_crop, img.shape, mode='reflect', anti_aliasing=True)
    #     img_crop_resized = np.clip(img_crop_resized, 0, 1)
    #     imagens_aumentadas.append(AugmentedImage((img_crop_resized * 255).astype(np.uint8), 'crop'))

    return imagens_aumentadas


def salvar_imagens_aumentadas(img_path, save_path):
    """
    Gera e salva imagens aumentadas a partir de uma imagem original.

    Args:
        img_path (str): Caminho para a imagem original.
        save_path (str): Diretório onde as imagens aumentadas serão salvas.
    """
    # Carregar a imagem
    img = io.imread(img_path, as_gray=True)

    # Aplicar data augmentation
    imagens_aumentadas = aplica_data_augmentation(img)

    # Verificar se o diretório de salvamento existe; se não, criar
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Salvar imagens aumentadas
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    for i, img_aug in enumerate(imagens_aumentadas, start=1):
        # Ajuste de contraste
        img_aug_contrast = exposure.rescale_intensity(img_aug.image, in_range='image', out_range=(0, 255))

        save_img_path = os.path.join(save_path, f"{img_name}_aug_{i}_{img_aug.method}.png")
        try:
            io.imsave(save_img_path, img_aug_contrast.astype(np.uint8))
           # print(f"Imagem aumentada salva em: {save_img_path}")
        except Exception as e:
            print(f"Erro ao salvar imagem em {save_img_path}: {e}")


def gerar_imagens_para_pasta(input_dir, output_dir):
    """
    Gera e salva imagens aumentadas para todos os arquivos em um diretório de entrada.

    Args:
        input_dir (str): Diretório contendo as imagens originais.
        output_dir (str): Diretório onde as imagens aumentadas serão salvas.
    """
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                img_path = os.path.join(root, file)
                salvar_imagens_aumentadas(img_path, output_dir)


# Caminho base e diretórios de treinamento e validação
base_dir = os.path.dirname(__file__)
input_dir = os.path.join(base_dir, 'data', 'Train')
output_dir = os.path.join(base_dir, 'data', 'Augmentation_train')
gerar_imagens_para_pasta(input_dir, output_dir)
