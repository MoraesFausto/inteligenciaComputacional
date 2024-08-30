from typing import List

import numpy as np
import matplotlib.pyplot as plt
from numpy.array_api._array_object import Array
from skimage import transform, exposure, util, io
import os
import random

class AugmentedImage:
    def __init__(self, image, method):
        self.image = image
        self.method = method


# Número de imagens que desejamos ter para cada classe
TARGET_IMAGES_PER_CLASS = 750

# Dados iniciais do número de imagens por classe
numero_imagens_por_classe = {
    0: 78,
    1: 61,
    2: 33,
    3: 30,
    4: 24
}
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
    # 1. Rotação com ângulos aleatórios
    angles = random.sample(range(-30, 31), 3)  # Escolhe 3 ângulos aleatórios entre -30 e 30 graus
    for angle in angles:
        img_rotacionada = transform.rotate(img_normalized, angle=angle, mode='reflect')
        img_rotacionada = (img_rotacionada - img_rotacionada.min()) / (img_rotacionada.max() - img_rotacionada.min())
        imagens_aumentadas.append(AugmentedImage(ajustar_contraste(img_rotacionada), 'rotate'))

    # 2. Flip Horizontal e Vertical
    flips = ['horizontal', 'vertical']
    flip_type = random.choice(flips)
    if flip_type == 'horizontal':
        img_flipped = np.fliplr(img_normalized)
    else:
        img_flipped = np.flipud(img_normalized)
    imagens_aumentadas.append(AugmentedImage(ajustar_contraste(img_flipped), 'flip'))

    # 3. Ajuste de Brilho e Contraste com valores aleatórios
    gammas = [0.8, 1.2]  # Ajustar para valores menos extremos
    gamma = random.choice(gammas)
    img_brilho = exposure.adjust_gamma(img_normalized, gamma=gamma)
    img_brilho = np.clip(img_brilho, 0, 1)
    imagens_aumentadas.append(AugmentedImage(ajustar_contraste(img_brilho), 'contrast'))

    # Aumenta o contraste usando sigmoid com ajuste
    img_contraste = exposure.adjust_sigmoid(img_normalized, cutoff=0.1, gain=5)
    img_contraste = np.clip(img_contraste, 0, 1)
    imagens_aumentadas.append(AugmentedImage(ajustar_contraste(img_contraste), 'contrast_sigmoid'))

    # 4. Zoom In e Zoom Out com escalas aleatórias
    zoom_scales = [1.1, 0.9]
    scale = random.choice(zoom_scales)
    img_zoom = transform.rescale(img_normalized, scale=scale, mode='reflect', anti_aliasing=True)

    if scale > 1.0:
        start_x = (img_zoom.shape[0] - img_normalized.shape[0]) // 2
        start_y = (img_zoom.shape[1] - img_normalized.shape[1]) // 2
        img_zoom_cropped = img_zoom[start_x:start_x + img_normalized.shape[0],
                           start_y:start_y + img_normalized.shape[1]]
        img_zoom_cropped = np.clip(img_zoom_cropped, 0, 1)
        imagens_aumentadas.append(AugmentedImage(ajustar_contraste(img_zoom_cropped), 'zoom'))
    else:
        img_zoom_resized = transform.resize(img_zoom, img_normalized.shape, mode='reflect', anti_aliasing=True)
        img_zoom_resized = np.clip(img_zoom_resized, 0, 1)
        imagens_aumentadas.append(AugmentedImage(ajustar_contraste(img_zoom_resized), 'zoom_09'))

    # 5. Ruído Gaussiano - reduzido
    img_ruido = util.random_noise(img_normalized, mode='gaussian', var=random.uniform(0.0000001, 0.0000005))
    img_ruido = np.clip(img_ruido, 0, 1)
    imagens_aumentadas.append(AugmentedImage(ajustar_contraste(img_ruido), 'gaussian'))

    # 6. Transformação Elástica - ajustada ou removida
    try:
        shear_value = random.uniform(-0.5, 0.5)  # Valor aleatório para shear
        img_elastica = transform.warp(img_normalized, inverse_map=transform.AffineTransform(shear=shear_value),
                                      mode='reflect')
        img_elastica = np.clip(img_elastica, 0, 1)
        imagens_aumentadas.append(AugmentedImage(ajustar_contraste(img_elastica), 'elastic'))
    except Exception as e:
        print(f"Erro na transformação elástica: {e}")

    # 7. Crop Aleatório - menos agressivo
    for _ in range(2):
        start_x = np.random.randint(0, img.shape[0] // 4)
        start_y = np.random.randint(0, img.shape[1] // 4)
        end_x = start_x + img.shape[0] * 3 // 4
        end_y = start_y + img.shape[1] * 3 // 4
        img_crop = img_normalized[start_x:end_x, start_y:end_y]
        img_crop_resized = transform.resize(img_crop, img.shape, mode='reflect', anti_aliasing=True)
        img_crop_resized = np.clip(img_crop_resized, 0, 1)
        imagens_aumentadas.append(AugmentedImage(ajustar_contraste(img_crop_resized), 'crop'))

    return imagens_aumentadas


def gerar_imagens_balanceadas(input_dir, output_dir, numero_imagens_por_classe, target_per_class):
    """
    Gera e salva imagens aumentadas para balancear o dataset.

    Args:
        input_dir (str): Diretório contendo as imagens originais organizadas por classe.
        output_dir (str): Diretório onde as imagens aumentadas serão salvas.
        numero_imagens_por_classe (dict): Número atual de imagens por classe.
        target_per_class (int): Número alvo de imagens por classe após o aumento.
    """
    for classe, num_images in numero_imagens_por_classe.items():
        class_input_dir = os.path.join(input_dir, str(classe))
        class_output_dir = output_dir

        if not os.path.exists(class_output_dir):
            os.makedirs(class_output_dir)

        # Calcula quantas imagens adicionais são necessárias para esta classe
        num_needed = target_per_class - num_images

        # Lista de imagens na classe
        class_images = [f for f in os.listdir(class_input_dir) if os.path.isfile(os.path.join(class_input_dir, f))]

        generated_images_count = 0
        while generated_images_count < num_needed:
            for img_name in class_images:
                if generated_images_count >= num_needed:
                    break

                img_path = os.path.join(class_input_dir, img_name)
                img = io.imread(img_path, as_gray=True)

                augmented_images = aplica_data_augmentation(img)

                for aug_img in augmented_images:
                    if generated_images_count >= num_needed:
                        break

                    save_img_path = os.path.join(class_output_dir,
                                                 f"{os.path.splitext(img_name)[0]}_aug_{generated_images_count}.png")
                    io.imsave(save_img_path, aug_img.image)
                    generated_images_count += 1

        print(f"Classe {classe}: Imagens aumentadas geradas: {generated_images_count}")


def executar_data_augmentation():
    # Caminhos de entrada e saída para os diretórios de imagem
    input_dir = os.path.join('data', 'classes')
    output_dir = os.path.join('data', 'Train')

    # Gerar imagens aumentadas para balancear o dataset
    gerar_imagens_balanceadas(input_dir, output_dir, numero_imagens_por_classe, TARGET_IMAGES_PER_CLASS)
