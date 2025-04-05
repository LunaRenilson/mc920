import cv2 as cv 
import numpy as np

image = cv.imread('imgs/macaco.png', cv.IMREAD_GRAYSCALE)
original_image = image.copy()

h, w = image.shape
imageH = h // 4
imageW = w // 4


# Dividindo em blocos de 4x4
# organizando os eixos (4x4 blocos, cada bloco de 128x128 elementos)
image = image.reshape(4, imageH, 4, imageW).swapaxes(1, 2) 

# Achatando em 16 blocos lineares de 128x128 elementos
flatted = image.reshape(16, 128, 128)

# Ordem desejada
order = np.array([6, 11, 13, 3,
         8, 16, 1, 9,
         12, 14, 2, 7, 
         4, 15, 10, 5]) - 1 # para começar indice em 0

# Aplicando ordem desejada
imageReordered = flatted[order]

# Voltando para 4x4 de 128x128
# Organizando os eixos de volta para o formato original
# Organizando os elementos pro formato original (512x512)
reordered = imageReordered.reshape(4,4,128,128).swapaxes(1, 2).reshape(512, 512)

resultado = np.clip(reordered, 0, 255).astype(np.uint8)


divisor = np.ones((h, 10), dtype=np.uint8) * 255 
concatenado = cv.hconcat([original_image, divisor, resultado])

cv.imwrite('imgs_geradas/atv1.3/source.png', concatenado)