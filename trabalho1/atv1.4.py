import numpy as np
import cv2 as cv


imagem = cv.imread('imgs/relogio.png')

transformacao = np.array([
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131]
    ])


resultado = np.matmul(imagem, transformacao)
resultado = np.clip(resultado, 0, 255)

resultado = resultado.astype(np.uint8)


cv.imshow('image', resultado)
cv.waitKey(0)
cv.destroyAllWindows()