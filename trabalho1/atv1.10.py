import numpy as np
import cv2 as cv

# Definição das matrizes hn
h1 = np.array([
    [0,  0, -1,  0,  0],
    [0, -1, -2, -1,  0],
    [-1, -2, 16, -2, -1],
    [0, -1, -2, -1,  0],
    [0,  0, -1,  0,  0]
])

h2 = (1/256) * np.array([
    [1,  4,  6,  4,  1],
    [4, 16, 24, 16, 4],
    [6, 24, 36, 24, 6],
    [4, 16, 24, 16, 4],
    [1,  4,  6,  4,  1]
])

h3 = np.array([
    [-1,  0,  1],
    [-2,  0,  2],
    [-1,  0,  1]
])

h4 = np.array([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
])

h5 = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
])

h6 = (1/9) * np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]
])

h7 = np.array([
    [-1, -1,  2],
    [-1,  2, -1],
    [ 2, -1, -1]
])

h8 = np.array([
    [ 2, -1, -1],
    [-1,  2, -1],
    [-1, -1,  2]
])

h9 = (1/9) * np.array([
    [1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1]
])

h10 = (1/8) * np.array([
    [-1, -1, -1, -1, -1],
    [-1,  2,  2,  2, -1],
    [-1,  2,  8,  2, -1],
    [-1,  2,  2,  2, -1],
    [-1, -1, -1, -1, -1]
])

h11 = np.array([
    [-1, -1,  0],
    [-1,  0,  1],
    [ 0,  1,  1]
])


def aplicarFiltro(imagem, filtro):
    resultado = cv.filter2D(imagem, ddepth=-1, kernel=filtro)
    resultado = np.clip(resultado, 0, 255).astype(np.uint8)
    return resultado


img = cv.imread('imgs/macaco.png', cv.IMREAD_GRAYSCALE)
novaImg = aplicarFiltro(img, h9)


cv.imshow('Filtro h1', novaImg)
cv.waitKey(0)
cv.destroyAllWindows()