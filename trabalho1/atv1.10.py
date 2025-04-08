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


def aplicarFiltro(imagem, filtro, nomeArq):
    resultado = cv.filter2D(imagem, ddepth=-1, kernel=filtro)
    resultado = np.clip(resultado, 0, 255).astype(np.uint8)
    concatenado = cv.hconcat([imagem, resultado])
    cv.imwrite(f'imgs_geradas/atv1.10/{nomeArq}.png', concatenado)
    return resultado


img = cv.imread('imgs/macaco.png', cv.IMREAD_GRAYSCALE)
img1 = aplicarFiltro(img, h1, 'filtroH1')
img2 = aplicarFiltro(img, h2, 'filtroH2')
img3 = aplicarFiltro(img, h3, 'filtroH3')
img4 = aplicarFiltro(img, h4, 'filtroH4')
img5 = aplicarFiltro(img, h5, 'filtroH5')
img6= aplicarFiltro(img, h6, 'filtroH6')
img7 = aplicarFiltro(img, h7, 'filtroH7')
img8 = aplicarFiltro(img, h8, 'filtroH8')
img9 = aplicarFiltro(img, h9, 'filtroH9')
img10 = aplicarFiltro(img, h10, 'filtroH10')
img11 = aplicarFiltro(img, h11, 'filtroH11')
