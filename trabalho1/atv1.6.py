import numpy as np
import cv2 as cv


imagem = cv.imread('imgs/macaco.png')



cv.imshow('image', imagem)
cv.waitKey(0)
cv.destroyAllWindows()