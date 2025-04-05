import numpy as np
import cv2 as cv


im1 = cv.imread('imgs/macaco.png')
im2 = im1.copy()
im2 = np.flip(im2)

media = np.average([im1, im2], axis=0, weights=[0.5, 0.5])
media = np.clip(media, 0, 255).astype(np.uint8)


concatenado = cv.hconcat([im1, media])
cv.imwrite('imgs_geradas/atv1.7/source.png', concatenado)

cv.imshow('image', media)
cv.waitKey(0)
cv.destroyAllWindows()