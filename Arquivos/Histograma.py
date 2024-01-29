import cv2
import numpy as np
from matplotlib import pyplot as plt


def histograma(image):

    plt.hist(image.ravel(), 256, [0, 256])
    plt.show()


imagem = cv2.imread('Lenna.png')

linhas = imagem.shape[0]
colunas = imagem.shape[1]

cv2.imshow("Original", imagem)

for x in range(0, linhas):
    for y in range(0, colunas):
        (b, g, r) = imagem[x, y]
        imagem[x, y] = ((int(r) + int(g) + int(b))/3)

cv2.imshow("Imagem modificada", imagem)

histograma(imagem)

cv2.waitKey(0)

