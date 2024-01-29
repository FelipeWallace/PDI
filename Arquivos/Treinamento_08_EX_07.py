import cv2
import numpy as np

# Carregue a imagem do arquivo
imagem = cv2.imread('Lenna.png', cv2.IMREAD_GRAYSCALE)

# Normalização (escala para valores entre 0 e 1)
imagem_normalizada = imagem.astype('float32') / 255.0

# Negativo (inversão das intensidades)
imagem_negativa = 255 - imagem

# Limiarização (defina um valor de limiar)
limiar = 128
imagem_limiarizada = cv2.threshold(imagem, limiar, 255, cv2.THRESH_BINARY)[1]

# Exiba as imagens resultantes
cv2.imshow("Imagem Original", imagem)
cv2.imshow("Imagem Normalizada", imagem_normalizada)
cv2.imshow("Imagem Negativa", imagem_negativa)
cv2.imshow("Imagem Limiarizada", imagem_limiarizada)

# cv2.imwrite("Imagem Original.png", imagem)
# cv2.imwrite("ImagemNormalizada.png", imagem_normalizada)
# cv2.imwrite("Imagem Negativa.png", imagem_negativa)
# cv2.imwrite("Imagem Limiarizada.png", imagem_limiarizada)

cv2.waitKey(0)
cv2.destroyAllWindows()
