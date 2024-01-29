import cv2
import numpy as np
import matplotlib.pyplot as plt



# Carregue a imagem em escala de cinza
imagem = cv2.imread('Lenna.png', cv2.IMREAD_GRAYSCALE)

# Calcula o histograma
hist = cv2.calcHist([imagem], [0], None, [256], [0,256])
    
# Calcula o histograma acumulado
hist_acumulado = np.cumsum(hist)
    
# Normaliza o histograma acumulado
hist_acumulado = (hist_acumulado / hist_acumulado[-1]) * 255
    
# Aplica a equalização na imagem
imagem_equalizada = cv2.LUT(imagem, hist_acumulado.astype('uint8'))

# Exiba a imagem original e a imagem equalizada
plt.figure(figsize=(10,5))
plt.subplot(121)
plt.imshow(imagem, cmap='gray')
plt.title('Imagem Original')

plt.subplot(122)
plt.imshow(imagem_equalizada, cmap='gray')
plt.title('Imagem Equalizada')

# Exiba os histogramas
plt.figure(figsize=(10,5))
plt.subplot(121)
plt.plot(hist)
plt.title('Histograma Original')

plt.subplot(122)
plt.plot(hist_acumulado)
plt.title('Histograma Acumulado Equalizado')

plt.show()
