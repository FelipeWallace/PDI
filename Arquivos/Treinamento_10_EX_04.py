import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.fftpack import fft2, ifft2, fftshift

def apply_ideal_filter(image, cutoff):
    f_transform = fft2(image)
    f_transform_shifted = fftshift(f_transform)

    rows, cols = image.shape
    center_row, center_col = rows // 2, cols // 2

    mask = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - center_row)**2 + (j - center_col)**2)
            if distance <= cutoff:
                mask[i, j] = 1

    f_transform_filtered = f_transform_shifted * mask
    result = np.abs(ifft2(fftshift(f_transform_filtered)))

    return result

def apply_butterworth_filter(image, cutoff, order):
    f_transform = fft2(image)
    f_transform_shifted = fftshift(f_transform)

    rows, cols = image.shape
    center_row, center_col = rows // 2, cols // 2

    mask = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - center_row)**2 + (j - center_col)**2)
            mask[i, j] = 1 / (1 + (distance / cutoff)**(2 * order))

    f_transform_filtered = f_transform_shifted * mask
    result = np.abs(ifft2(fftshift(f_transform_filtered)))

    return result

def apply_gaussian_filter(image, sigma):
    f_transform = fft2(image)
    f_transform_shifted = fftshift(f_transform)

    rows, cols = image.shape
    center_row, center_col = rows // 2, cols // 2

    mask = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - center_row)**2 + (j - center_col)**2)
            mask[i, j] = np.exp(-distance**2 / (2 * sigma**2))

    f_transform_filtered = f_transform_shifted * mask
    result = np.abs(ifft2(fftshift(f_transform_filtered)))

    return result

# Carregando uma imagem de exemplo
image = cv2.imread('Lenna.png', cv2.IMREAD_GRAYSCALE)

# Convertendo a imagem para float32
image = np.float32(image)

# Aplicando os filtros
cutoff_frequency = 30
butterworth_order = 2
gaussian_sigma = 20

result_ideal = apply_ideal_filter(image, cutoff_frequency)
result_butterworth = apply_butterworth_filter(image, cutoff_frequency, butterworth_order)
result_gaussian = apply_gaussian_filter(image, gaussian_sigma)

# Exibindo os resultados
plt.figure(figsize=(10, 6))

plt.subplot(231), plt.imshow(image, cmap='gray'), plt.title('Imagem Original')
plt.subplot(232), plt.imshow(np.log(1 + np.abs(fftshift(fft2(image)))), cmap='gray'), plt.title('Espectro de Frequência')
plt.subplot(233), plt.imshow(result_ideal, cmap='gray'), plt.title('Filtro Ideal')
plt.subplot(234), plt.imshow(result_butterworth, cmap='gray'), plt.title('Filtro Butterworth')
plt.subplot(235), plt.imshow(result_gaussian, cmap='gray'), plt.title('Filtro Gaussiano')

plt.show()
