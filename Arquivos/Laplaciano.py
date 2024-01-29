import cv2
import numpy as np
from matplotlib import pyplot as plt


def laplaciano():
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=int)
    return kernel


img = np.array([[252, 46, 115, 18, 73, 203, 60, 229, 112, 183],
                  [109, 31, 20, 53, 225, 58, 54, 28, 170, 94],
                  [99, 73, 116, 115, 183, 146, 177, 88, 14, 141],
                  [79, 176, 132, 54, 144, 148, 231, 157, 244, 187],
                  [207, 28, 4, 194, 111, 122, 172, 61, 211, 71],
                  [185, 199, 124, 123, 40, 195, 134, 112, 17, 194],
                  [26, 3, 168, 251, 12, 85, 98, 205, 174, 34],
                  [234, 222, 166, 121, 99, 167, 33, 35, 43, 183],
                  [237, 102, 254, 45, 206, 234, 49, 144, 1, 70],
                  [17, 231, 44, 224, 67, 195, 148, 68, 127, 42]], dtype=np.uint8)


img_pad = np.pad(img, pad_width=1, mode='constant', constant_values=0)

m, n = img_pad.shape


kernel = laplaciano()

nova_image = np.zeros([m, n])

for i in range(1, m-1):
    for j in range(1, n-1):

        valor1 = img_pad[i-1, j-1]*kernel[0, 0] + img_pad[i-1, j]*kernel[0,1] + img_pad[i-1, j+1]*kernel[0, 2] +\
                 img_pad[i, j-1]*kernel[1, 0] + img_pad[i, j]*kernel[1,1] + img_pad[i, j+1]*kernel[1, 2] +\
                 img_pad[i+1, j-1]*kernel[2, 0] + img_pad[i+1, j]*kernel[2,1] + img_pad[i+1, j+1]*kernel[2, 2]
        
        if valor1 < 0:
            valor1 = 0
        elif valor1 > 255:
            valor1 = 255

        nova_image[i, j] = int(valor1)


print(nova_image)
