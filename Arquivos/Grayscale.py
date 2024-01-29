import cv2

# Carrega a imagem
imagem = cv2.imread('Lenna.png')

# Obtém as dimensões da imagem
linhas = imagem.shape[0]
colunas = imagem.shape[1]

# Exibe a imagem original
cv2.imshow("Original", imagem)

# Converte a imagem para escala de cinza
for x in range(0, linhas):
    for y in range(0, colunas):
        # Calcula a média das intensidades dos canais R, G e B
        (b, g, r) = imagem[x, y]
        intensidade = (int(r) + int(g) + int(b)) // 3
        # Define a mesma intensidade nos canais R, G e B
        imagem[x, y] = (intensidade, intensidade, intensidade)

# Exibe a imagem em escala de cinza
cv2.imshow("Imagem modificada", imagem)

# Salva a imagem em escala de cinza
cv2.imwrite("Lenna_grayscale.jpg", imagem)

# Aguarda até que uma tecla seja pressionada e fecha a janela
cv2.waitKey(0)
