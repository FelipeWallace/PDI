import cv2
import numpy as np

# Funções morfológicas 


def erosao(imagem, elemento_estruturante):
    array_pad = np.pad(imagem, ((3, 3), (3, 3)), mode='constant', constant_values=0)

    elemento_estruturante_bin = np.where(elemento_estruturante != 0, 1, 0)
    #Aumentar a imagem com valor 0 nas bordas para não perder informação

    alutra_pad, largura_pad = array_pad.shape
    ee_altura, ee_largura = elemento_estruturante.shape
    temp = np.zeros((alutra_pad, largura_pad), dtype=np.uint8)

    for i in range(ee_altura // 2, alutra_pad - ee_altura // 2):
        for j in range(ee_largura // 2, largura_pad - ee_largura // 2):
            array = array_pad[i-ee_altura//2:i+ee_altura//2+1, j-ee_largura//2:j+ee_largura//2+1] * elemento_estruturante_bin
            temp[i, j] = np.min(array[array!=0])
    
    resultado = temp.copy()[3:-3, 3:-3]

    return resultado


def dilatacao(imagem, elemento_estruturante):
    array_pad = np.pad(imagem, ((3, 3), (3, 3)), mode='constant', constant_values=0)

    elemento_estruturante_bin = np.where(elemento_estruturante != 0, 1, 0)
    #Aumentar a imagem com valor 0 nas bordas para não perder informação

    alutra_pad, largura_pad = array_pad.shape
    ee_altura, ee_largura = elemento_estruturante.shape
    temp = np.zeros((alutra_pad, largura_pad), dtype=np.uint8)

    for i in range(ee_altura // 2, alutra_pad - ee_altura // 2):
        for j in range(ee_largura // 2, largura_pad - ee_largura // 2):
            array = array_pad[i-ee_altura//2:i+ee_altura//2+1, j-ee_largura//2:j+ee_largura//2+1] * elemento_estruturante_bin
            temp[i, j] = np.max(array[array!=0])
    
    resultado = temp.copy()[3:-3, 3:-3]

    return resultado


def gradiente_morfologico(imagem, elemento_estruturante):
    erosao_resultado = erosao(imagem, elemento_estruturante)
    dilatacao_resultado = dilatacao(imagem, elemento_estruturante)

    resultado = dilatacao_resultado - erosao_resultado

    return resultado


def abertura_morfologica(imagem, elemento_estruturante):
    erosao_resultado = erosao(imagem, elemento_estruturante)
    resultado = dilatacao(erosao_resultado, elemento_estruturante)

    return resultado


def fechamento_morfologico(imagem, elemento_estruturante):
    dilatacao_resultado = dilatacao(imagem, elemento_estruturante)
    resultado = erosao(dilatacao_resultado, elemento_estruturante)

    return resultado

# Imagem fornecida
imagem = np.array([[252, 46, 115, 18, 73, 203, 60, 229, 112, 183],
                  [109, 31, 20, 53, 225, 58, 54, 28, 170, 94],
                  [99, 73, 116, 115, 183, 146, 177, 88, 14, 141],
                  [79, 176, 132, 54, 144, 148, 231, 157, 244, 187],
                  [207, 28, 4, 194, 111, 122, 172, 61, 211, 71],
                  [185, 199, 124, 123, 40, 195, 134, 112, 17, 194],
                  [26, 3, 168, 251, 12, 85, 98, 205, 174, 34],
                  [234, 222, 166, 121, 99, 167, 33, 35, 43, 183],
                  [237, 102, 254, 45, 206, 234, 49, 144, 1, 70],
                  [17, 231, 44, 224, 67, 195, 148, 68, 127, 42]], dtype=np.uint8)

# Elemento estruturante
elemento_estruturante = np.array([[0, 0, 0, 1, 0, 0, 0],
                                  [0, 0, 1, 10, 1, 0, 0],
                                  [0, 1, 10, 20, 10, 1, 0],
                                  [1, 10, 20, 40, 20, 10, 1],
                                  [0, 1, 10, 20, 10, 1, 0],
                                  [0, 0, 1, 10, 1, 0, 0],
                                  [0, 0, 0, 1, 0, 0, 0],], dtype=np.uint8)

# Processos
erosao_resultado = erosao(imagem, elemento_estruturante)
dilatacao_resultado = dilatacao(imagem, elemento_estruturante)
gradiente_morfologico_resultado = gradiente_morfologico(imagem, elemento_estruturante)
abertura_morfologica_resultado = abertura_morfologica(imagem, elemento_estruturante)
fechamento_morfologico_resultado = fechamento_morfologico(imagem, elemento_estruturante)

# Resultados
print("Resultado Erosão:")
print(erosao_resultado)
print("\nResultado Dilatação:")
print(dilatacao_resultado)
print("\nResultado Gradiente Morfológico:")
print(gradiente_morfologico_resultado)
print("\nResultado Abertura Morfológica:")
print(abertura_morfologica_resultado)
print("\nResultado Fechamento Morfológico:")
print(fechamento_morfologico_resultado)
