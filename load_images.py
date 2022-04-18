from keras.utils import to_categorical
from keras.preprocessing import image
import numpy as np
import os, copy

def rotate(matriz):
    nova_matriz = copy.deepcopy(matriz)
    for i in range(len(matriz)):
        for j in range(len(matriz[i])):
            nova_matriz[j][len(matriz)-(i+1)] = matriz[i][j]
    return nova_matriz

def calculaAngulo(img_p):
    img = image.load_img(img_p,color_mode="grayscale")
    x = image.img_to_array(img)
    altura = x.shape[0]
    largura = x.shape[1]
    img = image.load_img(img_p,color_mode="grayscale", target_size=(256,256))
    x = image.img_to_array(img)
    if(largura/altura<1):
        x1 = rotate(x)
        x2 =rotate(rotate(rotate(x)))
    else:
        x1 = x
        x2 = rotate(rotate(x))
    return x1, x2

def load_images():
    images=[]
    resultados = []
    caminho_local = '/home/ricardo/Documentos/comprovei/DATASETS/RECONHECIMENTO_CANHOTO/3 tentativa/verdadeiro/'
    caminhos = [os.path.join(caminho_local, nome) for nome in os.listdir(caminho_local)]
    arquivos = [arq for arq in caminhos if os.path.isfile(arq)]
    jpgs = [arq for arq in arquivos if arq.lower().endswith(".jpg")]
    k=0
    for img_p in jpgs:
        try:
            k+=1
            print(k)
            x1,x2 = calculaAngulo(img_p)
            images.append(1-x1/255)
            resultados.append(1)
            images.append(1-x2/255)
            resultados.append(1)
        except:
            pass
            
    caminho_local = '/home/ricardo/Documentos/comprovei/DATASETS/RECONHECIMENTO_CANHOTO/3 tentativa/falso/'
    caminhos = [os.path.join(caminho_local, nome) for nome in os.listdir(caminho_local)]
    arquivos = [arq for arq in caminhos if os.path.isfile(arq)]
    jpgs = [arq for arq in arquivos if arq.lower().endswith(".jpg")]
    k=0
    for img_p in jpgs:
        try:
            k+=1
            print(k)
            x1,x2 = calculaAngulo(img_p)
            images.append(1-x1/255)
            resultados.append(0)
            images.append(1-x2/255)
            resultados.append(0)
        except:
            pass
        
    img_p = '/home/ricardo/Documentos/comprovei/DATASETS/RECONHECIMENTO_CANHOTO/3 tentativa/verdadeiro/1545.jpg'
    img = image.load_img(img_p,color_mode="grayscale", target_size=(256,256))
    x = image.img_to_array(img)
    images.append(1-x/255)
    resultados.append(1)
    
    
    images = np.array(images)
    resultados = to_categorical(resultados)
    np.savez('images-reconhecimentoCanhoto.npz', images=images, resultados=resultados)
    return (images, resultados)