from keras.preprocessing import image
from keras.utils import to_categorical
import numpy as np
from load_model import load_model

def predict():
    images_teste =[]
    label_images = []
    resultados=[]
    
    for i in range (3379870,3385739):
        try:
            img_predict = 'image/recortes/{}.jpg'.format(i)    
            img = image.load_img(img_predict,color_mode="grayscale", target_size=(64,256))
            x = image.img_to_array(img)
            images_teste.append(1-x/255)
            label_images.append(i)
            resultados.append(1)
        except:
            pass
        
    images_teste = np.array(images_teste)
    resultados = to_categorical(resultados)
    resultados_images_teste = load_model(images_teste)
    return (resultados_images_teste, label_images)
