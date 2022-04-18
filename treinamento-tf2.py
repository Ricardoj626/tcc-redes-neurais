# Import do TF e da ferramentas usadas
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

# Import de outras bibliotecas que serão usada
import numpy as np

import datetime
import os

num = 24
loads = np.load('images-reconhecimentoCanhoto.npz')
images = loads['images']
resultados = loads['resultados']

mcp_save = tf.keras.callbacks.ModelCheckpoint(filepath='modelos_treinados/best{}.h5'.format(num), save_best_only=True, monitor='val_loss', mode='min')

modelo = tf.keras.Sequential()
modelo.add(Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu', input_shape=(256,256,1), padding='same'))
modelo.add(Conv2D(filters = 128, kernel_size = (3,3), activation = 'relu', padding='same'))
modelo.add(Conv2D(filters = 128, kernel_size = (2,2), activation = 'relu', padding='same'))
modelo.add(MaxPooling2D(pool_size = (2,2)))
Dropout(0.1) 
modelo.add(Conv2D(filters = 128, kernel_size = (3,3), activation = 'relu', padding='same'))
modelo.add(Conv2D(filters = 128, kernel_size = (2,2), activation = 'relu', padding='same'))
modelo.add(MaxPooling2D(pool_size = (2,2), padding='same'))  
Dropout(0.1) 
modelo.add(Conv2D(filters = 128, kernel_size = (3,3), activation = 'relu', padding='same'))
modelo.add(Conv2D(filters = 256, kernel_size = (2,2), activation = 'relu', padding='same'))
modelo.add(MaxPooling2D(pool_size = (2,2), padding='same'))
Dropout(0.1) 
modelo.add(Conv2D(filters = 128, kernel_size = (3,3), activation = 'relu', padding='same'))
modelo.add(Conv2D(filters = 128, kernel_size = (2,2), activation = 'relu', padding='same'))
modelo.add(MaxPooling2D(pool_size = (2,2), padding='same'))
Dropout(0.1) 
modelo.add(Conv2D(filters = 128, kernel_size = (3,3), activation = 'relu', padding='same'))
modelo.add(Conv2D(filters = 128, kernel_size = (2,2), activation = 'relu', padding='same'))
modelo.add(MaxPooling2D(pool_size = (2,2), padding='same'))
Dropout(0.1) 
modelo.add(Conv2D(filters = 128, kernel_size = (2,2), activation = 'relu', padding='same', strides = (2,2)))
modelo.add(Flatten())
modelo.add(Dense(units=288, activation='relu'))
modelo.add(Dense(units=512, activation='relu'))
modelo.add(Dense(units=1024, activation='relu'))
modelo.add(Dense(units=1024, activation='relu'))
modelo.add(Dense(units=512, activation='relu'))
modelo.add(Dense(units=288, activation='relu'))
modelo.add(Dense(units=2, activation='softmax'))
modelo.compile(loss='categorical_crossentropy', optimizer='adam')

print('---------------------')
print(modelo.summary())
print('---------------------')
results = modelo.fit(images, resultados, batch_size = 48, epochs=20, callbacks=[mcp_save],  validation_split=0.25)

## salvando modelos

modelo.save('modelos_treinados/model{}.h5'.format(num))
#modelo.save_weights('modelos_treinados/model_weights.ckpt', save_format='tf')

model_json = modelo.to_json()
with open("modelos_treinados/model{}.json".format(num), "w") as json_file:
    json_file.write(model_json)
## serialize weights to HDF5
#modelo.save_weights("modelos_treinados/date-model_2.h5")
print("Saved model to disk")