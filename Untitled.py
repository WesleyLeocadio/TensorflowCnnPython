#!/usr/bin/env python
# coding: utf-8

# In[1]:


#pip install opencv-python
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm


# In[2]:


DATADIR = "/home/weslley/Downloads/Projeto tcc/Processadas/Train"
DATADIR_VALIDACAO = "/home/weslley/Downloads/Projeto tcc/Processadas/Validacao"
DATADIR_TEST = "/home/weslley/Downloads/Projeto tcc/Processadas/Test"


CATEGORIES = ["germinou", "naogerminou"]

for category in CATEGORIES:
    path = os.path.join(DATADIR,category)  
    for img in os.listdir(path):  
        img_array = cv2.imread(os.path.join(path,img) )  # convert to array
        plt.imshow(img_array, cmap='gray')  # graph it
        plt.show()  # display!

        break  # we just want one for now so break
    break  #...and one more!


# In[3]:


print(img_array.shape)


# In[4]:


IMG_SIZE = 100

new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
plt.imshow(new_array, cmap='gray')
plt.show()


# In[5]:


print(new_array.shape)


# In[6]:



training_data  =  [] 

def create_training_data():
    for category in CATEGORIES:  #  cães e gatos 
        # cria caminho para cães e gatos 
        path = os.path.join(DATADIR,category)  
        # obtém a classificação (0 ou 1). 0 = cão 1 = gato 
        class_num = CATEGORIES.index(category)  

         # itere sobre cada imagem por 
        for img in tqdm(os.listdir(path)):  
            try:
                # convertido a matriz 
                img_array = cv2.imread(os.path.join(path,img) ) 
                # redimensione para normalizar o tamanho dos dados 
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE)) 
                 # adicione isso aos nossos dados de treinamento, 
                training_data.append([new_array, class_num]) 
                 # no interesse em manter a saída limpa ... 
            except Exception as e: 
                pass
          # exceto OSError como e: 
            # print ("OSErrroBad img provavelmente", e, os.path.join ( path, img)) 
            #except Exceção como e: 
            # print ("exceção geral", e, os.path.join (path, img)) 

create_training_data()

print(len(training_data))


# In[ ]:





# In[7]:


validacao_data  =  [] 

def create_validacao_data():
    for category in CATEGORIES:  #  cães e gatos 
        # cria caminho para cães e gatos 
        path = os.path.join(DATADIR_VALIDACAO,category)  
        # obtém a classificação (0 ou 1). 0 = cão 1 = gato 
        class_num = CATEGORIES.index(category)  

         # itere sobre cada imagem por 
        for img in tqdm(os.listdir(path)):  
            try:
                # convertido a matriz 
                img_array = cv2.imread(os.path.join(path,img) ) 
                # redimensione para normalizar o tamanho dos dados 
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE)) 
                 # adicione isso aos nossos dados de treinamento, 
                validacao_data.append([new_array, class_num]) 
                 # no interesse em manter a saída limpa ... 
            except Exception as e: 
                pass
          # exceto OSError como e: 
            # print ("OSErrroBad img provavelmente", e, os.path.join ( path, img)) 
            #except Exceção como e: 
            # print ("exceção geral", e, os.path.join (path, img)) 

create_validacao_data()

print(len(validacao_data))


# In[8]:


test_data  =  [] 

def create_test_data():
    for category in CATEGORIES:  #  cães e gatos 
        # cria caminho para cães e gatos 
        path = os.path.join(DATADIR_TEST,category)  
        # obtém a classificação (0 ou 1). 0 = cão 1 = gato 
        class_num = CATEGORIES.index(category)  

         # itere sobre cada imagem por 
        for img in tqdm(os.listdir(path)):  
            try:
                # convertido a matriz 
                img_array = cv2.imread(os.path.join(path,img) ) 
                # redimensione para normalizar o tamanho dos dados 
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE)) 
                 # adicione isso aos nossos dados de treinamento, 
                test_data.append([new_array, class_num]) 
                 # no interesse em manter a saída limpa ... 
            except Exception as e: 
                pass
          # exceto OSError como e: 
            # print ("OSErrroBad img provavelmente", e, os.path.join ( path, img)) 
            #except Exceção como e: 
            # print ("exceção geral", e, os.path.join (path, img)) 

create_test_data()

print(len(test_data))


# In[9]:


import random

random.shuffle(training_data)
random.shuffle(validacao_data)
random.shuffle(test_data)


# In[10]:


for sample in training_data[:10]:
    print(sample[1])


# In[11]:


X = []
y = []

for features,label in training_data:
    X.append(features)
    y.append(label)

#print(X[0].reshape(-1, IMG_SIZE, IMG_SIZE, 1))

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
y= np.array(y)


# In[12]:


print(y[6])


# In[13]:


X_val = []
y_val = []

for features,label in validacao_data:
    X_val.append(features)
    y_val.append(label)

#print(X[0].reshape(-1, IMG_SIZE, IMG_SIZE, 1))

X_val = np.array(X_val).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
X_val= X_val/255.0


# In[14]:


print(y_val[1])


# In[15]:


X_test = []
y_test = []

for features,label in test_data:
    X_test.append(features)
    y_test.append(label)

#print(X[0].reshape(-1, IMG_SIZE, IMG_SIZE, 1))

X_test = np.array(X_test).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
X_test=X_test/255.0


# In[16]:


print((y_test))


# In[ ]:





# In[17]:


import pickle

pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()


# In[18]:


import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D


# In[19]:


import pickle

pickle_in = open("X.pickle","rb")
X= pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
b = pickle.load(pickle_in)
y = np.array(b)

X = X/255.0


# In[ ]:





# In[39]:


model = Sequential()

model.add(Conv2D(64, (3,3), input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

model.add(Dense(64))
model.add(Activation('relu'))

model.add(Dense(64))
model.add(Activation('relu'))


model.add(Dense(2))
model.add(Activation('softmax'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# In[40]:



early_stopping= tf.keras.callbacks.EarlyStopping(patience=2)
model.fit(X, y, batch_size=100, epochs=10, callbacks=[early_stopping],validation_data=(X_val,y_val),verbose=2)
 


# In[38]:


print(y_test[25])


# In[23]:


test_loss, test_accuracy=model.evaluate(X_test,y_test)


# In[24]:


classe = model.predict(X_test)


# In[37]:


print(classe[25])


# In[ ]:




