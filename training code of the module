#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import keras libraries
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten


# In[2]:


model=Sequential()


# In[3]:


model.add(Conv2D(32,3,3,input_shape=(64,64,3),activation='relu'))#1St parameter =no of features detectors 2nd& 3rd =Size of feature detector, 
#4th input image size,5 th parameter is channel for color=3 gray scale=1,6 th to avoid negative pixels we use activation function


# In[4]:


model.add(MaxPooling2D(pool_size=(2,2)))#1parmeter=size of pooling matrix


# In[5]:


model.add(Flatten())


# In[6]:


model.add(Dense(output_dim=64,activation='relu',init='random_uniform'))


# In[7]:


model.add(Dense(output_dim=3,activation='softmax',init='random_uniform'))


# In[8]:


model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])


# In[9]:


from keras.preprocessing.image import ImageDataGenerator


# In[10]:


train_datagen=ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)


# In[11]:


x_train = train_datagen.flow_from_directory(r'C:/Users/akura/New folder (4)/training',target_size=(64,64),batch_size=32,class_mode='categorical')
x_test = test_datagen.flow_from_directory(r'C:/Users/akura/New folder (4)/test',target_size=(64,64),batch_size=32,class_mode='categorical')


# In[18]:


print(x_train.class_indices)


# In[13]:


print(x_test.class_indices)


# In[17]:


model.fit_generator(x_train,samples_per_epoch = 280,epochs=25,validation_data=x_test,nb_val_samples=82)#(samples_per_epoch= no of traininig or testing images/batch size)
         #  =8000/32=250


# In[19]:


model.save('mymodel.h5')


# In[ ]:




