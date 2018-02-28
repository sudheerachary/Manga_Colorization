
# coding: utf-8

# In[1]:


import os
import cv2
import numpy as np
import keras
from keras.layers import *
from keras.optimizers import *
from keras.models import *
import matplotlib.pyplot as plt
from keras.regularizers import *
from keras.utils import plot_model


# In[ ]:


fixed_seed_num = 1234
np.random.seed(fixed_seed_num)
random.seed(fixed_seed_num) # not sure if needed or not
tf.set_random_seed(fixed_seed_num)
keras.set_random_seed(fixed_seed_num)


# In[2]:


def generator_model():
    
    generator_input = Input(batch_shape=(None, 512, 512, 1), name='generator_input')
    
    # encoder
    conv1_32 = Conv2D(32,kernel_size=(3,3),strides=(1,1),padding='same',activation='elu')(generator_input)
    
    conv2_64 = Conv2D(64,kernel_size=(3,3),strides=(2,2),padding='same',activation='elu')(conv1_32)
    conv2_64 = BatchNormalization()(conv2_64)
    
    conv3_128 = Conv2D(128,kernel_size=(3,3),strides=(2,2),padding='same',activation='elu')(conv2_64)
    conv3_128 = BatchNormalization()(conv3_128)
    
    conv4_256 = Conv2D(256,kernel_size=(3,3),strides=(2,2),padding='same',activation='elu')(conv3_128)
    conv4_256 = BatchNormalization()(conv4_256)
    
    conv5_512 = Conv2D(512,kernel_size=(3,3),strides=(2,2),padding='same',activation='elu')(conv4_256)
    conv5_512 = BatchNormalization()(conv5_512)
    
    conv6_512 = Conv2D(512,kernel_size=(3,3),strides=(2,2),padding='same',activation='elu')(conv5_512)
    conv6_512 = BatchNormalization()(conv6_512)
    
    conv7_512 = Conv2D(512,kernel_size=(3,3),padding='same',activation='elu')(conv6_512)
    conv7_512 = BatchNormalization()(conv7_512)
    
    # decoder
    conv8_512 = Conv2D(512,kernel_size=(3,3),padding='same',activation='elu')(conv7_512)
    conv8_512 = BatchNormalization(axis=1)(conv8_512)
    
    deconv9_512 = Conv2DTranspose(512,kernel_size=(3,3),padding='same',activation='elu',strides=(2,2))(conv8_512)
    deconv9_512 = Concatenate()([deconv9_512,conv5_512])
    deconv9_512 = Conv2D(512,kernel_size=(3,3),padding='same',activation='elu')(deconv9_512)
    
    deconv10_256 = Conv2DTranspose(256,kernel_size=(3,3),padding='same',activation='elu',strides=(2,2))(deconv9_512)
    deconv10_256 = Concatenate()([deconv10_256,conv4_256])
    deconv10_256 = Conv2D(256,kernel_size=(3,3),padding='same',activation='elu')(deconv10_256)
    
    deconv11_128 = Conv2DTranspose(128,kernel_size=(3,3),padding='same',activation='elu',strides=(2,2))(deconv10_256)
    deconv11_128 = Concatenate()([deconv11_128,conv3_128])
    deconv11_128 = Conv2D(128,kernel_size=(3,3),padding='same',activation='elu')(deconv11_128)
    
    deconv12_64 = Conv2DTranspose(64,kernel_size=(3,3),padding='same',activation='elu',strides=(2,2))(deconv11_128)
    deconv12_64 = Concatenate()([deconv12_64,conv2_64])
    deconv12_64 = Conv2D(512,kernel_size=(3,3),padding='same',activation='elu')(deconv12_64)
    
    deconv13_32 = Conv2DTranspose(32,kernel_size=(3,3),padding='same',activation='elu',strides=(2,2))(deconv12_64)
    deconv13_32 = Concatenate()([deconv13_32,conv1_32])
    deconv13_32 = Conv2D(32,kernel_size=(3,3),padding='same',activation='elu')(deconv13_32)
    
    deconv14_3 = Conv2DTranspose(3,kernel_size=(3,3),padding='same',activation='elu')(deconv13_32)
    deconv14_3 = Concatenate()([deconv14_3,generator_input])
    
    output = Conv2D(3,kernel_size=(1,1),padding='same',activation='relu')(deconv14_3)
    
    model = Model(inputs=generator_input,outputs=output)
    
    return model

def discriminator_model():
    
    generator_input = Input(batch_shape=(None, 512, 512, 1), name='generator_output')
    generator_output = Input(batch_shape=(None, 512, 512, 3), name='generator_input')
    
    input1 = BatchNormalization()(generator_input)
    input2 = BatchNormalization()(generator_output)
    
    convi = Conv2D(32,kernel_size=(3,3),activation='elu',padding='same')(generator_input)
    convi = BatchNormalization()(convi)
    convo = Conv2D(32,kernel_size=(3,3),activation='elu',padding='same')(generator_output)
    convo = BatchNormalization()(convo)

    
    convi = Conv2D(64,kernel_size=(3,3),activation='elu',padding='same')(convi)
    convi = BatchNormalization()(convi)
    convo = Conv2D(64,kernel_size=(3,3),activation='elu',padding='same')(convo)
    convo = BatchNormalization()(convo)

    
    convi = Conv2D(64,kernel_size=(3,3),strides=(2,2),activation='elu',padding='same')(convi)
    convo = Conv2D(64,kernel_size=(3,3),strides=(2,2),activation='elu',padding='same')(convo)
    
    conv = Concatenate()([convi,convo])
    conv = BatchNormalization()(conv)
    
    conv = Conv2D(128,kernel_size=(3,3),strides=(2,2),activation='elu',padding='same')(conv)
    conv = BatchNormalization()(conv)
    
    conv = Conv2D(128,kernel_size=(3,3),strides=(2,2),activation='elu',padding='same')(conv)
    conv = BatchNormalization()(conv)
    
    conv = Conv2D(128,kernel_size=(3,3),strides=(2,2),activation='elu',padding='same')(conv)
    conv = BatchNormalization()(conv)
    
    conv = Conv2D(256,kernel_size=(3,3),strides=(2,2),activation='elu',padding='same')(conv)
    conv = BatchNormalization()(conv)
    
    conv = Conv2D(256,kernel_size=(3,3),strides=(2,2),activation='elu',padding='same')(conv)
    conv = BatchNormalization()(conv)
    
    conv = Flatten()(conv)
    conv = Dropout(0.5)(conv)
    
    conv = Dense(100,activation='elu')(conv)
    conv = Dropout(0.5)(conv)
    
    output = Dense(1,activation='sigmoid')(conv)
    
    model = Model(inputs=([generator_input,generator_output]),outputs=[output])
    
    return model

def cGAN_model(generator,discriminator):
    
    discriminator.trainable = False
    model = Model(inputs=generator.inputs,outputs=[discriminator([generator.input,generator.output]), generator.output])
    
    return model


# In[3]:


gen = generator_model()
gen.summary()

disc = discriminator_model()
disc.summary()

cGAN = cGAN_model(gen, disc)
cGAN.summary()

# plot_model(x, to_file='generator_model.png')
# SVG(model_to_dot(x).create(prog='dot', format='svg'))

disc.compile(loss=['binary_crossentropy'], optimizer=Adam(lr=0.0001), metrics=['accuracy'])

# plot_model(model, to_file='discriminator_model.png')
# SVG(model_to_dot(y).create(prog='dot', format='svg'))

cGAN.compile(loss=['binary_crossentropy','mse'], loss_weights=[1, 10], optimizer=Adam(lr=0.001))

# plot_model(model, to_file='cGAN_visualization.png')
# SVG(model_to_dot(z).create(prog='dot', format='svg'))


# In[8]:


samples = 250
dataset = '../../manga_dataset/' 
rgb = np.zeros((samples, 512, 512, 3))
gray = np.zeros((samples, 512, 512, 1))
for i, image in enumerate(os.listdir(dataset)[:samples]):
    I = cv2.imread(dataset+image)
    I = cv2.resize(I, (512, 512))
    J = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    J = J.reshape(J.shape[0], J.shape[1], 1)
    rgb[i] = I; gray[i] = J


# In[ ]:


epochs = 1000
for i in range(epochs):
    gen_image = gen.predict(gray, batch_size=16)   
    inputs = np.concatenate([gray, gray])
    outputs = np.concatenate([rgb, gen_image])
    y = np.concatenate([np.ones((samples, 1)), np.zeros((samples, 1))])
    disc.fit([inputs, outputs], y, epochs=1, batch_size=16)
    disc.trainable = False
    cGAN.fit(gray, [np.ones((samples, 1)), rgb], epochs=1, batch_size=1)
    disc.trainable = True
    if i%10 == 0:
        gen.save_weights('../../gen_imgs/'+str(i)+'.h5')
    if i%2 == 0:
        for j in range(10):
            if not os.path.exists('../../gen_imgs/'+str(j)+'/'):
                os.mkdir('../../gen_imgs/'+str(j)+'/')
            cv2.imwrite('../../gen_imgs/'+str(j)+'/'+str(i)+'.jpg', gen_image[j])


# In[ ]:


print os.listdir(dataset)

