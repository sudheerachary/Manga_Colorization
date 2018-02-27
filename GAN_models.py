import numpy as np
import keras
from keras.layers import *
from keras.optimizers import *
from keras.models import *
from keras.regularizers import *
def generator_model():
    
    generator_input = Input(batch_shape=(None, 512, 512, 3), name='generator_input')
    
    #encoder
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
    
    #decoder
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
    
    generator_input = Input(batch_shape=(None, 512, 512, 3), name='generator_output')
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
    
    model = Model(inputs=([generator_input,generator_output]),outputs=[output,generator_output])
    
    return model

def cGAN_model(generator,discriminator):
    
    discriminator.trainable = False
    model = Model(inputs=generator.inputs,outputs=discriminator([generator.input,generator.output]))
    
    return model
