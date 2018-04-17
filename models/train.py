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
from keras.callbacks import TensorBoard
from time import time
from keras.preprocessing.image import ImageDataGenerator
from GAN_models import *
from losses import *

# constants
epochs = 50
x_shape = 512
y_shape = 512
fixed_seed_num = 1234
np.random.seed(fixed_seed_num)
tf.set_random_seed(fixed_seed_num)

# trains cGAN model
def train(gen, disc, cGAN, gray, rgb, gray_val, rgb_val, batch):
    samples = len(rgb)
    gen_image = gen.predict(gray, batch_size=16)   
    gen_image_val = gen.predict(gray_val, batch_size=8)
    inputs = np.concatenate([gray, gray])
    outputs = np.concatenate([rgb, gen_image])
    y = np.concatenate([np.ones((samples, 1)), np.zeros((samples, 1))])
    disc.fit([inputs, outputs], y, epochs=1, batch_size=4)
    disc.trainable = False
    # cGAN.fit(gray, [np.ones((samples, 1)), rgb], epochs=1, batch_size=batch, callbacks=[tensorboard],validation_data=[gray_val,[np.ones((val_samples,1)),rgb_val]])
    cGAN.fit(gray, [np.ones((samples, 1)), rgb], epochs=1, batch_size=batch,validation_data=[gray_val,[np.ones((val_samples,1)),rgb_val]])
    disc.trainable = True

# initialises the cGAN model with (generator, discriminator)
gen = generator_model(x_shape,y_shape)
disc = discriminator_model(x_shape,y_shape)
cGAN = cGAN_model(gen, disc)


# compile with custom loss functions
disc.compile(loss=['binary_crossentropy'], optimizer=Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08), metrics=['accuracy'])
cGAN.compile(loss=['binary_crossentropy',custom_loss_2], loss_weights=[5, 100], optimizer=Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08))
tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
cGAN.load_weights("../datasets/generated_images/model.h5")

# constants
dataset = '../datasets/train/' 
store2 = '../generated_images/'
val_data = '../datasets/validation/'
store = '../datasets/generated_Images/'

samples = len(os.listdir(dataset))
val_samples = len(os.listdir(val_data))
rgb = np.zeros((samples, x_shape, y_shape, 3))
gray = np.zeros((samples, x_shape, y_shape, 1))
rgb_val = np.zeros((val_samples, x_shape, y_shape, 3))
gray_val = np.zeros((val_samples, x_shape, y_shape, 1))

# get train dataset
y_train = np.zeros((samples, 1))
for i, image in enumerate(os.listdir(dataset)[:samples]):
    I = cv2.imread(dataset+image)
    I = cv2.resize(I, (x_shape, y_shape))
    J = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    J = J.reshape(J.shape[0], J.shape[1], 1)
    rgb[i] = I; gray[i] = J

# get validation dataset 
for i, image in enumerate(os.listdir(val_data)[:val_samples]):
    I = cv2.imread(val_data+image)
    I = cv2.resize(I, (x_shape, y_shape))
    J = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    J = J.reshape(J.shape[0], J.shape[1], 1)
    rgb_val[i] = I; gray_val[i] = J


# initialise data generator
datagen = ImageDataGenerator(zoom_range=0.2,fill_mode='wrap',horizontal_flip=True,
                            vertical_flip=True, rotation_range=15)
datagen.fit(rgb)

# train the cGAN model with specified number of epochs
for e in range(epochs):
    batches = 0
    print('Epoch', e)
    for x_batch, y_batch in datagen.flow(rgb, y_train, batch_size=samples):
        for i in range(len(x_batch)):
            gray[i] = cv2.cvtColor(x_batch[i], cv2.COLOR_BGR2GRAY).reshape((x_shape, y_shape, 1))
        
        params = (gen, disc, cGAN, gray, 
                x_batch, gray_val, rgb_val, 1)
        train(*params)
        batches += 1
        if batches >= 1:
            # we need to break the loop by hand because
            # the generator loops indefinitely
            break
    if e%5 == 0:
        cGAN.save_weights(store+str(e)+'.h5') 
    gen_image_val = gen.predict(gray_val, batch_size=8)
    # if e%1 == 0: 
    #     for j in range(val_samples):
    #         if not os.path.exists(store2+str(j)+'/'):
    #             os.mkdir(store+str(j)+'/')
    #         cv2.imwrite(store+str(j)+'/'+str(e)+'.jpg', gen_image_val[j])