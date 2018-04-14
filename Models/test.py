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


fixed_seed_num = 1234
np.random.seed(fixed_seed_num)
tf.set_random_seed(fixed_seed_num)
x_shape = 512
y_shape = 512

gen = generator_model(x_shape,y_shape)
# gen.summary()

disc = discriminator_model(x_shape,y_shape)
# disc.summary()

cGAN = cGAN_model(gen, disc)

disc.compile(loss=['binary_crossentropy'], optimizer=Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08), metrics=['accuracy'])

cGAN.compile(loss=['binary_crossentropy',custom_loss_2], loss_weights=[5, 100], optimizer=Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08))

tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

cGAN.load_weights("../Datasets/Generated_Images/525.h5")

dataset = '../Datasets/Train/' 
val_data = '../Datasets/Test/'
store = "../Datasets/Generated_Images/"
store2 = '../Generated_Images/'
samples = len(os.listdir(dataset))
# samples = 70
val_samples = len(os.listdir(val_data))
# model_path = store+str(1200)+'.h5'
# samples = 6
rgb = np.zeros((samples, x_shape, y_shape, 3))
gray = np.zeros((samples, x_shape, y_shape, 1))
rgb_val = np.zeros((val_samples, x_shape, y_shape, 3))
gray_val = np.zeros((val_samples, x_shape, y_shape, 1))
y_train = np.zeros((samples,1))


for i, image in enumerate(os.listdir(val_data)[:val_samples]):
    I = cv2.imread(val_data+image)
    I = cv2.resize(I, (x_shape, y_shape))
    J = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    J = J.reshape(J.shape[0], J.shape[1], 1)
    rgb_val[i] = I; gray_val[i] = J

gen_image_val = gen.predict(gray_val, batch_size=8)

for j in range(val_samples):
    cv2.imwrite(store+'/'+str(j)+'.jpg', gen_image_val[j])