from tensorflow import keras
from keras.models import Model
from keras import layers
import tensorflow as tf
import numpy as np
import os
import h5py
import matplotlib.pyplot as plt
import keras.backend as K
import time
from tqdm import tqdm
import random
from PIL import Image


def read_dataset(h5_path):
    data = []
      
    with h5py.File(h5_path, "r") as f:
        inputs  = f['inputs'][()]
        outputs = f['outputs'][()]
    
    
    inputs = inputs.reshape(-1,image_size,image_size,3)
    data.append(tf.multiply(inputs , 1))
    
    outputs = outputs.reshape(-1,1,2)
    data.append(tf.multiply(outputs, 1))

    return data

def read_dataset2(h5_path):
    data = []
      
    with h5py.File(h5_path, "r") as f:
        outputs = f['outputs'][()]

    outputs = outputs.reshape(-1,1,2)
    data.append(tf.multiply(outputs, 1))

    return data


ver = "ver1"
MODEL_PATH = f"Image_Generate/{ver}/"
os.makedirs(MODEL_PATH,exist_ok=True)
image_size = 256

# save_files = MODEL_PATH+"/learning_data/1210_test.h5"
# maketraindata(save_files)

model = keras.models.load_model(MODEL_PATH+"model/Image_Generate_ver1_1.h5")
model.summary()

train = read_dataset(MODEL_PATH+"learning_data/train_1212.h5")
test  = read_dataset(MODEL_PATH+"learning_data/val_1212.h5")
test1121=read_dataset2(MODEL_PATH+"learning_data/1210_test.h5")
np.set_printoptions(threshold=np.inf)



opt = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.99, epsilon=None, decay=0.0, amsgrad=False)
# pi_2 = tf.constant(1)
# pi_1 = tf.constant(1)

# for j in range(1):
# default = tf.Variable([[0.9982813 , 0.46913874]])
default= tf.Variable(train[1][38])
y_true = train[0][38]/255
x_true = train[1][38]
print("pre\n",default,"\ntrue\n",x_true)
images = []
for i in range(1):
# LOSS=1
# while LOSS > 0.008:
    with tf.GradientTape() as tape:
        tape.watch(default)
        y_pred = model(default)
        loss = tf.keras.losses.mean_squared_error(y_true,y_pred)
        LOSS = np.sum(loss.numpy())/65536
        print("LOSS",LOSS)
        grad = tape.gradient(loss,default)
        print("grad",grad)
        opt.apply_gradients([(grad,default)])
        
        # im = Image.fromarray((y_pred.numpy()*255).reshape(image_size,image_size,3).astype(np.uint8)).convert('RGB')
        # images.append(im)
        
        # if default[0][0]<0:
        #     default.assign([[tf.add(default[0][0],tf.constant(1.0)),default[0][1]]])
        # if default[0][1]<0:
        #     default.assign([[default[0][0], tf.add(default[0][1],tf.constant(1.0))]])
        
        # if default[0][0]>1:
        #     default.assign([[tf.math.mod(default[0][0],tf.constant(1.0)),default[0][1]]])
        # if default[0][1]>1:
        #     default.assign([[default[0][0], tf.math.mod(default[0][1],tf.constant(1.0))]])
       
        # print(i,"\ntrue\n",x_true,"\npre\n",default,"\ngrad\n",grad)

# im.save('out.gif', save_all=True, append_images=images)
print("true\n",x_true,"\npre\n",default,"\n")