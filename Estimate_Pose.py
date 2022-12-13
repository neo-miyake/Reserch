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



class Adam:

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None
        
    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            # for key in range(4):
            #     print(key)
            self.m = np.zeros([1,4])
            self.v = np.zeros([1,4])
        
        self.iter += 1
        lr_t  = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)         
        
        for key in range(2):
            #self.m[key] = self.beta1*self.m[key] + (1-self.beta1)*grads[key]
            #self.v[key] = self.beta2*self.v[key] + (1-self.beta2)*(grads[key]**2)
            self.m[0][key] += (1 - self.beta1) * (grads[0][key]    - self.m[0][key])
            self.v[0][key] += (1 - self.beta2) * (grads[0][key]**2 - self.v[0][key])
         
            params[0][key] = lr_t * self.m[0][key] / (np.sqrt(self.v[0][key]) + 1e-7)
        


ver = "ver1"
MODEL_PATH = f"model_AutoEncoder/{ver}/"
os.makedirs(MODEL_PATH,exist_ok=True)
image_size = 256

# save_files = MODEL_PATH+"/learning_data/1210_test.h5"
# maketraindata(save_files)

model = keras.models.load_model(MODEL_PATH+"/AutoEncoder/Image_Generate_ver1_17.h5")
model.summary()

train = read_dataset(MODEL_PATH+"learning_data/train_1212.h5")
test  = read_dataset(MODEL_PATH+"learning_data/val_1212.h5")
test1121=read_dataset2(MODEL_PATH+"learning_data/1210_test.h5")
np.set_printoptions(threshold=np.inf)



default = tf.Variable(test1121[0][0])
# default = tf.Variable(test[1][7])
y_true = test[0][750]/255
x_true = test[1][750]
print("pre\n",default,"\ntrue\n",x_true)
# opt = keras.optimizers.SGD(lr=0.0001, momentum=0.99, decay=0.0, nesterov=False)
opt =keras.optimizers.Adam(lr=0.5, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
# optimizer = Adam(lr=0.5, beta1=0.9, beta2=0.99)

    
for i in range(10):
    with tf.GradientTape() as tape:
        tape.watch(default)
        y_pred = model(default)
        loss = tf.keras.losses.mean_squared_error(y_true,y_pred)
        grad = tape.gradient(loss,default)
        opt.apply_gradients([(grad,default)])
        print("grad\n",grad,"\npre\n",default)
     
print("\ntrue\n",x_true,"\npre\n",default)