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

def maketraindata(save_files):
    outputs = np.zeros((1,1,2),np.float32)

    # for i, file in tqdm(enumerate(files)):

    q = np.asarray([0,0],np.float32)
    outputs[0] = q

    with h5py.File(save_files,"a") as h5:
        h5.create_dataset('outputs',data = outputs)

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

# def quatmul(q1,q2):
#     q_rel = np.array([q1[0][0]*q2[0][0]-q1[0][1]*q2[0][1]-q1[0][2]*q2[0][2]-q1[0][3]*q2[0][3],
#                       q1[0][2]*q2[0][3]-q1[0][3]*q2[0][2]+q1[0][0]*q2[0][1]+q1[0][1]*q2[0][0],
#                       q1[0][3]*q2[0][1]-q1[0][1]*q2[0][3]+q1[0][0]*q2[0][2]+q1[0][2]*q2[0][0],
#                       q1[0][1]*q2[0][2]-q1[0][2]*q2[0][1]+q1[0][0]*q2[0][3]+q1[0][3]*q2[0][0]])
    
#     return q_rel

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

grad = [[100,0]]
ite = 0
lr = 0.001
# while abs(grad[0][0]*lr)>=0.0872665 or abs(grad[0][1]*lr)>=0.0872665:
for i in range(10):
    with tf.GradientTape() as tape:
        tape.watch(default)
        y_pred = model(default)
        loss = tf.keras.losses.mean_squared_error(y_true,y_pred)
        grad = tape.gradient(loss,default)
        print(grad)
        # a = [1,-1]
        # grad=grad*a
        # optimizer.update(grad, grad)
        
        
        # default = grad.numpy()[0]*lr*default.numpy()[0]
        opt.apply_gradients([(grad,default)])
        # 
        # default = tf.multiply(default.reshape(1,4),1)
        # default = default - tf.multiply(grad,1)
        print("grad\n",grad,"\npre\n",default)
        ite += 1
     
        
# print("grad\n",grad,"\npre\n",default)
print("\ntrue\n",x_true,"\npre\n",default)
print(ite)