import cv2
import math
import os
import glob
import h5py
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from natsort import natsorted
from tensorflow import keras
from keras import layers
from tensorflow import Tensor
from tensorflow.python import traceback
from keras.models import Model
from PIL import Image


def make_traindata(files,save_files):
    inputs  = np.zeros((len(files), image_size, image_size,3), np.uint8)
    outputs = np.zeros((len(files),1,2),np.float32)

    for i, file in tqdm(enumerate(files)):
        image = Image.open(file)
        image = image.resize((image_size, image_size))
        data_in = np.asarray(image, np.uint8)
        inputs[i] = data_in
        
        theta = float(os.path.splitext(os.path.basename(file))[0].split(',')[1])/(2*math.pi)
        phi   = float(os.path.splitext(os.path.basename(file))[0].split(',')[2])/(2*math.pi)
        
        q = np.asarray([theta,phi],np.float32)
        outputs[i] = q

    with h5py.File(save_files,"a") as h5:
        h5.create_dataset('inputs', data = inputs)
        h5.create_dataset('outputs',data = outputs)

def read_dataset(h5_path):
    data = []
      
    with h5py.File(h5_path, "r") as f:
        inputs  = f['inputs'][()]
        outputs = f['outputs'][()]
    
    
    inputs = inputs.reshape(-1,image_size,image_size,3)
    data.append(tf.multiply(inputs , 1))
    
    outputs = outputs.reshape(-1,2)
    data.append(tf.multiply(outputs, 1))

    return data

def plot_graph(loss_list,val_loss_list):
    loss     = loss_list
    val_loss = val_loss_list

    fig = plt.figure()
    nb_epoch = len(loss)
    plt.plot(range(nb_epoch), loss, marker='None', label='loss')
    plt.plot(range(nb_epoch), val_loss, marker='None', label='val_loss')
    plt.legend(loc='best', fontsize=10)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    fig.set_figheight(2)
    fig.set_figwidth(8)
    plt.show()


ver = "ver1"
MODEL_PATH = f"Image_Generate/{ver}/"
os.makedirs(MODEL_PATH,exist_ok=True)

image_size = 256
dim=16

files = natsorted(glob.glob("learning_picture/cle_train_1/*.jpg"))
save_files = MODEL_PATH+"/learning_data/train_0105.h5"
make_traindata(files,save_files)
files = natsorted(glob.glob("learning_picture/cle_val_1/*.jpg"))
save_files = MODEL_PATH+"/learning_data/val_0105.h5"
make_traindata(files,save_files)


# input quaternion
input_ = layers.Input(shape=(2,))

# 低解像度の画像を生成
x = layers.Dense(128, activation = 'tanh')(input_)
x = layers.Dense(512, activation = 'tanh')(x)
x = layers.Dense(2048,activation = 'tanh')(x)
x_1 = layers.Reshape((16,16,8))(x)

"================================================================="
x = layers.Conv2D(dim*16,kernel_size=(3, 3), padding='same')(x_1)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
x = layers.Conv2D(dim*16,kernel_size=(3, 3), padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
x = layers.Conv2D(dim*16,kernel_size=(3, 3), padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
x = layers.Conv2D(dim*16,kernel_size=(3, 3), padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)

x = layers.UpSampling2D(size=(2,2))(x)
x_2 = layers.UpSampling2D(size=(2,2))(x_1)
x = layers.Concatenate(axis=3)([x, x_2])
"================================================================="


"================================================================="
x = layers.Conv2D(dim*8,kernel_size=(3, 3), padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
x = layers.Conv2D(dim*8,kernel_size=(3, 3), padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
x = layers.Conv2D(dim*8,kernel_size=(3, 3), padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
x = layers.Conv2D(dim*8,kernel_size=(3, 3), padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)

x = layers.UpSampling2D(size=(2,2))(x)
x_3 = layers.UpSampling2D(size=(4,4))(x_1)
x = layers.Concatenate(axis=3)([x, x_3])
"================================================================="


"================================================================="
x = layers.Conv2D(dim*4,kernel_size=(3, 3), padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
x = layers.Conv2D(dim*4,kernel_size=(3, 3), padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
x = layers.Conv2D(dim*4,kernel_size=(3, 3), padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
x = layers.Conv2D(dim*4,kernel_size=(3, 3), padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)

x = layers.UpSampling2D(size=(2,2))(x)
x_4 = layers.UpSampling2D(size=(8,8))(x_1)
x = layers.Concatenate(axis=3)([x, x_4])
"================================================================="


"================================================================="
x = layers.Conv2D(dim*2,kernel_size=(3, 3), padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
x = layers.Conv2D(dim*2,kernel_size=(3, 3), padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
x = layers.Conv2D(dim*2,kernel_size=(3, 3), padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
x = layers.Conv2D(dim*2,kernel_size=(3, 3), padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)

x = layers.UpSampling2D(size=(2,2))(x)
x_5 = layers.UpSampling2D(size=(16,16))(x_1)
x = layers.Concatenate(axis=3)([x, x_5])
"================================================================="


"================================================================="
x = layers.Conv2D(dim*1,kernel_size=(3, 3), padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
x = layers.Conv2D(dim*1,kernel_size=(3, 3), padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
x = layers.Conv2D(dim*1,kernel_size=(3, 3), padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
x = layers.Conv2D(dim*1,kernel_size=(3, 3), padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)

x = layers.Conv2D(3,kernel_size=(3, 3), padding='same')(x)
x = layers.BatchNormalization()(x)
output_ = layers.Activation('sigmoid')(x)
"================================================================="

model = Model(input_,output_)
model.compile(optimizer='adam',loss='mean_squared_error')
model.summary()

# #NNの可視化
# # from tensorflow.keras.utils import plot_model
# # num=len(glob.glob(MODEL_PATH+"/*.png"))
# # plot_model(model, show_shapes=True, show_layer_names=True, to_file=os.path.join(MODEL_PATH,"model_cvnn_ver1-"+str(num+1)+".png"))
# # tf.keras.utils.plot_model(model, to_file=os.path.join(MODEL_PATH,"model_cvnn_ver1-"+str(num+1)+".png"), show_shapes=True)

reducelr=keras.callbacks.ReduceLROnPlateau(monitor='loss',factor=0.5,patience=5,min_lr=0.000000001)
early_stopping=keras.callbacks.EarlyStopping(monitor='val_loss',patience=10,min_delta=0.0)

train = read_dataset(MODEL_PATH+"learning_data/train_0105.h5")
test  = read_dataset(MODEL_PATH+"learning_data/val_0105.h5")

history = model.fit(train[1], train[0]/255, batch_size = 64, epochs=500,
          callbacks=[reducelr,early_stopping],
          validation_data=(test[1], test[0]/255))

loss     = history.history['loss']
val_loss = history.history['val_loss']
plot_graph(loss, val_loss)

path, dirs, files = next(os.walk(MODEL_PATH+"result/"))
dir_num = len(dirs)+1
os.makedirs(MODEL_PATH+"result/"+str(dir_num), exist_ok=True)
files = natsorted(glob.glob("learning_picture/cle_valid_1/*.jpg"))
print(files)

# model = keras.models.load_model(MODEL_PATH+"model/Image_Generate_ver1_1.h5")
# model.summary()
pre = model.predict(test[1])
for i, file in tqdm(enumerate(files)):
    im1 = Image.open(file).convert('RGB')
    im1 = im1.resize((image_size, image_size))
    im1 = np.asarray(im1, np.uint8)
    im2 = Image.fromarray((pre[i]*255).reshape(image_size,image_size,3).astype(np.uint8)).convert('RGB')

    plt.figure()
    plt.subplots_adjust(wspace=0.4)
    plt.subplot(1,2,1)
    plt.title("true",fontsize=18)
    plt.imshow(im1)
    plt.subplot(1,2,2)
    plt.title("pred",fontsize=18)
    plt.imshow(im2)
    
    plt.savefig('{0}/{1}.jpg'.format(MODEL_PATH+"result/"+str(dir_num),i))
    plt.close()
    
num = len(glob.glob(MODEL_PATH+"/AutoEncoder/Image_Generate_ver1_"+"*"+".h5"))
model.save(MODEL_PATH+"/model/Image_Generate_ver1_"+str(num+1)+".h5")