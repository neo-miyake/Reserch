import os
import cv2
import glob
from tqdm import tqdm
from natsort import natsorted

ver = "ver1"
MODEL_PATH = f"model_AutoEncoder/{ver}/"
os.makedirs(MODEL_PATH,exist_ok=True)


path, dirs, files = next(os.walk(MODEL_PATH+"result/"))
dir_num = len(dirs)+1
os.makedirs(MODEL_PATH+"result/"+str(dir_num), exist_ok=True)

files = natsorted(glob.glob("learning_picture/AE_cle_train_v13/*.jpg"))#
for i, file in tqdm(enumerate(files)):
    img = cv2.imread(file, 0)
    threshold = 10

    ret, img_thresh = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)

    # 二値化画像の表示
    name = os.path.splitext(os.path.basename(file))[0]
    cv2.imwrite('{0}/{1}.jpg'.format(MODEL_PATH+"result/train_1",str(name)),img_thresh)#
