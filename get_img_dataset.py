import cv2
import os
import glob
import shutil
BASE = './dataset/test_data/'
IMGS_PATH = '../../vaipe-data/prescription/pre/'
files = os.listdir(BASE)
image_paths = os.listdir(IMGS_PATH)
print(image_paths[0])
files = [f[:-5] for f in files]
for f in files:
    for p in image_paths:
        if f == p.split('.')[0]:
            print('Coping file ....')
            shutil.copy(os.path.join(IMGS_PATH,p), os.path.join(BASE,p))
            break

    