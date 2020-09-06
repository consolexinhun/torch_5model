import pandas as pd
import numpy as np
import os
import shutil


data = pd.read_csv('car/train.csv')

col = data.iloc[:, 1] #第一列是文件名，第二列是标签

arrs = col.values

file_dir = 'car/train/'

file_list = sorted(os.listdir(file_dir), key=lambda x:int(x[:-4]))

if not os.path.exists("./data/0"):
    os.makedirs("./data/0")
if not os.path.exists("./data/1"):
    os.makedirs("./data/1")
if not os.path.exists("./data/2"):
    os.makedirs("./data/2")

for image in file_list:
    name = int(image.split('.')[0])
    if arrs[name] == 0:
        shutil.move(file_dir + image, './data/0')
    elif arrs[name] == 1:
        shutil.move(file_dir + image, './data/1')
    elif arrs[name] == 2:
        shutil.move(file_dir + image, './data/2')


