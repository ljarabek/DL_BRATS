import os
from configparser import ConfigParser
from pprint import pprint
import SimpleITK as sitk
import matplotlib.pyplot as plt
import re
import numpy as np
config = ConfigParser()
config.read("config.ini")
default_config = config['DEFAULT']
train_path = default_config['train_path']
arr = []
ids = []
for root, dirs, files in os. walk(train_path):
    for file in [f for f in files if f.endswith(".mha")]:
        s = str(root + '/' + file)
        arr.append(s)
        pat_id = re.sub(r'(pat)', '', re.findall(r'(pat[0-9]+)', s)[0])
        ids.append(int(pat_id))
dictionary = {}
for idx, id in enumerate(arr, 0):
    if idx%5==0:
        dictionary.update({ids[idx] : arr[idx:idx+5]})
def get():
    return dictionary


"""
k = sitk.GetArrayFromImage(sitk.ReadImage(arr[52])) #tole dela
print(k[70].shape)
k=k[70]
plt.imshow(k)
plt.show()"""
