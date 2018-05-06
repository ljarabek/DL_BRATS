import dictionary
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
from tqdm import tqdm
from patient import Patient



dct = dictionary.get()
#sitk.Show(sitk.ReadImage(dct[5][2]))
arr = []
arr10m = []
arr10s = []
for idx, d in tqdm(enumerate(dct)):

    arr.append(ma.masked_values(sitk.GetArrayFromImage(sitk.ReadImage(dct[d][0])), 0))
    if idx%10 == 0:
        arr10m.append(np.mean(arr))
        arr10s.append(np.std(arr))
        arr = []
    #print(np.array(arr).shape)
#arr = np.array(arr10)
m = np.mean(arr10m)
s = np.mean(arr10s)
print(np.array([m,s]))

#np.save('C:/BRATS/meanstd1.npy', [m,s])
print(np.load('C:/BRATS/meanstd.npy'))

"""arr = []
for d in tqdm(dct):
    arr.append(ma.masked_values(sitk.GetArrayFromImage(sitk.ReadImage(dct[d][2])), 0))
    #print(np.array(arr).shape)
arr = np.array(arr)
m = np.mean(arr)
s = np.std(arr)
np.save('C:/BRATS/meanstd2.npy', [m,s])
print('saved2')
arr = []
for d in tqdm(dct):
    arr.append(ma.masked_values(sitk.GetArrayFromImage(sitk.ReadImage(dct[d][3])), 0))
    #print(np.array(arr).shape)
arr = np.array(arr)
m = np.mean(arr)
s = np.std(arr)
np.save('C:/BRATS/meanstd3.npy', [m,s])
print('saved3')
print(np.load('C:/BRATS/meanstd.npy'))
print(np.load('C:/BRATS/meanstd1.npy'))
print(np.load('C:/BRATS/meanstd2.npy'))
print(np.load('C:/BRATS/meanstd3.npy'))"""


"""arr = sitk.GetArrayFromImage(sitk.ReadImage(dct[10][2]))
(z,y,x) = arr.shape  # 155 240 240  --> 128 144 160  t(160  144  128)

arr_cropped = arr[77-64:77+64,128-80:128+80,120-72:120+72]
print(np.mean(arr_cropped))
arr_cropped = ma.masked_values(arr_cropped, 0)
print(np.mean(arr_cropped))
arr_cropped = (arr_cropped - np.mean(arr_cropped))/np.std(arr_cropped)

print(Patient(5).input_arrays.shape)
plt.imshow(Patient(5).input_arrays[3,64])
plt.show()


#he = Patient(25)
#print(he.dirs)"""
