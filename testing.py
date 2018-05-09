import dictionary
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
from tqdm import tqdm
from patient import Patient



dct = dictionary.get() #vse je dolgo 188
#sitk.Show(sitk.ReadImage(dct[5][2]))


def getMaskedArray(i, mod):
    '''returns un-cropped array'''
    return ma.masked_values(sitk.GetArrayFromImage(sitk.ReadImage(dct[i][mod])),0)
###     STEVILO VREDNOSTI

def getN(modality=0, _maxid = 1e9): #267839313
    '''Returns avg and SD
    modality: 0-flair, 1-T1, 2-T1c, 3-T2'''
    n = np.uint64(0)
    sum = np.uint64(0)
    sumOfSquares = np.uint64(0)
    sums =[]
    if (modality>=4):
        print('Cant preprocess output images')
        return 0
    for d in tqdm(dct):
        buffer = getMaskedArray(d, modality)
        """if d%10==0:
            ha = [d for d in tqdm(buffer.flatten()) if d>0]
            plt.hist(ha)
            plt.show()"""
        sum += np.sum(buffer)
        sums.append(np.sum(buffer))
        sumOfSquares += np.sum(np.square(buffer))
        n += buffer.count()
    average = np.divide(sum, n)
    averageSquared = np.square(average)
    averageOfSquares = np.divide(sumOfSquares,n)
    variance = averageSquared - averageOfSquares
    SD = np.sqrt(variance)

    return average, SD  # (325.77115552114634, 325.515874279643)

def outputToChannels(id):
    """ ta koda je sexy """
    arr = np.array(sitk.GetArrayFromImage(sitk.ReadImage(dct[id][4])))
    zeros = np.expand_dims(np.zeros(arr.size),0)  # or shape...
    buffer = np.zeros(arr.size)
    buffer = np.append([buffer, buffer, buffer, buffer], zeros, 0)
    flat = arr.flatten()
    buffer[flat, np.arange(arr.size)] = 1
    [x,y,z] = arr.shape
    return np.reshape(buffer, newshape=(5,x,y,z))
"""heh = outputToChannels(3)
print(outputToChannels(3))
plt.imshow(heh[4,65])
plt.show()
plt.imshow(heh[3,65])
plt.show()
plt.imshow(heh[2,65])
plt.show()
plt.imshow(heh[1,65])
plt.show()
plt.imshow(heh[0,65])
plt.show()
#def loss(a, b):"""
print(getN(0))

#print(np.array(sitk.GetArrayFromImage(sitk.ReadImage(dct[3][0]))[77-64:77+64,128-80:128+80,120-72:120+72]).shape)
#print(getN()) #(267839313.0, 87254322490.0, 44531328854.0)


"""arr = []
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

arr = []
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
