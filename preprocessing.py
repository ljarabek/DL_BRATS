import dictionary
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import random as rand
import os
from presentation import multi_slice_viewer
from configparser import ConfigParser

cp = ConfigParser()
cp.read("config.ini")
cfg_crop = cp['CROP']
for v in cfg_crop:
    print(v)
    exec(v + '= int(cfg_crop["'+v+'"])')
print(x_0, x_r, y_0, y_r, z_0, z_r) # magic - don't delete
print(int(cfg_crop['x_0']))
if not os.path.exists('./preprocess/'):
    os.makedirs('./preprocess/')




#dictionary of patients (keys: patient ids, values: image paths)
#training dictionary
dct = dictionary.get() #length = 254

#validation dictionary
dctVal = dictionary.getVal()    # len = val_len = 20

#test dictionary
dctTest = dictionary.getTest() #length = 110



def getMaskedArray(i, mod):
    '''returns cropped array, where values 0 are masked'''
    arr = sitk.GetArrayFromImage(sitk.ReadImage(dct[i][mod]))[77 - 64:77 + 64, 128 - 80:128 + 80, 120 - 72:120 + 72]
    return ma.masked_values(arr,0)

def getMaskedArrayVal(i, mod):
    '''returns cropped array, where values 0 are masked'''
    arr = sitk.GetArrayFromImage(sitk.ReadImage(dctVal[i][mod]))[77 - 64:77 + 64, 128 - 80:128 + 80, 120 - 72:120 + 72]
    return ma.masked_values(arr,0)

def getMaskedArrayTest(i, mod):
    '''returns cropped array, where values 0 are masked'''
    arr = sitk.GetArrayFromImage(sitk.ReadImage(dctTest[i][mod]))[77 - 64:77 + 64, 128 - 80:128 + 80, 120 - 72:120 + 72]
    return ma.masked_values(arr,0)

#calculating the average value and standard deviation
def getAvg(modality = 0):
    try:
        avg = np.load('./preprocess/avg{}.npy'.format(modality))
        return avg
    except IOError:
        print('AVG for modality {} not found, generating average'.format(modality))
    n = np.uint64(0)
    sum = np.uint(0)
    for d in dct:
        buffer = getMaskedArray(d, modality)
        sum += buffer.sum()
        n += ma.count(buffer)  # 326.62273804171326
        #n += buffer.size - ma.count_masked(buffer)  #326.62273804171326
    avg = np.float32(sum/n)
    np.save('./preprocess/avg{}.npy'.format(modality), avg)
    return(avg)

def getStd(modality = 0):
    try:
        avg = np.load('./preprocess/std{}.npy'.format(modality))
        return avg
    except IOError:
        print('STD for modality {} not found, generating std'.format(modality))
    n = np.uint64(0)
    sum = np.uint(0)
    avg = getAvg(modality=modality)
    for d in dct:
        buffer = getMaskedArray(d, modality)
        sum += np.square(buffer - avg).sum()
        n += ma.count(buffer)  # 326.62273804171326
        # n += buffer.size - ma.count_masked(buffer)  #326.62273804171326
    std = np.float32(np.sqrt(np.divide(sum, n)))
    np.save('./preprocess/std{}.npy'.format(modality), std)
    return(std)

"""print('flair')
print(getAvg(0))
print(getStd(0))
print('t1')
print(getAvg(1))
print(getStd(1))
print('t1c')
print(getAvg(2))
print(getStd(2))
print('t2')
print(getAvg(3))
print(getStd(3))"""


def standardize(arr, mod):
    return (arr-getAvg(mod))/(getStd(mod) + 0.000000001)


def getBatchTraining():
    """OUTPUTA MODEL-READY VERZIJO"""
    key = rand.choice(list(dct))
    arr = []
    answers = outputToChannels(key)
    for i in range(4):
        arr.append(standardize(getMaskedArray(key, mod =i),
                               i))

    data = np.ma.masked_array(arr).filled(0)

    return np.expand_dims(data,0), np.expand_dims(answers,0)

def getVal(returnID=False): # vrne cel set slik za validacijo
    #key = rand.choice(list(dctVal))
    arr = []
    output
    for key in list(dctVal):
        for i in range(4):
            arr.append(standardize(getMaskedArrayVal(key, mod=i),
                                   i))

        data = np.ma.masked_array(arr).filled(0)
        if returnID:
            return np.expand_dims(data, 0), key
        else:
            return np.expand_dims(data, 0)

def getBatchTest(returnID=False):
    key = rand.choice(list(dctTest))
    arr = []
    for i in range(4):
        arr.append(standardize(getMaskedArrayTest(key, mod=i),
                               i))

    data = np.ma.masked_array(arr).filled(0)
    if returnID:
        return np.expand_dims(data, 0), key
    else:
        return np.expand_dims(data, 0)



#multi_slice_viewer(getBatchTest()[0,0])
#print(getN(2))
#print(getN(1))

""""""

def outputToChannels(id):
    arr = np.array(sitk.GetArrayFromImage(sitk.ReadImage(dct[id][4])))[77 - 64:77 + 64, 128 - 80:128 + 80, 120 - 72:120 + 72]

    type0 = np.zeros(arr.shape)
    type0[arr==0] = 1

    type1 = np.zeros(arr.shape)
    type1[arr == 1] = 1

    type2 = np.zeros(arr.shape)
    type2[arr==2]=1

    type3 = np.zeros(arr.shape)
    type3[arr == 3] = 1

    type4 = np.zeros(arr.shape)
    type4[arr == 4]=1

    return np.stack((type0,type1,type2,type3,type4))

def channelsToOutput(image, ID=0):  ## TODO : dub ven ID pr getbatchTest!!
    if image.shape[0]==1: #da ignorira batch=1
        image=image[0]
    return np.argmax(image, axis=0), ID

def resizeOutput(out):
    zeros = np.zeros(shape=[155,240,240])
    for x, n in enumerate(out):
        for y, m in enumerate(n):
            for z, o in enumerate(m):
                zeros[x+13, y+48,z+48]=o
    return zeros

def saveOutput(otpt, filename):
    #out, _ = channelsToOutput(otpt, None)
    out = sitk.GetImageFromArray(np.array(otpt, dtype=float))
    sitk.WriteImage(out, filename)




def save_numpy(picture, batch, dir='C:/activations/', filename = "/graph.png"):
    if not os.path.exists(dir):
        os.makedirs(dir + '{}'.format(batch))
    #dct = dictionary.get()
    savedir = dir + '{}/'.format(batch)
    fig = plt.figure()
    iter = int(len(picture) /30)
    for num,slice in enumerate(picture):
        if num>=30:
            break
        y = fig.add_subplot(5,6,num+1)
        y.imshow(picture[num*iter], cmap='gray')
    plt.savefig(dir + str(batch)+filename, dpi=500, format = "png")
    plt.close('all')
    #plt.show()
    return



"""display_numpy(outputToChannels(10)[0])
display_numpy(outputToChannels(10)[1])
display_numpy(outputToChannels(10)[2])
display_numpy(outputToChannels(10)[3])"""


"""def getBatch(size = 15, min= 0, max = 146):              BATCH SIZE je 1!!!
    arr = []
    for i in range(size):
        int = np.random.randint(min, max, None)
        for m in range(4):
            arr.append(standardize)
getMaskedArray(1, 0)"""

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
#print(he.dirs)

def outputToChannels(id):
    #  one way of doing it:
    arr = np.array(sitk.GetArrayFromImage(sitk.ReadImage(dct[id][4])))[77 - 64:77 + 64, 128 - 80:128 + 80, 120 - 72:120 + 72]
    zeros = np.expand_dims(np.zeros(arr.size),0)  # or shape...
    buffer = np.zeros(arr.size)
    buffer = np.append([buffer, buffer, buffer, buffer], zeros, 0)
    flat = arr.flatten()
    buffer[flat, np.arange(arr.size)] = 1
    [x,y,z] = arr.shape
    return np.reshape(buffer, newshape=(5,x,y,z)) 

"""
