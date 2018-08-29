import numpy as np
import matplotlib.pyplot as plt
import os
import SimpleITK as sitk
from preprocessing import getBatchVal, channelsToOutput

#displays a 3D picture
def display_numpy(picture):
    fig = plt.figure()
    iter = int(len(picture) /30)
    for num,slice in enumerate(picture):
        if num>=30:
            break
        y = fig.add_subplot(5,6,num+1)

        y.imshow(picture[num*iter], cmap='gray')
    plt.show()
    return


def saveSegmentation( arrayT1c,out, batch, dir='C:/activations_Segmentations/', filename = "graph.png"):
    if not os.path.exists(dir):
        os.makedirs(dir)
    #dct = dictionary.get()
    #savedir = dir + '{}/'.format(batch)
    fig = plt.figure()
    iter = int(len(arrayT1c) /30)
    for num,slice in enumerate(arrayT1c):
        if num>=30:
            break
        y = fig.add_subplot(5,6,num+1)
        y.imshow(out[1][num*iter],cmap="Blues",alpha=1)
        y.imshow(out[2][num*iter], cmap="Reds", alpha=0.5)
        y.imshow(out[3][num*iter], cmap="Greens", alpha=0.33333333)
        y.imshow(out[4][num*iter],cmap="Purples", alpha=0.25)
        y.imshow(arrayT1c[num * iter], cmap='gray', alpha=0.6)
    plt.savefig(dir + str(batch)+filename, dpi=500, format = "png")
    plt.close('all')
    #plt.show()
    return


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


#TEST getBatchVal():
#i == 20 mora bit isto kot i==0

#for i in range(30):
#    i_, a_ = getBatchVal()
#    print("%s : %s" %(i, i_[0, 0, 64, 80, 70]))
#    print(a_.shape)
#    if i ==5:
#        display_numpy(channelsToOutput(a_))
#        display_numpy(channelsToOutput(a_[0]))
#
def getOutput(out):
    "Funkcija prejme output oblike (1,5,128,160,144) in vrne matriko (128,160,144), ki ima na mestih kjer je zdrava 0 ostalo paštevila 1-4"
    newOut = np.zeros(shape=(128, 160, 144))
    out = out[0]
    for i in range(1,5):
        newOut[out[i] == 1]= i
    return newOut

def saveSeg( arrayT1c,out, batch, dir='C:/activations_Segmentations/', filename = "graph.png"):
    if not os.path.exists(dir):
        os.makedirs(dir)

    fig = plt.figure()
    iter = int(len(arrayT1c) /30)

    for num,slice in enumerate(arrayT1c):
        if num>=30:
            break
        y = fig.add_subplot(5,6,num+1)
        y.imshow(arrayT1c[num * iter], cmap='gray')
        y.imshow(out[num * iter], cmap="Reds", alpha=0.25)

    plt.savefig(dir + str(batch)+filename, dpi=500, format = "png")
    plt.close('all')
    return

