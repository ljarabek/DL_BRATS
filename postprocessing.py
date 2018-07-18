import numpy as np
import matplotlib.pyplot as plt
import os

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

