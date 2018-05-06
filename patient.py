import dictionary
import SimpleITK as sitk
import numpy as np



dict = dictionary.get()
#print(dict[5])
"""PRIMER UPORABE = dict[id]
dict[5] je tole:
['C:/BRATS/BRATS2015_Training/HGG\\brats_2013_pat0005_1\\VSD.Brain.XX.O.MR_Flair.54536/VSD.Brain.XX.O.MR_Flair.54536.mha',
'C:/BRATS/BRATS2015_Training/HGG\\brats_2013_pat0005_1\\VSD.Brain.XX.O.MR_T1.54537/VSD.Brain.XX.O.MR_T1.54537.mha',
'C:/BRATS/BRATS2015_Training/HGG\\brats_2013_pat0005_1\\VSD.Brain.XX.O.MR_T1c.54538/VSD.Brain.XX.O.MR_T1c.54538.mha',
'C:/BRATS/BRATS2015_Training/HGG\\brats_2013_pat0005_1\\VSD.Brain.XX.O.MR_T2.54539/VSD.Brain.XX.O.MR_T2.54539.mha',
'C:/BRATS/BRATS2015_Training/HGG\\brats_2013_pat0005_1\\VSD.Brain_3more.XX.O.OT.54541/VSD.Brain_3more.XX.O.OT.54541.mha']
"""

#slike so ble croppane: 240 240 155 --> 160  144  128


class Patient(object):


    def __init__(self, patient_id):
        #self.flair = self.getFlair()
        #self.t1 = self.getT1()
        self.id = patient_id
        self.dirs = dict[patient_id]
        self.input_arrays = self.getArrays()
        """self.flair_dir = self.dirs[0]
        self.t1_dir = self.dirs[1]
        self.t1c_dir = self.dirs[2]
        self."""


    def getArrays(self):
        images = []
        for idx, i in enumerate(dict[self.id]):
            images.append(sitk.GetArrayFromImage(sitk.ReadImage(dict[self.id][idx]))[77-64:77+64,128-80:128+80,120-72:120+72])
        return np.array(images)[:4]
    def getLabels(self):
        # TODO: make each class own array with onehot encoding
        return 0

    def getImages(self):
        return self.dirs[1]
    def getT1cDir(self):
        return self.dirs[2]
    def getT2Dir(self):
        return self.dirs[3]
    def getTruthDir(self):
        return self.dirs[4]
