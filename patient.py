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


    def __init__(self, patient_id,flair, t1, t1c, t2):
        self.id = patient_id
        self.Arrays = [flair, t1, t1c, t2] #input

    def getFlair(self):
        return self.Arrays[0]

    def setFlair(self, newFlair):
        self.Arrays[0] = newFlair

    def getT1(self):
        return self.Arrays[1]

    def setT1(self, newT1):
        self.Arrays[1] = newT1

    def getT1c(self):
        return self.Arrays[2]

    def setT1c(self, newT1c):
        self.Arrays[2] = newT1c

    def getT2(self):
        return self.Arrays[3]

    def setT2(self, newT2):
        self.Arrays[3] = newT2

    def getLabels(self):
        # TODO: make each class own array with onehot encoding
        return 0

