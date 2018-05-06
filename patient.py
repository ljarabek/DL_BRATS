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
        self.flair = flair
        self.t1 = t1
        self.t1c = t1c
        self.t2 = t2


    def getArray(self):
        "Vrne matriko inputa "
        return [self.flair, self.t1, self.t1c,self.t2]


    def getLabels(self):
        # TODO: make each class own array with onehot encoding
        return 0

