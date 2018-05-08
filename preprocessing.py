import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import dictionary
import patient
from sklearn import preprocessing
import numpy.ma as ma

#število pacientov
n = 50


dict = dictionary.get() #a dictionary of patients, every patient has 5 values (images)

print(dict[5])
"""PRIMER UPORABE = dict[id]
dict[5] je tole:
['C:/BRATS/BRATS2015_Training/HGG\\brats_2013_pat0005_1\\VSD.Brain.XX.O.MR_Flair.54536/VSD.Brain.XX.O.MR_Flair.54536.mha',
'C:/BRATS/BRATS2015_Training/HGG\\brats_2013_pat0005_1\\VSD.Brain.XX.O.MR_T1.54537/VSD.Brain.XX.O.MR_T1.54537.mha',
'C:/BRATS/BRATS2015_Training/HGG\\brats_2013_pat0005_1\\VSD.Brain.XX.O.MR_T1c.54538/VSD.Brain.XX.O.MR_T1c.54538.mha',
'C:/BRATS/BRATS2015_Training/HGG\\brats_2013_pat0005_1\\VSD.Brain.XX.O.MR_T2.54539/VSD.Brain.XX.O.MR_T2.54539.mha',
'C:/BRATS/BRATS2015_Training/HGG\\brats_2013_pat0005_1\\VSD.Brain_3more.XX.O.OT.54541/VSD.Brain_3more.XX.O.OT.54541.mha']
"""

#temporary dicitonary(za testiranje)
tmpDict = {}

for i, dict_id in enumerate(dict):
    tmpDict[i] = dict[dict_id]
    if (i+2>n):
        break

print(len(tmpDict))
patients = []

#funkcija, ki bo naredila seznam pacientov, kjer ima vsak pacient atributa ID in Arrays(input slike)
#slike so tu že predstavljene kot numpy matrike imajo že masked values 0

for id, images in tmpDict.items():
    flair = sitk.GetArrayFromImage(sitk.ReadImage(images[0])[77-64:77+64,128-80:128+80,120-72:120+72])
    t1 = sitk.GetArrayFromImage(sitk.ReadImage(images[1])[77-64:77+64,128-80:128+80,120-72:120+72])
    t1c = sitk.GetArrayFromImage(sitk.ReadImage(images[2])[77-64:77+64,128-80:128+80,120-72:120+72])
    t2 = sitk.GetArrayFromImage(sitk.ReadImage(images[3])[77-64:77+64,128-80:128+80,120-72:120+72])

    pat = patient.Patient(id,flair,t1,t1c,t2)
    patients.append(pat)

#Standardizacija

#število pacientov
size = patients[0].flair.size #število elemntov v matriki

sumFlair = 0 #vsota elemntov
sumT1 = 0
sumT1c = 0
sumT2 = 0

NFlair= 0 #število nemaskiranih elemntov
NT1 = 0
NT1c = 0
NT2 = 0


for i in range(n):
    #vsem matrikam pacienta damo masked_values 0
    patients[i].flair = ma.masked_values(patients[i].flair, 0)
    patients[i].t1= ma.masked_values(patients[i].t1, 0)
    patients[i].t1c = ma.masked_values(patients[i].t1c, 0)
    patients[i].t2 = ma.masked_values(patients[i].t2, 0)

    flair = patients[i].flair
    t1 = patients[i].t1
    t1c = patients[i].t1c
    t2 = patients[i].t2

    sumFlair += np.int64(flair.sum())
    NFlair += size - ma.count_masked(flair)

    sumT1 += np.int64(t1.sum())
    NT1 = size - ma.count_masked(t1)

    sumT1c += np.int64(t1c.sum())
    NT1c += size - ma.count_masked(t1c)

    sumT2 += np.int64(t2.sum())
    NT2 += size - ma.count_masked(t2)

#izračun povprečnih vrednosti
meanFlair = sumFlair / NFlair
meanT1 = sumT1 / NT1
meanT1c = sumT1c / NT1c
meanT2 = sumT2 / NT2


sumVarFlair = 0 # vsota (x_i - mean)**2
sumVarT1 = 0
sumVarT1c = 0
sumVarT2 = 0

#izračun vsote za varianco
for i in range(n):
    flair = patients[i].flair
    t1 = patients[i].t1
    t1c = patients[i].t1c
    t2 = patients[i].t2

    sumVarFlair += ( ( flair - meanFlair ) ** 2).sum()
    sumVarT1 += ( (t1 - meanT1) ** 2).sum()
    sumVarT1c += ( (t1c - meanT1c) ** 2).sum()
    sumVarT2 += ( (t2 - meanT2) ** 2).sum()

#izračun standardnega odklona....std(X) = ( VarX )**0.5

stdFlair = (sumVarFlair / NFlair )**0.5
stdT1 = ( sumVarT1 / NT1) ** 0.5
stdT1c = ( sumVarT1c / NT1c) ** 0.5
stdT2 = (sumVarT2 / NT2) ** 0.5














