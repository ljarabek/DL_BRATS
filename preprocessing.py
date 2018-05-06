import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import dictionary
import patient
from sklearn import preprocessing
import numpy.ma as ma



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

for i in list(range(1,11)):
    tmpDict[i] = dict[i]

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

#izračunajmo povprečne vrednosti

flairMeans = [] #seznam povprečnih vrednosti vseh matrik
t1Means =[]
t1cMeans = []
t2Means = []

for i in range(len(patients)):
    patients[i].setFlair(3)


