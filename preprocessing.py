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

#izračun povprečnih vrednosti kvadratov
#flairMeansSqr =[]
#t1MeansSqr =[]
#t1cMeansSqr =[]
#t2MeansSqr =[]

#število pacientov
n = len(patients)

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

    #dodamo povprečja matrik v sezname
    flairMeans.append(flair.mean())
    t1Means.append(t1.mean())
    t1cMeans.append(t1c.mean())
    t2Means.append(t2.mean())

#končna povprečja (E(X))
meanFlair = sum(flairMeans)/ n
meanT1 = sum(t1Means)/n
meanT1c = sum(t1cMeans)/n
meanT2 = sum(t2Means)/n


#TODO: Izračun standardnih deviacij
stdFlair = 1
stdT1 = 1
stdT1c = 1
stdT2 = 1

#Standardizacija

for patient in patients:
    patient.flair = (patient.flair - meanFlair)/stdFlair
    patient.t1 = (patient.t1 - meanT1)/stdT1
    patient.t1c = (patient.t1c - meanT1c) / stdT1c
    patient.t2 = (patient.t2 - meanT2) / stdT2

#TODO: Shraniti vse standardizirane matirke
