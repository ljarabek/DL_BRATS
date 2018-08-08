import os
import re
from glob import glob
from configparser import ConfigParser

cp = ConfigParser()
cp.read("config.ini")
cfg = cp['DEFAULT']
val_size = int(cfg['val_size'])
for d in cfg:
    #print(cfg[d])
    exec("%s = %s"%(d, cfg[d]))
#print(type(deal))
#val_size = 20   # koliko slik za validacijo

def ending(fileList, end = ".mha"):
    "Funkcija  prejme seznam datotek in vrne seznam datotek, ki se koncajo na dano končnico end"
    newList = []
    for file in fileList:
        if file.endswith(end):
            newList.append(file)

    return newList


#ta funkcija je nepotrebna, zato ker se taki ID-ji ponavljajo
#zato slovar ne shrani vseh pacientov
def getID(file):
    "Vrne ID stevilko pacienta iz imena njegove datoteke"
    id = re.search("pat0*([0-9]*)_.*", file).group(1) #string potrebno se pretvoriti v integer
    return id

#naredimo slovar za trening slike
##training dictionary

train_path1 = 'C:\\BRATS\\BRATS2015_Training\\HGG'
train_path2 = 'C:\\BRATS\\BRATS2015_Training\\LGG'

#Nareidmo slovar, kjer so ključi zaporedna stevilka pacienta vrednosti pa absoltune poti datotek Flair,T1,T1c,T2, answer
trainingDictionary ={}
validationDictionary = {}
id = 1



#najprej poberemo slike iz HGG
#os.chdir(train_path1)

for file in os.listdir(train_path1):
    absolutPath = train_path1 + "\\" + file

    #seznam vseh končnih map
    endFiles = glob(absolutPath + "\\*\\*")
    #izberemo samo datoteke, ki  se končajo na .mha
    endFiles = ending(endFiles)
    #shranimo v slovar
    if id>val_size:    # prvih val_size slik gre v validationDict, ostale pa v trainingDict
        trainingDictionary[id] = endFiles
    else:
        validationDictionary[id] = endFiles
    id += 1


# nato poberemo slike iz LGG
#os.chdir(train_path2)

for file in os.listdir(train_path2):
    absolutPath = train_path2 + "\\" + file
    # seznam vseh končnih map
    endFiles = glob(absolutPath + "\\*\\*")
    # izberemo samo datoteke, ki  se končajo na .mha
    endFiles = ending(endFiles)
    # shranimo v slovar
    trainingDictionary[id] = endFiles
    id += 1

#naredimo še slovar s test slikami
test_path = "C:\BRATS\Testing\HGG_LGG"
#os.chdir(test_path)


testDictionary = {}

for file in os.listdir(test_path):
    absolutPath = test_path + "\\" + file
    # seznam vseh končnih map
    endFiles = glob(absolutPath + "\\*\\*")
    # izberemo samo datoteke, ki  se končajo na .mha
    endFiles = ending(endFiles)
    # shranimo v slovar
    testDictionary[id] = endFiles
    id += 1

#zbrisati zadnji dve vrstici v test dictionary
del testDictionary[385]
del testDictionary[386]

#vrne slovar slik za učiti
def get():
    return trainingDictionary




def getVal():
    return validationDictionary



#vrne slovar slik za testiranje natančnosti
def getTest():
    return testDictionary

print(testDictionary)

"""
k = sitk.GetArrayFromImage(sitk.ReadImage(arr[52])) #tole dela
print(k[70].shape)
k=k[70]
plt.imshow(k)
plt.show()"""
