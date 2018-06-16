import os
from configparser import ConfigParser
import re
from glob import glob


##training dictionary
train_path = 'C:\\BRATS\\BRATS2015_Training\\HGG'

def ending(fileList, end = ".mha"):
    "Funkcija  prejme seznam datotek in vrne seeznam datotek, ki se koncajo na dano končnico"
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


def getDictionary(train_path):
    "Vrne slovar, kjer so ključi ID pacientov vrednosti pa absoltune poti datotek Flair,T1,T1c,T2, answer"
    os.chdir(train_path)

    dictionary ={}
    id = 1
    for file in os.listdir():

        absolutPath = train_path + "\\" + file

        #seznam vseh končnih map
        endFiles = glob(absolutPath + "\\*\\*")

        #izberemo samo datoteke, ki  se končajo na .mha
        endFiles = ending(endFiles)

        #shranimo v slovar
        dictionary[id] = endFiles
        id += 1

    return dictionary

#naredimo slovar
dictionary = getDictionary(train_path)

def get():
    return dictionary


"""
k = sitk.GetArrayFromImage(sitk.ReadImage(arr[52])) #tole dela
print(k[70].shape)
k=k[70]
plt.imshow(k)
plt.show()"""
