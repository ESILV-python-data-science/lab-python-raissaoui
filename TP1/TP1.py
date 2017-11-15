import csv
import pandas as pd
import math

#Question1
file='jfkrelease-2017-dce65d0ec70a54d5744de17d280f3ad2.csv'

def CountLine(filejfk):
    with open(filejfk,'r') as filejfk:
        reader = csv.reader(filejfk, delimiter=':', quoting=csv.QUOTE_NONE)

    ligne=1
    for row in reader:
        count = 0
        for field in row:
            count=count+1
        ligne=ligne+1
        print('La ligne'+str(ligne)+', il y a '+str(count)+' fields')


#Question2
def CountMath(filejfk):

    liste = list(filejfk)
    nom = liste[0].split(';')
    nbField = len(nom)
    count = 0

    mean = 0
    min = 999
    max = 0
    count = 0
    missing = 0
    nbPages = 0
    sum = 0
    col = nom.index('Num Pages')
    nbDocs = 0
    for line in liste:
        nbDocs += 1
        field = line.split(';')
        try:
            nbPages = int(field[col])
            sum = int(field[col]) + sum
            if nbPages < min:
                min = nbPages
            if nbPages > max:
                max = nbPages
        except ValueError:
            missing += 1
    mean = sum / nbDocs
    print("La moyenne est ", mean)
    print("La valeur minimum est ", min)
    print("La valeur maximum", max)
    print("Le nombre de valeurs manquantes est", missing)



#Question3

def Doctype(filejfk):
    liste = list(filejfk)
    nom = liste[0].split(';')
    type = nom.index('Doc Type')
    ListType = []

    for line in liste:
        field = line.split(';')
        try:
            Doctype = (field[type]);
            ListType.append(Doctype)
        except ValueError:
            print("Error")
    print(ListType)
    lenListe = len(ListType)
    for field in ListType:
        try:
            count = 0
            for champ in ListType:
                if field in champ:
                    count += 1
            print('Le nombre de ', field, 'est ', count)
        except ValueError:
            print("Error")


def Dictionary(filejfk):
    liste = list(filejfk)
    nom = liste[0].split(';')
    type = nom.index('Doc Type')
    ListType = []

    for line in liste:
        field = line.split(';')
        try:
            Doctype = (field[type]);
            ListType.append(Doctype)
        except ValueError:
            print("Error")
    print(ListType)
    lenListe = len(ListType)
    for field in ListType:
        try:
            count = 0
            for champ in ListType:
                if field in champ:
                    count += 1
            print('Le nombre de ', field, 'est ', count)
        except ValueError:
            print("Error")


#Question4
#What are the oldest and the more recent documents ?
#Compute the number of documents per year
def Oldest(filejfk):
    liste = list(filejfk)
    nom = liste[0].split(';')
    type = nom.index('Doc Type')
    ListType = []

    for line in liste:
        field = line.split(';')
        try:
            Doctype = (field[type]);
            ListType.append(Doctype)
        except ValueError:
            print("Error")
    print(ListType)
    lenListe = len(ListType)
    for field in ListType:
        try:
            count = 0
            for champ in ListType:
                if field in champ:
                    count += 1
            print('Le nombre de ', field, 'est ', count)
        except ValueError:
            print("Error")


#Question5
def Pandaa(filejfk):
    df = pd.read_csv(filejfk, na_values=['.'], error_bad_lines=False,
                     sep=";")

    # print(df.ix[:, 11])
    myList = list(df.ix[:, 11])
    somme = 0
    total = 0
    mean = 0
    max = myList[0]
    min = myList[0]
    nbPages = 0
    missing = 0
    for i in [0, len(myList)]:
        try:
            somme += myList[i]
            if myList[i] < min:
                min = nbPages
            if myList[i] > max:
                max = nbPages
        except ValueError:
            missing += 1

    mean = somme / len(myList)
    print(somme)
    print(mean)
    print(max)
    print(min)

#Main

#Question1
filejfk='jfkrelease-2017-dce65d0ec70a54d5744de17d280f3ad2.csv'
#CountLine(filejfk)
#Question2
#CountMath(filejfk)
#Question3
#Doctype(filejfk)
#Dictionary(filejfk)
#Question4
#Oldest(filejfk)
#Question 5
#Pandaa(filejfk)











