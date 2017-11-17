import csv
import pandas as pd
import math
import seaborn
import jupyter
import time
import string

#Question1

def ListeFile(file):
    missing=0
    try:
        filejfk = open(file, 'r')
        liste = list(filejfk)
        return liste
    except:
        missing+=1


def CountLine(file):

    with open(file, 'r') as filejfk:
        reader = csv.reader(filejfk, delimiter=':', quoting=csv.QUOTE_NONE)

    line = 1
    for row in reader:
        count = 0
        for field in row:
            count = count+1
        line = line+1
        print('The Line'+str(line)+', there are '+str(count)+' fields')


#Question2

def CountMath(file):
    filejfk = open(file, 'r')
    liste = list(filejfk)
    nom = liste[0].split(';')
    col = nom.index('Num Pages')
    minimum = 999
    maximum = 0
    missing = 0
    somme = 0
    nbDocs = 0
    for line in liste:
        nbDocs += 1
        field = line.split(';')
        try:
            nbPages = int(field[col])
            somme = int(field[col]) + sum
            if nbPages < minimum:
                minimum = nbPages
            if nbPages > maximum:
                maximum = nbPages
        except ValueError:
            missing += 1
    mean = somme / nbDocs
    print("The mean is ", mean)
    print("The minimum is ", minimum)
    print("The maximum is", maximum)
    print("Missing are ", missing)

#Question3

def Doctype(file):
    filejfk=open(file,'r')
    liste = list(filejfk)
    nom = liste[0].split(';')
    type = nom.index('Doc Type')
    ListType = []

    for line in liste:
        field = line.split(';')
        try:
            Doctype = (field[type]);
            ListType.append(Doctype)
            return ListType
        except ValueError:
            print("Error")
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


def Dictionary(file):
    liste = list(file)
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
def Date(file):
    missing = 0
    filejfk=open(file,'r')
    liste = list(filejfk)
    nom = liste[0].split(';')
    date = nom.index('Doc Date')
    ListDate = []

    for line in liste:
        field = line.split(';')
        try:
            DocDate = field[date]
            if DocDate != "Doc Date":
                ListDate.append(DocDate)
        except :
            missing += 1
    return ListDate

def Oldest(file):
    ListDate=Date(file)
    minimum = 2000
    missing = 0
    for date in ListDate:
        field = date.split(';')
        dat = field[0]
        annnee = dat[6:10]
        try:
            year = eval(annnee)
            if year < minimum and year != 0:
                minimum = year
        except:
            missing += 1;
    print(minimum)

def Recent(file):

    ListDate=Date(file)
    maximum = 0
    missing = 0
    for date in ListDate:
        field = date.split(';')
        dat = field[0]
        annnee = dat[6:10]
        try:
            year = eval(annnee)
            if year > maximum and year != 0:
                maximum = year
        except:
            missing += 1;
    print(maximum)

def parsestring(date):
    missing=0
    try:
        field = date.split(';')
        dat = field[0]
        annnee = dat[6:10]
        year = eval(annnee)
        return year
    except:
        missing += 1

def DocumentPerYear(file):
    ListDate=Date(file)
    missing = 0
    n = len(ListDate)
    count = 0
    i = 0
    while i < n:
        year = parsestring(ListDate[i])
        for j in [0, n]:
            try:

                years = parsestring(ListDate[j])
                if year == years:
                    count += 1
            except:
                missing += 1;
        print("The number of ", year, "is ", count)
        i += 1

#Question5
def Pandaa(file):
    df = pd.read_csv(file, na_values=['.'], error_bad_lines=False,sep=";")
    # print(df.ix[:, 11])
    myList = list(df.ix[:, 11])
    minimum = 999
    maximum = 0
    missing = 0
    somme = 0
    total = 0

    for line in myList:

        try:
            somme += line
            total += 1

            if line < minimum:
                minimum = line
            if line > maximum:
                maximum = line
        except ValueError:
            missing += 1

    print(somme)
    print(total)
    mean = somme / total
    print("The mean is ", mean)
    print("The minimum is ", minimum)
    print("The maximum is", maximum)
    print("Missing are ", missing)


# Main
# Question1
file='jfkrelease-2017-dce65d0ec70a54d5744de17d280f3ad2.csv'
# CountLine(file)
# Question2
# CountMath(file)
# Question3
# Doctype(file)
# Dictionary(file)
# Question4
# Date(file)
# Oldest(file)
# Recent(file)
# DocumentPerYear(file)
# Question 5
Pandaa(file)











