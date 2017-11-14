import csv
import pandas as pd
import math

#Question1
with open('jfkrelease-2017-dce65d0ec70a54d5744de17d280f3ad2.csv','r') as filejfk:
    reader = csv.reader(filejfk, delimiter=':', quoting=csv.QUOTE_NONE)

    ligne=1
    for row in reader:
        count = 0
        for field in row:
            count=count+1
        ligne=ligne+1
        print('La ligne'+str(ligne)+', il y a '+str(count)+' fields')


#Question2
#Compute the mean number of pages per document, the minimum and maximum of pages per document

#with open('jfkrelease-2017-dce65d0ec70a54d5744de17d280f3ad2.csv','r') as filejfk:
#reader = csv.reader(filejfk, delimiter=':', quoting=csv.QUOTE_NONE)
filejfk = open('jfkrelease-2017-dce65d0ec70a54d5744de17d280f3ad2.csv','r')

liste = list(filejfk)
nom = liste[0].split(';')
nbField = len(nom)
count=0

mean=0
min=999
max=0
count=0
missing = 0
nbPages=0
sum=0
col=nom.index('Num Pages')
nbDocs=0
for line in liste:
	nbDocs+=1
	field = line.split(';')
	try :
		nbPages=int(field[col])
		sum=int(field[col])+sum
		if nbPages<min:
			min=nbPages
		if nbPages>max:
			max=nbPages
	except ValueError:
		missing += 1
mean=sum/nbDocs
print("La moyenne est ", mean)
print("La valeur minimum est ",min)
print("La valeur maximum", max)
print("Le nombre de valeurs manquantes est", missing)


#Question3
#doc type =7-1

type=nom.index('Doc Type')
ListType=[]

for line in liste:
	field = line.split(';')
	try :
		Doctype=(field[type]);ListType.append(Doctype)
	except ValueError:
		missing += 1
print(ListType)
lenListe=len(ListType)
for field in ListType:
    try:
        count = 0
        for champ in ListType:
            if field in champ:
                count += 1
        print('Le nombre de ',field,'est ',count)
    except ValueError:
        missing+=1


#Question4

#Question5
df = pd.read_csv('jfkrelease-2017-dce65d0ec70a54d5744de17d280f3ad2.csv',na_values=['.'],error_bad_lines=False,sep=";")

#print(df.ix[:, 11])
myList = list(df.ix[:, 11])
somme=0
total=0
mean=0
max=myList[0]
min=myList[0]
nbPages=0
missing=0

#Question6








