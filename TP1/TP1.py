import csv
import pandas as pd

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
Doctype = field[type]

for line in [1,liste]:
	field = line.split(';'); Doctype = field[type]
    #print(Doctype)
	try :
		if field[type]!=Doctype : print(field[type])
	except ValueError:
		missing += 1


