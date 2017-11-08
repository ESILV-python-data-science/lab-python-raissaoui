import csv

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

somme=0
total=0
mean=0
max=0
min=0

with open('jfkrelease-2017-dce65d0ec70a54d5744de17d280f3ad2.csv','r') as filejfk:
    reader = csv.reader(filejfk, delimiter=':', quoting=csv.QUOTE_NONE)

    for row in reader:
        field=row
        #print(field)
        nbpages=list(field)
        #print(nbpages)
        #print(type(nbpages))
        #print(len(nbpages))

import pandas as pd

df = pd.read_csv('jfkrelease-2017-dce65d0ec70a54d5744de17d280f3ad2.csv',na_values=['.'],error_bad_lines=False,sep=";")


print(df.ix[:, 11])
