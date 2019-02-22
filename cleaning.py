import pandas as pd
import numpy as np
from sklearn import neighbors,preprocessing,cross_validation
import matplotlib.pyplot as plt
import math
import warnings
from collections import Counter
import csv
from itertools import zip_longest
import statistics as s1

def find(a):
    sum1=[]
    k=0
    x=[]
    for i in a:
        if(k==128):
            b=s1.mean(sum1)
            x.append(b)
            sum1=[]
            k=0
        else:
            sum1.append(i)
            k=k+1
    return(x)

def find1(a):
    sum1=[]
    k=0
    x=[]
    for i in a:
        if(k==128):
            b=s1.pvariance(sum1)
            x.append(b)
            sum1=[]
            k=0
        else:
            sum1.append(i)
            k=k+1
    return(x)

def find1(a):
    sum1=[]
    k=0
    x=[]
    for i in a:
        if(k==128):
            b=s1.pstdev(sum1)
            x.append(b)
            sum1=[]
            k=0
        else:
            sum1.append(i)
            k=k+1
    return(x)

d1=pd.read_csv('default__1.csv')
d2=pd.read_csv('default__2.csv')
d3=pd.read_csv('default__3.csv')
d4=pd.read_csv('default__4.csv')
d5=pd.read_csv('default__5.csv')
d6=pd.read_csv('default__6.csv')
x1=find(d1.ix[:,0].values)+find(d1.ix[:,1].values)+find(d1.ix[:,2].values)+find(d1.ix[:,3].values)
x2=find(d2.ix[:,0].values)+find(d2.ix[:,1].values)+find(d2.ix[:,2].values)+find(d2.ix[:,3].values)
x3=find(d3.ix[:,0].values)+find(d3.ix[:,1].values)+find(d3.ix[:,2].values)+find(d3.ix[:,3].values)
x4=find(d4.ix[:,0].values)+find(d4.ix[:,1].values)+find(d4.ix[:,2].values)+find(d4.ix[:,3].values)
x5=find(d5.ix[:,0].values)+find(d5.ix[:,1].values)+find(d5.ix[:,2].values)+find(d5.ix[:,3].values)
x6=find(d6.ix[:,0].values)+find(d6.ix[:,1].values)+find(d6.ix[:,2].values)+find(d6.ix[:,3].values)
y1=find1(d1.ix[:,0].values)+find1(d1.ix[:,1].values)+find1(d1.ix[:,2].values)+find(d1.ix[:,3].values)
y2=find1(d2.ix[:,0].values)+find1(d2.ix[:,1].values)+find1(d2.ix[:,2].values)+find1(d2.ix[:,3].values)
y3=find1(d3.ix[:,0].values)+find1(d3.ix[:,1].values)+find1(d3.ix[:,2].values)+find1(d3.ix[:,3].values)
y4=find1(d4.ix[:,0].values)+find1(d4.ix[:,1].values)+find1(d4.ix[:,2].values)+find1(d4.ix[:,3].values)
y5=find1(d5.ix[:,0].values)+find1(d5.ix[:,1].values)+find1(d5.ix[:,2].values)+find1(d5.ix[:,3].values)
y6=find1(d6.ix[:,0].values)+find1(d6.ix[:,1].values)+find1(d6.ix[:,2].values)+find1(d6.ix[:,3].values)
z1=find(d1.ix[:,0].values)+find(d1.ix[:,1].values)+find(d1.ix[:,2].values)+find(d1.ix[:,3].values)
z2=find(d2.ix[:,0].values)+find(d2.ix[:,1].values)+find(d2.ix[:,2].values)+find(d2.ix[:,3].values)
z3=find(d3.ix[:,0].values)+find(d3.ix[:,1].values)+find(d3.ix[:,2].values)+find(d3.ix[:,3].values)
z4=find(d4.ix[:,0].values)+find(d4.ix[:,1].values)+find(d4.ix[:,2].values)+find(d4.ix[:,3].values)
z5=find(d5.ix[:,0].values)+find(d5.ix[:,1].values)+find(d5.ix[:,2].values)+find(d5.ix[:,3].values)
z6=find(d6.ix[:,0].values)+find(d6.ix[:,1].values)+find(d6.ix[:,2].values)+find(d6.ix[:,3].values)
#print(x6)

a1=x1+x2+x3
a2=y1+y2+y3
a3=z1+z2+z3
b1=x4+x5
b2=y4+y5
b3=z4+z5

r=['Standing']
p=[a1,a2,a3,r]
r=['Walking']
q=[b1,b2,b3,r]
s=[x6,y6,z6]
data=zip_longest(*p,fillvalue='Standing')
data1=zip_longest(*q,fillvalue='Walking')
data2=zip_longest(*s,fillvalue='')
with open('Fdata1.csv','w') as f:
    w=csv.writer(f,lineterminator='\n')
    #w.writerow(("Mean","Variance","Standard Deviation","Class"))
    w.writerows(data)
    w.writerows(data1)
f.close()
with open('Fdata.csv','w') as f1:
    w=csv.writer(f1,lineterminator='\n')
    #w.writerow(("Mean","Variance","Standard Deviation"))
    w.writerows(data2)
f1.close()

    

