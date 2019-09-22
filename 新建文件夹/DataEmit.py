import random
import numpy as np 
import sys

list = [sys.argv[1].split(',')]

w0 = int(list[0][0][1])

w1 = int(list[0][1])

w2 = int(list[0][2][0])

m = int(sys.argv[2])

n = int(sys.argv[3]) 


positives = []

neigitivs = []

fo1 = open('train.txt','w')

for i in range(m):
    while True:
        x1 = np.random.uniform(-10,10,500)
        x2 = np.random.uniform(-10,10,500)
        if x1[i] * w1 + x2[i] * w2 + w0 > 0:
            positives.append('{},{},{}'.format(x1[i],x2[i],"+"))
            print(x1[i],x2[i],'+')
            break

for j in range(n):
    while True:
        x1 = np.random.uniform(-10,10,500)
        x2 = np.random.uniform(-10,10,500)
        if x1[j] * w1 + x2[j] * w2 + w0 < 0:
            neigitivs.append('{},{},{}'.format(x1[j],x2[j],"-"))
            print(x1[j],x2[j],'-')
            break
                    
total = neigitivs + positives

random.shuffle(total)

for i in range(len(total)):
    fo1.write(total[i]+'\n')
    
fo1.close()
