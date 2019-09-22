import sys
import numpy as np
import matplotlib.pyplot as plt
 
file = open(sys.argv[1],'r')

data_set = []

data_label = []

data_back = []

data_reback1 = []

data_reback2 = []



for line in file:
    line=line.rstrip("\n")
    line1 = line.split(',')
    data_back.append(line1)
    
file.close()    

for i in range(len(data_back)):
    if data_back[i][-1] == '+':
        data_reback1.append('{},{},{}'.format(data_back[i][0],data_back[i][1],'1'))
    else:
        data_reback2.append('{},{},{}'.format(data_back[i][0],data_back[i][1],'-1'))

data_reback = data_reback1+data_reback2   

fo1 = open('3.txt','w')
for i in range(len(data_reback)):
    fo1.write(data_reback[i]+'\n')     
fo1.close()

file1 = open('3.txt','r')

for line in file1:
    line=line.rstrip("\n")
    line1 = line.split(',')
    for i in range(len(line1)):
        line1[i] = float(line1[i])
    data_set.append(line1[0:2])
    data_label.append(int(line1[-1]))    
file1.close()
data = np.array(data_set)
label = np.array(data_label)
# 初始化w, b, alpha
w = np.array([0, 0])
w0 = 0
alpha = 1
# 计算 y*(w*x+b)
f = (np.dot(data, w.T) + w0) * label

idx = np.where(f <= 0)

# 使用随机梯度下降法求解w, b

w_countlist = []

while f[idx].size != 0:
    
    point = np.random.randint((f[idx].shape[0]))
    
    x = data[idx[0][point], :]
    
    y = label[idx[0][point]]
    
    w = w + alpha * y * x
    
    w0 = w0 + alpha * y
    
    w_countlist.append('{},{},{}'.format(w0,w[0],w[1]))
                        
    f = (np.dot(data, w.T) + w0) * label
    
    idx = np.where(f <= 0)
    
file2 = open('PLA.txt','w')

for i in range(len(w_countlist)):
    file2.write(w_countlist[i]+'\n')
    print([w_countlist[i]])
    
file2.close()

# 绘图显示
x1 = np.arange(-10, 10, 0.01)
x2 = (w[0] * x1 + w0) / (-w[1])
idx_p = np.where(label == 1)
idx_n = np.where(label != 1)
data_p = data[idx_p]
data_n = data[idx_n]

plt.plot(x1, x2,color='black')

plt.scatter(data_p[:, 0], data_p[:, 1], color='red',marker="+",label='+')
plt.scatter(data_n[:, 0], data_n[:, 1], color='blue',marker="o",label='--')
plt.legend(loc='best')
plt.show()
