import os
import csv
import matplotlib.pyplot as plt
import pandas as pd

path = "./result"
file_list = os.listdir(path)

instance_list = []
scheduler_list = []

for i in file_list:
      if i.find("instance") != -1:
            instance_list.append(i)
      else:
            scheduler_list.append(i)

results = []

for name in instance_list:
      
      filename = "./result/" + name
      df = pd.read_csv(filename, skiprows=12)
      data = list(df.columns)
      
      bar1 = name.find('_') + 1
      bar2 = name.find('_', bar1) + 1
      bar3 = name.find('_', bar2) + 1
      bar4 = name.find('_', bar3) + 1
      bar5 = name.find('_', bar4) + 1
      bar6 = name.find('_', bar5) + 1
      bar7 = name.find('_', bar6) + 1

      algo = name[bar1:bar2-1]
      mecha = name[bar2:bar3-1]
      batch = int(name[bar4:bar5-1])
      num = int(name[bar6:bar7-1])

      

      result = []
      result.extend((algo, mecha, batch, num))
      result.extend((data[1], data[3], data[5]))
      results.append(result)

with open("summary.csv", 'w', newline='') as file:
      writer = csv.writer(file)
      writer.writerow(["Algorithm", "Mechanism", "Batch Size", "Instance Number", "ANTT", "STP", "Fairness"])
      for i in results:
            writer.writerow(i)

df = pd.read_csv("summary.csv")
row = df.shape[0]
antt = [0] * 12
stp = [0] * 12
fairness = [0] * 12
num = [0] * 12
for i in range(row):
      ser = df.iloc[i]
      algo_num = 0
      if ser['Algorithm'] == 'FCFS':
            algo_num = 0
      elif ser['Algorithm'] == 'RRB':
            algo_num = 1
      elif ser['Algorithm'] == 'HPF':
            algo_num = 2
      elif ser['Algorithm'] == 'SJF':
            algo_num = 3
      elif ser['Algorithm'] == 'TOKEN':
            algo_num = 4
      elif ser['Algorithm'] == 'PREMA':
            algo_num = 5

      mecha_num = 0
      if ser['Mechanism'] == 'STATIC':
            mecha_num = 1
      
      index = algo_num + mecha_num * 6
      
      antt[index] += ser['ANTT']
      stp[index] += ser['STP']
      fairness[index] += ser['Fairness']
      num[index] += 1

for i in range(12):
      antt[i] /= num[i]
      stp[i] /= num[i]
      fairness[i] /= num[i]

index = ['FCFS', 'RRB', 'HPF', 'SJF', 'TOKEN', 'PREMA', 'FCFS_S', 'RRB_S', 'HPF_S', 'SJF_S', 'TOKEN_S', 'PREMA_S']

plt.subplot(3, 1, 1)
plt.bar(index, antt)
plt.title('ANTT')
plt.subplot(3, 1, 2)
plt.bar(index, stp)
plt.title('STP')
plt.subplot(3, 1, 3)
plt.bar(index, fairness)
plt.title('Fairness')
plt.show()