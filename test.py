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
antt = [0] * 8
stp = [0] * 8
fairness = [0] * 8
num = [0] * 8
for i in range(row):
      ser = df.iloc[i]
      algo_num = 0
      if ser['Algorithm'] == 'HPF':
            algo_num = 0
      elif ser['Algorithm'] == 'SJF':
            algo_num = 1
      elif ser['Algorithm'] == 'TOKEN':
            algo_num = 2
      elif ser['Algorithm'] == 'PREMA':
            algo_num = 3
      else:
            continue

      mecha_num = 0
      if ser['Mechanism'] == 'STATIC':
            mecha_num = 1
      
      index = algo_num + mecha_num * 4
      
      antt[index] += ser['ANTT']
      stp[index] += ser['STP']
      fairness[index] += ser['Fairness']
      num[index] += 1

for i in range(8):
      if num[i] == 0:
            continue
      else:
            antt[i] /= num[i]
            stp[i] /= num[i]
            fairness[i] /= num[i]

index = ['HPF', 'SJF', 'TOKEN', 'PREMA',  'HPF', 'SJF', 'TOKEN', 'PREMA']

plt.subplot(2, 3, 1)
plt.bar(index[0:4], antt[0:4])
plt.title('ANTT')
plt.subplot(2, 3, 2)
plt.bar(index[0:4], fairness[0:4])
plt.title('Fairness')
plt.subplot(2, 3, 3)
plt.bar(index[0:4], stp[0:4])
plt.title('STP')

plt.subplot(2, 3, 4)
plt.bar(index[4:], antt[4:])
plt.title('ANTT(Static)')
plt.subplot(2, 3, 5)
plt.bar(index[4:], fairness[4:])
plt.title('Fairness(Static)')
plt.subplot(2, 3, 6)
plt.bar(index[4:], stp[4:])
plt.title('STP(Static)')

plt.rc('font', size=20)

plt.show()