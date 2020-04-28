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

alex = "AlexNet"
vgg = "VGG16"
google = "GoogLeNet"
mobile = "MobileNet"

asr = "Automatic Speech Recognition"
mt = "Machine Translation"
sa = "Sentiment Analysis"
'''
run_dict = {alex:{'FCFS':[], 'RRB':[], 'HPF':[], 'SJF':[], 'TOKEN':[], 'PREMA':[]} , vgg:{'FCFS':[], 'RRB':[], 'HPF':[], 'SJF':[], 'TOKEN':[], 'PREMA':[]}, \
     google:{'FCFS':[], 'RRB':[], 'HPF':[], 'SJF':[], 'TOKEN':[], 'PREMA':[]}, mobile:{'FCFS':[], 'RRB':[], 'HPF':[], 'SJF':[], 'TOKEN':[], 'PREMA':[]}, \
          asr:{'FCFS':[], 'RRB':[], 'HPF':[], 'SJF':[], 'TOKEN':[], 'PREMA':[]}, mt:{'FCFS':[], 'RRB':[], 'HPF':[], 'SJF':[], 'TOKEN':[], 'PREMA':[]}, \
               sa:{'FCFS':[], 'RRB':[], 'HPF':[], 'SJF':[], 'TOKEN':[], 'PREMA':[]}}
'''

run_dict = {alex:{'Isolated':[0], 'HPF':[], 'SJF':[], 'TOKEN':[], 'PREMA':[]} , vgg:{'Isolated':[0], 'HPF':[], 'SJF':[], 'TOKEN':[], 'PREMA':[]}, \
     google:{'Isolated':[0], 'HPF':[], 'SJF':[], 'TOKEN':[], 'PREMA':[]}, mobile:{'Isolated':[0], 'HPF':[], 'SJF':[], 'TOKEN':[], 'PREMA':[]}, \
          asr:{'Isolated':[0], 'HPF':[], 'SJF':[], 'TOKEN':[], 'PREMA':[]}, mt:{'Isolated':[0], 'HPF':[], 'SJF':[], 'TOKEN':[], 'PREMA':[]}, \
               sa:{'Isolated':[0], 'HPF':[], 'SJF':[], 'TOKEN':[], 'PREMA':[]}}


for name in instance_list:
      
    filename = "./result/" + name
    df = pd.read_csv(filename, nrows=11)
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

    if mecha == 'STATIC':
        continue
    if algo == 'FCFS' or algo == 'RRB':
        continue

    for i in range(1, num+1):
        data = df[str(i)]
        name = data[0]
        run_dict[name][algo].append(int(data[7]))
        run_dict[name]['Isolated'][0] = int(data[1])

for i in run_dict:
    for j in run_dict[i]:
        run_dict[i][j] = min(run_dict[i][j])

index = 1

for i in run_dict:
    plt.subplot(4, 2, index)
    plt.bar(range(len(run_dict[i])), list(run_dict[i].values()))
    plt.xticks(range(len(run_dict[i])), list(run_dict[i].keys()))
    plt.title(i)
    index += 1

plt.tight_layout ()
plt.show()
        




