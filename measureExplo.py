import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle as pk
import os

#--- Initialization --#

os.chdir('D:/Universitaet Mannheim/MMDS 7. Semester/Master Thesis/Outline/Data/Cleaning Robots')

irectory = 'D:/Universitaet Mannheim/MMDS 7. Semester/Master Thesis/Outline/Data/Cleaning Robots/'

#patent_lda_ipc = pd.read_csv( directory + 'patent_lda_ipc.csv', quotechar='"', skipinitialspace=True)
patent_lda_ipc = pd.read_csv('patent_lda_ipc.csv', quotechar='"', skipinitialspace=True)
topics = pd.read_csv(r'D:\Universitaet Mannheim\MMDS 7. Semester\Master Thesis\Outline\Data\Cleaning Robots\patent_topics.csv', quotechar='"', skipinitialspace=True)
og_ipc = pd.read_csv(r'D:\Universitaet Mannheim\MMDS 7. Semester\Master Thesis\Outline\Data\Cleaning Robots\cleaning_robot_EP_patents_IPC.csv', quotechar='"', skipinitialspace=True)
#parent = pd.read_csv(r'D:\Universitaet Mannheim\MMDS 7. Semester\Master Thesis\Outline\Data\Cleaning Robots\cleaning_robot_EP_backward_citations.csv', quotechar='"', skipinitialspace=True)

patent_lda_ipc = patent_lda_ipc.to_numpy()
topics = topics.to_numpy()
og_ipc = og_ipc.to_numpy()



print(patent_lda_ipc)
print(np.shape(patent_lda_ipc))

print(og_ipc)
print(np.shape(og_ipc))

print(len(np.unique(og_ipc[:,0])))
print(len(np.unique(og_ipc[:,1]))) # 970 unique ipcs (and topics)

print(patent_lda_ipc[0,:])

with open('window90by1bbb', 'rb') as handle:
    b = pk.load(handle)

print(b)



#print(topics)
