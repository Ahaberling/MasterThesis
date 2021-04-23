import pandas as pd
import numpy as np

# not that important for now

pd.set_option('display.max_columns', None)

patent = pd.read_csv(r'D:\Universitaet Mannheim\MMDS 7. Semester\Master Thesis\Outline\Data\Cleaning Robots\cleaning_robot_EP_patents.csv', quotechar='"', skipinitialspace=True)
patent_ipc = pd.read_csv(r'D:\Universitaet Mannheim\MMDS 7. Semester\Master Thesis\Outline\Data\Cleaning Robots\cleaning_robot_EP_patents_IPC.csv', quotechar='"', skipinitialspace=True)

parent = pd.read_csv(r'D:\Universitaet Mannheim\MMDS 7. Semester\Master Thesis\Outline\Data\Cleaning Robots\cleaning_robot_EP_backward_citations.csv', quotechar='"', skipinitialspace=True)
parent_ipc = pd.read_csv(r'D:\Universitaet Mannheim\MMDS 7. Semester\Master Thesis\Outline\Data\Cleaning Robots\cleaning_robot_EP_backward_citations_IPC.csv', quotechar='"', skipinitialspace=True)

print(patent)
print('\n--------------------------------\n')
print(patent_ipc)

patent = patent.to_numpy()
patent_ipc = patent_ipc.to_numpy()

parent = parent.to_numpy()
parent_ipc = parent_ipc.to_numpy()

'''
comp_list = []
for i in range(len(patent)):
    if patent[i,0] in patent_ipc[:,0]:
        comp_list.append(patent[i,0])

print(len(comp_list))'''

val, count = np.unique(patent_ipc[:,0], return_counts=True)

print(len(val))
print(len(patent))

print(max(count))

patent_join = np.empty((np.shape(patent)[0],np.shape(patent)[1]+max(count)*3), dtype=object)

print(np.shape(patent))
print(np.shape(patent_join))

patent_helper = np.empty((np.shape(patent)[0],max(count)*3+1), dtype=object)

patent_helper[:,0] = patent[:,0]

#print(patent_helper.T)

#for i in patent[:,0]:



