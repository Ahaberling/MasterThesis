import numpy as np
import pandas as pd
import matplotlib as plt
import networkx as nx

import sys
import io



pd.set_option('display.max_columns', None)

#patents = np.genfromtxt(r'D:\Universitaet Mannheim\MMDS 7. Semester\Master Thesis\Outline\Data\Cleaning Robots\cleaning_robot_EP_patents.csv', delimiter=',', dtype=None, encoding='utf-8')
#patents_IPC = np.genfromtxt(r'D:\Universitaet Mannheim\MMDS 7. Semester\Master Thesis\Outline\Data\Cleaning Robots\cleaning_robot_EP_patents_IPC.csv', delimiter=',', dtype=None, encoding=None)


patent = pd.read_csv(r'D:\Universitaet Mannheim\MMDS 7. Semester\Master Thesis\Outline\Data\Cleaning Robots\cleaning_robot_EP_patents.csv', quotechar='"', skipinitialspace=True)
patent_IPC = pd.read_csv(r'D:\Universitaet Mannheim\MMDS 7. Semester\Master Thesis\Outline\Data\Cleaning Robots\cleaning_robot_EP_patents_IPC.csv', quotechar='"', skipinitialspace=True)

parent = pd.read_csv(r'D:\Universitaet Mannheim\MMDS 7. Semester\Master Thesis\Outline\Data\Cleaning Robots\cleaning_robot_EP_backward_citations.csv', quotechar='"', skipinitialspace=True)
parent_IPC = pd.read_csv(r'D:\Universitaet Mannheim\MMDS 7. Semester\Master Thesis\Outline\Data\Cleaning Robots\cleaning_robot_EP_backward_citations_IPC.csv', quotechar='"', skipinitialspace=True)


print("\n PATENT: \n", patent.head())
print("-------------------------------------------------------------------------")
print("\n PARENT: \n", parent.head())
print("-------------------------------------------------------------------------")
print("\n PARENT IPC: \n", patent_IPC.head())
print("-------------------------------------------------------------------------")
print("\n PARENT IPC: \n", parent_IPC.head())
print("-------------------------------------------------------------------------")

'''
print(len(patent))
print(len(parent))
print(len(patent_IPC))
print(len(parent_IPC))
'''
#df = np.where(parent['cited_pat_publn_id'] !=0)

print(len(parent.cited_pat_publn_id))
print(parent.cited_pat_publn_id.value_counts())     # 1910/18548 are 0

#print(len(parent.cited_appln_id))
#print(parent.cited_appln_id.value_counts())        # almost all 0

#print(df)

