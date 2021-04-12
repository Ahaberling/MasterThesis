import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

pd.set_option('display.max_columns', None)

patent_topicDist = pd.read_csv(r'D:\Universitaet Mannheim\MMDS 7. Semester\Master Thesis\Outline\Data\Cleaning Robots\cleaning_robot_EP_patents_07_04_topicNameMissing.csv', quotechar='"', skipinitialspace=True)
topics = pd.read_csv(r'D:\Universitaet Mannheim\MMDS 7. Semester\Master Thesis\Outline\Data\Cleaning Robots\cleaning_robot_EP_patents_07_04_topics.csv', quotechar='"', skipinitialspace=True)
parent = pd.read_csv(r'D:\Universitaet Mannheim\MMDS 7. Semester\Master Thesis\Outline\Data\Cleaning Robots\cleaning_robot_EP_backward_citations.csv', quotechar='"', skipinitialspace=True)

patent_topicDist = patent_topicDist.to_numpy()
parent = parent.to_numpy()
#print(patent_topicDist)


cited_pat_publn_id = parent[:,(0,2)]
print(cited_pat_publn_id)

#todo gibt es node ids in parent[:,(0,2)], die nicht in patent_topicDist auftauchen? Falls ja, how do we handle them?
#todo how do we handle patents that cite things that are not patents (parent[,2] = 0)?

print(patent_topicDist)
#print(patent_topicDist[0])
#print(patent_topicDist.T[0])

plain = nx.Graph()
bipart = nx.Graph()
topic_similar = nx.Graph()
topic_net = nx.Graph()

plain.add_nodes_from(patent_topicDist.T[0], test = patent_topicDist.T[1])

print(len(plain.nodes))
print(plain.nodes.data)

'''
unique, counts = np.unique(cited_pat_publn_id, return_counts=True)
print(np.asarray((unique, counts)).T)
unique, counts = np.unique(counts, return_counts=True)
print(np.asarray((unique, counts)).T)                                   # 1910 cited id 0. They will probably be excluded
'''

