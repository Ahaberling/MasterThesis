import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

patent_topicDist = pd.read_csv(r'D:\Universitaet Mannheim\MMDS 7. Semester\Master Thesis\Outline\Data\Cleaning Robots\cleaning_robot_EP_patents_07_04_topicDist.csv', quotechar='"', skipinitialspace=True)
topics = pd.read_csv(r'D:\Universitaet Mannheim\MMDS 7. Semester\Master Thesis\Outline\Data\Cleaning Robots\cleaning_robot_EP_patents_07_04_topics.csv', quotechar='"', skipinitialspace=True)
parent = pd.read_csv(r'D:\Universitaet Mannheim\MMDS 7. Semester\Master Thesis\Outline\Data\Cleaning Robots\cleaning_robot_EP_backward_citations.csv', quotechar='"', skipinitialspace=True)

patent_topicDist = patent_topicDist.to_numpy()
topics = topics.to_numpy()
parent = parent.to_numpy()

cited_pat_publn_id = parent[:,2]


unique, counts = np.unique(cited_pat_publn_id, return_counts=True)
print(np.asarray((unique, counts)).T)
unique, counts = np.unique(counts, return_counts=True)
print(np.asarray((unique, counts)).T)                                   # 1910 cited id 0. They will probably be excluded

plain = nx.Graph()
bipart = nx.Graph()
topic_similar = nx.Graph()
topic_net = nx.Graph()