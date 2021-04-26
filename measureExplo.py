import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle as pk

#--- Initialization --#

directory = 'D:/Universitaet Mannheim/MMDS 7. Semester/Master Thesis/Outline/Data/Cleaning Robots/'

patent_lda_ipc = pd.read_csv( directory + 'patent_lda_ipc.csv', quotechar='"', skipinitialspace=True)
topics = pd.read_csv(r'D:\Universitaet Mannheim\MMDS 7. Semester\Master Thesis\Outline\Data\Cleaning Robots\patent_topics.csv', quotechar='"', skipinitialspace=True)
#parent = pd.read_csv(r'D:\Universitaet Mannheim\MMDS 7. Semester\Master Thesis\Outline\Data\Cleaning Robots\cleaning_robot_EP_backward_citations.csv', quotechar='"', skipinitialspace=True)

patent_lda_ipc = patent_lda_ipc.to_numpy()
topics = topics.to_numpy()


print(patent_lda_ipc)
print(np.shape(patent_lda_ipc))

print(topics)
