import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle as pk
import os
import itertools

#--- Initialization --#

os.chdir('D:/Universitaet Mannheim/MMDS 7. Semester/Master Thesis/Outline/Data/Cleaning Robots')

#directory = 'D:/Universitaet Mannheim/MMDS 7. Semester/Master Thesis/Outline/Data/Cleaning Robots/'

#patent_lda_ipc = pd.read_csv( directory + 'patent_lda_ipc.csv', quotechar='"', skipinitialspace=True)
patent_lda_ipc = pd.read_csv('patent_lda_ipc.csv', quotechar='"', skipinitialspace=True)
topics = pd.read_csv(r'D:\Universitaet Mannheim\MMDS 7. Semester\Master Thesis\Outline\Data\Cleaning Robots\patent_topics.csv', quotechar='"', skipinitialspace=True)
og_ipc = pd.read_csv(r'D:\Universitaet Mannheim\MMDS 7. Semester\Master Thesis\Outline\Data\Cleaning Robots\cleaning_robot_EP_patents_IPC.csv', quotechar='"', skipinitialspace=True)
#parent = pd.read_csv(r'D:\Universitaet Mannheim\MMDS 7. Semester\Master Thesis\Outline\Data\Cleaning Robots\cleaning_robot_EP_backward_citations.csv', quotechar='"', skipinitialspace=True)

patent_lda_ipc = patent_lda_ipc.to_numpy()
topics = topics.to_numpy()
og_ipc = og_ipc.to_numpy()


'''
print(patent_lda_ipc)
print(np.shape(patent_lda_ipc))

print(og_ipc)
print(np.shape(og_ipc))

print(len(np.unique(og_ipc[:,0])))
print(len(np.unique(og_ipc[:,1]))) # 970 unique ipcs (and topics)

print(patent_lda_ipc[0,:])
'''
with open('window90by1', 'rb') as handle:
    window90by1 = pk.load(handle)

#print(window90by1)
print(window90by1['window_0'])
print(np.shape(window90by1['window_0']))

print(window90by1['window_0'][0])
print(window90by1['window_0'][0][0])

# now I want for each window the distribution of the ipc/topics

window90by1_dist_ipc = {}

#ipc_position = range(53,91,3)

ipc_position = np.r_[range(52,91,3)]
topic_position = np.r_[range(10,52,3)]

window90by1_ipcs = {}
window90by1_topics = {}

window90by1_ipcs_allComb = {}
window90by1_topics_allComb = {}

window90by1_ipcs_twoComb = {}
window90by1_topics_twoComb = {}

window90by1_ipcs_threeComb = {}
window90by1_topics_threeComb = {}

c = 0

for window in window90by1.values():
    #print('----')
    #print(window.key)
    #print(window)

    ipc_list = []
    topic_list = []

    ipc_allComb_list = []
    topic_allComb_list = []

    ipc_twoComb_list = []
    topic_twoComb_list = []

    ipc_threeComb_list = []
    topic_threeComb_list = []

    for patent in window:
        #print(patent[ipc_position4])
        #print(patent[np.r_[52,55,58]])
        #print(patent[ipc_position5])
        #print(patent[9:15])
        ipc_list.append(patent[ipc_position])
        topic_list.append(patent[topic_position])

        # ipc_allComb_list
        y = [x for x in patent[ipc_position] if x == x]             # nan elimination
        y = np.unique(y)                                            # todo probably/hopefully not neccessary (because hopefully the data is clean enough so that one paper is not classfied with the same ipc more than once)
        ipc_allComb_list.append(tuple(y))

        # topic_allComb_list
        z = [x for x in patent[topic_position] if x == x]           # nan elimination
        z = np.unique(z)                                            # todo probably/hopefully not neccessary (because hopefully the data is clean enough so that one paper is not classfied with the same topic more than once)
        topic_allComb_list.append(tuple(z))

        # ipc_twoComb_list
        ipc_twoComb_list.append(list(itertools.combinations(y, r=2)))

        # topic_twoComb_list
        topic_twoComb_list.append(list(itertools.combinations(z, r=2)))

        # ipc_threeComb_list
        ipc_threeComb_list.append(list(itertools.combinations(y, r=3)))

        # topic_threeComb_list
        topic_threeComb_list.append(list(itertools.combinations(z, r=3)))


    #print(ipc_comb_list)

    # all ipcs that occured in the window in general
    ipc_list = np.concatenate(ipc_list).ravel().tolist()
    ipc_list = [x for x in ipc_list if x == x]
    ipc_list = np.unique(ipc_list)
    window90by1_ipcs['window_{0}'.format(c)] = ipc_list

    # all topics that occured in the window in general
    topic_list = np.concatenate(topic_list).ravel().tolist()
    topic_list = [x for x in topic_list if x == x]
    topic_list = np.unique(topic_list)
    window90by1_topics['window_{0}'.format(c)] = topic_list

    # all ipcs combinations as tuple that occured in the window
    # meaning one patent -> one tuple
    #ipc_comb_list = np.unique(ipc_comb_list)                       # todo Error message, but I probably also dont want to do that in general
    window90by1_ipcs_allComb['window_{0}'.format(c)] = ipc_allComb_list

    # all topic combinations as tuple that occured in the window
    # meaning one patent -> one tuple
    #topic_comb_list = np.unique(topic_comb_list)                   # todo Error message, but I probably also dont want to do that in general
    window90by1_topics_allComb['window_{0}'.format(c)] = topic_allComb_list

    # all ipc inside a patent as pairs in the window
    # meaning one patent -> (possibly) multiple tuples of size two
    #print(window[1])
    #print(ipc_twoComb_list)                                             #todo somehow we got empty lists in here? is it for patents with only one ipc? -> no combination possible?
    ipc_twoComb_list = [item for sublist in ipc_twoComb_list for item in sublist]
    #print(ipc_twoComb_list)
    #ipc_twoComb_list = np.array(ipc_twoComb_list).ravel()
    #print(ipc_twoComb_list)
    window90by1_ipcs_twoComb['window_{0}'.format(c)] = ipc_twoComb_list


    # all topic inside a patent as pairs in the window
    # meaning one patent -> (possibly) multiple tuples of size two
    #print(topic_twoComb_list)
    topic_twoComb_list = [item for sublist in topic_twoComb_list for item in sublist]
    #print(topic_twoComb_list)
    window90by1_topics_twoComb['window_{0}'.format(c)] = topic_twoComb_list

    # all ipc inside a patent as triples in the window
    # meaning one patent -> (possibly) multiple tuples of size three
    #print(ipc_threeComb_list)
    ipc_threeComb_list = [item for sublist in ipc_threeComb_list for item in sublist]
    #print(ipc_threeComb_list)
    window90by1_ipcs_threeComb['window_{0}'.format(c)] = ipc_threeComb_list

    # all topic inside a patent as triples in the window
    # meaning one patent -> (possibly) multiple tuples of size three
    topic_threeComb_list = [item for sublist in topic_threeComb_list for item in sublist]
    window90by1_topics_threeComb['window_{0}'.format(c)] = topic_threeComb_list

    c = c + 1

print(window90by1_ipcs_twoComb)


# I need all pair combinations that occur in the whole timeframe
# construct heatmap with  x = combination, y = window, z = increase of occurence
# for this find list with all windows
# find list with all unique pairs
# interate through dictionary and fill he dict

#todo idea: right now window90by1_ipcs_twoComb contains tuples like ('C12M   1', 'C12M   3'). If this is to fine grained (no real inovation/ recombination) then go more course graind (or fine grained)

# recombination:
# is when a combination (2+) of ipc's/topics is cited together for the first time in X
# or if the number of patents combining them cross a threshold for the first time in X

# diffusion:
# is active as long as the number of a topic/ipc or the number of a combination of them is above a certain threshold

