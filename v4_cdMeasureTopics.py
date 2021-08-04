if __name__ == '__main__':
# --- Import Libraries ---#
    print('\n#--- Import Libraries ---#\n')

    import pandas as pd
    import numpy as np
    import pickle as pk

    import networkx as nx
    from cdlib import algorithms
    # import wurlitzer                   #not working for windows

    import tqdm
    import itertools
    import operator
    import os
    import sys

# --- Initialization ---#
    print('\n# --- Initialization ---#\n')

    os.chdir('D:/Universitaet Mannheim/MMDS 7. Semester/Master Thesis/Outline/Data/Cleaning Robots')

    patent_lda_ipc = pd.read_csv('patent_lda_ipc.csv', quotechar='"', skipinitialspace=True)
    patent_lda_ipc = patent_lda_ipc.to_numpy()

    with open('windows_topicSim', 'rb') as handle:
        topicSim = pk.load(handle)

    with open('lp_recombinations', 'rb') as handle:
        lp_recombinations = pk.load(handle)
    with open('lp_labeled', 'rb') as handle:
        lp_labeled = pk.load(handle)
    with open('lp_topD_dic', 'rb') as handle:
        lp_topD_dic = pk.load(handle)
    with open('lp_topD', 'rb') as handle:
        lp_topD = pk.load(handle)

    with open('gm_recombinations', 'rb') as handle:
        gm_recombinations = pk.load(handle)
    with open('gm_labeled', 'rb') as handle:
        gm_labeled = pk.load(handle)
    with open('gm_topD_dic', 'rb') as handle:
        gm_topD_dic = pk.load(handle)
    with open('gm_topD', 'rb') as handle:
        gm_topD = pk.load(handle)

    with open('kclique_recombinations', 'rb') as handle:
        kclique_recombinations = pk.load(handle)
    with open('kclique_labeled', 'rb') as handle:
        kclique_labeled = pk.load(handle)
    with open('kclique_topD_dic', 'rb') as handle:
        kclique_topD_dic = pk.load(handle)
    with open('kclique_topD', 'rb') as handle:
        kclique_topD = pk.load(handle)

    with open('lais2_recombinations', 'rb') as handle:
        lais2_recombinations = pk.load(handle)
    with open('lais2_labeled', 'rb') as handle:
        lais2_labeled = pk.load(handle)
    with open('lais2_topD_dic', 'rb') as handle:
        lais2_topD_dic = pk.load(handle)
    with open('lais2_topD', 'rb') as handle:
        lais2_topD = pk.load(handle)


    print(lp_recombinations['window_0'])
    print(lp_labeled['window_0'])
    print(lp_topD_dic['window_0'])
    print(lp_topD['window_0'])


    community_ids_all = []
    for window_id, window in lp_labeled.items():
        for community in window:
            community_ids_all.append(community[1][0])

    column_length = max(community_ids_all)

    column_unique = np.unique(community_ids_all)
    column_unique.sort()
    column_unique_length = len(column_unique)

    print(community_ids_all)
    print(column_length)
    #print(column_unique)
    print(column_unique_length)

    print(patent_lda_ipc[0])

    # get biggest community each window:

    community_size_dic = {}
    for window_id, window in lp_labeled.items():
        size_list = []
        for community in window:
            size_list.append(len(community))
        community_size_dic[window_id] = max(size_list)


    #for i in range(column_unique_length):
    #    for window_id, window in lp_labeled.items():

    community_topicDist_dic = {}

    for window_id, window in lp_labeled.items():
        window_list = []

        #for i in range(column_unique_length):
            
        for community in window:
            community_topics = np.zeros((len(community[0]), 325))
            topic_list = []

            #if community[1][0] == i:
            for patent in community[0]:

                paten_pos = np.where(patent_lda_ipc[:,0] == patent)

                topic_list.append(patent_lda_ipc[paten_pos[0][0]][9:30])

            topic_list = [item for sublist in topic_list for item in sublist]
            topic_list = [x for x in topic_list if x == x]
            #print(topic_list)

            for i in range(0, len(topic_list), 2):

                for row in range(len(community_topics)):

                    #print(community_topics[row, int(topic_list[i])])
                    if community_topics[row, int(topic_list[i])] == 0:
                        #print(community_topics[row, int(topic_list[i])])
                        community_topics[row, int(topic_list[i])] = topic_list[i+1]
                        break

            community_topics = np.sum(community_topics, axis=0)

            window_list.append([community[1][0], list(community_topics)])

        community_topicDist_dic[window_id] = window_list

    # 1. create dic with: each window, list of tuple with (communityID, highest topic)

    # 2. go through pattern array.
    #   for each entry == 1, find what patent recombines.
    #   same in a seperate structure
    #   delete recombinations that are no recombinations (189, 189)
    #   create new pattern array
