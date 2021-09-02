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



    # 2. go through pattern array.
    #   for each entry == 1, find what patent recombines.
    #   same in a seperate structure
    #   delete recombinations that are no recombinations (189, 189)
    #   create new pattern array

    filename = 'lp_community_topicDist_dic'
    outfile = open(filename, 'wb')
    pk.dump(community_topicDist_dic, outfile)
    outfile.close()

    filename = 'lp_community_topTopic_dic'
    outfile = open(filename, 'wb')
    pk.dump(community_topTopic_dic, outfile)
    outfile.close()
