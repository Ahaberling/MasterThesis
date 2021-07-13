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

# --- Initialization ---#
    print('\n# --- Initialization ---#\n')

    os.chdir('D:/Universitaet Mannheim/MMDS 7. Semester/Master Thesis/Outline/Data/Cleaning Robots')

    #patent_lda_ipc = pd.read_csv('patent_lda_ipc.csv', quotechar='"', skipinitialspace=True)
    #patent_lda_ipc = patent_lda_ipc.to_numpy()

    #with open('windows_topicSim', 'rb') as handle:
        #topicSim = pk.load(handle)

    with open('lp_recombinations', 'rb') as handle:
        lp_recombinations = pk.load(handle)
    with open('lp_labeled', 'rb') as handle:
        lp_labeled = pk.load(handle)

    with open('gm_recombinations', 'rb') as handle:
        gm_recombinations = pk.load(handle)
    with open('gm_labeled', 'rb') as handle:
        gm_labeled = pk.load(handle)

    with open('kclique_recombinations', 'rb') as handle:
        kclique_recombinations = pk.load(handle)
    with open('kclique_labeled', 'rb') as handle:
        kclique_labeled = pk.load(handle)

    with open('lais2_recombinations', 'rb') as handle:
        lais2_recombinations = pk.load(handle)
    with open('lais2_labeled', 'rb') as handle:
        lais2_labeled = pk.load(handle)



#--- Constructing Diffusion Array ---#

    #1. Compute all recombinations present in data
    #2. span np.arrays
    #3. fill np array either with count or threshold
    #4. present way to query it for long strings of

    def single_diffusion(cd_labeled):

        row_length = len(cd_labeled)

        only_id_dic = {}
        community_ids_all = []
        for window_id, window in lp_labeled.items():
            community_ids_window = []
            for community in window:
                community_ids_window.append(community[1][0])
                community_ids_all.append(community[1][0])

            only_id_dic[window_id] = community_ids_window

        column_length = max(community_ids_all)

        cd_singleDiffusion = np.zeros((row_length, column_length), dtype=int)

        for i in range(len(cd_singleDiffusion)):
            for j in range(len(cd_singleDiffusion.T)):
                if j in only_id_dic['window_{0}'.format(i * 30)]:
                    cd_singleDiffusion[i,j] = 1

        return cd_singleDiffusion

    lp_singleDiffusion = single_diffusion(lp_labeled)
    gm_singleDiffusion = single_diffusion(gm_labeled)

    kclique_singleDiffusion = single_diffusion(kclique_labeled)
    lais2_singleDiffusion = single_diffusion(lais2_labeled)

    def recombination_diffusion(cd_recombinations):

        row_length = len(cd_recombinations)

        recombinations_dic = {}
        recombinations_all = []
        for window_id, window in cd_recombinations.items():
            recombinations_window = []
            for recombination in window:
                community_id1 = recombination[1][0][1][0]
                community_id2 = recombination[1][1][1][0]
                recombinations_all.append((community_id1, community_id2))
                recombinations_window.append((community_id1, community_id2))

            recombinations_dic[window_id] = recombinations_window

        column_length = len(np.unique(recombinations_all, axis=0))

        cd_recombinationDiffusion_count = np.zeros((row_length, column_length), dtype=float)
        cd_recombinationDiffusion_threshold = np.zeros((row_length, column_length), dtype=float)

        return

    lp_recombinationDiffusion = recombination_diffusion(lp_recombinations)
    #gm_recombinationDiffusion = recombination_diffusion(gm_recombinations)

    #kclique_recombinationDiffusion = recombination_diffusion(kclique_recombinations)
    #lais2_recombinationDiffusion = recombination_diffusion(lais2_recombinations)

    #print(lp_recombinations)






#--- Recombination in Overlapping CD---#

    #1. Make Recombination Dic
    #2. Make diffusion patten array