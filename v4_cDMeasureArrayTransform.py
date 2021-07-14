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

    #patent_lda_ipc = pd.read_csv('patent_lda_ipc.csv', quotechar='"', skipinitialspace=True)
    #patent_lda_ipc = patent_lda_ipc.to_numpy()

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

    with open('kclique_recombinations', 'rb') as handle:
        kclique_recombinations = pk.load(handle)
    with open('kclique_labeled', 'rb') as handle:
        kclique_labeled = pk.load(handle)
    with open('kclique_topD_dic', 'rb') as handle:
        kclique_topD_dic = pk.load(handle)

    with open('lais2_recombinations', 'rb') as handle:
        lais2_recombinations = pk.load(handle)
    with open('lais2_labeled', 'rb') as handle:
        lais2_labeled = pk.load(handle)
    with open('lais2_topD_dic', 'rb') as handle:
        lais2_topD_dic = pk.load(handle)


# ---  cleaning topD_dic ---#


    # 1. cleaning topD_dic
    #
    #   go through topD_dic.
    #   if topD vanishes in t+1, but it's community id persists:
    #       get all nodes of the community on t
    #       get their life span
    #       remove community id of every entry in topD_dic, once the life span is over


    def cleaning_topD_dic(topD_dic, cd_topD):
        merging_communities_dic = {}

        for i in range(len(topD_dic.items())):

            if i == 11:
                print(1+1)

            window_id = 'window_{0}'.format(i * 30)
            window = topD_dic[window_id]

            next_window_id = 'window_{0}'.format((i+1) * 30)
            next_window = topD_dic[next_window_id]

            swallowed_communities = []

            if i != 0:

                for topD in window.keys():

                    if topD not in next_window.keys():
                        print(window[topD])

                        for community_id in window[topD]:

                            next_window_values_flattend = [item for sublist in next_window.values() for item in sublist]

                            if community_id in next_window_values_flattend:
                                swallowed_communities.append([topD, community_id])


                # get life time
                # I want for every window a list of community_ids that are swallowd, and their death time
                if len(swallowed_communities) != 0:
                    for swallowed_community in swallowed_communities:
                        if len(swallowed_community) != 0:
                            for community in cd_topD[window_id]:

                                if swallowed_community[0] == community[1][0][0]:
                                    swallowed_community.append(community[0])
                                    break
                    print(swallowed_communities)

                    for swallowed_community in swallowed_communities:
                        community_death = False
                        j = i

                        while community_death == False:
                            members = swallowed_community[2]
                            for member in members:
                                if member not in cd_topD['window_{0}'.format(j*30)]:
                                    members.pop(member)
                            j = j + 1
                            if len(members) == 0:
                                community_death = True

                        swallowed_community.append(j)

            merging_communities_dic[window_id] = swallowed_communities

        return


    lp_topD_dic_cleaned = cleaning_topD_dic(lp_topD_dic, lp_topD)
    print(1+1)


#--- Constructing Diffusion Array ---#

    #1. Compute all recombinations present in data
    #2. span np.arrays
    #3. fill np array either with count or threshold
    #4. present way to query it for long strings of

    def single_diffusion(cd_labeled):

        row_length = len(cd_labeled)

        only_id_dic = {}
        community_ids_all = []
        for window_id, window in cd_labeled.items():
            community_ids_window = []
            for community in window:
                community_ids_window.append(community[1][0])
                community_ids_all.append(community[1][0])

            only_id_dic[window_id] = community_ids_window

        column_length = max(community_ids_all)

        singleDiffusion_array = np.zeros((row_length, column_length), dtype=int)

        for i in range(len(singleDiffusion_array)):
            for j in range(len(singleDiffusion_array.T)):
                if j in only_id_dic['window_{0}'.format(i * 30)]:
                    singleDiffusion_array[i,j] = 1

        return singleDiffusion_array

    lp_singleDiffusion = single_diffusion(lp_labeled)
    gm_singleDiffusion = single_diffusion(gm_labeled)

    kclique_singleDiffusion = single_diffusion(kclique_labeled)
    lais2_singleDiffusion = single_diffusion(lais2_labeled)

    #print(lp_singleDiffusion)
    #print(gm_singleDiffusion)
    #print(kclique_singleDiffusion)
    #print(lais2_singleDiffusion)

    def recombination_diffusion_crips(cd_recombinations):

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
        #print(recombinations_dic)

        recombinations_all.sort()
        column_length = len(np.unique(recombinations_all, axis=0))

        recombinationDiffusion_count = np.zeros((row_length, column_length), dtype=float)
        recombinationDiffusion_fraction = np.zeros((row_length, column_length), dtype=float)
        #recombinationDiffusion_threshold = np.zeros((row_length, column_length), dtype=float)

        for i in range(len(recombinationDiffusion_count)):
            for j in range(len(recombinationDiffusion_count.T)):
                window = recombinations_dic['window_{0}'.format(i * 30)]
                recombinationDiffusion_count[i,j] = window.count(recombinations_all[j])

        for i in range(len(topicSim)):
            all_nodes_in_window = topicSim['window_{0}'.format(i * 30)].nodes()
            recombinationDiffusion_fraction[i,:] = (recombinationDiffusion_count[i,:] / len(all_nodes_in_window) )

        recombinationDiffusion_threshold = np.where(recombinationDiffusion_fraction < 0.005, 0, 1)

        return recombinationDiffusion_count, recombinationDiffusion_fraction, recombinationDiffusion_threshold


    def recombination_diffusion_overlapping(cd_recombinations):

        row_length = len(cd_recombinations)

        recombinations_dic = {}
        recombinations_all = []
        for window_id, window in cd_recombinations.items():
            recombinations_window = []
            for recombination in window:
                community_id1 = recombination[1][0]
                community_id2 = recombination[1][1]
                recombinations_all.append((community_id1, community_id2))
                recombinations_window.append((community_id1, community_id2))

            recombinations_dic[window_id] = recombinations_window
        #print(recombinations_dic)

        recombinations_all.sort()
        column_length = len(np.unique(recombinations_all, axis=0))

        recombinationDiffusion_count = np.zeros((row_length, column_length), dtype=float)
        recombinationDiffusion_fraction = np.zeros((row_length, column_length), dtype=float)
        recombinationDiffusion_threshold = np.zeros((row_length, column_length), dtype=float)

        for i in range(len(recombinationDiffusion_count)):
            for j in range(len(recombinationDiffusion_count.T)):
                window = recombinations_dic['window_{0}'.format(i * 30)]
                recombinationDiffusion_count[i,j] = window.count(recombinations_all[j])

        for i in range(len(topicSim)):
            all_nodes_in_window = topicSim['window_{0}'.format(i * 30)].nodes()
            recombinationDiffusion_fraction[i,:] = (recombinationDiffusion_count[i,:] / len(all_nodes_in_window) )

        recombinationDiffusion_threshold = np.where(recombinationDiffusion_fraction < 0.005, 0, 1)

        return recombinationDiffusion_count, recombinationDiffusion_fraction, recombinationDiffusion_threshold


    lp_recombinationDiffusion_count, lp_recombinationDiffusion_fraction, lp_recombinationDiffusion_threshold = recombination_diffusion_crips(lp_recombinations)
    gm_recombinationDiffusion_count, gm_recombinationDiffusion_fraction, gm_recombinationDiffusion_threshold = recombination_diffusion_crips(gm_recombinations)

    kclique_recombinationDiffusion_count, kclique_recombinationDiffusion_fraction, kclique_recombinationDiffusion_threshold = recombination_diffusion_overlapping(kclique_recombinations)
    lais2_recombinationDiffusion_count, lais2_recombinationDiffusion_fraction, lais2_recombinationDiffusion_threshold = recombination_diffusion_overlapping(lais2_recombinations)

    np.set_printoptions(threshold=sys.maxsize)
    np.set_printoptions(precision=2)
    np.set_printoptions(suppress=True)

    print(lp_recombinationDiffusion_count[0:99,0:24])
    print(lp_recombinationDiffusion_fraction[0:99,0:24])
    print(lp_recombinationDiffusion_threshold[0:99,0:24])

    print(np.size(lp_recombinationDiffusion_count))
    print(sum(sum(lp_recombinationDiffusion_count)))



# Diffusion is not working like this. The reason is that a recombination of B & C is lost if afterwards B megres with A
# with A being the dominant community (giving both id A)

#How to solve this:

    # 2. populating diffusion array:
    #
    #   x = all t
    #   y = all recombinations (no change from what i have now)
    #   for all columns:
    #       for all rows:
    #           big_community1 = community_list where tuple[0] is present (for this window)
    #           big_community2 = community_list where tuple[1] is present (for this window)
    #           overall_count = 0
    #           for all recombinations:
    #               if recombination[0] in big_community1:
    #                   if recombination[1] in big_community2:
    #                       overall_count = overall_count + 1
    #                       break
    #               elis: recombination[0] in big_community2:
    #                   if recombination[1] in big_community1:
    #                       overall_count = overall_count + 1
    #                       break