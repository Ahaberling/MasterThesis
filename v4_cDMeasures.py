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

    patent_lda_ipc = pd.read_csv('patent_lda_ipc.csv', quotechar='"', skipinitialspace=True)
    patent_lda_ipc = patent_lda_ipc.to_numpy()

    with open('windows_topicSim', 'rb') as handle:
        topicSim = pk.load(handle)

    with open('lp_labeled', 'rb') as handle:
        lp_clean = pk.load(handle)

    with open('gm_labeled', 'rb') as handle:
        gm_clean = pk.load(handle)

    with open('kclique_labeled', 'rb') as handle:
        kclique_clean = pk.load(handle)

    with open('lais2_labeled', 'rb') as handle:
        lais2_clean = pk.load(handle)


#--- ---#

#--- Recombination - crisp ---# (semi cool, because no idea of communities are stable, yet)

    # label propagation #
    lp_window_all_ids = {}

    for i in range(0, len(lp_commu)-1):

        all_ids_t = lp_commu['window_{0}'.format(i*30)]
        all_ids_t = [item for sublist in all_ids_t for item in sublist]
        lp_window_all_ids['window_{0}'.format(i * 30)] = all_ids_t

    lp_recombination_dic = {}

    for i in range(0, len(lp_window_all_ids)-2):
        t = set(lp_window_all_ids['window_{0}'.format(i * 30)])
        t_plus1 = set(lp_window_all_ids['window_{0}'.format((i+1) * 30)])

        new_patents = t_plus1.difference(t)

        window_list = []

        for patent in new_patents:

            neighbors = list(topicSim['window_{0}'.format((i+1) * 30)].neighbors(patent))

            patent_list = []

            if len(neighbors) >=2:

                bridge_list = []
                already_found_community = []

                for neighbor in neighbors:

                    for community in lp_commu_clean['window_{0}'.format((i+1) * 30)]:

                        if set([neighbor]).issubset(community):
                            if community not in already_found_community:
                                bridge_list.append(neighbor)
                                already_found_community.append(community)

                if len(bridge_list) >= 2:
                    patent_list.append(bridge_list)

            if len(patent_list) != 0:
                window_list.append([patent, patent_list])

        lp_recombination_dic['window_{0}'.format((i + 1) * 30)] = window_list # list of all patents that recombine  [[patent, [neighbor, neighbor]],...]
    #print(lp_recombination_dic)    # {'window_30': [], 'window_60': [], ...,  'window_300': [[287657442, [[287933459, 290076304]]], ...




    # greedy_modularity #
    gm_window_all_ids = {}

    for i in range(0, len(greedy_modularity_commu_transf)-1):

        all_ids_t = greedy_modularity_commu_transf['window_{0}'.format(i*30)]
        all_ids_t = [item for sublist in all_ids_t for item in sublist]
        gm_window_all_ids['window_{0}'.format(i * 30)] = all_ids_t

    gm_recombination_dic = {}

    for i in range(0, len(gm_window_all_ids)-2):
        t = set(gm_window_all_ids['window_{0}'.format(i * 30)])
        t_plus1 = set(gm_window_all_ids['window_{0}'.format((i+1) * 30)])

        new_patents = t_plus1.difference(t)

        window_list = []

        for patent in new_patents:

            neighbors = list(topicSim['window_{0}'.format((i+1) * 30)].neighbors(patent))

            patent_list = []

            if len(neighbors) >=2:

                bridge_list = []
                already_found_community = []

                for neighbor in neighbors:

                    for community in greedy_modularity_commu_clean['window_{0}'.format((i+1) * 30)]:

                        if set([neighbor]).issubset(community):
                            if community not in already_found_community:
                                bridge_list.append(neighbor)
                                already_found_community.append(community)
                                #print(bridge_list)

                if len(bridge_list) >= 2:
                    patent_list.append(bridge_list)

            if len(patent_list) != 0:
                window_list.append([patent, patent_list])

        gm_recombination_dic['window_{0}'.format((i + 1) * 30)] = window_list # list of all patents that recombine  [[patent, [neighbor, neighbor]],...]
    #print(gm_recombination_dic)    # {'window_30': [], 'window_60': [], 'window_90': [], 'window_120': [], ...          all empty :(



# --- Recombination Thrshold  - crisp ---# (semi cool, because no idea of communities are stable, yet)

    # label propagation #
    for window_id, window in lp_recombination_dic.items():

        threshold_meet = 0       # not meet

        if len(window) != 0:
            value = len(window) / len(topicSim[window_id])

            if value >= 0.05:
                threshold_meet = 1

        lp_recombination_dic[window_id].append(threshold_meet)

    #print(lp_recombination_dic)    # {'window_30': [0], 'window_60': [0], 'window_90': [0], 'window_120': [0], 'window_150': [0],
                                #  'window_180': [0], 'window_210': [0], 'window_240': [0], 'window_270': [0],
                                # 'window_300': [[287657442, [[287933459, 290076304]]], ...

    # label propagation #
    for window_id, window in gm_recombination_dic.items():

        threshold_meet = 0  # not meet

        if len(window) != 0:
            value = len(window) / len(topicSim[window_id])
            # This can be done relative to community size instead of relative to overall size, but latter makes more sense for me right now

            if value >= 0.05:
                threshold_meet = 1

        gm_recombination_dic[window_id].append(threshold_meet)

    #print(gm_recombination_dic)  # {'window_30': [0], 'window_60': [0], 'window_90': [0], ...
    '''

#--- Recombination - overlapping ---# (semi cool, because no idea of communities are stable, yet)





