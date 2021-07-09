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

    with open('windows_topicSim', 'rb') as handle:
        topicSim = pk.load(handle)

    with open('lp_labeled', 'rb') as handle:
        lp_labeled = pk.load(handle)
    with open('lp_clean', 'rb') as handle:
        lp_clean = pk.load(handle)


    with open('gm_labeled', 'rb') as handle:
        gm_labeled = pk.load(handle)
    with open('gm_clean', 'rb') as handle:
        gm_clean = pk.load(handle)
    '''
    with open('kclique_labeled', 'rb') as handle:
        kclique_labeled = pk.load(handle)
    with open('kclique_clean', 'rb') as handle:
        kclique_clean = pk.load(handle)

    with open('lais2_labeled', 'rb') as handle:
        lais2_labeled = pk.load(handle)
    with open('lais2_clean', 'rb') as handle:
        lais2_clean = pk.load(handle)
    '''

#--- Finding Recombination - crisp ---#

    ### Find all unique id's per window ###
    def all_unique_ids(cd_clean):

        cd_all_ids = {}
        for i in range(len(cd_clean)):
            all_ids_t = cd_clean['window_{0}'.format(i * 30)]
            all_ids_t = [item for sublist in all_ids_t for item in sublist]
            all_ids_t = np.unique(all_ids_t)
            cd_all_ids['window_{0}'.format(i * 30)] = all_ids_t

        return cd_all_ids

    # Label Propagation #
    lp_all_unique_ids = all_unique_ids(lp_clean)

    # Greedy Modularity #
    gm_all_unique_ids = all_unique_ids(gm_clean)

    # K-Clique #
    #kclique_all_unique_ids = all_unique_ids(kclique_clean)

    # Lais2 #
    #lais2_all_unique_ids = all_unique_ids(lais2_clean)


    ### Create Recombination dict ###

    def find_recombinations(cd_all_unique_ids, cd_labeled):
        cd_recombination_dic = {}

        for i in range(len(cd_all_unique_ids)-1):

            #if i * 30 == 450:
                #print(1+1)

            t = set(cd_all_unique_ids['window_{0}'.format(i * 30)])
            t_plus1 = set(cd_all_unique_ids['window_{0}'.format((i+1) * 30)])
            new_patents = t_plus1.difference(t)

            window_list = []

            for patent in new_patents:
                neighbor_list = list(topicSim['window_{0}'.format((i+1) * 30)].neighbors(patent))

                patent_list = []

                if len(neighbor_list) >=2:

                    bridge_list = []
                    already_found_community = []

                    for neighbor in neighbor_list:

                        for community in cd_labeled['window_{0}'.format((i+1) * 30)]:

                            #if set([neighbor]).issubset(community):
                            if neighbor in community[0]:
                                if community not in already_found_community:
                                    bridge_list.append((neighbor, community[1]))
                                    already_found_community.append(community)

                    if len(bridge_list) >= 2:
                        patent_list.append(bridge_list)

                if len(patent_list) != 0:
                    #window_list.append((patent, patent_list[0]))
                    patent_list_comb = list(itertools.combinations(patent_list[0], r=2))
                    for comb in patent_list_comb:
                        window_list.append([patent, comb])

            cd_recombination_dic['window_{0}'.format((i + 1) * 30)] = window_list # list of all patents that recombine  [[patent, [neighbor, neighbor]],...]
        #print(lp_recombination_dic)    # {'window_30': [], 'window_60': [], ...,  'window_300': [[287657442, [[287933459, 290076304]]], ...

        return cd_recombination_dic

    lp_recombinations = find_recombinations(lp_all_unique_ids, lp_labeled)
    #gm_recombinations = find_recombinations(gm_all_unique_ids, gm_labeled)

    # Problem:  [286551500, ((288550404, [24]), (287503390, [25])), 2, 1, 1, 1]

    #print(lp_recombinations['window_450'])

    ### Recombination Threshold ###


    def recombination_threshold(cd_recombinations, threshold):

        recombination_threshold = {}

        for window_id, window in cd_recombinations.items():
            recombination_types_plusCount = []

            if window_id == 'window_450':
                print(1+1)

            if len(window) != 0:

                total_number_patents = len(topicSim[window_id].nodes())
                recombination_types = []

                for recombination in window:
                    community_id1 = recombination[1][0][1][0]
                    community_id2 = recombination[1][1][1][0]

                    recombination_types.append((community_id1, community_id2))

                recombination_types_unique, index, count = np.unique(recombination_types, axis=0, return_counts=True, return_index=True)
                print(recombination_types_unique)
                print(index)
                print(count)
                '''
                zipped_lists = zip(index, count)
                sorted_pairs = sorted(zipped_lists)

                tuples = zip(*sorted_pairs)
                index_sorted, count_sorted = [list(tuple) for tuple in tuples]
                '''

                #fraction_sorted = count_sorted / total_number_patents
                #fraction_sorted = [x / total_number_patents for x in count_sorted]
                print(count)
                print(total_number_patents)
                fraction = [x / total_number_patents for x in count]
                print(fraction)

                #threshold_sorted = []
                threshold = []

                for i in range(len(fraction)):
                    threshold_meet = 0      # default
                    if fraction[i] >= threshold:
                        threshold_meet = 1

                    threshold.append(threshold_meet)

                for i in range(len(recombination_types_unique)):
                    recombination_types_plusCount.append(((recombination_types_unique[i]), count[i], threshold[i])) #, fraction[i]))
                    print(recombination_types_plusCount)

            recombination_threshold[window_id] = recombination_types_plusCount

        return recombination_threshold

    lp_recombination_threshold = recombination_threshold(lp_recombinations, 0.005)
    #gm_recombination_threshold = recombination_threshold(gm_recombinations, 0.05)

    #print(lp_recombination_count)      #((839, 811), 1, 0)

    print(lp_recombination_threshold['window_450']) # here it got ((24, 25), 2, 1) and ((24, 25), 1, 1) in window 450. why? I only want it once, correcly count

    ###  ###
    def enrich_recombinations_dic_with_thresholds(cd_recombinations, cd_recombination_threshold):

        recombinations_dic_with_thresholds = {}

        for window_id, window in cd_recombinations.items():

            new_window = []
            for recombination in window:

                if recombination[1][0][1][0] == 24:
                    if recombination[1][1][1][0] == 25:
                        #print(1+1)
                        1+1

                community_id1 = recombination[1][0][1][0]
                community_id2 = recombination[1][1][1][0]

                for recombination_threshold in cd_recombination_threshold[window_id]:
                    if recombination_threshold[0] == (community_id1, community_id2):

                        count = recombination_threshold[1]
                        threshold = recombination_threshold[2]

                        recombination.append(count)
                        recombination.append(threshold)
                        if len(recombination) >= 5:
                            print(1+1)

                        #print(recombination)

                new_window.append(recombination)
            recombinations_dic_with_thresholds[window_id] = new_window

        return recombinations_dic_with_thresholds

        # a dict like cd_recombination_dic, but with an additional entry per recombination list. the additional entry indicates if a threshold was meet

    lp_recombinations_enriched = enrich_recombinations_dic_with_thresholds(lp_recombinations, lp_recombination_threshold)

    '''    

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





