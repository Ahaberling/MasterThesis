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

    with open('kclique_labeled', 'rb') as handle:
        kclique_labeled = pk.load(handle)
    with open('kclique_clean', 'rb') as handle:
        kclique_clean = pk.load(handle)

    with open('lais2_labeled', 'rb') as handle:
        lais2_labeled = pk.load(handle)
    with open('lais2_clean', 'rb') as handle:
        lais2_clean = pk.load(handle)


#--- Finding Recombination ---#


    ### Create Recombination dict - crisp ###

    def find_recombinations_crisp(cd_labeled):
        cd_recombination_dic = {}

        for i in range(len(topicSim)-1):

            t = set(topicSim['window_{0}'.format(i * 30)].nodes())
            t_plus1 = set(topicSim['window_{0}'.format((i+1) * 30)].nodes())
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
                        bridge_list.sort(key=operator.itemgetter(1))
                        patent_list.append(bridge_list)

                if len(patent_list) != 0:
                    #window_list.append((patent, patent_list[0]))
                    patent_list_comb = list(itertools.combinations(patent_list[0], r=2)) # sorting order is preserved here
                    for comb in patent_list_comb:
                        window_list.append([patent, comb])

            cd_recombination_dic['window_{0}'.format((i + 1) * 30)] = window_list # list of all patents that recombine  [[patent, [neighbor, neighbor]],...]
        #print(lp_recombination_dic)    # {'window_30': [], 'window_60': [], ...,  'window_300': [[287657442, [[287933459, 290076304]]], ...

        return cd_recombination_dic

    def find_recombinations_overlapping(cd_labeled):
        cd_recombination_dic = {}

        for window_id, window in cd_labeled.items():
            recombination_list = []

            for patent in topicSim[window_id].nodes():

                patent_list = []
                for community in window:


                    if patent in community[0]:
                        patent_list.append((patent, community[1]))

                if len(patent_list) >= 2:
                    community_ids = [community[1][0] for community in cd_labeled[window_id]]

                    if len(community_ids) != len(np.unique(community_ids)):

                        print(window_id)
                        print(cd_labeled[window_id])
                        print(community_ids)
                    recombination_list.append(patent_list)      # these community ids are often identical- Shoulnd I want community ides to be unique per window????

            cd_recombination_dic[window_id] = recombination_list

        return cd_recombination_dic


    lp_recombinations = find_recombinations_crisp(lp_labeled)
    gm_recombinations = find_recombinations_crisp(gm_labeled)

    kclique_recombinations = find_recombinations_overlapping(kclique_labeled)
    lais2_recombinations = find_recombinations_overlapping(lais2_labeled)
    print(kclique_recombinations)
    #print(lp_recombinations)
    #print(gm_recombinations)

    #print(lp_recombinations['window_450'])


    ### Recombination Threshold ###


    def recombination_threshold(cd_recombinations, threshold_constant):

        recombination_threshold = {}

        for window_id, window in cd_recombinations.items():
            recombination_types_plusCount = []

           #if window_id == 'window_450':
                #print(1+1)

            if len(window) != 0:

                total_number_patents = len(topicSim[window_id].nodes())
                recombination_types = []

                for recombination in window:

                    if recombination[1][0][1][0] <= recombination[1][1][1][0]:          # probably not necessary anymore, because it was sorted in the pre function as well
                        community_id1 = recombination[1][0][1][0]                       #
                        community_id2 = recombination[1][1][1][0]
                    else:
                        community_id1 = recombination[1][1][1][0]
                        community_id2 = recombination[1][0][1][0]

                    recombination_types.append((community_id1, community_id2))

                recombination_types_unique, index, count = np.unique(recombination_types, axis=0, return_counts=True, return_index=True)
                #print(recombination_types_unique)
                #print(index)
                #print(count)
                '''
                zipped_lists = zip(index, count)
                sorted_pairs = sorted(zipped_lists)

                tuples = zip(*sorted_pairs)
                index_sorted, count_sorted = [list(tuple) for tuple in tuples]
                '''

                #fraction_sorted = count_sorted / total_number_patents
                #fraction_sorted = [x / total_number_patents for x in count_sorted]
                #print(count)
                #print(total_number_patents)

                fraction = [x / total_number_patents for x in count]
                #print(fraction)

                #threshold_sorted = []
                threshold_meet_list = []

                for i in range(len(fraction)):
                    threshold_meet = 0      # default
                    if fraction[i] >= threshold_constant:
                        threshold_meet = 1

                    threshold_meet_list.append(threshold_meet)
                #print(threshold_meet_list)

                for i in range(len(recombination_types_unique)):
                    recombination_types_plusCount.append((tuple(recombination_types_unique[i]), count[i], threshold_meet_list[i])) #, fraction[i]))
                    #print(recombination_types_plusCount)

            recombination_threshold[window_id] = recombination_types_plusCount

        return recombination_threshold

    lp_recombination_threshold = recombination_threshold(lp_recombinations, 0.005)
    gm_recombination_threshold = recombination_threshold(gm_recombinations, 0.005)

    #print(lp_recombination_threshold)      #((839, 811), 1, 0)
    #print(lp_recombination_threshold['window_450'])

    ###  ###
    def enrich_recombinations_dic_with_thresholds(cd_recombinations, cd_recombination_threshold):

        recombinations_dic_with_thresholds = {}

        for window_id, window in cd_recombinations.items():

            new_window = []
            for recombination in window:

                community_id1 = recombination[1][0][1][0]
                community_id2 = recombination[1][1][1][0]

                for recombination_threshold in cd_recombination_threshold[window_id]:
                    if recombination_threshold[0] == (community_id1, community_id2):

                        count = recombination_threshold[1]
                        threshold = recombination_threshold[2]

                        recombination.append(count)
                        recombination.append(threshold)


                new_window.append(recombination)
            recombinations_dic_with_thresholds[window_id] = new_window

        return recombinations_dic_with_thresholds

        # a dict like cd_recombination_dic, but with an additional entry per recombination list. the additional entry indicates if a threshold was meet

    lp_recombinations_enriched = enrich_recombinations_dic_with_thresholds(lp_recombinations, lp_recombination_threshold)

    #print(lp_recombinations_enriched['window_450'])




#--- Constructing Diffusion Array ---#

    #1. Compute all recombinations present in data
    #2. span np.arrays
    #3. fill np array either with count or threshold
    #4. present way to query it for long strings of 1




#--- Recombination in Overlapping CD---#

    #1. Make Recombination Dic
    #2. Make diffusion patten array