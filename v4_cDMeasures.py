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


#--- Make sure community ids are unique in each window ---#

    def is_community_id_unique(cd_labeled):

        for window_id, window in cd_labeled.items():

            id_list = []
            for community in window:

                id_list.append(community[1][0])

            if len(id_list) != len(np.unique(id_list)):
                print('PROBLEM:')
                print(id_list)
                print(window_id)
                print(window)

        print('done')
        return

    is_community_id_unique(lp_labeled)
    is_community_id_unique(gm_labeled)
    is_community_id_unique(kclique_labeled)
    is_community_id_unique(lais2_labeled)

#--- Finding Recombination ---#


    ### Create Recombination dict - crisp ###

    def find_recombinations_crisp(cd_labeled):
        cd_recombination_dic = {}

        for i in range(len(topicSim)):
            window_list = []

            if i != 0:
                t = set(topicSim['window_{0}'.format((i-1) * 30)].nodes())
                t_plus1 = set(topicSim['window_{0}'.format(i * 30)].nodes())
                new_patents = t_plus1.difference(t)

                for patent in new_patents:
                    neighbor_list = list(topicSim['window_{0}'.format(i * 30)].neighbors(patent))

                    patent_list = []

                    if len(neighbor_list) >=2:

                        bridge_list = []
                        already_found_community = []

                        for neighbor in neighbor_list:

                            for community in cd_labeled['window_{0}'.format(i * 30)]:

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

            cd_recombination_dic['window_{0}'.format(i * 30)] = window_list # list of all patents that recombine  [[patent, [neighbor, neighbor]],...]

        #print(lp_recombination_dic)    # {'window_30': [], 'window_60': [], ...,  'window_300': [[287657442, [[287933459, 290076304]]], ...

        return cd_recombination_dic


    def find_recombinations_overlapping(cd_labeled):
        cd_recombination_dic = {}

        for window_id, window in cd_labeled.items():
            recombination_list = []

            for patent in topicSim[window_id].nodes():

                recombinations = []
                for community in window:
                    if patent in community[0]:
                        community_id = community[1]
                        recombinations.append([patent, community_id])

                if len(recombinations) >= 2:

                    community_id_list = []
                    for tuple in recombinations:

                        community_id_list.append(tuple[1][0])

                    community_id_list.sort()

                    #if len(community_id_list) >= 3:
                        #print(community_id_list)

                    community_id_list_comb = list(itertools.combinations(community_id_list, r=2))

                    for i in community_id_list_comb:
                        recombination_list.append((recombinations[0][0], i))

            helper = []
            for j in recombination_list:
                if j not in helper:
                    helper.append(j)

            cd_recombination_dic[window_id] = recombination_list

        return cd_recombination_dic

    lp_recombinations = find_recombinations_crisp(lp_labeled)
    gm_recombinations = find_recombinations_crisp(gm_labeled)

    kclique_recombinations = find_recombinations_overlapping(kclique_labeled)
    lais2_recombinations = find_recombinations_overlapping(lais2_labeled)


# check if all cd_labeled are label unique in its windows!

    ### Recombination Threshold ###


    def recombination_threshold_crisp(cd_recombinations, threshold_constant):

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

    def recombination_threshold_overlapping(cd_recombinations, threshold_constant):
        recombination_threshold = {}

        for window_id, window in cd_recombinations.items():
            recombination_types_plusCount = []

           #if window_id == 'window_450':
                #print(1+1)

            if len(window) != 0:

                total_number_patents = len(topicSim[window_id].nodes())
                recombination_types = []

                for recombination in window:

                    if recombination[1][0] <= recombination[1][1]:          # probably not necessary anymore, because it was sorted in the pre function as well
                        community_id1 = recombination[1][0]                      #
                        community_id2 = recombination[1][1]
                    else:
                        community_id1 = recombination[1][1]
                        community_id2 = recombination[1][0]

                    recombination_types.append((community_id1, community_id2))

                recombination_types_unique, index, count = np.unique(recombination_types, axis=0, return_counts=True, return_index=True)

                fraction = [x / total_number_patents for x in count]

                threshold_meet_list = []

                for i in range(len(fraction)):
                    threshold_meet = 0  # default
                    if fraction[i] >= threshold_constant:
                        threshold_meet = 1

                    threshold_meet_list.append(threshold_meet)

                for i in range(len(recombination_types_unique)):
                    recombination_types_plusCount.append((tuple(recombination_types_unique[i]), count[i], threshold_meet_list[i]))  # , fraction[i]))
                    # print(recombination_types_plusCount)

            recombination_threshold[window_id] = recombination_types_plusCount

        return recombination_threshold

    lp_recombination_threshold = recombination_threshold_crisp(lp_recombinations, 0.005)
    gm_recombination_threshold = recombination_threshold_crisp(gm_recombinations, 0.005)

    kclique_recombination_threshold = recombination_threshold_overlapping(kclique_recombinations, 0.005)
    lais2_recombination_threshold = recombination_threshold_overlapping(lais2_recombinations, 0.005)


    ###  ###
    def enrich_recombinations_dic_with_thresholds_crips(cd_recombinations, cd_recombination_threshold):

        recombinations_dic_with_thresholds = {}

        for window_id, window in cd_recombinations.items():

            new_window = []
            for recombination in window:
                #print(recombination)
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

    def enrich_recombinations_dic_with_thresholds_overlapping(cd_recombinations, cd_recombination_threshold):
        recombinations_dic_with_thresholds = {}

        for window_id, window in cd_recombinations.items():

            new_window = []
            for recombination in window:
                community_id1 = recombination[1][0]
                community_id2 = recombination[1][1]

                recombination_value = list(recombination)

            for recombination_threshold in cd_recombination_threshold[window_id]:
                if recombination_threshold[0] == (community_id1, community_id2):

                    count = recombination_threshold[1]
                    threshold = recombination_threshold[2]

                    recombination_value.append(count)
                    recombination_value.append(threshold)

                new_window.append(recombination_value)
            recombinations_dic_with_thresholds[window_id] = new_window

        return recombinations_dic_with_thresholds

        # a dict like cd_recombination_dic, but with an additional entry per recombination list. the additional entry indicates if a threshold was meet

    lp_recombinations_enriched = enrich_recombinations_dic_with_thresholds_crips(lp_recombinations, lp_recombination_threshold)
    gm_recombinations_enriched = enrich_recombinations_dic_with_thresholds_crips(gm_recombinations, gm_recombination_threshold)


    kclique_recombinations_enriched = enrich_recombinations_dic_with_thresholds_overlapping(kclique_recombinations, kclique_recombination_threshold)
    lais2_recombinations_enriched = enrich_recombinations_dic_with_thresholds_overlapping(lais2_recombinations, lais2_recombination_threshold)



#--- Saving ---#

    filename = 'lp_recombinations'
    outfile = open(filename, 'wb')
    pk.dump(lp_recombinations_enriched, outfile)
    outfile.close()

    filename = 'gm_recombinations'
    outfile = open(filename, 'wb')
    pk.dump(gm_recombinations_enriched, outfile)
    outfile.close()

    filename = 'kclique_recombinations'
    outfile = open(filename, 'wb')
    pk.dump(kclique_recombinations_enriched, outfile)
    outfile.close()

    filename = 'lais2_recombinations'
    outfile = open(filename, 'wb')
    pk.dump(lais2_recombinations_enriched, outfile)
    outfile.close()

