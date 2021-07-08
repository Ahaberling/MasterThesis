if __name__ == '__main__':

#--- Import Libraries ---#
    print('\n#--- Import Libraries ---#\n')

    import pandas as pd
    import numpy as np
    import pickle as pk

    import networkx as nx
    from cdlib import algorithms
    #import wurlitzer                   #not working for windows

    import tqdm
    import itertools
    import operator
    import os

#--- Initialization ---#
    print('\n# --- Initialization ---#\n')

    os.chdir('D:/Universitaet Mannheim/MMDS 7. Semester/Master Thesis/Outline/Data/Cleaning Robots')

    patent_lda_ipc = pd.read_csv('patent_lda_ipc.csv', quotechar='"', skipinitialspace=True)
    patent_lda_ipc = patent_lda_ipc.to_numpy()


    with open('windows_topicSim', 'rb') as handle:
        topicSim = pk.load(handle)


    with open('lp_clean', 'rb') as handle:
        lp_clean = pk.load(handle)


    with open('gm_clean', 'rb') as handle:
        gm_clean = pk.load(handle)
    
    with open('kclique_clean', 'rb') as handle:
        kclique_clean = pk.load(handle)
    
    with open('lais2_clean', 'rb') as handle:
        lais2_clean = pk.load(handle)



#--- Identify TopD degree nodes of communities ---#

    def identify_topD(cd_clean):
        cd_topD = {}

        for i in range(len(cd_clean)):
            cd_window = cd_clean['window_{0}'.format(i*30)]
            topD_window = []

            for community in cd_window:
                topD_candidate = []

                for patent in community:

                    # get all degrees of all nodes
                    topD_candidate.append((patent, topicSim['window_{0}'.format(i * 30)].degree(patent)))

                # sort and only take top D (here D = 1)
                topD_candidate.sort(key=operator.itemgetter(1), reverse=True)
                topD = topD_candidate[0:1]                              # If multiple, just take one (This can be optimized as well)
                topD_window.append(topD)

            communities_plusTopD = []

            for j in range(len(cd_window)):

                # add communities and topD tuple to new dict
                communities_plusTopD.append([cd_window[j], topD_window[j]])

            cd_topD['window_{0}'.format(i * 30)] = communities_plusTopD

        return cd_topD


    # label propagation #
    lp_topD = identify_topD(lp_clean)

    # greedy_modularity #
    gm_topD = identify_topD(gm_clean)

    # kclique #
    kclique_topD = identify_topD(kclique_clean)

    # lais2 #
    lais2_topD = identify_topD(lais2_clean)


    # FOR OVERLAPPING: CHECK IF THERE IS A WINDOW WHERE ONE TOPD IDENTIFIES TWO COMMUNITIES. IF SO: LET ONE COMMUNITY BE IDENTIFIED BY
    # THE SECOND HIGEST DEGREE THAT IS ONLY IN ONE COMMUNITY!

#---  Community Tracing ---#

    ### Identify max number of possible community id's ###

    def max_number_community(cd_topD):
        max_number = 0
        for window in cd_topD.values():
                max_number = max_number + len(window)
        return max_number


    # label propagation #
    lp_max_number_community = max_number_community(lp_topD)

    # greedy_modularity #
    gm_max_number_community = max_number_community(gm_topD)

    # kclique #
    kclique_max_number_community = max_number_community(kclique_topD)

    # lais2 #
    lais2_max_number_community = max_number_community(lais2_topD)


    ### Tracing arrays ###

    def tracing_array(max_number, cd_topD):

        # Create Arrays #
        community_tracing_array = np.zeros((len(topicSim), max_number), dtype=int)
        community_size_array = np.zeros((len(topicSim), max_number), dtype=int)

        for row in range(len(community_tracing_array)):
            current_window = cd_topD['window_{0}'.format(row * 30)]

            # Part1: Trace existing TopD's #
            if row != 0:  # skip in first row, since there is nothing to trace
                prev_window = cd_topD['window_{0}'.format((row - 1) * 30)]

                for column in range(len(community_tracing_array.T)):

                    prev_topD = community_tracing_array[row - 1, column]

                    if prev_topD == 285449519:
                        print(1+1)

                                                 # community[1][0][0] = TopD of community                             community[0] = set of id's of community
                    current_topD_candidate      = [community[1][0][0] for community in current_window if prev_topD in community[0]]
                    current_topD_candidate_size = [len(community[0]) for community in current_window if prev_topD in community[0]]

                    if len(current_topD_candidate) == 1:  # >=2 only possible for overlapping CD
                        community_tracing_array[row, column] = current_topD_candidate[0]
                        community_size_array[row, column] = current_topD_candidate_size[0]

                    else:  # (e.g. 0 because the node disappears or 2 because it is in two communities)
                        community_candidate = [community[0] for community in prev_window if prev_topD in community[0]]

                        if len(community_candidate) >= 2:
                            community_size, community_candidate = max([(len(x), x) for x in community_candidate])
                            community_candidate = [community_candidate]
                            # alternative: take the one for which prev_topk has most edges in or biggest edge weight sum in
                        if len(community_candidate) != 0:

                            all_new_candidates = []
                            for candidate in community_candidate[0]:
                                all_new_candidates.append(
                                    (candidate, topicSim['window_{0}'.format((row - 1) * 30)].degree(candidate)))

                            all_new_candidates.sort(key=operator.itemgetter(1), reverse=True)

                            for degree_candidate in all_new_candidates:

                                next_topk_candidate = [community[1][0][0] for community in current_window if degree_candidate[0] in community[0]]
                                next_topk_candidate_size = [len(community[0]) for community in current_window if degree_candidate[0] in community[0]]

                                if len(next_topk_candidate) == 1:
                                    community_tracing_array[row, column] = next_topk_candidate[0]
                                    community_size_array[row, column] = next_topk_candidate_size[0]
                                    break

            # Part2: Create new communitiy entries if tracing did not create them #
            for community in current_window:

                community_identifier = community[1][0][0]

                if community_identifier not in community_tracing_array[row]:

                    for column_id in range(len(community_tracing_array.T)):

                        if sum(community_tracing_array[:,
                               column_id]) == 0:  # RuntimeWarning: overflow encountered in long_scalars

                            community_tracing_array[row, column_id] = community[1][0][0]
                            community_size_array[row, column_id] = len(community[0])
                            break


        # Resize the arrays and exclude non relevant columns #
        for i in range(len(community_tracing_array.T)):
            if sum(community_tracing_array[:, i]) == 0:
                cutoff = i
                break

        community_tracing_array = community_tracing_array[:, 0:cutoff]
        community_size_array = community_size_array[:, 0:cutoff]

        return community_tracing_array, community_size_array


    # label propagation #
    lp_tracing, lp_tracing_size = tracing_array(lp_max_number_community, lp_topD)

    # greedy_modularity #
    #gm_tracing, gm_tracing_size = tracing_array(gm_max_number_community, gm_topD)

    # kclique #
    #kclique_tracing, kclique_tracing_size = tracing_array(kclique_max_number_community, kclique_topD)

    # lais2 #
    #lais2_tracing, lais2_tracing_size = tracing_array(lais2_max_number_community, lais2_topD)



#---  Community Labeling ---#

    def community_labeling(cd_tracing, cd_tracing_size, cd_topD):

        ### Create dict with all unique topD per window
        topD_dic = {}

        for row in range(len(cd_tracing)):
            topD_dic['window_{0}'.format(row * 30)] = np.unique(cd_tracing[row, :])[1:]

        ### Create dict that associates a topD identifier with a stable community id (column number) for each window ###
        topD_associ = {}

        for i in range(len(topD_dic)):
            tuple_list = []

            for topD in topD_dic['window_{0}'.format(i * 30)]:

                if topD == 287921238:
                    print(1+1)

                column_pos = np.where(cd_tracing[i, :] == topD)

                # if topD is present in more then 1 column of a row:
                if len(column_pos[0]) != 1:

                    prev_topD_candidates = []

                    for column in column_pos[0]:
                        prev_topD_candidates.append((cd_tracing[i-1,column], column))

                    community_candidates = []
                    for prev_topD in prev_topD_candidates:
                        communities = [(community, prev_topD[1]) for community in cd_topD['window_{0}'.format((i-1) * 30)] if prev_topD[0] in community[0]]
                        community_candidates.append(communities)

                    community_candidates_withTopD = []

                    for community in community_candidates:
                        print(community[0][0][0])
                        if topD in community[0][0][0]:
                            community_candidates_withTopD.append(community)

                    if len(community_candidates_withTopD) == 1:
                        print(community[0][1])
                        column_pos = [community[0][1]]

                    else:
                        print(1+1)
                        current_community = [community for community in cd_topD['window_{0}'.format(i * 30)] if topD in community[1][0]]

                        #Assumption. if topD is identifier for a community, the it is the identifier for only that community and not for multiple

                        next_topD_candidates = []
                        print(current_community[0][0])
                        for patent in current_community[0][0]:
                            next_topD_candidates.append((patent, topicSim['window_{0}'.format(i * 30)].degree(patent)))

                        next_topD_candidates.sort(key=operator.itemgetter(1), reverse=True)
                        next_topD_candidates = next_topD_candidates[1:]         # we already checked for topD

                        for candidate in next_topD_candidates:
                            for community in community_candidates:
                                if candidate[0] in community[0][0]:
                                    column_pos = community[1]
                                    break


                # CHECK IF PREVIOUS TOPDS ARE IDENTICAL. IF SO: TAKE THE SAME AS IN THE PREVIOUS DICT.

                tuple_list.append((topD, int(column_pos[0])))

            topD_associ['window_{0}'.format(i * 30)] = tuple_list  # list of tuples (topk, community_id)

        ### Relabel all communities in cd_topD with static community id instead of dynamic TopD ###

        cd_labeled = {}

        for window_id, window in cd_topD.items():
            new_window = []

            for community in window:
                topD = community[1][0][0]

                community_id = [tuple[1] for tuple in topD_associ[window_id] if tuple[0] == topD]

                new_window.append([community[0], community_id])

            cd_labeled[window_id] = new_window

        return cd_labeled, topD_associ

    lp_labeled, lp_topD_associ= community_labeling(lp_tracing, lp_tracing_size, lp_topD)
    #gm_labeled, gm_topD_associ = community_labeling(gm_tracing, gm_tracing_size, gm_topD)
    #kclique_labeled, kclique_topD_associ = community_labeling(kclique_tracing, kclique_tracing_size, kclique_topD)
    #lais2_labeled, lais2_topD_associ = community_labeling(lais2_tracing, lais2_tracing_size, lais2_topD)



#--- Community Visualization ---#

    def visual_array(cd_tracing, topD_associ):

        #visual_array = np.zeros((len(topicSim), max_number), dtype=int)
        visual_array = np.full((np.shape(cd_tracing)[0], np.shape(cd_tracing)[1]), 9999999)

        for row in range(len(visual_array)):
            for column in range(len(visual_array.T)):

                if cd_tracing[row, column] != 0:

                    topD = cd_tracing[row, column]

                    label_entry = [tuple[1] for tuple in topD_associ['window_{0}'.format(row * 30)] if topD == tuple[0]]
                    visual_array[row, column] = label_entry[0]

        return visual_array


    lp_visual = visual_array(lp_tracing, lp_topD_associ)

    print(lp_labeled['window_630'])
    print(lp_topD['window_630'])
    print(lp_topD_associ['window_630'], '\n')

    print(lp_labeled['window_660'])
    print(lp_topD['window_660'])
    print(lp_topD_associ['window_660'], '\n')


    print(lp_labeled['window_690'])
    print(lp_topD['window_690'])
    print(lp_topD_associ['window_690'], '\n')


    print(1+1)
#--- Saving ---#

    #filename = 'windows_lp_communities'
    #outfile = open(filename, 'wb')
    #pk.dump(lp_labeled, outfile)
    #outfile.close()


