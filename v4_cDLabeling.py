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
    gm_tracing, gm_tracing_size = tracing_array(gm_max_number_community, gm_topD)

    # kclique #
    kclique_tracing, kclique_tracing_size = tracing_array(kclique_max_number_community, kclique_topD)

    # lais2 #
    lais2_tracing, lais2_tracing_size = tracing_array(lais2_max_number_community, lais2_topD)



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

                if topD == 56690516:
                    print(1+1)

                column_pos = np.where(cd_tracing[i, :] == topD)

                # if topD is present in more then 1 column of a row:
                if len(column_pos[0]) != 1:

                    prev_id_list = []

                    for column in column_pos[0]:

                        prev_topD = cd_tracing[i-1, column]

                        prev_id_list.append((prev_topD, column))

                    prev_id_list_unique = np.unique([prev_id[0] for prev_id in prev_id_list])

                    if topD in prev_id_list_unique:

                        column_pos = [prev_topD[1] for prev_topD in topD_associ['window_{0}'.format((i-1) * 30)] if prev_topD[0] == topD]

                    #elif len(np.unique(prev_id_list_unique)) == 1:
                    #    column_pos = [topD[1] for prev_topD in topD_associ['window_{0}'.format((i-1) * 30)] if prev_topD[0] == prev_id_list[0][0]]

                    else:
                        prev_topDs_withColumn = []

                        for column in column_pos[0]:
                            prev_topDs_withColumn.append((cd_tracing[i-1,column], column))

                        prev_topD_communities_withColumn = []
                        for prev_topD in prev_topDs_withColumn:
                            communities = [(community, prev_topD[1]) for community in cd_topD['window_{0}'.format((i-1) * 30)] if prev_topD[0] in community[0]]
                            if len(communities) >= 2:
                                for community in communities:
                                    prev_topD_communities_withColumn.append(community)
                            else:
                                prev_topD_communities_withColumn.append(communities)
                            # prev_topD_communities_withColumn =
                            # [ [([[28548992319, 284829374], [(286192743, 9)]], 18)], [([[288104405, 287461654], [(288007617, 10)]], 22)] ]
                            # [ [([[18754      , 280261300], [(279785951, 9)]], 115), ([[279785951, 278805366], [(279785951, 9)]], 115), ([[279785951, 49907, 359857, 277646092, 278123929, 278124396, 278805366, 277608328, 277636754], [(279785951, 9)]], 115), ([[279785951, 49907, 359857, 277646092, 278123929, 278124396, 278805366], [(279785951, 9)]], 115)], [([[18754, 50183, 113417, 189276, 360126, 55007658, 55327817, 55535358, 56567877, 276841737, 277256328, 279785951, 280021002, 280086124, 280110345, 280261300], [(279785951, 9)]], 117), ([[279785951, 12654, 49907, 359857, 55535386, 56690516, 277646092, 277719199, 278123929, 278124396, 278805366], [(279785951, 9)]], 117), ([[279785951, 49907, 359857, 277646092, 278123929, 278124396, 278805366, 277608328, 277636754], [(279785951, 9)]], 117), ([[279785951, 49907, 359857, 277646092, 278123929, 278124396, 278805366], [(279785951, 9)]], 117)], [([[18754, 50183, 113417, 189276, 360126, 55007658, 55327817, 55535358, 56567877, 276841737, 277256328, 279785951, 280021002, 280086124, 280110345, 280261300], [(279785951, 9)]], 121), ([[18754, 50183, 113417, 189276, 360126, 55007658, 55327817, 189501, 56573594, 278415848, 280050280], [(55007658, 8)]], 121)]]
                            # [ [([[285489923, 286192743], [(286192743, 9)]], 18)  ], [([[288007617, 288007607], [(288007617, 10)]], 22)] ]
                        #prev_topD_communities_withColumn = [item for sublist in prev_topD_communities_withColumn for item in sublist]

                        current_community = [community for community in cd_topD['window_{0}'.format(i * 30)] if topD in community[1][0]]

                        #Assumption. if topD is identifier for a community, the it is the identifier for only that community and not for multiple

                        current_community_degree_list = []
                        for patent in current_community[0][0]:
                            current_community_degree_list.append((patent, topicSim['window_{0}'.format(i * 30)].degree(patent)))

                        current_community_degree_list.sort(key=operator.itemgetter(1), reverse=True)

                        for candidate in current_community_degree_list:
                            checklist_inMultipleCommunities = []
                            prev_topD_communities_withColumn_mod = [prev_community[0][0] for prev_community in prev_topD_communities_withColumn]
                            for s in prev_topD_communities_withColumn:
                                print(s)
                                print(s[0])
                                print(s[0][0])
                            community_helper_list = []
                            for community_helper in prev_topD_communities_withColumn_mod:
                                if community_helper not in community_helper_list:
                                    community_helper_list.append(community_helper)

                            prev_topD_communities_withColumn_unique = community_helper_list

                            for prev_community in prev_topD_communities_withColumn_unique:
                                if candidate[0] in prev_community[0]:
                                    checklist_inMultipleCommunities.append(prev_community)
                            if len(checklist_inMultipleCommunities) == 1:
                                new_topD = checklist_inMultipleCommunities[0][1][0][0]

                                column_pos = [prev_topD[1] for prev_topD in topD_associ['window_{0}'.format((i-1) * 30)] if prev_topD[0] == new_topD]

                                break


                # CHECK IF PREVIOUS TOPDS ARE IDENTICAL. IF SO: TAKE THE SAME AS IN THE PREVIOUS DICT.
                print(topD)
                print(column_pos)
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

    # Label Propagation #
    #lp_labeled, lp_topD_associ= community_labeling(lp_tracing, lp_tracing_size, lp_topD)

    # Greedy Modularity #
    gm_labeled, gm_topD_associ = community_labeling(gm_tracing, gm_tracing_size, gm_topD)

    # Kclique #
    kclique_labeled, kclique_topD_associ = community_labeling(kclique_tracing, kclique_tracing_size, kclique_topD)

    # Lais2 #
    lais2_labeled, lais2_topD_associ = community_labeling(lais2_tracing, lais2_tracing_size, lais2_topD)



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

    # Label Propagation #
    lp_visual = visual_array(lp_tracing, lp_topD_associ)

    # Greedy Modularity #
    gm_visual = visual_array(gm_tracing, gm_topD_associ)

    # Kclique #
    kclique_visual = visual_array(kclique_tracing, kclique_topD_associ)

    # Lais2 #
    lais2_visual = visual_array(lais2_tracing, lais2_topD_associ)

#--- Saving ---#

    filename = 'lp_labeled'
    outfile = open(filename, 'wb')
    pk.dump(lp_labeled, outfile)
    outfile.close()

    filename = 'gm_labeled'
    outfile = open(filename, 'wb')
    pk.dump(gm_labeled, outfile)
    outfile.close()

    filename = 'kclique_labeled'
    outfile = open(filename, 'wb')
    pk.dump(kclique_labeled, outfile)
    outfile.close()

    filename = 'lais2_labeled'
    outfile = open(filename, 'wb')
    pk.dump(lais2_labeled, outfile)
    outfile.close()


