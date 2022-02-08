import numpy as np 
import tqdm
import itertools
import operator
import copy

import scipy.signal as sciSignal

from cdlib import algorithms
from cdlib import evaluation

import networkx as nx
import networkx.algorithms.community as nx_comm



###--- Class Direct_Measurement ---### ------------------------------------------------------------------------

class Direct_Measurement:

    @staticmethod
    def extract_knowledgeComponent_per_window(slidingWindow_dict, kC, unit): # kC = knowledgeComponent

        if kC == 'topic':
            position = np.r_[range(9, 25, 2)]

        elif kC == 'ipc':
            position = np.r_[range(25, np.shape(slidingWindow_dict['window_0'])[1], 3)]

        else:
            raise Exception("kC must be string value 'topic' or 'ipc'")

        slidingWindow_kC_unit = {}
        pbar = tqdm.tqdm(total=len(slidingWindow_dict))

        i = 0
        for window_id, window in slidingWindow_dict.items():

            kC_list = []

            for patent in window:

                if kC == 'topic':
                    kC_inPatent = [int(x) for x in patent[position] if x != None]  # nan elimination
                elif kC == 'ipc':
                    kC_inPatent = [x for x in patent[position] if x != None]
                else:
                    raise Exception("kC must be string value 'topic' or 'ipc'")


                kC_inPatent = np.unique(kC_inPatent)

                if unit == 1:
                    kC_list.append(tuple(kC_inPatent))

                else:
                    kC_list.append(list(itertools.combinations(kC_inPatent, r=unit)))

            kC_list = [item for sublist in kC_list for item in sublist]

            slidingWindow_kC_unit[window_id] = kC_list

            pbar.update(1)
            i = i + 1
        pbar.close()

        return slidingWindow_kC_unit

    @staticmethod
    def create_pattern_array(knowledgeComponent_dict):

        row_list = []
        column_list = []

        for window_id, window in knowledgeComponent_dict.items():
            row_list.append(window_id)
            column_list.append(window)

        column_list = [item for sublist in column_list for item in sublist]
        column_list, column_list_counts = np.unique(column_list, return_counts=True, axis=0)

        if np.issubdtype(type(column_list[0]), np.integer) or np.issubdtype(type(column_list[0]), np.str_):
            column_list.sort()

        else:
            ind = np.lexsort((column_list[:, 1], column_list[:, 0]))  # if 'unite' exceeds tuples, it is not sorted once again here
            # This sort would be redundant non the lest in the current version
            column_list = column_list[ind]

        pattern_array = np.zeros((len(row_list), len(column_list)), dtype=int)

        if np.issubdtype(type(column_list[0]), np.integer) or np.issubdtype(type(column_list[0]), np.str_):
            pbar = tqdm.tqdm(total=len(row_list))
            c_row = 0
            for row in row_list:
                c_column = 0

                for column in column_list:
                    if column in knowledgeComponent_dict[row]:
                        pattern_array[c_row, c_column] = list(knowledgeComponent_dict[row]).count(column)

                    c_column = c_column + 1
                c_row = c_row + 1
                pbar.update(1)

            pbar.close()

        else:
            pbar = tqdm.tqdm(total=len(row_list))
            c_row = 0
            for row in row_list:
                c_column = 0

                for column in column_list:

                    if tuple(column) in knowledgeComponent_dict[row]:
                        pattern_array[c_row, c_column] = list(knowledgeComponent_dict[row]).count(tuple(column))

                    c_column = c_column + 1
                c_row = c_row + 1
                pbar.update(1)
            pbar.close()

        return pattern_array, column_list

    

###--- Class Community_Measurement ---### ------------------------------------------------------------------------

class Community_Measurement:

    @staticmethod
    def detect_communities(patentProject_graphs, cD_algorithm, weight_bool=None, k_clique_size=None):

        community_dict = {}
        modularity_dict = {}

        if cD_algorithm == 'label_propagation':

            if weight_bool == True:
                pbar = tqdm.tqdm(total=len(patentProject_graphs))
                for window_id, window in patentProject_graphs.items():
                    lp = nx.algorithms.community.label_propagation.asyn_lpa_communities(window, weight='weight', seed=123)
                    community_dict[window_id] = list(lp)
                    modularity_dict[window_id] = nx_comm.modularity(window, nx.algorithms.community.label_propagation.asyn_lpa_communities(window, weight='weight', seed=123))
                    pbar.update(1)
                pbar.close()

            else:
                pbar = tqdm.tqdm(total=len(patentProject_graphs))
                for window_id, window in patentProject_graphs.items():
                    lp = nx.algorithms.community.label_propagation.asyn_lpa_communities(window, seed=123)
                    community_dict[window_id] = list(lp)
                    pbar.update(1)
                pbar.close()

        elif cD_algorithm == 'greedy_modularity':
            pbar = tqdm.tqdm(total=len(patentProject_graphs))
            for window_id, window in patentProject_graphs.items():
                gm = algorithms.greedy_modularity(window)
                community_dict[window_id] = gm.to_node_community_map()
                modularity_dict[window_id] = evaluation.newman_girvan_modularity(window, gm)
                pbar.update(1)
            pbar.close()

        elif cD_algorithm == 'k_clique':
            pbar = tqdm.tqdm(total=len(patentProject_graphs))
            for window_id, window in patentProject_graphs.items():
                kclique = algorithms.kclique(window, k=k_clique_size)  # no seed needed i think
                community_dict[window_id] = kclique.to_node_community_map()           # link_mod 0.048218915888596885
                modularity_dict[window_id] = evaluation.newman_girvan_modularity(window, kclique)
                pbar.update(1)
            pbar.close()

        elif cD_algorithm == 'lais2':
            pbar = tqdm.tqdm(total=len(patentProject_graphs))
            for window_id, window in patentProject_graphs.items():
                lais2 = algorithms.lais2(window) 
                community_dict[window_id] = lais2.to_node_community_map()           
                modularity_dict[window_id] = modularity_dict[window_id] = evaluation.newman_girvan_modularity(window, lais2)
                pbar.update(1)
            pbar.close()

        else:
            raise Exception("cD_algorithm must be 'label_propagation','greedy_modularity','k_clique' or 'lais2'")

        return community_dict, modularity_dict

    @staticmethod
    def align_cD_dataStructure(community_dict, cD_algorithm):

        if cD_algorithm == 'label_propagation':
            community_dict_transf = community_dict

        elif cD_algorithm == 'greedy_modularity':
            community_dict_transf = {}

            for window_id, window in community_dict.items():
                community_list = []
                focal_commu = []
                c = 0

                for patent_id, community_id in window.items():
                    if community_id[0] == c:
                        focal_commu.append(patent_id)
                    else:
                        community_list.append(focal_commu)
                        focal_commu = []
                        focal_commu.append(patent_id)
                        c = c + 1

                community_dict_transf[window_id] = community_list

        elif cD_algorithm == 'k_clique':

            community_dict_transf = {}

            for window_id, window in community_dict.items():
                community_list = []
                max_commu_counter = []

                for patent_id, community_id in window.items():
                    max_commu_counter.append(len(community_id))

                if len(max_commu_counter) >= 1:
                    max_commu_counter = max(max_commu_counter)

                    for j in range(max_commu_counter + 1):
                        focal_commu = []

                        for patent_id, community_id in window.items():
                            if j in community_id:
                                focal_commu.append(patent_id)

                        community_list.append(focal_commu)

                else:
                    community_list.append([])

                community_dict_transf[window_id] = community_list

        elif cD_algorithm == 'lais2':
            community_dict_transf = {}

            for window_id, window in community_dict.items():
                community_list = []
                max_commu_counter = []

                for patent_id, community_id in window.items():
                    max_commu_counter.append(len(community_id))

                max_commu_counter = max(max_commu_counter)

                for j in range(max_commu_counter + 1):
                    focal_commu = []

                    for patent_id, community_id in window.items():
                        if j in community_id:
                            focal_commu.append(patent_id)

                    community_list.append(focal_commu)
                community_dict_transf[window_id] = community_list

        else:
            raise Exception("cD_algorithm must be 'label_propagation','greedy_modularity','k_clique' or 'lais2'")

        return community_dict_transf

    @staticmethod
    def community_cleaning(community_dict_transf, min_community_size):
        community_dict_clean = {}
        communities_removed_list = []
        for window_id, window in community_dict_transf.items():
            community_dict_clean[window_id] = [x for x in window if len(x) >= min_community_size]
            communities_removed_list.append([len(x) for x in window if len(x) <= min_community_size - 1])
        communities_removed_list = len([item for sublist in communities_removed_list for item in sublist])
        return community_dict_clean, communities_removed_list

    @staticmethod
    def identify_topDegree(community_dict_clean, patentProject_graphs):
        community_dict_topD = {}

        for i in range(len(community_dict_clean)):
            cd_window = community_dict_clean['window_{0}'.format(i * 30)]
            topD_window = []

            # First find nodes that are in more than one community, they shall not be topD (otherwise identifying recombination gets iffy)
            multi_community_patents = []

            for node in patentProject_graphs['window_{0}'.format(i * 30)].nodes():
                community_counter = []
                for community in cd_window:
                    if node in community:
                        community_counter.append(node)

                if len(community_counter) >= 2:
                    multi_community_patents.append(node)

            for community in cd_window:
                topD_candidate = []

                for patent in community:

                    if patent not in multi_community_patents:
                        # get all degrees of all nodes
                        topD_candidate.append(
                            (patent, patentProject_graphs['window_{0}'.format(i * 30)].degree(patent)))

                if len(topD_candidate) == 0:
                    for patent in community:
                        # get all degrees of all nodes
                        topD_candidate.append(
                            (patent, patentProject_graphs['window_{0}'.format(i * 30)].degree(patent)))

                # sort and only take top D (here D = 1)
                topD_candidate.sort(key=operator.itemgetter(1), reverse=True)
                topD = topD_candidate[0:1]  # If multiple, just take one (This can be optimized as well)
                topD_window.append(topD)

            communities_plusTopD = []

            for j in range(len(cd_window)):
                # add communities and topD tuple to new dict
                communities_plusTopD.append([cd_window[j], topD_window[j]])

            community_dict_topD['window_{0}'.format(i * 30)] = communities_plusTopD

        return community_dict_topD

    @staticmethod
    def merging_completly_overlapping_communities(community_dict_topD):
        merged_community_dic = {}

        for window_id, window in community_dict_topD.items():

            new_window = []
            all_topD = []
            for community in window:
                all_topD.append(community[1][0][0])

            communites_unique, communites_unique_index, communities_unique_count = np.unique(all_topD, return_index=True, return_counts=True)

            communites_non_unique = communites_unique[communities_unique_count >= 2]
            communites_unique_exclusive = communites_unique[communities_unique_count == 1]

            merged_communities = []
            for community_non_unique in communites_non_unique:

                non_unique_pos = np.where(all_topD == community_non_unique)

                toBeMerged = []
                for pos in non_unique_pos[0]:
                    toBeMerged.append(window[pos])

                toBeMerged_len = [(community, len(community[0])) for community in toBeMerged]
                toBeMerged_len.sort(key=operator.itemgetter(1), reverse=True)
                toBeMerged_sort = [community[0] for community in toBeMerged_len]
                merged_community = toBeMerged_sort[0]

                for community in toBeMerged_sort[1:]:
                    for patent in community[0]:
                        if patent not in merged_community[0]:
                            merged_community[0].append(patent)

                merged_communities.append(merged_community)

            normal_communities = []

            for community_unique in communites_unique_exclusive:
                unique_pos = np.where(all_topD == community_unique)

                normal_communities.append(window[unique_pos[0][0]])

            new_window.append(normal_communities)

            if len(merged_communities) != 0:
                new_window.append(merged_communities)

            merged_community_dic[window_id] = new_window[0]

        return merged_community_dic

    @staticmethod
    def max_number_community(cd_topD):
        max_number = 0
        for window in cd_topD.values():
                max_number = max_number + len(window)
        return max_number

    @staticmethod
    def create_tracing_array(max_number_community, community_dict_topD, patentProject_graphs):

        # Create Arrays #
        community_tracing_array = np.zeros((len(patentProject_graphs), max_number_community), dtype=int)

        for row in range(len(community_tracing_array)):
            current_window = community_dict_topD['window_{0}'.format(row * 30)]

            # Part1: Trace existing TopD's #
            if row != 0:  # skiping first row, since there is nothing to trace
                prev_window = community_dict_topD['window_{0}'.format((row - 1) * 30)]

                for column in range(len(community_tracing_array.T)):

                    prev_topD = community_tracing_array[row - 1, column]

                    # community[1][0][0] = TopD of community                             community[0] = set of id's of community
                    current_topD_candidate = [community[1][0][0] for community in current_window if prev_topD in community[0]]

                    if len(current_topD_candidate) == 1:  # >=2 only possible for overlapping CD
                        community_tracing_array[row, column] = current_topD_candidate[0]

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
                                    (candidate,
                                     patentProject_graphs['window_{0}'.format((row - 1) * 30)].degree(candidate)))

                            all_new_candidates.sort(key=operator.itemgetter(1), reverse=True)

                            for degree_candidate in all_new_candidates:

                                next_topk_candidate = [community[1][0][0] for community in current_window if
                                                       degree_candidate[0] in community[0]]

                                if len(next_topk_candidate) == 1:
                                    community_tracing_array[row, column] = next_topk_candidate[0]
                                    break

            # Part2: Create new communitiy entries if tracing did not create them #
            for community in current_window:

                community_identifier = community[1][0][0]

                if community_identifier not in community_tracing_array[row]:

                    for column_id in range(len(community_tracing_array.T)):

                        if sum(community_tracing_array[:,
                               column_id]) == 0:  # RuntimeWarning: overflow encountered in long_scalars

                            community_tracing_array[row, column_id] = community[1][0][0]
                            break

        # Resize the arrays and exclude non relevant columns #
        for i in range(len(community_tracing_array.T)):
            if sum(community_tracing_array[:, i]) == 0:
                cutoff = i
                break

        community_tracing_array = community_tracing_array[:, 0:cutoff]

        return community_tracing_array

    @staticmethod
    def community_labeling(tracingArray, community_dict_topD, patentProject_graphs):

        ### Create dict with all topD per window and the community sequences aka columns they are associated with

        topD_dic = {}

        for row in range(len(tracingArray)):

            topD_pos = {}
            for j in range(len(tracingArray[row, :])):

                # find for every topD the community sequences (columns) topD is identifying.
                if tracingArray[row, j] != 0:
                    if tracingArray[row, j] in topD_pos.keys():
                        topD_pos[tracingArray[row, j]].append(j)
                    else:
                        topD_pos[tracingArray[row, j]] = [j]

            topD_dic['window_{0}'.format(row * 30)] = topD_pos

        ### Create dict that associates a topD identifier with a stable community id (column number) for each window ###
        topD_associ = {}

        for i in range(len(topD_dic)):

            tuple_list = []

            for topD, column_pos in topD_dic['window_{0}'.format(i * 30)].items():

                if len(column_pos) != 1:  # this can never be 0


                    prev_id_list = []

                    for column in column_pos:
                        prev_topD = tracingArray[i - 1, column]

                        prev_id_list.append((prev_topD, column))

                    prev_id_list_unique = np.unique([prev_id[0] for prev_id in prev_id_list])

                    # wouldnt there be cases where 0 is added to the list as well? No, this is never 0 because: new columns (community sequences)
                    # are only opened, if a topD appears, that is not linkable to previous topDs. if one of the pos in column_pos is linked to a previous id
                    # then the other one is as well. Cases in which two identical topDs appear in two columns/sequences are not possible, because they would
                    # just be subsumed in one sequence/one topD
                    # THIS MEANS SOME COMMUNITIES MUST HAVE MERGED.
                    # Now has to be decided how to label this merged community. Which column id is chosen?
                    # Take the previous topDs and ... see below

                    # if the current multiple occuring topD was already a topD in the previous window, then simply take the same column id as in the last window
                    # this is never ambigious, because topDs are always only associated with one column/ sequence id per window.
                    if topD in prev_id_list_unique:

                        column_pos = [prev_topD[1] for prev_topD in topD_associ['window_{0}'.format((i - 1) * 30)] if
                                      prev_topD[0] == topD]
                        # first row does not have to be accounted for, because ambigious topDs could only occure in Lais2,
                        # but 'After all core are identified, communities with the same core are merged.'

                    # case: ambigious topD, but topD not observed in any previous sequence points
                    else:
                        multi_community_edgeCase = []
                        prev_topD_communities_withColumn = []

                        # for prev_topD in prev_topDs_withColumn:
                        for prev_topD in prev_id_list:
                            communities = [(community, prev_topD[1]) for community in
                                           community_dict_topD['window_{0}'.format((i - 1) * 30)] if
                                           prev_topD[0] == community[1][0][0]]
                            # if selected previous topD was in more then one community...
                            # this is never the case, if two communities would have the same topD, then either antohe one would have been chosen
                            # or in the case for total subsets in lais2, the communities would have been merged -> one community

                            prev_topD_communities_withColumn.append(communities)

                        # 'current_community' is always just 1 community
                        current_community = [community for community in community_dict_topD['window_{0}'.format(i * 30)]
                                             if topD in community[1][0]]

                        current_community_degree_list = []
                        for patent in current_community[0][0]:
                            current_community_degree_list.append(
                                (patent, patentProject_graphs['window_{0}'.format(i * 30)].degree(patent)))

                        current_community_degree_list.sort(key=operator.itemgetter(1), reverse=True)

                        for candidate in current_community_degree_list:
                            checklist_inMultipleCommunities = []

                            # get all communities of previous topDs
                            prev_topD_communities_withColumn_mod = [prev_community[0][0] for prev_community in
                                                                    prev_topD_communities_withColumn]

                            # get only the unqiue ones.
                            community_helper_list = []
                            for community_helper in prev_topD_communities_withColumn_mod:
                                if community_helper not in community_helper_list:
                                    community_helper_list.append(community_helper)
                            prev_topD_communities_withColumn_unique = community_helper_list

                            # check if candidate node is in one and only one community
                            for prev_community in prev_topD_communities_withColumn_unique:
                                if candidate[0] in prev_community[0]:
                                    checklist_inMultipleCommunities.append(prev_community)

                            # if the node is in one and only one of the previous communities of topD, then get the topD of this previous community
                            # and the previous column position. This one is used then.
                            if len(checklist_inMultipleCommunities) == 1:

                                # if this is the case:
                                new_topD = checklist_inMultipleCommunities[0][1][0][0]

                                column_pos = [prev_topD[1] for prev_topD in
                                              topD_associ['window_{0}'.format((i - 1) * 30)] if
                                              prev_topD[0] == new_topD]

                                break

                            elif len(checklist_inMultipleCommunities) >= 2:
                                multi_community_edgeCase.append(checklist_inMultipleCommunities)

                                
                        # multi_community_edgeCase_unique now equals all communties that inhibit node candidates that are not
                        # unique to one community
                        if len(column_pos) != 1:
                            multi_community_edgeCase = [item for sublist in multi_community_edgeCase for item in
                                                        sublist]

                            # get only the unique communities
                            multi_community_edgeCase_unique = []
                            for community in multi_community_edgeCase:
                                if community not in multi_community_edgeCase_unique:
                                    multi_community_edgeCase_unique.append(community)

                            multi_community_edgeCase_count = []

                            # get the most frequent communities in this list
                            for unique_item in multi_community_edgeCase_unique:
                                c = 0
                                for item in multi_community_edgeCase:
                                    if unique_item == item:
                                        c = c + 1

                                multi_community_edgeCase_count.append((unique_item, c))

                            multi_community_edgeCase_count.sort(key=operator.itemgetter(1), reverse=True)

                            # chose column of the community with the highest count as new label
                            new_topD = multi_community_edgeCase_count[0][0][1][0][0]

                            column_pos = [prev_topD[1] for prev_topD in topD_associ['window_{0}'.format((i - 1) * 30)]
                                          if prev_topD[0] == new_topD]

                tuple_list.append((topD, int(column_pos[0])))

            topD_associ['window_{0}'.format(i * 30)] = tuple_list  # list of tuples (topD, community_id)

            
        ### Relabel all communities in cd_topD with static community id instead of dynamic TopD ###
        cd_labeled = {}

        for window_id, window in community_dict_topD.items():
            new_window = []

            for community in window:
                topD = community[1][0][0]

                community_id = [tuple[1] for tuple in topD_associ[window_id] if tuple[0] == topD]

                new_window.append([community[0], community_id])

            cd_labeled[window_id] = new_window

        return cd_labeled, topD_associ, topD_dic

    @staticmethod
    def is_community_id_unique(cd_labeled):

        for window_id, window in cd_labeled.items():

            id_list = []
            for community in window:
                id_list.append(community[1][0])

            if len(id_list) != len(np.unique(id_list)):
                raise Exception("Community labeling faulty: {} contains non-unique community id's".format(window_id))

        return

    @staticmethod
    def create_visualization_array(tracingArray, topD_communityID_association_perWindow):

        visual_array = np.full((np.shape(tracingArray)[0], np.shape(tracingArray)[1]), 9999999)

        for row in range(len(visual_array)):
            for column in range(len(visual_array.T)):

                if tracingArray[row, column] != 0:
                    topD = tracingArray[row, column]

                    label_entry = [tuple[1] for tuple in
                                   topD_communityID_association_perWindow['window_{0}'.format(row * 30)] if
                                   topD == tuple[0]]
                    visual_array[row, column] = label_entry[0]

        return visual_array

    @staticmethod
    def find_recombinations_crisp(community_dict_labeled, patentProject_graphs):
        cd_recombination_dic = {}  # community_dict_labeled = window: [{member ids}, [community id]]

        for i in range(len(patentProject_graphs)):
            window_list = []

            if i != 0:
                t_minus1 = set(patentProject_graphs['window_{0}'.format((i - 1) * 30)].nodes())
                t = set(patentProject_graphs['window_{0}'.format(i * 30)].nodes())
                new_patents = t.difference(t_minus1)

                for patent in new_patents:
                    neighbor_list = list(patentProject_graphs['window_{0}'.format(i * 30)].neighbors(patent))

                    patent_list = []

                    if len(neighbor_list) >= 2:

                        bridge_list = []
                        already_found_community = []

                        for neighbor in neighbor_list:

                            for community in community_dict_labeled['window_{0}'.format((i - 1) * 30)]:

                                if neighbor in community[0]:
                                    if community not in already_found_community:
                                        bridge_list.append((neighbor, community[1]))
                                        already_found_community.append(community)

                        if len(bridge_list) >= 2:
                            bridge_list.sort(key=operator.itemgetter(1))
                            patent_list.append(bridge_list)

                    if len(patent_list) != 0:
                        patent_list_comb = list(
                            itertools.combinations(patent_list[0], r=2))  # sorting order is preserved here
                        for comb in patent_list_comb:
                            window_list.append([patent, comb])

            cd_recombination_dic['window_{0}'.format(
                i * 30)] = window_list  # list of all patents that recombine  [[patent, [neighbor, neighbor]],...]

        return cd_recombination_dic

    @staticmethod
    def find_recombinations_overlapping(community_dict_labeled, patentProject_graphs):
        cd_recombination_dic = {}

        for window_id, window in community_dict_labeled.items():
            recombination_list = []

            for patent in patentProject_graphs[window_id].nodes():

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

                    community_id_list_comb = list(itertools.combinations(community_id_list, r=2))

                    for i in community_id_list_comb:
                        recombination_list.append((recombinations[0][0], i))

            helper = []
            for j in recombination_list:
                if j not in helper:
                    helper.append(j)

            cd_recombination_dic[window_id] = recombination_list

        return cd_recombination_dic

    @staticmethod
    def creat_dict_topicDistriburionOfCommunities(community_dict_labeled, patent_lda_ipc):

        community_ids_all = []
        for window_id, window in community_dict_labeled.items():
            for community in window:
                community_ids_all.append(community[1][0])

        column_unique = np.unique(community_ids_all)
        column_unique.sort()

        # get biggest community each window:

        community_size_dic = {}
        for window_id, window in community_dict_labeled.items():
            size_list = []
            for community in window:
                size_list.append(len(community[0]))
            if size_list != []:
                community_size_dic[window_id] = max(size_list)
            else:
                community_size_dic[window_id] = 0

        community_topicDist_dic = {}

        for window_id, window in community_dict_labeled.items():
            window_list = []

            for community in window:
                community_topics = np.zeros((len(community[0]), 330))
                topic_list = []

                for patent in community[0]:
                    paten_pos = np.where(patent_lda_ipc[:, 0] == patent)
                    topic_list.append(patent_lda_ipc[paten_pos[0][0]][9:25])

                topic_list = [item for sublist in topic_list for item in sublist]
                topic_list = [x for x in topic_list if x == x]

                for i in range(0, len(topic_list), 2):
                    for row in range(len(community_topics)):  # for all patents in the community
                        if community_topics[row, int(topic_list[i])] == 0:
                            community_topics[row, int(topic_list[i])] = topic_list[i + 1]
                            break

                community_topics = np.sum(community_topics, axis=0)
                window_list.append([community[1][0], list(community_topics)])
            community_topicDist_dic[window_id] = window_list
        return community_topicDist_dic

    @staticmethod
    def create_dict_communityTopicAssociation(community_topicDist_dic):

        # create dic with: each window, list of tuple with (communityID, highest topic)

        community_topTopic_dic = {}
        confidence_list = []
        for window_id, window in community_topicDist_dic.items():
            community_list = []
            for community in window:
                topTopic = max(community[1])
                topicSum = sum(community[1])
                confidence = topTopic / topicSum

                topTopic_index = community[1].index(max(community[1]))
                community_list.append([community[0], topTopic_index, round(confidence, 2)])

                confidence_list.append(confidence)

            community_topTopic_dic[window_id] = community_list

        avg_confidence = sum(confidence_list) / len(confidence_list)

        return community_topTopic_dic, avg_confidence

    @staticmethod
    def create_diffusionArray_Topics(communityTopicAssociation_dict, maxTopicNumber):
        topic_diffusion_array = np.zeros((len(communityTopicAssociation_dict), maxTopicNumber), dtype=int)
        # community id, topic id, confidence
        for i in range(len(topic_diffusion_array)):
            window = communityTopicAssociation_dict['window_{}'.format(i * 30)]

            for j in range(len(topic_diffusion_array.T)):

                if any(j == community[1] for community in window) == True:
                    topic_diffusion_array[i, j] = 1
        return topic_diffusion_array, list(range(maxTopicNumber))

    @staticmethod
    def created_recombination_dict_Topics_crisp(communityTopicAssociation_dict, recombination_dict):
        recombination_dict_mod_lp = {}
        for i in range(
                len(recombination_dict)):
            new_window = []
            for recombination in recombination_dict['window_{}'.format(i * 30)]:
                new_recombination = []
                if i != 0:
                    for community in communityTopicAssociation_dict['window_{}'.format((i - 1) * 30)]:
                        if recombination[1][0][1][0] == community[0]:
                            new_recombination.append(community[1])
                        if recombination[1][1][1][0] == community[0]:
                            new_recombination.append(community[1])
                        if len(new_recombination) >= 2:
                            break
                new_recombination.sort()
                new_window.append(tuple(new_recombination))
            recombination_dict_mod_lp['window_{}'.format(i * 30)] = new_window

        return recombination_dict_mod_lp

    @staticmethod
    def created_recombination_dict_Topics_overlap(communityTopicAssociation_dict, recombination_dict):
        recombination_dict_mod_lp = {}
        for i in range(len(recombination_dict)):
            new_window = []
            for recombination in recombination_dict['window_{}'.format(i * 30)]:
                new_recombination = []
                for community in communityTopicAssociation_dict[
                    'window_{}'.format(i * 30)]:
                    if recombination[1][0] == community[0]:
                        new_recombination.append(community[1])
                    if recombination[1][1] == community[0]:
                        new_recombination.append(community[1])
                    if len(new_recombination) >= 2:
                        break
                new_recombination.sort()
                new_window.append(tuple(new_recombination))
            recombination_dict_mod_lp['window_{}'.format(i * 30)] = new_window

        return recombination_dict_mod_lp

    @staticmethod
    def doubleCheck_recombination_dict_Topics_crisp(recombination_dict_mod, recombination_dict):
        for i in range(len(recombination_dict_mod)):
            if len(recombination_dict_mod['window_{}'.format(i * 30)]) != len(
                    recombination_dict['window_{}'.format(i * 30)]):
                raise Exception("At least one window is not consistent")

        for i in range(len(recombination_dict_mod)):
            helper = []
            for recombination in recombination_dict_mod['window_{}'.format(i * 30)]:
                count = recombination_dict_mod['window_{0}'.format(i * 30)].count(recombination)
                if count >= 2:
                    new_entry = (i * 30, count, recombination)
                    if new_entry not in helper:
                        helper.append(new_entry)
            helper2 = []
            helper3 = []
            for recombination2 in recombination_dict['window_{}'.format(i * 30)]:
                helper2.append([recombination2[1][0][1][0], recombination2[1][1][1][0]])
            for recombination2 in recombination_dict['window_{}'.format(i * 30)]:
                recombination2 = [recombination2[1][0][1][0], recombination2[1][1][1][0]]
                count2 = helper2.count(recombination2)
                if count2 >= 2:
                    new_entry2 = (i * 30, count2, recombination2)
                    if new_entry2 not in helper3:
                        helper.append(new_entry2)
                        helper3.append(new_entry2)
        return

    @staticmethod
    def doubleCheck_recombination_dict_Topics_overlap(recombination_dict_mod, recombination_dict):
        for i in range(len(recombination_dict_mod)):
            if len(recombination_dict_mod['window_{}'.format(i * 30)]) != len(
                    recombination_dict['window_{}'.format(i * 30)]):
                raise Exception("At least one window is not consistent")

        for i in range(len(recombination_dict_mod)):
            helper = []
            for recombination in recombination_dict_mod['window_{}'.format(i * 30)]:
                count = recombination_dict_mod['window_{0}'.format(i * 30)].count(recombination)
                if count >= 2:
                    new_entry = (i * 30, count, recombination)
                    if new_entry not in helper:
                        helper.append(new_entry)
            helper2 = []
            helper3 = []
            for recombination2 in recombination_dict['window_{}'.format(i * 30)]:
                helper2.append([recombination2[1][0], recombination2[1][1]])
            for recombination2 in recombination_dict['window_{}'.format(i * 30)]:
                recombination2 = [recombination2[1][0], recombination2[1][1]]
                count2 = helper2.count(recombination2)
                if count2 >= 2:
                    new_entry2 = (i * 30, count2, recombination2)
                    if new_entry2 not in helper3:
                        helper.append(new_entry2)
                        helper3.append(new_entry2)
        return

    @staticmethod
    def create_recombinationArray_Topics(recombination_dict_Topics):
        all_recombinations = []
        for window_id, window in recombination_dict_Topics.items():
            for recombination in window:
                if recombination[0] != recombination[1]:
                    all_recombinations.append(recombination)

        all_recombinations = np.unique(all_recombinations, axis=0)
        all_recombinations.sort()
        all_recombinations_tuple = []
        for recombination in all_recombinations:
            all_recombinations_tuple.append(tuple(recombination))

        topic_recombination_array = np.zeros((len(recombination_dict_Topics), len(all_recombinations_tuple)), dtype=int)

        for i in range(len(topic_recombination_array)):
            for j in range(len(topic_recombination_array.T)):
                count = recombination_dict_Topics['window_{}'.format(i * 30)].count(all_recombinations_tuple[j])
                topic_recombination_array[i, j] = count

        return topic_recombination_array, all_recombinations



###--- Class EdgeWeight_Measurement ---### ------------------------------------------------------------------------

class EdgeWeight_Measurement:

    @staticmethod
    def create_diffusion_array(topicProject_graphs):
        # get row length
        row_length = len(topicProject_graphs)

        # get column length
        all_nodes = []
        for window_id, graph in topicProject_graphs.items():
            for n in graph.nodes():
                all_nodes.append(int(n[6:]))

        all_nodes_unique = np.unique(all_nodes, axis=0)
        column_length = len(all_nodes_unique)
        all_nodes_unique.sort()

        diffusion_array = np.zeros((row_length, column_length), dtype=int)

        pbar = tqdm.tqdm(total=len(diffusion_array))

        for i in range(len(diffusion_array)):

            all_edgeNodes = []
            for (u, v, wt) in topicProject_graphs['window_{0}'.format(i * 30)].edges.data('weight'):

                all_edgeNodes.append(int(u[6:]))
                all_edgeNodes.append(int(v[6:]))

            for j in range(len(diffusion_array.T)):
                diffusion_array[i, j] = all_edgeNodes.count(all_nodes_unique[j])

            pbar.update(1)

        pbar.close()

        return diffusion_array, all_nodes_unique

    @staticmethod
    def create_recombination_array(topicProject_graphs):
        # get row length
        row_length = len(topicProject_graphs)

        # get column length
        all_edges = []
        for window_id, graph in topicProject_graphs.items():
            for (u, v) in graph.edges():

                if int(u[6:]) < int(v[6:]):
                    all_edges.append((int(u[6:]), int(v[6:])))
                elif int(u[6:]) > int(v[6:]):
                    all_edges.append((int(v[6:]), int(u[6:])))
                else:
                    raise Exception("Graph contains selfloop")

        all_edges.sort()
        all_edges_unique = np.unique(all_edges, axis=0)
        column_length = len(all_edges_unique)
        all_edges_unique.sort()

        recombinationDiffusion = np.zeros((row_length, column_length), dtype=float)

        pbar = tqdm.tqdm(total=len(recombinationDiffusion))

        for i in range(len(recombinationDiffusion)):
            for j in range(len(recombinationDiffusion.T)):

                for (u, v, wt) in topicProject_graphs['window_{0}'.format(i * 30)].edges.data('weight'):

                    if int(u[6:]) == all_edges_unique[j][0]:
                        if int(v[6:]) == all_edges_unique[j][1]:
                            recombinationDiffusion[i, j] = wt

                    elif int(u[6:]) == all_edges_unique[j][1]:
                        if int(v[6:]) == all_edges_unique[j][0]:
                            recombinationDiffusion[i, j] = wt
            pbar.update(1)

        pbar.close()

        return recombinationDiffusion, all_edges_unique


###--- Class Similarities ---### ------------------------------------------------------------------------

class Similarities:

    @staticmethod
    def check_columnLength(list_of_allArrays, diffusionArray_Topics_lp_columns):
        for i in range(len(list_of_allArrays)):
            if len(list_of_allArrays[i].T) != len((list(diffusionArray_Topics_lp_columns))):
                raise Exception("Diffusion arrays vary in their columns")
        return

    @staticmethod
    def normalized_and_binarize(list_of_allArrays, threshold, leeway):
        array_rowNorm_list = []
        array_binariz_list = []
        for i in range(len(list_of_allArrays)):
            array_toBeMod = list_of_allArrays[i]

            array_threshold, array_rowNorm = Similarities.modify_arrays(array_toBeMod, threshold)

            if leeway == True:
                array_threshold_leeway = Similarities.introcude_leeway(array_threshold, np.array([1, 0, 1]), 1)

                array_rowNorm_list.append(array_rowNorm)
                array_binariz_list.append(array_threshold_leeway)

            else:
                array_rowNorm_list.append(array_rowNorm)
                array_binariz_list.append(array_threshold)

        return array_binariz_list, array_rowNorm_list

    @staticmethod
    def modify_arrays(array, threshold):
        row_sum = array.sum(axis=1)
        row_sum = np.where(row_sum < 1, 0.0000000001, row_sum)  # smoothing to avoid dividing by 0
        array_frac = array / row_sum[:, np.newaxis]

        array_threshold = np.where(array_frac < threshold, 0, 1)

        return array_threshold, array_frac

    @staticmethod
    def introcude_leeway(pattern_array_thresh, sequence, impute_value):
        c = 0
        for row in pattern_array_thresh.T:
            row[(sciSignal.convolve(row, sequence, 'same') == 2) & (row == 0)] = impute_value
            pattern_array_thresh.T[c, :] = row
            c = c + 1
        return pattern_array_thresh

    @staticmethod
    def find_patternStart(pattern_array_thresh):

        recom_pos = []
        c = 0

        pbar = tqdm.tqdm(total=len(pattern_array_thresh.T))

        for column in pattern_array_thresh.T:
            for row in range(len(pattern_array_thresh)):
                if row != 0:
                    if column[row] == 1:
                        if column[row - 1] == 0:
                            recom_pos.append([row, c])

            c = c + 1
            pbar.update(1)

        pbar.close()
        return recom_pos

    @staticmethod
    def find_pattern_length(pattern_array_thresh, recombinationPos):

        diffu_list = []
        pbar = tqdm.tqdm(total=len(recombinationPos))

        for pos in recombinationPos:
            diffusion = -1
            i = 0

            while pattern_array_thresh[pos[0] + i, pos[1]] == 1:
                diffusion = diffusion + 1

                i = i + 1
                if pos[0] + i == len(pattern_array_thresh):
                    break

            diffu_list.append(diffusion)

            pbar.update(1)

        pbar.close()

        # Merge both lists to get final data structure #
        for i in range(len(recombinationPos)):
            recombinationPos[i].append(diffu_list[i])

        return recombinationPos

    @staticmethod
    def CM_similarities_byPair(list_of_allArrays_names, list_of_allArrays_threshold):
        namePair_list = []
        arrayPair_list = []

        namePair_list.append(list(itertools.combinations(list_of_allArrays_names, r=2)))
        namePair_list = namePair_list[0]
        arrayPair_list.append(list(itertools.combinations(list_of_allArrays_threshold, r=2)))
        arrayPair_list = arrayPair_list[0]

        similarityPair_list_cosine = []
        similarityPair_list_manhattan = []

        for matrixPair in arrayPair_list:
            patternArray1 = matrixPair[0]
            patternArray2 = matrixPair[1]

            cosine_list = []
            manhattan_list = []
            for column_id in range(len(patternArray1.T)):

                if not (sum(patternArray1[:, column_id]) == 0 and sum(patternArray2[:, column_id]) == 0):
                    cosine = Similarities.cosine_sim_mod(patternArray1[:, column_id], patternArray2[:, column_id])
                    cosine_list.append(cosine)

                manhattan = Similarities.manhattan_sim_mod(patternArray1[:, column_id], patternArray2[:, column_id])
                manhattan_list.append(manhattan)

            if len(cosine_list) != 0:  # this means: if at least in one column pair both columns were not completely 0
                cosine_avg = sum(cosine_list) / len(cosine_list)
            else:
                cosine_avg = 0

            if len(manhattan_list) != 0:  # this means: if at least in one column pair both columns were not completely 0
                manhattan_avg = sum(manhattan_list) / len(manhattan_list)
            else:
                manhattan_avg = 0

            similarityPair_list_cosine.append(cosine_avg)
            similarityPair_list_manhattan.append(manhattan_avg)

        return namePair_list, similarityPair_list_cosine, similarityPair_list_manhattan

    @staticmethod
    def CM_similarities_byTopic(list_of_allArrays_names, list_of_allArrays_threshold):
        namePair_list = []
        arrayPair_list = []
        namePair_list.append(list(itertools.combinations(list_of_allArrays_names, r=2)))
        namePair_list = namePair_list[0]
        arrayPair_list.append(list(itertools.combinations(list_of_allArrays_threshold, r=2)))
        arrayPair_list = arrayPair_list[0]

        simScores_withinTopic_list_cosine_avg = []
        simScores_withinTopic_list_manhattan_avg = []

        for column_id in range(len(list_of_allArrays_threshold[0].T)):

            simScores_withinTopic_cosine = []
            simScores_withinTopic_manhattan = []

            for matrixPair in arrayPair_list:
                patternArray1 = matrixPair[0]
                patternArray2 = matrixPair[1]

                if not (sum(patternArray1[:, column_id]) == 0 and sum(patternArray2[:, column_id]) == 0):
                    cosine = Similarities.cosine_sim_mod(patternArray1[:, column_id],
                                                                patternArray2[:, column_id])
                    simScores_withinTopic_cosine.append(cosine)
                manhattan = Similarities.manhattan_sim_mod(patternArray1[:, column_id],
                                                                  patternArray2[:, column_id])

                simScores_withinTopic_manhattan.append(manhattan)

            if len(simScores_withinTopic_cosine) != 0:
                simScores_withinTopic_list_cosine_avg.append(
                    sum(simScores_withinTopic_cosine) / len(simScores_withinTopic_cosine))
            else:
                simScores_withinTopic_list_cosine_avg.append(0)

            if len(simScores_withinTopic_manhattan) != 0:
                simScores_withinTopic_list_manhattan_avg.append(
                    sum(simScores_withinTopic_manhattan) / len(simScores_withinTopic_manhattan))
            else:
                simScores_withinTopic_list_manhattan_avg.append(-9999)

        return simScores_withinTopic_list_cosine_avg, simScores_withinTopic_list_manhattan_avg

    @staticmethod
    def cosine_sim_mod(List1, List2):
        if sum(List1) == 0 or sum(List2) == 0:
            result = 0
        else:
            result = np.dot(List1, List2) / (np.linalg.norm(List1) * np.linalg.norm(List2))
        return (result)

    @staticmethod
    def manhattan_sim_mod(List1, List2):
        return 1 - sum(abs(a - b) for a, b in zip(List1, List2)) / len(List1)

    @staticmethod
    def extend_cD_recombinationDiffuion(cd_Arrays, slidingWindow_size, cd_CCM_posStart, cd_CCM_posEnd):
        for cd_array in cd_Arrays[cd_CCM_posStart:cd_CCM_posEnd]:

            for j in range(len(cd_array.T)):
                for i in range(len(cd_array) - 2, -1,
                               -1):  # -2 because otherwise i+1 would cause problems in the following lines

                    if cd_array[i, j] >= 1:
                        if len(cd_array[i:, j]) >= slidingWindow_size:
                            for k in range(1, slidingWindow_size):
                                cd_array[i + k, j] = cd_array[i + k, j] + cd_array[i, j]
                        else:
                            for k in range(1, len(cd_array[i:, j])):
                                cd_array[i + k, j] = cd_array[i + k, j] + cd_array[i, j]
        return

    @staticmethod
    def extend_recombination_columns(column_lists, recoArrays_threshold_list):
        all_recombs = [item for sublist in column_lists for item in sublist]
        all_recombs = np.unique(all_recombs, axis=0)
        all_recombs = [tuple(x) for x in all_recombs]
        extended_arrays = []

        pbar = tqdm.tqdm(total=len(column_lists))
        for i in range(len(column_lists)):
            extended_array = recoArrays_threshold_list[i]

            tuple_list = [tuple(x) for x in column_lists[i]]

            for j in range(len(all_recombs)):
                if all_recombs[j] not in tuple_list:
                    extended_array = np.c_[extended_array[:, :j], np.zeros(len(extended_array)), extended_array[:, j:]]

            extended_array = extended_array.astype(int)
            pbar.update(1)
            extended_arrays.append(extended_array)
        pbar.close()
        return extended_arrays



###--- Class Misc ---### ------------------------------------------------------------------------

class Misc:

    @staticmethod
    def find_diffusionPatterns(CM):
        diff_pos = []
        c = 0
        for column in CM.T:
            for row in range(len(CM)):
                if row != 0:
                    if column[row] != 0:
                        if column[row - 1] == 0:
                            diff_pos.append([row, c])
                else:
                    if column[row] != 0:
                        diff_pos.append([row, c])
            c = c + 1
        return diff_pos

    @staticmethod
    def find_diffusionSequenceAndLength(diffusionPatternPos, CM):
        diff_sequence_list = []
        diff_sequence_sum_list = []
        diff_list = []
        for pos in diffusionPatternPos:
            diffusion = 0
            diff_sequence = []
            i = 0

            while CM[pos[0] + i, pos[1]] != 0:
                diffusion = diffusion + 1
                diff_sequence.append(CM[pos[0] + i, pos[1]])

                i = i + 1
                if pos[0] + i == len(CM):
                    break

            diff_list.append(diffusion)
            diff_sequence_list.append(diff_sequence)
            diff_sequence_sum_list.append(sum(diff_sequence))

        # Merge both lists to get final data structure #
        for i in range(len(diffusionPatternPos)):
            diffusionPatternPos[i].append(diff_list[i])

        return diffusionPatternPos, diff_sequence_list, diff_sequence_sum_list

    @staticmethod
    def find_diffusionStepsAndPatternPerDiffusion(diffusionPatternPos, diff_sequence_list): # still faulty
        diffusion_counter_list = []
        PatentsPerDiffPattern_list = []

        for diff_seq in diff_sequence_list:
            indicator_list = []
            diff_seq_mod = []

            indicator_list.append(0)
            diff_seq_mod.append(0)

            for i in diff_seq:
                indicator_list.append(0)
                diff_seq_mod.append(i)

            for i in range(len(indicator_list)):
                if i != 0:
                    if indicator_list[i] == 0:
                        if (diff_seq_mod[i] - diff_seq_mod[i - 1]) >= 1:
                            indicator_list[i] = diff_seq_mod[i] - diff_seq_mod[i - 1]
                            if i + 12 <= len(indicator_list) - 1:
                                indicator_list[i + 12] = (diff_seq_mod[i] - diff_seq_mod[i - 1]) * (-1)

                    elif indicator_list[i] <= -1:
                        if diff_seq_mod[i] == (diff_seq_mod[i - 1] + indicator_list[i]):
                            indicator_list[i] = 0

                        else:
                            indicator_list[i] = diff_seq_mod[i] - (diff_seq_mod[i - 1] + indicator_list[i])
                            if i + 12 <= len(indicator_list) - 1:
                                indicator_list[i + 12] = indicator_list[i] * (-1)
                            if indicator_list[i] <= -1:
                                raise Exception('current diff_seq_mod[i] is smaller then expected ')
                    else:
                        raise Exception('indicator_list[i] was positive before referencing')

            DiffsInDiffPattern = -1
            for i in indicator_list:
                if i <= -1:
                    raise Exception('indicator_list contains negative element after process')

                if i >= 1:
                    DiffsInDiffPattern = DiffsInDiffPattern + 1

            numPatentsInDiffPattern = sum(indicator_list)

            diffusion_counter_list.append(DiffsInDiffPattern)
            PatentsPerDiffPattern_list.append(numPatentsInDiffPattern)

        for i in range(len(diffusionPatternPos)):
            diffusionPatternPos[i].append(diffusion_counter_list[i])
            diffusionPatternPos[i].append(PatentsPerDiffPattern_list[i])

        return diffusionPatternPos

    @staticmethod
    def search_sequence(pattern_array_thresh, sequence):
        sequencePos = []

        c = 0
        for row in pattern_array_thresh.T:
            result = Misc.search_sequence_helper(row, sequence)
            if len(result) != 0:
                sequencePos.append((c, result))

            c = c + 1
        return sequencePos

    @staticmethod
    def search_sequence_helper(arr, seq):

        # Store sizes of input array and sequence
        Na, Nseq = arr.size, seq.size

        # Range of sequence
        r_seq = np.arange(Nseq)

        # Create a 2D array of sliding indices across the entire length of input array.
        # Match up with the input sequence & get the matching starting indices.
        M = (arr[np.arange(Na - Nseq + 1)[:, None] + r_seq] == seq).all(1)

        # Get the range of those indices as final output
        if M.any() > 0:
            return np.where(np.convolve(M, np.ones((Nseq), dtype=int)) > 0)[0]
        else:
            return []  # No match found
