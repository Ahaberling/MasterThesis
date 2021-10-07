import numpy as np
import tqdm
import itertools
import scipy.signal as sciSignal
from cdlib import algorithms
from cdlib import evaluation
import networkx as nx
import networkx.algorithms.community as nx_comm
import operator
import copy

class ReferenceMeasures:

    @staticmethod
    def extract_knowledgeComponent_per_window(slidingWindow_dict, kC, unit):

        if kC == 'topic':
            position = np.r_[range(9, 23, 2)]

        elif kC == 'ipc':
            position = np.r_[range(23, np.shape(slidingWindow_dict['window_0'])[1], 3)]

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

                #print(kC_inPatent)
                #if 0 in kC_inPatent:
                    #print(patent)

                kC_inPatent = np.unique(kC_inPatent)

                #if 0 in kC_inPatent:
                    #print(patent)

                if unit == 1:
                    kC_list.append(tuple(kC_inPatent))

                else:
                    kC_list.append(list(itertools.combinations(kC_inPatent, r=unit)))

            # dictionary with all singularly occuring ipc's within a window
            kC_list = [item for sublist in kC_list for item in sublist]

            #if 0 in kC_list:
                #print(1+1)
            '''
            #print(kC_list)
            if 0 in kC_list:
                print(i)
                #print(kC_list)
                print('\t', kC_list.count(0))
                for patent in window:
                    #print(patent[position])
                    if '0' in patent[position]:
                        print(patent[0])
            # 276807342 was in 13 sliding windows
            '''

            slidingWindow_kC_unit[window_id] = kC_list

            pbar.update(1)
            i = i+1
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

        #print(type(column_list[0]))
        if np.issubdtype(type(column_list[0]), np.integer) or np.issubdtype(type(column_list[0]), np.str_):
            column_list.sort()

        else:
            ind = np.lexsort((column_list[:, 1], column_list[:, 0]))    # if 'unite' exceeds tuples, it is not sorted once more here
                                                                        # However this sort is redundant non the lest in the current version
            column_list = column_list[ind]

        pattern_array = np.zeros((len(row_list), len(column_list)), dtype=int)

        if np.issubdtype(type(column_list[0]), np.integer) or np.issubdtype(type(column_list[0]), np.str_):
            pbar = tqdm.tqdm(total=len(row_list))
            c_row = 0
            for row in row_list:
                c_column = 0

                for column in column_list:
                    # print(tuple(column))
                    # print(knowledgeComponent_dict[row])
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
                    # print(tuple(column))
                    # print(knowledgeComponent_dict[row])
                    if tuple(column) in knowledgeComponent_dict[row]:
                        pattern_array[c_row, c_column] = list(knowledgeComponent_dict[row]).count(tuple(column))

                    c_column = c_column + 1
                c_row = c_row + 1
                pbar.update(1)

            pbar.close()

        return pattern_array, column_list

    @staticmethod
    def find_recombination(pattern_array_thresh):

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
    def find_diffusion(pattern_array_thresh, recombinationPos):

        diff_list = []
        pbar = tqdm.tqdm(total=len(recombinationPos))

        for pos in recombinationPos:
            diffusion = -1
            i = 0

            while pattern_array_thresh[pos[0] + i, pos[1]] == 1:
                diffusion = diffusion + 1

                i = i + 1
                if pos[0] + i == len(pattern_array_thresh):
                    break

            diff_list.append(diffusion)

            pbar.update(1)

        pbar.close()

        # print(diffusion_duration_list)          # [0, 0, 0, 13, 0, 0, 20, 0, 0, 0, 13, 0, 0, 20, 41, 7, 0, 89, 89, 152, 5, 229, 90, 0, 6,
        # print(len(diffusion_duration_list))     # 3095

        # Merge both lists to get final data structure #

        for i in range(len(recombinationPos)):
            recombinationPos[i].append(diff_list[i])

        return recombinationPos

    @staticmethod
    def search_sequence_helper(arr, seq):
        """
        Parameters
        ----------
        arr    : input 1D array
        seq    : input 1D array

        Output
        ------
        Output : 1D Array of indices in the input array that satisfy the
        matching of input sequence in the input array.
        In case of no match, an empty list is returned.
        """

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

    @staticmethod
    def search_sequence(pattern_array_thresh, sequence):
        sequencePos = []

        c = 0
        for row in pattern_array_thresh.T:
            result = ReferenceMeasures.search_sequence_helper(row, sequence)
            # print(result)
            if len(result) != 0:
                sequencePos.append((c, result))

            c = c + 1
        return sequencePos

    @staticmethod
    def introcude_leeway(pattern_array_thresh, sequence, impute_value):

        c = 0
        for row in pattern_array_thresh.T:
            row[(sciSignal.convolve(row, sequence, 'same') == 2) & (row == 0)] = impute_value

            pattern_array_thresh.T[c, :] = row

            c = c + 1
        return pattern_array_thresh

class CommunityMeasures:

    @staticmethod
    def detect_communities(patentProject_graphs, cD_algorithm, weight_bool=None, k_clique_size=None):

        community_dict = {}
        modularity_dict = {}

        if cD_algorithm == 'label_propagation':
            c = 0

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
            c = 0
            pbar = tqdm.tqdm(total=len(patentProject_graphs))
            for window_id, window in patentProject_graphs.items():
                gm = algorithms.greedy_modularity(window)  # , weight='weight')                               # no seed needed i think, weights yield less communities (right now)
                community_dict[window_id] = gm.to_node_community_map()
                modularity_dict[window_id] = evaluation.newman_girvan_modularity(window, gm)
                pbar.update(1)





            pbar.close()

        elif cD_algorithm == 'k_clique':
            c = 0
            pbar = tqdm.tqdm(total=len(patentProject_graphs))
            for window_id, window in patentProject_graphs.items():
                kclique = algorithms.kclique(window, k=k_clique_size)  # no seed needed i think
                community_dict[window_id] = kclique.to_node_community_map()           # link_mod 0.048218915888596885
                modularity_dict[window_id] = evaluation.newman_girvan_modularity(window, kclique)
                pbar.update(1)

            pbar.close()

        elif cD_algorithm == 'lais2':
            c = 0
            pbar = tqdm.tqdm(total=len(patentProject_graphs))
            for window_id, window in patentProject_graphs.items():
                lais2 = algorithms.lais2(window)  # no seed needed i think
                community_dict[window_id] = lais2.to_node_community_map()           # link_mod 0.1350135445281584
                modularity_dict[window_id] = modularity_dict[window_id] = evaluation.newman_girvan_modularity(window, lais2)
                pbar.update(1)

            pbar.close()

        else:
            raise Exception("cD_algorithm must be 'label_propagation','greedy_modularity','k_clique' or 'lais2'")

        return community_dict , modularity_dict

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

                if len(max_commu_counter) >= 1:  # WHY CHECK THIS FOR K CLIQUE BUT NOT FOR LAIS2?
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
        for window_id, window in community_dict_transf.items():
            community_dict_clean[window_id] = [x for x in window if len(x) >= min_community_size]
        return community_dict_clean

    @staticmethod
    def identify_topDegree(community_dict_clean, patentProject_graphs):
        community_dict_topD = {}

        for i in range(len(community_dict_clean)):              # 189
            cd_window = community_dict_clean['window_{0}'.format(i*30)]
            topD_window = []

            # First find nodes that are in more than one community, they shall not be topD (otherwise identifying recombination gets iffy) #
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
                        topD_candidate.append((patent, patentProject_graphs['window_{0}'.format(i * 30)].degree(patent)))

                if len(topD_candidate) == 0:
                    for patent in community:
                        # get all degrees of all nodes
                        topD_candidate.append((patent, patentProject_graphs['window_{0}'.format(i * 30)].degree(patent)))

                # sort and only take top D (here D = 1)
                topD_candidate.sort(key=operator.itemgetter(1), reverse=True)
                topD = topD_candidate[0:1]                              # If multiple, just take one (This can be optimized as well)
                topD_window.append(topD)

                # FOR OVERLAPPING: DONT TAKE TOPD IN GENERAL. ONLY TAKE TOPD IF pART OF ONLY ONE COMMUNITY

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
            #if max(communities_unique_count) >= 2:
                #print(1+1)

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

                            # [[282911021, 283400389, 283460432, 283731668, 283988244, 284174115, 284201343, 284255419, 285349637, 285854177, 286019050, 286710232, 286800270], [(285349637, 13)]]
                            # 286068579
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
        #community_size_array = np.zeros((len(patentProject_graphs), max_number_community), dtype=int)

        for row in range(len(community_tracing_array)):
            current_window = community_dict_topD['window_{0}'.format(row * 30)]

            # Part1: Trace existing TopD's #
            if row != 0:  # skip in first row, since there is nothing to trace
                prev_window = community_dict_topD['window_{0}'.format((row - 1) * 30)]

                for column in range(len(community_tracing_array.T)):

                    prev_topD = community_tracing_array[row - 1, column]

                                                 # community[1][0][0] = TopD of community                             community[0] = set of id's of community
                    current_topD_candidate      = [community[1][0][0] for community in current_window if prev_topD in community[0]]
                    #current_topD_candidate_size = [len(community[0]) for community in current_window if prev_topD in community[0]]

                    if len(current_topD_candidate) == 1:  # >=2 only possible for overlapping CD
                        community_tracing_array[row, column] = current_topD_candidate[0]
                        #community_size_array[row, column] = current_topD_candidate_size[0]

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
                                    (candidate, patentProject_graphs['window_{0}'.format((row - 1) * 30)].degree(candidate)))

                            all_new_candidates.sort(key=operator.itemgetter(1), reverse=True)

                            for degree_candidate in all_new_candidates:

                                next_topk_candidate = [community[1][0][0] for community in current_window if degree_candidate[0] in community[0]]
                                next_topk_candidate_size = [len(community[0]) for community in current_window if degree_candidate[0] in community[0]]

                                if len(next_topk_candidate) == 1:
                                    community_tracing_array[row, column] = next_topk_candidate[0]
                                    #community_size_array[row, column] = next_topk_candidate_size[0]
                                    break

            # Part2: Create new communitiy entries if tracing did not create them #
            for community in current_window:

                community_identifier = community[1][0][0]

                if community_identifier not in community_tracing_array[row]:

                    for column_id in range(len(community_tracing_array.T)):

                        if sum(community_tracing_array[:,
                               column_id]) == 0:  # RuntimeWarning: overflow encountered in long_scalars

                            community_tracing_array[row, column_id] = community[1][0][0]
                            #community_size_array[row, column_id] = len(community[0])
                            break


        # Resize the arrays and exclude non relevant columns #
        for i in range(len(community_tracing_array.T)):
            if sum(community_tracing_array[:, i]) == 0:
                cutoff = i
                break

        community_tracing_array = community_tracing_array[:, 0:cutoff]
        #community_size_array = community_size_array[:, 0:cutoff]

        return community_tracing_array #, community_size_array


    @staticmethod
    def community_labeling(tracingArray, community_dict_topD, patentProject_graphs):

        ### Create dict with all topD per window and the community sequences aka columns they are associated with

        topD_dic = {}
        #topD_dic_unique = {}

        for row in range(len(tracingArray)):

            #topD_dic_unique['window_{0}'.format(row * 30)] = np.unique(tracingArray[row, :])[1:]       #[1:] to exclude the 0s

            topD_pos = {}
            for j in range(len(tracingArray[row, :])):

                # find for every topD the community sequences (columns) topD is identifying.
                if tracingArray[row,j] != 0:
                    if tracingArray[row,j] in topD_pos.keys():
                        topD_pos[tracingArray[row,j]].append(j)
                    else:
                        topD_pos[tracingArray[row, j]] = [j]

            #print(topD_pos)
            topD_dic['window_{0}'.format(row * 30)] = topD_pos

            #print(topD_dic_unique['window_{0}'.format(row * 30)])
            #print(list(topD_dic['window_{0}'.format(row * 30)].keys()))


        ### Create dict that associates a topD identifier with a stable community id (column number) for each window ###
        topD_associ = {}

        #for i in range(len(topD_dic_unique)):
        for i in range(len(topD_dic)):
            #if i * 30 == 4470:
                #print(1+1)

            tuple_list = []
            #                             (412413192, 337)  (412862058, 338)  (413103388, 328)  (416974172, 330)  (418775075, 339)  (419259320, 330)

            #for topD in topD_dic_unique['window_{0}'.format(i * 30)]:
            for topD, column_pos in topD_dic['window_{0}'.format(i * 30)].items():

                #column_pos = np.where(tracingArray[i, :] == topD)

                # if topD is present in more then 1 column of a row:
                #if len(column_pos[0]) != 1:
                if len(column_pos) != 1:        # this can never be 0
                    #print(i)
                    #print(column_pos)

                    prev_id_list = []

                    #for column in column_pos[0]:
                    for column in column_pos:

                        prev_topD = tracingArray[i-1, column]

                        prev_id_list.append((prev_topD, column))

                    prev_id_list_unique = np.unique([prev_id[0] for prev_id in prev_id_list])

                    # wouldnt there be cases where 0 is added to the list as well? No, this is never 0 because: new columns (community sequences)
                    # are only opened, if a topD appears, that is not linkable to previous topDs. if one of the pos in column_pos is linked to a previous id
                    # then the other one is as well. Casesin which two identical topDs appear in two columns/sequences are not possible, because they would
                    # just be subsumed in one sequence/one topD
                    # THIS MEANS SOME COMMUNITIES MUST HAVEMERGED.
                    #Now has to be decided how to label this merged community. Which column id is chosen?
                    # we take the previous topDs and ... see below

                    # if the current multiple occuring topD was already a topD in the previous window, then simply take the same column id as in the last window
                    # this is never ambigious, because topDs are always only associated with one column/ sequence id per window.
                    if topD in prev_id_list_unique:

                        #print(1+1)
                        column_pos = [prev_topD[1] for prev_topD in topD_associ['window_{0}'.format((i-1) * 30)] if prev_topD[0] == topD]
                        # first row does not have to be accounted for, because ambigious topDs could only occure in Lais2, but 'After all core are identified, communities with the same core are merged.'

                    #elif len(np.unique(prev_id_list_unique)) == 1:
                    #    column_pos = [topD[1] for prev_topD in topD_associ['window_{0}'.format((i-1) * 30)] if prev_topD[0] == prev_id_list[0][0]]


                    # case: ambigious topD, but topD not observed in any previous sequence points
                    else:
                        #prev_topDs_withColumn = []
                        multi_community_edgeCase = []

                        #for column in column_pos[0]:
                        #for column in column_pos:
                            #prev_topDs_withColumn.append((tracingArray[i-1,column], column))

                        prev_topD_communities_withColumn = []

                        #for prev_topD in prev_topDs_withColumn:
                        for prev_topD in prev_id_list:
                            print(prev_topD)
                            #communities = [(community, prev_topD[1]) for community in cd_topD['window_{0}'.format((i-1) * 30)] if prev_topD[0] in community[0]]
                            communities = [(community, prev_topD[1]) for community in community_dict_topD['window_{0}'.format((i-1) * 30)] if prev_topD[0] == community[1][0][0]]
                            # if selected previous topD was in more then one community...
                            # this is never the case, if two communities would have the same topD, then either antohe one would have been chosen
                            # or in the case for total subsets in lais2, the communities would have been merged -> one community
                            #print(1+1)
                            #if len(communities) >= 2:
                                #print(1+1)
                                #for community in communities:
                                    #prev_topD_communities_withColumn.append([community])
                            #else:
                                #prev_topD_communities_withColumn.append(communities)

                            # MAYBE DELETE THIS AND JUST WORK WITH 'communties' aka community later
                            prev_topD_communities_withColumn.append(communities)

                        # 'current_community' is always just 1 community
                        current_community = [community for community in community_dict_topD['window_{0}'.format(i * 30)] if topD in community[1][0]]


                        #Assumption. if topD is identifier for a community, the it is the identifier for only that community and not for multiple

                        current_community_degree_list = []
                        for patent in current_community[0][0]:
                            current_community_degree_list.append((patent, patentProject_graphs['window_{0}'.format(i * 30)].degree(patent)))

                        current_community_degree_list.sort(key=operator.itemgetter(1), reverse=True)

                        for candidate in current_community_degree_list:
                            checklist_inMultipleCommunities = []
                            #maybe i can delete the prev_community[1] aka the column pos from prev_topD_communities_withColumn

                            # get me all communities of previous topDs
                            #print(prev_topD_communities_withColumn)
                            prev_topD_communities_withColumn_mod = [prev_community[0][0] for prev_community in prev_topD_communities_withColumn]

                            # get me only the unqiue ones.
                            community_helper_list = []
                            for community_helper in prev_topD_communities_withColumn_mod:
                                if community_helper not in community_helper_list:
                                    community_helper_list.append(community_helper)
                            prev_topD_communities_withColumn_unique = community_helper_list

                            # check if my candidate node is in one and only one community
                            for prev_community in prev_topD_communities_withColumn_unique:
                                if candidate[0] in prev_community[0]:       # (290444528, 5)
                                    checklist_inMultipleCommunities.append(prev_community)

                            # if the node is in one and only one of the previous communities of topD, then get the topD of this previous community
                            # and the previous column position. This one is used then.
                            if len(checklist_inMultipleCommunities) == 1:

                                # is this is the case:
                                new_topD = checklist_inMultipleCommunities[0][1][0][0]

                                #if new_topD not in topD_dic['window_{0}'.format((i+1) * 30)]:

                                column_pos = [prev_topD[1] for prev_topD in topD_associ['window_{0}'.format((i-1) * 30)] if prev_topD[0] == new_topD]
                                #print(new_topD)
                                #print(topD_dic['window_{0}'.format((i + 1) * 30)])
                                #print(column_pos)


                                break

                            elif len(checklist_inMultipleCommunities) >= 2:
                                multi_community_edgeCase.append(checklist_inMultipleCommunities)

                        # if column_pos has not be narrowed down, but is still a list (multiple colum pos) then:
                        #if isinstance(column_pos[0], int) == False:
                        #if isinstance(column_pos, int) == False:


                        # multi_community_edgeCase_unique now equals all communties that inhibit node candidates that are not
                        # unique to one community
                        #if len(column_pos[0]) != 1:
                        if len(column_pos) != 1:
                            multi_community_edgeCase = [item for sublist in multi_community_edgeCase for item in sublist]

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

                            column_pos = [prev_topD[1] for prev_topD in topD_associ['window_{0}'.format((i - 1) * 30)] if prev_topD[0] == new_topD]

                #tuple_list.append((topD, int(column_pos[0])))
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
    def create_visualization_array(tracingArray, topD_communityID_association_perWindow):

        #visual_array = np.zeros((len(topicSim), max_number), dtype=int)
        visual_array = np.full((np.shape(tracingArray)[0], np.shape(tracingArray)[1]), 9999999)

        for row in range(len(visual_array)):
            for column in range(len(visual_array.T)):

                if tracingArray[row, column] != 0:

                    topD = tracingArray[row, column]

                    label_entry = [tuple[1] for tuple in topD_communityID_association_perWindow['window_{0}'.format(row * 30)] if topD == tuple[0]]
                    visual_array[row, column] = label_entry[0]

        return visual_array

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
    def find_recombinations_crisp(community_dict_labeled, patentProject_graphs):
        cd_recombination_dic = {}   # community_dict_labeled = window: [{member ids}, [community id]]

        for i in range(len(patentProject_graphs)):
            window_list = []

            if i != 0:
                t_minus1 = set(patentProject_graphs['window_{0}'.format((i-1) * 30)].nodes())
                t = set(patentProject_graphs['window_{0}'.format(i * 30)].nodes())
                new_patents = t.difference(t_minus1)

                for patent in new_patents:
                    neighbor_list = list(patentProject_graphs['window_{0}'.format(i * 30)].neighbors(patent))

                    patent_list = []

                    if len(neighbor_list) >=2:

                        bridge_list = []
                        already_found_community = []

                        for neighbor in neighbor_list:

                            for community in community_dict_labeled['window_{0}'.format((i-1) * 30)]:

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

    @staticmethod
    def recombination_threshold_crisp(recombination_dict, patentProject_graphs, threshold_value):
        recombination_threshold = {}

        for window_id, window in recombination_dict.items():
            recombination_types_plusCount = []

            if len(window) != 0:
                total_number_patents = len(patentProject_graphs[window_id].nodes())
                recombination_types = []

                for recombination in window:
                    if recombination[1][0][1][0] <= recombination[1][1][1][
                        0]:  # probably not necessary anymore, because it was sorted in the pre function as well
                        community_id1 = recombination[1][0][1][0]  #
                        community_id2 = recombination[1][1][1][0]
                    else:
                        community_id1 = recombination[1][1][1][0]
                        community_id2 = recombination[1][0][1][0]

                    recombination_types.append((community_id1, community_id2))

                recombination_types_unique, index, count = np.unique(recombination_types, axis=0, return_counts=True,
                                                                     return_index=True)
                fraction = [x / total_number_patents for x in count]
                threshold_meet_list = []

                for i in range(len(fraction)):
                    threshold_meet = 0  # default
                    if fraction[i] >= threshold_value:
                        threshold_meet = 1

                    threshold_meet_list.append(threshold_meet)

                for i in range(len(recombination_types_unique)):
                    recombination_types_plusCount.append(
                        (tuple(recombination_types_unique[i]), count[i], threshold_meet_list[i]))  # , fraction[i]))

            recombination_threshold[window_id] = recombination_types_plusCount

        return recombination_threshold

    @staticmethod
    def recombination_threshold_overlapping(recombination_dict, patentProject_graphs, threshold_value):
        recombination_threshold = {}

        for window_id, window in recombination_dict.items():
            recombination_types_plusCount = []

            if len(window) != 0:
                total_number_patents = len(patentProject_graphs[window_id].nodes())
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
                    if fraction[i] >= threshold_value:
                        threshold_meet = 1

                    threshold_meet_list.append(threshold_meet)

                for i in range(len(recombination_types_unique)):
                    recombination_types_plusCount.append((tuple(recombination_types_unique[i]), count[i], threshold_meet_list[i]))  # , fraction[i]))

            recombination_threshold[window_id] = recombination_types_plusCount

        return recombination_threshold

    @staticmethod
    def enrich_recombinations_dic_with_thresholds_crips(recombination_dict, recombination_dict_threshold):

        recombinations_dic_with_thresholds = {}

        for window_id, window in recombination_dict.items():

            new_window = []
            for recombination in window:
                #print(recombination)
                community_id1 = recombination[1][0][1][0]
                community_id2 = recombination[1][1][1][0]

                for recombination_threshold in recombination_dict_threshold[window_id]:
                    if recombination_threshold[0] == (community_id1, community_id2):

                        count = recombination_threshold[1]
                        threshold = recombination_threshold[2]

                        recombination.append(count)
                        recombination.append(threshold)


                new_window.append(recombination)
            recombinations_dic_with_thresholds[window_id] = new_window

        return recombinations_dic_with_thresholds

    @staticmethod
    def enrich_recombinations_dic_with_thresholds_overlapping(recombination_dict, recombination_dict_threshold):
        recombinations_dic_with_thresholds = {}

        for window_id, window in recombination_dict.items():

            new_window = []
            for recombination in window:
                community_id1 = recombination[1][0]
                community_id2 = recombination[1][1]

                recombination_value = list(recombination)

            for recombination_threshold in recombination_dict_threshold[window_id]:
                if recombination_threshold[0] == (community_id1, community_id2):

                    count = recombination_threshold[1]
                    threshold = recombination_threshold[2]

                    recombination_value.append(count)
                    recombination_value.append(threshold)

                new_window.append(recombination_value)
            recombinations_dic_with_thresholds[window_id] = new_window

        return recombinations_dic_with_thresholds

    @staticmethod
    def create_cleaningIndex_associationAccumulated(topD_communityID_association_accumulated, community_dict_topD, patentProject_graphs):
        merging_communities_dic = {}
        # window 1140 should contain 37 and 90
        for i in range(len(topD_communityID_association_accumulated) - 1):

            #if i == 89:
                #print(1+1)
            #if i == 25:
                #print(1 + 1)
            #if i == 185:
                #print(1 + 1)

            window_id = 'window_{0}'.format(i * 30)
            window = topD_communityID_association_accumulated[window_id]

            next_window_id = 'window_{0}'.format((i + 1) * 30)
            next_window = topD_communityID_association_accumulated[next_window_id]

            swallowed_communities = []

            if i != 0:

                # if i == 12:
                # print(1+1)

                # get all topDs and check if they are present in the next window as well
                for topD in window.keys():
                    if topD not in next_window.keys():

                        next_community_lists = list(next_window.values())

                        # print(next_community_lists)

                        for community in next_community_lists:

                            if set(window[topD]).issubset(set(community)):

                                if len(set(community) - set(window[topD])) != 0:
                                    swallowed_communities.append([topD, window[topD]])
                                break

                # get life time
                # I want for every window a list of community_ids that are swallowed, and their death time
                if len(swallowed_communities) != 0:
                    for swallowed_community in swallowed_communities:
                        # if len(swallowed_community) != 0:
                        for community in community_dict_topD[window_id]:

                            if swallowed_community[0] == community[1][0][0]:
                                swallowed_community.append(community[0])
                                break
                    # print(swallowed_communities)

                    for swallowed_community in swallowed_communities:
                        community_death = False
                        j = i
                        members = list(swallowed_community[2])

                        while community_death == False:  # [288465877, 287698910, 286963357, 289531190]

                            all_id_in_next_window = list(patentProject_graphs['window_{0}'.format((j + 1) * 30)].nodes())
                            #all_id_in_next_window = [item for sublist in community_dict_topD['window_{0}'.format((j + 1) * 30)] for item in sublist]
                            #all_id_in_next_window = [item for sublist in all_id_in_next_window for item in sublist]

                            # todo: CHECK FOR ALL WITHIN A WINDOW (IF ALL DISAPPEAR)
                            # 237 in window 185 is not in swallowed?
                            missing_members = []
                            for member in members:
                                if member not in all_id_in_next_window:
                                    missing_members.append(member)
                            if len(missing_members) >= len(members):
                                community_death = True

                            else:
                                j = j + 1

                                if j == 188:
                                    community_death = True

                        swallowed_community.append(i)  # i = last point before merge
                        swallowed_community.append(j)  # j = point of death (first row were not alive)

            merging_communities_dic[window_id] = swallowed_communities

        return merging_communities_dic

    @staticmethod

    def cleaning_associationAccumulated(topD_communityID_association_accumulated, topD_communityID_association_accumulated_cleanID):
        cd_topD_dic_clean = copy.deepcopy(topD_communityID_association_accumulated)

        for i in range(len(topD_communityID_association_accumulated_cleanID)):
            window_id = 'window_{0}'.format(i * 30)
            window = topD_communityID_association_accumulated_cleanID[window_id]
            if i == 79:
                print(1+1)

            for cleaning_entry in window:
                community_id = cleaning_entry[1]
                last_point_before_swallowed = cleaning_entry[3]
                point_of_death = cleaning_entry[4]

                # for j2 in range(last_point_before_swallowed,point_of_death):
                # print(cd_topD_dic_clean['window_{0}'.format(j2 * 30)])

                for j in range(point_of_death, len(cd_topD_dic_clean)):
                    cleaning_window = cd_topD_dic_clean['window_{0}'.format(j * 30)]
                    for community_toBeCleaned in cleaning_window.values():
                        if set(community_id).issubset(set(community_toBeCleaned)):
                            for id in community_id:
                                community_toBeCleaned.remove(id)

                # for j3 in range(last_point_before_swallowed, point_of_death+10):
                # print(cd_topD_dic_clean['window_{0}'.format(j3 * 30)])

                # print(cd_topD_dic_clean['window_330'])

        return cd_topD_dic_clean

    @staticmethod
    def single_diffusion_v2(cd_topD_dic_clean):
        row_length = len(cd_topD_dic_clean)

        all_ids = []
        for window_id, window in cd_topD_dic_clean.items():

            for community in window:
                all_ids.append(window[community])
        all_ids = [item for sublist in all_ids for item in sublist]

        column_length = max(all_ids)+1

        singleDiffusion_array = np.zeros((row_length, column_length), dtype=int)

        pbar = tqdm.tqdm(total=len(singleDiffusion_array))
        for i in range(len(singleDiffusion_array)):
            for j in range(len(singleDiffusion_array.T)):

                window = cd_topD_dic_clean['window_{0}'.format(i * 30)]
                #print(window)
                #print(window.values())
                #print(list(window.values()))
                #print(j)

                if any(j in sublist for sublist in list(window.values())) == True:
                    # A count is not necessary, since the value can not exceed 1. Community ids are unique within a window.
                    singleDiffusion_array[i, j] = 1

                '''
                for k in range(len(list(window.values()))):
                    if j in list(window.values())[k]:
                        big_community = list(window.values())[k]
                        break
                #print(big_community)

                overall_count = 0

                big_community = 1
                '''
            pbar.update(1)

        pbar.close()

        return singleDiffusion_array

    @staticmethod
    def recombination_diffusion_crip_v2(topD_communityID_association_accumulated_clean, recombination_dict, patentProject_graphs):

        # NOTE: a recombination in a window of cd_recombinations contains a patent that recombines knowledge with it's links.
        # This patent occures in the said window.
        # BUT the two nodes that are identifying the recombination are extracted from the window before (t-1). This means that
        # the following case is possible:
        # In window 10, patent A recombines knowledge via neighbor B (community X) and neighbor C (Community Y). Patent A is always
        # present in window 10, but it's nighboors B and C were extracted in window 9. They, or their community might have vanished in window 10
        # For this reason, we are ... #

        row_length = len(recombination_dict)

        recombinations_dic = {}
        recombinations_all = []
        # for window_id, window in cd_recombinations.items():
        for l in range(1, len(recombination_dict)):

            recombinations_window = []
            for recombination in recombination_dict['window_{0}'.format(l * 30)]:
                community_id1 = recombination[1][0][1][0]
                community_id2 = recombination[1][1][1][0]
                if community_id1 < community_id2:
                    recombinations_all.append((community_id1, community_id2))
                    recombinations_window.append((community_id1, community_id2))
                elif community_id1 > community_id2:
                    recombinations_all.append((community_id2, community_id1))
                    recombinations_window.append((community_id2, community_id1))
                else:
                    raise Exception("recombination dictionary contains recombinations between the same communities")

            recombinations_dic['window_{0}'.format((l - 1) * 30)] = recombinations_window
        # print(len(recombinations_dic))
        recombinations_dic['window_{0}'.format((len(recombination_dict) - 1) * 30)] = []
        # print(len(recombinations_dic))

        # print(recombinations_dic)

        # print(len(recombinations_all))
        recombinations_all.sort()
        # print(recombinations_all)
        recombinations_all = np.unique(recombinations_all, axis=0)
        # print(recombinations_all)

        recombinations_all_tuple = []
        for recombination in recombinations_all:
            recombinations_all_tuple.append(tuple(recombination))

        recombinations_all = recombinations_all_tuple
        # print(len(recombinations_all))

        # print(recombinations_all)
        column_length = len(recombinations_all)
        # column 79 = (37, 90)

        # recombinationDiffusion_count = np.zeros((row_length, column_length), dtype=int)
        recombinationDiffusion_count = np.zeros((row_length, column_length), dtype=int)
        recombinationDiffusion_fraction = np.zeros((row_length, column_length), dtype=float)
        recombinationDiffusion_threshold = np.zeros((row_length, column_length), dtype=float)

        pbar = tqdm.tqdm(total=len(recombinationDiffusion_count))
        for i in range(len(recombinationDiffusion_count)):  # 63: [[59], [63], [64], [65], [66]]
            # if i == 10:
            # print(1+1)
            for j in range(len(recombinationDiffusion_count.T)):

                window_topD_dic = topD_communityID_association_accumulated_clean['window_{0}'.format(i * 30)]
                # print(window_topD_dic)

                # count how often a recombination appears in a window
                # Recombinations are identified over community id. These community id's are dominant.
                # print(recombinations_all[j])
                # print(recombinations_dic['window_{0}'.format(i*30)])

                recombination_count = recombinations_dic['window_{0}'.format(i * 30)].count(recombinations_all[j])
                # print(recombination_count)
                #if recombination_count >=2:
                    #print(i*30)

                if recombination_count != 0:
                    #if j == 89:
                        #print(recombinations_all[j])
                        #print(recombinations_dic['window_{0}'.format(i * 30)])
                        #print(1 + 1)
                    # this count has to be placed in all columns that are the same recombination under different community ids
                    # (e.g. because of a community merge where the dominant id overwrite the original one used in the prior recombination

                    big_community1 = []
                    big_community2 = []

                    # print(list(window_topD_dic.values()))
                    for k in range(len(list(window_topD_dic.values()))):
                        # print(recombinations_all[j][0])
                        # print(list(window_topD_dic.values())[k])
                        if recombinations_all[j][0] in list(window_topD_dic.values())[k]:
                            big_community1 = list(window_topD_dic.values())[k]
                            # print(big_community1)
                            break

                    list(window_topD_dic.values())
                    for k in range(len(list(window_topD_dic.values()))):
                        # print(recombinations_all[j][1])
                        # print(list(window_topD_dic.values())[k])
                        if recombinations_all[j][1] in list(window_topD_dic.values())[k]:
                            big_community2 = list(window_topD_dic.values())[k]
                            # print(big_community2)
                            break

                    # find all j's where the count has to be written in as well

                    weak_recombination_list = []
                    # print(recombinations_all)
                    for h in range(len(recombinations_all)):
                        # print(recombinations_all[h][0])
                        # print(big_community1)
                        # print(big_community2)
                        if recombinations_all[h][0] in big_community1:
                            # print(recombinations_all[h][1])
                            # print(big_community2)
                            if recombinations_all[h][1] in big_community2:
                                # print(h)
                                weak_recombination_list.append(h)
                        elif recombinations_all[h][0] in big_community2:
                            # print(recombinations_all[h][1])
                            # print(big_community1)
                            if recombinations_all[h][1] in big_community1:
                                # print(h)
                                weak_recombination_list.append(h)

                    for weak_recombination_pos in weak_recombination_list:
                        # print(weak_recombination_pos)
                        # print(recombination_count)
                        recombinationDiffusion_count[i, weak_recombination_pos] = recombination_count
                        # print(recombinationDiffusion_count[i, weak_recombination_pos])

            pbar.update(1)
        pbar.close()

        for n in range(len(patentProject_graphs)):
            # if n == 50:
            # print(1+1)
            all_nodes_window = len(patentProject_graphs['window_{0}'.format(n * 30)].nodes())
            # print(all_nodes_window)
            # print(recombinationDiffusion_count[n,:])
            recombinationDiffusion_fraction[n, :] = recombinationDiffusion_count[n, :] / all_nodes_window
            # print(recombinationDiffusion_fraction[n,:])

        recombinationDiffusion_threshold = np.where(recombinationDiffusion_fraction < 0.005, 0, 1)
        #print(1 + 1)

        return recombinationDiffusion_count, recombinationDiffusion_fraction, recombinationDiffusion_threshold, recombinations_all


    @staticmethod
    def recombination_diffusion_overlapping_v2(cd_topD_dic_clean, cd_recombinations, patentProject_graphs):
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
        # print(recombinations_dic)

        # print(len(recombinations_all))
        recombinations_all.sort()
        # print(recombinations_all)
        recombinations_all = np.unique(recombinations_all, axis=0)
        # print(recombinations_all)
        recombinations_all_tuple = []
        for recombination in recombinations_all:
            recombinations_all_tuple.append(tuple(recombination))
        recombinations_all = recombinations_all_tuple
        # print(len(recombinations_all))

        # print(recombinations_all)
        column_length = len(recombinations_all)

        # recombinationDiffusion_count = np.zeros((row_length, column_length), dtype=int)
        recombinationDiffusion_count = np.zeros((row_length, column_length), dtype=int)
        recombinationDiffusion_fraction = np.zeros((row_length, column_length), dtype=float)
        # recombinationDiffusion_threshold = np.zeros((row_length, column_length), dtype=float)

        pbar = tqdm.tqdm(total=len(recombinationDiffusion_count))
        for i in range(len(recombinationDiffusion_count)):
            # if i == 10:
            # print(1+1)
            for j in range(len(recombinationDiffusion_count.T)):

                window_topD_dic = cd_topD_dic_clean['window_{0}'.format(i * 30)]
                # print(window_topD_dic)

                # count how often a recombination appears in a window
                # Recombinations are identified over community id. These community id's are dominant.
                # print(recombinations_all[j])
                # print(recombinations_dic['window_{0}'.format(i*30)])
                recombination_count = recombinations_dic['window_{0}'.format(i * 30)].count(recombinations_all[j])
                # print(recombination_count)

                if recombination_count != 0:
                    # this count has to be placed in all columns that are the same recombination under different community ids
                    # (e.g. because of a community merge where the dominant id overwrite the original one used in the prior recombination

                    # print(list(window_topD_dic.values()))
                    for k in range(len(list(window_topD_dic.values()))):
                        # print(recombinations_all[j][0])
                        # print(list(window_topD_dic.values())[k])
                        if recombinations_all[j][0] in list(window_topD_dic.values())[k]:
                            big_community1 = list(window_topD_dic.values())[k]
                            # print(big_community1)
                            break

                    # print(list(window_topD_dic.values()))
                    for k in range(len(list(window_topD_dic.values()))):
                        # print(recombinations_all[j][1])
                        # print(list(window_topD_dic.values())[k])
                        if recombinations_all[j][1] in list(window_topD_dic.values())[k]:
                            big_community2 = list(window_topD_dic.values())[k]
                            # print(big_community2)
                            break

                    # find all j's where the count has to be written in as well

                    # 'weak' not fitting anymore
                    weak_recombination_list = []
                    # print(recombinations_all)
                    for h in range(len(recombinations_all)):
                        # print(recombinations_all[h][0])
                        # print(big_community1)
                        # print(big_community2)
                        if recombinations_all[h][0] in big_community1:
                            # print(recombinations_all[h][1])
                            # print(big_community2)
                            if recombinations_all[h][1] in big_community2:
                                # print(h)
                                weak_recombination_list.append(h)
                        elif recombinations_all[h][0] in big_community2:
                            # print(recombinations_all[h][1])
                            # print(big_community1)
                            if recombinations_all[h][1] in big_community1:
                                # print(h)
                                weak_recombination_list.append(h)

                    for weak_recombination_pos in weak_recombination_list:
                        # print(weak_recombination_pos)
                        # print(recombination_count)
                        recombinationDiffusion_count[i, weak_recombination_pos] = recombination_count
                        # print(recombinationDiffusion_count[i, weak_recombination_pos])

            pbar.update(1)
        pbar.close()

        for n in range(len(patentProject_graphs)):
            # if n == 50:
            # print(1+1)
            all_nodes_window = len(patentProject_graphs['window_{0}'.format(n * 30)].nodes())
            # print(all_nodes_window)
            # print(recombinationDiffusion_count[n,:])
            recombinationDiffusion_fraction[n, :] = recombinationDiffusion_count[n, :] / all_nodes_window
            # print(recombinationDiffusion_fraction[n,:])

        recombinationDiffusion_threshold = np.where(recombinationDiffusion_fraction < 0.005, 0, 1)

        #print(1 + 1)

        return recombinationDiffusion_count, recombinationDiffusion_fraction, recombinationDiffusion_threshold, recombinations_all

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
                    topic_list.append(patent_lda_ipc[paten_pos[0][0]][9:23])

                topic_list = [item for sublist in topic_list for item in sublist]
                topic_list = [x for x in topic_list if x == x]

                for i in range(0, len(topic_list), 2):
                    for row in range(len(community_topics)):  # for all patents in the community
                        if community_topics[row, int(topic_list[i])] == 0:
                            community_topics[row, int(topic_list[i])] = topic_list[i + 1]
                            break
                        #else:
                            #print(1+1)

                community_topics = np.sum(community_topics, axis=0)
                window_list.append([community[1][0], list(community_topics)])
            community_topicDist_dic[window_id] = window_list
        return community_topicDist_dic

    @staticmethod
    def create_dict_communityTopicAssociation(community_topicDist_dic):

        # 1. create dic with: each window, list of tuple with (communityID, highest topic)

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
    def create_diffusionArray_Topics(communityTopicAssociation_dict):
        topic_diffusion_array = np.zeros((len(communityTopicAssociation_dict), 330), dtype=int)
        # community id, topic id, confidence
        for i in range(len(topic_diffusion_array)):
            window = communityTopicAssociation_dict['window_{}'.format(i * 30)]

            for j in range(len(topic_diffusion_array.T)):

                if any(j == community[1] for community in window) == True:
                    topic_diffusion_array[i, j] = 1
        return topic_diffusion_array, list(range(330))

    @staticmethod
    def created_recombination_dict_Topics_crisp(communityTopicAssociation_dict, recombination_dict):
        recombination_dict_mod_lp = {}      # communityTopicAssociation_dict =  window: [community id, topic, confidence]
        for i in range(len(recombination_dict)): #recombination_dict =          window: [broker patent id((neighbor id, community id),(neighbor id, community id)]
            new_window = []
            for recombination in recombination_dict['window_{}'.format(i*30)]:
                new_recombination = []
                if i != 0:
                    for community in communityTopicAssociation_dict['window_{}'.format((i-1)*30)]:
                        if recombination[1][0][1][0] == community[0]:
                            new_recombination.append(community[1])
                        if recombination[1][1][1][0] == community[0]:
                            new_recombination.append(community[1])
                        if len(new_recombination) >= 2:
                            break
                new_recombination.sort()
                new_window.append(tuple(new_recombination))
            recombination_dict_mod_lp['window_{}'.format(i*30)] = new_window

        return recombination_dict_mod_lp

    @staticmethod
    def created_recombination_dict_Topics_overlap(communityTopicAssociation_dict, recombination_dict):
        recombination_dict_mod_lp = {}
        for i in range(len(recombination_dict)):
            new_window = []
            for recombination in recombination_dict['window_{}'.format(i*30)]:
                new_recombination = []
                #if i != 0:
                for community in communityTopicAssociation_dict['window_{}'.format(i*30)]:  # todo does this have to be i or i-1?
                    if recombination[1][0] == community[0]:
                        new_recombination.append(community[1])
                    if recombination[1][1] == community[0]:
                        new_recombination.append(community[1])
                    if len(new_recombination) >= 2:
                        break
                new_recombination.sort()
                new_window.append(tuple(new_recombination))
            recombination_dict_mod_lp['window_{}'.format(i*30)] = new_window

        return recombination_dict_mod_lp

    @staticmethod
    def doubleCheck_recombination_dict_Topics_crisp(recombination_dict_mod, recombination_dict, communityTopicAssociation_dict):
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

            # stochastically check if recombinations in both dictionaries are consistent
            #print(recombination_dict['window_{}'.format(i * 30)])
            #print(recombination_dict_mod['window_{}'.format(i * 30)])
            #print(helper)
            #print(communityTopicAssociation_dict['window_{}'.format(i * 30)])

        return

    @staticmethod
    def doubleCheck_recombination_dict_Topics_overlap(recombination_dict_mod, recombination_dict, communityTopicAssociation_dict):
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

            # stochastically check if recombinations in both dictionaries are consistent
            #print(recombination_dict['window_{}'.format(i * 30)])
            #print(recombination_dict_mod['window_{}'.format(i * 30)])
            #print(helper)
            #print(communityTopicAssociation_dict['window_{}'.format(i * 30)])

        return

    # [[14, 121, 0.17], [21, 121, 0.38], [22, 65, 0.19], [23, 144, 0.14], [19, 178, 0.26], [24, 8, 0.33], [18, 316, 0.35]]
    #14, 18, 21, 22, 23, 24

    # recombination_dict #16
    # [(14, 21), (22, 23), (22, 24), (23, 24), (22, 23), (22, 24), (23, 24), (14, 21), (14, 21), (14, 21), (14, 18), (14, 21), (18, 21), (22, 23), (22, 24), (23, 24)]

    # recombination_dict_mod #16
    # [(121, 121), (65, 144), (8, 65), (8, 144), (65, 144), (8, 65), (8, 144), (121, 121), (121, 121), (121, 121), (121, 316), (121, 121), (121, 316), (65, 144), (8, 65), (8, 144)]

    # helper
    # [(121, 121), (65, 144), (8, 65), (8, 144), (121, 316), [14, 21], [22, 23], [22, 24], [23, 24]]

    @staticmethod
    def create_recombinationArray_Topics(recombination_dict_Topics): # recombination_dict_Topics = window: [(topic id, topic id), ...]
        all_recombinations = []
        for window_id, window in recombination_dict_Topics.items():
            for recombination in window:
                if recombination[0] != recombination[1]:
                    all_recombinations.append(recombination) # 6779 in total

        for recombination in all_recombinations:
            if recombination[0] >= recombination[1]:
                print('problem')
                print(recombination)

        all_recombinations = np.unique(all_recombinations, axis=0)

        for recombination in all_recombinations:
            if recombination[0] >= recombination[1]:
                print('problem')
                print(recombination)

        #print(len(all_recombinations))  # 3061
        all_recombinations.sort()

        for recombination in all_recombinations:
            if recombination[0] >= recombination[1]:
                print('problem')
                print(recombination)

        all_recombinations_tuple = []
        for recombination in all_recombinations:
            all_recombinations_tuple.append(tuple(recombination))

        for recombination in all_recombinations_tuple:
            if recombination[0] >= recombination[1]:
                print('problem')
                print(recombination)

        topic_recombination_array = np.zeros((len(recombination_dict_Topics), len(all_recombinations_tuple)), dtype=int)
        #topic_recombination_array_frac = np.zeros((len(recombination_dict_Topics), len(all_recombinations_tuple)), dtype=float)
        # print(np.shape(lp_recombination_diffusion_crip_count_v2))   # 3710 --> 550 rekombinations are merged in other recombinations

        for i in range(len(topic_recombination_array)):
            for j in range(len(topic_recombination_array.T)):
                count = recombination_dict_Topics['window_{}'.format(i * 30)].count(all_recombinations_tuple[j])
                topic_recombination_array[i, j] = count

            #rowsum = topic_recombination_array[i, :].sum()

            #if rowsum != 0:
                #topic_recombination_array_frac[i, :] = topic_recombination_array[i, :] / rowsum

        #topic_recombination_array_threshold = np.where(topic_recombination_array_frac < 0.005, 0, 1)

        #return topic_recombination_array_threshold, all_recombinations
        return topic_recombination_array, all_recombinations




class EdgeWeightMeasures:

    @staticmethod
    def create_diffusion_array(topicProject_graphs, threshold, edge_threshold_quantil):
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
        diffusion_array_frac = np.zeros((row_length, column_length), dtype=float)

        pbar = tqdm.tqdm(total=len(diffusion_array))

        for i in range(len(diffusion_array)):

            weight_list = []
            for (u, v, wt) in topicProject_graphs['window_{0}'.format(i * 30)].edges.data('weight'):
                weight_list.append(wt)

            weight_threshold = np.quantile(weight_list, edge_threshold_quantil)


            all_edgeNodes = []
            for (u, v, wt) in topicProject_graphs['window_{0}'.format(i * 30)].edges.data('weight'):
                # to get rid of noice
                if wt >= weight_threshold:
                    all_edgeNodes.append(int(u[6:]))
                    all_edgeNodes.append(int(v[6:]))

            for j in range(len(diffusion_array.T)):

                diffusion_array[i, j] = all_edgeNodes.count(all_nodes_unique[j])

            pbar.update(1)
            #print(len(topicProject_graphs['window_{0}'.format(i * 30)].edges()))
            diffusion_array_frac[i] = diffusion_array[i] / len(topicProject_graphs['window_{0}'.format(i * 30)].edges())

        pbar.close()
        diffusion_array_thresh = np.where(diffusion_array_frac < threshold, 0, 1)



        return diffusion_array, diffusion_array_frac, diffusion_array_thresh, all_nodes_unique

    @staticmethod
    def create_recombination_array(topicProject_graphs, threshold):
        # get row length
        row_length = len(topicProject_graphs)

        # get column length
        all_edges = []
        for window_id, graph in topicProject_graphs.items():
            for (u, v) in graph.edges():
                #all_edges.append((int(u[6:]), int(v[6:])))

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

        row_sum = recombinationDiffusion.sum(axis=1)
        recombinationDiffusion_frac = recombinationDiffusion / row_sum[:, np.newaxis]
        recombinationDiffusion_thresh = np.where(recombinationDiffusion_frac < threshold, 0, 1)

        return recombinationDiffusion, recombinationDiffusion_frac, recombinationDiffusion_thresh, all_edges_unique

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
        diff_list = []
        for pos in diffusionPatternPos:
            diffusion = 0
            diff_sequence = []
            i = 0
            # diff_sequence.append(pattern_array_reference_diff[pos[0],pos[1]])

            while CM[pos[0] + i, pos[1]] != 0:
                diffusion = diffusion + 1
                diff_sequence.append(CM[pos[0] + i, pos[1]])

                i = i + 1
                if pos[0] + i == len(CM):
                    break

            diff_list.append(diffusion)
            diff_sequence_list.append(diff_sequence)

        # Merge both lists to get final data structure #

        for i in range(len(diffusionPatternPos)):
            diffusionPatternPos[i].append(diff_list[i])
        return diffusionPatternPos, diff_sequence_list

    @staticmethod
    def find_diffusionStepsAndPatternPerDiffusion(diffusionPatternPos, diff_sequence_list):
        diffusion_counter_list = []
        PatentsPerDiffPattern_list = []

        for diff_seq in diff_sequence_list:
            indicator_list = []
            diff_seq_mod = []

            # if diff_seq == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 3, 3, 3, 3, 3, 4, 4, 5, 5, 5, 5, 5, 3, 4, 4, 5, 5, 4, 3, 2, 3, 3, 3, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 10, 9, 8, 10, 10, 8, 8, 8, 8, 7, 6, 6, 5, 4, 4, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3]:
            # print(1+1)

            indicator_list.append(0)
            diff_seq_mod.append(0)

            for i in diff_seq:
                indicator_list.append(0)
                diff_seq_mod.append(i)

            # if diff_seq_mod == [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 3, 3, 3, 3, 3, 4, 4, 5, 5, 5, 5, 5, 3, 4, 4, 5, 5, 4, 3, 2, 3, 3, 3, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 10, 9, 8, 10, 10, 8, 8, 8, 8, 7, 6, 6, 5, 4, 4, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3]:
            # print(1+1)

            for i in range(len(indicator_list)):
                # if i == 74-12-12:
                # print(1+1)   # indicator pos 50 jumps from -1 to 1
                # if i == 74-12:
                # print(1+1)
                # if i == 74:
                # print(1+1)
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
                            # print(indicator_list[i-12])
                            # print(diff_seq_mod[i])
                            # print(diff_seq_mod[i-1])
                            # print(indicator_list[i])
                            indicator_list[i] = diff_seq_mod[i] - (diff_seq_mod[i - 1] + indicator_list[i])
                            if i + 12 <= len(indicator_list) - 1:
                                indicator_list[i + 12] = indicator_list[i] * (-1)
                            if indicator_list[i] <= -1:
                                # print(diff_seq_mod)
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