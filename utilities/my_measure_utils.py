import numpy as np
import tqdm
import itertools
import scipy.signal as sciSignal
from cdlib import algorithms
import networkx as nx
import operator

class ReferenceMeasures:

    @staticmethod
    def extract_knowledgeComponent_per_window(slidingWindow_dict, kC, unit):

        if kC == 'topic':
            position = np.r_[range(9, 23, 2)]

        elif kC == 'ipc':
            position = np.r_[range(23, np.shape(slidingWindow_dict['window_0'])[1], 3)]

        else:
            raise Exception("kC must be string value 'topic' or 'ipc'")

        slidingWindow_kC_unite = {}


        pbar = tqdm.tqdm(total=len(slidingWindow_dict))

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

            # dictionary with all singularly occuring ipc's within a window
            kC_list = [item for sublist in kC_list for item in sublist]
            #print(kC_list)
            slidingWindow_kC_unite[window_id] = kC_list

            pbar.update(1)

        pbar.close()

        return slidingWindow_kC_unite

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

        pattern_array = np.zeros((len(row_list), len(column_list)))

        pbar = tqdm.tqdm(total=len(row_list))
        c_row = 0

        for row in row_list:
            c_column = 0

            for column in column_list:
                #print(tuple(column))
                #print(knowledgeComponent_dict[row])
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
    def detect_communities(patentProject_graphs, cD_algorithm, weight_bool, k_clique_size):

        community_dict = {}

        if cD_algorithm == 'label_propagation':
            c = 0

            if weight_bool == True:
                pbar = tqdm.tqdm(total=len(patentProject_graphs))
                for window_id, window in patentProject_graphs.items():
                    lp = nx.algorithms.community.label_propagation.asyn_lpa_communities(window, weight='weight', seed=123)
                    community_dict[window_id] = list(lp)
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
                pbar.update(1)

            pbar.close()

        elif cD_algorithm == 'k_clique':
            c = 0
            pbar = tqdm.tqdm(total=len(patentProject_graphs))
            for window_id, window in patentProject_graphs.items():
                kclique = algorithms.kclique(window, k=k_clique_size)  # no seed needed i think
                community_dict[window_id] = kclique.to_node_community_map()
                pbar.update(1)

            pbar.close()

        elif cD_algorithm == 'lais2':
            c = 0
            pbar = tqdm.tqdm(total=len(patentProject_graphs))
            for window_id, window in patentProject_graphs.items():
                lais2 = algorithms.lais2(window)  # no seed needed i think
                community_dict[window_id] = lais2.to_node_community_map()
                pbar.update(1)

            pbar.close()

        else:
            raise Exception("cD_algorithm must be 'label_propagation','greedy_modularity','k_clique' or 'lais2'")

        return community_dict

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