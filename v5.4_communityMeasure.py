if __name__ == '__main__':

    # --- Import Libraries ---#
    print('\n#--- Import Libraries ---#\n')

    import pickle as pk
    import os

    # --- Initialization ---#
    print('\n# --- Initialization ---#\n')

    os.chdir('D:/Universitaet Mannheim/MMDS 7. Semester/Master Thesis/Outline/Data/Cleaning Robots')

    with open('patentProject_graphs', 'rb') as handle:
        patentProject_graphs = pk.load(handle)

    # --- Applying Community detection to each graph/window and populate respective dictionaries ---#

    ### Creating dictionaries to save communities ###

    from utilities.my_measure_utils import CommunityMeasures
    '''
    community_dict_lp = CommunityMeasures.detect_communities(patentProject_graphs, cD_algorithm='label_propagation', weight=True)
    community_dict_gm = CommunityMeasures.detect_communities(patentProject_graphs, cD_algorithm='greedy_modularity')
    community_dict_kc = CommunityMeasures.detect_communities(patentProject_graphs, cD_algorithm='k_clique', k_clique_size=3)
    community_dict_l2 = CommunityMeasures.detect_communities(patentProject_graphs, cD_algorithm='lais2')

    filename = 'community_dict_lp'
    outfile = open(filename, 'wb')
    pk.dump(community_dict_lp, outfile)
    outfile.close()

    filename = 'community_dict_gm'
    outfile = open(filename, 'wb')
    pk.dump(community_dict_gm, outfile)
    outfile.close()

    filename = 'community_dict_kc'
    outfile = open(filename, 'wb')
    pk.dump(community_dict_kc, outfile)
    outfile.close()

    filename = 'community_dict_l2'
    outfile = open(filename, 'wb')
    pk.dump(community_dict_l2, outfile)
    outfile.close()
    '''
    # --- Transform data structure ---#

    with open('community_dict_lp', 'rb') as handle:
        community_dict_lp = pk.load(handle)
    '''
    with open('community_dict_gm', 'rb') as handle:
        community_dict_gm = pk.load(handle)

    with open('community_dict_kc', 'rb') as handle:
        community_dict_kc = pk.load(handle)

    with open('community_dict_l2', 'rb') as handle:
        community_dict_l2 = pk.load(handle)
    '''
    '''
    community_dict_transf_lp = CommunityMeasures.align_cD_dataStructure(community_dict_lp, cD_algorithm='label_propagation')
    #community_dict_transf_gm = CommunityMeasures.align_cD_dataStructure(community_dict_gm, cD_algorithm='greedy_modularity')
    #community_dict_transf_kc = CommunityMeasures.align_cD_dataStructure(community_dict_kc, cD_algorithm='k_clique')
    #community_dict_transf_l2 = CommunityMeasures.align_cD_dataStructure(community_dict_l2, cD_algorithm='lais2')



    # --- Clean Communties ---#

    community_dict_clean_lp = CommunityMeasures.community_cleaning(community_dict_transf_lp,min_community_size=3)
    #community_dict_clean_gm = CommunityMeasures.community_cleaning(community_dict_transf_gm,min_community_size=3)
    #community_dict_clean_kc = CommunityMeasures.community_cleaning(community_dict_transf_kc,min_community_size=3)
    #community_dict_clean_l2 = CommunityMeasures.community_cleaning(community_dict_transf_l2,min_community_size=3)

### New File ###


#--- Identify TopD degree nodes of communities ---#

    community_dict_topD_lp = CommunityMeasures.identify_topDegree(community_dict_clean_lp, patentProject_graphs)
    #community_dict_topD_gm = CommunityMeasures.identify_topDegree(community_dict_clean_gm, patentProject_graphs)
    #community_dict_topD_kc = CommunityMeasures.identify_topDegree(community_dict_clean_kc, patentProject_graphs)
    #community_dict_topD_l2 = CommunityMeasures.identify_topDegree(community_dict_clean_l2, patentProject_graphs)


# --- Mainly for Lais2: merge communities with same topd ---#

    community_dict_topD_lp = CommunityMeasures.merging_completly_overlapping_communities(community_dict_topD_lp)
    #community_dict_topD_gm = CommunityMeasures.merging_completly_overlapping_communities(community_dict_topD_gm)
    #community_dict_topD_kc = CommunityMeasures.merging_completly_overlapping_communities(community_dict_topD_kc)
    #community_dict_topD_l2 = CommunityMeasures.merging_completly_overlapping_communities(community_dict_topD_l2)


#---  Community Tracing ---#

    ### Identify max number of possible community id's ###

    max_number_community_lp = CommunityMeasures.max_number_community(community_dict_topD_lp)
    #max_number_community_gm = CommunityMeasures.max_number_community(community_dict_topD_gm)
    #max_number_community_kc = CommunityMeasures.max_number_community(community_dict_topD_kc)
    #max_number_community_l2 = CommunityMeasures.max_number_community(community_dict_topD_l2)


    ### Tracing arrays ###

    tracingArray_lp, tracingArraySize_lp = CommunityMeasures.create_tracing_array(max_number_community_lp, community_dict_topD_lp, patentProject_graphs)
    #tracingArray_gm, tracingArraySize_gm = CommunityMeasures.create_tracing_array(max_number_community_gm, community_dict_topD_gm, patentProject_graphs)
    #tracingArray_kc, tracingArraySize_kc = CommunityMeasures.create_tracing_array(max_number_community_kc, community_dict_topD_kc, patentProject_graphs)
    #tracingArray_l2, tracingArraySize_l2 = CommunityMeasures.create_tracing_array(max_number_community_l2, community_dict_topD_l2, patentProject_graphs)



#---  Community Labeling ---#

    # Label Propagation #
    community_dict_labeled_lp, topD_communityID_association_perWindow_lp, topD_communityID_association_accumulated_lp = CommunityMeasures.community_labeling(tracingArray_lp, community_dict_topD_lp, patentProject_graphs)
    #community_dict_labeled_gm, topD_communityID_association_perWindow_gm, topD_communityID_association_accumulated_gm = CommunityMeasures.community_labeling(tracingArray_gm, community_dict_topD_gm, patentProject_graphs)
    #community_dict_labeled_kc, topD_communityID_association_perWindow_kc, topD_communityID_association_accumulated_kc = CommunityMeasures.community_labeling(tracingArray_kc, community_dict_topD_kc, patentProject_graphs)
    #community_dict_labeled_l2, topD_communityID_association_perWindow_l2, topD_communityID_association_accumulated_l2 = CommunityMeasures.community_labeling(tracingArray_l2, community_dict_topD_l2, patentProject_graphs)

    # --- Make sure community ids are unique in each window ---#

    CommunityMeasures.is_community_id_unique(community_dict_labeled_lp)
    #CommunityMeasures.is_community_id_unique(community_dict_labeled_gm)
    #CommunityMeasures.is_community_id_unique(community_dict_labeled_kc)
    #CommunityMeasures.is_community_id_unique(community_dict_labeled_l2)

#--- Community Visualization ---#
    # NOT REALLY NECESSARY

    visualizationArray_lp = CommunityMeasures.create_visualization_array(tracingArray_lp, topD_communityID_association_perWindow_lp)
    #visualizationArray_gm = CommunityMeasures.create_visualization_array(tracingArray_gm, topD_communityID_association_perWindow_gm)
    #visualizationArray_kc = CommunityMeasures.create_visualization_array(tracingArray_kc, topD_communityID_association_perWindow_kc)
    #visualizationArray_l2 = CommunityMeasures.create_visualization_array(tracingArray_l2, topD_communityID_association_perWindow_l2)

### new file (cDMeasures) ###


#--- Finding Recombination ---#

    ### Create Recombination dict - crisp ###

    recombination_dict_lp = CommunityMeasures.find_recombinations_crisp(community_dict_labeled_lp, patentProject_graphs)
    #recombination_dict_gm = CommunityMeasures.find_recombinations_crisp(community_dict_labeled_gm, patentProject_graphs)
    #recombination_dict_kc = CommunityMeasures.find_recombinations_overlapping(community_dict_labeled_kc, patentProject_graphs)
    #recombination_dict_l2 = CommunityMeasures.find_recombinations_overlapping(community_dict_labeled_l2, patentProject_graphs)

    ### Recombination Threshold ###
    # NOT REALLY NECESSARY

    recombination_dict_threshold_lp = CommunityMeasures.recombination_threshold_crisp(recombination_dict_lp, patentProject_graphs, 0.005)
    #recombination_dict_threshold_gm = CommunityMeasures.recombination_threshold_crisp(recombination_dict_gm, patentProject_graphs, 0.005)
    #recombination_dict_threshold_kc = CommunityMeasures.recombination_threshold_overlapping(recombination_dict_kc, patentProject_graphs, 0.005)
    #recombination_dict_threshold_l2 = CommunityMeasures.recombination_threshold_overlapping(recombination_dict_l2, patentProject_graphs, 0.005)

    ###  ###
    # NOT REALLY NECESSARY
        # a dict like cd_recombination_dic, but with an additional entry per recombination list. the additional entry indicates if a threshold was meet

    recombination_dict_enriched_lp = CommunityMeasures.enrich_recombinations_dic_with_thresholds_crips(recombination_dict_lp, recombination_dict_threshold_lp)
    #recombination_dict_enriched_gm = CommunityMeasures.enrich_recombinations_dic_with_thresholds_crips(recombination_dict_gm, recombination_dict_threshold_gm)
    #recombination_dict_enriched_kc = CommunityMeasures.enrich_recombinations_dic_with_thresholds_overlapping(recombination_dict_kc, recombination_dict_threshold_kc)
    #recombination_dict_enriched_l2 = CommunityMeasures.enrich_recombinations_dic_with_thresholds_overlapping(recombination_dict_l2, recombination_dict_threshold_l2)

### NEW FILE cDMeasureArrayTransform #########


    # ---  cleaning topD_dic ---#

    topD_communityID_association_accumulated_cleanID_lp = CommunityMeasures.create_cleaningIndex_associationAccumulated(topD_communityID_association_accumulated_lp, community_dict_topD_lp)
    #topD_communityID_association_accumulated_cleanID_gm = CommunityMeasures.create_cleaningIndex_associationAccumulated(topD_communityID_association_accumulated_gm, community_dict_topD_gm)
    #topD_communityID_association_accumulated_cleanID_kc = CommunityMeasures.create_cleaningIndex_associationAccumulated(topD_communityID_association_accumulated_kc, community_dict_topD_kc)
    #topD_communityID_association_accumulated_cleanID_l2 = CommunityMeasures.create_cleaningIndex_associationAccumulated(topD_communityID_association_accumulated_l2, community_dict_topD_l2)

    topD_communityID_association_accumulated_clean_lp = CommunityMeasures.cleaning_associationAccumulated(topD_communityID_association_accumulated_lp, topD_communityID_association_accumulated_cleanID_lp)
    #topD_communityID_association_accumulated_clean_gm = CommunityMeasures.cleaning_associationAccumulated(topD_communityID_association_accumulated_gm, topD_communityID_association_accumulated_cleanID_gm)
    #topD_communityID_association_accumulated_clean_kc = CommunityMeasures.cleaning_associationAccumulated(topD_communityID_association_accumulated_kc, topD_communityID_association_accumulated_cleanID_kc)
    #topD_communityID_association_accumulated_clean_l2 = CommunityMeasures.cleaning_associationAccumulated(topD_communityID_association_accumulated_l2, topD_communityID_association_accumulated_cleanID_l2)


    # --- Constructing Diffusion Array ---#

    lp_singleDiffusion_v2 = CommunityMeasures.single_diffusion_v2(topD_communityID_association_accumulated_clean_lp)
    #gm_singleDiffusion_v2 = CommunityMeasures.single_diffusion_v2(topD_communityID_association_accumulated_clean_gm)
    #kclique_singleDiffusion_v2 = CommunityMeasures.single_diffusion_v2(topD_communityID_association_accumulated_clean_kc)
    #lais2_singleDiffusion_v2 = CommunityMeasures.single_diffusion_v2(topD_communityID_association_accumulated_clean_l2)


    lp_recombination_diffusion_crip_count_v2, lp_recombination_diffusion_crip_fraction_v2, lp_recombination_diffusion_crip_threshold_v2, lp_recombination_diffusion_crip_columns = \
        CommunityMeasures.recombination_diffusion_crip_v2(topD_communityID_association_accumulated_clean_lp, recombination_dict_lp, patentProject_graphs)

    #gm_recombination_diffusion_crip_count_v2, gm_recombination_diffusion_crip_fraction_v2, gm_recombination_diffusion_crip_threshold_v2, gm_recombination_diffusion_crip_columns = \
    #    CommunityMeasures.recombination_diffusion_crip_v2(topD_communityID_association_accumulated_clean_gm, recombination_dict_gm, patentProject_graphs)

    #kclique_recombination_diffusion_overlapping_count_v2, kclique_recombination_diffusion_overlapping_fraction_v2, kclique_recombination_diffusion_overlapping_threshold_v2, kclique_recombination_diffusion_crip_columns = \
    #    CommunityMeasures.recombination_diffusion_overlapping_v2(topD_communityID_association_accumulated_clean_kc, recombination_dict_kc, patentProject_graphs)

    #lais2_recombination_diffusion_overlapping_count_v2, lais2_recombination_diffusion_overlapping_fraction_v2, lais2_recombination_diffusion_overlapping_threshold_v2, lais2_recombination_diffusion_crip_columns = \
    #    CommunityMeasures.recombination_diffusion_overlapping_v2(topD_communityID_association_accumulated_clean_l2, recombination_dict_l2, patentProject_graphs)
    '''
    '''
    filename = 'lp_singleDiffusion_v2'
    outfile = open(filename, 'wb')
    pk.dump(lp_singleDiffusion_v2, outfile)
    outfile.close()

    filename = 'lp_recombination_diffusion_crip_count_v2'
    outfile = open(filename, 'wb')
    pk.dump(lp_recombination_diffusion_crip_count_v2, outfile)
    outfile.close()

    filename = 'lp_recombination_diffusion_crip_columns'
    outfile = open(filename, 'wb')
    pk.dump(lp_recombination_diffusion_crip_columns, outfile)
    outfile.close()
    
    filename = 'community_dict_labeled_lp'
    outfile = open(filename, 'wb')
    pk.dump(community_dict_labeled_lp, outfile)
    outfile.close()
    
    '''

    with open('lp_singleDiffusion_v2', 'rb') as handle:
        lp_singleDiffusion_v2 = pk.load(handle)

    with open('lp_recombination_diffusion_crip_count_v2', 'rb') as handle:
        lp_recombination_diffusion_crip_count_v2 = pk.load(handle)

    with open('lp_recombination_diffusion_crip_columns', 'rb') as handle:
        lp_recombination_diffusion_crip_columns = pk.load(handle)

    with open('community_dict_labeled_lp', 'rb') as handle:
        community_dict_labeled_lp = pk.load(handle)
# NEW FILE ################

    import numpy as np
    import pandas as pd

    os.chdir('D:/Universitaet Mannheim/MMDS 7. Semester/Master Thesis/Outline/Data/Cleaning Robots')

    patent_lda_ipc = pd.read_csv('patent_lda_ipc.csv', quotechar='"', skipinitialspace=True)
    patent_lda_ipc = patent_lda_ipc.to_numpy()

    community_ids_all = []
    for window_id, window in community_dict_labeled_lp.items():
        for community in window:
            community_ids_all.append(community[1][0])

    column_length = max(community_ids_all)

    column_unique = np.unique(community_ids_all)
    column_unique.sort()
    column_unique_length = len(column_unique)

    print(community_ids_all)
    print(column_length)
    # print(column_unique)
    print(column_unique_length)

    # print(patent_lda_ipc[0])

    # get biggest community each window:

    community_size_dic = {}
    for window_id, window in community_dict_labeled_lp.items():
        size_list = []
        for community in window:
            size_list.append(len(community))
        community_size_dic[window_id] = max(size_list)

    # for i in range(column_unique_length):
    #    for window_id, window in lp_labeled.items():

    community_topicDist_dic = {}

    for window_id, window in community_dict_labeled_lp.items():
        window_list = []

        # for i in range(column_unique_length):

        for community in window:
            community_topics = np.zeros((len(community[0]), 330))
            topic_list = []

            # if community[1][0] == i:
            for patent in community[0]:
                paten_pos = np.where(patent_lda_ipc[:, 0] == patent)

                topic_list.append(patent_lda_ipc[paten_pos[0][0]][9:23])

            topic_list = [item for sublist in topic_list for item in sublist]
            topic_list = [x for x in topic_list if x == x]
            # print(topic_list)

            for i in range(0, len(topic_list), 2):

                for row in range(len(community_topics)):        # for all patents in the community

                    # print(community_topics[row, int(topic_list[i])])
                    if community_topics[row, int(topic_list[i])] == 0:
                        # print(community_topics[row, int(topic_list[i])])
                        community_topics[row, int(topic_list[i])] = topic_list[i + 1]
                        break

            community_topics = np.sum(community_topics, axis=0)

            window_list.append([community[1][0], list(community_topics)])

        community_topicDist_dic[window_id] = window_list

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
    print(community_topTopic_dic)

    print(sum(confidence_list) / len(confidence_list))


    # 1. recompute diffusion and recombination pattern arrays. Note that dimension will shrink, because one topic is refered to many times with
    # different communities
    #   1.1 transform column/recombination list to topics
    #   1.2 add columns with same topics together
    #   1.3 sort columns new

    topic_diffusion_array = np.zeros((len(community_topTopic_dic), 330))

    for i in range(len(topic_diffusion_array)):
        for j in range(len(topic_diffusion_array.T)):
            window = community_topTopic_dic['window_{}'.format(i*30)]
            pos_list = []
            for community in window:
                if community[1] == j:
                    pos_list.append(community[0])

            count = 0
            for pos in pos_list:
                count = count + lp_singleDiffusion_v2[i, pos]

            topic_diffusion_array[i,j] = count

    print(1+1)

    #lp_singleDiffusion_v2
    # 0 - 880 (column length) all number are ids of communities

    #lp_recombination_diffusion_crip_count_v2 lp_recombination_diffusion_crip_columns


    #2. compute average change of confidence in a community id
    #3. compute average confidence in window
    #4. average condience in general

