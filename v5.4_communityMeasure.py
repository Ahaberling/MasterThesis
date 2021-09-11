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
    community_dict_lp = CommunityMeasures.detect_communities(patentProject_graphs, cD_algorithm='label_propagation', weight_bool=True)
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
    #'''
    with open('community_dict_lp', 'rb') as handle:
        community_dict_lp = pk.load(handle)
    
    with open('community_dict_gm', 'rb') as handle:
        community_dict_gm = pk.load(handle)

    with open('community_dict_kc', 'rb') as handle:
        community_dict_kc = pk.load(handle)

    with open('community_dict_l2', 'rb') as handle:
        community_dict_l2 = pk.load(handle)
    #'''

    community_dict_transf_lp = CommunityMeasures.align_cD_dataStructure(community_dict_lp, cD_algorithm='label_propagation')
    community_dict_transf_gm = CommunityMeasures.align_cD_dataStructure(community_dict_gm, cD_algorithm='greedy_modularity')
    community_dict_transf_kc = CommunityMeasures.align_cD_dataStructure(community_dict_kc, cD_algorithm='k_clique')
    community_dict_transf_l2 = CommunityMeasures.align_cD_dataStructure(community_dict_l2, cD_algorithm='lais2')



    # --- Clean Communties ---#

    community_dict_clean_lp = CommunityMeasures.community_cleaning(community_dict_transf_lp,min_community_size=3)
    community_dict_clean_gm = CommunityMeasures.community_cleaning(community_dict_transf_gm,min_community_size=3)
    community_dict_clean_kc = CommunityMeasures.community_cleaning(community_dict_transf_kc,min_community_size=3)
    community_dict_clean_l2 = CommunityMeasures.community_cleaning(community_dict_transf_l2,min_community_size=3)

### New File ###


#--- Identify TopD degree nodes of communities ---#

    community_dict_topD_lp = CommunityMeasures.identify_topDegree(community_dict_clean_lp, patentProject_graphs)
    community_dict_topD_gm = CommunityMeasures.identify_topDegree(community_dict_clean_gm, patentProject_graphs)
    community_dict_topD_kc = CommunityMeasures.identify_topDegree(community_dict_clean_kc, patentProject_graphs)
    community_dict_topD_l2 = CommunityMeasures.identify_topDegree(community_dict_clean_l2, patentProject_graphs)


# --- Mainly for Lais2: merge communities with same topd ---#

    community_dict_topD_lp = CommunityMeasures.merging_completly_overlapping_communities(community_dict_topD_lp)
    community_dict_topD_gm = CommunityMeasures.merging_completly_overlapping_communities(community_dict_topD_gm)
    community_dict_topD_kc = CommunityMeasures.merging_completly_overlapping_communities(community_dict_topD_kc)
    community_dict_topD_l2 = CommunityMeasures.merging_completly_overlapping_communities(community_dict_topD_l2)


#---  Community Tracing ---#

    ### Identify max number of possible community id's ###

    max_number_community_lp = CommunityMeasures.max_number_community(community_dict_topD_lp)
    max_number_community_gm = CommunityMeasures.max_number_community(community_dict_topD_gm)
    max_number_community_kc = CommunityMeasures.max_number_community(community_dict_topD_kc)
    max_number_community_l2 = CommunityMeasures.max_number_community(community_dict_topD_l2)


    ### Tracing arrays ###

    tracingArray_lp, tracingArraySize_lp = CommunityMeasures.create_tracing_array(max_number_community_lp, community_dict_topD_lp, patentProject_graphs)
    tracingArray_gm, tracingArraySize_gm = CommunityMeasures.create_tracing_array(max_number_community_gm, community_dict_topD_gm, patentProject_graphs)
    tracingArray_kc, tracingArraySize_kc = CommunityMeasures.create_tracing_array(max_number_community_kc, community_dict_topD_kc, patentProject_graphs)
    tracingArray_l2, tracingArraySize_l2 = CommunityMeasures.create_tracing_array(max_number_community_l2, community_dict_topD_l2, patentProject_graphs)



#---  Community Labeling ---#

    # Label Propagation #
    community_dict_labeled_lp, topD_communityID_association_perWindow_lp, topD_communityID_association_accumulated_lp = CommunityMeasures.community_labeling(tracingArray_lp, community_dict_topD_lp, patentProject_graphs)
    community_dict_labeled_gm, topD_communityID_association_perWindow_gm, topD_communityID_association_accumulated_gm = CommunityMeasures.community_labeling(tracingArray_gm, community_dict_topD_gm, patentProject_graphs)
    community_dict_labeled_kc, topD_communityID_association_perWindow_kc, topD_communityID_association_accumulated_kc = CommunityMeasures.community_labeling(tracingArray_kc, community_dict_topD_kc, patentProject_graphs)
    community_dict_labeled_l2, topD_communityID_association_perWindow_l2, topD_communityID_association_accumulated_l2 = CommunityMeasures.community_labeling(tracingArray_l2, community_dict_topD_l2, patentProject_graphs)

    # --- Make sure community ids are unique in each window ---#

    CommunityMeasures.is_community_id_unique(community_dict_labeled_lp)
    CommunityMeasures.is_community_id_unique(community_dict_labeled_gm)
    CommunityMeasures.is_community_id_unique(community_dict_labeled_kc)
    CommunityMeasures.is_community_id_unique(community_dict_labeled_l2)

#--- Community Visualization ---#
    # NOT REALLY NECESSARY

    visualizationArray_lp = CommunityMeasures.create_visualization_array(tracingArray_lp, topD_communityID_association_perWindow_lp)
    visualizationArray_gm = CommunityMeasures.create_visualization_array(tracingArray_gm, topD_communityID_association_perWindow_gm)
    visualizationArray_kc = CommunityMeasures.create_visualization_array(tracingArray_kc, topD_communityID_association_perWindow_kc)
    visualizationArray_l2 = CommunityMeasures.create_visualization_array(tracingArray_l2, topD_communityID_association_perWindow_l2)

### new file (cDMeasures) ###


#--- Finding Recombination ---#

    ### Create Recombination dict - crisp ###

    recombination_dict_lp = CommunityMeasures.find_recombinations_crisp(community_dict_labeled_lp, patentProject_graphs)
    recombination_dict_gm = CommunityMeasures.find_recombinations_crisp(community_dict_labeled_gm, patentProject_graphs)
    recombination_dict_kc = CommunityMeasures.find_recombinations_overlapping(community_dict_labeled_kc, patentProject_graphs)
    recombination_dict_l2 = CommunityMeasures.find_recombinations_overlapping(community_dict_labeled_l2, patentProject_graphs)

    ### Recombination Threshold ###
    # NOT REALLY NECESSARY

    recombination_dict_threshold_lp = CommunityMeasures.recombination_threshold_crisp(recombination_dict_lp, patentProject_graphs, 0.005)
    recombination_dict_threshold_gm = CommunityMeasures.recombination_threshold_crisp(recombination_dict_gm, patentProject_graphs, 0.005)
    recombination_dict_threshold_kc = CommunityMeasures.recombination_threshold_overlapping(recombination_dict_kc, patentProject_graphs, 0.005)
    recombination_dict_threshold_l2 = CommunityMeasures.recombination_threshold_overlapping(recombination_dict_l2, patentProject_graphs, 0.005)

    ###  ###
    # NOT REALLY NECESSARY
        # a dict like cd_recombination_dic, but with an additional entry per recombination list. the additional entry indicates if a threshold was meet

    recombination_dict_enriched_lp = CommunityMeasures.enrich_recombinations_dic_with_thresholds_crips(recombination_dict_lp, recombination_dict_threshold_lp)
    recombination_dict_enriched_gm = CommunityMeasures.enrich_recombinations_dic_with_thresholds_crips(recombination_dict_gm, recombination_dict_threshold_gm)
    recombination_dict_enriched_kc = CommunityMeasures.enrich_recombinations_dic_with_thresholds_overlapping(recombination_dict_kc, recombination_dict_threshold_kc)
    recombination_dict_enriched_l2 = CommunityMeasures.enrich_recombinations_dic_with_thresholds_overlapping(recombination_dict_l2, recombination_dict_threshold_l2)

### NEW FILE cDMeasureArrayTransform #########


    # ---  cleaning topD_dic ---#

    topD_communityID_association_accumulated_cleanID_lp = CommunityMeasures.create_cleaningIndex_associationAccumulated(topD_communityID_association_accumulated_lp, community_dict_topD_lp, patentProject_graphs)
    topD_communityID_association_accumulated_cleanID_gm = CommunityMeasures.create_cleaningIndex_associationAccumulated(topD_communityID_association_accumulated_gm, community_dict_topD_gm, patentProject_graphs)
    topD_communityID_association_accumulated_cleanID_kc = CommunityMeasures.create_cleaningIndex_associationAccumulated(topD_communityID_association_accumulated_kc, community_dict_topD_kc, patentProject_graphs)
    topD_communityID_association_accumulated_cleanID_l2 = CommunityMeasures.create_cleaningIndex_associationAccumulated(topD_communityID_association_accumulated_l2, community_dict_topD_l2, patentProject_graphs)

    topD_communityID_association_accumulated_clean_lp = CommunityMeasures.cleaning_associationAccumulated(topD_communityID_association_accumulated_lp, topD_communityID_association_accumulated_cleanID_lp)
    topD_communityID_association_accumulated_clean_gm = CommunityMeasures.cleaning_associationAccumulated(topD_communityID_association_accumulated_gm, topD_communityID_association_accumulated_cleanID_gm)
    topD_communityID_association_accumulated_clean_kc = CommunityMeasures.cleaning_associationAccumulated(topD_communityID_association_accumulated_kc, topD_communityID_association_accumulated_cleanID_kc)
    topD_communityID_association_accumulated_clean_l2 = CommunityMeasures.cleaning_associationAccumulated(topD_communityID_association_accumulated_l2, topD_communityID_association_accumulated_cleanID_l2)


    # --- Constructing Diffusion Array ---#

    lp_singleDiffusion_v2 = CommunityMeasures.single_diffusion_v2(topD_communityID_association_accumulated_clean_lp)
    gm_singleDiffusion_v2 = CommunityMeasures.single_diffusion_v2(topD_communityID_association_accumulated_clean_gm)
    kclique_singleDiffusion_v2 = CommunityMeasures.single_diffusion_v2(topD_communityID_association_accumulated_clean_kc)
    lais2_singleDiffusion_v2 = CommunityMeasures.single_diffusion_v2(topD_communityID_association_accumulated_clean_l2)


    lp_recombination_diffusion_crip_count_v2, lp_recombination_diffusion_crip_fraction_v2, lp_recombination_diffusion_crip_threshold_v2, lp_recombination_diffusion_crip_columns = \
        CommunityMeasures.recombination_diffusion_crip_v2(topD_communityID_association_accumulated_clean_lp, recombination_dict_lp, patentProject_graphs)

    import numpy as np
    print(len(lp_recombination_diffusion_crip_count_v2.T))
    columSum_vec = np.sum(lp_recombination_diffusion_crip_count_v2, axis=0)
    print(len(columSum_vec))
    pos = np.where(columSum_vec == 0)
    print(pos)

    gm_recombination_diffusion_crip_count_v2, gm_recombination_diffusion_crip_fraction_v2, gm_recombination_diffusion_crip_threshold_v2, gm_recombination_diffusion_crip_columns = \
        CommunityMeasures.recombination_diffusion_crip_v2(topD_communityID_association_accumulated_clean_gm, recombination_dict_gm, patentProject_graphs)

    kclique_recombination_diffusion_overlapping_count_v2, kclique_recombination_diffusion_overlapping_fraction_v2, kclique_recombination_diffusion_overlapping_threshold_v2, kclique_recombination_diffusion_crip_columns = \
        CommunityMeasures.recombination_diffusion_overlapping_v2(topD_communityID_association_accumulated_clean_kc, recombination_dict_kc, patentProject_graphs)

    lais2_recombination_diffusion_overlapping_count_v2, lais2_recombination_diffusion_overlapping_fraction_v2, lais2_recombination_diffusion_overlapping_threshold_v2, lais2_recombination_diffusion_crip_columns = \
        CommunityMeasures.recombination_diffusion_overlapping_v2(topD_communityID_association_accumulated_clean_l2, recombination_dict_l2, patentProject_graphs)

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
        
    filename = 'community_dict_labeled_lp'
    outfile = open(filename, 'wb')
    pk.dump(community_dict_labeled_lp, outfile)
    outfile.close()
    
    

    with open('lp_singleDiffusion_v2', 'rb') as handle:
        lp_singleDiffusion_v2 = pk.load(handle)

    with open('lp_recombination_diffusion_crip_count_v2', 'rb') as handle:
        lp_recombination_diffusion_crip_count_v2 = pk.load(handle)

    with open('lp_recombination_diffusion_crip_columns', 'rb') as handle:
        lp_recombination_diffusion_crip_columns = pk.load(handle)

    with open('community_dict_labeled_lp', 'rb') as handle:
        community_dict_labeled_lp = pk.load(handle)

    with open('recombination_dict_lp', 'rb') as handle:
        recombination_dict_lp = pk.load(handle)
    '''




# NEW FILE ################

    import numpy as np
    import pandas as pd

    os.chdir('D:/Universitaet Mannheim/MMDS 7. Semester/Master Thesis/Outline/Data/Cleaning Robots')

    patent_lda_ipc = pd.read_csv('patent_lda_ipc.csv', quotechar='"', skipinitialspace=True)
    patent_lda_ipc = patent_lda_ipc.to_numpy()



    topicDistriburionOfCommunities_dict_lp = CommunityMeasures.creat_dict_topicDistriburionOfCommunities(community_dict_labeled_lp, patent_lda_ipc)
    topicDistriburionOfCommunities_dict_gm = CommunityMeasures.creat_dict_topicDistriburionOfCommunities(community_dict_labeled_gm, patent_lda_ipc)
    topicDistriburionOfCommunities_dict_kc = CommunityMeasures.creat_dict_topicDistriburionOfCommunities(community_dict_labeled_kc, patent_lda_ipc)
    topicDistriburionOfCommunities_dict_l2 = CommunityMeasures.creat_dict_topicDistriburionOfCommunities(community_dict_labeled_l2, patent_lda_ipc)

    communityTopicAssociation_dict_lp, avg_confidence_lp = CommunityMeasures.create_dict_communityTopicAssociation(topicDistriburionOfCommunities_dict_lp)
    communityTopicAssociation_dict_gm, avg_confidence_gm = CommunityMeasures.create_dict_communityTopicAssociation(topicDistriburionOfCommunities_dict_gm)
    communityTopicAssociation_dict_kc, avg_confidence_kc = CommunityMeasures.create_dict_communityTopicAssociation(topicDistriburionOfCommunities_dict_kc)
    communityTopicAssociation_dict_l2, avg_confidence_l2 = CommunityMeasures.create_dict_communityTopicAssociation(topicDistriburionOfCommunities_dict_l2)

    diffusionArray_Topics_lp, diffusionArray_Topics_lp_columns = CommunityMeasures.create_diffusionArray_Topics(communityTopicAssociation_dict_lp)
    diffusionArray_Topics_gm, diffusionArray_Topics_gm_columns = CommunityMeasures.create_diffusionArray_Topics(communityTopicAssociation_dict_gm)
    diffusionArray_Topics_kc, diffusionArray_Topics_kc_columns = CommunityMeasures.create_diffusionArray_Topics(communityTopicAssociation_dict_kc)
    diffusionArray_Topics_l2, diffusionArray_Topics_l2_columns = CommunityMeasures.create_diffusionArray_Topics(communityTopicAssociation_dict_l2)

    # they cant be the same, because lp_singleDiffusion_v2 measure the lifetime of communities and not topic diffusion.
    # in lp_singleDiffusion_v2 subset of communities are listed as well, if they were swallowed by bigger communities.
    # this is irrelevant for topics.

    recombination_dict_Topics_lp = CommunityMeasures.created_recombination_dict_Topics_crisp(communityTopicAssociation_dict_lp, recombination_dict_lp)
    recombination_dict_Topics_gm = CommunityMeasures.created_recombination_dict_Topics_crisp(communityTopicAssociation_dict_gm, recombination_dict_gm)
    recombination_dict_Topics_kc = CommunityMeasures.created_recombination_dict_Topics_overlap(communityTopicAssociation_dict_kc, recombination_dict_kc)
    recombination_dict_Topics_l2 = CommunityMeasures.created_recombination_dict_Topics_overlap(communityTopicAssociation_dict_l2, recombination_dict_l2)

    CommunityMeasures.doubleCheck_recombination_dict_Topics_crisp(recombination_dict_Topics_lp, recombination_dict_lp, communityTopicAssociation_dict_lp)
    CommunityMeasures.doubleCheck_recombination_dict_Topics_crisp(recombination_dict_Topics_gm, recombination_dict_gm, communityTopicAssociation_dict_gm)
    CommunityMeasures.doubleCheck_recombination_dict_Topics_overlap(recombination_dict_Topics_kc, recombination_dict_kc, communityTopicAssociation_dict_kc)
    CommunityMeasures.doubleCheck_recombination_dict_Topics_overlap(recombination_dict_Topics_l2, recombination_dict_l2, communityTopicAssociation_dict_l2)

    recombinationArray_Topics_lp, recombinationArray_Topics_lp_columns = CommunityMeasures.create_recombinationArray_Topics(recombination_dict_Topics_lp)
    recombinationArray_Topics_gm, recombinationArray_Topics_gm_columns = CommunityMeasures.create_recombinationArray_Topics(recombination_dict_Topics_gm)
    recombinationArray_Topics_kc, recombinationArray_Topics_kc_columns = CommunityMeasures.create_recombinationArray_Topics(recombination_dict_Topics_kc)
    recombinationArray_Topics_l2, recombinationArray_Topics_l2_columns = CommunityMeasures.create_recombinationArray_Topics(recombination_dict_Topics_l2)

    filename = 'diffusionArray_Topics_lp'
    outfile = open(filename, 'wb')
    pk.dump(diffusionArray_Topics_lp, outfile)
    outfile.close()

    filename = 'diffusionArray_Topics_lp_columns'
    outfile = open(filename, 'wb')
    pk.dump(diffusionArray_Topics_lp_columns, outfile)
    outfile.close()

    filename = 'diffusionArray_Topics_gm'
    outfile = open(filename, 'wb')
    pk.dump(diffusionArray_Topics_gm, outfile)
    outfile.close()

    filename = 'diffusionArray_Topics_gm_columns'
    outfile = open(filename, 'wb')
    pk.dump(diffusionArray_Topics_gm_columns, outfile)
    outfile.close()

    filename = 'diffusionArray_Topics_kc'
    outfile = open(filename, 'wb')
    pk.dump(diffusionArray_Topics_kc, outfile)
    outfile.close()

    filename = 'diffusionArray_Topics_kc_columns'
    outfile = open(filename, 'wb')
    pk.dump(diffusionArray_Topics_kc_columns, outfile)
    outfile.close()

    filename = 'diffusionArray_Topics_l2'
    outfile = open(filename, 'wb')
    pk.dump(diffusionArray_Topics_l2, outfile)
    outfile.close()

    filename = 'diffusionArray_Topics_l2_columns'
    outfile = open(filename, 'wb')
    pk.dump(diffusionArray_Topics_l2_columns, outfile)
    outfile.close()

    filename = 'recombinationArray_Topics_lp'
    outfile = open(filename, 'wb')
    pk.dump(recombinationArray_Topics_lp, outfile)
    outfile.close()

    filename = 'recombinationArray_Topics_lp_columns'
    outfile = open(filename, 'wb')
    pk.dump(recombinationArray_Topics_lp_columns, outfile)
    outfile.close()

    filename = 'recombinationArray_Topics_gm'
    outfile = open(filename, 'wb')
    pk.dump(recombinationArray_Topics_gm, outfile)
    outfile.close()

    filename = 'recombinationArray_Topics_gm_columns'
    outfile = open(filename, 'wb')
    pk.dump(recombinationArray_Topics_gm_columns, outfile)
    outfile.close()

    filename = 'recombinationArray_Topics_kc'
    outfile = open(filename, 'wb')
    pk.dump(recombinationArray_Topics_kc, outfile)
    outfile.close()

    filename = 'recombinationArray_Topics_kc_columns'
    outfile = open(filename, 'wb')
    pk.dump(recombinationArray_Topics_kc_columns, outfile)
    outfile.close()

    filename = 'recombinationArray_Topics_l2'
    outfile = open(filename, 'wb')
    pk.dump(recombinationArray_Topics_l2, outfile)
    outfile.close()

    filename = 'recombinationArray_Topics_l2_columns'
    outfile = open(filename, 'wb')
    pk.dump(recombinationArray_Topics_l2_columns, outfile)
    outfile.close()

    print(1+1)
    print(1+1)
    print(1+1)
    print(1+1)

    #2. compute average change of confidence in a community id
    #3. compute average confidence in window
    #4. average condience in general

