if __name__ == '__main__':

    # --- Import Libraries ---#
    print('\n#--- Import Libraries ---#\n')

    import pickle as pk




    import tqdm
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

    with open('community_dict_gm', 'rb') as handle:
        community_dict_gm = pk.load(handle)

    with open('community_dict_kc', 'rb') as handle:
        community_dict_kc = pk.load(handle)

    with open('community_dict_l2', 'rb') as handle:
        community_dict_l2 = pk.load(handle)

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
    community_dict_topD_gm = CommunityMeasures.identify_topDegree(community_dict_clean_gm,patentProject_graphs)
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

#--- Community Visualization ---#


    visualizationArray_lp = CommunityMeasures.create_visualization_array(community_dict_labeled_lp, topD_communityID_association_perWindow_lp)
    visualizationArray_gm = CommunityMeasures.create_visualization_array(community_dict_labeled_gm, topD_communityID_association_perWindow_gm)
    visualizationArray_kc = CommunityMeasures.create_visualization_array(community_dict_labeled_kc, topD_communityID_association_perWindow_kc)
    visualizationArray_l2 = CommunityMeasures.create_visualization_array(community_dict_labeled_l2, topD_communityID_association_perWindow_l2)

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


    filename = 'lp_topD_dic'
    outfile = open(filename, 'wb')
    pk.dump(lp_topD_dic, outfile)
    outfile.close()

    filename = 'gm_topD_dic'
    outfile = open(filename, 'wb')
    pk.dump(gm_topD_dic, outfile)
    outfile.close()

    filename = 'kclique_topD_dic'
    outfile = open(filename, 'wb')
    pk.dump(kclique_topD_dic, outfile)
    outfile.close()

    filename = 'lais2_topD_dic'
    outfile = open(filename, 'wb')
    pk.dump(lais2_topD_dic, outfile)
    outfile.close()



    filename = 'lp_topD'
    outfile = open(filename, 'wb')
    pk.dump(lp_topD, outfile)
    outfile.close()

    filename = 'gm_topD'
    outfile = open(filename, 'wb')
    pk.dump(gm_topD, outfile)
    outfile.close()

    filename = 'kclique_topD'
    outfile = open(filename, 'wb')
    pk.dump(kclique_topD, outfile)
    outfile.close()

    filename = 'lais2_topD'
    outfile = open(filename, 'wb')
    pk.dump(lais2_topD, outfile)
    outfile.close()


