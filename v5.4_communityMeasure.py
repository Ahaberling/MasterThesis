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
    import numpy as np
    from utilities_old.my_measure_utils import CommunityMeasures
    '''
    
    import seaborn as sns
    import matplotlib.pyplot as plt

    f, ax = plt.subplots()
    sns.heatmap(np.zeros((3,3)), cbar_kws={'label': 'Component Count in Window', 'ticks': [0, 1]}, vmax=1,
            vmin=0) #, cmap="YlGnBu") #, annot=True, fmt="d", linewidths=.5, ax=ax)
    #colorbar = ax.collections[0].colorbar
    #colorbar.set_ticks([0,1,0.2])
    #plt.yticks(range(0,80,10))
    #ax.set_xticklabels(range(20,30))
    #ax.set_yticklabels(range(0,80,10))
    plt.xlabel("Knowledge Component ID")
    plt.ylabel("Sliding Window ID ")
    plt.show()
    plt.close()
    '''

    '''
    import numpy as np
    
    community_dict_lp, modularity_dict_lp = CommunityMeasures.detect_communities(patentProject_graphs, cD_algorithm='label_propagation', weight_bool=True)
    #print(community_dict_lp['window_0'])
    #print(modularity_dict_lp['window_0'])                   # 0.8547171828975307
    modularity_lp = list(modularity_dict_lp.values())
    print('average modularity lp: ', np.mean(modularity_lp))       # 0.7322156796307513

    community_dict_gm, modularity_dict_gm = CommunityMeasures.detect_communities(patentProject_graphs, cD_algorithm='greedy_modularity')
    #print(community_dict_gm['window_0'])
   # print(modularity_dict_gm['window_0'])                   # -0.05733820147057286
    modularity_gm = []
    for i in modularity_dict_gm.values():
        modularity_gm.append(i[2])
    #print(helper[0])
    print(np.mean(modularity_gm))                                  # 0.06982563520602868

    community_dict_kc, modularity_dict_kc = CommunityMeasures.detect_communities(patentProject_graphs, cD_algorithm='k_clique', k_clique_size=3)
    #print(modularity_dict_kc['window_0'])                   # 0.19167160178624776 newman_girvan_modularity
    modularity_kc = []                                             # 0.048218915888596885 link
    for i in modularity_dict_kc.values():
        modularity_kc.append(i[2])
    #print(helper[0])
    print(np.mean(modularity_kc))                                  # 0.3798751013811473  newman_girvan_modularity
                                                            # 0.09104628306679274 link

    community_dict_l2, modularity_dict_l2 = CommunityMeasures.detect_communities(patentProject_graphs, cD_algorithm='lais2')
    #print(modularity_dict_l2['window_0'])                   # 0.796357503685926 newman_girvan_modularity
    modularity_l2 = []                                             # 0.1350135445281584 link
    for i in modularity_dict_l2.values():
        modularity_l2.append(i[2])
    #print(helper[0])
    print(np.mean(modularity_l2))                                  # 0.4571296942392671 newman_girvan_modularity
                                                            # 0.1041329499095561 link
    import matplotlib.pyplot as plt

    

    fig, ax = plt.subplots()
    ax.plot(range(len(patentProject_graphs)), modularity_lp, color='darkblue', label='Label Propagation')
    ax.plot(range(len(patentProject_graphs)), modularity_gm, color='darkred', label='Greedy Modularity')
    ax.plot(range(len(patentProject_graphs)), modularity_kc, color='darkgreen', label='K-Clique')
    ax.plot(range(len(patentProject_graphs)), modularity_l2, color='black', label='Lais2')
    ax.set_ylim([-0.2, 1.2])
    plt.legend(loc='upper left')
    plt.xlabel("Patent Network Representation of Sliding Windows")
    plt.ylabel("Modularity")
    #plt.show()
    os.chdir('D:/Universitaet Mannheim/MMDS 7. Semester/Master Thesis/Outline/Plots')
    plt.savefig('Modularities.png')
    os.chdir('D:/Universitaet Mannheim/MMDS 7. Semester/Master Thesis/Outline/Data/Cleaning Robots')
    plt.close()
    
    #community_dict_lp = CommunityMeasures.detect_communities(patentProject_graphs, cD_algorithm='label_propagation', weight_bool=True)
    #community_dict_gm = CommunityMeasures.detect_communities(patentProject_graphs, cD_algorithm='greedy_modularity')
    #community_dict_kc = CommunityMeasures.detect_communities(patentProject_graphs, cD_algorithm='k_clique', k_clique_size=3)
    #community_dict_l2 = CommunityMeasures.detect_communities(patentProject_graphs, cD_algorithm='lais2')
    '''
    '''
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
    #[1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 38 39 40 44 45 48 53 55]
    #[6806 1992 1340  930  684  480  351  249  198  160  104  101   72   58 64   34   32   29   22    9   16   10    6    8    8    1    9    6 7    3    3    3    2    2    5    1    2    1    1    1    1    1 1    1]
    community_dict_transf_gm = CommunityMeasures.align_cD_dataStructure(community_dict_gm, cD_algorithm='greedy_modularity')
    #[1 2 3 4 5]
    #[30216  4767   829    48    13]
    community_dict_transf_kc = CommunityMeasures.align_cD_dataStructure(community_dict_kc, cD_algorithm='k_clique')
    #[  0   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19 20  21  22  23  24  25  26  27  28  29  31  32  33  34  36  37  39  40 42  53  54  64  66  73  77  78  80  81  86  87  90  91 102 104 111 112 117 125 128 130 135 136 137 139 140 146 152 156 166 201]
    #[3 24 21 46 53 74 61 67 39 48 43 29 16 16 12 19 16  7  7 10 10  8  7 12 12  5  1  1  1  2  1  2  1  2  1  2  1  3  1  1  2  1  1  1  1  1  1  1 2  1  1  1  1  1  1  1  1  1  1  1  1  3  1  1  1  1  1  1]
    community_dict_transf_l2 = CommunityMeasures.align_cD_dataStructure(community_dict_l2, cD_algorithm='lais2')
    # [ 3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 36 37 38 39 40 45 49 50]
    # [  2  24  21  20  43  65  95 103  86  76  77  57  52  72  56  59  33  39 23  26  26  16  21  17   9  15   5   6   3   6   3   3   3   2   1   2  1   2   1   2]



    '''
    lengthList = []
    for windowid, window in community_dict_transf_lp.items():
        for community in window:
            lengthList.append(len(community))
            if len(community) <= 2:
                print(windowid)
                print(community)

    print(lengthList)
    import numpy as np
    val, count = np.unique(lengthList, return_counts=True)
    print(val)
    print(count)
    '''

    # --- Clean Communties ---#

    community_dict_clean_lp, communities_removed_lp = CommunityMeasures.community_cleaning(community_dict_transf_lp,min_community_size=3)
    community_dict_clean_gm, communities_removed_gm = CommunityMeasures.community_cleaning(community_dict_transf_gm,min_community_size=3)
    community_dict_clean_kc, communities_removed_kc = CommunityMeasures.community_cleaning(community_dict_transf_kc,min_community_size=3)
    community_dict_clean_l2, communities_removed_l2 = CommunityMeasures.community_cleaning(community_dict_transf_l2,min_community_size=3)

    community_length_list = []
    community_size_list = []
    for window_id, window in community_dict_clean_lp.items():
        community_length_list.append(len(window))

        for community in window:
            community_size_list.append(len(community))

    import statistics

    print('Average Size of communities in Window LP: ', np.mean(community_size_list))
    print('Average Number of communities in Window LP: ', np.mean(community_length_list))
    print('Median Number of communities in Window LP: ', np.median(community_length_list))
    print('Mode Number of communities in Window LP: ', statistics.mode(community_length_list))
    print('Max Number of communities in Window LP: ', max(community_length_list))
    print('Min Number of communities in Window LP: ', min(community_length_list))

    community_length_list = []
    community_size_list = []
    for window_id, window in community_dict_clean_gm.items():
        community_length_list.append(len(window))

        for community in window:
            community_size_list.append(len(community))

    print('Average Size of communities in Window gm: ', np.mean(community_size_list))
    print('Average Number of communities in Window gm: ', np.mean(community_length_list))
    print('Median Number of communities in Window gm: ', np.median(community_length_list))
    print('Mode Number of communities in Window gm: ', statistics.mode(community_length_list))
    print('Max Number of communities in Window gm: ', max(community_length_list))
    print('Min Number of communities in Window gm: ', min(community_length_list))

    community_length_list = []
    community_size_list = []
    for window_id, window in community_dict_clean_kc.items():
        community_length_list.append(len(window))

        for community in window:
            community_size_list.append(len(community))
    print('Average Size of communities in Window kc: ', np.mean(community_size_list))
    print('Average Number of communities in Window kc: ', np.mean(community_length_list))
    print('Median Number of communities in Window kc: ', np.median(community_length_list))
    print('Mode Number of communities in Window kc: ', statistics.mode(community_length_list))
    print('Max Number of communities in Window kc: ', max(community_length_list))
    print('Min Number of communities in Window kc: ', min(community_length_list))

    community_length_list = []
    community_size_list = []
    for window_id, window in community_dict_clean_l2.items():
        community_length_list.append(len(window))

        for community in window:
            community_size_list.append(len(community))
    print('Average Size of communities in Window L2: ', np.mean(community_size_list))
    print('Average Number of communities in Window L2: ', np.mean(community_length_list))
    print('Median Number of communities in Window L2: ', np.median(community_length_list))
    print('Mode Number of communities in Window L2: ', statistics.mode(community_length_list))
    print('Max Number of communities in Window L2: ', max(community_length_list))
    print('Min Number of communities in Window L2: ', min(community_length_list))


    #--- removed communities
    print('Average Number of removed communities lp: ', communities_removed_lp / len(community_dict_clean_lp))

    print('Average Number of removed communities gm: ', communities_removed_gm / len(community_dict_clean_lp))

    print('Average Number of removed communities kc: ', communities_removed_kc / len(community_dict_clean_lp))
    print('actually 0. Only 3 communities of size zero were filtered out')

    print('Average Number of removed communities l2: ', communities_removed_l2 / len(community_dict_clean_lp))



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

    #tracingArray_lp, tracingArraySize_lp = CommunityMeasures.create_tracing_array(max_number_community_lp, community_dict_topD_lp, patentProject_graphs)
    #tracingArray_gm, tracingArraySize_gm = CommunityMeasures.create_tracing_array(max_number_community_gm, community_dict_topD_gm, patentProject_graphs)
    #tracingArray_kc, tracingArraySize_kc = CommunityMeasures.create_tracing_array(max_number_community_kc, community_dict_topD_kc, patentProject_graphs)
    #tracingArray_l2, tracingArraySize_l2 = CommunityMeasures.create_tracing_array(max_number_community_l2, community_dict_topD_l2, patentProject_graphs)

    # CORRECT
    tracingArray_lp = CommunityMeasures.create_tracing_array(max_number_community_lp, community_dict_topD_lp, patentProject_graphs)
    tracingArray_gm = CommunityMeasures.create_tracing_array(max_number_community_gm, community_dict_topD_gm, patentProject_graphs)
    tracingArray_kc = CommunityMeasures.create_tracing_array(max_number_community_kc, community_dict_topD_kc, patentProject_graphs)
    tracingArray_l2 = CommunityMeasures.create_tracing_array(max_number_community_l2, community_dict_topD_l2, patentProject_graphs)



#---  Community Labeling ---#

    # Label Propagation #
    # CORRECT. 'topD_communityID_association_accumulated' PROBABLY USELESS
    community_dict_labeled_lp, topD_communityID_association_perWindow_lp, topD_communityID_association_accumulated_lp = CommunityMeasures.community_labeling(tracingArray_lp, community_dict_topD_lp, patentProject_graphs)
    community_dict_labeled_gm, topD_communityID_association_perWindow_gm, topD_communityID_association_accumulated_gm = CommunityMeasures.community_labeling(tracingArray_gm, community_dict_topD_gm, patentProject_graphs)
    community_dict_labeled_kc, topD_communityID_association_perWindow_kc, topD_communityID_association_accumulated_kc = CommunityMeasures.community_labeling(tracingArray_kc, community_dict_topD_kc, patentProject_graphs)
    community_dict_labeled_l2, topD_communityID_association_perWindow_l2, topD_communityID_association_accumulated_l2 = CommunityMeasures.community_labeling(tracingArray_l2, community_dict_topD_l2, patentProject_graphs)

    # --- Make sure community ids are unique in each window ---#

    # CORRECT
    CommunityMeasures.is_community_id_unique(community_dict_labeled_lp)
    CommunityMeasures.is_community_id_unique(community_dict_labeled_gm)
    CommunityMeasures.is_community_id_unique(community_dict_labeled_kc)
    CommunityMeasures.is_community_id_unique(community_dict_labeled_l2)

#--- Community Visualization ---#

    # CORRECT
    visualizationArray_lp = CommunityMeasures.create_visualization_array(tracingArray_lp, topD_communityID_association_perWindow_lp)
    visualizationArray_gm = CommunityMeasures.create_visualization_array(tracingArray_gm, topD_communityID_association_perWindow_gm)
    visualizationArray_kc = CommunityMeasures.create_visualization_array(tracingArray_kc, topD_communityID_association_perWindow_kc)
    visualizationArray_l2 = CommunityMeasures.create_visualization_array(tracingArray_l2, topD_communityID_association_perWindow_l2)

### new file (cDMeasures) ###


#--- Finding Recombination ---#

    ### Create Recombination dict - crisp ###

    # CORRECT
    recombination_dict_lp = CommunityMeasures.find_recombinations_crisp(community_dict_labeled_lp, patentProject_graphs)
    recombination_dict_gm = CommunityMeasures.find_recombinations_crisp(community_dict_labeled_gm, patentProject_graphs)
    recombination_dict_kc = CommunityMeasures.find_recombinations_overlapping(community_dict_labeled_kc, patentProject_graphs)
    recombination_dict_l2 = CommunityMeasures.find_recombinations_overlapping(community_dict_labeled_l2, patentProject_graphs)

    ### Recombination Threshold ###
    '''
    # # CORRECT BUT MAYBE NOT REALLY NECESSARY
    recombination_dict_threshold_lp = CommunityMeasures.recombination_threshold_crisp(recombination_dict_lp, patentProject_graphs, 0.005)
    recombination_dict_threshold_gm = CommunityMeasures.recombination_threshold_crisp(recombination_dict_gm, patentProject_graphs, 0.005)
    recombination_dict_threshold_kc = CommunityMeasures.recombination_threshold_overlapping(recombination_dict_kc, patentProject_graphs, 0.005)
    recombination_dict_threshold_l2 = CommunityMeasures.recombination_threshold_overlapping(recombination_dict_l2, patentProject_graphs, 0.005)
    '''
    ###  ###
    # # CORRECT BUT MAYBE NOT REALLY NECESSARY
        # a dict like cd_recombination_dic, but with an additional entry per recombination list. the additional entry indicates if a threshold was meet
    '''
    recombination_dict_enriched_lp = CommunityMeasures.enrich_recombinations_dic_with_thresholds_crips(recombination_dict_lp, recombination_dict_threshold_lp)
    recombination_dict_enriched_gm = CommunityMeasures.enrich_recombinations_dic_with_thresholds_crips(recombination_dict_gm, recombination_dict_threshold_gm)
    recombination_dict_enriched_kc = CommunityMeasures.enrich_recombinations_dic_with_thresholds_overlapping(recombination_dict_kc, recombination_dict_threshold_kc)
    recombination_dict_enriched_l2 = CommunityMeasures.enrich_recombinations_dic_with_thresholds_overlapping(recombination_dict_l2, recombination_dict_threshold_l2)
    '''
### NEW FILE cDMeasureArrayTransform #########

    # PROBABLY DONT NEED THIS
    # ---  cleaning topD_dic ---#
    # PROBABLY USELESS
    '''
    topD_communityID_association_accumulated_cleanID_lp = CommunityMeasures.create_cleaningIndex_associationAccumulated(topD_communityID_association_accumulated_lp, community_dict_topD_lp, patentProject_graphs)
    topD_communityID_association_accumulated_cleanID_gm = CommunityMeasures.create_cleaningIndex_associationAccumulated(topD_communityID_association_accumulated_gm, community_dict_topD_gm, patentProject_graphs)
    topD_communityID_association_accumulated_cleanID_kc = CommunityMeasures.create_cleaningIndex_associationAccumulated(topD_communityID_association_accumulated_kc, community_dict_topD_kc, patentProject_graphs)
    topD_communityID_association_accumulated_cleanID_l2 = CommunityMeasures.create_cleaningIndex_associationAccumulated(topD_communityID_association_accumulated_l2, community_dict_topD_l2, patentProject_graphs)

    topD_communityID_association_accumulated_clean_lp = CommunityMeasures.cleaning_associationAccumulated(topD_communityID_association_accumulated_lp, topD_communityID_association_accumulated_cleanID_lp)
    topD_communityID_association_accumulated_clean_gm = CommunityMeasures.cleaning_associationAccumulated(topD_communityID_association_accumulated_gm, topD_communityID_association_accumulated_cleanID_gm)
    topD_communityID_association_accumulated_clean_kc = CommunityMeasures.cleaning_associationAccumulated(topD_communityID_association_accumulated_kc, topD_communityID_association_accumulated_cleanID_kc)
    topD_communityID_association_accumulated_clean_l2 = CommunityMeasures.cleaning_associationAccumulated(topD_communityID_association_accumulated_l2, topD_communityID_association_accumulated_cleanID_l2)
    '''

    # --- Constructing Diffusion Array ---#
    # PROBABLY USELESS
    '''
    lp_singleDiffusion_v2 = CommunityMeasures.single_diffusion_v2(topD_communityID_association_accumulated_clean_lp)
    gm_singleDiffusion_v2 = CommunityMeasures.single_diffusion_v2(topD_communityID_association_accumulated_clean_gm)
    kclique_singleDiffusion_v2 = CommunityMeasures.single_diffusion_v2(topD_communityID_association_accumulated_clean_kc)
    lais2_singleDiffusion_v2 = CommunityMeasures.single_diffusion_v2(topD_communityID_association_accumulated_clean_l2)
    '''
    #PROBABLY USELESS
    '''
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


    # CORRECT
    #window: [community id [topic distribution], community id [...], ... window: ...
    topicDistriburionOfCommunities_dict_lp = CommunityMeasures.creat_dict_topicDistriburionOfCommunities(community_dict_labeled_lp, patent_lda_ipc)
    topicDistriburionOfCommunities_dict_gm = CommunityMeasures.creat_dict_topicDistriburionOfCommunities(community_dict_labeled_gm, patent_lda_ipc)
    topicDistriburionOfCommunities_dict_kc = CommunityMeasures.creat_dict_topicDistriburionOfCommunities(community_dict_labeled_kc, patent_lda_ipc)
    topicDistriburionOfCommunities_dict_l2 = CommunityMeasures.creat_dict_topicDistriburionOfCommunities(community_dict_labeled_l2, patent_lda_ipc)

    # CORRECT
    # COMMUNITY ID , MOST DOMINANT TOPIC, CONFIDENCE
    communityTopicAssociation_dict_lp, avg_confidence_lp = CommunityMeasures.create_dict_communityTopicAssociation(topicDistriburionOfCommunities_dict_lp)
    communityTopicAssociation_dict_gm, avg_confidence_gm = CommunityMeasures.create_dict_communityTopicAssociation(topicDistriburionOfCommunities_dict_gm)
    communityTopicAssociation_dict_kc, avg_confidence_kc = CommunityMeasures.create_dict_communityTopicAssociation(topicDistriburionOfCommunities_dict_kc)
    communityTopicAssociation_dict_l2, avg_confidence_l2 = CommunityMeasures.create_dict_communityTopicAssociation(topicDistriburionOfCommunities_dict_l2)

    # list of confidence averages for every topic
    avgConfidence_perTopic = []
    for i in range(330):
        confidences_inTopic = []
        for window in communityTopicAssociation_dict_lp.values():
            for commuTopic in window:
                #[community id, topic, confidence]
                topic = list(commuTopic)[1]
                confidence = list(commuTopic)[2]
                if topic == i:
                    confidences_inTopic.append(confidence)
        avgConfidence_perTopic.append(np.mean(confidences_inTopic))

    avgConfidence_perTopic = [x for x in avgConfidence_perTopic if x == x]
    print(avgConfidence_perTopic)

    print('Averaged confidence average over all topics -LP: ', np.mean(avgConfidence_perTopic))
    print('Median confidence average over all topics -LP: ', np.median(avgConfidence_perTopic))
    print('Mode confidence average over all topics -LP: ', statistics.mode(avgConfidence_perTopic))
    print('Max confidence average over all topics -LP: ', max(avgConfidence_perTopic))
    print('Min confidence average over all topics -LP: ', min(avgConfidence_perTopic))

    avgConfidence_perTopic = []
    for i in range(330):
        confidences_inTopic = []
        for window in communityTopicAssociation_dict_gm.values():
            for commuTopic in window:
                #[community id, topic, confidence]
                topic = list(commuTopic)[1]
                confidence = list(commuTopic)[2]
                if topic == i:
                    confidences_inTopic.append(confidence)
        avgConfidence_perTopic.append(np.mean(confidences_inTopic))

    avgConfidence_perTopic = [x for x in avgConfidence_perTopic if x == x]
    print(avgConfidence_perTopic)

    print('Averaged confidence average over all topics -gm: ', np.mean(avgConfidence_perTopic))
    print('Median confidence average over all topics -gm: ', np.median(avgConfidence_perTopic))
    print('Mode confidence average over all topics -gm: ', statistics.mode(avgConfidence_perTopic))
    print('Max confidence average over all topics -gm: ', max(avgConfidence_perTopic))
    print('Min confidence average over all topics -gm: ', min(avgConfidence_perTopic))

    avgConfidence_perTopic = []
    for i in range(330):
        confidences_inTopic = []
        for window in communityTopicAssociation_dict_kc.values():
            for commuTopic in window:
                #[community id, topic, confidence]
                topic = list(commuTopic)[1]
                confidence = list(commuTopic)[2]
                if topic == i:
                    confidences_inTopic.append(confidence)
        avgConfidence_perTopic.append(np.mean(confidences_inTopic))

    avgConfidence_perTopic = [x for x in avgConfidence_perTopic if x == x]
    print(avgConfidence_perTopic)

    print('Averaged confidence average over all topics -kc: ', np.mean(avgConfidence_perTopic))
    print('Median confidence average over all topics -kc: ', np.median(avgConfidence_perTopic))
    print('Mode confidence average over all topics -kc: ', statistics.mode(avgConfidence_perTopic))
    print('Max confidence average over all topics -kc: ', max(avgConfidence_perTopic))
    print('Min confidence average over all topics -kc: ', min(avgConfidence_perTopic))

    avgConfidence_perTopic = []
    for i in range(330):
        confidences_inTopic = []
        for window in communityTopicAssociation_dict_l2.values():
            for commuTopic in window:
                #[community id, topic, confidence]
                topic = list(commuTopic)[1]
                confidence = list(commuTopic)[2]
                if topic == i:
                    confidences_inTopic.append(confidence)
        avgConfidence_perTopic.append(np.mean(confidences_inTopic))

    avgConfidence_perTopic = [x for x in avgConfidence_perTopic if x == x]
    print(avgConfidence_perTopic)

    print('Averaged confidence average over all topics -L2: ', np.mean(avgConfidence_perTopic))
    print('Median confidence average over all topics -L2: ', np.median(avgConfidence_perTopic))
    print('Mode confidence average over all topics -L2: ', statistics.mode(avgConfidence_perTopic))
    print('Max confidence average over all topics -L2: ', max(avgConfidence_perTopic))
    print('Min confidence average over all topics -L2: ', min(avgConfidence_perTopic))












    # CORRECT
    diffusionArray_Topics_lp, diffusionArray_Topics_lp_columns = CommunityMeasures.create_diffusionArray_Topics(communityTopicAssociation_dict_lp)
    diffusionArray_Topics_gm, diffusionArray_Topics_gm_columns = CommunityMeasures.create_diffusionArray_Topics(communityTopicAssociation_dict_gm)
    diffusionArray_Topics_kc, diffusionArray_Topics_kc_columns = CommunityMeasures.create_diffusionArray_Topics(communityTopicAssociation_dict_kc)
    diffusionArray_Topics_l2, diffusionArray_Topics_l2_columns = CommunityMeasures.create_diffusionArray_Topics(communityTopicAssociation_dict_l2)

    from utilities_old.my_measure_utils import Misc

    diffusionPatternPos_SCM_LP = Misc.find_diffusionPatterns(diffusionArray_Topics_lp)
    diffusionPatternPos_SCM_LP, diff_sequence_list_SCM, irrelevant = Misc.find_diffusionSequenceAndLength(diffusionPatternPos_SCM_LP, diffusionArray_Topics_lp)
    # diff_pos = [ row, column, diffLength, diffSteps, patentsInDiff ]
    diffusionPatternPos_SCM_LP = np.array(diffusionPatternPos_SCM_LP)
    print('SCM Number of diffusion cycles / patterns in the scm - LP: ', len(diffusionPatternPos_SCM_LP))
    print('SCM Average diffusion pattern length - LP: ', np.mean(diffusionPatternPos_SCM_LP[:, 2]))
    print('SCM median diffusion pattern length - LP: ', np.median(diffusionPatternPos_SCM_LP[:, 2]))
    print('SCM mode diffusion pattern length - LP: ', statistics.mode(diffusionPatternPos_SCM_LP[:, 2]))
    print('SCM max diffusion pattern length - LP: ', max(diffusionPatternPos_SCM_LP[:, 2]))
    print('SCM min diffusion pattern length - LP: ', min(diffusionPatternPos_SCM_LP[:, 2]))


    diffusionPatternPos_SCM_GM = Misc.find_diffusionPatterns(diffusionArray_Topics_gm)
    diffusionPatternPos_SCM_GM, diff_sequence_list_SCM, irrelevant = Misc.find_diffusionSequenceAndLength(diffusionPatternPos_SCM_GM, diffusionArray_Topics_gm)
    # diff_pos = [ row, column, diffLength, diffSteps, patentsInDiff ]
    diffusionPatternPos_SCM_GM = np.array(diffusionPatternPos_SCM_GM)
    print('SCM Number of diffusion cycles / patterns in the scm - GM: ', len(diffusionPatternPos_SCM_GM))
    print('SCM Average diffusion pattern length - GM: ', np.mean(diffusionPatternPos_SCM_GM[:, 2]))
    print('SCM median diffusion pattern length - LP: ', np.median(diffusionPatternPos_SCM_GM[:, 2]))
    print('SCM mode diffusion pattern length - LP: ', statistics.mode(diffusionPatternPos_SCM_GM[:, 2]))
    print('SCM max diffusion pattern length - LP: ', max(diffusionPatternPos_SCM_GM[:, 2]))
    print('SCM min diffusion pattern length - LP: ', min(diffusionPatternPos_SCM_GM[:, 2]))


    diffusionPatternPos_SCM_KC = Misc.find_diffusionPatterns(diffusionArray_Topics_kc)
    diffusionPatternPos_SCM_KC, diff_sequence_list_SCM, irrelevant = Misc.find_diffusionSequenceAndLength(diffusionPatternPos_SCM_KC, diffusionArray_Topics_kc)
    # diff_pos = [ row, column, diffLength, diffSteps, patentsInDiff ]
    diffusionPatternPos_SCM_KC = np.array(diffusionPatternPos_SCM_KC)
    print('SCM Number of diffusion cycles / patterns in the scm - KC: ', len(diffusionPatternPos_SCM_KC))
    print('SCM Average diffusion pattern length - KC: ', np.mean(diffusionPatternPos_SCM_KC[:, 2]))
    print('SCM median diffusion pattern length - LP: ', np.median(diffusionPatternPos_SCM_KC[:, 2]))
    print('SCM mode diffusion pattern length - LP: ', statistics.mode(diffusionPatternPos_SCM_KC[:, 2]))
    print('SCM max diffusion pattern length - LP: ', max(diffusionPatternPos_SCM_KC[:, 2]))
    print('SCM min diffusion pattern length - LP: ', min(diffusionPatternPos_SCM_KC[:, 2]))


    diffusionPatternPos_SCM_L2 = Misc.find_diffusionPatterns(diffusionArray_Topics_l2)
    diffusionPatternPos_SCM_L2, diff_sequence_list_SCM, irrelevant = Misc.find_diffusionSequenceAndLength(diffusionPatternPos_SCM_L2, diffusionArray_Topics_l2)
    # diff_pos = [ row, column, diffLength, diffSteps, patentsInDiff ]
    diffusionPatternPos_SCM_L2 = np.array(diffusionPatternPos_SCM_L2)
    print('SCM Number of diffusion cycles / patterns in the scm - L2: ', len(diffusionPatternPos_SCM_L2))
    print('SCM Average diffusion pattern length - L2: ', np.mean(diffusionPatternPos_SCM_L2[:, 2]))
    print('SCM median diffusion pattern length - LP: ', np.median(diffusionPatternPos_SCM_L2[:, 2]))
    print('SCM mode diffusion pattern length - LP: ', statistics.mode(diffusionPatternPos_SCM_L2[:, 2]))
    print('SCM max diffusion pattern length - LP: ', max(diffusionPatternPos_SCM_L2[:, 2]))
    print('SCM min diffusion pattern length - LP: ', min(diffusionPatternPos_SCM_L2[:, 2]))


    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, axes = plt.subplots(1, 2)  # , figsize=(15, 5), sharey=True)
    # fig.suptitle('test')
    sns.heatmap(diffusionArray_Topics_lp[0:80,20:30], ax=axes[0], cbar=True, cmap="bone_r",
                #cbar_kws={
                #'ticks': [0, 0.2, 0.4, 0.6, 0.8, 1]}, vmax=1,
                #vmin=0
                )
    sns.heatmap(diffusionArray_Topics_gm[0:80,20:30], ax=axes[1], cbar=True,
                cmap="bone_r")

    axes[0].set_title('SCM - Label Propagation')
    axes[1].set_title('SCM - Greedy Modularity')
    axes[0].set_xticklabels(range(20, 30))
    axes[0].set_yticks(range(0, 80, 10))
    axes[0].set_yticklabels(range(0, 80, 10))
    axes[1].set_xticklabels(range(20, 30))
    axes[1].set_yticks(range(0, 80, 10))
    axes[1].set_yticklabels(range(0, 80, 10))

    plt.tight_layout()
    #plt.xlabel('Knowledge Component ID')
    #plt.ylabel('Sliding Window ID')
    #plt.show()

    os.chdir('D:/Universitaet Mannheim/MMDS 7. Semester/Master Thesis/Outline/Plots')
    plt.savefig('SCM_LP_GM.png')
    os.chdir('D:/Universitaet Mannheim/MMDS 7. Semester/Master Thesis/Outline/Data/Cleaning Robots')

    plt.close()



    fig, axes = plt.subplots(1, 2)  # , figsize=(15, 5), sharey=True)
    # fig.suptitle('test')
    sns.heatmap(diffusionArray_Topics_kc[0:80,20:30], ax=axes[0], cbar=True, cmap="bone_r",
                cbar_kws={
                'ticks': [0, 0.2, 0.4, 0.6, 0.8, 1]}, vmax=1,
                vmin=0
                )
    sns.heatmap(diffusionArray_Topics_l2[0:80,20:30], ax=axes[1], cbar=True,
                cmap="bone_r")

    axes[0].set_title('SCM - K-Clique')
    axes[1].set_title('SCM - Lais2')
    axes[0].set_xticklabels(range(20, 30))
    axes[0].set_yticks(range(0, 80, 10))
    axes[0].set_yticklabels(range(0, 80, 10))
    axes[1].set_xticklabels(range(20, 30))
    axes[1].set_yticks(range(0, 80, 10))
    axes[1].set_yticklabels(range(0, 80, 10))

    plt.tight_layout()
    #plt.xlabel('Knowledge Component ID')
    #plt.ylabel('Sliding Window ID')
    #plt.show()

    os.chdir('D:/Universitaet Mannheim/MMDS 7. Semester/Master Thesis/Outline/Plots')
    plt.savefig('SCM_KC_L2.png')
    os.chdir('D:/Universitaet Mannheim/MMDS 7. Semester/Master Thesis/Outline/Data/Cleaning Robots')
    plt.close()



    '''
    f, ax = plt.subplots()
    sns.heatmap(diffusionArray_Topics_lp[0:80,20:30], cbar_kws={'label': 'Component Count in Window'}) #, cmap="YlGnBu") #, annot=True, fmt="d", linewidths=.5, ax=ax)
    plt.yticks(range(0,80,10))
    ax.set_xticklabels(range(20,30))
    ax.set_yticklabels(range(0,80,10))
    plt.xlabel("Knowledge Component ID")
    plt.ylabel("Sliding Window ID ")
    #plt.show()
    plt.close()

    f, ax = plt.subplots()
    sns.heatmap(diffusionArray_Topics_gm[0:80,20:30], cbar_kws={'label': 'Component Count in Window'}) #, cmap="YlGnBu") #, annot=True, fmt="d", linewidths=.5, ax=ax)
    plt.yticks(range(0,80,10))
    ax.set_xticklabels(range(20,30))
    ax.set_yticklabels(range(0,80,10))
    plt.xlabel("Knowledge Component ID")
    plt.ylabel("Sliding Window ID ")
    #plt.show()
    plt.close()

    print(diffusionArray_Topics_kc[0:80,20:30])

    f, ax = plt.subplots()
    #sns.heatmap(diffusionArray_Topics_kc[0:80,20:30], cbar_kws={'label': 'Component Count in Window', 'boundaries': range(0,1)}) #, cmap="YlGnBu") #, annot=True, fmt="d", linewidths=.5, ax=ax)
    sns.heatmap(diffusionArray_Topics_kc[0:80,20:30], cbar_kws={'label': 'Component Count in Window', 'ticks': [0, 0.2, 0.4, 0.6, 0.8, 1]}, vmax=1, vmin=0)
    #colorbar = ax.collections[0].colorbar
    #colorbar.set_ticks([0,1,0.2])
    plt.yticks(range(0,80,10))
    ax.set_xticklabels(range(20,30))
    ax.set_yticklabels(range(0,80,10))
    plt.xlabel("Knowledge Component ID")
    plt.ylabel("Sliding Window ID ")
    #plt.show()
    plt.close()

    f, ax = plt.subplots()
    sns.heatmap(diffusionArray_Topics_l2[0:80,20:30], cbar_kws={'label': 'Component Count in Window'}) #, cmap="YlGnBu") #, annot=True, fmt="d", linewidths=.5, ax=ax)
    plt.yticks(range(0,80,10))
    ax.set_xticklabels(range(20,30))
    ax.set_yticklabels(range(0,80,10))
    plt.xlabel("Knowledge Component ID")
    plt.ylabel("Sliding Window ID ")
    #plt.show()
    plt.close()
    '''
    # they cant be the same, because lp_singleDiffusion_v2 measure the lifetime of communities and not topic diffusion.
    # in lp_singleDiffusion_v2 subset of communities are listed as well, if they were swallowed by bigger communities.
    # this is irrelevant for topics.
    # CORRECT
    recombination_dict_Topics_lp = CommunityMeasures.created_recombination_dict_Topics_crisp(communityTopicAssociation_dict_lp, recombination_dict_lp)
    recombination_dict_Topics_gm = CommunityMeasures.created_recombination_dict_Topics_crisp(communityTopicAssociation_dict_gm, recombination_dict_gm)
    recombination_dict_Topics_kc = CommunityMeasures.created_recombination_dict_Topics_overlap(communityTopicAssociation_dict_kc, recombination_dict_kc)
    recombination_dict_Topics_l2 = CommunityMeasures.created_recombination_dict_Topics_overlap(communityTopicAssociation_dict_l2, recombination_dict_l2)
    # CORRECT
    CommunityMeasures.doubleCheck_recombination_dict_Topics_crisp(recombination_dict_Topics_lp, recombination_dict_lp, communityTopicAssociation_dict_lp)
    CommunityMeasures.doubleCheck_recombination_dict_Topics_crisp(recombination_dict_Topics_gm, recombination_dict_gm, communityTopicAssociation_dict_gm)
    CommunityMeasures.doubleCheck_recombination_dict_Topics_overlap(recombination_dict_Topics_kc, recombination_dict_kc, communityTopicAssociation_dict_kc)
    CommunityMeasures.doubleCheck_recombination_dict_Topics_overlap(recombination_dict_Topics_l2, recombination_dict_l2, communityTopicAssociation_dict_l2)

    recombinationArray_Topics_lp, recombinationArray_Topics_lp_columns = CommunityMeasures.create_recombinationArray_Topics(recombination_dict_Topics_lp)
    recombinationArray_Topics_gm, recombinationArray_Topics_gm_columns = CommunityMeasures.create_recombinationArray_Topics(recombination_dict_Topics_gm)
    recombinationArray_Topics_kc, recombinationArray_Topics_kc_columns = CommunityMeasures.create_recombinationArray_Topics(recombination_dict_Topics_kc)
    recombinationArray_Topics_l2, recombinationArray_Topics_l2_columns = CommunityMeasures.create_recombinationArray_Topics(recombination_dict_Topics_l2)
    '''
    f, ax = plt.subplots()
    sns.heatmap(recombinationArray_Topics_lp[100:180,0:11], cbar_kws={'label': 'Component Combination Count in Window'}) #, cmap="YlGnBu") #, annot=True, fmt="d", linewidths=.5, ax=ax)
    plt.yticks(range(0,80,10))
    ax.set_xticklabels(range(0,11))
    ax.set_yticklabels(range(100,180,10))
    plt.xlabel("Knowledge Component Combination ID")
    plt.ylabel("Sliding Window ID ")
    #plt.show()
    plt.close()

    f, ax = plt.subplots()
    sns.heatmap(recombinationArray_Topics_gm[100:180,0:11], cbar_kws={'label': 'Component Combination Count in Window'}) #, cmap="YlGnBu") #, annot=True, fmt="d", linewidths=.5, ax=ax)
    plt.yticks(range(0,80,10))
    ax.set_xticklabels(range(0,11))
    ax.set_yticklabels(range(100,180,10))
    plt.xlabel("Knowledge Component Combination ID")
    plt.ylabel("Sliding Window ID ")
    #plt.show()
    plt.close()

    f, ax = plt.subplots()
    sns.heatmap(recombinationArray_Topics_kc[100:180,0:11], cbar_kws={'label': 'Component Combination Count in Window'}) #, cmap="YlGnBu") #, annot=True, fmt="d", linewidths=.5, ax=ax)
    plt.yticks(range(0,80,10))
    ax.set_xticklabels(range(0,11))
    ax.set_yticklabels(range(100,180,10))
    plt.xlabel("Knowledge Component Combination ID")
    plt.ylabel("Sliding Window ID ")
    #plt.show()
    plt.close()

    f, ax = plt.subplots()
    sns.heatmap(recombinationArray_Topics_l2[100:180,0:11], cbar_kws={'label': 'Component Combination Count in Window', 'ticks': [0, 0.2, 0.4, 0.6, 0.8, 1]}, vmax=1, vmin=0) #, cmap="YlGnBu") #, annot=True, fmt="d", linewidths=.5, ax=ax)
    plt.yticks(range(0,80,10))
    ax.set_xticklabels(range(0,11))
    ax.set_yticklabels(range(100,180,10))
    plt.xlabel("Knowledge Component Combination ID")
    plt.ylabel("Sliding Window ID ")
    #plt.show()
    plt.close()
    '''


    diffusionPatternPos_CCM_LP = Misc.find_diffusionPatterns(recombinationArray_Topics_lp)
    diffusionPatternPos_CCM_LP, diff_sequence_list_CCM_LP, diff_sequence_sum_list_CCM_LP = Misc.find_diffusionSequenceAndLength(diffusionPatternPos_CCM_LP, recombinationArray_Topics_lp)
    # diff_pos = [ row, column, diffLength, diffSteps, patentsInDiff ]
    diffusionPatternPos_CCM_LP = np.array(diffusionPatternPos_CCM_LP)
    print('CCM Number of diffusion cycles / patterns in the scm - LP: ', len(diffusionPatternPos_CCM_LP))
    print('CCM Average diffusion pattern length - LP: ', np.mean(diffusionPatternPos_CCM_LP[:, 2]))
    print('CCM Average diffusion pattern patent engagement - LP: ', np.mean(diff_sequence_sum_list_CCM_LP))

    diffusionPatternPos_CCM_GM = Misc.find_diffusionPatterns(recombinationArray_Topics_gm)
    diffusionPatternPos_CCM_GM, diff_sequence_list_CCM_GM, diff_sequence_sum_list_CCM_GM = Misc.find_diffusionSequenceAndLength(diffusionPatternPos_CCM_GM, recombinationArray_Topics_gm)
    # diff_pos = [ row, column, diffLength, diffSteps, patentsInDiff ]
    diffusionPatternPos_CCM_GM = np.array(diffusionPatternPos_CCM_GM)
    print('CCM Number of diffusion cycles / patterns in the scm - GM: ', len(diffusionPatternPos_CCM_GM))
    print('CCM Average diffusion pattern length - GM: ', np.mean(diffusionPatternPos_CCM_GM[:, 2]))
    print('CCM Average diffusion pattern patent engagement - GM: ', np.mean(diff_sequence_sum_list_CCM_GM))

    diffusionPatternPos_CCM_KC = Misc.find_diffusionPatterns(recombinationArray_Topics_kc)
    diffusionPatternPos_CCM_KC, diff_sequence_list_CCM_KC, diff_sequence_sum_list_CCM_KC = Misc.find_diffusionSequenceAndLength(diffusionPatternPos_CCM_KC, recombinationArray_Topics_kc)
    # diff_pos = [ row, column, diffLength, diffSteps, patentsInDiff ]
    diffusionPatternPos_CCM_KC = np.array(diffusionPatternPos_CCM_KC)
    print('CCM Number of diffusion cycles / patterns in the scm - KC: ', len(diffusionPatternPos_CCM_KC))
    print('CCM Average diffusion pattern length - KC: ', np.mean(diffusionPatternPos_CCM_KC[:, 2]))
    print('CCM Average diffusion pattern patent engagement - KC: ', np.mean(diff_sequence_sum_list_CCM_KC))

    diffusionPatternPos_CCM_L2 = Misc.find_diffusionPatterns(recombinationArray_Topics_l2)
    diffusionPatternPos_CCM_L2, diff_sequence_list_CCM_L2, diff_sequence_sum_list_CCM_L2 = Misc.find_diffusionSequenceAndLength(diffusionPatternPos_CCM_L2, recombinationArray_Topics_l2)
    # diff_pos = [ row, column, diffLength, diffSteps, patentsInDiff ]
    diffusionPatternPos_CCM_L2 = np.array(diffusionPatternPos_CCM_L2)
    print('CCM Number of diffusion cycles / patterns in the scm - L2: ', len(diffusionPatternPos_CCM_L2))
    print('CCM Average diffusion pattern length - L2: ', np.mean(diffusionPatternPos_CCM_L2[:, 2]))
    print('CCM Average diffusion pattern patent engagement - L2: ', np.mean(diff_sequence_sum_list_CCM_L2))




    fig, axes = plt.subplots(1, 2)  # , figsize=(15, 5), sharey=True)
    # fig.suptitle('test')
    sns.heatmap(recombinationArray_Topics_lp[100:180, 5:11], ax=axes[0], cbar=True, cmap="bone_r", cbar_kws={
        'ticks': [0, 0.2, 0.4, 0.6, 0.8, 1]}, vmax=1,
                vmin=0)
    sns.heatmap(recombinationArray_Topics_gm[100:180, 5:11], ax=axes[1], cbar=True,
                cmap="bone_r")

    axes[0].set_title('CCM - Label Propagation')
    axes[1].set_title('CCM - Greedy Modularity')
    axes[0].set_xticklabels(range(5, 11))
    axes[0].set_yticks(range(0, 80, 10))
    axes[0].set_yticklabels(range(100, 180, 10))
    axes[1].set_xticklabels(range(5, 11))
    axes[1].set_yticks(range(0, 80, 10))
    axes[1].set_yticklabels(range(100, 180, 10))

    plt.tight_layout()
    #plt.xlabel('Knowledge Component Combination ID')
    #plt.ylabel('Sliding Window ID')
    #plt.show()

    os.chdir('D:/Universitaet Mannheim/MMDS 7. Semester/Master Thesis/Outline/Plots')
    plt.savefig('CCM_LP_GM.png')
    os.chdir('D:/Universitaet Mannheim/MMDS 7. Semester/Master Thesis/Outline/Data/Cleaning Robots')
    plt.close()
    plt.close()





    fig, axes = plt.subplots(1, 2)  # , figsize=(15, 5), sharey=True)
    # fig.suptitle('test')

    sns.heatmap(recombinationArray_Topics_kc[100:180, 5:11], ax=axes[0], cbar=True, cmap="bone_r") #, cbar_kws={'label': 'Component Combination Count in Window'})  # , cmap="YlGnBu") #, annot=True, fmt="d", linewidths=.5, ax=ax)
    sns.heatmap(recombinationArray_Topics_l2[100:180, 5:11], ax=axes[1], cbar=True,
                cmap = "bone_r",
                cbar_kws={
                            #'label': 'Component Combination Count in Window',
                            'ticks': [0, 1, 2, 3, 4, 5, 6]}, vmax=6, vmin=0)  # , cmap="YlGnBu") #, annot=True, fmt="d", linewidths=.5, ax=ax)

    axes[0].set_title('CCM - K-Clique')
    axes[1].set_title('CCM - Lais2')
    #axes[1, 0].set_title('kc')
    # axes[1, 1].set_title('l2')

    axes[0].set_xticklabels(range(5, 11))
    axes[0].set_yticks(range(0, 80, 10))
    axes[0].set_yticklabels(range(100, 180, 10))

    axes[1].set_xticklabels(range(5, 11))
    axes[1].set_yticks(range(0, 80, 10))
    axes[1].set_yticklabels(range(100, 180, 10))

    plt.tight_layout()
    #plt.xlabel('Knowledge Component Combination ID')
    #plt.ylabel('Sliding Window ID')
    #plt.show()

    os.chdir('D:/Universitaet Mannheim/MMDS 7. Semester/Master Thesis/Outline/Plots')
    plt.savefig('CCM_KC_L2.png')
    os.chdir('D:/Universitaet Mannheim/MMDS 7. Semester/Master Thesis/Outline/Data/Cleaning Robots')
    plt.close()
    plt.close()


    print(np.shape(recombinationArray_Topics_lp))
    print(np.shape(recombinationArray_Topics_gm))
    print(np.shape(recombinationArray_Topics_kc))
    print(np.shape(recombinationArray_Topics_l2))

    columSum_vec = np.sum(recombinationArray_Topics_lp, axis= 0)
    print(len(np.where(columSum_vec == 0)[0]))

    #(189, 3061) 154 empty columns
    #(189, 42)
    #(189, 128)
    #(189, 454)

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

    #2. compute average change of confidence in a community id
    #3. compute average confidence in window
    #4. average condience in general

