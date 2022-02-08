if __name__ == '__main__':

    #--- Import Libraries ---#
    print('\n#--- Import Libraries ---#\n')

    # Utility
    import os
    import statistics

    # Data handling
    import pandas as pd
    import pickle as pk
    import numpy as np

    # Visualization
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Custom functions
    from utilities.Measurement_utils import Community_Measurement
    from utilities.Measurement_utils import Misc



    #--- Initialization ---#
    print('\n# --- Initialization ---#\n')

    # Hyperparameter value given to LDA
    maxTopicNumber = 330 
    
    path = 'D:/'

    #load data
    os.chdir(path)
    
    patent_lda_ipc = pd.read_csv('patent_lda_ipc.csv', quotechar='"', skipinitialspace=True)
    patent_lda_ipc = patent_lda_ipc.to_numpy()

    with open('patentProject_graphs', 'rb') as handle:
        patentProject_graphs = pk.load(handle)

    #'''
    #--- Applying Community detection to each graph/window and populate respective dictionaries ---#
    print('\n#--- Applying Community detection to each graph/window and populate respective dictionaries ---#\n')

    # Creating dictionaries with communities and get Modularity

    community_dict_lp, modularity_dict_lp = Community_Measurement.detect_communities(patentProject_graphs, cD_algorithm='label_propagation', weight_bool=True)
    modularity_lp = list(modularity_dict_lp.values())
    print('average modularity lp: ', np.mean(modularity_lp))

    community_dict_gm, modularity_dict_gm = Community_Measurement.detect_communities(patentProject_graphs, cD_algorithm='greedy_modularity')
    modularity_gm = []
    for i in modularity_dict_gm.values():
        modularity_gm.append(i[2])
    print('average modularity gm: ', np.mean(modularity_gm))

    community_dict_kc, modularity_dict_kc = Community_Measurement.detect_communities(patentProject_graphs, cD_algorithm='k_clique', k_clique_size=3)
    modularity_kc = []
    for i in modularity_dict_kc.values():
        modularity_kc.append(i[2])
    print('average modularity kc: ', np.mean(modularity_kc))

    community_dict_l2, modularity_dict_l2 = Community_Measurement.detect_communities(patentProject_graphs, cD_algorithm='lais2')
    modularity_l2 = []
    for i in modularity_dict_l2.values():
        modularity_l2.append(i[2])
    print('average modularity l2: ', np.mean(modularity_l2))

    # visualization
    fig, ax = plt.subplots()
    ax.plot(range(len(patentProject_graphs)), modularity_lp, color='darkblue', label='Label Propagation')
    ax.plot(range(len(patentProject_graphs)), modularity_gm, color='darkred', label='Greedy Modularity')
    ax.plot(range(len(patentProject_graphs)), modularity_kc, color='darkgreen', label='K-Clique')
    ax.plot(range(len(patentProject_graphs)), modularity_l2, color='black', label='Lais2')
    ax.set_ylim([-0.2, 1.2])
    plt.legend(loc='upper left')
    plt.xlabel("Patent Network Representation of Sliding Windows")
    plt.ylabel("Modularity")
    plt.savefig('Modularities.png')
    plt.close()
    #'''

    # saving intermediate results because Lais2 takes a while
    #'''
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
    #'''

    #--- Community Measurement Preprocessing ---#
    print('\n#--- Community Measurement Preprocessing ---#\n')

    # load intermediate results #
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

    # --- Aligne Data Structure of Communties ---#
    community_dict_transf_lp = Community_Measurement.align_cD_dataStructure(community_dict_lp, cD_algorithm='label_propagation')
    community_dict_transf_gm = Community_Measurement.align_cD_dataStructure(community_dict_gm, cD_algorithm='greedy_modularity')
    community_dict_transf_kc = Community_Measurement.align_cD_dataStructure(community_dict_kc, cD_algorithm='k_clique')
    community_dict_transf_l2 = Community_Measurement.align_cD_dataStructure(community_dict_l2, cD_algorithm='lais2')


    # --- Clean Communties ---#
    community_dict_clean_lp, communities_removed_lp = Community_Measurement.community_cleaning(community_dict_transf_lp,min_community_size=3)
    community_dict_clean_gm, communities_removed_gm = Community_Measurement.community_cleaning(community_dict_transf_gm,min_community_size=3)
    community_dict_clean_kc, communities_removed_kc = Community_Measurement.community_cleaning(community_dict_transf_kc,min_community_size=3)
    community_dict_clean_l2, communities_removed_l2 = Community_Measurement.community_cleaning(community_dict_transf_l2,min_community_size=3)


    # Descriptives
    community_length_list = []
    community_size_list = []
    for window_id, window in community_dict_clean_lp.items():
        community_length_list.append(len(window))

        for community in window:
            community_size_list.append(len(community))


    print('Average Size of communities in Window LP: ', np.mean(community_size_list))
    print('Average Number of communities in Window LP: ', np.mean(community_length_list))
    print('Median Number of communities in Window LP: ', np.median(community_length_list))
    print('Mode Number of communities in Window LP: ', statistics.mode(community_length_list))
    print('Max Number of communities in Window LP: ', max(community_length_list))
    print('Min Number of communities in Window LP: ', min(community_length_list), '\n')

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
    print('Min Number of communities in Window gm: ', min(community_length_list), '\n')

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
    print('Min Number of communities in Window kc: ', min(community_length_list), '\n')

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
    print('Min Number of communities in Window L2: ', min(community_length_list), '\n')

    #--- removed communities ---#
    print('Average Number of removed communities lp: ', communities_removed_lp / len(community_dict_clean_lp))
    print('Average Number of removed communities gm: ', communities_removed_gm / len(community_dict_clean_lp))
    print('Average Number of removed communities kc: ', communities_removed_kc / len(community_dict_clean_lp))
    print('Average Number of removed communities l2: ', communities_removed_l2 / len(community_dict_clean_lp), '\n')

    #---  Correct parts of the Flawed Approach are partly utilized below  ---#
    print('\n#--- Correct parts of the Flawed Approach are partly utilized below ---#\n')

    # --- Identify TopD degree nodes of communities ---#
    community_dict_topD_lp = Community_Measurement.identify_topDegree(community_dict_clean_lp, patentProject_graphs)
    community_dict_topD_gm = Community_Measurement.identify_topDegree(community_dict_clean_gm, patentProject_graphs)
    community_dict_topD_kc = Community_Measurement.identify_topDegree(community_dict_clean_kc, patentProject_graphs)
    community_dict_topD_l2 = Community_Measurement.identify_topDegree(community_dict_clean_l2, patentProject_graphs)

    # --- Mainly for Lais2: merge communities with same topd ---#
    community_dict_topD_lp = Community_Measurement.merging_completly_overlapping_communities(community_dict_topD_lp)
    community_dict_topD_gm = Community_Measurement.merging_completly_overlapping_communities(community_dict_topD_gm)
    community_dict_topD_kc = Community_Measurement.merging_completly_overlapping_communities(community_dict_topD_kc)
    community_dict_topD_l2 = Community_Measurement.merging_completly_overlapping_communities(community_dict_topD_l2)

    # ---  Community Tracing ---#

    ### Identify max number of possible community id's ###
    max_number_community_lp = Community_Measurement.max_number_community(community_dict_topD_lp)
    max_number_community_gm = Community_Measurement.max_number_community(community_dict_topD_gm)
    max_number_community_kc = Community_Measurement.max_number_community(community_dict_topD_kc)
    max_number_community_l2 = Community_Measurement.max_number_community(community_dict_topD_l2)

    ### Tracing arrays ###
    tracingArray_lp = Community_Measurement.create_tracing_array(max_number_community_lp, community_dict_topD_lp, patentProject_graphs)
    tracingArray_gm = Community_Measurement.create_tracing_array(max_number_community_gm, community_dict_topD_gm, patentProject_graphs)
    tracingArray_kc = Community_Measurement.create_tracing_array(max_number_community_kc, community_dict_topD_kc, patentProject_graphs)
    tracingArray_l2 = Community_Measurement.create_tracing_array(max_number_community_l2, community_dict_topD_l2, patentProject_graphs)

    #--- Community Labeling ---#

    community_dict_labeled_lp, topD_communityID_association_perWindow_lp, topD_communityID_association_accumulated_lp = Community_Measurement.community_labeling(tracingArray_lp, community_dict_topD_lp, patentProject_graphs)
    community_dict_labeled_gm, topD_communityID_association_perWindow_gm, topD_communityID_association_accumulated_gm = Community_Measurement.community_labeling(tracingArray_gm, community_dict_topD_gm, patentProject_graphs)
    community_dict_labeled_kc, topD_communityID_association_perWindow_kc, topD_communityID_association_accumulated_kc = Community_Measurement.community_labeling(tracingArray_kc, community_dict_topD_kc, patentProject_graphs)
    community_dict_labeled_l2, topD_communityID_association_perWindow_l2, topD_communityID_association_accumulated_l2 = Community_Measurement.community_labeling(tracingArray_l2, community_dict_topD_l2, patentProject_graphs)

    # --- Make sure community ids are unique in each window ---#

    Community_Measurement.is_community_id_unique(community_dict_labeled_lp)
    Community_Measurement.is_community_id_unique(community_dict_labeled_gm)
    Community_Measurement.is_community_id_unique(community_dict_labeled_kc)
    Community_Measurement.is_community_id_unique(community_dict_labeled_l2)

    # --- Community Label Visualization ---#
    visualizationArray_lp = Community_Measurement.create_visualization_array(tracingArray_lp, topD_communityID_association_perWindow_lp)
    visualizationArray_gm = Community_Measurement.create_visualization_array(tracingArray_gm, topD_communityID_association_perWindow_gm)
    visualizationArray_kc = Community_Measurement.create_visualization_array(tracingArray_kc, topD_communityID_association_perWindow_kc)
    visualizationArray_l2 = Community_Measurement.create_visualization_array(tracingArray_l2, topD_communityID_association_perWindow_l2)

    # --- Finding Recombination ---#
    # Create Recombination dict - crisp 

    recombination_dict_lp = Community_Measurement.find_recombinations_crisp(community_dict_labeled_lp, patentProject_graphs)
    recombination_dict_gm = Community_Measurement.find_recombinations_crisp(community_dict_labeled_gm, patentProject_graphs)
    recombination_dict_kc = Community_Measurement.find_recombinations_overlapping(community_dict_labeled_kc, patentProject_graphs)
    recombination_dict_l2 = Community_Measurement.find_recombinations_overlapping(community_dict_labeled_l2, patentProject_graphs)


    # --- Calculate Topic Distributions ---#
    print('\n# --- Calculate Topic Distributions ---#\n')
    
    # window: [community id [topic distribution], community id [...], ... window: ...
    topicDistriburionOfCommunities_dict_lp = Community_Measurement.creat_dict_topicDistriburionOfCommunities(community_dict_labeled_lp, patent_lda_ipc)
    topicDistriburionOfCommunities_dict_gm = Community_Measurement.creat_dict_topicDistriburionOfCommunities(community_dict_labeled_gm, patent_lda_ipc)
    topicDistriburionOfCommunities_dict_kc = Community_Measurement.creat_dict_topicDistriburionOfCommunities(community_dict_labeled_kc, patent_lda_ipc)
    topicDistriburionOfCommunities_dict_l2 = Community_Measurement.creat_dict_topicDistriburionOfCommunities(community_dict_labeled_l2, patent_lda_ipc)

    # community id , most dominant topic, confidence
    communityTopicAssociation_dict_lp, avg_confidence_lp = Community_Measurement.create_dict_communityTopicAssociation(topicDistriburionOfCommunities_dict_lp)
    communityTopicAssociation_dict_gm, avg_confidence_gm = Community_Measurement.create_dict_communityTopicAssociation(topicDistriburionOfCommunities_dict_gm)
    communityTopicAssociation_dict_kc, avg_confidence_kc = Community_Measurement.create_dict_communityTopicAssociation(topicDistriburionOfCommunities_dict_kc)
    communityTopicAssociation_dict_l2, avg_confidence_l2 = Community_Measurement.create_dict_communityTopicAssociation(topicDistriburionOfCommunities_dict_l2)

    # Descriptives
    # list of confidence averages for every topic
    avgConfidence_perTopic = []
    for i in range(maxTopicNumber):
        confidences_inTopic = []
        for window in communityTopicAssociation_dict_lp.values():
            for commuTopic in window:
                # [community id, topic, confidence]
                topic = list(commuTopic)[1]
                confidence = list(commuTopic)[2]
                if topic == i:
                    confidences_inTopic.append(confidence)
        avgConfidence_perTopic.append(np.mean(confidences_inTopic))

    avgConfidence_perTopic = [x for x in avgConfidence_perTopic if x == x]

    print('Averaged confidence average over all topics -LP: ', np.mean(avgConfidence_perTopic))
    print('Median confidence average over all topics -LP: ', np.median(avgConfidence_perTopic))
    print('Mode confidence average over all topics -LP: ', statistics.mode(avgConfidence_perTopic))
    print('Max confidence average over all topics -LP: ', max(avgConfidence_perTopic))
    print('Min confidence average over all topics -LP: ', min(avgConfidence_perTopic))

    avgConfidence_perTopic = []
    for i in range(maxTopicNumber):
        confidences_inTopic = []
        for window in communityTopicAssociation_dict_gm.values():
            for commuTopic in window:
                # [community id, topic, confidence]
                topic = list(commuTopic)[1]
                confidence = list(commuTopic)[2]
                if topic == i:
                    confidences_inTopic.append(confidence)
        avgConfidence_perTopic.append(np.mean(confidences_inTopic))

    avgConfidence_perTopic = [x for x in avgConfidence_perTopic if x == x]

    print('Averaged confidence average over all topics -gm: ', np.mean(avgConfidence_perTopic))
    print('Median confidence average over all topics -gm: ', np.median(avgConfidence_perTopic))
    print('Mode confidence average over all topics -gm: ', statistics.mode(avgConfidence_perTopic))
    print('Max confidence average over all topics -gm: ', max(avgConfidence_perTopic))
    print('Min confidence average over all topics -gm: ', min(avgConfidence_perTopic))

    avgConfidence_perTopic = []
    for i in range(maxTopicNumber):
        confidences_inTopic = []
        for window in communityTopicAssociation_dict_kc.values():
            for commuTopic in window:
                # [community id, topic, confidence]
                topic = list(commuTopic)[1]
                confidence = list(commuTopic)[2]
                if topic == i:
                    confidences_inTopic.append(confidence)
        avgConfidence_perTopic.append(np.mean(confidences_inTopic))

    avgConfidence_perTopic = [x for x in avgConfidence_perTopic if x == x]

    print('Averaged confidence average over all topics -kc: ', np.mean(avgConfidence_perTopic))
    print('Median confidence average over all topics -kc: ', np.median(avgConfidence_perTopic))
    print('Mode confidence average over all topics -kc: ', statistics.mode(avgConfidence_perTopic))
    print('Max confidence average over all topics -kc: ', max(avgConfidence_perTopic))
    print('Min confidence average over all topics -kc: ', min(avgConfidence_perTopic))

    avgConfidence_perTopic = []
    for i in range(maxTopicNumber):
        confidences_inTopic = []
        for window in communityTopicAssociation_dict_l2.values():
            for commuTopic in window:
                # [community id, topic, confidence]
                topic = list(commuTopic)[1]
                confidence = list(commuTopic)[2]
                if topic == i:
                    confidences_inTopic.append(confidence)
        avgConfidence_perTopic.append(np.mean(confidences_inTopic))

    avgConfidence_perTopic = [x for x in avgConfidence_perTopic if x == x]

    print('Averaged confidence average over all topics -L2: ', np.mean(avgConfidence_perTopic))
    print('Median confidence average over all topics -L2: ', np.median(avgConfidence_perTopic))
    print('Mode confidence average over all topics -L2: ', statistics.mode(avgConfidence_perTopic))
    print('Max confidence average over all topics -L2: ', max(avgConfidence_perTopic))
    print('Min confidence average over all topics -L2: ', min(avgConfidence_perTopic))


    #--- Creating SCMs ---#
    print('\n#--- Creating SCMs ---#\n')
    CommunityMeasure_SCM_LP, CommunityMeasure_SCM_LP_columns = Community_Measurement.create_diffusionArray_Topics(communityTopicAssociation_dict_lp, maxTopicNumber)
    CommunityMeasure_SCM_GM, CommunityMeasure_SCM_GM_columns = Community_Measurement.create_diffusionArray_Topics(communityTopicAssociation_dict_gm, maxTopicNumber)
    CommunityMeasure_SCM_KC, CommunityMeasure_SCM_KC_columns = Community_Measurement.create_diffusionArray_Topics(communityTopicAssociation_dict_kc, maxTopicNumber)
    CommunityMeasure_SCM_L2, CommunityMeasure_SCM_L2_columns = Community_Measurement.create_diffusionArray_Topics(communityTopicAssociation_dict_l2, maxTopicNumber)

    # Descriptives
    diffusionPatternPos_SCM_LP = Misc.find_diffusionPatterns(CommunityMeasure_SCM_LP)
    diffusionPatternPos_SCM_LP, diff_sequence_list_SCM, placeholder = Misc.find_diffusionSequenceAndLength(diffusionPatternPos_SCM_LP, CommunityMeasure_SCM_LP)
    # diff_pos = [ row, column, diffLength, diffSteps, patentsInDiff ]
    diffusionPatternPos_SCM_LP = np.array(diffusionPatternPos_SCM_LP)
    print('SCM Number of diffusion cycles / patterns in the scm - LP -SCM-: ', len(diffusionPatternPos_SCM_LP))
    print('SCM Average diffusion pattern length - LP -SCM-: ', np.mean(diffusionPatternPos_SCM_LP[:, 2]))
    print('SCM median diffusion pattern length - LP -SCM-: ', np.median(diffusionPatternPos_SCM_LP[:, 2]))
    print('SCM mode diffusion pattern length - LP -SCM-: ', statistics.mode(diffusionPatternPos_SCM_LP[:, 2]))
    print('SCM max diffusion pattern length - LP -SCM-: ', max(diffusionPatternPos_SCM_LP[:, 2]))
    print('SCM min diffusion pattern length - LP -SCM-: ', min(diffusionPatternPos_SCM_LP[:, 2]))


    diffusionPatternPos_SCM_GM = Misc.find_diffusionPatterns(CommunityMeasure_SCM_GM)
    diffusionPatternPos_SCM_GM, diff_sequence_list_SCM, placeholder = Misc.find_diffusionSequenceAndLength(diffusionPatternPos_SCM_GM, CommunityMeasure_SCM_GM)
    # diff_pos = [ row, column, diffLength, diffSteps, patentsInDiff ]
    diffusionPatternPos_SCM_GM = np.array(diffusionPatternPos_SCM_GM)
    print('SCM Number of diffusion cycles / patterns in the scm - GM -SCM-: ', len(diffusionPatternPos_SCM_GM))
    print('SCM Average diffusion pattern length - GM -SCM-: ', np.mean(diffusionPatternPos_SCM_GM[:, 2]))
    print('SCM median diffusion pattern length - LP -SCM-: ', np.median(diffusionPatternPos_SCM_GM[:, 2]))
    print('SCM mode diffusion pattern length - LP -SCM-: ', statistics.mode(diffusionPatternPos_SCM_GM[:, 2]))
    print('SCM max diffusion pattern length - LP -SCM-: ', max(diffusionPatternPos_SCM_GM[:, 2]))
    print('SCM min diffusion pattern length - LP -SCM-: ', min(diffusionPatternPos_SCM_GM[:, 2]))


    diffusionPatternPos_SCM_KC = Misc.find_diffusionPatterns(CommunityMeasure_SCM_KC)
    diffusionPatternPos_SCM_KC, diff_sequence_list_SCM, placeholder = Misc.find_diffusionSequenceAndLength(diffusionPatternPos_SCM_KC, CommunityMeasure_SCM_KC)
    # diff_pos = [ row, column, diffLength, diffSteps, patentsInDiff ]
    diffusionPatternPos_SCM_KC = np.array(diffusionPatternPos_SCM_KC)
    print('SCM Number of diffusion cycles / patterns in the scm - KC -SCM-: ', len(diffusionPatternPos_SCM_KC))
    print('SCM Average diffusion pattern length - KC -SCM-: ', np.mean(diffusionPatternPos_SCM_KC[:, 2]))
    print('SCM median diffusion pattern length - LP -SCM-: ', np.median(diffusionPatternPos_SCM_KC[:, 2]))
    print('SCM mode diffusion pattern length - LP -SCM-: ', statistics.mode(diffusionPatternPos_SCM_KC[:, 2]))
    print('SCM max diffusion pattern length - LP -SCM-: ', max(diffusionPatternPos_SCM_KC[:, 2]))
    print('SCM min diffusion pattern length - LP -SCM-: ', min(diffusionPatternPos_SCM_KC[:, 2]))


    diffusionPatternPos_SCM_L2 = Misc.find_diffusionPatterns(CommunityMeasure_SCM_L2)
    diffusionPatternPos_SCM_L2, diff_sequence_list_SCM, placeholder = Misc.find_diffusionSequenceAndLength(diffusionPatternPos_SCM_L2, CommunityMeasure_SCM_L2)
    # diff_pos = [ row, column, diffLength, diffSteps, patentsInDiff ]
    diffusionPatternPos_SCM_L2 = np.array(diffusionPatternPos_SCM_L2)
    print('SCM Number of diffusion cycles / patterns in the scm - L2 -SCM-: ', len(diffusionPatternPos_SCM_L2))
    print('SCM Average diffusion pattern length - L2 -SCM-: ', np.mean(diffusionPatternPos_SCM_L2[:, 2]))
    print('SCM median diffusion pattern length - LP -SCM-: ', np.median(diffusionPatternPos_SCM_L2[:, 2]))
    print('SCM mode diffusion pattern length - LP -SCM-: ', statistics.mode(diffusionPatternPos_SCM_L2[:, 2]))
    print('SCM max diffusion pattern length - LP -SCM-: ', max(diffusionPatternPos_SCM_L2[:, 2]))
    print('SCM min diffusion pattern length - LP -SCM-: ', min(diffusionPatternPos_SCM_L2[:, 2]))

    # Visualization
    fig, axes = plt.subplots(1, 2)
    sns.heatmap(CommunityMeasure_SCM_LP[0:80,20:30], ax=axes[0], cbar=True, cmap="bone_r",)
    sns.heatmap(CommunityMeasure_SCM_GM[0:80,20:30], ax=axes[1], cbar=True, cmap="bone_r",
                cbar_kws={'ticks': [0, 0.2, 0.4, 0.6, 0.8, 1]}, vmax=1, vmin=0)

    axes[0].set_title('SCM - Label Propagation')
    axes[1].set_title('SCM - Greedy Modularity')
    axes[0].set_xticklabels(range(20, 30))
    axes[0].set_yticks(range(0, 80, 10))
    axes[0].set_yticklabels(range(0, 80, 10))
    axes[1].set_xticklabels(range(20, 30))
    axes[1].set_yticks(range(0, 80, 10))
    axes[1].set_yticklabels(range(0, 80, 10))

    plt.tight_layout()
    plt.savefig('SCM_LP_GM.png')
    plt.close()



    fig, axes = plt.subplots(1, 2)
    sns.heatmap(CommunityMeasure_SCM_KC[0:80,20:30], ax=axes[0], cbar=True, cmap="bone_r") #,
                #cbar_kws={'ticks': [0, 0.2, 0.4, 0.6, 0.8, 1]}, vmax=1, vmin=0)
    sns.heatmap(CommunityMeasure_SCM_L2[0:80,20:30], ax=axes[1], cbar=True, cmap="bone_r")

    axes[0].set_title('SCM - K-Clique')
    axes[1].set_title('SCM - Lais2')
    axes[0].set_xticklabels(range(20, 30))
    axes[0].set_yticks(range(0, 80, 10))
    axes[0].set_yticklabels(range(0, 80, 10))
    axes[1].set_xticklabels(range(20, 30))
    axes[1].set_yticks(range(0, 80, 10))
    axes[1].set_yticklabels(range(0, 80, 10))

    plt.tight_layout()
    plt.savefig('SCM_KC_L2.png')
    plt.close()



    #--- Creating CCMs ---#
    print('\n#--- Creating CCMs ---#\n')
    
    recombination_dict_Topics_lp = Community_Measurement.created_recombination_dict_Topics_crisp(communityTopicAssociation_dict_lp, recombination_dict_lp)
    recombination_dict_Topics_gm = Community_Measurement.created_recombination_dict_Topics_crisp(communityTopicAssociation_dict_gm, recombination_dict_gm)
    recombination_dict_Topics_kc = Community_Measurement.created_recombination_dict_Topics_overlap(communityTopicAssociation_dict_kc, recombination_dict_kc)
    recombination_dict_Topics_l2 = Community_Measurement.created_recombination_dict_Topics_overlap(communityTopicAssociation_dict_l2, recombination_dict_l2)

    Community_Measurement.doubleCheck_recombination_dict_Topics_crisp(recombination_dict_Topics_lp, recombination_dict_lp)
    Community_Measurement.doubleCheck_recombination_dict_Topics_crisp(recombination_dict_Topics_gm, recombination_dict_gm)
    Community_Measurement.doubleCheck_recombination_dict_Topics_overlap(recombination_dict_Topics_kc, recombination_dict_kc)
    Community_Measurement.doubleCheck_recombination_dict_Topics_overlap(recombination_dict_Topics_l2, recombination_dict_l2)

    CommunityMeasure_CCM_LP, CommunityMeasure_CCM_LP_columns = Community_Measurement.create_recombinationArray_Topics(recombination_dict_Topics_lp)
    CommunityMeasure_CCM_GM, CommunityMeasure_CCM_GM_columns = Community_Measurement.create_recombinationArray_Topics(recombination_dict_Topics_gm)
    CommunityMeasure_CCM_KC, CommunityMeasure_CCM_KC_columns = Community_Measurement.create_recombinationArray_Topics(recombination_dict_Topics_kc)
    CommunityMeasure_CCM_L2, CommunityMeasure_CCM_L2_columns = Community_Measurement.create_recombinationArray_Topics(recombination_dict_Topics_l2)


    # Descriptives
    print('--------- CCM Descriptibes ---------')
    print('Shape of CCM LP: ', np.shape(CommunityMeasure_CCM_LP))
    print('Shape of CCM GM: ', np.shape(CommunityMeasure_CCM_GM))
    print('Shape of CCM KC: ', np.shape(CommunityMeasure_CCM_KC))
    print('Shape of CCM L2: ', np.shape(CommunityMeasure_CCM_L2))
    
    diffusionPatternPos_CCM_LP = Misc.find_diffusionPatterns(CommunityMeasure_CCM_LP)
    diffusionPatternPos_CCM_LP, diff_sequence_list_CCM_LP, diff_sequence_sum_list_CCM_LP = Misc.find_diffusionSequenceAndLength(diffusionPatternPos_CCM_LP, CommunityMeasure_CCM_LP)
    # diff_pos = [ row, column, diffLength, diffSteps, patentsInDiff ]
    diffusionPatternPos_CCM_LP = np.array(diffusionPatternPos_CCM_LP)
    print('CCM Number of diffusion cycles / patterns in the ccm - LP -CCM-: ', len(diffusionPatternPos_CCM_LP))
    print('CCM Average diffusion pattern length - LP: ', np.mean(diffusionPatternPos_CCM_LP[:, 2]))
    print('CCM Average diffusion pattern patent engagement - LP -CCM-: ', np.mean(diff_sequence_sum_list_CCM_LP))

    diffusionPatternPos_CCM_GM = Misc.find_diffusionPatterns(CommunityMeasure_CCM_GM)
    diffusionPatternPos_CCM_GM, diff_sequence_list_CCM_GM, diff_sequence_sum_list_CCM_GM = Misc.find_diffusionSequenceAndLength(diffusionPatternPos_CCM_GM, CommunityMeasure_CCM_GM)
    # diff_pos = [ row, column, diffLength, diffSteps, patentsInDiff ]
    diffusionPatternPos_CCM_GM = np.array(diffusionPatternPos_CCM_GM)
    print('CCM Number of diffusion cycles / patterns in the ccm - GM -CCM-: ', len(diffusionPatternPos_CCM_GM))
    print('CCM Average diffusion pattern length - GM -CCM-: ', np.mean(diffusionPatternPos_CCM_GM[:, 2]))
    print('CCM Average diffusion pattern patent engagement - GM -CCM-: ', np.mean(diff_sequence_sum_list_CCM_GM))

    diffusionPatternPos_CCM_KC = Misc.find_diffusionPatterns(CommunityMeasure_CCM_KC)
    diffusionPatternPos_CCM_KC, diff_sequence_list_CCM_KC, diff_sequence_sum_list_CCM_KC = Misc.find_diffusionSequenceAndLength(diffusionPatternPos_CCM_KC, CommunityMeasure_CCM_KC)
    # diff_pos = [ row, column, diffLength, diffSteps, patentsInDiff ]
    diffusionPatternPos_CCM_KC = np.array(diffusionPatternPos_CCM_KC)
    print('CCM Number of diffusion cycles / patterns in the ccm - KC -CCM-: ', len(diffusionPatternPos_CCM_KC))
    print('CCM Average diffusion pattern length - KC -CCM-: ', np.mean(diffusionPatternPos_CCM_KC[:, 2]))
    print('CCM Average diffusion pattern patent engagement - KC -CCM-: ', np.mean(diff_sequence_sum_list_CCM_KC))

    diffusionPatternPos_CCM_L2 = Misc.find_diffusionPatterns(CommunityMeasure_CCM_L2)
    diffusionPatternPos_CCM_L2, diff_sequence_list_CCM_L2, diff_sequence_sum_list_CCM_L2 = Misc.find_diffusionSequenceAndLength(diffusionPatternPos_CCM_L2, CommunityMeasure_CCM_L2)
    # diff_pos = [ row, column, diffLength, diffSteps, patentsInDiff ]
    diffusionPatternPos_CCM_L2 = np.array(diffusionPatternPos_CCM_L2)
    print('CCM Number of diffusion cycles / patterns in the ccm - L2 -CCM-: ', len(diffusionPatternPos_CCM_L2))
    print('CCM Average diffusion pattern length - L2 -CCM-: ', np.mean(diffusionPatternPos_CCM_L2[:, 2]))
    print('CCM Average diffusion pattern patent engagement - L2 -CCM-: ', np.mean(diff_sequence_sum_list_CCM_L2))

    # Visualization
    fig, axes = plt.subplots(1, 2)
    sns.heatmap(CommunityMeasure_CCM_LP[100:180, 5:11], ax=axes[0], cbar=True, cmap="bone_r") #, cbar_kws={
        #'ticks': [0, 0.2, 0.4, 0.6, 0.8, 1]}, vmax=1, vmin=0)
    sns.heatmap(CommunityMeasure_CCM_GM[100:180, 5:11], ax=axes[1], cbar=True, cmap="bone_r")

    axes[0].set_title('CCM - Label Propagation')
    axes[1].set_title('CCM - Greedy Modularity')
    axes[0].set_xticklabels(range(5, 11))
    axes[0].set_yticks(range(0, 80, 10))
    axes[0].set_yticklabels(range(100, 180, 10))
    axes[1].set_xticklabels(range(5, 11))
    axes[1].set_yticks(range(0, 80, 10))
    axes[1].set_yticklabels(range(100, 180, 10))

    plt.tight_layout()
    plt.savefig('CCM_LP_GM.png')
    plt.close()

    fig, axes = plt.subplots(1, 2)
    sns.heatmap(CommunityMeasure_CCM_KC[100:180, 5:11], ax=axes[0], cbar=True, cmap="bone_r")  
    sns.heatmap(CommunityMeasure_CCM_L2[100:180, 5:11], ax=axes[1], cbar=True, cmap="bone_r") #, cbar_kws={
        #'ticks': [0, 1, 2, 3, 4, 5, 6]}, vmax=6, vmin=0)

    axes[0].set_title('CCM - K-Clique')
    axes[1].set_title('CCM - Lais2')
    axes[0].set_xticklabels(range(5, 11))
    axes[0].set_yticks(range(0, 80, 10))
    axes[0].set_yticklabels(range(100, 180, 10))
    axes[1].set_xticklabels(range(5, 11))
    axes[1].set_yticks(range(0, 80, 10))
    axes[1].set_yticklabels(range(100, 180, 10))

    plt.tight_layout()
    plt.savefig('CCM_KC_L2.png')
    plt.close()



    #--- Save Data ---#
    print('\n#--- Save Data ---#\n')
    
    filename = 'CommunityMeasure_SCM_LP'
    outfile = open(filename, 'wb')
    pk.dump(CommunityMeasure_SCM_LP, outfile)
    outfile.close()

    filename = 'CommunityMeasure_SCM_LP_columns'
    outfile = open(filename, 'wb')
    pk.dump(CommunityMeasure_SCM_LP_columns, outfile)
    outfile.close()

    filename = 'CommunityMeasure_SCM_GM'
    outfile = open(filename, 'wb')
    pk.dump(CommunityMeasure_SCM_GM, outfile)
    outfile.close()

    filename = 'CommunityMeasure_SCM_GM_columns'
    outfile = open(filename, 'wb')
    pk.dump(CommunityMeasure_SCM_GM_columns, outfile)
    outfile.close()

    filename = 'CommunityMeasure_SCM_KC'
    outfile = open(filename, 'wb')
    pk.dump(CommunityMeasure_SCM_KC, outfile)
    outfile.close()

    filename = 'CommunityMeasure_SCM_KC_columns'
    outfile = open(filename, 'wb')
    pk.dump(CommunityMeasure_SCM_KC_columns, outfile)
    outfile.close()

    filename = 'CommunityMeasure_SCM_L2'
    outfile = open(filename, 'wb')
    pk.dump(CommunityMeasure_SCM_L2, outfile)
    outfile.close()

    filename = 'CommunityMeasure_SCM_L2_columns'
    outfile = open(filename, 'wb')
    pk.dump(CommunityMeasure_SCM_L2_columns, outfile)
    outfile.close()

    filename = 'CommunityMeasure_CCM_LP'
    outfile = open(filename, 'wb')
    pk.dump(CommunityMeasure_CCM_LP, outfile)
    outfile.close()

    filename = 'CommunityMeasure_CCM_LP_columns'
    outfile = open(filename, 'wb')
    pk.dump(CommunityMeasure_CCM_LP_columns, outfile)
    outfile.close()

    filename = 'CommunityMeasure_CCM_GM'
    outfile = open(filename, 'wb')
    pk.dump(CommunityMeasure_CCM_GM, outfile)
    outfile.close()

    filename = 'CommunityMeasure_CCM_GM_columns'
    outfile = open(filename, 'wb')
    pk.dump(CommunityMeasure_CCM_GM_columns, outfile)
    outfile.close()

    filename = 'CommunityMeasure_CCM_KC'
    outfile = open(filename, 'wb')
    pk.dump(CommunityMeasure_CCM_KC, outfile)
    outfile.close()

    filename = 'CommunityMeasure_CCM_KC_columns'
    outfile = open(filename, 'wb')
    pk.dump(CommunityMeasure_CCM_KC_columns, outfile)
    outfile.close()

    filename = 'CommunityMeasure_CCM_L2'
    outfile = open(filename, 'wb')
    pk.dump(CommunityMeasure_CCM_L2, outfile)
    outfile.close()

    filename = 'CommunityMeasure_CCM_L2_columns'
    outfile = open(filename, 'wb')
    pk.dump(CommunityMeasure_CCM_L2_columns, outfile)
    outfile.close()
