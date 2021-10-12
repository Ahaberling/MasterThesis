
if __name__ == '__main__':

    # --- Import Libraries ---#
    print('\n#--- Import Libraries ---#\n')

    # Utility
    import os

    # Data handling
    import pickle as pk
    import numpy as np

    # Visualization
    import matplotlib.pyplot as plt

    # Custom functions
    from utilities_final.Measurement_utils import Similarities
    from utilities_final.Measurement_utils import Misc
    

    #--- Initialization ---#
    print('\n#--- Initialization ---#\n')

    path = 'D:/Universitaet Mannheim/MMDS 7. Semester/Master Thesis/Outline/Data/new'

    # load data
    os.chdir(path)

    #--- Load Data ---#
    print('#--- Load Data ---#')
    
    with open('direct_SCM', 'rb') as handle:
            direct_SCM = pk.load(handle)

    with open('direct_CCM', 'rb') as handle:
        direct_CCM = pk.load(handle)

    with open('columns_direct_CCM', 'rb') as handle:
        columns_direct_CCM = pk.load(handle)



    with open('CommunityMeasure_SCM_LP', 'rb') as handle:
        CommunityMeasure_SCM_LP = pk.load(handle)

    with open('CommunityMeasure_SCM_LP_columns', 'rb') as handle:
        CommunityMeasure_SCM_LP_columns = pk.load(handle)

    with open('CommunityMeasure_SCM_GM', 'rb') as handle:
        CommunityMeasure_SCM_GM = pk.load(handle)

    with open('CommunityMeasure_SCM_KC', 'rb') as handle:
        CommunityMeasure_SCM_KC = pk.load(handle)

    with open('CommunityMeasure_SCM_L2', 'rb') as handle:
        CommunityMeasure_SCM_L2 = pk.load(handle)


    with open('CommunityMeasure_CCM_LP', 'rb') as handle:
        CommunityMeasure_CCM_LP = pk.load(handle)

    with open('CommunityMeasure_CCM_LP_columns', 'rb') as handle:
        CommunityMeasure_CCM_LP_columns = pk.load(handle)

    with open('CommunityMeasure_CCM_GM', 'rb') as handle:
        CommunityMeasure_CCM_GM = pk.load(handle)

    with open('CommunityMeasure_CCM_GM_columns', 'rb') as handle:
        CommunityMeasure_CCM_GM_columns = pk.load(handle)

    with open('CommunityMeasure_CCM_KC', 'rb') as handle:
        CommunityMeasure_CCM_KC = pk.load(handle)

    with open('CommunityMeasure_CCM_KC_columns', 'rb') as handle:
        CommunityMeasure_CCM_KC_columns = pk.load(handle)

    with open('CommunityMeasure_CCM_L2', 'rb') as handle:
        CommunityMeasure_CCM_L2 = pk.load(handle)

    with open('CommunityMeasure_CCM_L2_columns', 'rb') as handle:
        CommunityMeasure_CCM_L2_columns = pk.load(handle)



    with open('EdgeWeight_SCM', 'rb') as handle:
        EdgeWeight_SCM = pk.load(handle)

    with open('EdgeWeight_CCM', 'rb') as handle:
        EdgeWeight_CCM = pk.load(handle)

    with open('EdgeWeight_CCM_columns', 'rb') as handle:
        EdgeWeight_CCM_columns = pk.load(handle)



    #--- Comparing SCMs ---#
    print('\n#--- Comparing SCMs ---#\n')

    list_allSCM = [direct_SCM, CommunityMeasure_SCM_LP, CommunityMeasure_SCM_GM,
                   CommunityMeasure_SCM_KC, CommunityMeasure_SCM_L2, EdgeWeight_SCM]
    list_allSCM_names = ['direct_SCM', 'CommunityMeasure_SCM_LP', 'CommunityMeasure_SCM_GM',
                         'CommunityMeasure_SCM_KC', 'CommunityMeasure_SCM_L2', 'EdgeWeight_SCM']


    # test if all SCMs have the appropriate column length
    Similarities.check_columnLength(list_allSCM, CommunityMeasure_SCM_LP_columns)

    # transform all diffusion arrays to row normalized and threshold arrays
    list_allSCM_threshold, list_allSCM_rowNorm = Similarities.normalized_and_binarize(list_allSCM, threshold=0.01, leeway=True)

    # find pattern start in SCM
    pattern_start_list = []
    for i in range(len(list_allSCM_threshold)):
        pattern_start = Similarities.find_patternStart(list_allSCM_threshold[i])  # [[row,column], ...]
        pattern_start_list.append(pattern_start)

    # find pattern length
    pattern_length_list = []
    min_length_threshold = 1
    for i in range(len(list_allSCM_threshold)):
        pattern_length = Similarities.find_pattern_length(list_allSCM_threshold[i], pattern_start_list[i])  # [[row, column, pattern_length], ...]
        pattern_length = [entry for entry in pattern_length if entry[2] >= min_length_threshold]
        pattern_length_list.append(pattern_length)

    # Alligned SCM descriptives
    print('-----Aligned SCM Descriptives-----')
    for i in range(len(list_allSCM_threshold)):
        diffusionPatternPos_SCM = Misc.find_diffusionPatterns(list_allSCM_threshold[i])
        diffusionPatternPos_SCM, diff_sequence_list_SCM, irrelevant = Misc.find_diffusionSequenceAndLength(diffusionPatternPos_SCM, list_allSCM_threshold[i])
        # diff_pos = [ row, column, diffLength, diffSteps, patentsInDiff ]
        diffusionPatternPos_SCM = np.array(diffusionPatternPos_SCM)

        print(list_allSCM_names[i])
        print('Number of diffusion cycles / patterns in the scm -SCM-: ', len(diffusionPatternPos_SCM))
        print('Average diffusion pattern length -SCM-: ', np.mean(diffusionPatternPos_SCM[:, 2]))

    #--- SCMs Similarities ---#
    print('\n#--- SCMs Similarities ---#\n')
    namePair_list, SimilaritiesPair_list_cosine, SimilaritiesPair_list_manhattan = Similarities.CM_similarities_byPair(list_allSCM_names, list_allSCM_threshold)

    print('Name of SCM Pairs: ', namePair_list)
    print('Mod Cosine by SCM Pair: ', SimilaritiesPair_list_cosine)
    print('Mod Manhattan by SCM Pair: ', SimilaritiesPair_list_manhattan)


    matrixSimilaritiesScore_cosine = 0
    matrixSimilaritiesScore_manhattan = 0

    # one Similarities score
    if len(SimilaritiesPair_list_cosine) != 0:
        matrixSimilaritiesScore_cosine = sum(SimilaritiesPair_list_cosine) / len(SimilaritiesPair_list_cosine)

    # one Similarities score
    if len(SimilaritiesPair_list_manhattan) != 0:
        matrixSimilaritiesScore_manhattan = sum(SimilaritiesPair_list_manhattan) / len(SimilaritiesPair_list_manhattan)

    print(matrixSimilaritiesScore_cosine)
    print(matrixSimilaritiesScore_manhattan)

    # Similarity by topic
    simScores_withinTopic_list_cosine_avg, simScores_withinTopic_list_manhattan_avg = Similarities.CM_similarities_byTopic(list_allSCM_names, list_allSCM_threshold)

    most_similarTopic_value_cosine = max(simScores_withinTopic_list_cosine_avg)
    most_similarTopic_value_manhattan = max(simScores_withinTopic_list_manhattan_avg)

    most_similarTopic_pos_cosine = np.where(simScores_withinTopic_list_cosine_avg == most_similarTopic_value_cosine)
    most_similarTopic_pos_manhattan = np.where(simScores_withinTopic_list_manhattan_avg == most_similarTopic_value_manhattan)

    least_similarTopic_value_cosine = min(simScores_withinTopic_list_cosine_avg)
    least_similarTopic_value_manhattan = min(simScores_withinTopic_list_manhattan_avg)

    least_similarTopic_pos_cosine = np.where(simScores_withinTopic_list_cosine_avg == least_similarTopic_value_cosine)
    least_similarTopic_pos_manhattan = np.where(simScores_withinTopic_list_manhattan_avg == least_similarTopic_value_manhattan)

    avg_similarTopic_cosine = sum(simScores_withinTopic_list_cosine_avg) / len(simScores_withinTopic_list_cosine_avg)
    avg_similarTopic_manhattan = sum(simScores_withinTopic_list_manhattan_avg) / len(simScores_withinTopic_list_manhattan_avg)


    print('\n Topic Cosine Similarities by Topic:')
    print('Highest Similarity -SCM- by Topic: ', most_similarTopic_value_cosine)
    print('Highest Similarity Topic Id -SCM- by Topic: : ', most_similarTopic_pos_cosine[0])
    print('Lowest Similarity -SCM- by Topic: : ', least_similarTopic_value_cosine)
    print('Lowest Similarity Topic Id -SCM- by Topic: : ', least_similarTopic_pos_cosine[0])
    print('Average Similarity between all topics -SCM-: ', avg_similarTopic_cosine)
    print('Number of columns excluded because at least one was column of each pair was always 0: ',len([x for x in simScores_withinTopic_list_cosine_avg if x == 0]))

    print('\n Topic Manhattan Similarities by Topic:')
    print('Highest Similarity -SCM- by Topic: ', most_similarTopic_value_manhattan)
    print('Highest Similarity Topic Id -SCM- by Topic: ', most_similarTopic_pos_manhattan[0])
    print('Lowest Similarity -SCM- by Topic: ', least_similarTopic_value_manhattan)
    print('Lowest Similarity Topic Id -SCM- by Topic: ', least_similarTopic_pos_manhattan[0])
    print('Average Similarity between all topics -SCM- by Topic: ', avg_similarTopic_manhattan)
    print('Number of columns excluded because both columns of each pair were always 0: ',len([x for x in simScores_withinTopic_list_manhattan_avg if x == 0]))


    # Visualization
    fig, ax = plt.subplots()
    plt.bar(range(100), simScores_withinTopic_list_cosine_avg[0:100], width=1.0, color='black', label='Cosine Similarity')
    plt.bar(range(100), simScores_withinTopic_list_manhattan_avg[0:100], width=1.0, color='darkblue', label='Manhattan Similarity', alpha=0.5)
    ax.set_ylim([0, 1.2])
    #plt.legend(loc='center left', bbox_to_anchor=(0.5, 0.6))
    plt.legend(loc='upper left') #, bbox_to_anchor=(0.5, 0.6))
    plt.xlabel("Topic IDs")
    plt.ylabel("Similarity")
    plt.savefig('TopicSimilarities_SCM.png')
    plt.close()




    #--- Comparing CCMs ---#
    print('\n#--- Comparing CCMs ---#\n')


    CCM_column_lists = [columns_direct_CCM, CommunityMeasure_CCM_LP_columns,
                        CommunityMeasure_CCM_GM_columns, CommunityMeasure_CCM_KC_columns,
                        CommunityMeasure_CCM_L2_columns, EdgeWeight_CCM_columns]
    CCM_column_list_names = ['columns_direct_CCM', 'CommunityMeasure_CCM_LP_columns',
                             'CommunityMeasure_CCM_GM_columns', 'CommunityMeasure_CCM_KC_columns',
                             'CommunityMeasure_CCM_L2_columns', 'EdgeWeight_CCM_columns']


    CCM_list = [direct_CCM, CommunityMeasure_CCM_LP, CommunityMeasure_CCM_GM,
                CommunityMeasure_CCM_KC, CommunityMeasure_CCM_L2, EdgeWeight_CCM]


    Similarities.extend_cD_recombinationDiffuion(CCM_list, slidingWindow_size=12, cd_CCM_posStart=1, cd_CCM_posEnd=5)

    recoArrays_threshold_list, recoArrays_rowNorm_list = Similarities.normalized_and_binarize(CCM_list, threshold=0.01, leeway=True)


    extended_threshold_arrays = Similarities.extend_recombination_columns(CCM_column_lists, recoArrays_threshold_list)

    # delete columns if they are 0 in all matrices (after aligning)
    columSum_vec_list = []
    for array in extended_threshold_arrays:
        columSum_vec = np.sum(array, axis=0)
        columSum_vec_list.append(columSum_vec)
        print(len(columSum_vec))

    columSum_vec_summed = np.sum(columSum_vec_list, axis=0)
    topicExclusionPosition = np.where(columSum_vec_summed == 0)
    print(topicExclusionPosition)
    print(len(topicExclusionPosition[0]))

    resized_threshold_arrays = []
    for array in extended_threshold_arrays:
        resized_threshold_arrays.append(np.delete(array, topicExclusionPosition, 1))

    # Alligned CCM descriptives
    print('-----Aligned CCM Descriptives-----')

    for i in range(len(resized_threshold_arrays)):
        diffusionPatternPos_SCM = Misc.find_diffusionPatterns(resized_threshold_arrays[i])
        diffusionPatternPos_SCM, diff_sequence_list_SCM, irrelevant = Misc.find_diffusionSequenceAndLength(diffusionPatternPos_SCM, resized_threshold_arrays[i])
        # diff_pos = [ row, column, diffLength, diffSteps, patentsInDiff ]
        diffusionPatternPos_SCM = np.array(diffusionPatternPos_SCM)

        print(CCM_column_list_names[i])
        print('Number of diffusion cycles / patterns in the ccm -CCM-: ', len(diffusionPatternPos_SCM))
        print('Average diffusion pattern length-CCM-: ', np.mean(diffusionPatternPos_SCM[:, 2]))

    print('shape of resized CCMs: ', np.shape(resized_threshold_arrays[0]))



    namePair_list, similarityPair_list_cosine, similarityPair_list_manhattan = Similarities.CM_similarities_byPair(CCM_column_list_names, resized_threshold_arrays)
    print('Name of CCM Pairs: ', namePair_list)
    print('Mod Cosine by CCM Pair: ', similarityPair_list_cosine)
    print('Mod Manhattan by CCM Pair: ', similarityPair_list_manhattan)

    # Similarities between all CCMs

    # Initiallize
    matrixSimilarityScore_cosine = 0
    matrixSimilarityScore_manhattan = 0

    # one similarity score
    if len(similarityPair_list_cosine) != 0:
        matrixSimilarityScore_cosine = sum(similarityPair_list_cosine) / len(similarityPair_list_cosine)

    # one similarity score
    if len(similarityPair_list_manhattan) != 0:
        matrixSimilarityScore_manhattan = sum(similarityPair_list_manhattan) / len(similarityPair_list_manhattan)

    print(matrixSimilarityScore_cosine)
    print(matrixSimilarityScore_manhattan)


    # Similarities by Topic
    simScores_withinTopic_list_cosine_avg, simScores_withinTopic_list_manhattan_avg = Similarities.CM_similarities_byTopic(CCM_column_list_names, resized_threshold_arrays)  # , slices_toExclude)

    most_similarTopic_value_cosine = max(simScores_withinTopic_list_cosine_avg)
    most_similarTopic_value_manhattan = max(simScores_withinTopic_list_manhattan_avg)

    most_similarTopic_pos_cosine = np.where(simScores_withinTopic_list_cosine_avg == most_similarTopic_value_cosine)
    most_similarTopic_pos_manhattan = np.where(
        simScores_withinTopic_list_manhattan_avg == most_similarTopic_value_manhattan)

    least_similarTopic_value_cosine = min(simScores_withinTopic_list_cosine_avg)
    least_similarTopic_value_manhattan = min(simScores_withinTopic_list_manhattan_avg)

    least_similarTopic_pos_cosine = np.where(simScores_withinTopic_list_cosine_avg == least_similarTopic_value_cosine)
    least_similarTopic_pos_manhattan = np.where(
        simScores_withinTopic_list_manhattan_avg == least_similarTopic_value_manhattan)

    avg_similarTopic_cosine = sum(simScores_withinTopic_list_cosine_avg) / len(simScores_withinTopic_list_cosine_avg)
    avg_similarTopic_manhattan = sum(simScores_withinTopic_list_manhattan_avg) / len(simScores_withinTopic_list_manhattan_avg)


    # descriptives
    print('\n Topic Cosine Similarities by Topic:')
    print('Highest Similarity by Topic -CCM-: ', most_similarTopic_value_cosine)
    print('Highest Similarity combination id by Topic -CCM-: ', most_similarTopic_pos_cosine[0])
    print('Lowest Similarity by Topic -CCM-: ', least_similarTopic_value_cosine)
    print('Lowest Similarity combination Id by Topic -CCM-: ', least_similarTopic_pos_cosine[0])
    print('Average Similarity between all topic combination by Topic -CCM-: ', avg_similarTopic_cosine)
    print('Number of columns excluded because at least one was column of each pair was always 0: ', len([x for x in simScores_withinTopic_list_cosine_avg if x == 0]))

    print('\n Topic Manhattan Similarities by Topic:')
    print('Highest Similarity by Topic -CCM-: ', most_similarTopic_value_manhattan)
    print('Highest Similarity Topic Id by Topic -CCM-: ', most_similarTopic_pos_manhattan[0])
    print('Lowest Similarity by Topic -CCM-: ', least_similarTopic_value_manhattan)
    print('Lowest Similarity Topic Id by Topic -CCM-: ', least_similarTopic_pos_manhattan[0])
    print('Average Similarity between all topics by Topic -CCM-: ', avg_similarTopic_manhattan)
    print('Number of columns excluded because both columns of each pair were always 0: ',len([x for x in simScores_withinTopic_list_manhattan_avg if x == 0]))

    # visualization
    fig, ax = plt.subplots()
    plt.bar(range(100), simScores_withinTopic_list_cosine_avg[0:100], width=1.0, color='black',
            label='Cosine Similarity')
    plt.bar(range(100), simScores_withinTopic_list_manhattan_avg[0:100], width=1.0, color='darkblue',
            label='Manhattan Similarity', alpha=0.5)
    plt.legend(loc='center left')
    plt.xlabel("Recombination IDs")
    plt.ylabel("Similarity")
    plt.savefig('TopicSimilarities_CCM.png')
    plt.close()