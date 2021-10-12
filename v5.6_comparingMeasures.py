'''
1. get diffusion and recombination arrays from all three approaches + recombination dicts
2. diffusion:
    create list with all topics used in all three approaches combined
    for every array, extent them with this list so all have the same dimensions
3. recombination:
    create list with all recombinations used in all three approaches combined
    for every array, extent them with this list so all have the same dimensions

how often was threshhold reached?
how many counts?
cosine / jaccard similarity between arrays
are there columns identical, that are not sum = 0?
pick confident cases of all three approaches and compare them within approaches
argument:   finding identical? cool, validates the finding
            finding (almost) no identical? cool, that shows how complex it is and that a multitude of measures is necessary
            finding knowledge recombination is complex, so one plain measure is not enough.
            different typed of recombination are found in different ways. not clear yet
            in what way the recombinations differ

rename referenceMeasure to intuitiveMeasure

idea:
comparing diffusion should be fine

idea:
for recombination comparability focus first on recombination counts.
in order to compare recombination, the cD measure might be adpated by adding 1 to all 10 or 11 cells following a 1, vertically
look for patterns 111 111 111 0
if there are patterns like this in array[10:,:] then overthink this

'''

if __name__ == '__main__':

    # --- Import Libraries ---#
    print('\n#--- Import Libraries ---#\n')

    import pickle as pk
    import numpy as np

    import tqdm
    import os

    import statistics
    from scipy import spatial

    # --- Initialization ---#
    print('\n# --- Initialization ---#\n')

    os.chdir('D:/Universitaet Mannheim/MMDS 7. Semester/Master Thesis/Outline/Data/Cleaning Robots')

    with open('pattern_array_reference_diff', 'rb') as handle:
        pattern_array_reference_diff = pk.load(handle)

    with open('columns_reference_diff', 'rb') as handle:
        columns_reference_diff = pk.load(handle)

    with open('pattern_array_reference_reco', 'rb') as handle:
        pattern_array_reference_reco = pk.load(handle)

    with open('columns_reference_reco', 'rb') as handle:
        columns_reference_reco = pk.load(handle)



    with open('diffusionArray_Topics_lp', 'rb') as handle:
        diffusionArray_Topics_lp = pk.load(handle)

    with open('diffusionArray_Topics_lp_columns', 'rb') as handle:
        diffusionArray_Topics_lp_columns = pk.load(handle)

    with open('diffusionArray_Topics_gm', 'rb') as handle:
        diffusionArray_Topics_gm = pk.load(handle)

    with open('diffusionArray_Topics_kc', 'rb') as handle:
        diffusionArray_Topics_kc = pk.load(handle)

    with open('diffusionArray_Topics_l2', 'rb') as handle:
        diffusionArray_Topics_l2 = pk.load(handle)


    with open('recombinationArray_Topics_lp', 'rb') as handle:
        recombinationArray_Topics_lp = pk.load(handle)

    with open('recombinationArray_Topics_lp_columns', 'rb') as handle:
        recombinationArray_Topics_lp_columns = pk.load(handle)

    with open('recombinationArray_Topics_gm', 'rb') as handle:
        recombinationArray_Topics_gm = pk.load(handle)

    with open('recombinationArray_Topics_gm_columns', 'rb') as handle:
        recombinationArray_Topics_gm_columns = pk.load(handle)

    with open('recombinationArray_Topics_kc', 'rb') as handle:
        recombinationArray_Topics_kc = pk.load(handle)

    with open('recombinationArray_Topics_kc_columns', 'rb') as handle:
        recombinationArray_Topics_kc_columns = pk.load(handle)

    with open('recombinationArray_Topics_l2', 'rb') as handle:
        recombinationArray_Topics_l2 = pk.load(handle)

    with open('recombinationArray_Topics_l2_columns', 'rb') as handle:
        recombinationArray_Topics_l2_columns = pk.load(handle)



    with open('diffusion_array_edgeWeight', 'rb') as handle:
        diffusion_array_edgeWeight = pk.load(handle)

    with open('columns_diff_edgeWeight', 'rb') as handle:
        columns_diff_edgeWeight = pk.load(handle)

    with open('recombinationDiffusion_edgeWeight', 'rb') as handle:
        recombinationDiffusion_edgeWeight = pk.load(handle)

    with open('columns_recom_edgeWeight', 'rb') as handle:
        columns_recom_edgeWeight = pk.load(handle)



#--- Comparing Diffusion arrays ---#


    list_allSCM = [pattern_array_reference_diff, diffusionArray_Topics_lp, diffusionArray_Topics_gm, diffusionArray_Topics_kc, diffusionArray_Topics_l2, diffusion_array_edgeWeight]
    list_allSCM_names = ['pattern_array_reference_diff', 'diffusionArray_Topics_lp', 'diffusionArray_Topics_gm', 'diffusionArray_Topics_kc', 'diffusionArray_Topics_l2', 'diffusion_array_edgeWeight']

    list_cdSCM = [diffusionArray_Topics_lp, diffusionArray_Topics_gm, diffusionArray_Topics_kc, diffusionArray_Topics_l2]
    list_cdSCM_names = ['diffusionArray_Topics_lp', 'diffusionArray_Topics_gm', 'diffusionArray_Topics_kc', 'diffusionArray_Topics_l2']


    from utilities_old.my_comparative_utils import ComparativeMeasures


    # test if all SCMs have the appropriate column length
    ComparativeMeasures.check_columnLength(list_allSCM, diffusionArray_Topics_lp_columns)

    # transform all diffusion arrays to row normalized and threshold arrays
    list_allSCM_threshold, list_allSCM_rowNorm = ComparativeMeasures.normalized_and_binarize(list_allSCM, threshold=0.01, leeway=True)

    # for comparative example, choose the topics with the higest similarity between the arrays or between direct, link and 1 cd measure (to decide on the cd most suitable)
    # find diffusion position
    #modArray_dict = {}
    
    # find pattern start in SCM
    pattern_start_list = []
    for i in range(len(list_allSCM_threshold)):
        pattern_start = ComparativeMeasures.find_patternStart(list_allSCM_threshold[i]) # [[row,column], ...]
        pattern_start_list.append(pattern_start)    

    # find pattern length
    pattern_length_list = []
    min_length_threshold = 1
    for i in range(len(list_allSCM_threshold)):
        pattern_length = ComparativeMeasures.find_pattern_length(list_allSCM_threshold[i], pattern_start_list[i]) # [[row, column, pattern_length], ...]
        pattern_length = [entry for entry in pattern_length if entry[2] >= min_length_threshold] 
        pattern_length_list.append(pattern_length)

    from utilities_old.my_measure_utils import Misc

    # Alligned SCM descriptives
    for i in range(len(list_allSCM_threshold)):

        diffusionPatternPos_SCM = Misc.find_diffusionPatterns(list_allSCM_threshold[i])
        diffusionPatternPos_SCM, diff_sequence_list_SCM, irrelevant = Misc.find_diffusionSequenceAndLength(diffusionPatternPos_SCM, list_allSCM_threshold[i])
        # diffusionPatternPos_CCM = Misc.find_diffusionStepsAndPatternPerDiffusion(diffusionPatternPos_CCM, diff_sequence_list_SCM)
        # diff_pos = [ row, column, diffLength, diffSteps, patentsInDiff ]
        diffusionPatternPos_SCM = np.array(diffusionPatternPos_SCM)

        print(list_allSCM_names[i])
        print('Number of diffusion cycles / patterns in the scm: ', len(diffusionPatternPos_SCM))
        print('Average diffusion pattern length: ', np.mean(diffusionPatternPos_SCM[:, 2]))

        '''
        patterns_perTopic, pattern_lengths_perTopic = ComparativeMeasures.alligned_SCM_descriptives(diffusionArray_Topics_lp_columns, pattern_length_list[i])
        PatternLength_withinTopic_avg = []
        for j in pattern_lengths_perTopic:
            if j != []:
                PatternLength_withinTopic_avg.append(np.mean(j))

        SCM_threshold = list_allSCM_threshold[i]
        SCM_threshold_size = np.size(SCM_threshold)
        SCM_threshold_sum = np.sum(SCM_threshold)
        rowSum_vec = np.sum(SCM_threshold, axis=0)
        patternStars_inTopics_avg = np.mean(rowSum_vec)

        print('\n', list_allSCM_threshold[i])
        print('Number of Diffusion Cycles total: ', len(pattern_length_list[i]))
        print('Average number of Diffusion Cycles per topic: ', (sum(patterns_perTopic) / len(patterns_perTopic)))
        print('Number of diffusion entries total: ', SCM_threshold_sum, ' of ', SCM_threshold_size)
        print('Average number of diffusion entries per topic: ', patternStars_inTopics_avg, ' of ', len(SCM_threshold))
        print('Average diffusion length: ', np.mean(PatternLength_withinTopic_avg), 'max: ', max(PatternLength_withinTopic_avg), 'min: ',
              min(PatternLength_withinTopic_avg), 'median: ', np.median(PatternLength_withinTopic_avg), 'mode: ', statistics.mode(PatternLength_withinTopic_avg))
        '''

    # cosine focuses more on the similarity in diffusion instead of similarity in diffusion and non-diffusion. This is because of the
    # numinator in the fraction. a match 0-0 match in the lists, does not increase the numinator. it is treated as a mismatch. 0s have no
    # influence on the denominator. Better: 0-0 matches have no impact neither the numerator nor the denominator

    '''
    x = [0,0,1,1,1]
    y = [0,0,0,1,1]

    print(ComparativeMeasures.manhattan_sim_mod(x,y))
    '''
    # Similarities between SCM pairs

    # namePair_list =                   [[patternArray1, patternArray2], [patternArray1, patternArray3], ...]
    # similarityPair_list_cosine =      [[similarity score],              [similarity score],            ...]
    # similarityPair_list_manhattan =   [[similarity score],              [similarity score],            ...]
    namePair_list, similarityPair_list_cosine, similarityPair_list_manhattan = ComparativeMeasures.SCM_similarities_byColumn(list_allSCM_names, list_allSCM_threshold)

    print(namePair_list)
    print(similarityPair_list_cosine)
    print(similarityPair_list_manhattan)

    # Similarities between all SCMs

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
    # position 1,9,10,11 are weird. do they correpsond to one array? all realted to gm. only lp + gm seems ok
    # I will probably exclude gm, because 4 out of 5 gm combinations are outliers with similarity scores
    # two orders of magnitude smaller then the rest.
    print(matrixSimilarityScore_manhattan)
    # print(namePair_list)                    # 1,5,9,10,11
    '''
    # New colculation with the exclution of the unfitting SCM (Greedy modularity)
    slices_toExclude = [1,5,9,10,11]

    # 4 out of 5 similarities involving the gm approach are very bad, so we exclude them ( similarity two magnitudes smaller)
    # lp is 0, 5, 6, 7, 8

    similarityPair_list_cosine_withoutGM = list(np.delete(similarityPair_list_cosine, slices_toExclude, axis=0))
    similarityPair_list_manhattan_withoutGM = list(np.delete(similarityPair_list_manhattan, slices_toExclude, axis=0))

    # Initialization
    matrixSimilarityScore_cosine_withoutGM = 0
    matrixSimilarityScore_manhattan_withoutGM = 0

    # one similarity score
    if len(similarityPair_list_cosine_withoutGM) != 0:
        matrixSimilarityScore_cosine_withoutGM = sum(similarityPair_list_cosine_withoutGM) / len(similarityPair_list_cosine_withoutGM)

    # one similarity score
    if len(similarityPair_list_manhattan_withoutGM) != 0:
        matrixSimilarityScore_manhattan_withoutGM = sum(similarityPair_list_manhattan_withoutGM) / len(similarityPair_list_manhattan_withoutGM)

    print('\n', matrixSimilarityScore_cosine_withoutGM)
    print(matrixSimilarityScore_manhattan_withoutGM)
    '''





###---------------------

    # get similarities between vectors of the same topic. max, min, avg, mode, media, distribution

    simScores_withinTopic_list_cosine_avg, simScores_withinTopic_list_manhattan_avg = ComparativeMeasures.SCM_topic_similarities(list_allSCM_names, list_allSCM_threshold) #, slices_toExclude)

        # calculate the following only with values that are not -9999
    # there are a lot topics falling through. check if this filter is correct
    # also check if there are really topics with similarity of 1 across all pairs
    #simScores_withinTopic_list_cosine_avg_clean = [x for x in simScores_withinTopic_list_cosine_avg if x != -9999]
    #simScores_withinTopic_list_manhattan_avg_clean = [x for x in simScores_withinTopic_list_manhattan_avg if x != -9999]

    # -9999 in cosine means: at least one column of all column pairs was always 0
    # -9999 in manhattan means: both columns of all column pairs were always 0

    #simScores_withinTopic_list_cosine_avg_clean = simScores_withinTopic_list_cosine_avg
    #simScores_withinTopic_list_manhattan_avg_clean = simScores_withinTopic_list_manhattan_avg


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

    import matplotlib.pyplot as plt

    print(len(range(100)))
    print(len(simScores_withinTopic_list_cosine_avg[0:99]))
    fig, ax = plt.subplots()
    #plt.bar(range(len(simScores_withinTopic_list_cosine_avg)), simScores_withinTopic_list_cosine_avg, width=1.0, color='white', label='Cosine Similarity')
    #plt.bar(range(len(simScores_withinTopic_list_cosine_avg)), simScores_withinTopic_list_manhattan_avg, width=1.0, color='darkblue', label='Manhattan Similarity', alpha=0.5)
    plt.bar(range(100), simScores_withinTopic_list_cosine_avg[0:100], width=1.0, color='black', label='Cosine Similarity')
    plt.bar(range(100), simScores_withinTopic_list_manhattan_avg[0:100], width=1.0, color='darkblue', label='Manhattan Similarity', alpha=0.5)
    #plt.legend(loc='upper left')
    plt.legend(loc='center left', bbox_to_anchor=(0.5,0.6))
    #ax.set_ylim([0, 1.2])
    #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel("Topic IDs")
    plt.ylabel("Similarity")
    #plt.show()
    os.chdir('D:/Universitaet Mannheim/MMDS 7. Semester/Master Thesis/Outline/Plots')
    plt.savefig('TopicSimilarities_SCM.png')
    os.chdir('D:/Universitaet Mannheim/MMDS 7. Semester/Master Thesis/Outline/Data/Cleaning Robots')
    plt.close()




    print('\n Topic Cosine Similarities between approaches:')
    print('Highest Similarity: ', most_similarTopic_value_cosine)
    print('Highest Similarity Topic Id: ', most_similarTopic_pos_cosine[0])
    print('Lowest Similarity: ', least_similarTopic_value_cosine)
    print('Lowest Similarity Topic Id: ', least_similarTopic_pos_cosine[0])
    print('Average Similarity between all topics: ', avg_similarTopic_cosine)
    #print('Number of columns excluded because at least one was column of each pair was always 0: ', len([x for x in simScores_withinTopic_list_cosine_avg if x == -9999]))
    print('Number of columns excluded because at least one was column of each pair was always 0: ', len([x for x in simScores_withinTopic_list_cosine_avg if x == 0]))

    print('\n Topic Manhattan Similarities between approaches:')
    print('Highest Similarity: ', most_similarTopic_value_manhattan)
    print('Highest Similarity Topic Id: ', most_similarTopic_pos_manhattan[0])
    print('Lowest Similarity: ', least_similarTopic_value_manhattan)
    print('Lowest Similarity Topic Id: ', least_similarTopic_pos_manhattan[0])
    print('Average Similarity between all topics: ', avg_similarTopic_manhattan)
    #print('Number of columns excluded because both columns of each pair were always 0: ', len([x for x in simScores_withinTopic_list_manhattan_avg if x == -9999]))
    print('Number of columns excluded because both columns of each pair were always 0: ', len([x for x in simScores_withinTopic_list_manhattan_avg if x == 0]))


    #print(len(list_of_allArrays_threshold[0]))
    #print(len(list_of_allArrays_threshold))

    vialization_higestTopicSim = np.zeros((len(list_allSCM_threshold[0]), len(list_allSCM_threshold)), dtype=int)
    #print(np.shape(vialization_higestTopicSim))
    for i in range(len(list_allSCM_threshold)):
        #print(list_of_allArrays_threshold[i][:,23])
        #print(vialization_higestTopicSim[:,i])
        vialization_higestTopicSim[:,i] = list_allSCM_threshold[i][:,23]

    #print(vialization_higestTopicSim)

    # A. matrix with i = j = pattern arrays.    Aij = similarity between arrays
    # B. matrix with i = j = pattern arrays.    Aij = similarity between topic vecs of most similar topic
    # C. matrix with i = j = pattern arrays.    Aij = similarity between topic vecs of least similar topic
    # D. x = average topic similarity between all topics and all pattern arrays


    #1. modify cd arrays to that they have 11 following ones. (ok maybe not for diffusion, just for recombination)
    #2. calculate fraction and threshold array for all
    #3. write diffusion dicts
    #4. count:
    #       average number of diffusion cycles for all topics
    #       average length of diffusion (without diffusions of length 0?)
    #       calculate cosine and jaccard sim between matricies (with all matricies?)
    #       average cosine and jaccard for a topic column
    #       pick examplary patent(s) for recombination and diffusion









###----------- Recombination ------------------------------------------------------------------------------
#--- Comparing Recombination arrays ---#



    CCM_column_lists = [columns_reference_reco, recombinationArray_Topics_lp_columns, recombinationArray_Topics_gm_columns, recombinationArray_Topics_kc_columns, recombinationArray_Topics_l2_columns, columns_recom_edgeWeight]
    CCM_column_list_names = ['columns_reference_reco', 'recombinationArray_Topics_lp_columns', 'recombinationArray_Topics_gm_columns', 'recombinationArray_Topics_kc_columns', 'recombinationArray_Topics_l2_columns', 'columns_recom_edgeWeight']

    print(len(recombinationDiffusion_edgeWeight.T))
    print(len(columns_recom_edgeWeight))
    print(len(np.unique(columns_recom_edgeWeight, axis=0)))

    CCM_list = [pattern_array_reference_reco, recombinationArray_Topics_lp, recombinationArray_Topics_gm,
                       recombinationArray_Topics_kc, recombinationArray_Topics_l2, recombinationDiffusion_edgeWeight]

    for CCM in CCM_list:
        columSum_vec = np.sum(CCM, axis= 0)
        print(np.where(columSum_vec == 0))
        print(len(np.where(columSum_vec == 0)[0]))
        #todo: why is this not empty for lp????

        # 0 154 0 0 0 0

    ComparativeMeasures.extend_cD_recombinationDiffuion(CCM_list, slidingWindow_size=12, cd_CCM_posStart=1, cd_CCM_posEnd=5)

    #CCM_list_mod = [CCM_list[0], extend_cD_arrays[:], CCM_list[-1]]

    recoArrays_threshold_list, recoArrays_rowNorm_list = ComparativeMeasures.normalized_and_binarize(CCM_list, threshold=0.01, leeway=True)

    #print(sum(sum(recoArrays_threshold_list[0])))
    #print(sum(sum(recoArrays_threshold_list[1])))

    for ccm in CCM_column_lists:
        print(np.shape(ccm))

        #(5668, 2)
        #(3061, 2)
        #(42, 2)
        #(128, 2)
        #(454, 2)
        #(4305, 2)

    print('+++++++++++++')
    for ccm in recoArrays_threshold_list:
        print(np.shape(ccm))

        #(189, 5668)
        #(189, 3061)
        #(189, 42)
        #(189, 128)
        #(189, 454)
        #(189, 4305)

        # todo maybe dont restrict to communities of size 3 or bigger, maybe do idk

    extended_threshold_arrays = ComparativeMeasures.extend_recombination_columns(CCM_column_lists, recoArrays_threshold_list)
    #extended_threshold_arrays = ComparativeMeasures.extend_recombination_columns(CCM_column_lists, CCM_list)

    # delete collumns if they are 0 in all matrices

    columSum_vec_list = []
    for array in extended_threshold_arrays:
        columSum_vec = np.sum(array, axis=0)
        columSum_vec_list.append(columSum_vec)
        print(len(columSum_vec))

    columSum_vec_summed = np.sum(columSum_vec_list, axis=0)
    topicExclusionPosition = np.where(columSum_vec_summed == 0)
    print(topicExclusionPosition)
    print(len(topicExclusionPosition[0])) # 7119 are deleted from 8157 -> 1038 left
    # todo why are these so many? maybe because a lot is lost in binarization. maybe extend befor binarizing


    resized_threshold_arrays = []
    for array in extended_threshold_arrays:
        resized_threshold_arrays.append(np.delete(array, topicExclusionPosition, 1))

    for i in range(len(resized_threshold_arrays)):
        diffusionPatternPos_SCM = Misc.find_diffusionPatterns(resized_threshold_arrays[i])
        diffusionPatternPos_SCM, diff_sequence_list_SCM, irrelevant = Misc.find_diffusionSequenceAndLength(
            diffusionPatternPos_SCM, resized_threshold_arrays[i])
        # diffusionPatternPos_CCM = Misc.find_diffusionStepsAndPatternPerDiffusion(diffusionPatternPos_CCM, diff_sequence_list_SCM)
        # diff_pos = [ row, column, diffLength, diffSteps, patentsInDiff ]
        diffusionPatternPos_SCM = np.array(diffusionPatternPos_SCM)

        print(CCM_column_list_names[i])
        print('Number of diffusion cycles / patterns in the ccm: ', len(diffusionPatternPos_SCM))
        print('Average diffusion pattern length: ', np.mean(diffusionPatternPos_SCM[:, 2]))



    for array in resized_threshold_arrays:
        print(np.shape(array))

    #todo: possible reason for the lp mistake of finding "to little" recombiantions. in the beginning al communitys of size 1 and 2 are excluded. check if
    # the dicts used later account for that or if they access the graph directly.

    #todo either community labeling or the recombination dict is wrong. the later indicates that the former should have  community id 37 and 119 in window 89 aka 2670

    for ccm in resized_threshold_arrays:
        print('new')
        print(np.size(ccm))
        print(np.count_nonzero(ccm))            # 694 3858 340 1667 3418 2197
        print(np.size(ccm)-np.count_nonzero(ccm))


    namePair_list, similarityPair_list_cosine, similarityPair_list_manhattan = ComparativeMeasures.SCM_similarities_byColumn(
        CCM_column_list_names, resized_threshold_arrays)
    # probably better to just compare direct and edgeweight and seperatly compare all the cd measures.
    print(namePair_list)    # cosine: gm is not usefull at all. maybe try it with more then just the 3 biggest topics for network creation.
    print(similarityPair_list_cosine)   # 0 = 0 1 2 ((3)) 5 (6) (7) ((8)) 9 10 11 (12) ((13)) ((14)) Only direct and edge weigth are similar
    print(similarityPair_list_manhattan)

    # Similarities between all SCMs

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
    # position 1,9,10,11 are weird. do they correpsond to one array? all realted to gm. only lp + gm seems ok
    # I will probably exclude gm, because 4 out of 5 gm combinations are outliers with similarity scores
    # two orders of magnitude smaller then the rest.
    print(matrixSimilarityScore_manhattan)
    # print(namePair_list)                    # 1,5,9,10,11
    '''
    # New colculation with the exclution of the unfitting SCM (Greedy modularity)
    slices_toExclude = [1, 5, 9, 10, 11]

    # 4 out of 5 similarities involving the gm approach are very bad, so we exclude them ( similarity two magnitudes smaller)
    # lp is 0, 5, 6, 7, 8

    similarityPair_list_cosine_withoutGM = list(np.delete(similarityPair_list_cosine, slices_toExclude, axis=0))
    similarityPair_list_manhattan_withoutGM = list(np.delete(similarityPair_list_manhattan, slices_toExclude, axis=0))

    # Initialization
    matrixSimilarityScore_cosine_withoutGM = 0
    matrixSimilarityScore_manhattan_withoutGM = 0

    # one similarity score
    if len(similarityPair_list_cosine_withoutGM) != 0:
        matrixSimilarityScore_cosine_withoutGM = sum(similarityPair_list_cosine_withoutGM) / len(
            similarityPair_list_cosine_withoutGM)

    # one similarity score
    if len(similarityPair_list_manhattan_withoutGM) != 0:
        matrixSimilarityScore_manhattan_withoutGM = sum(similarityPair_list_manhattan_withoutGM) / len(
            similarityPair_list_manhattan_withoutGM)

    print('\n', matrixSimilarityScore_cosine_withoutGM)
    print(matrixSimilarityScore_manhattan_withoutGM)
    '''
    ###---------------------

    # get similarities between vectors of the same topic. max, min, avg, mode, media, distribution

    simScores_withinTopic_list_cosine_avg, simScores_withinTopic_list_manhattan_avg = ComparativeMeasures.SCM_topic_similarities(
        CCM_column_list_names, resized_threshold_arrays) #, slices_toExclude)


    # calculate the following only with values that are not -9999
    # there are a lot topics falling through. check if this filter is correct
    # also check if there are really topics with similarity of 1 across all pairs
    # simScores_withinTopic_list_cosine_avg_clean = [x for x in simScores_withinTopic_list_cosine_avg if x != -9999]
    #simScores_withinTopic_list_manhattan_avg_clean = [x for x in simScores_withinTopic_list_manhattan_avg if x != -9999]

    # -9999 in cosine means: at least one column of all column pairs was always 0
    # -9999 in manhattan means: both columns of all column pairs were always 0

    #simScores_withinTopic_list_cosine_avg_clean = simScores_withinTopic_list_cosine_avg
    #simScores_withinTopic_list_manhattan_avg_clean = simScores_withinTopic_list_manhattan_avg

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

    avg_similarTopic_cosine = sum(simScores_withinTopic_list_cosine_avg) / len(
        simScores_withinTopic_list_cosine_avg)
    avg_similarTopic_manhattan = sum(simScores_withinTopic_list_manhattan_avg) / len(
        simScores_withinTopic_list_manhattan_avg)



    fig, ax = plt.subplots()
    #plt.bar(range(len(simScores_withinTopic_list_cosine_avg)), simScores_withinTopic_list_cosine_avg, width=1.0, color='darkblue', label='Cosine Similarity')
    #plt.bar(range(len(simScores_withinTopic_list_cosine_avg)), simScores_withinTopic_list_manhattan_avg, width=1.0, color='darkred', label='Manhattan Similarity', alpha=0.5)
    plt.bar(range(100), simScores_withinTopic_list_cosine_avg[0:100], width=1.0, color='black', label='Cosine Similarity')
    plt.bar(range(100), simScores_withinTopic_list_manhattan_avg[0:100], width=1.0, color='darkblue', label='Manhattan Similarity', alpha=0.5)
    plt.legend(loc='center left') #, bbox_to_anchor=(0.5,0.6))
    #plt.legend(loc='best')
    #ax.set_ylim([0, 1.2])
    #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel("Topic IDs")
    plt.ylabel("Similarity")
    #plt.show()
    os.chdir('D:/Universitaet Mannheim/MMDS 7. Semester/Master Thesis/Outline/Plots')
    plt.savefig('TopicSimilarities_CCM.png')
    os.chdir('D:/Universitaet Mannheim/MMDS 7. Semester/Master Thesis/Outline/Data/Cleaning Robots')
    plt.close()


    print('\n Topic Cosine Similarities between approaches:')
    print('Highest Similarity: ', most_similarTopic_value_cosine)
    print('Highest Similarity combination id: ', most_similarTopic_pos_cosine[0])
    print('Lowest Similarity: ', least_similarTopic_value_cosine)
    print('Lowest Similarity combination Id: ', least_similarTopic_pos_cosine[0])
    print('Average Similarity between all topic combination: ', avg_similarTopic_cosine)
    # print('Number of columns excluded because at least one was column of each pair was always 0: ', len([x for x in simScores_withinTopic_list_cosine_avg if x == -9999]))
    print('Number of columns excluded because at least one was column of each pair was always 0: ',
          len([x for x in simScores_withinTopic_list_cosine_avg if x == 0]))

    print('\n Topic Manhattan Similarities between approaches:')
    print('Highest Similarity: ', most_similarTopic_value_manhattan)
    print('Highest Similarity Topic Id: ', most_similarTopic_pos_manhattan[0])
    print('Lowest Similarity: ', least_similarTopic_value_manhattan)
    print('Lowest Similarity Topic Id: ', least_similarTopic_pos_manhattan[0])
    print('Average Similarity between all topics: ', avg_similarTopic_manhattan)
    # print('Number of columns excluded because both columns of each pair were always 0: ', len([x for x in simScores_withinTopic_list_manhattan_avg if x == -9999]))
    print('Number of columns excluded because both columns of each pair were always 0: ',
          len([x for x in simScores_withinTopic_list_manhattan_avg if x == 0]))



    # USE FIND SEQUENCE EXEMPLARY


    print('end')

    # print(len(list_of_allArrays_threshold[0]))
    # print(len(list_of_allArrays_threshold))
















    '''
    # find diffusion position
    # modArray_dict = {}
    recomb_pos_list = []
    recomb_length_list = []
    for i in range(len(CCM_list)):
        recomb_pos = ComparativeMeasures.find_recombination(CCM_list[i])
        recomb_length_list.append(recomb_pos)

    # fine diffusion length
    min_length_threshold = 1
    for i in range(len(list_of_allArrays_threshold)):
        pattern_length = ComparativeMeasures.find_diffusion(list_of_allArrays_threshold[i], pattern_start_list[i])
        pattern_length = [entry for entry in pattern_length if
                            entry[2] >= min_length_threshold]  # [row, column, diffusionDuration]
        pattern_length_list.append(pattern_length)

    # get descriptives
    for i in range(len(list_of_allArrays_threshold)):
        patterns_perTopic, pattern_lengths_perTopic = ComparativeMeasures.get_nonSimilarity_descriptives(
            diffusionArray_Topics_lp_columns, pattern_length_list[i])
        PatternLength_withinTopic_avg = []
        for j in pattern_lengths_perTopic:
            if j != []:
                PatternLength_withinTopic_avg.append(np.mean(j))

        threshold_array = list_of_allArrays_threshold[i]
        size_threshold_array = np.size(threshold_array)
        sum_threshold_array = np.sum(threshold_array)
        topic_sum_vec = np.sum(threshold_array, axis=0)
        avg_entry_per_topic = np.mean(topic_sum_vec)

        print('\n', list_of_allArrays_names[i])
        print('Number of Diffusion Cycles total: ', len(pattern_length_list[i]))
        print('Average number of Diffusion Cycles per topic: ', (sum(patterns_perTopic) / len(patterns_perTopic)))
        print('Number of diffusion entries total: ', sum_threshold_array, ' of ', size_threshold_array)
        print('Average number of diffusion entries per topic: ', avg_entry_per_topic, ' of ', len(threshold_array))
        print('Average diffusion length: ', np.mean(PatternLength_withinTopic_avg), 'max: ', max(PatternLength_withinTopic_avg),
              'min: ',
              min(PatternLength_withinTopic_avg), 'median: ', np.median(PatternLength_withinTopic_avg), 'mode: ',
              statistics.mode(PatternLength_withinTopic_avg))
    '''