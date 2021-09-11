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

    # test if all diffusion arrays are of column length 330
    list_of_allArrays = [pattern_array_reference_diff, diffusionArray_Topics_lp, diffusionArray_Topics_gm, diffusionArray_Topics_kc, diffusionArray_Topics_l2, diffusion_array_edgeWeight]
    list_of_allArrays_names = ['pattern_array_reference_diff', 'diffusionArray_Topics_lp', 'diffusionArray_Topics_gm', 'diffusionArray_Topics_kc', 'diffusionArray_Topics_l2', 'diffusion_array_edgeWeight']

    list_of_cdArrays = [diffusionArray_Topics_lp, diffusionArray_Topics_gm, diffusionArray_Topics_kc, diffusionArray_Topics_l2]
    list_of_cdArrays_names = ['diffusionArray_Topics_lp', 'diffusionArray_Topics_gm', 'diffusionArray_Topics_kc', 'diffusionArray_Topics_l2']

    from utilities.my_comparative_utils import ComparativeMeasures

    # for comparative example, choose the topics with the higest similarity between the arrays or between intuitive, link and 1 cd measure (to decide on the cd most suitable)
    '''
    ComparativeMeasures.check_dimensions(list_of_allArrays, diffusionArray_Topics_lp_columns)

    list_of_allArrays_threshold, list_of_allArrays_rowNorm = ComparativeMeasures.normalized_and_binarize(list_of_allArrays, threshold=0.01, leeway=True)


    # find diffusion position
    #modArray_dict = {}
    diffusion_pos_list = []
    diffusion_length_list = []
    for i in range(len(list_of_allArrays_threshold)):
        diffusion_pos = ComparativeMeasures.find_recombination(list_of_allArrays_threshold[i])
        diffusion_pos_list.append(diffusion_pos)

    # fine diffusion length
    min_diffusion_threshold = 1
    for i in range(len(list_of_allArrays_threshold)):
        diffusion_length = ComparativeMeasures.find_diffusion(list_of_allArrays_threshold[i], diffusion_pos_list[i])
        diffusion_length = [entry for entry in diffusion_length if entry[2] >= min_diffusion_threshold] # [row, column, diffusionDuration]
        diffusion_length_list.append(diffusion_length)


    # get descriptives
    for i in range(len(list_of_allArrays_threshold)):
        diff_count_per_topic, diff_duration_per_topic = ComparativeMeasures.get_nonSimilarity_descriptives(diffusionArray_Topics_lp_columns, diffusion_length_list[i])
        avg_duration_withinTopic = []
        for j in diff_duration_per_topic:
            if j != []:
                avg_duration_withinTopic.append(np.mean(j))

        threshold_array = list_of_allArrays_threshold[i]
        size_threshold_array = np.size(threshold_array)
        sum_threshold_array = np.sum(threshold_array)
        topic_sum_vec = np.sum(threshold_array, axis=0)
        avg_entry_per_topic = np.mean(topic_sum_vec)

        print('\n', list_of_allArrays_names[i])
        print('Number of Diffusion Cycles total: ', len(diffusion_length_list[i]))
        print('Average number of Diffusion Cycles per topic: ', (sum(diff_count_per_topic) / len(diff_count_per_topic)))
        print('Number of diffusion entries total: ', sum_threshold_array, ' of ', size_threshold_array)
        print('Average number of diffusion entries per topic: ', avg_entry_per_topic, ' of ', len(threshold_array))
        print('Average diffusion length: ', np.mean(avg_duration_withinTopic), 'max: ', max(avg_duration_withinTopic), 'min: ',
              min(avg_duration_withinTopic), 'median: ', np.median(avg_duration_withinTopic), 'mode: ', statistics.mode(avg_duration_withinTopic))


    # cosine focuses more on the similarity in diffusion instead of similarity in diffusion and non-diffusion. This is because of the
    # numinator in the fraction. a match 0-0 match in the lists, does not increase the numinator. it is treated as a mismatch. 0s have no
    # influence on the denominator. Better: 0-0 matches have no impact neither the numerator nor the denominator


    # get similarities between matrices and over all matrix similarity score

    import itertools

    namePair_list, similarityPair_list_cosine, similarityPair_list_manhattan = ComparativeMeasures.get_matrix_similarities_byTopic(list_of_allArrays_names, list_of_allArrays_threshold)

    print(namePair_list)
    print(similarityPair_list_cosine)
    print(similarityPair_list_manhattan)

    matrixSimilarityScore_cosine = 0
    matrixSimilarityScore_manhattan = 0

    if len(similarityPair_list_cosine) != 0:
        matrixSimilarityScore_cosine = sum(similarityPair_list_cosine) / len(similarityPair_list_cosine)

    if len(similarityPair_list_manhattan) != 0:
        matrixSimilarityScore_manhattan = sum(similarityPair_list_manhattan) / len(similarityPair_list_manhattan)


    print(
        matrixSimilarityScore_cosine)  # position 1,9,10,11 are weird. do they correpsond to one array? all realted to gm. only lp + gm seems ok
    # I will probably exclude gm, because 4 out of 5 gm combinations are outliers with similarity scores
    # two orders of magnitude smaller then the rest.
    print(matrixSimilarityScore_manhattan)
    # print(namePair_list)                    # 1,5,9,10,11

    slices_toExclude = [1,5,9,10,11]

    # 4 out of 5 similarities involving the gm approach are very bad, so we exclude them

    similarityPair_list_cosine_withoutGM = list(np.delete(similarityPair_list_cosine, slices_toExclude, axis=0))
    similarityPair_list_manhattan_withoutGM = list(np.delete(similarityPair_list_manhattan, slices_toExclude, axis=0))

    matrixSimilarityScore_cosine_withoutGM = 0
    matrixSimilarityScore_manhattan_withoutGM = 0

    if len(similarityPair_list_cosine_withoutGM) != 0:
        matrixSimilarityScore_cosine_withoutGM = sum(similarityPair_list_cosine_withoutGM) / len(similarityPair_list_cosine_withoutGM)

    if len(similarityPair_list_manhattan_withoutGM) != 0:
        matrixSimilarityScore_manhattan_withoutGM = sum(similarityPair_list_manhattan_withoutGM) / len(similarityPair_list_manhattan_withoutGM)

    print('\n', matrixSimilarityScore_cosine_withoutGM)
    print(matrixSimilarityScore_manhattan_withoutGM)



###---------------------

    # get similarities between vectors of the same topic. max, min, avg, mode, media, distribution

    simScores_withinTopic_list_cosine_avg, simScores_withinTopic_list_manhattan_avg = ComparativeMeasures.get_topic_similarity(list_of_allArrays_names, list_of_allArrays_threshold, slices_toExclude)

        # calculate the following only with values that are not -9999
    # there are a lot topics falling through. check if this filter is correct
    # also check if there are really topics with similarity of 1 across all pairs
    simScores_withinTopic_list_cosine_avg_clean =  [x for x in simScores_withinTopic_list_cosine_avg if x != -9999]
    simScores_withinTopic_list_manhattan_avg_clean = [x for x in simScores_withinTopic_list_manhattan_avg if x != -9999]



    most_similarTopic_value_cosine = max(simScores_withinTopic_list_cosine_avg_clean)
    most_similarTopic_value_manhattan = max(simScores_withinTopic_list_manhattan_avg_clean)

    most_similarTopic_pos_cosine = np.where(simScores_withinTopic_list_cosine_avg == most_similarTopic_value_cosine)
    most_similarTopic_pos_manhattan = np.where(simScores_withinTopic_list_manhattan_avg == most_similarTopic_value_manhattan)

    least_similarTopic_value_cosine = min(simScores_withinTopic_list_cosine_avg_clean)
    least_similarTopic_value_manhattan = min(simScores_withinTopic_list_manhattan_avg_clean)

    least_similarTopic_pos_cosine = np.where(simScores_withinTopic_list_cosine_avg == least_similarTopic_value_cosine)
    least_similarTopic_pos_manhattan = np.where(simScores_withinTopic_list_manhattan_avg == least_similarTopic_value_manhattan)

    avg_similarTopic_cosine = sum(simScores_withinTopic_list_cosine_avg_clean) / len(simScores_withinTopic_list_cosine_avg_clean)
    avg_similarTopic_manhattan = sum(simScores_withinTopic_list_manhattan_avg_clean) / len(simScores_withinTopic_list_manhattan_avg_clean)


    print('\n Topic Cosine Similarities between approaches:')
    print('Highest Similarity: ', most_similarTopic_value_cosine)
    print('Highest Similarity Topic Id: ', most_similarTopic_pos_cosine[0])
    print('Lowest Similarity: ', least_similarTopic_value_cosine)
    print('Lowest Similarity Topic Id: ', least_similarTopic_pos_cosine[0])
    print('Average Similarity between all topics: ', avg_similarTopic_cosine)

    print('\n Topic Manhattan Similarities between approaches:')
    print('Highest Similarity: ', most_similarTopic_value_manhattan)
    print('Highest Similarity Topic Id: ', most_similarTopic_pos_manhattan[0])
    print('Lowest Similarity: ', least_similarTopic_value_manhattan)
    print('Lowest Similarity Topic Id: ', least_similarTopic_pos_manhattan[0])
    print('Average Similarity between all topics: ', avg_similarTopic_manhattan)

    #print(len(list_of_allArrays_threshold[0]))
    #print(len(list_of_allArrays_threshold))

    vialization_higestTopicSim = np.zeros((len(list_of_allArrays_threshold[0]), len(list_of_allArrays_threshold)), dtype=int)
    #print(np.shape(vialization_higestTopicSim))
    for i in range(len(list_of_allArrays_threshold)):
        #print(list_of_allArrays_threshold[i][:,23])
        #print(vialization_higestTopicSim[:,i])
        vialization_higestTopicSim[:,i] = list_of_allArrays_threshold[i][:,23]

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


    '''


###----------- Recombination ----------------
#--- Comparing Recombination arrays ---#



    column_lists = [columns_reference_reco, recombinationArray_Topics_lp_columns, recombinationArray_Topics_gm_columns, recombinationArray_Topics_kc_columns, recombinationArray_Topics_l2_columns, columns_recom_edgeWeight]
    columns_list_names = ['columns_reference_reco', 'recombinationArray_Topics_lp_columns', 'recombinationArray_Topics_gm_columns', 'recombinationArray_Topics_kc_columns', 'recombinationArray_Topics_l2_columns', 'columns_recom_edgeWeight']

    print(len(recombinationDiffusion_edgeWeight.T))
    print(len(columns_recom_edgeWeight))
    print(len(np.unique(columns_recom_edgeWeight, axis=0)))

    recoArrays_list = [pattern_array_reference_reco, recombinationArray_Topics_lp, recombinationArray_Topics_gm,
                       recombinationArray_Topics_kc, recombinationArray_Topics_l2, recombinationDiffusion_edgeWeight]

    for array in recoArrays_list:
        columSum_vec = np.sum(array, axis= 0)
        print(np.where(columSum_vec == 0))
        #todo: why is this not empty for lp????

    ComparativeMeasures.extend_cD_recombinationDiffuion(recoArrays_list, 12, 1, 5)

    #recoArrays_list_mod = [recoArrays_list[0], extend_cD_arrays[:], recoArrays_list[-1]]

    recoArrays_threshold_list, recoArrays_rowNorm_list = ComparativeMeasures.normalized_and_binarize(recoArrays_list, threshold=0.01, leeway=True)

    #print(sum(sum(recoArrays_threshold_list[0])))
    #print(sum(sum(recoArrays_threshold_list[1])))

    extended_threshold_arrays = ComparativeMeasures.extend_recombination_columns(column_lists, recoArrays_threshold_list)
    #extended_threshold_arrays = ComparativeMeasures.extend_recombination_columns(column_lists, recoArrays_list)

    # delete collumns if they are 0 in all matrices

    columSum_vec_list = []
    for array in extended_threshold_arrays:
        columSum_vec = np.sum(array, axis=0)
        columSum_vec_list.append(columSum_vec)
        print(len(columSum_vec))

    columSum_vec_summed = np.sum(columSum_vec_list, axis=0)
    topicExclusionPosition = np.where(columSum_vec_summed == 0)
    print(topicExclusionPosition)

    resized_threshold_arrays = []
    for array in extended_threshold_arrays:
        resized_threshold_arrays.append(np.delete(array, topicExclusionPosition, 1))

    for array in resized_threshold_arrays:
        print(np.shape(array))














    '''
    # find diffusion position
    # modArray_dict = {}
    recomb_pos_list = []
    recomb_length_list = []
    for i in range(len(recoArrays_list)):
        recomb_pos = ComparativeMeasures.find_recombination(recoArrays_list[i])
        recomb_length_list.append(recomb_pos)

    # fine diffusion length
    min_diffusion_threshold = 1
    for i in range(len(list_of_allArrays_threshold)):
        diffusion_length = ComparativeMeasures.find_diffusion(list_of_allArrays_threshold[i], diffusion_pos_list[i])
        diffusion_length = [entry for entry in diffusion_length if
                            entry[2] >= min_diffusion_threshold]  # [row, column, diffusionDuration]
        diffusion_length_list.append(diffusion_length)

    # get descriptives
    for i in range(len(list_of_allArrays_threshold)):
        diff_count_per_topic, diff_duration_per_topic = ComparativeMeasures.get_nonSimilarity_descriptives(
            diffusionArray_Topics_lp_columns, diffusion_length_list[i])
        avg_duration_withinTopic = []
        for j in diff_duration_per_topic:
            if j != []:
                avg_duration_withinTopic.append(np.mean(j))

        threshold_array = list_of_allArrays_threshold[i]
        size_threshold_array = np.size(threshold_array)
        sum_threshold_array = np.sum(threshold_array)
        topic_sum_vec = np.sum(threshold_array, axis=0)
        avg_entry_per_topic = np.mean(topic_sum_vec)

        print('\n', list_of_allArrays_names[i])
        print('Number of Diffusion Cycles total: ', len(diffusion_length_list[i]))
        print('Average number of Diffusion Cycles per topic: ', (sum(diff_count_per_topic) / len(diff_count_per_topic)))
        print('Number of diffusion entries total: ', sum_threshold_array, ' of ', size_threshold_array)
        print('Average number of diffusion entries per topic: ', avg_entry_per_topic, ' of ', len(threshold_array))
        print('Average diffusion length: ', np.mean(avg_duration_withinTopic), 'max: ', max(avg_duration_withinTopic),
              'min: ',
              min(avg_duration_withinTopic), 'median: ', np.median(avg_duration_withinTopic), 'mode: ',
              statistics.mode(avg_duration_withinTopic))
    '''