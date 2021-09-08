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
    list_of_allArraysNames = ['pattern_array_reference_diff', 'diffusionArray_Topics_lp', 'diffusionArray_Topics_gm', 'diffusionArray_Topics_kc', 'diffusionArray_Topics_l2', 'diffusion_array_edgeWeight']

    list_of_cdArrays = [diffusionArray_Topics_lp, diffusionArray_Topics_gm, diffusionArray_Topics_kc, diffusionArray_Topics_l2]
    list_of_cdArraysNames = ['diffusionArray_Topics_lp', 'diffusionArray_Topics_gm', 'diffusionArray_Topics_kc', 'diffusionArray_Topics_l2']

    from utilities.my_comparative_utils import ComparativeMeasures

    # for comparative example, choose the topics with the higest similarity between the arrays or between intuitive, link and 1 cd measure (to decide on the cd most suitable)

    modArray_dict = {}
    diffusion_dict = {}
    for i in range(len(list_of_allArrays)):
        if len(list_of_allArrays[i].T) != len((list(diffusionArray_Topics_lp_columns))):
            raise Exception("Diffusion arrays vary in their columns")

        array_threshold, array_frac = ComparativeMeasures.modify_arrays(list_of_allArrays[i], 0.01)
        array_threshold = ComparativeMeasures.introcude_leeway(array_threshold, np.array([1, 0, 1]), 1)
        # if there is 101 this means that there was one year without a occurrence. 1001 is one year and one month ...

        diffusion_pos = ComparativeMeasures.find_recombination(array_threshold)
        diffusion_length = ComparativeMeasures.find_diffusion(array_threshold, diffusion_pos)

        name = list_of_allArraysNames[i] + '_mod'

        modArray_dict[name] = array_threshold

        min_diffusion_threshold = 1
        diffusion_length = [entry for entry in diffusion_length if entry[2] >= min_diffusion_threshold]
        diffusion_dict[name] = diffusion_length     # [row, column, diffusionDuration]


    #descriptives_diffusion = {}
    for approach, diff_dict in diffusion_dict.items():
        diff_count_per_topic = []
        avgDiff_duration_per_topic = []
        for topic in range(len(diffusionArray_Topics_lp_columns)):
            diff_count = 0
            diff_duration = []
            for entry in diff_dict:
                if entry[1] == topic:
                    diff_count = diff_count+1
                    diff_duration.append(entry[2])
                    #print(diff_count)
                    #print(diff_duration)
            #print(diff_count)
            #print(diff_duration)
            diff_count_per_topic.append(diff_count)
            if diff_duration != []:
                avgDiff_duration_per_topic.append(np.mean(diff_duration)) #, np.median(diff_duration), statistics.mode(diff_duration), max(diff_duration), min(diff_duration))

        threshold_array = modArray_dict[approach]
        size_threshold_array = np.size(threshold_array)
        sum_threshold_array = np.sum(threshold_array)
        topic_sum_vec = np.sum(threshold_array, axis=0)
        avg_entry_per_topic = np.mean(topic_sum_vec)

        print('\n', approach)
        print('Number of Diffusion Cycles total: ', len(diff_dict))
        print('Average number of Diffusion Cycles per topic: ', (sum(diff_count_per_topic) / len(diff_count_per_topic)))
        print('Number of diffusion entries total: ', sum_threshold_array, ' of ', size_threshold_array)
        print('Average number of diffusion entries per topic: ', avg_entry_per_topic, ' of ', len(threshold_array))
        print('Average diffusion length: ', np.mean(avgDiff_duration_per_topic), 'max: ', max(avgDiff_duration_per_topic), 'min: ',
              min(avgDiff_duration_per_topic), 'median: ', np.median(avgDiff_duration_per_topic), 'mode: ', statistics.mode(avgDiff_duration_per_topic))
        '''
        print('\n', diff_dict)
        print(diff_count_per_topic)
        print(size_threshold_array)
        print(sum_threshold_array)
        print(avgDiff_duration_per_topic)
        break
        '''

        #descriptives_diffusion[approach] = [('number of diffusionCycles', len(diff_dict)),(),(),()]

    print(modArray_dict['pattern_array_reference_diff_mod'][:,50])
    print(modArray_dict['diffusionArray_Topics_lp_mod'][:,50])


    def manhattan_sim_mod(x, y):
        return 1 - sum(abs(a - b) for a, b in zip(x, y)) / len(x)


    from numpy import dot
    from numpy.linalg import norm


    def cosine_sim_mod(List1, List2):
        if sum(List1) == 0 or sum(List2) == 0:
            result = 0
        else:
            result = dot(List1, List2) / (norm(List1) * norm(List2))

        return (result)

    # cosine focuses more on the similarity in diffusion instead of similarity in diffusion and non-diffusion. This is because of the
    # numinator in the fraction. a match 0-0 match in the lists, does not increase the numinator. it is treated as a mismatch. 0s have no
    # influence on the denominator. Better: 0-0 matches have no impact neither the numerator nor the denominator


    result = cosine_sim_mod(modArray_dict['pattern_array_reference_diff_mod'][:,50], modArray_dict['diffusionArray_Topics_lp_mod'][:,50])
    result2 = manhattan_sim_mod(modArray_dict['pattern_array_reference_diff_mod'][:,50], modArray_dict['diffusionArray_Topics_lp_mod'][:,50])
    print(result)
    print(result2)

    # get similarities between matrizes and over all matrix similarity score

    print(cosine_sim_mod([0,0,1],[0,1,1]))
    print(manhattan_sim_mod([0,0,1],[0,1,1]))


    import itertools

    name_list = []
    array_list = []
    namePair_list = []
    arrayPair_list = []
    for name, array in modArray_dict.items():
        name_list.append(name) # , array))
        array_list.append(array)

    namePair_list.append(list(itertools.combinations(name_list, r=2)))
    namePair_list = namePair_list[0]
    arrayPair_list.append(list(itertools.combinations(array_list, r=2)))
    arrayPair_list = arrayPair_list[0]

    similarityPair_list_cosine = []
    similarityPair_list_manhattan = []

    for matrixPair in arrayPair_list:
        patternArray1 = matrixPair[0]
        print(np.shape(matrixPair))
        patternArray2 = matrixPair[1]

        cosine_list = []
        manhattan_list = []
        for column_id in range(len(patternArray1.T)):

            print(max(patternArray1[:,column_id]))
            print(max(patternArray2[:,column_id]))

            print(min(patternArray1[:, column_id]))
            print(min(patternArray2[:, column_id]))

            print(len(patternArray1[:, column_id]))
            print(len(patternArray2[:, column_id]))

            cosine = 1 - cosine_sim_mod(patternArray1[:,column_id],patternArray2[:,column_id])
            manhattan = manhattan_sim_mod(patternArray1[:,column_id],patternArray2[:,column_id])

            cosine_list.append(cosine)
            manhattan_list.append(manhattan)

        cosine_avg = sum(cosine_list) / len(cosine_list)
        manhattan_avg = sum(manhattan_list) / len(manhattan_list)

        similarityPair_list_cosine.append(cosine_avg)
        similarityPair_list_manhattan.append(manhattan_avg)

    matrixSimilarityScore_cosine = sum(similarityPair_list_cosine) / len(similarityPair_list_cosine)
    matrixSimilarityScore_manhattan = sum(similarityPair_list_manhattan) / len(similarityPair_list_manhattan)

    # get similarities between vectors of the same topic. max, min, avg, mode, media, distribution

    topicVectore_simScore_list = []

    for column_id in range(len(array_list[0].T)):

        simScores_withinTopic_cosine = []
        simScores_withinTopic_manhattan = []
        for matrixPair in arrayPair_list:
            patternArray1 = matrixPair[0]
            patternArray2 = matrixPair[1]

            cosine = 1 - cosine_sim_mod(patternArray1[:, column_id], patternArray2[:, column_id])
            manhattan = manhattan_sim_mod(patternArray1[:, column_id], patternArray2[:, column_id])

            simScores_withinTopic_cosine.append(cosine)
            simScores_withinTopic_manhattan.append(manhattan)

        simScores_withinTopic_cosine_avg = sum(simScores_withinTopic_cosine) / len(simScores_withinTopic_cosine)
        simScores_withinTopic_manhattan_avg = sum(simScores_withinTopic_manhattan) / len(simScores_withinTopic_manhattan)

    most_similarTopic_value_cosine = max(simScores_withinTopic_cosine)
    most_similarTopic_value_manhattan = max(simScores_withinTopic_manhattan)

    most_similarTopic_pos_cosine = np.where(simScores_withinTopic_cosine == max(simScores_withinTopic_cosine))
    most_similarTopic_pos_manhattan = np.where(simScores_withinTopic_manhattan == max(simScores_withinTopic_manhattan))

    least_similarTopic_value_cosine = min(simScores_withinTopic_cosine)
    least_similarTopic_value_manhattan = min(simScores_withinTopic_manhattan)

    least_similarTopic_pos_cosine = np.where(simScores_withinTopic_cosine == min(simScores_withinTopic_cosine))
    least_similarTopic_pos_manhattan = np.where(simScores_withinTopic_manhattan == min(simScores_withinTopic_manhattan))

    avg_similarTopic_cosine = sum(simScores_withinTopic_cosine) / len(simScores_withinTopic_cosine)
    avg_similarTopic_manhattan = sum(simScores_withinTopic_manhattan) / len(simScores_withinTopic_manhattan)


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




#--- Comparing Recombination arrays ---#

    '''
    pattern_array_reference_reco
    recombinationArray_Topics_lp
    recombinationArray_Topics_gm
    recombinationArray_Topics_kc
    recombinationArray_Topics_l2
    recombinationDiffusion_edgeWeight
    
    columns_reference_reco
    recombinationArray_Topics_lp_columns
    recombinationArray_Topics_gm_columns
    recombinationArray_Topics_kc_columns
    recombinationArray_Topics_l2_columns
    columns_recom_edgeWeight
        
    '''

