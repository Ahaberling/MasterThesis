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

    def modify_arrays(array, threshold):

        row_sum = array.sum(axis=1)
        array_frac = array / row_sum[:, np.newaxis]

        nonzeros = [row[np.nonzero(row)] for row in array_frac]
        helper = np.mean(nonzeros[0])
        helper4 = np.median(nonzeros[0])
        helper2 = np.quantile(nonzeros[0], 0.5)
        helper3 = np.quantile(nonzeros[0], 0.25)
        helper3 = np.quantile(nonzeros[0], 0.1)
        quantil_vec = [np.quantile(row, 0.1) for row in nonzeros]
        quantil_vec_test = [np.quantile(row, 0.0000001) for row in nonzeros]

        threshold_array = np.zeros((np.shape(array)[0],np.shape(array)[1]))
        threshold_array_test = np.zeros((np.shape(array)[0],np.shape(array)[1]))
        diff_list = []

        for i in range(len(threshold_array)):
            threshold_array[i,:] = np.where(array_frac[i,:] < quantil_vec[i], 0, 1)
            threshold_array_test[i,:] = np.where(array_frac[i,:] < quantil_vec_test[i], 0, 1)
            c = 0
            for j in range(len(threshold_array.T)):
                if threshold_array[i,j] !=  threshold_array_test[i,j]:
                    c = c +1
            diff_list.append(c)
        print(diff_list)
        print(sum(diff_list))

        array_threshold = np.where(array_frac < threshold, 0, 1)

        return array_threshold, array_frac

    modArray_dict = {}
    for i in range(len(list_of_allArrays)):
        if len(list_of_allArrays[i].T) != len((list(diffusionArray_Topics_lp_columns))):
            raise Exception("Diffusion arrays vary in their columns")
        a = list_of_allArrays[i]

        array_threshold, array_frac = modify_arrays(list_of_allArrays[i], 0.01)

        name = list_of_allArraysNames[i] + '_mod'
        modArray_dict[name] = array_threshold

    print(modArray_dict.keys())



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