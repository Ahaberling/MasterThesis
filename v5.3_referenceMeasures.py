
if __name__ == '__main__':

    # --- Import Libraries ---#
    print('\n#--- Import Libraries ---#\n')

    import pandas as pd
    import numpy as np
    import pickle as pk

    import os


    # --- Initialization --#
    print('\n#--- Initialization ---#\n')

    os.chdir('D:/Universitaet Mannheim/MMDS 7. Semester/Master Thesis/Outline/Data/Cleaning Robots')

    patent_topicDist = pd.read_csv('patent_topicDist_mallet.csv', quotechar='"', skipinitialspace=True)
    patent_IPC = pd.read_csv('cleaning_robot_EP_patents_IPC.csv', quotechar='"', skipinitialspace=True)

    patent_topicDist = patent_topicDist.to_numpy()
    patent_IPC = patent_IPC.to_numpy()

    patent_lda_ipc = patent_topicDist

    with open('slidingWindow_dict', 'rb') as handle:
        slidingWindow_dict = pk.load(handle)

    # --- Preprocessing for pattern arrays (IPC's, topics) x (singular, pair, tripple)  --#
    print('\n#--- Preprocessing for pattern arrays (IPC\'s, topics) x (singluar, pair, tripple) ---#\n')

    ### Create dictionaries containing ALL ipc/topic singles/pairs/triples for a window ###

    from utilities.my_measure_utils import ReferenceMeasures

    knowledgeComponent_dict = ReferenceMeasures.extract_knowledgeComponent_per_window(slidingWindow_dict, kC='topic', unit=2)


    # --- Constructing pattern arrays (IPC\'s, topics) x (singular, pair, tripple) ---#
    print('\n#--- Constructing pattern arrays (IPC\'s, topics) x (singular, pair, tripple) ---#\n')

    pattern_array_reference, columns_reference  = ReferenceMeasures.create_pattern_array(knowledgeComponent_dict)


    # --- Normalized plain pattern arrays (IPC\'s, topics) x (singular, pair, triple)  ---#
    print('\n#--- Normalized plain pattern arrays (IPC\'s, topics) x (singular, pair, triple)  ---#\n')


    row_sum = pattern_array_reference.sum(axis=1)
    pattern_array_norm_reference = pattern_array_reference / row_sum[:, np.newaxis]


    # --- Binarized plain pattern arrays (IPC\'s, topics) x (singular, pair, triple)  ---#
    print('\n#--- Binarized plain pattern arrays (IPC\'s, topics) x (singular, pair, triple)  ---#\n')

    ### IPC singles ###

    threshold = 0.005

    pattern_array_thresh_reference = np.where(pattern_array_norm_reference < threshold, 0, 1)  # arbitrary threshold of 0.01


    # --- Measuring recombination, diffusion of recombination and diffusion of singular ipc's/topics  ---#
    print(
        '\n#--- Measuring recombination, diffusion of recombination and diffusion of singular ipc\'s/topics  ---#\n')

    # The resulting data structure looks like [[window_position, ipc_position, duration],[w,i,d],...]
    # For further insights the w and i coordinates can be used to review the unnormalized and unbinarized pattern array.

    ### IPC singles ###
    print('\n\t### IPC singles ###')
    print('\t Locating diffusion\n')

    # finding diffusion patterns #

    recombinationPos_reference = ReferenceMeasures.find_recombination(pattern_array_thresh_reference)



    # print(diff_pos)             # [[2171, 2], [2206, 2], [2213, 2], [2220, 2], [3074, 2], [3081, 2], [3088, 2], [2171, 5], [2206, 5], [2213, 5]
    #  [2220, 5], [3074, 5], [3081, 5], [3088, 5], [3383, 5], [3431, 5], [3445, 5], [3635, 5], [3747, 5], ...
    # print(len(diff_pos))        # 3095



    # counting diffusion #
    print('\t Measuring diffusion\n')

    Diffusion_reference = ReferenceMeasures.find_diffusion(pattern_array_thresh_reference, recombinationPos_reference)


    ### Identifying where sequences occur ###

    sequence = np.array([1, 0, 1])

    sequencePos = ReferenceMeasures.search_sequence(pattern_array_thresh_reference, sequence)

    print(sequencePos)

    ### Replacing sequences 101 with 111 ###

    impute_value = 1

    pattern_array_thresh_leeway_reference = ReferenceMeasures.introcude_leeway(pattern_array_thresh_reference, sequence, impute_value)


    sequencePos = ReferenceMeasures.search_sequence(pattern_array_thresh_leeway_reference, sequence)

    print(sequencePos)

        # break

    # todo problem 1: imputing sequences only works for 101 case, not for 100001, and so on

