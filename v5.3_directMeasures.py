
if __name__ == '__main__':

    # --- Import Libraries ---#
    print('\n#--- Import Libraries ---#\n')

    import pandas as pd
    import numpy as np
    import pickle as pk

    import os
    import statistics

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

    import matplotlib.pyplot as plt
    import seaborn as sns
    '''
    flights_long = sns.load_dataset("flights")
    flights = flights_long.pivot("month", "year", "passengers")
    print(flights)
    print(np.shape(flights)) #(12, 12)

    # Draw a heatmap with the numeric values in each cell
    #f, ax = plt.subplots(figsize=(9, 6))
    f, ax = plt.subplots()
    sns.heatmap(flights, annot=True, fmt="d", linewidths=.5, ax=ax)
    plt.show()
    '''


    #sns.set_theme()
    #uniform_data = np.random.rand(10, 12)
    #print(uniform_data)
    #print(np.shape(uniform_data))   # (10, 12)
    #ax = sns.heatmap(uniform_data)



    from utilities.my_measure_utils import ReferenceMeasures

    knowledgeComponent_dict_diff = ReferenceMeasures.extract_knowledgeComponent_per_window(slidingWindow_dict, kC='topic', unit=1)
    knowledgeComponent_dict_reco = ReferenceMeasures.extract_knowledgeComponent_per_window(slidingWindow_dict, kC='topic', unit=2)

    all_components = []
    component_perWindow = []
    component_perWindow_unique = []
    for i in knowledgeComponent_dict_diff.values():
        all_components.append(i)
        component_perWindow.append(len(i))
        component_perWindow_unique.append(len(np.unique(i)))
    all_components = [item for sublist in all_components for item in sublist]
    print('Number of retrieved components: ', len(all_components))
    print('Number of unique retrieved components: ', len(np.unique(all_components)))

    print('Average Number retrieved components per window: ', np.mean(component_perWindow))
    print('Median Number retrieved components per window: ', np.median(component_perWindow))
    print('Mode Number retrieved components per window: ', statistics.mode(component_perWindow))
    print('Max Number retrieved components per window: ', max(component_perWindow))
    print('Min Number retrieved components per window: ', min(component_perWindow))

    print('Average Number unique retrieved components per window: ', np.mean(component_perWindow_unique))
    print('Median Number unique retrieved components per window: ', np.median(component_perWindow_unique))
    print('Mode Number unique retrieved components per window: ', statistics.mode(component_perWindow_unique))
    print('Max Number unique retrieved components per window: ', max(component_perWindow_unique))
    print('Min Number unique retrieved components per window: ', min(component_perWindow_unique))


    all_components = []
    component_perWindow = []
    component_perWindow_unique = []
    for i in knowledgeComponent_dict_reco.values():
        all_components.append(i)
        component_perWindow.append(len(i))
        component_perWindow_unique.append(len(np.unique(i, axis=0)))
    all_components = [item for sublist in all_components for item in sublist]
    print('Number of retrieved component combinations: ', len(all_components))
    print('Number of unique retrieved component combinations: ', len(np.unique(all_components, axis=0)))

    print('Average Number retrieved components per window: ', np.mean(component_perWindow))
    print('Median Number retrieved components per window: ', np.median(component_perWindow))
    print('Mode Number retrieved components per window: ', statistics.mode(component_perWindow))
    print('Max Number retrieved components per window: ', max(component_perWindow))
    print('Min Number retrieved components per window: ', min(component_perWindow))

    print('Average Number unique retrieved components per window: ', np.mean(component_perWindow_unique))
    print('Median Number unique retrieved components per window: ', np.median(component_perWindow_unique))
    print('Mode Number unique retrieved components per window: ', statistics.mode(component_perWindow_unique))
    print('Max Number unique retrieved components per window: ', max(component_perWindow_unique))
    print('Min Number unique retrieved components per window: ', min(component_perWindow_unique))

    # --- Constructing pattern arrays (IPC\'s, topics) x (singular, pair, tripple) ---#
    print('\n#--- Constructing pattern arrays (IPC\'s, topics) x (singular, pair, tripple) ---#\n')

    pattern_array_reference_diff, columns_reference_diff = ReferenceMeasures.create_pattern_array(knowledgeComponent_dict_diff)

    f, ax = plt.subplots()
    sns.heatmap(pattern_array_reference_diff[0:80,20:30], cbar_kws={'label': 'Component Count in Window'}) #, cmap="YlGnBu") #, annot=True, fmt="d", linewidths=.5, ax=ax)
    plt.yticks(range(0,80,10))
    ax.set_xticklabels(range(20,30))
    ax.set_yticklabels(range(0,80,10))
    plt.xlabel("Knowledge Component ID")
    plt.ylabel("Sliding Window ID ")
    #plt.show()
    plt.close()

    pattern_array_reference_reco, columns_reference_reco = ReferenceMeasures.create_pattern_array(knowledgeComponent_dict_reco)

    f, ax = plt.subplots()
    sns.heatmap(pattern_array_reference_reco[100:180,635:645], cbar_kws={'label': 'Component Combination Count in Window'}) #, cmap="YlGnBu") #, annot=True, fmt="d", linewidths=.5, ax=ax)
    plt.yticks(range(0,80,10))
    ax.set_xticklabels(range(635,645))
    ax.set_yticklabels(range(100,180,10))
    plt.xlabel("Knowledge Component Combination ID")
    plt.ylabel("Sliding Window ID ")
    plt.show()
    plt.close()

    print('Dimensions of the SCM: ', np.shape(pattern_array_reference_diff))
    print('Number of cells SCM: ', np.size(pattern_array_reference_diff))
    print('Number of non zero cells SCM: ', np.count_nonzero(pattern_array_reference_diff))
    print('Number of zero cells SCM: ', np.size(pattern_array_reference_diff) - np.count_nonzero(pattern_array_reference_diff))
    print('Sum of cells SCM: ', sum(sum(pattern_array_reference_diff)))




    print('Dimensions of the CCM: ', np.shape(pattern_array_reference_reco))
    print('Number of cells CCM: ', np.size(pattern_array_reference_reco))
    print('Number of non zero cells CCM: ', np.count_nonzero(pattern_array_reference_reco))
    print('Number of zero cells CCM: ', np.size(pattern_array_reference_reco) - np.count_nonzero(pattern_array_reference_reco))
    print('Sum of cells CCM: ', sum(sum(pattern_array_reference_reco)))


    filename = 'pattern_array_reference_diff'
    outfile = open(filename, 'wb')
    pk.dump(pattern_array_reference_diff, outfile)
    outfile.close()

    filename = 'columns_reference_diff'
    outfile = open(filename, 'wb')
    pk.dump(columns_reference_diff, outfile)
    outfile.close()

    filename = 'pattern_array_reference_reco'
    outfile = open(filename, 'wb')
    pk.dump(pattern_array_reference_reco, outfile)
    outfile.close()

    filename = 'columns_reference_reco'
    outfile = open(filename, 'wb')
    pk.dump(columns_reference_reco, outfile)
    outfile.close()

    # --- Normalized plain pattern arrays (IPC\'s, topics) x (singular, pair, triple)  ---#
    print('\n#--- Normalized plain pattern arrays (IPC\'s, topics) x (singular, pair, triple)  ---#\n')

    pattern_array_reference = pattern_array_reference_reco

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



