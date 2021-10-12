
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



    from utilities_old.my_measure_utils import ReferenceMeasures

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

    #print(np.sum(pattern_array_reference_diff))
    #print(np.count_nonzero(pattern_array_reference_diff))
    #print(np.sum(pattern_array_reference_diff[np.nonzero(pattern_array_reference_diff)]) / np.count_nonzero(pattern_array_reference_diff))

    from utilities_old.my_measure_utils import Misc

    diffusionPatternPos_SCM = Misc.find_diffusionPatterns(pattern_array_reference_diff)
    diffusionPatternPos_SCM, diff_sequence_list_SCM, irrelevant = Misc.find_diffusionSequenceAndLength(diffusionPatternPos_SCM, pattern_array_reference_diff)
    diffusionPatternPos_SCM = Misc.find_diffusionStepsAndPatternPerDiffusion(diffusionPatternPos_SCM, diff_sequence_list_SCM)
    # diff_pos = [ row, column, diffLength, diffSteps, patentsInDiff ]
    diffusionPatternPos_SCM = np.array(diffusionPatternPos_SCM)



    diffPatternLength_perTopic = []
    diffPatternSteps_perTopic = []
    diffPatternPatents_perTopic = []
    diff_perTopic = []


    for topic in range(np.max(diffusionPatternPos_SCM[:,1])):
        diffPatternLength_helper = []
        diffPatternSteps_helper = []
        diffPatternPatents_helper = []
        diff_perTopic_helper = []

        for diffPattern in diffusionPatternPos_SCM:
            if diffPattern[1] == topic:
                diffPatternLength_helper.append(diffPattern[2])
                diffPatternSteps_helper.append(diffPattern[3])
                diffPatternPatents_helper.append(diffPattern[4])
                diff_perTopic_helper.append(1)

        diffPatternLength_perTopic.append(np.mean(diffPatternLength_helper))
        diffPatternSteps_perTopic.append(np.mean(diffPatternSteps_helper))
        diffPatternPatents_perTopic.append(np.mean(diffPatternPatents_helper))
        diff_perTopic.append(sum(diff_perTopic_helper))

    print('Number of diffusion cycles / patterns in the scm: ', len(diffusionPatternPos_SCM))
    print('Average diffusion pattern length: ', np.mean(diffusionPatternPos_SCM[:, 2]))
    print('Average diffusion steps: ', np.mean(diffusionPatternPos_SCM[:, 3]))
    print('Average diffusion patents: ', np.mean(diffusionPatternPos_SCM[:, 4]))


    print('Averaged Average number of cycles/patterns per topic: ', np.mean(diff_perTopic))
    print('Median number of Average cycles/patterns per topic: ', np.median(diff_perTopic))
    print('Mode number of Average cycles/patterns per topic: ', statistics.mode(diff_perTopic))
    print('Max number of Average cycles/patterns per topic: ', max(diff_perTopic))
    print('Min number of Average cycles/patterns per topic: ', min(diff_perTopic))

    print('Average of the average topic diffusion steps: ', np.mean(diffPatternSteps_perTopic))
    print('Median of the average topic diffusion steps: ', np.median(diffPatternSteps_perTopic))
    print('Mode of the average topic diffusion steps: ', statistics.mode(diffPatternSteps_perTopic))
    print('Max of the average topic diffusion steps: ', max(diffPatternSteps_perTopic))
    print('Min of the average topic diffusion steps: ', min(diffPatternSteps_perTopic))

    print('Average of the average topic diffusion length: ', np.mean(diffPatternLength_perTopic))
    print('Median of the average topic diffusion length: ', np.median(diffPatternLength_perTopic))
    print('Mode of the average topic diffusion length: ', statistics.mode(diffPatternLength_perTopic))
    print('Max of the average topic diffusion length: ', max(diffPatternLength_perTopic))
    print('Min of the average topic diffusion length: ', min(diffPatternLength_perTopic))

    print('Average of the average topic diffusion patents: ', np.mean(diffPatternPatents_perTopic))
    print('Median of the average topic diffusion patents: ', np.median(diffPatternPatents_perTopic))
    print('Mode of the average topic diffusion patents: ', statistics.mode(diffPatternPatents_perTopic))
    print('Max of the average topic diffusion patents: ', max(diffPatternPatents_perTopic))
    print('Min of the average topic diffusion patents: ', min(diffPatternPatents_perTopic))

    # ---
    topicWithMostDiffSteps = []
    topicWithLeastDiffSteps = []
    for i in range(len(diffPatternSteps_perTopic)):
        #print(diffPatternSteps_perTopic[i])
        if diffPatternSteps_perTopic[i] == max(diffPatternSteps_perTopic):
            topicWithMostDiffSteps.append(i)
        if diffPatternSteps_perTopic[i] == min(diffPatternSteps_perTopic):
            topicWithLeastDiffSteps.append(i)

    topicWithMostDiffSteps = np.unique(topicWithMostDiffSteps)
    topicWithLeastDiffSteps = np.unique(topicWithLeastDiffSteps)

    print('topic with most diff steps: ', topicWithMostDiffSteps)
    print('topic with elast diff steps: ', topicWithLeastDiffSteps)

    # ---
    topicWithMostDiffpats = []
    topicWithLeastDiffpats = []
    for i in range(len(diffPatternPatents_perTopic)):
        #print(diffPatternSteps_perTopic[i])
        if diffPatternPatents_perTopic[i] == max(diffPatternPatents_perTopic):
            topicWithMostDiffpats.append(i)
        if diffPatternPatents_perTopic[i] == min(diffPatternPatents_perTopic):
            topicWithLeastDiffpats.append(i)

    topicWithMostDiffpats = np.unique(topicWithMostDiffpats)
    topicWithLeastDiffpats = np.unique(topicWithLeastDiffpats)

    print('topic with most diff patents: ', topicWithMostDiffpats)
    print('topic with elast diff patents: ', topicWithLeastDiffpats)



    #print(diffusionPatternPos_SCM[0])
    #print(diffusionPatternPos_SCM[:,2])






    '''
    for diff_seq in diff_sequence_list:
        
        
        diff_seq_mod = []
        expected_value_list = []
        difference_value_list = []

        for i in range(12):
            diff_seq_mod.append(0)
        for i in diff_seq:
            diff_seq_mod.append(i)

        for i in range(len(diff_seq_mod)):
            if diff_seq_mod[i] == 0:
                expected_value_list.append(0)
            else:
                expected_value = abs(diff_seq_mod[i-1] - diff_seq_mod[i-12])
                expected_value_list.append(expected_value)

            difference_value_list.append(diff_seq_mod[i] - expected_value_list[i])

        diffusion = -1
        for i in difference_value_list:
            if i != 0:
                diffusion = diffusion + 1
            elif i <= -1:
                raise Exception('faulty expected value of difference value computation')

        diffusion_counter_list.append(diffusion)

    for i in range(len(diff_pos)):
        diff_pos[i].append(diffusion_counter_list[i])
    '''
    f, ax = plt.subplots()
    sns.heatmap(pattern_array_reference_diff[0:80,20:30],
                #cmap='plasma_r',
                #cmap='magma_r',
                cmap='mako_r',
                #cmap='bone_r',
                cbar_kws={'label': 'Component Count in Window'}) #, cmap="YlGnBu") #, annot=True, fmt="d", linewidths=.5, ax=ax)
    plt.yticks(range(0,80,10))
    ax.set_xticklabels(range(20,30))
    ax.set_yticklabels(range(0,80,10))
    plt.xlabel("Knowledge Component ID")
    plt.ylabel("Sliding Window ID ")
    #plt.show()
    os.chdir('D:/Universitaet Mannheim/MMDS 7. Semester/Master Thesis/Outline/Plots')
    plt.savefig('SCM_direct.png')
    os.chdir('D:/Universitaet Mannheim/MMDS 7. Semester/Master Thesis/Outline/Data/Cleaning Robots')
    plt.close()

    pattern_array_reference_reco, columns_reference_reco = ReferenceMeasures.create_pattern_array(knowledgeComponent_dict_reco)

    print('number of recombinations: ', len(columns_reference_reco))

    from utilities_old.my_measure_utils import Misc

    diffusionPatternPos_CCM = Misc.find_diffusionPatterns(pattern_array_reference_reco)
    diffusionPatternPos_CCM, diff_sequence_list_SCM, irrelevant = Misc.find_diffusionSequenceAndLength(diffusionPatternPos_CCM, pattern_array_reference_reco)
    diffusionPatternPos_CCM = Misc.find_diffusionStepsAndPatternPerDiffusion(diffusionPatternPos_CCM, diff_sequence_list_SCM)
    # diff_pos = [ row, column, diffLength, diffSteps, patentsInDiff ]
    diffusionPatternPos_CCM = np.array(diffusionPatternPos_CCM)



    diffPatternLength_perTopic = []
    diffPatternSteps_perTopic = []
    diffPatternPatents_perTopic = []
    diff_perTopic = []


    for topic in range(np.max(diffusionPatternPos_CCM[:,1])):
        diffPatternLength_helper = []
        diffPatternSteps_helper = []
        diffPatternPatents_helper = []
        diff_perTopic_helper = []

        for diffPattern in diffusionPatternPos_CCM:
            if diffPattern[1] == topic:
                diffPatternLength_helper.append(diffPattern[2])
                diffPatternSteps_helper.append(diffPattern[3])
                diffPatternPatents_helper.append(diffPattern[4])
                diff_perTopic_helper.append(1)

        diffPatternLength_perTopic.append(np.mean(diffPatternLength_helper))
        diffPatternSteps_perTopic.append(np.mean(diffPatternSteps_helper))
        diffPatternPatents_perTopic.append(np.mean(diffPatternPatents_helper))
        diff_perTopic.append(sum(diff_perTopic_helper))

    print('Number of diffusion cycles / patterns in the scm: ', len(diffusionPatternPos_CCM))
    print('Average diffusion pattern length: ', np.mean(diffusionPatternPos_CCM[:, 2]))
    print('Average diffusion steps: ', np.mean(diffusionPatternPos_CCM[:, 3]))
    print('Average diffusion patents: ', np.mean(diffusionPatternPos_CCM[:, 4]))


    print('Averaged Average number of cycles/patterns per topic: ', np.mean(diff_perTopic))
    print('Median number of Average cycles/patterns per topic: ', np.median(diff_perTopic))
    print('Mode number of Average cycles/patterns per topic: ', statistics.mode(diff_perTopic))
    print('Max number of Average cycles/patterns per topic: ', max(diff_perTopic))
    print('Min number of Average cycles/patterns per topic: ', min(diff_perTopic))

    print('Average of the average topic diffusion steps: ', np.mean(diffPatternSteps_perTopic))
    print('Median of the average topic diffusion steps: ', np.median(diffPatternSteps_perTopic))
    print('Mode of the average topic diffusion steps: ', statistics.mode(diffPatternSteps_perTopic))
    print('Max of the average topic diffusion steps: ', max(diffPatternSteps_perTopic))
    print('Min of the average topic diffusion steps: ', min(diffPatternSteps_perTopic))

    print('Average of the average topic diffusion length: ', np.mean(diffPatternLength_perTopic))
    print('Median of the average topic diffusion length: ', np.median(diffPatternLength_perTopic))
    print('Mode of the average topic diffusion length: ', statistics.mode(diffPatternLength_perTopic))
    print('Max of the average topic diffusion length: ', max(diffPatternLength_perTopic))
    print('Min of the average topic diffusion length: ', min(diffPatternLength_perTopic))

    print('Average of the average topic diffusion patents: ', np.mean(diffPatternPatents_perTopic))
    print('Median of the average topic diffusion patents: ', np.median(diffPatternPatents_perTopic))
    print('Mode of the average topic diffusion patents: ', statistics.mode(diffPatternPatents_perTopic))
    print('Max of the average topic diffusion patents: ', max(diffPatternPatents_perTopic))
    print('Min of the average topic diffusion patents: ', min(diffPatternPatents_perTopic))

    # ---
    topicWithMostDiffSteps = []
    topicWithLeastDiffSteps = []
    for i in range(len(diffPatternSteps_perTopic)):
        #print(diffPatternSteps_perTopic[i])
        if diffPatternSteps_perTopic[i] == max(diffPatternSteps_perTopic):
            topicWithMostDiffSteps.append(i)
        if diffPatternSteps_perTopic[i] == min(diffPatternSteps_perTopic):
            topicWithLeastDiffSteps.append(i)

    topicWithMostDiffSteps = np.unique(topicWithMostDiffSteps)
    topicWithLeastDiffSteps = np.unique(topicWithLeastDiffSteps)

    print('recombination with most diff steps: ', topicWithMostDiffSteps)
    print('recombination with elast diff steps: ', topicWithLeastDiffSteps)
    print('number if recombination with elast diff steps: ', len(topicWithLeastDiffSteps))

    # ---
    topicWithMostDiffpats = []
    topicWithLeastDiffpats = []
    for i in range(len(diffPatternPatents_perTopic)):
        #print(diffPatternSteps_perTopic[i])
        if diffPatternPatents_perTopic[i] == max(diffPatternPatents_perTopic):
            topicWithMostDiffpats.append(i)
        if diffPatternPatents_perTopic[i] == min(diffPatternPatents_perTopic):
            topicWithLeastDiffpats.append(i)

    topicWithMostDiffpats = np.unique(topicWithMostDiffpats)
    topicWithLeastDiffpats = np.unique(topicWithLeastDiffpats)

    print('recombination with most diff patents: ', topicWithMostDiffpats)
    print('recombination with elast diff patents: ', topicWithLeastDiffpats)
    print('number of recombination with elast diff patents: ', len(topicWithLeastDiffpats))





    f, ax = plt.subplots()
    sns.heatmap(pattern_array_reference_reco[100:180,635:645], cmap='mako_r',
                cbar_kws={'label': 'Component Combination Count in Window'}) #, cmap="YlGnBu") #, annot=True, fmt="d", linewidths=.5, ax=ax)
    plt.yticks(range(0,80,10))
    ax.set_xticklabels(range(635,645))
    ax.set_yticklabels(range(100,180,10))
    plt.xlabel("Knowledge Component Combination ID")
    plt.ylabel("Sliding Window ID ")
    #plt.show()
    os.chdir('D:/Universitaet Mannheim/MMDS 7. Semester/Master Thesis/Outline/Plots')
    plt.savefig('CCM_direct.png')
    os.chdir('D:/Universitaet Mannheim/MMDS 7. Semester/Master Thesis/Outline/Data/Cleaning Robots')
    plt.close()

    #'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r',
    # 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r',
    # 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn',
    # 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu',
    # 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r',
    # 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r',
    # 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis',
    # 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'crest', 'crest_r', 'cubehelix', 'cubehelix_r',
    # 'flag', 'flag_r', 'flare', 'flare_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r',
    # 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot',
    # 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'icefire', 'icefire_r', 'inferno',
    # 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'mako', 'mako_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r',
    # 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'rocket', 'rocket_r', 'seismic', 'seismic_r',
    # 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r',
    # 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis',
    # 'viridis_r', 'vlag', 'vlag_r', 'winter', 'winter_r'

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



