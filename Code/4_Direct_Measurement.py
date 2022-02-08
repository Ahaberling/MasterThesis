if __name__ == '__main__':

    # --- Import Libraries ---#
    print('\n#--- Import Libraries ---#\n')
 
    # Utility
    import os
    import statistics

    # Data handling
    import pandas as pd
    import numpy as np
    import pickle as pk

    # Visualization
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Custom functions
    from utilities.Measurement_utils import Direct_Measurement
    from utilities.Measurement_utils import Misc



    # --- Initialization --#
    print('\n#--- Initialization ---#\n')

    path = 'D:/'

    os.chdir(path)

    patent_topicDist = pd.read_csv('patent_topicDistribution_mallet.csv', quotechar='"', skipinitialspace=True)
    #patent_topicDist = pd.read_csv('patent_topicDist_mallet.csv', quotechar='"', skipinitialspace=True)
    patent_IPC = pd.read_csv('cleaning_robot_EP_patents_IPC.csv', quotechar='"', skipinitialspace=True)

    patent_topicDist = patent_topicDist.to_numpy()
    patent_IPC = patent_IPC.to_numpy()

    patent_lda_ipc = patent_topicDist

    with open('slidingWindow_dict', 'rb') as handle:
        slidingWindow_dict = pk.load(handle)



    #--- Preprocessing for pattern arrays ---#
    print('\n#--- Preprocessing for pattern arrays ---#\n')

    # Create dictionaries containing ALL ipc/topic singles/pairs/triples for a window
    knowledgeComponent_dict_SCM = Direct_Measurement.extract_knowledgeComponent_per_window(slidingWindow_dict, kC='topic', unit=1)
    knowledgeComponent_dict_CCM = Direct_Measurement.extract_knowledgeComponent_per_window(slidingWindow_dict, kC='topic', unit=2)

    # Descriptives
    all_components = []
    component_perWindow = []
    component_perWindow_unique = []

    for i in knowledgeComponent_dict_SCM.values():
        all_components.append(i)
        component_perWindow.append(len(i))
        component_perWindow_unique.append(len(np.unique(i)))
    all_components = [item for sublist in all_components for item in sublist]

    print('----- First Descirptives SCM -----')
    print('Number of retrieved components -SCM-: ', len(all_components))
    print('Number of unique retrieved components -SCM-: ', len(np.unique(all_components)), '\n')

    print('Average Number retrieved components per window -SCM-: ', np.mean(component_perWindow))
    print('Median Number retrieved components per window -SCM-: ', np.median(component_perWindow))
    print('Mode Number retrieved components per window -SCM-: ', statistics.mode(component_perWindow))
    print('Max Number retrieved components per window -SCM-: ', max(component_perWindow))
    print('Min Number retrieved components per window -SCM-: ', min(component_perWindow), '\n')

    print('Average Number unique retrieved components per window -SCM-: ', np.mean(component_perWindow_unique))
    print('Median Number unique retrieved components per window -SCM-: ', np.median(component_perWindow_unique))
    print('Mode Number unique retrieved components per window -SCM-: ', statistics.mode(component_perWindow_unique))
    print('Max Number unique retrieved components per window -SCM-: ', max(component_perWindow_unique))
    print('Min Number unique retrieved components per window -SCM-: ', min(component_perWindow_unique), '\n')

    all_components = []
    component_perWindow = []
    component_perWindow_unique = []
    for i in knowledgeComponent_dict_CCM.values():
        all_components.append(i)
        component_perWindow.append(len(i))
        component_perWindow_unique.append(len(np.unique(i, axis=0)))
    all_components = [item for sublist in all_components for item in sublist]

    print('----- First Descirptives CCM -----')
    print('Number of retrieved component combinations -CCM-: ', len(all_components))
    print('Number of unique retrieved component combinations -CCM-: ', len(np.unique(all_components, axis=0)), '\n')

    print('Average Number retrieved components per window -CCM-: ', np.mean(component_perWindow))
    print('Median Number retrieved components per window -CCM-: ', np.median(component_perWindow))
    print('Mode Number retrieved components per window -CCM-: ', statistics.mode(component_perWindow))
    print('Max Number retrieved components per window -CCM-: ', max(component_perWindow))
    print('Min Number retrieved components per window -CCM-: ', min(component_perWindow), '\n')

    print('Average Number unique retrieved components per window -CCM-: ', np.mean(component_perWindow_unique))
    print('Median Number unique retrieved components per window -CCM-: ', np.median(component_perWindow_unique))
    print('Mode Number unique retrieved components per window -CCM-: ', statistics.mode(component_perWindow_unique))
    print('Max Number unique retrieved components per window -CCM-: ', max(component_perWindow_unique))
    print('Min Number unique retrieved components per window -CCM-: ', min(component_perWindow_unique), '\n')


    #--- Constructing pattern arrays---#
    print('\n#--- Constructing SCM ---#\n')

    direct_SCM, columns_Direct_SCM = Direct_Measurement.create_pattern_array(knowledgeComponent_dict_SCM)


    # Descriptives SCM
    diffusionPatternPos_SCM = Misc.find_diffusionPatterns(direct_SCM)
    diffusionPatternPos_SCM, diff_sequence_list_SCM, placeholder = Misc.find_diffusionSequenceAndLength(diffusionPatternPos_SCM, direct_SCM)
    #diffusionPatternPos_SCM = Misc.find_diffusionStepsAndPatternPerDiffusion(diffusionPatternPos_SCM, diff_sequence_list_SCM)  # needs to be revisited

    # diff_pos = [ row, column, diffLength, diffSteps, patentsInDiff ]
    diffusionPatternPos_SCM = np.array(diffusionPatternPos_SCM)

    diffPatternLength_perTopic = []
    #diffPatternSteps_perTopic = []
    #diffPatternPatents_perTopic = []
    diff_perTopic = []

    for topic in range(np.max(diffusionPatternPos_SCM[:, 1])):
        diffPatternLength_helper = []
        #diffPatternSteps_helper = []
        #diffPatternPatents_helper = []
        diff_perTopic_helper = []

        for diffPattern in diffusionPatternPos_SCM:
            if diffPattern[1] == topic:
                diffPatternLength_helper.append(diffPattern[2])
                #diffPatternSteps_helper.append(diffPattern[3])
                #diffPatternPatents_helper.append(diffPattern[4])
                diff_perTopic_helper.append(1)

        diffPatternLength_perTopic.append(np.mean(diffPatternLength_helper))
        #diffPatternSteps_perTopic.append(np.mean(diffPatternSteps_helper))
        #diffPatternPatents_perTopic.append(np.mean(diffPatternPatents_helper))
        diff_perTopic.append(sum(diff_perTopic_helper))

    print('------ Descriptives SCM ------')
    print('Number of diffusion cycles / patterns in the scm: ', len(diffusionPatternPos_SCM))
    print('Average diffusion pattern length -SCM-: ', np.mean(diffusionPatternPos_SCM[:, 2]))
    #print('Average diffusion steps -SCM-: ', np.mean(diffusionPatternPos_SCM[:, 3]))
    #print('Average diffusion patents -SCM-: ', np.mean(diffusionPatternPos_SCM[:, 4]), '\n')

    print('Averaged Average number of cycles/patterns per topic -SCM-: ', np.mean(diff_perTopic))
    print('Median number of Average cycles/patterns per topic -SCM-: ', np.median(diff_perTopic))
    print('Mode number of Average cycles/patterns per topic -SCM-: ', statistics.mode(diff_perTopic))
    print('Max number of Average cycles/patterns per topic -SCM-: ', max(diff_perTopic))
    print('Min number of Average cycles/patterns per topic -SCM-: ', min(diff_perTopic), '\n')
    '''
    print('Average of the average topic diffusion steps -SCM-: ', np.mean(diffPatternSteps_perTopic))
    print('Median of the average topic diffusion steps -SCM-: ', np.median(diffPatternSteps_perTopic))
    print('Mode of the average topic diffusion steps -SCM-: ', statistics.mode(diffPatternSteps_perTopic))
    print('Max of the average topic diffusion steps -SCM-: ', max(diffPatternSteps_perTopic))
    print('Min of the average topic diffusion steps -SCM-: ', min(diffPatternSteps_perTopic), '\n')
    '''
    print('Average of the average topic diffusion length -SCM-: ', np.mean(diffPatternLength_perTopic))
    print('Median of the average topic diffusion length -SCM-: ', np.median(diffPatternLength_perTopic))
    print('Mode of the average topic diffusion length -SCM-: ', statistics.mode(diffPatternLength_perTopic))
    print('Max of the average topic diffusion length -SCM-: ', max(diffPatternLength_perTopic))
    print('Min of the average topic diffusion length -SCM-: ', min(diffPatternLength_perTopic), '\n')
    '''
    print('Average of the average topic diffusion patents -SCM-: ', np.mean(diffPatternPatents_perTopic))
    print('Median of the average topic diffusion patents -SCM-: ', np.median(diffPatternPatents_perTopic))
    print('Mode of the average topic diffusion patents -SCM-: ', statistics.mode(diffPatternPatents_perTopic))
    print('Max of the average topic diffusion patents -SCM-: ', max(diffPatternPatents_perTopic))
    print('Min of the average topic diffusion patents -SCM-: ', min(diffPatternPatents_perTopic), '\n')
    '''
    '''
    topicWithMostDiffSteps = []
    topicWithLeastDiffSteps = []
    for i in range(len(diffPatternSteps_perTopic)):
        if diffPatternSteps_perTopic[i] == max(diffPatternSteps_perTopic):
            topicWithMostDiffSteps.append(i)
        if diffPatternSteps_perTopic[i] == min(diffPatternSteps_perTopic):
            topicWithLeastDiffSteps.append(i)

    topicWithMostDiffSteps = np.unique(topicWithMostDiffSteps)
    topicWithLeastDiffSteps = np.unique(topicWithLeastDiffSteps)

    print('topic with most diff steps -SCM-: ', topicWithMostDiffSteps)
    print('topic with elast diff steps -SCM-: ', topicWithLeastDiffSteps, '\n')


    topicWithMostDiffpats = []
    topicWithLeastDiffpats = []
    for i in range(len(diffPatternPatents_perTopic)):
        if diffPatternPatents_perTopic[i] == max(diffPatternPatents_perTopic):
            topicWithMostDiffpats.append(i)
        if diffPatternPatents_perTopic[i] == min(diffPatternPatents_perTopic):
            topicWithLeastDiffpats.append(i)

    topicWithMostDiffpats = np.unique(topicWithMostDiffpats)
    topicWithLeastDiffpats = np.unique(topicWithLeastDiffpats)

    print('topic with most diff patents  -SCM-: ', topicWithMostDiffpats)
    print('topic with elast diff patents  -SCM-: ', topicWithLeastDiffpats, '\n')
    '''

    f, ax = plt.subplots()
    sns.heatmap(direct_SCM[0:80, 20:30],
                cmap='mako_r',
                cbar_kws={'label': 'Component Count in Window'})
    plt.yticks(range(0, 80, 10))
    ax.set_xticklabels(range(20, 30))
    ax.set_yticklabels(range(0, 80, 10))
    plt.xlabel("Knowledge Component ID")
    plt.ylabel("Sliding Window ID ")
    plt.savefig('SCM_direct.png')
    plt.close()

    # Creating CCM
    direct_CCM, columns_Direct_CCM = Direct_Measurement.create_pattern_array(knowledgeComponent_dict_CCM)

    # Descriptives
    diffusionPatternPos_CCM = Misc.find_diffusionPatterns(direct_CCM)
    diffusionPatternPos_CCM, diff_sequence_list_CCM, placeholder = Misc.find_diffusionSequenceAndLength(diffusionPatternPos_CCM, direct_CCM)
    #diffusionPatternPos_CCM = Misc.find_diffusionStepsAndPatternPerDiffusion(diffusionPatternPos_CCM, diff_sequence_list_CCM) # needs to be revisited

    # diff_pos = [ row, column, diffLength, diffSteps, patentsInDiff ]
    diffusionPatternPos_CCM = np.array(diffusionPatternPos_CCM)

    diffPatternLength_perTopic = []
    #diffPatternSteps_perTopic = []
    #diffPatternPatents_perTopic = []
    diff_perTopic = []

    for topic in range(np.max(diffusionPatternPos_CCM[:, 1])):
        diffPatternLength_helper = []
        #diffPatternSteps_helper = []
        #diffPatternPatents_helper = []
        diff_perTopic_helper = []

        for diffPattern in diffusionPatternPos_CCM:
            if diffPattern[1] == topic:
                diffPatternLength_helper.append(diffPattern[2])
                #diffPatternSteps_helper.append(diffPattern[3])
                #diffPatternPatents_helper.append(diffPattern[4])
                diff_perTopic_helper.append(1)

        diffPatternLength_perTopic.append(np.mean(diffPatternLength_helper))
        #diffPatternSteps_perTopic.append(np.mean(diffPatternSteps_helper))
        #diffPatternPatents_perTopic.append(np.mean(diffPatternPatents_helper))
        diff_perTopic.append(sum(diff_perTopic_helper))

    print('------ Descriptives SCM ------')
    print('Number of recombinations -CCM-: ', len(columns_Direct_CCM), '\n')

    print('Number of diffusion cycles / patterns in the -CCM-: ', len(diffusionPatternPos_CCM))
    print('Average diffusion pattern length -CCM-: ', np.mean(diffusionPatternPos_CCM[:, 2]))
    #print('Average diffusion steps -CCM-: ', np.mean(diffusionPatternPos_CCM[:, 3]))
    #print('Average diffusion patents -CCM-: ', np.mean(diffusionPatternPos_CCM[:, 4]), '\n')

    print('Averaged Average number of cycles/patterns per topic -CCM-: ', np.mean(diff_perTopic))
    print('Median number of Average cycles/patterns per topic -CCM-: ', np.median(diff_perTopic))
    print('Mode number of Average cycles/patterns per topic -CCM-: ', statistics.mode(diff_perTopic))
    print('Max number of Average cycles/patterns per topic -CCM-: ', max(diff_perTopic))
    print('Min number of Average cycles/patterns per topic -CCM-: ', min(diff_perTopic), '\n')
    '''
    print('Average of the average topic diffusion steps -CCM-: ', np.mean(diffPatternSteps_perTopic))
    print('Median of the average topic diffusion steps -CCM-: ', np.median(diffPatternSteps_perTopic))
    print('Mode of the average topic diffusion steps -CCM-: ', statistics.mode(diffPatternSteps_perTopic))
    print('Max of the average topic diffusion steps -CCM-: ', max(diffPatternSteps_perTopic))
    print('Min of the average topic diffusion steps -CCM-: ', min(diffPatternSteps_perTopic), '\n')
    '''
    print('Average of the average topic diffusion length -CCM-: ', np.mean(diffPatternLength_perTopic))
    print('Median of the average topic diffusion length -CCM-: ', np.median(diffPatternLength_perTopic))
    print('Mode of the average topic diffusion length -CCM-: ', statistics.mode(diffPatternLength_perTopic))
    print('Max of the average topic diffusion length -CCM-: ', max(diffPatternLength_perTopic))
    print('Min of the average topic diffusion length -CCM-: ', min(diffPatternLength_perTopic), '\n')
    '''
    print('Average of the average topic diffusion patents -CCM-: ', np.mean(diffPatternPatents_perTopic))
    print('Median of the average topic diffusion patents -CCM-: ', np.median(diffPatternPatents_perTopic))
    print('Mode of the average topic diffusion patents -CCM-: ', statistics.mode(diffPatternPatents_perTopic))
    print('Max of the average topic diffusion patents -CCM-: ', max(diffPatternPatents_perTopic))
    print('Min of the average topic diffusion patents -CCM-: ', min(diffPatternPatents_perTopic), '\n')
    '''
    '''
    topicWithMostDiffSteps = []
    topicWithLeastDiffSteps = []
    for i in range(len(diffPatternSteps_perTopic)):
        if diffPatternSteps_perTopic[i] == max(diffPatternSteps_perTopic):
            topicWithMostDiffSteps.append(i)
        if diffPatternSteps_perTopic[i] == min(diffPatternSteps_perTopic):
            topicWithLeastDiffSteps.append(i)

    topicWithMostDiffSteps = np.unique(topicWithMostDiffSteps)
    topicWithLeastDiffSteps = np.unique(topicWithLeastDiffSteps)

    print('recombination with most diff steps -CCM-: ', topicWithMostDiffSteps)
    print('recombination with elast diff steps -CCM-: ', topicWithLeastDiffSteps)
    print('number of recombination with least diff steps -CCM-: ', len(topicWithLeastDiffSteps), '\n')


    topicWithMostDiffpats = []
    topicWithLeastDiffpats = []
    for i in range(len(diffPatternPatents_perTopic)):
        if diffPatternPatents_perTopic[i] == max(diffPatternPatents_perTopic):
            topicWithMostDiffpats.append(i)
        if diffPatternPatents_perTopic[i] == min(diffPatternPatents_perTopic):
            topicWithLeastDiffpats.append(i)

    topicWithMostDiffpats = np.unique(topicWithMostDiffpats)
    topicWithLeastDiffpats = np.unique(topicWithLeastDiffpats)

    print('recombination with most diff patents -CCM-: ', topicWithMostDiffpats)
    print('recombination with elast diff patents -CCM-: ', topicWithLeastDiffpats)
    print('number of recombination with elast diff patents -CCM-: ', len(topicWithLeastDiffpats), '\n')
    '''

    # Visulization
    f, ax = plt.subplots()
    sns.heatmap(direct_CCM[100:180, 770:780], cmap='mako_r',
                cbar_kws={
                    'label': 'Component Combination Count in Window'})
    plt.yticks(range(0, 80, 10))
    ax.set_xticklabels(range(770, 780))
    ax.set_yticklabels(range(100, 180, 10))
    plt.xlabel("Knowledge Component Combination ID")
    plt.ylabel("Sliding Window ID ")
    plt.savefig('CCM_direct.png')
    plt.close()

    #--- Saving results ---#
    print('\n#--- Saving results ---#\n')

    filename = 'direct_SCM'
    outfile = open(filename, 'wb')
    pk.dump(direct_SCM, outfile)
    outfile.close()

    filename = 'columns_Direct_SCM'
    outfile = open(filename, 'wb')
    pk.dump(columns_Direct_SCM, outfile)
    outfile.close()

    filename = 'direct_CCM'
    outfile = open(filename, 'wb')
    pk.dump(direct_CCM, outfile)
    outfile.close()

    filename = 'columns_Direct_CCM'
    outfile = open(filename, 'wb')
    pk.dump(columns_Direct_CCM, outfile)
    outfile.close()
