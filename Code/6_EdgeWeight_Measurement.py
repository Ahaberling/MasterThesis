if __name__ == '__main__': 

    # --- Import Libraries ---#
    print('\n#--- Import Libraries ---#\n')

    # Utility
    import os
    import tqdm
    import statistics

    # Data handling
    import pandas as pd
    import pickle as pk
    import numpy as np

    # Visualization
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Custom functions
    from utilities.Measurement_utils import EdgeWeight_Measurement
    from utilities.Measurement_utils import Misc


    # --- Initialization ---#
    print('\n# --- Initialization ---#\n')

    path = 'D:/'

    # load data
    os.chdir(path)

    with open('topicProject_graphs', 'rb') as handle:
            topicProject_graphs = pk.load(handle)

    # --- Creating SCMs ---#
    print('\n#--- Creating SCMs ---#\n')

    EdgeWeight_SCM, EdgeWeight_SCM_columns = EdgeWeight_Measurement.create_diffusion_array(topicProject_graphs)

    # Descriptives
    diffusionPatternPos_SCM = Misc.find_diffusionPatterns(EdgeWeight_SCM)
    diffusionPatternPos_SCM, diff_sequence_list_SCM, placeholder = Misc.find_diffusionSequenceAndLength(diffusionPatternPos_SCM, EdgeWeight_SCM)
    # diff_pos = [ row, column, diffLength, diffSteps, patentsInDiff ]
    diffusionPatternPos_SCM = np.array(diffusionPatternPos_SCM)

    diffPatternLength_perTopic = []
    diff_perTopic = []

    print('----- SCM Descriptives -----')
    print('Number of diffusion cycles / patterns in the -SCM-: ', len(diffusionPatternPos_SCM))
    print('Average diffusion pattern length -SCM-: ', np.mean(diffusionPatternPos_SCM[:, 2]))

    # Visualization
    f, ax = plt.subplots()
    sns.heatmap(EdgeWeight_SCM[0:80, 20:30], cmap='mako_r', cbar_kws={'label': 'Component Count in Window'})  
    plt.yticks(range(0, 80, 10))
    ax.set_xticklabels(range(20, 30))
    ax.set_yticklabels(range(0, 80, 10))
    plt.xlabel("Knowledge Component ID")
    plt.ylabel("Sliding Window ID ")
    plt.savefig('SCM_EdgeWeight.png')
    plt.close()


    # --- Creating CCMs ---#
    print('\n#--- Creating CCMs ---#\n')

    EdgeWeight_CCM, EdgeWeight_CCM_columns = EdgeWeight_Measurement.create_recombination_array(topicProject_graphs)

    # Descriptives
    diffusionPatternPos_CCM = Misc.find_diffusionPatterns(EdgeWeight_CCM)
    diffusionPatternPos_CCM, diff_sequence_list_CCM, placeholder = Misc.find_diffusionSequenceAndLength(diffusionPatternPos_CCM, EdgeWeight_CCM)
    # diff_pos = [ row, column, diffLength, diffSteps, patentsInDiff ]
    diffusionPatternPos_CCM = np.array(diffusionPatternPos_CCM)

    print('----- CCM Descriptives -----')
    print('CCM size: ', np.shape(EdgeWeight_CCM))
    print('Number of diffusion cycles / patterns in the -CCM-: ', len(diffusionPatternPos_CCM))
    print('Average diffusion pattern length -CCM-: ', np.mean(diffusionPatternPos_CCM[:, 2]))

    # Visualization
    f, ax = plt.subplots()
    sns.heatmap(EdgeWeight_CCM[100:180, 770:780], cmap='mako_r', cbar_kws={
                    'label': 'Component Combination Count in Window'}) 
    plt.yticks(range(0, 80, 10))
    ax.set_xticklabels(range(770, 780))
    ax.set_yticklabels(range(100, 180, 10))
    plt.xlabel("Knowledge Component Combination ID")
    plt.ylabel("Sliding Window ID ")
    plt.savefig('CCM_EdgeWeight.png')
    plt.close()
    

    
    # --- Saving Data ---#
    print('\n#--- Saving Data ---#\n')

    filename = 'EdgeWeight_SCM'
    outfile = open(filename, 'wb')
    pk.dump(EdgeWeight_SCM, outfile)
    outfile.close()

    filename = 'EdgeWeight_SCM_columns'
    outfile = open(filename, 'wb')
    pk.dump(EdgeWeight_SCM_columns, outfile)
    outfile.close()

    filename = 'EdgeWeight_CCM'
    outfile = open(filename, 'wb')
    pk.dump(EdgeWeight_CCM, outfile)
    outfile.close()

    filename = 'EdgeWeight_CCM_columns'
    outfile = open(filename, 'wb')
    pk.dump(EdgeWeight_CCM_columns, outfile)
    outfile.close()
