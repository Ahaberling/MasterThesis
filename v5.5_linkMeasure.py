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

    with open('topicProject_graphs', 'rb') as handle:
        topicProject_graphs = pk.load(handle)

    # --- pattern array ---#

    from utilities.my_measure_utils import EdgeWeightMeasures

    x = [1,2,3,4,5,6,7,8,9,10]
    print(np.quantile(x, 0.25))
    print(np.quantile(x, 0.5))
    print(np.quantile(x, 0.75))

    for i in range(len(topicProject_graphs)):
        if i >= 145:
            print('WINDOW: ', i)
            for (u, v, wt) in topicProject_graphs['window_{}'.format(i*30)].edges.data('weight'):
                if u == 'topic_5':
                    print(u, v)
                if v == 'topic_5':
                    print(u, v)


    #with open('diffusion_array_edgeWeight', 'rb') as handle:
    #    diffusion_array_edgeWeight = pk.load(handle)

    diffusion_array_edgeWeight, columns_diff_edgeWeight = EdgeWeightMeasures.create_diffusion_array(topicProject_graphs, 0.005, 0.25)

    import statistics

    from utilities.my_measure_utils import Misc

    diffusionPatternPos_SCM = Misc.find_diffusionPatterns(diffusion_array_edgeWeight)
    diffusionPatternPos_SCM, diff_sequence_list_SCM, irrelevant = Misc.find_diffusionSequenceAndLength(diffusionPatternPos_SCM,
                                                                                           diffusion_array_edgeWeight)
    #diffusionPatternPos_SCM = Misc.find_diffusionStepsAndPatternPerDiffusion(diffusionPatternPos_SCM,
    #                                                                         diff_sequence_list_SCM)
    # diff_pos = [ row, column, diffLength, diffSteps, patentsInDiff ]
    diffusionPatternPos_SCM = np.array(diffusionPatternPos_SCM)

    diffPatternLength_perTopic = []
    diff_perTopic = []



    print('Number of diffusion cycles / patterns in the scm: ', len(diffusionPatternPos_SCM))
    print('Average diffusion pattern length: ', np.mean(diffusionPatternPos_SCM[:, 2]))





    import matplotlib.pyplot as plt
    import seaborn as sns

    f, ax = plt.subplots()
    sns.heatmap(diffusion_array_edgeWeight[0:80, 20:30],
                # cmap='plasma_r',
                # cmap='magma_r',
                cmap='mako_r',
                # cmap='bone_r',
                cbar_kws={
                    'label': 'Component Count in Window'})  # , cmap="YlGnBu") #, annot=True, fmt="d", linewidths=.5, ax=ax)
    plt.yticks(range(0, 80, 10))
    ax.set_xticklabels(range(20, 30))
    ax.set_yticklabels(range(0, 80, 10))
    plt.xlabel("Knowledge Component ID")
    plt.ylabel("Sliding Window ID ")
    #plt.show()
    os.chdir('D:/Universitaet Mannheim/MMDS 7. Semester/Master Thesis/Outline/Plots')
    plt.savefig('SCM_EdgeWeight.png')
    os.chdir('D:/Universitaet Mannheim/MMDS 7. Semester/Master Thesis/Outline/Data/Cleaning Robots')
    plt.close()



    #with open('recombinationDiffusion_edgeWeight', 'rb') as handle:
    #    recombinationDiffusion_edgeWeight = pk.load(handle)

    recombinationDiffusion_edgeWeight, recombinationDiffusion_frac, pattern_array_thresh_reference, columns_recom_edgeWeight = EdgeWeightMeasures.create_recombination_array(topicProject_graphs, 0.005)



    diffusionPatternPos_CCM = Misc.find_diffusionPatterns(recombinationDiffusion_edgeWeight)
    diffusionPatternPos_CCM, diff_sequence_list_SCM, irrelevant = Misc.find_diffusionSequenceAndLength(diffusionPatternPos_CCM, recombinationDiffusion_edgeWeight)
    #diffusionPatternPos_CCM = Misc.find_diffusionStepsAndPatternPerDiffusion(diffusionPatternPos_CCM, diff_sequence_list_SCM)
    # diff_pos = [ row, column, diffLength, diffSteps, patentsInDiff ]
    diffusionPatternPos_CCM = np.array(diffusionPatternPos_CCM)



    print('Number of diffusion cycles / patterns in the scm: ', len(diffusionPatternPos_CCM))
    print('Average diffusion pattern length: ', np.mean(diffusionPatternPos_CCM[:, 2]))





    f, ax = plt.subplots()
    sns.heatmap(recombinationDiffusion_edgeWeight[100:180,635:645], cmap='mako_r',
                cbar_kws={'label': 'Component Combination Count in Window'}) #, cmap="YlGnBu") #, annot=True, fmt="d", linewidths=.5, ax=ax)
    plt.yticks(range(0,80,10))
    ax.set_xticklabels(range(635,645))
    ax.set_yticklabels(range(100,180,10))
    plt.xlabel("Knowledge Component Combination ID")
    plt.ylabel("Sliding Window ID ")
    #plt.show()
    os.chdir('D:/Universitaet Mannheim/MMDS 7. Semester/Master Thesis/Outline/Plots')
    plt.savefig('CCM_EdgeWeight.png')
    os.chdir('D:/Universitaet Mannheim/MMDS 7. Semester/Master Thesis/Outline/Data/Cleaning Robots')
    plt.close()










    filename = 'diffusion_array_edgeWeight'
    outfile = open(filename, 'wb')
    pk.dump(diffusion_array_edgeWeight, outfile)
    outfile.close()

    filename = 'columns_diff_edgeWeight'
    outfile = open(filename, 'wb')
    pk.dump(columns_diff_edgeWeight, outfile)
    outfile.close()

    filename = 'recombinationDiffusion_edgeWeight'
    outfile = open(filename, 'wb')
    pk.dump(recombinationDiffusion_edgeWeight, outfile)
    outfile.close()

    filename = 'columns_recom_edgeWeight'
    outfile = open(filename, 'wb')
    pk.dump(columns_recom_edgeWeight, outfile)
    outfile.close()