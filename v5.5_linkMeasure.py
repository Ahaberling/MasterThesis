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

    diffusion_array_edgeWeight, diffusion_array_frac, diffusion_array_thresh, columns_diff_edgeWeight = EdgeWeightMeasures.create_diffusion_array(topicProject_graphs, 0.005, 0.25)
    recombinationDiffusion_edgeWeight, recombinationDiffusion_frac, pattern_array_thresh_reference, columns_recom_edgeWeight = EdgeWeightMeasures.create_recombination_array(topicProject_graphs, 0.005)

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