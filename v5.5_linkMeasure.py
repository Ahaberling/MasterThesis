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

    def create_diffusion_array(topicProject_graphs, threshold):
        # get row length
        row_length = len(topicProject_graphs)

        # get column length
        all_nodes = []
        for window_id, graph in topicProject_graphs.items():
            for n in graph.nodes():
                all_nodes.append(int(n[6:]))

        all_nodes_unique = np.unique(all_nodes, axis=0)
        column_length = len(all_nodes_unique)
        all_nodes_unique.sort()

        diffusion_array = np.zeros((row_length, column_length), dtype=int)
        diffusion_array_frac = np.zeros((row_length, column_length), dtype=float)

        pbar = tqdm.tqdm(total=len(diffusion_array))

        for i in range(len(diffusion_array)):
            for j in range(len(diffusion_array.T)):

                all_edgeNodes = []
                for (u, v) in topicProject_graphs['window_{0}'.format(i * 30)].edges():
                    all_edgeNodes.append(int(u[6:]))
                    all_edgeNodes.append(int(v[6:]))

                diffusion_array[i, j] = all_edgeNodes.count(all_nodes_unique[j])

            pbar.update(1)
            #print(len(topicProject_graphs['window_{0}'.format(i * 30)].edges()))
            diffusion_array_frac[i] = diffusion_array[i] / len(topicProject_graphs['window_{0}'.format(i * 30)].edges())

        pbar.close()
        diffusion_array_thresh = np.where(diffusion_array_frac < threshold, 0, 1)



        return diffusion_array, diffusion_array_frac, diffusion_array_thresh

    diffusion_array, diffusion_array_frac, diffusion_array_thresh = create_diffusion_array(topicProject_graphs, 0.005)


    def create_recombination_array(topicProject_graphs, threshold):
        # get row length
        row_length = len(topicProject_graphs)

        # get column length
        all_edges = []
        for window_id, graph in topicProject_graphs.items():
            for (u, v) in graph.edges():
                all_edges.append((int(u[6:]), int(v[6:])))

        all_edges_unique = np.unique(all_edges, axis=0)
        column_length = len(all_edges_unique)
        all_edges_unique.sort()

        recombinationDiffusion = np.zeros((row_length, column_length), dtype=float)

        pbar = tqdm.tqdm(total=len(recombinationDiffusion))

        for i in range(len(recombinationDiffusion)):
            for j in range(len(recombinationDiffusion.T)):

                for (u, v, wt) in topicProject_graphs['window_{0}'.format(i * 30)].edges.data('weight'):

                    if int(u[6:]) == all_edges_unique[j][0]:
                        if int(v[6:]) == all_edges_unique[j][1]:
                            recombinationDiffusion[i, j] = wt

                    elif int(u[6:]) == all_edges_unique[j][1]:
                        if int(v[6:]) == all_edges_unique[j][0]:
                            recombinationDiffusion[i, j] = wt
            pbar.update(1)

        pbar.close()

        row_sum = recombinationDiffusion.sum(axis=1)
        recombinationDiffusion_frac = recombinationDiffusion / row_sum[:, np.newaxis]
        recombinationDiffusion_thresh = np.where(recombinationDiffusion_frac < threshold, 0, 1)

        return recombinationDiffusion, recombinationDiffusion_frac, recombinationDiffusion_thresh


    recombinationDiffusion, recombinationDiffusion_frac, pattern_array_thresh_reference = create_recombination_array(topicProject_graphs, 0.005)
    print(1+1)
    print(1+1)
    print(1+1)
    print(1+1)
    print(1+1)
