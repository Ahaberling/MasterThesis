if __name__ == '__main__':

    # --- Import Libraries ---#
    print('\n#--- Import Libraries ---#\n')

    import pickle as pk
    import numpy as np

    import networkx as nx
    from cdlib import algorithms
    # import wurlitzer                   #not working for windows

    import tqdm
    import os

    # --- Initialization ---#
    print('\n# --- Initialization ---#\n')

    os.chdir('D:/Universitaet Mannheim/MMDS 7. Semester/Master Thesis/Outline/Data/Cleaning Robots')

    with open('windows_topicOccu', 'rb') as handle:
        topicOccu = pk.load(handle)

    '''
    for window_id, graph in topicOccu.items():
        #print(graph)
        print(len(graph.nodes()))
        print(graph.nodes())
        print(len(graph.edges()))
        print(graph.edges())

        helper = []
        for (u, v, wt) in graph.edges.data('weight'):
            if wt >= 0.005:
                print(u, v, wt)
                helper.append((u, v, wt))
        print(len(helper))
        break
    '''
    # --- pattern array ---#

    # get row length
    row_length = len(topicOccu)
    print(row_length)

    # get column length

    all_edges = []
    for window_id, graph in topicOccu.items():
        for (u, v) in graph.edges():
            all_edges.append((u, v))
    # print(all_edges)
    print(len(all_edges))

    all_edges_unique = np.unique(all_edges, axis=0)

    print(len(all_edges_unique))

    column_length = len(all_edges_unique)

    all_edges_unique.sort()
    print(all_edges_unique)

    # recombinationDiffusion = np.zeros((row_length, column_length), dtype=float)
    recombinationDiffusion = np.full((row_length, column_length), 9999999, dtype=float)
    # recombinationDiffusion_fraction = np.zeros((row_length, column_length), dtype=int)
    # recombinationDiffusion = np.zeros((row_length, column_length), dtype=int)

    pbar = tqdm.tqdm(total=len(recombinationDiffusion))

    for i in range(len(recombinationDiffusion)):
        for j in range(len(recombinationDiffusion.T)):

            for (u, v, wt) in topicOccu['window_{0}'.format(i * 30)].edges.data('weight'):
                # print(u)
                # print(v)
                # print(wt)
                # print(all_edges_unique[j][0])
                # print(all_edges_unique[j][1])
                if u == all_edges_unique[j][0]:
                    if v == all_edges_unique[j][1]:
                        recombinationDiffusion[i, j] = wt

                elif u == all_edges_unique[j][1]:
                    if v == all_edges_unique[j][0]:
                        recombinationDiffusion[i, j] = wt
        pbar.update(1)
    pbar.close()

    # column 270 has 27 rows with values (varying)

    print(1 + 1)
    print(1 + 1)

