if __name__ == '__main__':

#--- Import Libraries ---#
    print('\n#--- Import Libraries ---#\n')

    import pandas as pd
    import numpy as np
    import pickle as pk

    import networkx as nx

    import tqdm
    import itertools
    import os

#--- Initialization ---#
    print('\n# --- Initialization ---#\n')

    os.chdir('D:/Universitaet Mannheim/MMDS 7. Semester/Master Thesis/Outline/Data/Cleaning Robots')

    patent_lda_ipc = pd.read_csv('patent_lda_ipc.csv', quotechar='"', skipinitialspace=True)
    topics = pd.read_csv('patent_topics_mallet.csv', quotechar='"', skipinitialspace=True)
    parent = pd.read_csv('cleaning_robot_EP_backward_citations.csv', quotechar='"', skipinitialspace=True)

    patent_lda_ipc = patent_lda_ipc.to_numpy()
    topics = topics.to_numpy()
    parent = parent.to_numpy()

    '''
    with open('window90by1_bipartite', 'rb') as handle:
        bipartite = pk.load(handle)
    '''
    '''
    with open('window90by1_topicOccu', 'rb') as handle:
        topicOccu = pk.load(handle)
    '''


    with open('window90by1_topicSim', 'rb') as handle:
        topicSim = pk.load(handle)

#--- Applying Community detection to each graph/window ---#

    #print(topicSim)
    print(topicSim['window_0'])
    print(topicSim['window_0'].nodes())
    print(topicSim['window_0'].edges())

    # apply community detection on very window
    # save result in dict
    # find most likely topic/s for each community with some sort of confidence score
    # result: {winow_0: {{[2356, 3332, 2345, 3434, ...], Most frequent Topic Id, confidence score},{[12334], mf topic id, score}} , window_1: {...}, ...}
    # every time there is 1 or [threshold] patents spanning between two communities/between two topics, the knowledge has been recombined.
    # Diffusion is how long these communities keep being connected.

    # how do we measure recombination?
    # maybe with overlapping community detection. Every node in an overlap is recombining the knowledge of both communities. Diffusion = how long the overlap lasts.

    # recombination with label propagagtion:
    # community detection with label propagation: recombination occures whenever a patent is linked to at least two communities, or a threshhold of x patents is linked to at least two communities.
    # Problem with labelpropagation with weights: this middle patent(s) would always take the label of the community it has a stronger bond to. it is very very unlikely, that these nodes have same weights on the outgoing ties
    # and even then it would randomly be labeled with one of the two labels. label propagation seems to be not fitting. It might be still prossible to measure diffusion by measuring how long
    # a community is alive/grows.

    # further community detection:  https://python-louvain.readthedocs.io/en/latest/
    #                               https://python-louvain.readthedocs.io/en/latest/api.html
    # and more