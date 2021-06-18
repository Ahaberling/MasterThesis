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
    #print(topicSim['window_0'])
    #print(topicSim['window_0'].nodes())
    #print(topicSim['window_0'].edges())

    ### Label Propagation       # nx.algorithms.community.label_propagation.asyn_lpa_communities

    lp_commu = {}

    for window_id, window in topicSim.items():

        lp = nx.algorithms.community.label_propagation.asyn_lpa_communities(window, weight='weight')
        #for i in lp:
            #print(i)

        #print('+++++++++++++++++++++++++++++++++++++++++')

        lp_commu[window_id] = lp

    print(lp_commu)
    print(lp_commu['window_0'])
    for i in lp_commu['window_0']:
        print(i)


    # non-overlapping:
    # label propagation weighted
    # leiden weighted (what is 'initial_membership' parameter? Important?)
    # walk trap (seems to be implemented only for unweigthed graphs -> rethink bipartite link creation of only taking the 3 most prominent topics)

    # overlapping:
    # kclique   # can be used on weighted, but not sure if implemented here
    # wCommunity    (weighted but have to reread because of parameters
    # lais2 # relies on density function. reread. seems not implemented to consider weights




#---------------------------------------

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

    # label propagation:
    # compute communities for time t, evaluate density of communities, then at t+1 check if there are patents linked to two communities (threshold based). These patents
    # recombine knowledge. for diffusion: check for how many t's this thereshold is met.

    # label propagation:
    # we probably just take all edges, not only the top 3 (in the bipartite stage) and apply this weighted label propagation. and then look at the connection patens at t+1.
    # denisty and most frequent topics in these clusters can be used to evaluate the preformance of the label propagation.

    # Louvain:
    # Do the same
    # look at resulting partition. not clear yet, if connection patens at t can be view, or if we have to check t+1 again
    # lovain and modularity in general only good for few big communties (https://towardsdatascience.com/community-detection-algorithms-9bd8951e7dae)
    # Better? : (many small communities)
    #from cdlib import algorithms
    # import networkx as nx
    # G = nx.karate_club_graph()
    # coms = algorithms.surprise_communities(G)

    # Leiden: (https://towardsdatascience.com/community-detection-algorithms-9bd8951e7dae)
    # from cdlib import algorithms
    # import networkx as nx
    # G = nx.karate_club_graph()
    # coms = algorithms.leiden(G)

    # Walktrap Community Detection

    # preformance: https://yoyoinwanderland.github.io/2017/08/08/Community-Detection-in-Python/ igraph better then networkx



    # further community detection:  https://python-louvain.readthedocs.io/en/latest/
    #                               https://python-louvain.readthedocs.io/en/latest/api.html
    # and more

    # new approach: overlapping communities:
    # if to communities start to overlap (with the adding of new patents) then these patents building the overlapping recombined knowledge. difussion: how long does the overlap stay.
    # problem: what to do if two overlapping communties become one?

    # maybe put this in futur works

    # Clique Percolation Method
    # not sure if appropraite. maybe duable, but hard to find correct value for k and to find the appropriate number of bipartite links.
    # Weighted version possible, but additional threshold has to be set.

    # lfm
    # cool, but is it implemented with weights??

    # big_clam
    # assuming high density in overlapping communities. is this resonable for us?

    # slpa
    # not weighted i think. build on lablepropa

    # lais2
    # not weighted i think but seems cool, no parameters

    # wCommunity
    # weighted! seems interesting

    # applying: kclique, wCommunity, lais2