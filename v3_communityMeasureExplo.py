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

        lp_commu[window_id] = list(lp)

    #print(lp_commu)
    #print(lp_commu['window_0'])
    #for i in lp_commu['window_0']:          # {291407609} {290720124, 290720623}
        #print(i)

    for i in range(0, len(lp_commu)-1):

        #print(lp_commu['window_{0}'.format(i)])
        all_ids_t = lp_commu['window_{0}'.format(i)]

        all_ids_t = [item for sublist in all_ids_t for item in sublist]

        #print(all_ids_t)

        #print(len(all_ids_t))
        #print(len(np.unique(all_ids_t)))
        #print(len(all_ids_t) == len(np.unique(all_ids_t)))


    # 0. delete all communities that are not a least of size x (for now 2)
    # 0.5 take communities and calculate the most relevant topic/s in them, with frequency

    # 1. take normal sliding window
    # 2. see which patents are new every t
    # 3. see if the new patents connects to patents of overall two or more different communities at t-1
    # 4. if some, save window_id, bridging patent, community endpoint 1, community topics 1, community endpoint 2, community topics 2,
    # 5.  check if the number of bridging patents between two communities meets a certain threshold ( x% of all new papers or x% of community size 1 + community size 2, or the average of the two)
    # Measure of diffusion a lot more difficult. One would have to follow the communities over t. if it is possible to identify communities that are stable over t, then
    # it would be possible to assess for how many t's a threshold of bridging patents is meet. This seems too be to much for my thesis. Alternatively one could check how many citations
    # a bridging patent gets over time, but this is contrary to the reason why we are doing it with topics in the first place, and only a lazy approximation

    # Overlapping communities:

    # same approach
    # calculate communities for every window
    # check if there are overlapping communities. check if they meet a threshold (x% of all new patets vs x% of community sizes) if so then these patents that are overlapping are recombining information. Same limitation for diffusion: to evaluate how long the threshold is meet
    # one has to establish the consistency / stability of communtiies over time.

    # how to check stability over time of communities:
    # 0. establish core of community (node with most internal connection within the community) (one core for each window and each communitiy)
    # if the core drops out, then look for the second/third/... most connected node in the network, that exists in t+1
    # 1. how many patents that left the community also left the graph in general
    # 2. how many patents that joined the community also joined the graph in general
    #
    # 3. Can I develop a stability measure when doing this for all communities (how much movement is there between communities)
    # is there a already established measure i can use? I guess smth like modularity onyl talks about the quality of the partition, not about the sbaility over time.



    '''
    #test = [x for x in lp_commu['window_0']]
    #print(test)

    x = {290720124}
    #rnd_bool = x in lp_commu['window_0']
    #rnd_bool = x.issubset(lp_commu['window_0'])

    test = [x for set in lp_commu['window_0'] if x.issubset(set)]

    print(test)
    '''


    # is the similarity between patens changing over time? No, because chnage in the network is only caused by added patents. these patents instantly bring their
    # topic affiliatio. this affiliation is not changing over time, since the abstract and the lda model is not changing. this leads to no change in the patent similarity as well.
    # this means that all networks are not dynamic, expect for the topic occurance network.

    #print(rnd_bool)

    #s = (val for val in range(10))
    #print(1 in s)

    #x = {range(10) }
    #print(x)
    '''
    if  in lp_commu['window_0']:
        print('yeees')
    else:
        print('noooo')

    print(lp_commu['window_0'])
    '''
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