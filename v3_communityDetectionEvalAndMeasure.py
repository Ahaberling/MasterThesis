if __name__ == '__main__':

#--- Import Libraries ---#
    print('\n#--- Import Libraries ---#\n')

    import pandas as pd
    import numpy as np
    import pickle as pk

    import networkx as nx
    from cdlib import algorithms
    #import wurlitzer                   #not working for windows

    import tqdm
    import itertools
    import operator
    import os

#--- Initialization ---#
    print('\n# --- Initialization ---#\n')

    os.chdir('D:/Universitaet Mannheim/MMDS 7. Semester/Master Thesis/Outline/Data/Cleaning Robots')

    patent_lda_ipc = pd.read_csv('patent_lda_ipc.csv', quotechar='"', skipinitialspace=True)
    #topics = pd.read_csv('patent_topics_mallet.csv', quotechar='"', skipinitialspace=True)
    #parent = pd.read_csv('cleaning_robot_EP_backward_citations.csv', quotechar='"', skipinitialspace=True)

    patent_lda_ipc = patent_lda_ipc.to_numpy()
    #topics = topics.to_numpy()
    #parent = parent.to_numpy()

    '''
    with open('windows_bipartite', 'rb') as handle:
        bipartite = pk.load(handle)
    '''
    '''
    with open('windows_topicOccu', 'rb') as handle:
        topicOccu = pk.load(handle)
    '''

    with open('windows_topicSim', 'rb') as handle:
        topicSim = pk.load(handle)

    with open('windows_lp_communities', 'rb') as handle:
        lp_communities = pk.load(handle)

    with open('windows_gm_communities', 'rb') as handle:
        gm_communities = pk.load(handle)
    '''
    with open('windows_lais2_communities', 'rb') as handle:
        lais2_communities = pk.load(handle)

    with open('windows_kclique_communities', 'rb') as handle:
        kclique_communities = pk.load(handle)

    '''

    print(topicSim['window_0'])
    print(topicSim['window_0'].nodes())


#--- Recombination - crisp ---#

    # dont take the community dicts for this part below. we want all ids in the window, not only the id that are new in the window!

    # label propagation #
    '''
    lp_window_all_ids = {}

    for i in range(0, len(topicSim)-1):

        all_ids_t = topicSim['window_{0}'.format(i*30)]
        all_ids_t = [community[0] for community in all_ids_t]
        all_ids_t = [item for sublist in all_ids_t for item in sublist]

        lp_window_all_ids['window_{0}'.format(i * 30)] = all_ids_t
    '''

    lp_recombination_dic = {}

    for i in range(0, len(topicSim)-2):
        #t = set(lp_window_all_ids['window_{0}'.format(i * 30)])
        t = set(topicSim['window_{0}'.format(i*30)])
        #t_plus1 = set(lp_window_all_ids['window_{0}'.format((i+1) * 30)])
        t_plus1 = set(topicSim['window_{0}'.format((i+1)*30)])

        new_patents = t_plus1.difference(t)

        window_list = []

        for patent in new_patents:

            neighbors = list(topicSim['window_{0}'.format((i+1) * 30)].neighbors(patent))

            patent_list = []

            if len(neighbors) >=2:

                bridge_list = []
                already_found_community = []

                for neighbor in neighbors:

                    for community in lp_communities['window_{0}'.format((i+1) * 30)]:
                        #community = community[0]
                        if set([neighbor]).issubset(community[0]):
                            if community not in already_found_community:
                                #community_id =
                                bridge_list.append((neighbor, community[1][0]))
                                #bridge_list.append(neighbor)
                                already_found_community.append(community)

                if len(bridge_list) >= 2:
                    patent_list.append(bridge_list)
                    #patent_list.append(list(itertools.combinations(bridge_list, r=2)))
                    #patent_list.append(list(itertools.combinations(bridge_list, r=3)))



            if len(patent_list) != 0:
                #window_list.append([patent, patent_list])
                patent_list_comb = list(itertools.combinations(patent_list[0], r=2))
                for comb in patent_list_comb:
                    window_list.append([patent, comb])

        lp_recombination_dic['window_{0}'.format((i + 1) * 30)] = window_list # list of all patents that recombine  [[patent, [neighbor, neighbor]],...]
    #print(lp_recombination_dic)    # {'window_30': [], 'window_60': [], ...,  'window_300': [[287657442, [[287933459, 290076304]]], ...




    # greedy_modularity #

    '''
    gm_window_all_ids = {}

    for i in range(0, len(greedy_modularity_commu_transf)-1):

        all_ids_t = greedy_modularity_commu_transf['window_{0}'.format(i*30)]
        all_ids_t = [item for sublist in all_ids_t for item in sublist]
        gm_window_all_ids['window_{0}'.format(i * 30)] = all_ids_t
    '''
    gm_recombination_dic = {}

    for i in range(0, len(topicSim) - 2):
        t = set(topicSim['window_{0}'.format(i * 30)])
        t_plus1 = set(topicSim['window_{0}'.format((i + 1) * 30)])

        new_patents = t_plus1.difference(t)

        window_list = []

        for patent in new_patents:

            neighbors = list(topicSim['window_{0}'.format((i + 1) * 30)].neighbors(patent))

            patent_list = []

            if len(neighbors) >= 2:

                bridge_list = []
                already_found_community = []

                for neighbor in neighbors:

                    for community in gm_communities['window_{0}'.format((i + 1) * 30)]:
                        # community = community[0]
                        if set([neighbor]).issubset(community[0]):
                            if community not in already_found_community:
                                # community_id =
                                bridge_list.append((neighbor, community[1][0]))
                                # bridge_list.append(neighbor)
                                already_found_community.append(community)

                if len(bridge_list) >= 2:
                    patent_list.append(bridge_list)
                    # patent_list.append(list(itertools.combinations(bridge_list, r=2)))
                    # patent_list.append(list(itertools.combinations(bridge_list, r=3)))

            if len(patent_list) != 0:
                # window_list.append([patent, patent_list])
                patent_list_comb = list(itertools.combinations(patent_list[0], r=2))
                for comb in patent_list_comb:
                    window_list.append([patent, comb])

        gm_recombination_dic['window_{0}'.format((i + 1) * 30)] = window_list  # list of all patents that recombine  [[patent, [neighbor, neighbor]],...]
    #print(gm_recombination_dic)  # {'window_30': [], 'window_60': [], ...,  'window_300': [[287657442, [[287933459, 290076304]]], ...



# --- Recombination Thrshold  - crisp ---#

    # label propagation #
    for window_id, window in lp_recombination_dic.items():

        threshold_meet = 0       # 0 = not meet | 1 = meet

        if len(window) != 0:
            value = len(window) / len(topicSim[window_id])

            if value >= 0.05:
                threshold_meet = 1

        lp_recombination_dic[window_id].append(threshold_meet)

    #print(lp_recombination_dic)    # {'window_30': [0], 'window_60': [0], 'window_90': [0], 'window_120': [0], 'window_150': [0],
                                #  'window_180': [0], 'window_210': [0], 'window_240': [0], 'window_270': [0],
                                # 'window_300': [[287657442, [[287933459, 290076304]]], ...

    # greedy modularity #
    for window_id, window in gm_recombination_dic.items():

        threshold_meet = 0  # not meet

        if len(window) != 0:
            value = len(window) / len(topicSim[window_id])
            # This can be done relative to community size instead of relative to overall size, but latter makes more sense for me right now

            if value >= 0.05:
                threshold_meet = 1

        gm_recombination_dic[window_id].append(threshold_meet)

    #print(gm_recombination_dic)  # {'window_30': [0], 'window_60': [0], 'window_90': [0], ...



# --- Recombination - overlapping ---# (semi cool, because no idea of communities are stable, yet)


# --- community stability ---# (ignored for now, because it is more suitbale to try this with good networks)

'''
c = 0
for window_id, window in lp_commu_clean.items():
    print(len(topicSim[window_id]))
    for community in window:
        degree_list = []
        for patent in community:
            degree_list.append((patent, topicSim[window_id].degree[patent]))
        print(degree_list)
    print('------')
    c = c +1
    if c ==4:
        break
'''

# --------------------------------


# 0. delete all communities that are not a least of size x (for now 2)
# 0.5 take communities and calculate the most relevant topic/s in them, with frequency

# 1. take normal sliding window
# 2. see which patents are new every t
# 3. see if the new patents connects to patents of overall two or more different communities at t-1
# 4. if some, save window_id, bridging patent, community endpoint 1, community topics 1, community endpoint 2, community topics 2,
# 5. check if the number of bridging patents between two communities meets a certain threshold ( x% of all new papers or x% of community size 1 + community size 2, or the average of the two)
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

# why do i want the stability of communities?
# 1. as validation of the community detection algorithm
# 2. as argument. If my communities are stable, then i can try to finding nodes linking between them


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

# print(rnd_bool)

# s = (val for val in range(10))
# print(1 in s)

# x = {range(10) }
# print(x)

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


# ---------------------------------------

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
# from cdlib import algorithms
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

# applying lable propa networkx weighted, leiden, walktrap
# applying: kclique, wCommunity, lais2