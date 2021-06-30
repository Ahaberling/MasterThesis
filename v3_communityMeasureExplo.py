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
    topics = pd.read_csv('patent_topics_mallet.csv', quotechar='"', skipinitialspace=True)
    parent = pd.read_csv('cleaning_robot_EP_backward_citations.csv', quotechar='"', skipinitialspace=True)

    patent_lda_ipc = patent_lda_ipc.to_numpy()
    topics = topics.to_numpy()
    parent = parent.to_numpy()

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

#--- Applying Community detection to each graph/window and populate respective dictionaries ---#

    ### Creating dictionaries to save communities ###

    # suitable algorithms - crisp #
    lp_commu = {}                           # Label Propagation (networkx)
    greedy_modularity_commu = {}            #


    # unsuitable algorithms - crip #
    '''
    lp2_commu = {}                          # Label Propagation (cdlib) | no weight consideration
    leiden_commu = {}                       #                           | need connected graph 
    walktrap_commu = {}                     #                           | need connected graph 
    eigenvector_commu = {}                  #                           | need connected graph 
    spinglass_commu = {}                    #                           | need connected graph 

    gdmp2_commu = {}                        #                           | does not procude communities
    paris_commu = {}                        #                           | does not seem to work porperly #todo check why?

    sbm_dl_commu = {}                       #                           | need GraphTool
    sbm_dl_nested_commu = {}                #                           | need GraphTool
    infomap_commu = {}                      #                           | needs Linux (I assume)
    aslpaw_commu = {}                       #                           | needs Linux (I assume)
    '''

    # suitable algorithms - overlapping #
    kclique_commu = {}
    lais2_commu = {}
    # try lfm as well if desperate

    # unsuitable algorithms - overlapping #
    '''
    wCommunity_commu = {}                   #                           | need connected graph OR nodes with at least one degree
    '''

    ### Applying Community Detection and filling dictionaries ###

    c = 0

    #todo place seeds if possible


    pbar = tqdm.tqdm(total=len(topicSim))
    for window_id, window in topicSim.items():

        # suitable algorithms - crisp #
        lp = nx.algorithms.community.label_propagation.asyn_lpa_communities(window, weight='weight')
        #greedy_modularity = algorithms.greedy_modularity(window)    #, weight='weight')
        #paris = algorithms.paris(window)


        lp_commu[window_id] = list(lp)
        #greedy_modularity_commu[window_id] = greedy_modularity.to_node_community_map()
        #paris_commu[window_id] = paris.to_node_community_map()


        # unsuitable algorithms - crisp #
        '''
        lp2 = algorithms.label_propagation(window) # (no weight)
        leiden = algorithms.leiden(window)
        walktrap = algorithms.walktrap(window)
        eigenvector = algorithms.eigenvector(window)
        spinglass = algorithms.spinglass(window)
        gdmp2 = algorithms.gdmp2(window)
        sbm_dl = algorithms.sbm_dl(window) 
        sbm_dl_nested = algorithms.sbm_dl_nested(window)
        aslpaw = algorithms.aslpaw(window) 
        infomap = algorithms.infomap(window) 
        
        lp2_commu[window_id] = lp2.to_node_community_map()
        leiden_commu[window_id] = leiden.to_node_community_map()
        walktrap_commu[window_id] = walktrap.to_node_community_map()
        eigenvector_commu[window_id] = eigenvector.to_node_community_map()  
        spinglass_commu[window_id] = spinglass.to_node_community_map()  
        gdmp2_commu[window_id] = gdmp2.to_node_community_map()
        sbm_dl_commu[window_id] = sbm_dl.to_node_community_map()  
        sbm_dl_nested_commu[window_id] = sbm_dl_nested.to_node_community_map()
        aslpaw_commu[window_id] = aslpaw.to_node_community_map()  
        infomap_commu[window_id] = infomap.to_node_community_map()
        '''

        # suitable algorithms - overlapping #
        #kclique = algorithms.kclique(window, k = 3)
        #lais2 = algorithms.lais2(window)

        #kclique_commu[window_id] = kclique.to_node_community_map()
        #lais2_commu[window_id] = lais2.to_node_community_map()

        # unsuitable algorithms - overlapping #
        '''
        wCommunity = algorithms.wCommunity(window, weightName='weight')  # min_bel_degree=0.6, threshold_bel_degree=0.6)
        
        wCommunity_commu[window_id] = wCommunity.to_node_community_map()
        '''

        pbar.update(1)

        c = c + 1
        if c >= 25:
            break


        # {'window_0': defaultdict(<class 'list'>, {288766563: [0, 3], 288803376: [0], 288819596: [0],
        # 290076304: [0, 1, 3, 5], 290106123: [0, 1, 3, 5], 290234572: [0], 291465230: [0, 1, 3, 5],
        # 289730801: [1], 290720988: [1], 290011409: [2], 290122867: [2], 290720623: [2], 290787054: [2],
        # 289643751: [4, 6], 291383952: [4, 6], 291793181: [4, 6], 293035547: [4], 290844808: [7], 291727396: [7],
        # 290373076: [8], 291482760: [8], 289802971: [9], 290768405: [9], 289649697: [10], 290146627: [10],
        # 290721004: [11], 290721071: [11], 289859057: [12], 290348470: [12], 288878152: [13], 289989447: [13],
        # ...
        # 291407609: [42], 290844922: [43], 290720124: [44]}), 'window_30': ...

        # change data structure.
        # filter every community of size 1 out (or even of size 1 and 2, but that will require more effort)
        # construct similar list containing recombinations

        # problem: some graphs are not connected:   'networkx.exception.AmbiguousSolution: Disconnected graph: Ambiguous solution for bipartite sets.'
        # solution 1: take not top 3 edges for bipartite graphs but all
        # if not working, take biggest component? Probably not so cool..
        # Take different cd algorithms that can handle disconnectedness?
        # keep in mind for overlapping community detection as well



    ### Transform data structure ###

    # greedy_modularity #
    greedy_modularity_commu_transf = {}

    for window_id, window in greedy_modularity_commu.items():

        community_list = []
        focal_commu = []
        c = 0

        for patent_id, community_id in window.items():

            if community_id[0] == c:
                focal_commu.append(patent_id)

            else:
                community_list.append(focal_commu)
                focal_commu = []
                focal_commu.append(patent_id)
                c = c + 1

        greedy_modularity_commu_transf[window_id] = community_list

    '''
    # paris #
    paris_commu_transf = {}

    for window_id, window in paris_commu.items():

        number_communities = window[list(window.keys())[-1]]

        community_list = []
        focal_commu = []
        c = 0

        for patent_id, community_id in window.items():

            if community_id[0] == c:
                focal_commu.append(patent_id)

            else:
                community_list.append(focal_commu)
                focal_commu = []
                focal_commu.append(patent_id)
                c = c + 1

        paris_commu_transf[window_id] = community_list
    '''

    # lais2 #

    lais2_commu_transf = {}

    for window_id, window in lais2_commu.items():

        community_list = []
        max_commu_counter = []

        for patent_id, community_id in window.items():
            max_commu_counter.append(len(community_id))

        max_commu_counter = max(max_commu_counter)

        for j in range(max_commu_counter+1):
            focal_commu = []

            for patent_id, community_id in window.items():

                if j in community_id:
                    focal_commu.append(patent_id)

            community_list.append(focal_commu)

        lais2_commu_transf[window_id] = community_list



    # kclique #

    kclique_commu_transf = {}

    for window_id, window in kclique_commu.items():

        community_list = []
        max_commu_counter = []

        for patent_id, community_id in window.items():
            max_commu_counter.append(len(community_id))

        if len(max_commu_counter) >= 1:
            max_commu_counter = max(max_commu_counter)


            for j in range(max_commu_counter+1):
                focal_commu = []

                for patent_id, community_id in window.items():

                    if j in community_id:
                        focal_commu.append(patent_id)

                community_list.append(focal_commu)

        else:
            community_list.append([])

        kclique_commu_transf[window_id] = community_list



    # First I need stbale communities and a way to identify tem over time
    # then I can transform lais2_commu['window_X'] into list of recombinations, e.g.:
    # 288803376: [0]            -> out
    # 288766563: [0, 3]         -> stay same
    # 290106123: [0, 1, 3, 5]   - 290106123: [0, 1], 290106123: [0, 3], 290106123: [0, 5], 290106123: [1, 3], 290106123: [1, 5], 290106123: [3, 5]
    # then I can index by recombination and list patent_ids that recombine afterwards
    # then I can calculate if thresholds are meet


    ### Clean Communties (if necessary) ###


    lp_commu_clean ={}

    for window_id, window in lp_commu.items():
        lp_commu_clean[window_id] = [x for x in window if len(x) >= 3]

    #print(lp_commu_clean)       # {'window_0': [{288766563, 290106123, 291465230, 290076304, 289730801, 290720988}, {288803376, 290234572, 288819596}, {291383952, ...


    greedy_modularity_commu_clean = {}

    for window_id, window in greedy_modularity_commu_transf.items():
        greedy_modularity_commu_clean[window_id] = [x for x in window if len(x) >= 3]

    #print(greedy_modularity_commu_clean)       # {'window_0': [[288766563, 290106123, 288819596, 290234572, 291465230, 290076304, 288803376, 289730801, 290720988], [291383952, ...

    '''
    paris_commu_clean = {}

    for window_id, window in paris_commu_transf.items():
        paris_commu_clean[window_id] = [x for x in window if len(x) >= 3]

    '''

    lais2_commu_clean = {}

    for window_id, window in lais2_commu_transf.items():
        lais2_commu_clean[window_id] = [x for x in window if len(x) >= 3]

    lais2_commu_clean = kclique_commu_transf # k already discriminates community size
    '''
    kclique_commu_clean = {}

    for window_id, window in kclique_commu_transf.items():
        kclique_commu_clean[window_id] = [x for x in window if len(x) >= 3]
    '''


#--- Stability of Communities ---#

    lp_commu_topK = {}

    for i in range(0, len(lp_commu_clean)-1):
        #print('i - lp')
        lp_window = lp_commu_clean['window_{0}'.format(i*30)]
        #print('i')
        #print(topicSim['window_{0}'.format(i*30)].nodes())
        #print('i+1')
        #print(topicSim['window_{0}'.format((i+1)*30)].nodes())

        #print(topicSim[window_id].degree(288766563))


        #print(next(topicSim[window_id]))

        surviver_window = []

        for community in lp_window:
            #print(community)

            suriviver =  []

            for patent in community:
                #if patent in topicSim['window_{0}'.format((i+1)*30)]:                  # Without this, the name 'surviver' is not really fitting anymore
                    #suriviver.append((patent, topicSim['window_{0}'.format(i*30)].degree(patent) ))

                suriviver.append((patent, topicSim['window_{0}'.format(i * 30)].degree(patent)))    # Here we take the overall degree, not the degree restricted to
                                                                                                    # nodes in the community. This is due to the assumption that most
                                                                                                    # high degree labeled to be in a community also have the most edges
                                                                                                    # to nodes in this community. This assumption can be falsified later.
                                                                                                    # Later on not only the degree, but rather the sum of edges weighes
                                                                                                    # might be used to find this core node of the community.
                                                                                                    # This approach might be extended to consider not only the top degree
                                                                                                    # node as core, but the top k degree nodes.

            suriviver.sort(key=operator.itemgetter(1), reverse=True)
            suriviver_topK = suriviver[0:1]
            #print(suriviver)
            #print(suriviver_topK)
            surviver_window.append(suriviver_topK)

        #print('surviver_window')
        #print(surviver_window)

        lp_window = lp_commu_clean['window_{0}'.format(i * 30)]
        #print(lp_window)
        communities_plusTopK = []

        for j in range(len(lp_window)):
            #print(lp_window[j])
            communities_plusTopK.append([lp_window[j], surviver_window[j]])

        lp_commu_topK['window_{0}'.format(i * 30)] = communities_plusTopK
        #print(lp_commu_topK['window_{0}'.format(i * 30)])



    ### Community Labeling ###

    # Assumption: highly connected nodes in (/within) a community are somewhat stable parts of communities. We assume that
    # these high degree nodes are not randomly changing community affiliation in the clustering algorithms, and even are
    # propably the least likely and thereby the least nodes that leave the community


    #for each window (array x-axis)
    #   if not first row:
    #       for each column (community id)
    #           if topk of above column is in a current community
    #               insert topk as community id
    #           elif highest surving of above topk community in a current community
    #               look for highest suring that is in a solo community
    #                   insert topk as community id
    #           else (if no nodes appears anymore)                                                     # This means: if a community is dying and on the last tick, at least one patent switches the community, than this counts as merging (which is ok, i guess. At least arguable)
    #               insert 0
    #   for each community in dic window
    #       if topk not in array window
    #          open new community
    #


    max_number_commu = 0
    for window_id, window in lp_commu_topK.items():
        for community in window:
            max_number_commu = max_number_commu+1

    community_tracing_array  = np.zeros((len(topicSim), max_number_commu))      # this is the max columns needed for the case that no community is tracable
    #print(np.shape(community_tracing_array))

    for row in range(len(community_tracing_array)):

        if row != 0:
            prev_window = lp_commu_topK['window_{0}'.format(row - 1 * 30)]
            current_window = lp_commu_topK['window_{0}'.format(row * 30)]

            for column in range(len(community_tracing_array.T)):

                prev_topk = community_tracing_array[row-1, column]
                topk_candidate = [community[1][0][0] for community in current_window if prev_topk in community[0]]

                if len(topk_candidate) == 1:
                    community_tracing_array[row, column] = topk_candidate

                else:                                                           # (e.g. 0 because the node disappears or 2 because it is in two communities)
                    community_candidate = [community[0] for community in prev_window if prev_topk in community[0]]

                    if len(community_candidate) >= 2:
                        community_candidate =   # take the bigger one
                                                # alternative: take the one for which prev_topk has most edges in or biggest edge weight sum in

                    candidate_list = []
                    for candidate in community_candidate:

                        candidate_list.append((candidate, topicSim['window_{0}'.format(row-1 * 30)].degree(candidate)))

                    candidate_list.sort(key=operator.itemgetter(1), reverse=True)

                    for degree_candidate in candidate_list:

                        next_topk_candidate = [community[1][0][0] for community in current_window if degree_candidate[0] in community[0]]

                        if len(next_topk_candidate) ==1:
                            community_tracing_array[row, column] = topk_candidate
                            break







    print('sd')
    print(lp_commu_topK['window_{0}'.format(0 * 30)])
    if 288766563 in lp_commu_topK['window_{0}'.format(0 * 30)]:
        print('yes')
    else:
        print('no')

    x = [item[1][0][0] for item in lp_commu_topK['window_{0}'.format(0 * 30)] if 288766563 in item[0]]
    print(x)

    '''
    lp_commu_labeled = {}
    topk_label_list = []
    available_ids = range(1000000)

    print(lp_commu_topK)

    for i in range(0, len(lp_commu_topK)-1):
        lp_window = lp_commu_topK['window_{0}'.format(i * 30)]
        #print('lp_window')
        #print(lp_window)

        for community in lp_window:
            #print(community)
            #print(community[1])

            new_community_threshold = 0

            for topK_patent in community[1]:
                #print(topK_patent[0])

                if topK_patent in topk_label_list:
                    1+1

                else:
                    new_community_threshold = new_community_threshold + 1

                    if

            if new_community_threshold == len(community[1]):
                community_id = available_ids.pop(0)

                for topK_patent in community[1]:
                    topk_label_list.append((topK_patent, community_id))
    '''




#--- Recombination - crisp ---# (semi cool, because no idea of communities are stable, yet)

    # label propagation #
    lp_window_all_ids = {}

    for i in range(0, len(lp_commu)-1):

        all_ids_t = lp_commu['window_{0}'.format(i*30)]
        all_ids_t = [item for sublist in all_ids_t for item in sublist]
        lp_window_all_ids['window_{0}'.format(i * 30)] = all_ids_t

    lp_recombination_dic = {}

    for i in range(0, len(lp_window_all_ids)-2):
        t = set(lp_window_all_ids['window_{0}'.format(i * 30)])
        t_plus1 = set(lp_window_all_ids['window_{0}'.format((i+1) * 30)])

        new_patents = t_plus1.difference(t)

        window_list = []

        for patent in new_patents:

            neighbors = list(topicSim['window_{0}'.format((i+1) * 30)].neighbors(patent))

            patent_list = []

            if len(neighbors) >=2:

                bridge_list = []
                already_found_community = []

                for neighbor in neighbors:

                    for community in lp_commu_clean['window_{0}'.format((i+1) * 30)]:

                        if set([neighbor]).issubset(community):
                            if community not in already_found_community:
                                bridge_list.append(neighbor)
                                already_found_community.append(community)

                if len(bridge_list) >= 2:
                    patent_list.append(bridge_list)

            if len(patent_list) != 0:
                window_list.append([patent, patent_list])

        lp_recombination_dic['window_{0}'.format((i + 1) * 30)] = window_list # list of all patents that recombine  [[patent, [neighbor, neighbor]],...]
    #print(lp_recombination_dic)    # {'window_30': [], 'window_60': [], ...,  'window_300': [[287657442, [[287933459, 290076304]]], ...




    # greedy_modularity #
    gm_window_all_ids = {}

    for i in range(0, len(greedy_modularity_commu_transf)-1):

        all_ids_t = greedy_modularity_commu_transf['window_{0}'.format(i*30)]
        all_ids_t = [item for sublist in all_ids_t for item in sublist]
        gm_window_all_ids['window_{0}'.format(i * 30)] = all_ids_t

    gm_recombination_dic = {}

    for i in range(0, len(gm_window_all_ids)-2):
        t = set(gm_window_all_ids['window_{0}'.format(i * 30)])
        t_plus1 = set(gm_window_all_ids['window_{0}'.format((i+1) * 30)])

        new_patents = t_plus1.difference(t)

        window_list = []

        for patent in new_patents:

            neighbors = list(topicSim['window_{0}'.format((i+1) * 30)].neighbors(patent))

            patent_list = []

            if len(neighbors) >=2:

                bridge_list = []
                already_found_community = []

                for neighbor in neighbors:

                    for community in greedy_modularity_commu_clean['window_{0}'.format((i+1) * 30)]:

                        if set([neighbor]).issubset(community):
                            if community not in already_found_community:
                                bridge_list.append(neighbor)
                                already_found_community.append(community)
                                #print(bridge_list)

                if len(bridge_list) >= 2:
                    patent_list.append(bridge_list)

            if len(patent_list) != 0:
                window_list.append([patent, patent_list])

        gm_recombination_dic['window_{0}'.format((i + 1) * 30)] = window_list # list of all patents that recombine  [[patent, [neighbor, neighbor]],...]
    #print(gm_recombination_dic)    # {'window_30': [], 'window_60': [], 'window_90': [], 'window_120': [], ...          all empty :(



# --- Recombination Thrshold  - crisp ---# (semi cool, because no idea of communities are stable, yet)

    # label propagation #
    for window_id, window in lp_recombination_dic.items():

        threshold_meet = 0       # not meet

        if len(window) != 0:
            value = len(window) / len(topicSim[window_id])

            if value >= 0.05:
                threshold_meet = 1

        lp_recombination_dic[window_id].append(threshold_meet)

    #print(lp_recombination_dic)    # {'window_30': [0], 'window_60': [0], 'window_90': [0], 'window_120': [0], 'window_150': [0],
                                #  'window_180': [0], 'window_210': [0], 'window_240': [0], 'window_270': [0],
                                # 'window_300': [[287657442, [[287933459, 290076304]]], ...

    # label propagation #
    for window_id, window in gm_recombination_dic.items():

        threshold_meet = 0  # not meet

        if len(window) != 0:
            value = len(window) / len(topicSim[window_id])
            # This can be done relative to community size instead of relative to overall size, but latter makes more sense for me right now

            if value >= 0.05:
                threshold_meet = 1

        gm_recombination_dic[window_id].append(threshold_meet)

    #print(gm_recombination_dic)  # {'window_30': [0], 'window_60': [0], 'window_90': [0], ...


#--- Recombination - overlapping ---# (semi cool, because no idea of communities are stable, yet)





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

    #--------------------------------






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

        # applying lable propa networkx weighted, leiden, walktrap
        # applying: kclique, wCommunity, lais2