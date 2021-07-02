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



# --- Applying Community detection to each graph/window and populate respective dictionaries ---#

    ### Creating dictionaries to save communities ###
    """
    # suitable algorithms - crisp #
    lp_commu = {}  # Label Propagation (networkx)
    greedy_modularity_commu = {}  #


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

    # todo place seeds if possible


    pbar = tqdm.tqdm(total=len(topicSim))
    for window_id, window in topicSim.items():
        # suitable algorithms - crisp #
        lp = nx.algorithms.community.label_propagation.asyn_lpa_communities(window, weight='weight')
        greedy_modularity = algorithms.greedy_modularity(window)    #, weight='weight')
        # paris = algorithms.paris(window)

        lp_commu[window_id] = list(lp)
        greedy_modularity_commu[window_id] = greedy_modularity.to_node_community_map()
        # paris_commu[window_id] = paris.to_node_community_map()

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
        kclique = algorithms.kclique(window, k = 3)
        lais2 = algorithms.lais2(window)

        kclique_commu[window_id] = kclique.to_node_community_map()
        lais2_commu[window_id] = lais2.to_node_community_map()

        # unsuitable algorithms - overlapping #
        '''
        wCommunity = algorithms.wCommunity(window, weightName='weight')  # min_bel_degree=0.6, threshold_bel_degree=0.6)
    
        wCommunity_commu[window_id] = wCommunity.to_node_community_map()
        '''

        pbar.update(1)

        c = c + 1
        #if c >= 35:
            #break

    pbar.close()

    filename = 'windows_lp_helper'
    outfile = open(filename, 'wb')
    pk.dump(lp_commu, outfile)
    outfile.close()

    filename = 'windows_gm_helper'
    outfile = open(filename, 'wb')
    pk.dump(greedy_modularity_commu, outfile)
    outfile.close()

    filename = 'windows_kclique_helper'
    outfile = open(filename, 'wb')
    pk.dump(kclique_commu, outfile)
    outfile.close()

    filename = 'windows_lais2_helper'
    outfile = open(filename, 'wb')
    pk.dump(lais2_commu, outfile)
    outfile.close()
    """
    #print('------------------------------------------')

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

    with open('windows_lp_helper', 'rb') as handle:
        lp_commu = pk.load(handle)

    with open('windows_gm_helper', 'rb') as handle:
        greedy_modularity_commu = pk.load(handle)

    with open('windows_kclique_helper', 'rb') as handle:
        kclique_commu = pk.load(handle)

    with open('windows_lais2_helper', 'rb') as handle:
        lais2_commu = pk.load(handle)



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

        for j in range(max_commu_counter + 1):
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

            for j in range(max_commu_counter + 1):
                focal_commu = []

                for patent_id, community_id in window.items():

                    if j in community_id:
                        focal_commu.append(patent_id)

                community_list.append(focal_commu)

        else:
            community_list.append([])

        kclique_commu_transf[window_id] = community_list


    #print(lais2_commu_transf, '\n')
    #print(kclique_commu_transf)
    #print(1+1)

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
    #print('++++++++++++++++++++ \n')
    #print(lais2_commu_transf['window_0'])

    lais2_commu_clean = {}

    for window_id, window in lais2_commu_transf.items():
        lais2_commu_clean[window_id] = [x for x in window if len(x) >= 3]

    #kclique_commu_clean  = kclique_commu_transf # k already discriminates community size

    kclique_commu_clean = {}

    for window_id, window in kclique_commu_transf.items():
        kclique_commu_clean[window_id] = [x for x in window if len(x) >= 3]

    #print(kclique_commu_clean)
    #print(1+1)

    '''
    kclique_commu_clean = {}

    for window_id, window in kclique_commu_transf.items():
        kclique_commu_clean[window_id] = [x for x in window if len(x) >= 3]
    '''


    #print(lp_commu_clean['window_0'], '\n')
    #print(lais2_commu_clean['window_0'], '\n')

    #print(lp_commu_clean['window_900'], '\n')
    #print(lais2_commu_clean['window_900'], '\n')


#--- Stability of Communities TopK ---#

    # label prop #
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



    # greedy_modularity #

    gm_commu_clean = greedy_modularity_commu_clean
    gm_commu_topK = {}

    for i in range(0, len(gm_commu_clean) - 1):
        # print('i - lp')
        gm_window = gm_commu_clean['window_{0}'.format(i * 30)]
        # print('i')
        # print(topicSim['window_{0}'.format(i*30)].nodes())
        # print('i+1')
        # print(topicSim['window_{0}'.format((i+1)*30)].nodes())

        # print(topicSim[window_id].degree(288766563))

        # print(next(topicSim[window_id]))

        surviver_window = []

        for community in gm_window:
            # print(community)

            suriviver = []

            for patent in community:
                # if patent in topicSim['window_{0}'.format((i+1)*30)]:                  # Without this, the name 'surviver' is not really fitting anymore
                # suriviver.append((patent, topicSim['window_{0}'.format(i*30)].degree(patent) ))

                suriviver.append((patent, topicSim['window_{0}'.format(i * 30)].degree(
                    patent)))  # Here we take the overall degree, not the degree restricted to
                # nodes in the community. This is due to the assumption that most
                # high degree labeled to be in a community also have the most edges
                # to nodes in this community. This assumption can be falsified later.
                # Later on not only the degree, but rather the sum of edges weighes
                # might be used to find this core node of the community.
                # This approach might be extended to consider not only the top degree
                # node as core, but the top k degree nodes.

            suriviver.sort(key=operator.itemgetter(1), reverse=True)
            suriviver_topK = suriviver[0:1]
            # print(suriviver)
            # print(suriviver_topK)
            surviver_window.append(suriviver_topK)

        # print('surviver_window')
        # print(surviver_window)

        gm_window = gm_commu_clean['window_{0}'.format(i * 30)]
        # print(lp_window)
        communities_plusTopK = []

        for j in range(len(gm_window)):
            # print(lp_window[j])
            communities_plusTopK.append([gm_window[j], surviver_window[j]])

        gm_commu_topK['window_{0}'.format(i * 30)] = communities_plusTopK
        # print(lp_commu_topK['window_{0}'.format(i * 30)])

    '''
    print(lp_commu_topK['window_0'], '\n')
    print(gm_commu_topK['window_0'], '\n')

    print(lp_commu_topK['window_900'], '\n')
    print(gm_commu_topK['window_900'], '\n')
    '''

    # lais2 #
    lais2_commu_topK = {}

    #print('----------------- \n')
    #print(lais2_commu_clean)
    #print(lais2_commu_clean['window_0'])

    for i in range(0, len(lais2_commu_clean) - 1):
        # print('i - lp')
        lais2_window = lais2_commu_clean['window_{0}'.format(i * 30)]
        # print('i')
        # print(topicSim['window_{0}'.format(i*30)].nodes())
        # print('i+1')
        # print(topicSim['window_{0}'.format((i+1)*30)].nodes())

        # print(topicSim[window_id].degree(288766563))

        # print(next(topicSim[window_id]))

        surviver_window = []

        for community in lais2_window:
            # print(community)

            suriviver = []

            for patent in community:
                # if patent in topicSim['window_{0}'.format((i+1)*30)]:                  # Without this, the name 'surviver' is not really fitting anymore
                # suriviver.append((patent, topicSim['window_{0}'.format(i*30)].degree(patent) ))

                suriviver.append((patent, topicSim['window_{0}'.format(i * 30)].degree(
                    patent)))  # Here we take the overall degree, not the degree restricted to
                # nodes in the community. This is due to the assumption that most
                # high degree labeled to be in a community also have the most edges
                # to nodes in this community. This assumption can be falsified later.
                # Later on not only the degree, but rather the sum of edges weighes
                # might be used to find this core node of the community.
                # This approach might be extended to consider not only the top degree
                # node as core, but the top k degree nodes.

            suriviver.sort(key=operator.itemgetter(1), reverse=True)
            suriviver_topK = suriviver[0:1]
            # print(suriviver)
            # print(suriviver_topK)
            surviver_window.append(suriviver_topK)

        # print('surviver_window')
        # print(surviver_window)

        lais2_window = lais2_commu_clean['window_{0}'.format(i * 30)]
        # print(lp_window)
        communities_plusTopK = []

        for j in range(len(lais2_window)):
            # print(lp_window[j])
            communities_plusTopK.append([lais2_window[j], surviver_window[j]])

        lais2_commu_topK['window_{0}'.format(i * 30)] = communities_plusTopK
        #print(lp_commu_topK['window_{0}'.format(i * 30)])


    #print(lp_commu_topK['window_0'], '\n')
    #print(lais2_commu_topK['window_0'], '\n')

    #print(lp_commu_topK['window_900'], '\n')
    #print(lais2_commu_topK['window_900'], '\n')

    # kclique #
    kclique_commu_topK = {}

    # print('----------------- \n')
    # print(lais2_commu_clean)
    # print(lais2_commu_clean['window_0'])

    for i in range(0, len(kclique_commu_clean) - 1):
        # print('i - lp')
        kclique_window = kclique_commu_clean['window_{0}'.format(i * 30)]
        # print('i')
        # print(topicSim['window_{0}'.format(i*30)].nodes())
        # print('i+1')
        # print(topicSim['window_{0}'.format((i+1)*30)].nodes())

        # print(topicSim[window_id].degree(288766563))

        # print(next(topicSim[window_id]))

        surviver_window = []

        for community in kclique_window:
            # print(community)

            suriviver = []

            for patent in community:
                # if patent in topicSim['window_{0}'.format((i+1)*30)]:                  # Without this, the name 'surviver' is not really fitting anymore
                # suriviver.append((patent, topicSim['window_{0}'.format(i*30)].degree(patent) ))

                suriviver.append((patent, topicSim['window_{0}'.format(i * 30)].degree(patent)))  # Here we take the overall degree, not the degree restricted to
                # nodes in the community. This is due to the assumption that most
                # high degree labeled to be in a community also have the most edges
                # to nodes in this community. This assumption can be falsified later.
                # Later on not only the degree, but rather the sum of edges weighes
                # might be used to find this core node of the community.
                # This approach might be extended to consider not only the top degree
                # node as core, but the top k degree nodes.

            suriviver.sort(key=operator.itemgetter(1), reverse=True)
            suriviver_topK = suriviver[0:1]
            # print(suriviver)
            # print(suriviver_topK)
            surviver_window.append(suriviver_topK)

        # print('surviver_window')
        # print(surviver_window)

        kclique_window = kclique_commu_clean['window_{0}'.format(i * 30)]
        # print(lp_window)
        communities_plusTopK = []

        for j in range(len(kclique_window)):
            # print(lp_window[j])
            communities_plusTopK.append([kclique_window[j], surviver_window[j]])

        kclique_commu_topK['window_{0}'.format(i * 30)] = communities_plusTopK
        # print(lp_commu_topK['window_{0}'.format(i * 30)])

    #print(lp_commu_topK['window_0'], '\n')
    #print(kclique_commu_topK['window_0'], '\n')

    #print(kclique_commu_clean)
    #print(kclique_commu_topK)

    #print(lp_commu_topK['window_900'], '\n')
    #print(kclique_commu_topK['window_900'], '\n')

    #print(1+1)


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

    # label prop # 
    max_number_commu = 0
    for window_id, window in lp_commu_topK.items():
        for community in window:
            max_number_commu = max_number_commu+1

    community_tracing_array  = np.zeros((len(topicSim), max_number_commu), dtype=int)      # this is the max columns needed for the case that no community is tracable
    #print(np.shape(community_tracing_array))


    for row in range(0, len(community_tracing_array)-1):

        current_window = lp_commu_topK['window_{0}'.format(row * 30)]

        if row != 0:
            prev_window = lp_commu_topK['window_{0}'.format((row - 1) * 30)]

            for column in range(len(community_tracing_array.T)):

                prev_topk = community_tracing_array[row-1, column]
                topk_candidate = [community[1][0][0] for community in current_window if prev_topk in community[0]]

                if len(topk_candidate) == 1:
                    community_tracing_array[row, column] = topk_candidate[0]

                    '''
                elif len(topk_candidate) >= 2:
                    print(topk_candidate, current_window)
                    '''

                else:                                                           # (e.g. 0 because the node disappears or 2 because it is in two communities)
                    community_candidate = [community[0] for community in prev_window if prev_topk in community[0]]

                    if len(community_candidate) >= 2:
                        community_size, community_candidate = max([(len(x), x) for x in community_candidate])
                        community_candidate = [community_candidate]
                                                # alternative: take the one for which prev_topk has most edges in or biggest edge weight sum in
                    if len(community_candidate) != 0:

                        candidate_list = []
                        for candidate in community_candidate[0]:

                            candidate_list.append((candidate, topicSim['window_{0}'.format((row-1) * 30)].degree(candidate)))

                        candidate_list.sort(key=operator.itemgetter(1), reverse=True)

                        for degree_candidate in candidate_list:

                            next_topk_candidate = [community[1][0][0] for community in current_window if degree_candidate[0] in community[0]]

                            if len(next_topk_candidate) ==1:
                                community_tracing_array[row, column] = next_topk_candidate[0]
                                break


        for community in current_window:

            community_identifier = community[1][0][0]

            if community_identifier not in community_tracing_array[row]:

                for column_id in range(len(community_tracing_array.T)):

                    if sum(community_tracing_array[:, column_id]) == 0:                 # RuntimeWarning: overflow encountered in long_scalars
                    #if len(np.unique(community_tracing_array[:, column_id])) >= 2:     # Takes way longer

                        community_tracing_array[row, column_id] = community[1][0][0]
                        break

                '''
                c = len(community_tracing_array[row])
                for back_column in reversed(community_tracing_array[row]):

                    if back_column != 0:
                        community_tracing_array[row, c] = community[1][0][0]
                        break

                    c = c - 1

                    if c == 0:
                        community_tracing_array[row, c] = community[1][0][0]
                        break
                '''

    # cut array to exlcude non relevant 0 columns


    for i in range(len(community_tracing_array.T)):
        if sum(community_tracing_array[:,i]) == 0:
            cutoff = i
            break

    community_tracing_array = community_tracing_array[0:len(community_tracing_array)-1,0:cutoff]

    # make list with flattened array and take only unique ids

    topk_list = np.unique(community_tracing_array.flatten())[1:]

    # for each id, look in which column the id first appeared

    topk_list_associ = []

    for topk in topk_list:
        candidate_list = []

        if topk == 291465230:
            print(1+1)

        for column in range(len(community_tracing_array.T)):

            if topk in community_tracing_array[:,column]:

                window_pos = np.where(community_tracing_array[:,column] == topk)
                #window_pos = community_tracing_array[:,column].index(topk)

                #window_pos = [i for i, x in enumerate(community_tracing_array[:,column]) if x == topk]



                #print('window_pos')
                #print(window_pos)
                window_pos = max(window_pos[0])
                #print(window_pos)
                window = lp_commu_topK['window_{0}'.format(window_pos * 30)]
                #print(window)
                #print(max([(len(x[0]), x[1][0][0]) for x in window]))
                community_size, community_topk = max([(len(x[0]), x[1][0][0]) for x in window])

                candidate_list.append((column, community_size))

        candidate_list.sort(key=operator.itemgetter(1), reverse=True)


        topk_list_associ.append((topk, candidate_list[-1][0]))



                #topk_list_associ.append((i, j))
                #break

    #print(topk_list_associ)
    #print(lp_commu_topK)
    #print('lp_commu_topK')

    lp_commu_id = {}

    for window_id, window in lp_commu_topK.items():
        new_window = []

        for community in window:
            topk = community[1][0][0]
            community_id = [tuple[1] for tuple in topk_list_associ if tuple[0] == topk]

            new_window.append([community[0], community_id])

        lp_commu_id[window_id] = new_window
    '''
    #print(lp_commu_id)
    #print(topk_list_associ)
    print(lp_commu_id['window_300'])
    print(lp_commu_topK['window_300'], '\n')

    print(lp_commu_id['window_0'])
    print(lp_commu_topK['window_0'])
    print('sdfgerg')
    '''

    filename = 'windows_lp_communities'
    outfile = open(filename, 'wb')
    pk.dump(lp_commu_id, outfile)
    outfile.close()


    # greedy modularity #

    max_number_commu = 0
    for window_id, window in gm_commu_topK.items():
        for community in window:
            max_number_commu = max_number_commu+1

    community_tracing_array  = np.zeros((len(topicSim), max_number_commu), dtype=int)      # this is the max columns needed for the case that no community is tracable
    #print(np.shape(community_tracing_array))


    for row in range(0, len(community_tracing_array)-1):

        current_window = gm_commu_topK['window_{0}'.format(row * 30)]

        if row != 0:
            prev_window = gm_commu_topK['window_{0}'.format((row - 1) * 30)]

            for column in range(len(community_tracing_array.T)):

                prev_topk = community_tracing_array[row-1, column]
                topk_candidate = [community[1][0][0] for community in current_window if prev_topk in community[0]]

                if len(topk_candidate) == 1:
                    community_tracing_array[row, column] = topk_candidate[0]

                    '''
                elif len(topk_candidate) >= 2:
                    print(topk_candidate, current_window)
                    '''

                else:                                                           # (e.g. 0 because the node disappears or 2 because it is in two communities)
                    community_candidate = [community[0] for community in prev_window if prev_topk in community[0]]

                    if len(community_candidate) >= 2:
                        community_size, community_candidate = max([(len(x), x) for x in community_candidate])
                        community_candidate = [community_candidate]
                                                # alternative: take the one for which prev_topk has most edges in or biggest edge weight sum in
                    if len(community_candidate) != 0:

                        candidate_list = []
                        for candidate in community_candidate[0]:

                            candidate_list.append((candidate, topicSim['window_{0}'.format((row-1) * 30)].degree(candidate)))

                        candidate_list.sort(key=operator.itemgetter(1), reverse=True)

                        for degree_candidate in candidate_list:

                            next_topk_candidate = [community[1][0][0] for community in current_window if degree_candidate[0] in community[0]]

                            if len(next_topk_candidate) ==1:
                                community_tracing_array[row, column] = next_topk_candidate[0]
                                break


        for community in current_window:

            community_identifier = community[1][0][0]

            if community_identifier not in community_tracing_array[row]:

                for column_id in range(len(community_tracing_array.T)):

                    if sum(community_tracing_array[:, column_id]) == 0:                 # RuntimeWarning: overflow encountered in long_scalars
                    #if len(np.unique(community_tracing_array[:, column_id])) >= 2:     # Takes way longer

                        community_tracing_array[row, column_id] = community[1][0][0]
                        break

                '''
                c = len(community_tracing_array[row])
                for back_column in reversed(community_tracing_array[row]):

                    if back_column != 0:
                        community_tracing_array[row, c] = community[1][0][0]
                        break

                    c = c - 1

                    if c == 0:
                        community_tracing_array[row, c] = community[1][0][0]
                        break
                '''

    # cut array to exlcude non relevant 0 columns


    for i in range(len(community_tracing_array.T)):
        if sum(community_tracing_array[:,i]) == 0:
            cutoff = i
            break

    community_tracing_array = community_tracing_array[0:len(community_tracing_array)-1,0:cutoff]

    # make list with flattened array and take only unique ids

    topk_list = np.unique(community_tracing_array.flatten())[1:]

    # for each id, look in which column the id first appeared

    topk_list_associ = []

    for topk in topk_list:
        candidate_list = []

        if topk == 291465230:
            print(1+1)

        for column in range(len(community_tracing_array.T)):

            if topk in community_tracing_array[:,column]:

                window_pos = np.where(community_tracing_array[:,column] == topk)
                #window_pos = community_tracing_array[:,column].index(topk)

                #window_pos = [i for i, x in enumerate(community_tracing_array[:,column]) if x == topk]



                #print('window_pos')
                #print(window_pos)
                window_pos = max(window_pos[0])
                #print(window_pos)
                window = gm_commu_topK['window_{0}'.format(window_pos * 30)]
                #print(window)
                #print(max([(len(x[0]), x[1][0][0]) for x in window]))
                community_size, community_topk = max([(len(x[0]), x[1][0][0]) for x in window])

                candidate_list.append((column, community_size))

        candidate_list.sort(key=operator.itemgetter(1), reverse=True)


        topk_list_associ.append((topk, candidate_list[-1][0]))



                #topk_list_associ.append((i, j))
                #break

    #print(topk_list_associ)
    #print(lp_commu_topK)
    #print('lp_commu_topK')

    gm_commu_id = {}

    for window_id, window in gm_commu_topK.items():
        new_window = []

        for community in window:
            topk = community[1][0][0]
            community_id = [tuple[1] for tuple in topk_list_associ if tuple[0] == topk]

            new_window.append([community[0], community_id])

        gm_commu_id[window_id] = new_window

    '''
    #print(lp_commu_id)
    #print(topk_list_associ)
    print(gm_commu_id['window_0'])
    print(gm_commu_topK['window_0'], '\n')

    print(gm_commu_id['window_900'])
    print(gm_commu_topK['window_900'])
    print('sdfgerg')
    '''

    filename = 'windows_gm_communities'
    outfile = open(filename, 'wb')
    pk.dump(gm_commu_id, outfile)
    outfile.close()


    # lais2 #


    max_number_commu = 0
    for window_id, window in lais2_commu_topK.items():
        for community in window:
            max_number_commu = max_number_commu + 1

    community_tracing_array = np.zeros((len(topicSim), max_number_commu),
                                       dtype=int)  # this is the max columns needed for the case that no community is tracable
    # print(np.shape(community_tracing_array))


    for row in range(0, len(community_tracing_array) - 1):

        current_window = lais2_commu_topK['window_{0}'.format(row * 30)]

        if row != 0:
            prev_window = lais2_commu_topK['window_{0}'.format((row - 1) * 30)]

            for column in range(len(community_tracing_array.T)):

                prev_topk = community_tracing_array[row - 1, column]
                topk_candidate = [community[1][0][0] for community in current_window if prev_topk in community[0]]

                if len(topk_candidate) == 1:
                    community_tracing_array[row, column] = topk_candidate[0]

                    '''
                elif len(topk_candidate) >= 2:
                    print(topk_candidate, current_window)
                    '''

                else:  # (e.g. 0 because the node disappears or 2 because it is in two communities)

                    # Limitation: for overlapping algorithms, taking only the node with the higest degree might be not suitable. Looking at this case below, 3 of 5 communities of that
                    # window are merged, even though they might not be the same
                    # todo: does this stability measure even make sense for overlapping? how is diffusion and recombination measured?

                    #print(prev_window)          # lais2:    # [[[288766563, 288803376, 288819596, 290076304, 290106123, 290234572, 291465230], [(291465230, 4)]], [[290076304, 290106123, 291465230, 289730801, 290720988], [(291465230, 4)]], [[290011409, 290122867, 290720623, 290787054], [(290011409, 2)]], [[288766563, 290076304, 290106123, 291465230], [(291465230, 4)]], [[289643751, 291383952, 291793181, 293035547], [(291383952, 3)]]]
                    #print(prev_topk)            # lais2:    # 291465230
                    # what i should get         #           # [[288766563, 288803376, 288819596, 290076304, 290106123, 290234572, 291465230], [290076304, 290106123, 291465230, 289730801, 290720988],[288766563, 290076304, 290106123, 291465230]]

                    #print(prev_window)          # LP:      # [[{288766563, 290106123, 291465230, 290076304, 289730801, 290720988}, [(291465230, 4)]], [{288803376, 288819596, 290234572}, [(288819596, 2)]], [{291383952, 293035547, 291793181, 289643751}, [(291383952, 3)]], [{290011409, 290122867, 290787054, 290720623}, [(290011409, 2)]]]
                    #print(prev_topk)            # LP:      # 291465230
                    #print(community_candidate)  # LP:      # [{288766563, 290106123, 291465230, 290076304, 289730801, 290720988}]

                    #print(prev_window)         # gm:       # [[[290115954, 286499669, 286404823], [(290115954, 2)]], [[289751121, 289458126, 289593087], [(289751121, 2)]]]
                    #print(prev_topk)           # gm:       # 290115954
                    #print(community_candidate) # gm:       # [[290115954, 286499669, 286404823]]

                    community_candidate = [community[0] for community in prev_window if prev_topk in community[0]]
                    #community_candidate2 = [community_candidate]
                    #community_candidate3 = [item for sublist in community_candidate2 for item in sublist]
                    #community_candidate = community_candidate3
                    #print(community_candidate)

                    if len(community_candidate) >= 2:
                        community_size, community_candidate = max([(len(x), x) for x in community_candidate])
                        community_candidate = [community_candidate]
                        # Just taking the biggest community where candidate occures. This might seem problematic, but the real "problem" is described in the to-do
                        # alternative: take the one for which prev_topk has most edges in or biggest edge weight sum in
                    if len(community_candidate) != 0:

                        candidate_list = []
                        for candidate in community_candidate[0]:
                            candidate_list.append(
                                (candidate, topicSim['window_{0}'.format((row - 1) * 30)].degree(candidate)))

                        candidate_list.sort(key=operator.itemgetter(1), reverse=True)

                        for degree_candidate in candidate_list:

                            next_topk_candidate = [community[1][0][0] for community in current_window if
                                                   degree_candidate[0] in community[0]]

                            if len(next_topk_candidate) == 1:
                                community_tracing_array[row, column] = next_topk_candidate[0]
                                break

        for community in current_window:
            #print(community)
            #print(community[1])
            #print(community[1][0])
            #print(community[1][0][0])

            community_identifier = community[1][0][0]

            if community_identifier not in community_tracing_array[row]:

                for column_id in range(len(community_tracing_array.T)):

                    if sum(community_tracing_array[:,
                           column_id]) == 0:  # RuntimeWarning: overflow encountered in long_scalars
                        # if len(np.unique(community_tracing_array[:, column_id])) >= 2:     # Takes way longer

                        community_tracing_array[row, column_id] = community[1][0][0]
                        break

                '''
                c = len(community_tracing_array[row])
                for back_column in reversed(community_tracing_array[row]):
    
                    if back_column != 0:
                        community_tracing_array[row, c] = community[1][0][0]
                        break
    
                    c = c - 1
    
                    if c == 0:
                        community_tracing_array[row, c] = community[1][0][0]
                        break
                '''

    # cut array to exlcude non relevant 0 columns


    for i in range(len(community_tracing_array.T)):
        if sum(community_tracing_array[:, i]) == 0:
            cutoff = i
            break

    community_tracing_array = community_tracing_array[0:len(community_tracing_array) - 1, 0:cutoff]

    # make list with flattened array and take only unique ids

    topk_list = np.unique(community_tracing_array.flatten())[1:]

    # for each id, look in which column the id first appeared

    topk_list_associ = []

    for topk in topk_list:
        candidate_list = []

        if topk == 291465230:
            print(1 + 1)

        for column in range(len(community_tracing_array.T)):

            if topk in community_tracing_array[:, column]:
                window_pos = np.where(community_tracing_array[:, column] == topk)
                # window_pos = community_tracing_array[:,column].index(topk)

                # window_pos = [i for i, x in enumerate(community_tracing_array[:,column]) if x == topk]

                # print('window_pos')
                # print(window_pos)
                window_pos = max(window_pos[0])
                # print(window_pos)
                window = lais2_commu_topK['window_{0}'.format(window_pos * 30)]
                # print(window)
                # print(max([(len(x[0]), x[1][0][0]) for x in window]))
                community_size, community_topk = max([(len(x[0]), x[1][0][0]) for x in window])

                candidate_list.append((column, community_size))

        candidate_list.sort(key=operator.itemgetter(1), reverse=True)

        topk_list_associ.append((topk, candidate_list[-1][0]))

        # topk_list_associ.append((i, j))
        # break

    # print(topk_list_associ)
    # print(lp_commu_topK)
    # print('lp_commu_topK')

    lais2_commu_id = {}

    for window_id, window in lais2_commu_topK.items():
        new_window = []

        for community in window:
            topk = community[1][0][0]
            community_id = [tuple[1] for tuple in topk_list_associ if tuple[0] == topk]

            new_window.append([community[0], community_id])

        lais2_commu_id[window_id] = new_window


    #print(lp_commu_id)
    #print(topk_list_associ)
    #print(lais2_commu_id['window_0'])
    #print(lais2_commu_topK['window_0'], '\n')

    #print(lais2_commu_id['window_900'])
    #print(lais2_commu_topK['window_900'])
    #print('sdfgerg')


    filename = 'windows_lais2_communities'
    outfile = open(filename, 'wb')
    pk.dump(lais2_commu_id, outfile)
    outfile.close()


    # kclique #

    max_number_commu = 0
    for window_id, window in kclique_commu_topK.items():
        for community in window:
            max_number_commu = max_number_commu + 1

    community_tracing_array = np.zeros((len(topicSim), max_number_commu), dtype=int)  # this is the max columns needed for the case that no community is tracable
    # print(np.shape(community_tracing_array))


    for row in range(0, len(community_tracing_array) - 1):

        current_window = kclique_commu_topK['window_{0}'.format(row * 30)]

        if row != 0:
            prev_window = kclique_commu_topK['window_{0}'.format((row - 1) * 30)]

            for column in range(len(community_tracing_array.T)):

                prev_topk = community_tracing_array[row - 1, column]
                topk_candidate = [community[1][0][0] for community in current_window if prev_topk in community[0]]

                if len(topk_candidate) == 1:
                    community_tracing_array[row, column] = topk_candidate[0]

                    '''
                elif len(topk_candidate) >= 2:
                    print(topk_candidate, current_window)
                    '''

                else:  # (e.g. 0 because the node disappears or 2 because it is in two communities)

                    # Limitation: for overlapping algorithms, taking only the node with the higest degree might be not suitable. Looking at this case below, 3 of 5 communities of that
                    # window are merged, even though they might not be the same
                    # todo: does this stability measure even make sense for overlapping? how is diffusion and recombination measured?

                    # print(prev_window)          # lais2:    # [[[288766563, 288803376, 288819596, 290076304, 290106123, 290234572, 291465230], [(291465230, 4)]], [[290076304, 290106123, 291465230, 289730801, 290720988], [(291465230, 4)]], [[290011409, 290122867, 290720623, 290787054], [(290011409, 2)]], [[288766563, 290076304, 290106123, 291465230], [(291465230, 4)]], [[289643751, 291383952, 291793181, 293035547], [(291383952, 3)]]]
                    # print(prev_topk)            # lais2:    # 291465230
                    # what i should get         #           # [[288766563, 288803376, 288819596, 290076304, 290106123, 290234572, 291465230], [290076304, 290106123, 291465230, 289730801, 290720988],[288766563, 290076304, 290106123, 291465230]]

                    # print(prev_window)          # LP:      # [[{288766563, 290106123, 291465230, 290076304, 289730801, 290720988}, [(291465230, 4)]], [{288803376, 288819596, 290234572}, [(288819596, 2)]], [{291383952, 293035547, 291793181, 289643751}, [(291383952, 3)]], [{290011409, 290122867, 290787054, 290720623}, [(290011409, 2)]]]
                    # print(prev_topk)            # LP:      # 291465230
                    # print(community_candidate)  # LP:      # [{288766563, 290106123, 291465230, 290076304, 289730801, 290720988}]

                    # print(prev_window)         # gm:       # [[[290115954, 286499669, 286404823], [(290115954, 2)]], [[289751121, 289458126, 289593087], [(289751121, 2)]]]
                    # print(prev_topk)           # gm:       # 290115954
                    # print(community_candidate) # gm:       # [[290115954, 286499669, 286404823]]

                    community_candidate = [community[0] for community in prev_window if prev_topk in community[0]]
                    # community_candidate2 = [community_candidate]
                    # community_candidate3 = [item for sublist in community_candidate2 for item in sublist]
                    # community_candidate = community_candidate3
                    # print(community_candidate)

                    if len(community_candidate) >= 2:
                        community_size, community_candidate = max([(len(x), x) for x in community_candidate])
                        community_candidate = [community_candidate]
                        # Just taking the biggest community where candidate occures. This might seem problematic, but the real "problem" is described in the to-do
                        # alternative: take the one for which prev_topk has most edges in or biggest edge weight sum in
                    if len(community_candidate) != 0:

                        candidate_list = []
                        for candidate in community_candidate[0]:
                            candidate_list.append(
                                (candidate, topicSim['window_{0}'.format((row - 1) * 30)].degree(candidate)))

                        candidate_list.sort(key=operator.itemgetter(1), reverse=True)

                        for degree_candidate in candidate_list:

                            next_topk_candidate = [community[1][0][0] for community in current_window if
                                                   degree_candidate[0] in community[0]]

                            if len(next_topk_candidate) == 1:
                                community_tracing_array[row, column] = next_topk_candidate[0]
                                break

        for community in current_window:
            #print(community)
            #print(community[1])
            #print(community[1][0])
            #print(community[1][0][0])

            community_identifier = community[1][0][0]

            if community_identifier not in community_tracing_array[row]:

                for column_id in range(len(community_tracing_array.T)):

                    if sum(community_tracing_array[:,column_id]) == 0:  # RuntimeWarning: overflow encountered in long_scalars
                        # if len(np.unique(community_tracing_array[:, column_id])) >= 2:     # Takes way longer

                        community_tracing_array[row, column_id] = community[1][0][0]
                        break

                '''
                c = len(community_tracing_array[row])
                for back_column in reversed(community_tracing_array[row]):
    
                    if back_column != 0:
                        community_tracing_array[row, c] = community[1][0][0]
                        break
    
                    c = c - 1
    
                    if c == 0:
                        community_tracing_array[row, c] = community[1][0][0]
                        break
                '''

    # cut array to exlcude non relevant 0 columns


    for i in range(len(community_tracing_array.T)):
        if sum(community_tracing_array[:, i]) == 0:
            cutoff = i
            break

    community_tracing_array = community_tracing_array[0:len(community_tracing_array) - 1, 0:cutoff]

    # make list with flattened array and take only unique ids

    topk_list = np.unique(community_tracing_array.flatten())[1:]

    # for each id, look in which column the id first appeared

    topk_list_associ = []

    for topk in topk_list:
        candidate_list = []

        if topk == 291465230:
            print(1 + 1)

        for column in range(len(community_tracing_array.T)):

            if topk in community_tracing_array[:, column]:
                window_pos = np.where(community_tracing_array[:, column] == topk)
                # window_pos = community_tracing_array[:,column].index(topk)

                # window_pos = [i for i, x in enumerate(community_tracing_array[:,column]) if x == topk]

                # print('window_pos')
                # print(window_pos)
                window_pos = max(window_pos[0])
                # print(window_pos)
                window = kclique_commu_topK['window_{0}'.format(window_pos * 30)]
                # print(window)
                # print(max([(len(x[0]), x[1][0][0]) for x in window]))
                community_size, community_topk = max([(len(x[0]), x[1][0][0]) for x in window])

                candidate_list.append((column, community_size))

        candidate_list.sort(key=operator.itemgetter(1), reverse=True)

        topk_list_associ.append((topk, candidate_list[-1][0]))

        # topk_list_associ.append((i, j))
        # break

    # print(topk_list_associ)
    # print(lp_commu_topK)
    # print('lp_commu_topK')

    kclique_commu_id = {}

    for window_id, window in kclique_commu_topK.items():
        new_window = []

        for community in window:
            topk = community[1][0][0]
            community_id = [tuple[1] for tuple in topk_list_associ if tuple[0] == topk]

            new_window.append([community[0], community_id])

        kclique_commu_id[window_id] = new_window

    # print(lp_commu_id)
    # print(topk_list_associ)
    #print(kclique_commu_id['window_0'])
    #print(kclique_commu_topK['window_0'], '\n')

    #print(kclique_commu_id['window_900'])
    #print(kclique_commu_topK['window_900'])
    #print('sdfgerg')


    filename = 'windows_kclique_communities'
    outfile = open(filename, 'wb')
    pk.dump(kclique_commu_id, outfile)
    outfile.close()


