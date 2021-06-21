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

    #with open('window90by1', 'rb') as handle:
    #    windows = pk.load(handle)

    with open('window365by30', 'rb') as handle:
        windows = pk.load(handle)

#--- Custom weighting function for preojection ---#

    def test_weight(G, u, v):
        #print(u)                # Topic1 of edge
        #print(v)                # Topic2 of edge
        #print(G[u])             # Neighbors of Topic1
        #print(G[v])             # Neighbors of Topic2

        u_nbrs = set(G[u])      # Neighbors of Topic1 in set format for later intersection
        v_nbrs = set(G[v])      # Neighbors of Topic2 in set format for later intersection

        shared_nbrs = u_nbrs.intersection(v_nbrs)       # Shared neighbors of both topic nodes (intersection)

        #print(shared_nbrs)

        list_of_poducts = []

        for i in shared_nbrs:
            #print(i)
            #print(G.edges[u,i])                         # Edge (-weight) of topic1 and shared neighbor
            #print(G.edges[v,i])                         # Edge (-weight) of topic2 and shared neighbor

            weight1 = list(G.edges[u,i].values())[0]
            weight2 = list(G.edges[v,i].values())[0]

            #print(weight1)
            #print(weight2)

            #print(weight1 * weight2)                    # product of weights (contrasting to taking the sum, the product penalized heavily skeewd topic distributions like 0.1 and 0.9

            list_of_poducts.append(weight1 * weight2)

        #print(list_of_poducts)

        projected_weight = sum(list_of_poducts) / len(list_of_poducts)
        #print(projected_weight)

        return projected_weight #, len(list_of_poducts)  # return the resultung weight and the number of shared neighbors


#--- Preparing overall node attributes ---#


    node_att_name = ['publn_auth', 'publn_nr', 'publn_date', 'publn_claims', 'nb_IPC']

    number_of_topics = int(len(patent_lda_ipc[0, 9:30]) / 3)

    for i in range(1, number_of_topics + 1):
        node_att_name.append('TopicID_{0}'.format(i))
        node_att_name.append('TopicName_{0}'.format(i))
        node_att_name.append('TopicCover_{0}'.format(i))

    number_of_ipcs = int(len(patent_lda_ipc[0, 30:]) / 3)

    for i in range(1, number_of_ipcs + 1):
        node_att_name.append('IpcID_{0}'.format(i))
        node_att_name.append('IpcSubCat1_{0}'.format(i))
        node_att_name.append('IpcSubCat2_{0}'.format(i))

#--- Creating a Graph for each windows---#


    bipartite_graphs = {}
    topicOccu_graphs = {}
    topicSim_graphs = {}

    pbar = tqdm.tqdm(total=len(windows))

    for window_id, window in windows.items():

        ### Create Graph ###

        sliding_graph = nx.Graph()


        ### Create Nodes - Paents ###

        nodes = window[:, 0]            # extract patent ids in window

        node_att_dic = []
        for i in range(len(window)):

            window_reduc = window[:, np.r_[1:5, 7, 9:len(window.T)]]

            dic_entry = dict(enumerate(window_reduc[i]))  # Here each patent is converted into a dictionary. dictionary keys are still numbers:
            # {0: 'EP', 1: nan, 2: '2007-10-10', ...} Note that the patent id is ommited, since it
            # serves as key for the outer dictionary encapsulating these inner once.


            for key, n_key in zip(dic_entry.copy().keys(), node_att_name):  # Renaming the keys of the inner dictionary from numbers to actually names saved in node_plain_att
                dic_entry[n_key] = dic_entry.pop(key)  # {'publn_auth': 'EP', 'publn_nr': nan, 'publn_date': '2009-06-17', ...}

            node_att_dic.append(dic_entry)


        nested_dic = dict(enumerate(node_att_dic))  # Here the nested (outer) dictionary is created. Each key is still represented as a number, each value as another dictionary

        for key, n_key in zip(nested_dic.copy().keys(), nodes):  # Here the key of the outer dictionary are renamed to the patent ids
            nested_dic[n_key] = nested_dic.pop(key)

        #print(len(window))


        sliding_graph.add_nodes_from(nodes, bipartite=1)
        nx.set_node_attributes(sliding_graph, nested_dic)


        #print(sliding_graph.number_of_nodes())
        #print(sliding_graph.nodes[290106123])
        # {'bipartite': 1, 'publn_auth': 'EP', 'publn_nr': 1139503.0, 'publn_date': '2001-10-04', 'publn_claims': 13, 'nb_IPC': 3, 'TopicID_1': 16.0, 'TopicName_1': nan, ...
        #break


        ### Create Nodes - Topics ###

        ipc_position = np.r_[range(30,np.shape(patent_lda_ipc)[1]-1,3)]             # right now, this has to be adjusted manually depending on the LDA results #todo adjust
        topic_position = np.r_[range(9,30,3)]                                       # right now, this has to be adjusted manually depending on the LDA results #todo adjust

        topics_inWindow = []

        for patent in window:
            topics_inWindow.append(patent[topic_position])


        topics_inWindow = [item for sublist in topics_inWindow for item in sublist]
        topics_inWindow = list(filter(lambda x: x == x, topics_inWindow))
        topics_inWindow = np.unique(topics_inWindow)

        topicNode_list = ['topic_{0}'.format(int(i)) for i in topics_inWindow]

        sliding_graph.add_nodes_from(topicNode_list, bipartite=0)


        ### Create Edges ###

        edges = window[:, np.r_[0, 9:18]]  # first three topics

        #edges = window[:, np.r_[0, 9:31]]  # topics
        #for i in range(1, 7 * 3, 3):

        c = 0
        for i in edges.T[1]:
            if np.isfinite(i):
                edges[c, 1] = 'topic_{0}'.format(int(i))
            c = c + 1

        c = 0
        for i in edges.T[4]:
            if np.isfinite(i):
                edges[c, 4] = 'topic_{0}'.format(int(i))
            c = c + 1

        c = 0
        for i in edges.T[7]:
            if np.isfinite(i):
                edges[c, 7] = 'topic_{0}'.format(int(i))
            c = c + 1

        topic1_edges = [(i[0], i[1], {'Weight_1': i[3]}) for i in edges]
        topic2_edges = [(i[0], i[4], {'Weight_2': i[6]}) for i in edges]
        topic3_edges = [(i[0], i[7], {'Weight_3': i[9]}) for i in edges]

        topic1_edges_clear = list(filter(lambda x: x[1] == x[1], topic1_edges))
        topic2_edges_clear = list(filter(lambda x: x[1] == x[1], topic2_edges))
        topic3_edges_clear = list(filter(lambda x: x[1] == x[1], topic3_edges))


        sliding_graph.add_edges_from(topic1_edges_clear)
        sliding_graph.add_edges_from(topic2_edges_clear)
        sliding_graph.add_edges_from(topic3_edges_clear)


        ### Project ###

        top_nodes = {n for n, d in sliding_graph.nodes(data=True) if d["bipartite"] == 0}
        bottom_nodes = set(sliding_graph) - top_nodes

        topicOccu_graph = nx.algorithms.bipartite.generic_weighted_projected_graph(sliding_graph, top_nodes, weight_function=test_weight)
        topicSim_graph = nx.algorithms.bipartite.generic_weighted_projected_graph(sliding_graph, bottom_nodes, weight_function=test_weight)

        ### Append ###

        bipartite_graphs[window_id] = sliding_graph
        topicOccu_graphs[window_id] = topicOccu_graph
        topicSim_graphs[window_id] = topicSim_graph

        pbar.update(1)

    pbar.close()

    #print(bipartite_graphs)
    #print(bipartite_graphs['window_0'].nodes())
    #print('\n', bipartite_graphs['window_0'].nodes[290106123])
    #print('\n', bipartite_graphs['window_0'].edges())
    #print('\n', bipartite_graphs['window_0'][290106123]['topic_16'])
    #print(topicOccu_graphs)
    #print(topicOccu_graphs['window_0'].nodes())
    #print('\n', topicOccu_graphs['window_0'].nodes['topic_270'])
    #print('\n', topicOccu_graphs['window_0'].edges())
    #print('\n', topicOccu_graphs['window_0']['topic_134']['topic_18'])
    #print(topicSim_graphs)
    #print(topicSim_graphs['window_0'].nodes())
    #print('\n', topicSim_graphs['window_0'].nodes[291383952])
    #print('\n', topicSim_graphs['window_0'].edges())
    #print('\n', topicSim_graphs['window_0'][291465230][291383952])

    #print(topicSim_g.edges.data('weight'))

#--- Save Sliding Graphs ---#

    filename = 'windows_bipartite'
    outfile = open(filename, 'wb')
    pk.dump(bipartite_graphs, outfile)
    outfile.close()

    filename = 'windows_topicOccu'
    outfile = open(filename, 'wb')
    pk.dump(topicOccu_graphs, outfile)
    outfile.close()

    filename = 'windows_topicSim'
    outfile = open(filename, 'wb')
    pk.dump(topicSim_graphs, outfile)
    outfile.close()

