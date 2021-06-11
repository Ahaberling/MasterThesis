if __name__ == '__main__':

#--- Import Libraries ---#
    print('\n#--- Import Libraries ---#\n')

    import pandas as pd
    import numpy as np

    import networkx as nx

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



#--- Plain citation network - Preparing data structure for networkx ---#
    print('\n#--- Plain citation network - Preparing data structure for networkx ---#\n')


    ### Plain citation nodes ###

    node_plain = patent_lda_ipc[:,0]
    #print(len(node_plain))                         # 3781


    ### Plain citation edges ###

    citation_source = parent[:,0]                   # pat_publn_id
    citation_target = parent[:,2]                   # cited_pat_publn_id

    #print(citation_source)                          # [468378953 285297843 336208959 ...]
    #print(len(citation_source))                     # 18548


    ### plain citation node attributes ###

    #node_plain_att_full = ['publn_auth', 'publn_nr', 'publn_date', 'publn_claims', 'publn_title',        # with column 5, 6, 8
    #                       'publn_abstract', 'nb_IPC', 'topic_list']           # pat_publn_id ommited, since it's the node id. No need to add it as attribute as well.

    node_plain_att = ['publn_auth', 'publn_nr', 'publn_date', 'publn_claims', 'nb_IPC']

    #print(patent_lda_ipc[1,:])
    #print(patent_lda_ipc[1,9:30])                                      # topic columns
    #print(patent_lda_ipc[1,30:])                                       # ipc columns


    number_of_topics = int(len(patent_lda_ipc[0,9:30])/3)
    #print(loop_helper)

    for i in range(1,number_of_topics+1):
        node_plain_att.append('TopicID_{0}'.format(i))
        node_plain_att.append('TopicName_{0}'.format(i))
        node_plain_att.append('TopicCover_{0}'.format(i))


    number_of_ipcs = int(len(patent_lda_ipc[0, 30:]) / 3)
    # print(loop_helper)

    for i in range(1, number_of_ipcs + 1):
        node_plain_att.append('IpcID_{0}'.format(i))
        node_plain_att.append('IpcSubCat1_{0}'.format(i))
        node_plain_att.append('IpcSubCat2_{0}'.format(i))

    #print(node_plain_att)                   # ['publn_auth', ... 'TopicID_1', ... 'IpcID_1', ... ]

    #print(len(node_plain_att))
    #print(len(patent_lda_ipc[0,:]))

    # the resulting data structure is a nested dictionary. First an inner dictionary is constructed for each patent, where each column represents an entry. Then the outer
    # dictionary is created, where each patent represents an entry. An entry of the outer entry thereby consists of a dictionary as well.


    patent_lda_ipc_reduc = patent_lda_ipc[:, np.r_[1:5, 7, 9:len(patent_lda_ipc.T)]]   # exclude title, abstract and topic list because of the data size

    inner_dic = []
    for i in range(len(patent_lda_ipc_reduc)):

        inner_dic_entry = dict(enumerate(patent_lda_ipc_reduc[i]))                     # Here each patent is converted into a dictionary. dictionary keys are still numbers:
                                                                                    # {0: 'EP', 1: nan, 2: '2007-10-10', ...} Note that the patent id is ommited, since it
                                                                                    # serves as key for the outer dictionary encapsulating these inner once.

        for key, n_key in zip(inner_dic_entry.copy().keys(), node_plain_att):       # Renaming the keys of the inner dictionary from numbers to actually names saved in node_plain_att
            inner_dic_entry[n_key] = inner_dic_entry.pop(key)                       # {'publn_auth': 'EP', 'publn_nr': nan, 'publn_date': '2009-06-17', ...}

        inner_dic.append(inner_dic_entry)


    outer_dic = dict(enumerate(inner_dic))                                          # Here the nested (outer) dictionary is created. Each key is still represented as a number, each value as another dictionary

    for key, n_key in zip(outer_dic.copy().keys(), node_plain):                     # Here the key of the outer dictionary are renamed to the patent ids
        outer_dic[n_key] = outer_dic.pop(key)

    nested_dic = outer_dic

    #print(nested_dic)



#--- ---#

    plain = nx.Graph()
    bipart = nx.Graph()
    topic_similar = nx.Graph()
    topic_net = nx.Graph()

    plain.add_nodes_from(node_plain)
    nx.set_node_attributes(plain, nested_dic)

    #print(plain.nodes[487838990])

    set_nodes = set(node_plain)
    set_edgeSource = set(citation_source)
    set_edgeTarget = set(citation_target)

    validEdges = []
    for i in range(len(parent)):
        if citation_source[i] in node_plain:
            if citation_target[i] in node_plain:
                validEdges.append(tuple((citation_source[i],citation_target[i])))


    print('Number of edges with source and target in the patent data: ', len(np.unique(validEdges, axis= 0)))

    plain.add_edges_from(validEdges)

    print('Number of nodes in plain: ', plain.number_of_nodes())
    print('Number of edges in plain: ', plain.number_of_edges())


#--- Bipartite ---#

    # 0 = top ~ topics

    bipart.add_nodes_from(node_plain, bipartite=1)
    nx.set_node_attributes(bipart, nested_dic)

    topicNode_list = ['topic_{0}'.format(i) for i in range(len(topics))]
    #print(topicNode_list)
    #print(len(topicNode_list))                                      # 325

    bipart.add_nodes_from(topicNode_list, bipartite=0)
    #print(bipart.number_of_nodes())                                 # 4106 = 3781 + 325


    ### edges ###


    bipart_edges = patent_lda_ipc[:,np.r_[0, 9:18]]         # first three topics

    c = 0
    for i in bipart_edges.T[1]:
        if np.isfinite(i):
            bipart_edges[c,1] = 'topic_{0}'.format(int(i))
        c = c + 1

    c = 0
    for i in bipart_edges.T[4]:
        if np.isfinite(i):
            bipart_edges[c, 4] = 'topic_{0}'.format(int(i))
        c = c + 1


    c = 0
    for i in bipart_edges.T[7]:
        if np.isfinite(i):
            bipart_edges[c,7] = 'topic_{0}'.format(int(i))
        c = c + 1



    topic1_edges = [(i[0], i[1], {'Weight_1': i[3]}) for i in bipart_edges]
    topic2_edges = [(i[0], i[4], {'Weight_2': i[6]}) for i in bipart_edges]
    topic3_edges = [(i[0], i[7], {'Weight_3': i[9]}) for i in bipart_edges]

    #print(bipart_edges[0])
    #print(topic1_edges)
    #print(topic2_edges)

    #print(len(topic1_edges))

    topic1_edges_clear = list(filter(lambda x: x[1] == x[1], topic1_edges))
    topic2_edges_clear = list(filter(lambda x: x[1] == x[1], topic2_edges))
    topic3_edges_clear = list(filter(lambda x: x[1] == x[1], topic3_edges))

    #print(topic1_edges_clear)
    #print(len(topic1_edges_clear))

    print(len(bipart.nodes))
    print(len(bipart.edges))

    bipart.add_edges_from(topic1_edges_clear)
    bipart.add_edges_from(topic2_edges_clear)
    bipart.add_edges_from(topic3_edges_clear)
    print(len(bipart.nodes))
    print(len(bipart.edges))

    print(nx.is_connected(bipart))

    #bottom_nodes, top_nodes = nx.algorithms.bipartite.sets(bipart) # not working, because not connected

    top_nodes = {n for n, d in bipart.nodes(data=True) if d["bipartite"] == 0}
    bottom_nodes = set(bipart) - top_nodes

    print(top_nodes)
    print(len(top_nodes))
    print(bottom_nodes)
    print(len(bottom_nodes))

    print(nx.algorithms.bipartite.density(bipart, top_nodes))
    print(round(nx.algorithms.bipartite.density(bipart, top_nodes), 2))
    print(nx.algorithms.bipartite.density(bipart, bottom_nodes))
    print(round(nx.algorithms.bipartite.density(bipart, bottom_nodes), 2))

    topic_g_unweig = nx.algorithms.bipartite.projected_graph(bipart, top_nodes)
    topicSim_g_unweig = nx.algorithms.bipartite.projected_graph(bipart, bottom_nodes)

    # nx.write_gml(plain, r'D:\Universitaet Mannheim\MMDS 7. Semester\Master Thesis\Outline\Data\Cleaning Robots\plain.gml')