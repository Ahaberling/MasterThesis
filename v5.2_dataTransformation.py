
if __name__ == '__main__':

#--- Import Libraries ---#
    print('\n#--- Import Libraries ---#\n')

    import os
    import tqdm

    import numpy as np
    import pandas as pd
    import pickle as pk
    import networkx as nx


#--- Initialization ---#
    print('\n#--- Initialization ---#\n')

    os.chdir('D:/Universitaet Mannheim/MMDS 7. Semester/Master Thesis/Outline/Data/Cleaning Robots')

    patent_topicDist = pd.read_csv('patent_topicDist_mallet.csv', quotechar='"', skipinitialspace=True)
    patent_topicDist = patent_topicDist.to_numpy()

    patent_IPC = pd.read_csv('cleaning_robot_EP_patents_IPC.csv', quotechar='"', skipinitialspace=True)
    patent_IPC = patent_IPC.to_numpy()

    topics = pd.read_csv('patent_topics_mallet.csv', quotechar='"', skipinitialspace=True)
    topics = topics.to_numpy()

    #parent = pd.read_csv('cleaning_robot_EP_backward_citations.csv', quotechar='"', skipinitialspace=True)




#--- Transforming topic representation ---#
    print('\n#--- Transforming topic representation ---#\n')

    from utilities.my_transform_utils import Transf_misc

    topic_list_helper, max_topics = Transf_misc.max_number_topics(patent_topicDist)
    print('Maximum number of topics a patent has: ', max_topics)               # 7 is the maximum of topics abstracts have
    print('Hence, number of new columns needed: ',   max_topics*2)             # space needed = 21 (7 topic_ids + 7 topic_names + 7 topic_coverages)

    ### Prepare dataframe with columns for topics (transformation result) ###
    patent_transf = np.empty((np.shape(patent_topicDist)[0], np.shape(patent_topicDist)[1] + int(max_topics*2)), dtype=object)
    patent_transf[:, :-int(max_topics*2)] = patent_topicDist

    ### Filling the new array ###
    patent_transf = Transf_misc.fill_with_topics(patent_transf, topic_list_helper, np.shape(patent_topicDist)[1])


#--- Check transformation ---#
    print('\n#--- Check transformation ---#\n')

    if sum(x is not None for x in patent_transf[:,np.shape(patent_transf)[1]-1]) == 0:
        raise Exception("Error: Not all created columns in patent_transf have been filled")


#--- Append IPC ---#
    print('\n#--- Append IPC ---#\n')

    # Each Patent has at least one IPC. These IPC'S are appended to the patent_transf array in order to facilitate
    # a brief, heuristic evaluation of topic modeling. Additionally they might be used to conceptualize first basic
    # recombination and diffusion measures


    ### Review patent_transf and patent_IPC ###
    print('Review patent_transf and patent_IPC:\n')

    # check if all patents in patent_transf are unique
    val, count = np.unique(patent_transf[:, 0], return_counts=True)
    if len(val) != len(patent_transf):
        raise Exception("Error: patent_transf contains non-unqiue patents")


    ### Find biggest number of IPCs a patent has (new space) ###
    patent_IPC_clean = [i[0] for i in patent_IPC if i[0] in patent_transf[:, 0]]
    val, count = np.unique(patent_IPC_clean, return_counts=True)
    new_space_needed = max(count) * 3       # x * 3 = x3 additional columns are needed in the new array

    ### New array, including space for IPC's ###
    patent_join = np.empty((np.shape(patent_transf)[0], np.shape(patent_transf)[1] + new_space_needed), dtype=object)
    patent_join[:, :-new_space_needed] = patent_transf


    ### Fill new array ###
    patent_join = Transf_misc.fill_with_IPC(patent_join, patent_IPC, new_space_needed)

    ### check if all created columns are used ###

    if sum(x is not None for x in patent_join[:, np.shape(patent_join)[1] - 1]) == 0:
        raise Exception("Error: Not all created columns in patent_join have been filled")



#--- Save transformation and IPC appendix---#
    #print('\n#--- Save transformation and IPC appendix---#\n')

    #pd.DataFrame(patent_join).to_csv('patent_lda_ipc.csv', index=False)


#### new file ############################

    patent_lda_ipc = patent_join

    ### Declare sliding window approach ###

    windowSize =  365
    slidingInterval = 30


#--- Overview ---#
    print('\n# --- Overview ---#\n')

    patent_time = patent_lda_ipc[:,3].astype('datetime64')

    print('Earliest day with publication: ', min(patent_time))          # earliest day with publication 2001-08-01
    print('Latest day with publication: ', max(patent_time))            # latest  day with publication 2018-01-31

    max_timeSpan = int((max(patent_time) - min(patent_time)) / np.timedelta64(1, 'D'))
    print('Days inbetween: ', max_timeSpan)                             # 6027 day between earliest and latest publication

    val, count = np.unique(patent_time, return_counts=True)
    print('Number of days with publications: ', len(val))               # On 817 days publications were made
                                                                        # -> on average every 7.37698898409 days a patent was published


#--- slinding window approache ---#
    print('\n#--- slinding window approach ---#\n')

    from utilities.my_transform_utils import Transf_slidingWindow

    slidingWindow_dict = Transf_slidingWindow.sliding_window_slizing(windowSize, slidingInterval, patent_lda_ipc,)

    filename = 'slidingWindow_dict'
    outfile = open(filename, 'wb')
    pk.dump(slidingWindow_dict, outfile)
    outfile.close()


### New File ################################


#--- Custom weighting function for preojection ---#



#--- Preparing overall node attributes ---#

    from utilities.my_transform_utils import Transf_network

    node_att_name = ['publn_auth', 'publn_nr', 'publn_date', 'publn_claims', 'nb_IPC']

    for i in range(1, int(max_topics) + 1):
        node_att_name.append('TopicID_{0}'.format(i))
        node_att_name.append('TopicCover_{0}'.format(i))

    number_of_ipcs = new_space_needed/3

    for i in range(1, int(number_of_ipcs) + 1):
        node_att_name.append('IpcID_{0}'.format(i))
        node_att_name.append('IpcSubCat1_{0}'.format(i))
        node_att_name.append('IpcSubCat2_{0}'.format(i))


#--- Creating a Graph for each windows---#

    bipartite_graphs = {}
    patentProject_graphs = {}
    topicProject_graphs = {}

    from utilities.my_transform_utils import Transf_network

    pbar = tqdm.tqdm(total=len(slidingWindow_dict))

    for window_id, window in slidingWindow_dict.items():

        ### Create Graph ###
        sliding_graph = nx.Graph()

        ### Create Nodes - Paents ###
        nodes = window[:, 0]            # extract patent ids in window
        window_reduc = window[:, np.r_[1:5, 7, 9:len(window.T)]]

        nested_dic = Transf_network.prepare_patentNodeAttr_Networkx(window_reduc, nodes, node_att_name)

        sliding_graph.add_nodes_from(nodes, bipartite=1)
        nx.set_node_attributes(sliding_graph, nested_dic)


        print(sliding_graph.number_of_nodes())
        print(sliding_graph.nodes[290106123])
        # {'bipartite': 1, 'publn_auth': 'EP', 'publn_nr': 1139503.0, 'publn_date': '2001-10-04', 'publn_claims': 13, 'nb_IPC': 3, 'TopicID_1': 16.0, 'TopicName_1': nan, ...
        #break


        ### Create Nodes - Topics ###

        ipc_position = np.r_[range(30,np.shape(patent_lda_ipc)[1]-1,3)]             # right now, this has to be adjusted manually depending on the LDA results #todo adjust
        topic_position = np.r_[range(9,(9+int(max_topics)),2)]                                       # right now, this has to be adjusted manually depending on the LDA results #todo adjust

        topicNode_list = Transf_network.prepare_topicNodes_Networkx(window, topic_position)

        sliding_graph.add_nodes_from(topicNode_list, bipartite=0)

        num_topics = 3

        ### Create Edges ###

        edges = window[:, np.r_[0, 9:(9+(2*num_topics))]]  # first three topics

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









