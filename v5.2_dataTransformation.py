
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


    import statistics

    ### Find biggest number of IPCs a patent has (new space) ###
    #print(patent_IPC)
    patent_IPC_clean = [i[0] for i in patent_IPC if i[0] in patent_transf[:, 0]]
    #print(patent_IPC_clean)
    val, count = np.unique(patent_IPC_clean, return_counts=True)
    #print(count)
    new_space_needed = max(count) * 3       # x * 3 = x3 additional columns are needed in the new array
    print(max(count))   # x * 3 = x3 additional columns are needed in the new array
    print(min(count))   # x * 3 = x3 additional columns are needed in the new array
    print(np.mean(count))   # x * 3 = x3 additional columns are needed in the new array
    print(np.median(count))   # x * 3 = x3 additional columns are needed in the new array
    print(statistics.mode(count))   # x * 3 = x3 additional columns are needed in the new array

    ### New array, including space for IPC's ###
    patent_join = np.empty((np.shape(patent_transf)[0], np.shape(patent_transf)[1] + new_space_needed), dtype=object)
    patent_join[:, :-new_space_needed] = patent_transf


    ### Fill new array ###
    patent_join = Transf_misc.fill_with_IPC(patent_join, patent_IPC, new_space_needed)

    ipc_list_full = []
    ipc_list_4 = []
    ipc_list_3 = []
    #ipc_list_2 = []            # Makes no sense
    ipc_list_1 = []
    # number of ipcs in dataset
    for patent in patent_join:
        #print(patent[23:])
        #print(patent)
        #print(len(patent[23:]))
        for ipc in range(0,len(patent[23:]),3):
            #print(ipc)
            if patent[23:][ipc] != None:
                ipc_list_full.append(patent[23:][ipc])
                ipc_list_4.append(patent[23:][ipc][0:4])
                ipc_list_3.append(patent[23:][ipc][0:3])
                #ipc_list_2.append(patent[23:][ipc][0:2])
                ipc_list_1.append(patent[23:][ipc][0:1])

                if patent[23:][ipc][0] == 'D':
                    print(patent)

    print("\n full:")
    print(ipc_list_full[0])
    print(len(ipc_list_full))
    ipc_list_full = np.unique(ipc_list_full)
    print(len(ipc_list_full))

    print("\n 4:")
    print(ipc_list_4[0])
    print(len(ipc_list_4))
    ipc_list_4 = np.unique(ipc_list_4)
    print(len(ipc_list_4))

    print("\n 3:")
    print(ipc_list_3[0])
    print(len(ipc_list_3))
    ipc_list_3 = np.unique(ipc_list_3)
    print(len(ipc_list_3))

    #print("\n 2:")
    #print(ipc_list_2[0])
    #print(len(ipc_list_2))
    #ipc_list_2 = np.unique(ipc_list_2)
    #print(len(ipc_list_2))

    print("\n 1:")
    print(ipc_list_1[0])
    print(len(ipc_list_1))
    ipc_list_1_clean = np.unique(ipc_list_1)
    print(ipc_list_1_clean)
    print(len(ipc_list_1_clean))



    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1)
    ax.hist(ipc_list_1, bins=8, color='darkred')
    #locator = mdates.AutoDateLocator()
    #ax.xaxis.set_major_locator(locator)
    #ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(locator))
    #plt.title("Histogram: Monthly number of patent publications")
    plt.xlabel("International Patent Classification - Sections")
    plt.ylabel("Number of Patents")
    plt.show()

    #os.chdir('D:/Universitaet Mannheim/MMDS 7. Semester/Master Thesis/Outline/Plots')

    #plt.savefig('hist_publications.png')


        #ipc_list.append(patent[1,23:])

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
    # todo check size of last window

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


        #print(sliding_graph.number_of_nodes())
        #print(sliding_graph.nodes[290106123])
        # {'bipartite': 1, 'publn_auth': 'EP', 'publn_nr': 1139503.0, 'publn_date': '2001-10-04', 'publn_claims': 13, 'nb_IPC': 3, 'TopicID_1': 16.0, 'TopicName_1': nan, ...
        #break


        ### Create Nodes - Topics ###

        ipc_position = np.r_[range(30,np.shape(patent_lda_ipc)[1]-1,3)]             # right now, this has to be adjusted manually depending on the LDA results #todo adjust
        topic_position = np.r_[range(9,(9+int(max_topics*2)),2)]                                       # right now, this has to be adjusted manually depending on the LDA results #todo adjust

        topicNode_list = Transf_network.prepare_topicNodes_Networkx(window, topic_position)

        sliding_graph.add_nodes_from(topicNode_list, bipartite=0)



        ### Create Edges ###

        num_topics = 3

        topic_edges_list = Transf_network.prepare_edgeLists_Networkx(window, num_topics, max_topics)

        #print(topic_edges_list)

        for edge_list in topic_edges_list:
            sliding_graph.add_edges_from(edge_list)


        ### Project ###

        top_nodes = {n for n, d in sliding_graph.nodes(data=True) if d["bipartite"] == 0}
        bottom_nodes = set(sliding_graph) - top_nodes

        topicProject_graph = nx.algorithms.bipartite.generic_weighted_projected_graph(sliding_graph, top_nodes, weight_function=Transf_network.test_weight)
        patentProject_graph = nx.algorithms.bipartite.generic_weighted_projected_graph(sliding_graph, bottom_nodes, weight_function=Transf_network.test_weight)

        ### Append ###

        bipartite_graphs[window_id] = sliding_graph
        patentProject_graphs[window_id] = patentProject_graph
        topicProject_graphs[window_id] = topicProject_graph

        pbar.update(1)

    pbar.close()


#--- Save Sliding Graphs ---#

    filename = 'windows_bipartite'
    outfile = open(filename, 'wb')
    pk.dump(bipartite_graphs, outfile)
    outfile.close()

    filename = 'topicProject_graphs'
    outfile = open(filename, 'wb')
    pk.dump(topicProject_graphs, outfile)
    outfile.close()

    filename = 'patentProject_graphs'
    outfile = open(filename, 'wb')
    pk.dump(patentProject_graphs, outfile)
    outfile.close()









