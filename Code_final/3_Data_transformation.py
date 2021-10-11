if __name__ == '__main__':

    #--- Import Libraries ---#
    print('\n#--- Import Libraries ---#\n')

    # Utility
    import os
    import tqdm
    import statistics

    # Data handling
    import numpy as np
    import pandas as pd
    import pickle as pk

    # Network X
    import networkx as nx

    # Visualization
    import matplotlib.pyplot as plt

    # Custom functions
    from utilities_final.Data_Preparation_utils import TransformationMisc
    from utilities_final.Data_Preparation_utils import Transformation_SlidingWindows
    from utilities_final.Data_Preparation_utils import Transformation_Network


    #--- Initialization ---#
    print('\n#--- Initialization ---#\n')

    path = 'D:/Universitaet Mannheim/MMDS 7. Semester/Master Thesis/Outline/Data/new'

    # Import data
    os.chdir(path)

    patent_topicDist = pd.read_csv('patent_topicDistribution_mallet.csv', quotechar='"', skipinitialspace=True)
    #patent_topicDist = pd.read_csv('patent_topicDist_mallet.csv', quotechar='"', skipinitialspace=True)
    patent_topicDist = patent_topicDist.to_numpy()

    patent_IPC = pd.read_csv('cleaning_robot_EP_patents_IPC.csv', quotechar='"', skipinitialspace=True)
    patent_IPC = patent_IPC.to_numpy()

    # specify sliding window approach
    windowSize =  360
    slidingInterval = 30



    #--- Transforming topic representation ---#
    print('\n#--- Transforming topic representation ---#\n')


    topic_list_helper, max_topics = TransformationMisc.max_number_topics(patent_topicDist)

    # descriptives
    numTopic_list = []
    topicFrequency_list = []
    coverageFrequency_list = []
    for TopicList in topic_list_helper:
        numTopic_list.append(int(len(TopicList)/2))
        c = 0
        for j in TopicList:

            if c % 2 == 0:
                topicFrequency_list.append(int(j))
            else:
                coverageFrequency_list.append(float(j))
            c = c +1

    print('Average number of topics per abstract: ', sum(numTopic_list)/len(numTopic_list))
    print('median number of topics per abstract: ', np.median(numTopic_list))
    print('mode number of topics per abstract: ', statistics.mode(numTopic_list))
    print('min number of topics per abstract: ', min(numTopic_list))
    print('max number of topics per abstract: ', max(numTopic_list), '\n')

    print('Average coverage of topics: ', sum(coverageFrequency_list) / len(coverageFrequency_list))
    print('median coverage of topics: ', np.median(coverageFrequency_list))
    print('mode coverage of topics: ', statistics.mode(coverageFrequency_list))
    print('max coverage of topics: ', max(coverageFrequency_list))
    print('min coverage of topics: ', min(coverageFrequency_list), '\n')

    val, count = np.unique(topicFrequency_list, return_counts=True)
    min_pos = np.where(count == (min(count)))
    max_pos = np.where(count == (max(count)))
    average = np.mean(count)
    print('Most common topic: ', val[max_pos])
    print('Least  common topic: ', val[min_pos], '\n')

    print('Average number of abstracts a topic appears in: ', average)
    print('Median number of abstracts a topic appears in: ', np.median(count))
    print('Mode number of abstracts a topic appears in: ', statistics.mode(count))
    print('Max number of abstracts a topic appears in: ', max(count))
    print('Min number of abstracts a topic appears in: ', min(count), '\n')

    # Visualization
    fig, ax = plt.subplots(1, 1)
    ax.hist(numTopic_list, bins=8, color='darkblue')
    plt.xlabel("Number of topics in an abstract")
    plt.ylabel("Frequency")
    plt.savefig('hist_topics_perAbstract.png')
    plt.close()

    topicFrequency_list.sort()
    fig, ax = plt.subplots(1, 1)
    ax.hist(topicFrequency_list, bins=np.arange(0, 331), color='darkblue')
    plt.xticks(np.arange(0, 331, 30))
    plt.xlabel("Number of abstract occurrences per topic")
    plt.ylabel("Frequency")
    plt.savefig('hist_abstracts_perTopic.png')
    plt.close()


    # Prepare dataframe with columns for topics (transformation result)
    patent_transf = np.empty((np.shape(patent_topicDist)[0], np.shape(patent_topicDist)[1] + int(max_topics*2)), dtype=object)
    patent_transf[:, :-int(max_topics*2)] = patent_topicDist

    # Filling the new array
    patent_transf = TransformationMisc.fill_with_topics(patent_transf, topic_list_helper, np.shape(patent_topicDist)[1])

    # check Topic transformation
    if sum(x is not None for x in patent_transf[:,np.shape(patent_transf)[1]-1]) == 0:
        raise Exception("Error: Not all created columns in patent_transf have been filled")



    #--- Append IPC ---#
    print('\n#--- Append IPC ---#\n')

    # check if all patents in patent_transf are unique
    val, count = np.unique(patent_transf[:, 0], return_counts=True)
    if len(val) != len(patent_transf):
        raise Exception("Error: patent_transf contains non-unqiue patents")


    # Find biggest number of IPCs a patent has (new space)
    patent_IPC_clean = [i[0] for i in patent_IPC if i[0] in patent_transf[:, 0]]

    val, count = np.unique(patent_IPC_clean, return_counts=True)
    new_space_needed = max(count) * 3       # x * 3 = x3 additional columns are needed in the new array


    # New array, including space for IPC's
    patent_lda_ipc = np.empty((np.shape(patent_transf)[0], np.shape(patent_transf)[1] + new_space_needed), dtype=object)
    patent_lda_ipc[:, :-new_space_needed] = patent_transf


    # Fill new array
    patent_join = TransformationMisc.fill_with_IPC(patent_lda_ipc, patent_IPC, new_space_needed)


    # check if all created columns are used
    if sum(x is not None for x in patent_join[:, np.shape(patent_join)[1] - 1]) == 0:
        raise Exception("Error: Not all created columns in patent_join have been filled")



    #--- Construct Sliding Windows ---#
    print('\n# --- Construct Sliding Windows ---#\n')

    slidingWindow_dict, patents_perWindow, topics_perWindow, topics_perWindow_unique = Transformation_SlidingWindows.sliding_window_slizing(windowSize, slidingInterval, patent_lda_ipc, max_topics)

    # Descriptives
    print('new latest patent date: ', max(slidingWindow_dict['window_5640'][:,3]))
    print('new earliest patent date: ', min(slidingWindow_dict['window_0'][:,3]), '\n')

    print('Average number of patents per window : ', np.average(patents_perWindow))
    print('Median number of patents per window : ', np.median(patents_perWindow))
    print('Mode number of patents per window : ', statistics.mode(patents_perWindow))
    print('Max number of patents per window : ', max(patents_perWindow))
    print('Min number of patents per window : ', min(patents_perWindow), '\n')

    print('Average number of topics per window : ', np.average(topics_perWindow))
    print('Median number of topics per window : ', np.median(topics_perWindow))
    print('Mode number of topics per window : ', statistics.mode(topics_perWindow))
    print('Max number of topics per window : ', max(topics_perWindow))
    print('Min number of topics per window : ', min(topics_perWindow), '\n')

    print('Average number of unique topics per window : ', np.average(topics_perWindow_unique))
    print('Median number of unique topics per window : ', np.median(topics_perWindow_unique))
    print('Mode number of unique topics per window : ', statistics.mode(topics_perWindow_unique))
    print('Max number of unique topics per window : ', max(topics_perWindow_unique))
    print('Min number of unique topics per window : ', min(topics_perWindow_unique), '\n')


    # Visualization
    x_patent = range(len(patents_perWindow))
    y_patent = patents_perWindow

    x_topic = range(len(topics_perWindow))
    y_topic = topics_perWindow

    fig, ax = plt.subplots()
    ax.plot(x_patent, y_patent, color='darkblue', label='Patents')
    ax.plot(x_topic, y_topic, color='darkred', label='Unique Topics')
    plt.legend(loc='upper left')
    plt.xlabel("Sliding Windows")
    plt.ylabel("Frequency")
    plt.savefig('PatentsAndUniqueTopics_perWindow.png')
    plt.close()


    fig, ax = plt.subplots()
    ax.plot(x_topic, y_topic, color='darkgreen')
    plt.xlabel("Sliding Windows")
    plt.ylabel("Number of unique topics")
    plt.savefig('Topics_perWindow.png')
    plt.close()



    # --- Network transformation ---#
    print('\n# --- Network transformation ---#\n')

    # Prepare nodes for bipartite network
    node_att_name = ['publn_auth', 'publn_nr', 'publn_date', 'publn_claims', 'nb_IPC']

    for i in range(1, int(max_topics) + 1):
        node_att_name.append('TopicID_{0}'.format(i))
        node_att_name.append('TopicCover_{0}'.format(i))

    number_of_ipcs = new_space_needed / 3

    for i in range(1, int(number_of_ipcs) + 1):
        node_att_name.append('IpcID_{0}'.format(i))
        node_att_name.append('IpcSubCat1_{0}'.format(i))
        node_att_name.append('IpcSubCat2_{0}'.format(i))

    # --- Creating a Network Representation for each windows---#

    bipartite_graphs = {}
    patentProject_graphs = {}
    topicProject_graphs = {}

    pbar = tqdm.tqdm(total=len(slidingWindow_dict))

    for window_id, window in slidingWindow_dict.items():

        # Create Graph
        sliding_graph = nx.Graph()

        # Create Nodes - Paents
        nodes = window[:, 0]  # extract patent ids in window
        window_reduc = window[:, np.r_[1:5, 7, 9:len(window.T)]]

        nested_dic = Transformation_Network.prepare_patentNodeAttr_Networkx(window_reduc, nodes, node_att_name)

        sliding_graph.add_nodes_from(nodes, bipartite=1)
        nx.set_node_attributes(sliding_graph, nested_dic)

        # Create Nodes - Topics
        ipc_position = np.r_[range(25, np.shape(patent_lda_ipc)[1] - 1,3)]  # right now, this has to be adjusted manually depending on the LDA results
        topic_position = np.r_[range(9, (9 + int(max_topics * 2)),2)]  # right now, this has to be adjusted manually depending on the LDA results

        topicNode_list = Transformation_Network.prepare_topicNodes_Networkx(window, topic_position)

        sliding_graph.add_nodes_from(topicNode_list, bipartite=0)

        # Create Edges

        num_topics = 3

        topic_edges_list = Transformation_Network.prepare_edgeLists_Networkx(window, num_topics, max_topics)

        for edge_list in topic_edges_list:
            sliding_graph.add_edges_from(edge_list)

        # Project
        top_nodes = {n for n, d in sliding_graph.nodes(data=True) if d["bipartite"] == 0}
        bottom_nodes = set(sliding_graph) - top_nodes

        topicProject_graph = nx.algorithms.bipartite.generic_weighted_projected_graph(sliding_graph, top_nodes, weight_function=Transformation_Network.custom_projection_function)
        patentProject_graph = nx.algorithms.bipartite.generic_weighted_projected_graph(sliding_graph, bottom_nodes, weight_function=Transformation_Network.custom_projection_function)

        # Append to dictionaries
        bipartite_graphs[window_id] = sliding_graph
        patentProject_graphs[window_id] = patentProject_graph
        topicProject_graphs[window_id] = topicProject_graph

        pbar.update(1)

    pbar.close()



    #--- bipartite network descriptives ---#
    topicNode_distribution = []
    edgeWeight_distribution = []
    centralityPatents_distribution = []
    centralityTopics_distribution = []
    density_distribution = []
    active_topicNode_distribution = []
    helper = []
    allTopicNodes_bipart = []

    for window_id, graph in bipartite_graphs.items():
        topicNode_distribution.append(len([n for n, d in graph.nodes(data=True) if d["bipartite"] == 0]))
        edgeWeight_distribution.append(np.average([float(list(w.values())[0]) for (u, v, w) in graph.edges(data=True)]))
        centralityPatents_distribution.append(np.average([graph.degree[n] for n, d in graph.nodes(data=True) if d["bipartite"] == 1]))
        centralityTopics_distribution.append(np.average([graph.degree[n] for n, d in graph.nodes(data=True) if d["bipartite"] == 0]))
        density_distribution.append(nx.density(graph))

        active_topicNode_distribution.append(len(np.unique([v for u, v in graph.edges(data=False)])))

        allTopicNodes_bipart.append([n for n, d in graph.nodes(data=True) if d["bipartite"] == 0])
        allTopicNodes = ([n for n, d in graph.nodes(data=True) if d["bipartite"] == 0])
        allActiveTopicNodes = (np.unique([v for u, v in graph.edges(data=False)]))

        for topicN in allTopicNodes:
            if topicN not in allActiveTopicNodes:
                helper.append(window_id)

    if len(helper) == 0:
        print('Every topic node in every window is active')

    print('---- Bipartite Network Descriptives ----')
    print('Average number of topic nodes per window : ', np.average(topicNode_distribution))
    print('Median number of topic nodes per window : ', np.median(topicNode_distribution))
    print('Mode number of topic nodes per window : ', statistics.mode(topicNode_distribution))
    print('Max number of topic nodes per window : ', max(topicNode_distribution))
    print('Min number of topic nodes per window : ', min(topicNode_distribution), '\n')

    print('Average number of active topic nodes per window : ', np.average(active_topicNode_distribution))
    print('Median number of active topic nodes per window : ', np.median(active_topicNode_distribution))
    print('Mode number of active topic nodes per window : ', statistics.mode(active_topicNode_distribution))
    print('Max number of active topic nodes per window : ', max(active_topicNode_distribution))
    print('Min number of active topic nodes per window : ', min(active_topicNode_distribution), '\n')

    print('Average of average edgeWeight per window : ', np.average(edgeWeight_distribution))
    print('Median of average edgeWeight per window : ', np.median(edgeWeight_distribution))
    print('Mode of average edgeWeight per window : ', statistics.mode(edgeWeight_distribution))
    print('Max of average edgeWeight per window : ', max(edgeWeight_distribution))
    print('Min of average edgeWeight per window : ', min(edgeWeight_distribution), '\n')

    print('Average number of average patent node centrality per window : ', np.average(centralityPatents_distribution))
    print('Median number of average patent node centrality per window : ', np.median(centralityPatents_distribution))
    print('Mode number of average patent node centrality per window : ',statistics.mode(centralityPatents_distribution))
    print('Max number of average patent node centrality per window : ', max(centralityPatents_distribution))
    print('Min number of average patent node centrality per window : ', min(centralityPatents_distribution), '\n')

    print('Average number of average topic node centrality per window : ', np.average(centralityTopics_distribution))
    print('Median number of average topic node centrality per window : ', np.median(centralityTopics_distribution))
    print('Mode number of average topic node centrality per window : ', statistics.mode(centralityTopics_distribution))
    print('Max number of average topic node centrality per window : ', max(centralityTopics_distribution))
    print('Min number of average topic node centrality per window : ', min(centralityTopics_distribution), '\n')

    print('Average network density in window : ', np.average(density_distribution))
    print('Median network density in window : ', np.median(density_distribution))
    print('Mode network density in window : ', statistics.mode(density_distribution))
    print('Max network density in window : ', max(density_distribution))
    print('Min network density in window : ', min(density_distribution), '\n')

    # Visualization (later plotted)
    x_bipart = range(len(density_distribution))
    y_bipart = density_distribution



    # Patent Network Descriptives
    edgeWeight_distribution_patentNetwork = []
    centrality_distribution_patentNetwork = []
    density_distribution_patentNetwork = []

    for window_id, graph in patentProject_graphs.items():
        edgeWeight_distribution_patentNetwork.append(np.average([float(list(w.values())[0]) for (u, v, w) in graph.edges(data=True)]))
        centrality_distribution_patentNetwork.append(np.average([graph.degree[n] for n in graph.nodes()]))
        density_distribution_patentNetwork.append(nx.density(graph))

    print('---- Patent Network Descriptives ----')
    print('Average patent network edge weight per window : ', np.average(edgeWeight_distribution_patentNetwork))
    print('Median patent network edge weight per window : ', np.median(edgeWeight_distribution_patentNetwork))
    print('Mode patent network edge weight per window : ', statistics.mode(edgeWeight_distribution_patentNetwork))
    print('Max patent network edge weight per window : ', max(edgeWeight_distribution_patentNetwork))
    print('Min patent network edge weight per window : ', min(edgeWeight_distribution_patentNetwork), '\n')

    print('Average patent network centrality per window : ', np.average(centrality_distribution_patentNetwork))
    print('Median patent network centrality per window : ', np.median(centrality_distribution_patentNetwork))
    print('Mode patent network centrality per window : ', statistics.mode(centrality_distribution_patentNetwork))
    print('Max patent network centrality per window : ', max(centrality_distribution_patentNetwork))
    print('Min patent network centrality per window : ', min(centrality_distribution_patentNetwork), '\n')

    print('Average patent network density in window : ', np.average(density_distribution_patentNetwork))
    print('Median patent network density in window : ', np.median(density_distribution_patentNetwork))
    print('Mode patent network density in window : ', statistics.mode(density_distribution_patentNetwork))
    print('Max patent network density in window : ', max(density_distribution_patentNetwork))
    print('Min patent network density in window : ', min(density_distribution_patentNetwork), '\n')

    x_patent = range(len(density_distribution_patentNetwork))
    y_patent = density_distribution_patentNetwork



    # Topic Network Descriptives
    edgeWeight_distribution_topicNetwork = []
    centrality_distribution_topicNetwork = []
    density_distribution_topicNetwork = []
    allTopicNodes_inPatentNet = []

    for window_id, graph in topicProject_graphs.items():
        edgeWeight_distribution_topicNetwork.append(np.average([float(list(w.values())[0]) for (u, v, w) in graph.edges(data=True)]))
        centrality_distribution_topicNetwork.append(np.average([graph.degree[n] for n in graph.nodes()]))
        density_distribution_topicNetwork.append(nx.density(graph))
        allTopicNodes_inPatentNet.append([n for n in graph.nodes(data=False)])


    print('---- Topic Network Descriptives ----')
    print('Average topic network edge weight per window : ', np.average(edgeWeight_distribution_topicNetwork))
    print('Median topic network edge weight per window : ', np.median(edgeWeight_distribution_topicNetwork))
    print('Mode topic network edge weight per window : ', statistics.mode(edgeWeight_distribution_topicNetwork))
    print('Max topic network edge weight per window : ', max(edgeWeight_distribution_topicNetwork))
    print('Min topic network edge weight per window : ', min(edgeWeight_distribution_topicNetwork), '\n')

    print('Average topic network centrality per window : ', np.average(centrality_distribution_topicNetwork))
    print('Median topic network centrality per window : ', np.median(centrality_distribution_topicNetwork))
    print('Mode topic network centrality per window : ', statistics.mode(centrality_distribution_topicNetwork))
    print('Max topic network centrality per window : ', max(centrality_distribution_topicNetwork))
    print('Min topic network centrality per window : ', min(centrality_distribution_topicNetwork), '\n')

    print('Average topic network density in window : ', np.average(density_distribution_topicNetwork))
    print('Median topic network density in window : ', np.median(density_distribution_topicNetwork))
    print('Mode topic network density in window : ', statistics.mode(density_distribution_topicNetwork))
    print('Max topic network density in window : ', max(density_distribution_topicNetwork))
    print('Min topic network density in window : ', min(density_distribution_topicNetwork), '\n')

    x_topic = range(len(density_distribution_topicNetwork))
    y_topic = density_distribution_topicNetwork


    fig, ax = plt.subplots()
    ax.plot(x_bipart, y_bipart, color='darkblue', label='Bipartite Networks')
    ax.plot(x_patent, y_patent, color='darkred', label='Patent Networks')
    ax.plot(x_topic, y_topic, color='darkgreen', label='Topic Networks')
    plt.legend(loc='upper left')
    ax.set_ylim([0, 0.04])
    plt.xlabel("Network Representations of Sliding Windows")
    plt.ylabel("Density")
    plt.savefig('Density_All_networks.png')
    plt.close()



    # --- Save Data ---#
    print('\n# --- Save Data ---#\n')

    pd.DataFrame(patent_join).to_csv('patent_lda_ipc.csv', index=False)

    filename = 'slidingWindow_dict'
    outfile = open(filename, 'wb')
    pk.dump(slidingWindow_dict, outfile)
    outfile.close()

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












