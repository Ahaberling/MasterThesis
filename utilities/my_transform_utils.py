import re
import numpy as np
import tqdm

class Transf_misc:

    @staticmethod
    def max_number_topics(dataset):
        ### Eliminating round brackets ###
        # e.g. [(45, 0.06), (145, 0.05), ...] to ['45', '0.06', '145', '0.05']

        transf_list = []
        for i in range(len(dataset)):
            topic_transf = re.findall("(\d*\.*?\d+)", dataset[i, len(dataset.T)-1])
            transf_list.append(topic_transf)

        ### Identify the number of topics the abstract/s with the most topics has/have ###

        list_len = [len(i) for i in transf_list]
        max_topics = max(list_len) / 2

        return transf_list, max_topics

    @staticmethod
    def fill_with_topics(array_toBeFilled, topic_list, column_start):

        c = 0
        for i in topic_list:

            # Create tuple list of topic_id and coverage to sort by coverage #
            tuple_list = []

            # e.g. ['45', '0.06', '145', '0.05'] to [('45', '0.06'), ('145', '0.05'), ...]
            for j in range(0, len(i) - 1, 2):
                tuple = (i[j], i[j + 1])
                tuple_list.append(tuple)

            # tod o: check code above for redundancy.('-signs)

            # Sort by coverage #
            tuple_list = sorted(tuple_list, key=lambda tup: tup[1], reverse=True)

            # Insert values ordered in new array #
            l = 0

            for k in range(len(tuple_list)):
                # np.shape(patent_topicDist)[1] represent the number of filled columns the patent_transf array already has
                # + l because the new data is appended to the empty columns following the filled ones
                array_toBeFilled[c, column_start + l] = tuple_list[k][0]  # topic_id

                # skip 1 column for topic_name (to be added later)
                l = l + 1

                array_toBeFilled[c, column_start + l] = tuple_list[k][1]  # topic_coverage
                l = l + 1

            c = c + 1

        array_filled = array_toBeFilled

        return array_filled

    @staticmethod
    def fill_with_IPC(array_toBeFilled, patent_IPC, max_numIPC):
        count_list = []
        count_l = 0

        for i in patent_IPC:

            if i[0] in array_toBeFilled[:, 0]:  # For each row in patent_IPC, check if id in patent_join (identical to patent_transf)
                count_l = count_list.count(i[0])  # Retrieve how often the id has been seen yet (how often ipc's where appended already

                # if patent_join[patent_join[:,0] == i[0],-(new_space_needed-count_l*3)] == None:

                array_toBeFilled[array_toBeFilled[:, 0] == i[0], -(max_numIPC - count_l * 3)] = i[1]
                array_toBeFilled[array_toBeFilled[:, 0] == i[0], -(max_numIPC - count_l * 3 - 1)] = i[2]
                array_toBeFilled[array_toBeFilled[:, 0] == i[0], -(max_numIPC - count_l * 3 - 2)] = i[3]

            count_list.append(i[0])

        array_filled = array_toBeFilled

        return array_filled

class Transf_slidingWindow:

    @staticmethod
    def sliding_window_slizing(windowSize, slidingInterval, array_toBeSlized):

        array_time = array_toBeSlized[:, 3].astype('datetime64')

        array_time_unique = np.unique(array_time)  # 817
        array_time_unique_filled = np.arange(np.min(array_time_unique), np.max(array_time_unique))  # 6027
        array_time_unique_filled_windowSize = array_time_unique_filled[
            array_time_unique_filled <= max(array_time_unique_filled) - windowSize]  # 5662

        slidingWindow_dict = {}
        patents_perWindow = []
        topics_perWindow = []
        uniqueTopics_inAllWindows = []

        c = 0
        pbar = tqdm.tqdm(total=len(array_time_unique_filled_windowSize))

        for i in array_time_unique_filled_windowSize:

            if c % slidingInterval == 0:
                lower_limit = i
                upper_limit = i + windowSize

                array_window = array_toBeSlized[(array_toBeSlized[:, 3].astype('datetime64') < upper_limit) & (
                        array_toBeSlized[:, 3].astype('datetime64') >= lower_limit)]
                patents_perWindow.append(len(array_window))

                topics_perWindow_helper = []
                for topic_list in array_window[:,9:23]:
                    for column_id in range(0,len(topic_list.T),2):
                        if topic_list[column_id] != None:
                            topics_perWindow_helper.append(topic_list[column_id])

                topics_perWindow_helper = np.unique(topics_perWindow_helper)


                topics_perWindow.append(len(topics_perWindow_helper))
                uniqueTopics_inAllWindows.append(topics_perWindow_helper)

                slidingWindow_dict['window_{0}'.format(c)] = array_window

            c = c + 1
            pbar.update(1)

        pbar.close()

        uniqueTopics_inAllWindows = [item for sublist in uniqueTopics_inAllWindows for item in sublist]
        uniqueTopics_inAllWindows = np.unique(uniqueTopics_inAllWindows)

        return slidingWindow_dict, patents_perWindow, topics_perWindow, uniqueTopics_inAllWindows


class Transf_network:

    @staticmethod
    def test_weight(G, u, v):

        u_nbrs = set(G[u])      # Neighbors of Topic1 in set format for later intersection
        v_nbrs = set(G[v])      # Neighbors of Topic2 in set format for later intersection
        shared_nbrs = u_nbrs.intersection(v_nbrs)       # Shared neighbors of both topic nodes (intersection)
        #if len(shared_nbrs) >= 2:
            #print(1+1)

        list_of_poducts = []
        for i in shared_nbrs:

            weight1 = list(G.edges[u,i].values())[0]
            weight2 = list(G.edges[v,i].values())[0]

            list_of_poducts.append(float(weight1) * float(weight2))

        projected_weight = sum(list_of_poducts) / len(list_of_poducts)

        return projected_weight

    @staticmethod
    def prepare_patentNodeAttr_Networkx(window, nodes, node_att_name):

        node_att_dic_list = []

        for i in range(len(window)):

            dic_entry = dict(enumerate(
                window[i]))  # Here each patent is converted into a dictionary. dictionary keys are still numbers:
            # {0: 'EP', 1: nan, 2: '2007-10-10', ...} Note that the patent id is ommited, since it
            # serves as key for the outer dictionary encapsulating these inner once.

            for key, n_key in zip(dic_entry.copy().keys(),
                                  node_att_name):  # Renaming the keys of the inner dictionary from numbers to actually names saved in node_plain_att
                dic_entry[n_key] = dic_entry.pop(
                    key)  # {'publn_auth': 'EP', 'publn_nr': nan, 'publn_date': '2009-06-17', ...}

            node_att_dic_list.append(dic_entry)

        nested_dic = dict(enumerate(
            node_att_dic_list))  # Here the nested (outer) dictionary is created. Each key is still represented as a number, each value as another dictionary

        for key, n_key in zip(nested_dic.copy().keys(),
                              nodes):  # Here the key of the outer dictionary are renamed to the patent ids
            nested_dic[n_key] = nested_dic.pop(key)

        # print(len(window))

        return nested_dic

    @staticmethod
    def prepare_topicNodes_Networkx(window, topic_position):
        topics_inWindow = []

        for patent in window:
            topics_inWindow.append(patent[topic_position])

        #print(topics_inWindow)
        topics_inWindow = [item for sublist in topics_inWindow for item in sublist]
        #print(topics_inWindow)
        #topics_inWindow = list(filter(lambda x: x == x, topics_inWindow))
        topics_inWindow = [x for x in topics_inWindow if x is not None]
        #print(topics_inWindow)
        topics_inWindow = np.unique(topics_inWindow)
        #print(topics_inWindow)

        topicNode_list = ['topic_{0}'.format(int(i)) for i in topics_inWindow]
        return topicNode_list


    @staticmethod
    def prepare_edgeLists_Networkx(window, num_topics, max_topics):

        if num_topics >= max_topics + 1:
            raise Exception("num_topics must be <= max_topics")

        edges = window[:, np.r_[0, 9:(9 + (num_topics * 2))]]  # first three topics

        topic_edges_list = []

        for i in range(1, (num_topics * 2)+1, 2):

            c = 0
            for j in edges.T[i]:
                # if np.isfinite(i):
                if j != None:
                    edges[c, i] = 'topic_{0}'.format(int(j))
                c = c + 1

            topic_edges = [(j[0], j[i], {'Weight': j[i + 1]}) for j in edges]
            topic_edges_clear = list(filter(lambda x: x[1] != None, topic_edges))

            topic_edges_list.append(topic_edges_clear)

        return topic_edges_list