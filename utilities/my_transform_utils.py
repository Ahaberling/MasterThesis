import re
import numpy as np

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
    def fill_with_IPC(array_toBeFilled, ipcs, max_noIPC):
        count_list = []
        count_l = 0

        for i in ipcs:

            if i[0] in array_toBeFilled[:,
                       0]:  # For each row in patent_IPC, check if id in patent_join (identical to patent_transf)
                count_l = count_list.count(
                    i[0])  # Retrieve how often the id has been seen yet (how often ipc's where appended already

                # if patent_join[patent_join[:,0] == i[0],-(new_space_needed-count_l*3)] == None:

                array_toBeFilled[array_toBeFilled[:, 0] == i[0], -(max_noIPC - count_l * 3)] = i[1]
                array_toBeFilled[array_toBeFilled[:, 0] == i[0], -(max_noIPC - count_l * 3 - 1)] = i[2]
                array_toBeFilled[array_toBeFilled[:, 0] == i[0], -(max_noIPC - count_l * 3 - 2)] = i[3]

            count_list.append(i[0])

        array_filled = array_toBeFilled

        return array_filled