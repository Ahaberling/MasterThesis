### This file relies on the results of v3_textPreproc_LDA_gridSeach.py. It takes the patent data set enriched with the LDA results
### and transforms the later into a more accessable structure. Without transformation the topic affiliation of each
### document is stored as list of tuples, ordered by topic_id (asc):     [(topic_id, coverage),(topic_id, coverage),...]
###
### Input:
### pat_publn_id, ... publn_abstract, nb_IPC, LDA_results
### Output:
### pat_publn_id, ... publn_abstract, nb_IPC, topic_id, topic_name, coverage, topic_id, ...   (ordered desc by coverage)
###
### Note: topic_name not inserted yet. 'None' value in place
###
### Futher more the IPC's of each patent (stored in cleaning_robot_EP_patents_IPC.csv) are added to the dataset in a
### similar manner
###
### Output:
### pat_publn_id, ... publn_abstract, nb_IPC, topic_id, ... , ipc_identifier, ipc_subcategory, ipc_subcategory2, ipc_identifier, ...


if __name__ == '__main__':

#--- Import Libraries ---#
    print('\n#--- Import Libraries ---#\n')

    import numpy as np
    import pandas as pd

    import os
    import sys

    import re


# --- Initialization ---#
    print('\n# --- Initialization ---#\n')

    os.chdir('D:/Universitaet Mannheim/MMDS 7. Semester/Master Thesis/Outline/Data/Cleaning Robots')

    #patent_topicDist = pd.read_csv('patent_topicDist_gensim.csv', quotechar='"', skipinitialspace=True)
    patent_topicDist = pd.read_csv('patent_topicDist_mallet.csv', quotechar='"', skipinitialspace=True)
    patent_IPC = pd.read_csv('cleaning_robot_EP_patents_IPC.csv', quotechar='"', skipinitialspace=True)


    patent_topicDist = patent_topicDist.to_numpy()
    patent_IPC = patent_IPC.to_numpy()


# --- Transformation of topic representation ---#
    print('\n# --- Transformation of topic representation ---#\n')


    ### Eliminating round brackets ###

    topic_transf_list = []
    for i in range(len(patent_topicDist)):
        topic_transf = re.findall("(\d*\.*?\d+)", patent_topicDist[i, 8])
        topic_transf_list.append(topic_transf)


    ### Identify the number of topics the abstract/s with the most topics has/have in order to identify the size of the new array ###

    list_len = [len(i) for i in topic_transf_list]
    highest_no_topics = max(list_len) / 2
    new_space_needed = int(highest_no_topics * 3)

    print('Maximum number of topics a patent has: ', highest_no_topics)         # 7 is the maximum of topics abstracts have
    print('Hence, number of new columns needed: ', new_space_needed)            # space needed = 21 (7 topic_ids + 7 topic_names + 7 topic_coverages)


    ### New array, including space for transformation ###

    patent_transf = np.empty((np.shape(patent_topicDist)[0], np.shape(patent_topicDist)[1] + new_space_needed), dtype=object)
    patent_transf[:, :-new_space_needed] = patent_topicDist


    ### Filling the new array ###

    c = 0
    for i in topic_transf_list:

        # Create tuple list of topic_id and coverage to sort by coverage #
        tuple_list = []

        for j in range(0, len(i) - 1, 2):
            tuple = (i[j], i[j + 1])
            tuple_list.append(tuple)

        #todo: check code above for redundancy.('-signs)

        # Sort by coverage #
        tuple_list = sorted(tuple_list, key=lambda tup: tup[1], reverse=True)

        # Insert values ordered in new array #
        l = 0

        for k in range(len(tuple_list)):

            # np.shape(patent_topicDist)[1] represent the number of filled columns the patent_transf array already has
            # + l because the new data is appended to the empty columns following the filled ones
            patent_transf[c, np.shape(patent_topicDist)[1] + l] = tuple_list[k][0]  # topic_id

            # skip 1 column for topic_name (to be added later)
            l = l + 2

            patent_transf[c, np.shape(patent_topicDist)[1] + l] = tuple_list[k][1]  # topic_coverage
            l = l + 1

        c = c + 1



# --- Check transformation ---#
    print('\n# --- Check transformation ---#\n')

    print('Shape of new array: ', np.shape(patent_transf))                                  # (3781, 30)
    print('Are all new columns used? Number of patents with maximum number of topics: ',
          sum(x is not None for x in patent_transf[:,np.shape(patent_transf)[1]-1]))        # 1



# --- Append IPC ---#
    print('\n# --- Append IPC ---#\n')

    # Each Patent has at least one IPC. These IPC'S are appended to the patent_transf array in order to facilitate
    # a brief, heuristic evaluation of topic modeling. Additionally they might be used to conceptualize first basic
    # recombination and diffusion measures


    ### Review patent_transf and patent_IPC ###
    print('Review patent_transf and patent_IPC:\n')

    val, count = np.unique(patent_transf[:, 0], return_counts=True)
    print('All patent ids in patent_topicDist_x are unique if: ', len(val), ' == ', len(patent_transf))


    val, count = np.unique(patent_IPC[:, 0], return_counts=True)
    print('Patent_IPC contains: ', len(val), 'unique id\'s. This is more then patent_topicDist_x, since the later was cleaned of german and fransh patents')
    print('Patent_IPC contains: ', len(patent_IPC), ' rows overall. Patents can be categorized with more than one IPC')


    ### Clean patent_IPC (remove patents with german and france abstracts) ###

    patent_IPC_clean = [i[0] for i in patent_IPC if i[0] in patent_transf[:, 0]]

    #print(len(np.unique(patent_IPC[:,1])))        #970

    val, count = np.unique(patent_IPC_clean, return_counts=True)
    # print(len(val))                             # 3781
    # print(len(patent_IPC_clean))                # 9449
    # print(max(count))                           # 13 -> patents have at most 13 IPC's

    # IPC's consist of 3 colums (IPC, subcategory, subcategory)
    # 13 * 3 = 39 additional columns are needed in the new array

    new_space_needed = max(count) * 3  # 39

    # print(np.argmax(count))                     #      3485 index of highest value in count
    # print(val[np.argmax(count)])                # 478449443 id of with highest value in count
    # print(patent_IPC[patent_IPC[:,0] == val[np.argmax(count)]])


    ### New array, including space for IPC's ###

    patent_join = np.empty((np.shape(patent_transf)[0], np.shape(patent_transf)[1] + new_space_needed), dtype=object)
    patent_join[:, :-new_space_needed] = patent_transf

    # print(np.shape(patent_transf))            # (3781, 52)
    # print(np.shape(patent_join))              # (3781, 91)

    ### Fill new array ###

    count_list = []
    count_l = 0

    for i in patent_IPC:

        if i[0] in patent_join[:,0]:  # For each row in patent_IPC, check if id in patent_join (identical to patent_transf)
            count_l = count_list.count(i[0])  # Retrieve how often the id has been seen yet (how often ipc's where appended already

            # if patent_join[patent_join[:,0] == i[0],-(new_space_needed-count_l*3)] == None:

            patent_join[patent_join[:, 0] == i[0], -(new_space_needed - count_l * 3)] = i[1]
            patent_join[patent_join[:, 0] == i[0], -(new_space_needed - count_l * 3 - 1)] = i[2]
            patent_join[patent_join[:, 0] == i[0], -(new_space_needed - count_l * 3 - 2)] = i[3]

        count_list.append(i[0])

    ### check if all created columns are used ###

    print('Are all new columns used? Number of patents with maximum number of IPC\'S: ',
          sum(x is not None for x in patent_join[:, np.shape(patent_join)[1] - 1]))

    # print(sum(x is not None for x in patent_join[:,90]))       they are
    # print(patent_join[patent_join[:,0] == 478449443])
    # print(patent_join[patent_join[:,0] == val[np.argmax(count)]])

    np.set_printoptions(threshold=sys.maxsize)

    #print(patent_join[100])

# --- Save transformation and IPC appendix---#
    print('\n# --- Save transformation and IPC appendix---#\n')

    #print('Preview of the resulting Array:\n\n', patent_join[0])                        #  [12568 'EP' 1946896.0 '2008-07-23' 15
                                                                                        # 'Method for adjusting at least one axle'
                                                                                        # 'An adjustment method for at least one axis (10) in which a robot has a control unit (12) for controlling an axis (10) via which at least two component parts (16,18) are mutually movable. The component parts (16,18) each has at least one marker (24,26) and the positions of the markers are detected by a sensor, with the actual value of a characteristic value ascertained as a relative position of the two mutually movable components (16,18). The adjustment position is repeated by comparing an actual value with a stored, desired value for the adjustment position. An independent claim is included for a device with signal processing unit.'
                                                                                        #  1 '[(139, 0.05602006688963211)]' '139' None '0.05602006688963211' None
                                                                                        #  None None None None None None None None None None None None None None
                                                                                        #  None None None 'B25J   9' 'Handling' 'Mechanical engineering' None None
                                                                                        #  None None None None None None None None None None None None None None
                                                                                        #  None None None None None None None None None None None None None None
                                                                                        #  None None None None None None]

    pd.DataFrame(patent_join).to_csv('patent_lda_ipc.csv', index=None)


