import pandas as pd
import numpy as np
import re
import sys

# --- Initialization ---#

# pd.set_option('display.max_columns', None)

directory = 'D:/Universitaet Mannheim/MMDS 7. Semester/Master Thesis/Outline/Data/Cleaning Robots/'

patent_topicDist = pd.read_csv(directory + 'patent_topicDist.csv', quotechar='"', skipinitialspace=True)
patent_IPC = pd.read_csv(directory + 'cleaning_robot_EP_patents_IPC.csv', quotechar='"', skipinitialspace=True)
parent = pd.read_csv(directory + 'cleaning_robot_EP_backward_citations.csv', quotechar='"', skipinitialspace=True)
parent_IPC = pd.read_csv(directory + 'cleaning_robot_EP_backward_citations_IPC.csv', quotechar='"',
                         skipinitialspace=True)
# topics              = pd.read_csv(directory + 'patent_topics.csv', quotechar='"', skipinitialspace=True)

patent_topicDist = patent_topicDist.to_numpy()
patent_IPC = patent_IPC.to_numpy()
parent = parent.to_numpy()
parent_IPC = parent_IPC.to_numpy()
# topics              = topics.to_numpy()

print(patent_topicDist)

# --- Transformation of topic representation ---#

# Structure up until now:
# One column containing full topic distribution of the patent as list   [(topic_id, coverage),(topic_id, coverage),...] Unordered

# Transformation to:
# three columns for each topic represented in a patent          topic_id, topic_name, topic_coverage, topic_id, ... desc ordered by coverage
# Note: Topic name not inserted yet. 'None' value in place


### Eliminating round brackets ###
topic_reg_list = []

for i in range(len(patent_topicDist)):
    topic_reg = re.findall("(\d*\.*?\d+)", patent_topicDist[i, 8])
    topic_reg_list.append(topic_reg)

### Identify number topics the abstract/s with the most topics has/have to identify size of new array ###
list_len = [len(i) for i in topic_reg_list]
highest_no_topics = max(list_len) / 2
new_space_needed = int(highest_no_topics * 3)

# print(highest_no_topics)                       # 14 is the maximum of topics abstracts have
# print(new_space_needed)                        # space needed = 42 (14 topic_ids + 14 topic_names + 14 topic_coverages)


### New array ###
patent_topicDist_filled = np.empty((np.shape(patent_topicDist)[0], np.shape(patent_topicDist)[1] + new_space_needed),
                                   dtype=object)
patent_topicDist_filled[:, :-new_space_needed] = patent_topicDist

### Filling new array ###
topic_reg_list_tuple = []

c = 0
for i in topic_reg_list:

    # Create tuple list of topic_id and coverage to sort by coverage
    tuple_list = []

    for j in range(0, len(i) - 1, 2):
        tuple = (i[j], i[j + 1])
        tuple_list.append(tuple)

    '''
    ggg = patent_topicDist[c, 9]
    print(tuple_list)                      
    print(ggg)
    '''
    # todo this is the same except for the ''. upper code seems kinda redundant

    # Sort by coverage
    tuple_list = sorted(tuple_list, key=lambda tup: tup[1], reverse=True)

    # Insert values ordered in new array
    l = 0

    for k in range(len(tuple_list)):
        # np.shape(patent_topicDist)[1] represent the number of filled columns the patent_topicDist_filled array already has
        # + l because the new data is appended to the empty columns following the filled ones
        patent_topicDist_filled[c, np.shape(patent_topicDist)[1] + l] = tuple_list[k][0]  # topic_id
        # skip 1 column for topic_name (to be added later)
        l = l + 2

        patent_topicDist_filled[c, np.shape(patent_topicDist)[1] + l] = tuple_list[k][1]  # topic_coverage
        l = l + 1

    c = c + 1

# --- Check transformation ---#

# print(np.shape(patent_topicDist_filled))              # (3781, 52)
# print(sum(x is not None for x in patent_topicDist_filled[:,51])) # check if all created columns are used (they are)


# --- Append IPC ---#

# Each Patent has at least one IPC. These IPC'S are appended to the patent_topicDist_filled array in order to facilitate
# a brief, heuristic evaluation of topic modeling. Additionally they might be used to conceptualize first basic
# recombination and diffusion measures

# Structure aimed for:
# columns of patent_topicDist_filled, IPC, subcategory, subcategory, IPC, subcategory, subcategory, IPC, ...


### Review patent_topicDist_filled and patent_IPC ###

val, count = np.unique(patent_topicDist_filled[:, 0], return_counts=True)
# print(len(val))                            # 3781
# print(len(patent_topicDist_transf))        # 3781 -> all ids of patent_topicDist_transf are unique

val, count = np.unique(patent_IPC[:, 0], return_counts=True)
# print(len(val))                             # 3844 -> There are patents in patent_IPC that have been cleaned earlier (german and france abstracts)
# print(len(patent_IPC))                      # 9596


### Clean patent_IPC (remove patents with german and france abstracts)
patent_IPC_clean = [i[0] for i in patent_IPC if i[0] in patent_topicDist_filled[:, 0]]

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


### Prepare new array patent_join ###
patent_join = np.empty((np.shape(patent_topicDist_filled)[0], np.shape(patent_topicDist_filled)[1] + new_space_needed),
                       dtype=object)
patent_join[:, :-new_space_needed] = patent_topicDist_filled

# print(np.shape(patent_topicDist_filled))    # (3781, 52)
# print(np.shape(patent_join))                # (3781, 91)

### Fill new array ###

count_list = []
count_l = 0

for i in patent_IPC:

    if i[0] in patent_join[:,
               0]:  # For each row in patent_IPC, check if id in patent_join (identical to patent_topicDist_filled)
        count_l = count_list.count(
            i[0])  # Retrieve how often the id has been seen yet (how often ipc's where appended already

        # if patent_join[patent_join[:,0] == i[0],-(new_space_needed-count_l*3)] == None:

        patent_join[patent_join[:, 0] == i[0], -(new_space_needed - count_l * 3)] = i[1]
        patent_join[patent_join[:, 0] == i[0], -(new_space_needed - count_l * 3 - 1)] = i[2]
        patent_join[patent_join[:, 0] == i[0], -(new_space_needed - count_l * 3 - 2)] = i[3]

    count_list.append(i[0])

### check if all created columns are used ###
# print(sum(x is not None for x in patent_join[:,90]))       they are
# print(patent_join[patent_join[:,0] == 478449443])
# print(patent_join[patent_join[:,0] == val[np.argmax(count)]])

np.set_printoptions(threshold=sys.maxsize)

print(patent_join[100])

# --- Save transformation and IPC appendix---#

pd.DataFrame(patent_join).to_csv(directory + 'patent_lda_ipc.csv', index=None)


