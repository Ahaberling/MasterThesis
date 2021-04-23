import pandas as pd
import numpy as np
import re



#--- Initialization ---#

#pd.set_option('display.max_columns', None)

directory = 'D:/Universitaet Mannheim/MMDS 7. Semester/Master Thesis/Outline/Data/Cleaning Robots/'

patent_topicDist    = pd.read_csv(directory + 'patent_topicDist.csv', quotechar='"', skipinitialspace=True)
#topics              = pd.read_csv(directory + 'patent_topics.csv', quotechar='"', skipinitialspace=True)
#parent              = pd.read_csv(directory + 'cleaning_robot_EP_backward_citations.csv', quotechar='"', skipinitialspace=True)

patent_topicDist    = patent_topicDist.to_numpy()
#topics              = topics.to_numpy()
#parent              = parent.to_numpy()



#--- Transformation of topic representation ---#

# Structure up until now:
# One column containing full topic distribution of the patent as list   [(topic_id, coverage),(topic_id, coverage),...] Unordered

# Transformation to:
# three columns for each topic represented in a patent          topic_id, topic_name, topic_coverage, topic_id, ... desc ordered by coverage
# Note: Topic name not inserted yet. 'None' value in place


### Eliminating round brackets ###
topic_reg_list = []

for i in range(len(patent_topicDist)):

    topic_reg = re.findall("(\d*\.*?\d+)", patent_topicDist[i, 9])
    topic_reg_list.append(topic_reg)


### Identify number topics the abstract/s with the most topics has/have to identify size of new array ###
list_len = [len(i) for i in topic_reg_list]
highest_no_topics = max(list_len)/2
new_space_needed = int(highest_no_topics*3)

#print(highest_no_topics)                       # 14 is the maximum of topics abstracts have
#print(new_space_needed)                        # space needed = 42 (14 topic_ids + 14 topic_names + 14 topic_coverages)




### New array ###
patent_topicDist_filled = np.empty((np.shape(patent_topicDist)[0], np.shape(patent_topicDist)[1] + new_space_needed), dtype=object)
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
        patent_topicDist_filled[c, np.shape(patent_topicDist)[1] + l] = tuple_list[k][0]      # topic_id
        # skip 1 column for topic_name (to be added later)
        l = l + 2


        patent_topicDist_filled[c, np.shape(patent_topicDist)[1] + l] = tuple_list[k][1]      # topic_coverage
        l = l + 1

    c = c + 1


#--- Check and save transformation ---#

#print(np.shape(patent_topicDist_filled))              # (3781, 52)
#print(sum(x is not None for x in patent_topicDist_filled[:,51])) # check if all created columns are used (they are)
pd.DataFrame(patent_topicDist_filled).to_csv(directory + 'patent_topicDist_transf.csv', index=None)

# todo append this to lda
