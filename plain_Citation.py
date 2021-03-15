import numpy as np
import pandas as pd
import matplotlib as plt
import networkx as nx

import sys
import io

pd.set_option('display.max_columns', None)

patent = pd.read_csv(r'D:\Universitaet Mannheim\MMDS 7. Semester\Master Thesis\Outline\Data\Cleaning Robots\cleaning_robot_EP_patents.csv', quotechar='"', skipinitialspace=True)
patent_IPC = pd.read_csv(r'D:\Universitaet Mannheim\MMDS 7. Semester\Master Thesis\Outline\Data\Cleaning Robots\cleaning_robot_EP_patents_IPC.csv', quotechar='"', skipinitialspace=True)

parent = pd.read_csv(r'D:\Universitaet Mannheim\MMDS 7. Semester\Master Thesis\Outline\Data\Cleaning Robots\cleaning_robot_EP_backward_citations.csv', quotechar='"', skipinitialspace=True)
parent_IPC = pd.read_csv(r'D:\Universitaet Mannheim\MMDS 7. Semester\Master Thesis\Outline\Data\Cleaning Robots\cleaning_robot_EP_backward_citations_IPC.csv', quotechar='"', skipinitialspace=True)



# keep data where pat_publn_id =! 0 Dont forget if 0 really means that parents are not patents and not in the data
# check if parent.pat_publn_id and patent.pat_publn_id contain 0. They should not
# check if patent.pat_publn_id is unique, it should be
# create list of tuple with (source, target) of an edge
# feed this to the the networkx
# check if number of nodes and edges are correct

# parent.pat_publn_id has no 0




### Checking for 0 in patent.pat_publn_id (ivalid rows)

print(len(patent.pat_publn_id))           #3844
print(len(patent.pat_publn_id.unique()))  #3844

pat_publn_id_no0 = patent.pat_publn_id[patent.pat_publn_id != 0]

print(len(pat_publn_id_no0))              #3844
print(len(pat_publn_id_no0.unique()))     #3844


### Checking for 0 in parent.cited_pat_publn_id (ivalid rows)

print(len(parent.cited_pat_publn_id))           #18548
print(len(parent.cited_pat_publn_id.unique()))  #13315

cited_pat_publn_id_no0 = parent.cited_pat_publn_id[parent.cited_pat_publn_id != 0]

print(len(cited_pat_publn_id_no0))              #16638
print(len(cited_pat_publn_id_no0.unique()))     #13314


### Checking for 0 in parent.cited_appln_id (not sure yet, what that means)
print(len(parent.cited_appln_id))           #18548
print(len(parent.cited_appln_id.unique()))  #191

cited_appln_id_no0 = parent.cited_appln_id[parent.cited_appln_id != 0]

print(len(cited_appln_id_no0))              #216
print(len(cited_appln_id_no0.unique()))     #190


### Checking for 0 in parent.pat_publn_id (Number of unique should be qual to 3844, because there are 3844 patents in df patents)
# maybe 3844-3390=454 patents dont cite anything? #todo ask leo about that
print(len(parent.pat_publn_id))           #18548
print(len(parent.pat_publn_id.unique()))  #3390

pat_publn_id_no0 = parent.pat_publn_id[parent.pat_publn_id != 0]

print(len(pat_publn_id_no0))              #18548
print(len(pat_publn_id_no0.unique()))     #3390


# excluding all entries with parent.cited_pat_publn_id == 0, in order to create the network afterwards with edge information

parent_cited_no0 = parent[parent.cited_pat_publn_id != 0]

print(len(parent_cited_no0))

### building the networks via edge info of parent:

### adding the missing nodes out of patent. The number of nodes added should be eagl to 3844-3390=454 (presumably isolated nodes)


plainC = nx.Graph()

#plainC.add_nodes_from(parent.pat_publn_id)

#print(list(plainC.nodes))
#print(len(list(plainC.nodes)))

tuple_list = []

for index, row in parent.iterrows():
    #print(row.pat_publn_id, row.cited_pat_publn_id)
    tup = (row.pat_publn_id, row.cited_pat_publn_id)
    tuple_list.append(tup)

print("Number of tuples: ", len(tuple_list))                          #18548
print("Number of unique tuples: ", len(set(tuple_list)))              #16612
# Structure: (id, cited_id) or (source, target) or (paper, cited_paper)
# X nodes and 16612 edges should be added with this tuplelist

plainC.add_edges_from(tuple_list)

print("Number of Nodes added: ", len(list(plainC.nodes)))
print("Number of Edges added: ", len(list(plainC.edges)))

# check how many unique row.pat_publn_id, row.cited_pat_publn_id there are


### Now adding the isolated (?) nodes
print(len(list(plainC.nodes)))

plainC.add_nodes_from(parent.pat_publn_id)

#print(list(plainC.nodes))
print(len(list(plainC.nodes)))