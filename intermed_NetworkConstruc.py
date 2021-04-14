import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

pd.set_option('display.max_columns', None)

patent_topicDist = pd.read_csv(r'D:\Universitaet Mannheim\MMDS 7. Semester\Master Thesis\Outline\Data\Cleaning Robots\cleaning_robot_EP_patents_07_04_topicNameMissing.csv', quotechar='"', skipinitialspace=True)
topics = pd.read_csv(r'D:\Universitaet Mannheim\MMDS 7. Semester\Master Thesis\Outline\Data\Cleaning Robots\cleaning_robot_EP_patents_07_04_topics.csv', quotechar='"', skipinitialspace=True)
parent = pd.read_csv(r'D:\Universitaet Mannheim\MMDS 7. Semester\Master Thesis\Outline\Data\Cleaning Robots\cleaning_robot_EP_backward_citations.csv', quotechar='"', skipinitialspace=True)
#print(parent)


patent_topicDist = patent_topicDist.to_numpy()
parent = parent.to_numpy()
#print(patent_topicDist)


cited_pat_publn_id = parent[:,(0,2)]
print(cited_pat_publn_id)
print(len(cited_pat_publn_id))
print(len(parent))

#todo gibt es node ids in parent[:,(0,2)], die nicht in patent_topicDist auftauchen? Falls ja, how do we handle them?
#todo how do we handle patents that cite things that are not patents (parent[,2] = 0)?

#print(patent_topicDist)
#print(patent_topicDist[-2])
#print(patent_topicDist.T[0])



#dict(enumerate(arr.sum(axis=1)))

#test_dict = dict(patent_topicDist[0,:])

#top_dic = dict(enumerate(patent_topicDist[:,1:]))
#top_dic = patent_topicDist

patent_topicDist_prep = patent_topicDist[:,1:]



inner_keys = ['publn_auth', 'publn_nr', 'publn_date', 'publn_claims', 'publn_title',
                    'publn_abstract', 'nb_IPC', 'abstract_clean', 'topic_list']
outer_keys = patent_topicDist[:,0]

helper = int(len(patent_topicDist_prep.T[9:,:])/3)

for i in range(1,helper+1):
    inner_keys.append('{0}_topicID'.format(i))
    inner_keys.append('{0}_topicName'.format(i))
    inner_keys.append('{0}_topicCover'.format(i))
    #print(i)

#print(len(inner_keys))



#inner_dic = [dict(enumerate(patent_topicDist_prep[i])) for i in range(len(patent_topicDist_prep))]

inner_dic = []
for i in range(len(patent_topicDist_prep)):

    helper = dict(enumerate(patent_topicDist_prep[i]))
    for key, n_key in zip(helper.copy().keys(), inner_keys):
        helper[n_key] = helper.pop(key)

    inner_dic.append(helper)

#print( inner_dic == inner_dic2)

#print(top_dic[0])
#print(dict(enumerate(top_dic[0])))
#print(inner_dic)
'''
for key, n_key in zip(inner_dic.copy().keys(), inner_keys):
    inner_dic[n_key] = inner_dic.pop(key)
    # print(key,n_key)
'''


outer_dic = dict(enumerate(inner_dic))

#print(outer_dic)

'''
print((len(patent_topicDist_prep.T[9:,:])+1)/3)
print(len(patent_topicDist_prep.T[9:,:])+1)
'''

'''
d = {}
for x in range(1, 10):
    d["string{0}".format(x)] = "Hello"
'''

#print(len(patent_topicDist_prep.T[9:,:]))
#print((len(patent_topicDist_prep.T[9:,:]))/3)
#print(patent_topicDist_prep.T[9:,:])

#print(outer_keys)
#print(len(outer_keys))
#print(outer_dic)
#print(len(outer_dic))


for key, n_key in zip(outer_dic.copy().keys(), outer_keys):
    outer_dic[n_key] = outer_dic.pop(key)
    # print(key,n_key)

#print(outer_dic[487839054])
#print(len(outer_dic[487839054]))
#print(outer_dic[487839054].keys())
#print(inner_keys)
#print(len(inner_keys))

#print(outer_dic)

#print(outer_dic)
#print(outer_dic.values())
#487838945 487838990 487839054
'''
487838990: {0: 'EP', 1: 3275601.0, 2: '2018-01-31', 3: 8, 4: 'ROBOT AND GEAR DEVICE', 5: 'A robot (100) includes a first member (111), a second member (121) provided to be capable of turning with respect to the first member (111), and a gear device (1) configured to transmit a driving force from one side to the other side of the first member (111) and the second member (121). The gear device (1) includes internal teeth (23) and external teeth (33) provided halfway in a transmission path of the driving force and configured to mesh with each other and lubricant (51) disposed between the internal teeth (23) and the external teeth (33). An average grain size of a constituent material of the external teeth (33) is smaller than an average grain size of a constituent material of the internal teeth (23).', 6: 3, 7: 'robot includ first member second member provid capabl turn respect first member gear devic configur transmit drive forc one side side first member second member gear devic includ intern teeth extern teeth provid halfway transmiss path drive forc configur mesh lubric dispos intern teeth extern teeth averag grain size constitu materi extern teeth smaller averag grain size constitu materi intern teeth', 8: '[(1, 0.13103338), (68, 0.058549438), (93, 0.053261172), (172, 0.26179296), (234, 0.099221155), (241, 0.05149606), (297, 0.05921663)]', 9: 172, 10: nan, 11: 0.26179296, 12: 1.0, 13: nan, 14: 0.13103338, 15: 234.0, 16: nan, 17: 0.099221155, 18: 297.0, 19: nan, 20: 0.05921663, 21: 68.0, 22: nan, 23: 0.058549438, 24: 93.0, 25: nan, 26: 0.053261172, 27: 241.0, 28: nan, 29: 0.05149606, 30: nan, 31: nan, 32: nan, 33: nan, 34: nan, 35: nan, 36: nan, 37: nan, 38: nan, 39: nan, 40: nan, 41: nan, 42: nan, 43: nan, 44: nan, 45: nan, 46: nan, 47: nan, 48: nan, 49: nan, 50: nan}
487839054: {0: 'EP', 1: 3275603.0, 2: '2018-01-31', 3: 9, 4: 'CONTROL DEVICE, ROBOT, AND ROBOT SYSTEM', 5: 'A control device which controls a robot having a moving part includes: a control unit which causes an end effector provided on the moving part to move an insertion object, bring the insertion object into contact with an insertion hole provided in an insertion target object in the state where the insertion object is tilted from a center axis of the insertion hole, and subsequently insert the insertion object into the insertion hole.', 6: 1, 7: 'control devic control robot move part includ control unit caus end effector provid move part move insert object bring insert object contact insert hole provid insert target object state insert object tilt center axi insert hole subsequ insert insert object insert hole', 8: '[(27, 0.12415153), (126, 0.1388967), (129, 0.05026767), (167, 0.32650527), (204, 0.120582916)]', 9: 167, 10: nan, 11: 0.32650527, 12: 126.0, 13: nan, 14: 0.1388967, 15: 27.0, 16: nan, 17: 0.12415153, 18: 204.0, 19: nan, 20: 0.120582916, 21: 129.0, 22: nan, 23: 0.05026767, 24: nan, 25: nan, 26: nan, 27: nan, 28: nan, 29: nan, 30: nan, 31: nan, 32: nan, 33: nan, 34: nan, 35: nan, 36: nan, 37: nan, 38: nan, 39: nan, 40: nan, 41: nan, 42: nan, 43: nan, 44: nan, 45: nan, 46: nan, 47: nan, 48: nan, 49: nan, 50: nan}
'''
'''
target_dict = {'k1':'v1', 'k2':'v2', 'k3':'v3'}
new_keys = ['k4','k5','k6']

for i in range(0,100):
    for key,n_key in zip(target_dict.keys(), new_keys):
        target_dict[n_key] = target_dict.pop(key)
        #print(key,n_key)

    print(target_dict)
'''
#print(patent_topicDist)
'''
columns = np.array(['pat_publn_id', 'publn_auth', 'publn_nr', 'publn_date', 'publn_claims', 'publn_title', 
                    'publn_abstract', 'nb_IPC', 'abstract_clean', 'topic_list', 
                    '1st_topic_id', '1st_topic_name', '1st_topic_cover', ...])
'''
'''
attrs = {0: {"attr1": 20, "attr2": "nothing"}, 1: {"attr2": 3}}
nx.set_node_attributes(G, attrs)
G.nodes[0]["attr1"]
20
G.nodes[0]["attr2"]
'nothing'
G.nodes[1]["attr2"]
3
G.nodes[2]
{}'''


plain = nx.Graph()
bipart = nx.Graph()
topic_similar = nx.Graph()
topic_net = nx.Graph()

plain.add_nodes_from(patent_topicDist.T[0])
nx.set_node_attributes(plain, outer_dic)

print(plain.nodes[487838990])
#print(plain.nodes[487839054])

#print(plain.nodes[487838990]["publn_claims"])
#print(plain.nodes[487839054]["publn_claims"])


### edge investigation ###

#plain.add_nodes_from(patent_topicDist.T[0], test = 'lala')

set_nodeID = set(patent_topicDist.T[0])
set_edgeID1 = set(cited_pat_publn_id.T[0])
set_edgeID2 = set(cited_pat_publn_id.T[1])

print("\n", 'Every edge source has a corresponding node in the patent data: ', set_edgeID1.issubset(set_nodeID))                 # So, every source of our edge has a corresponding node
print('Every edge target has a corresponding node in the patent data: ', set_edgeID1.issubset(set_edgeID2))          # So, not every target of our edge has a corresponding nodes

print('# of nodes in the patent data: ', len(patent_topicDist))
print('# of edges in the parent data: ', len(cited_pat_publn_id))
print('# of unique edge sources: ', len(np.unique(cited_pat_publn_id.T[0])))
print('# of unique edge targets: ', len(np.unique(cited_pat_publn_id.T[1])))

print('# of unique edge targets with no correpsonding node in the patent data: ',len(set_edgeID2.difference(set_nodeID)))          # what is in x that is not in y x.difference(y) / number of nodes that are cited in general (zeros not considered)
print('# of unique edge targets with correpsonding node in the patent data: ', len(set_edgeID2.intersection(set_nodeID)))  # what is in x that is also in y / number of nodes that are cited and present in the network

print('# of edges with source and target in the patent data: ', len(parent)-len(set_edgeID2.difference(set_nodeID))) # number of edges that are present in the network
# 18548 - 12950

#print(cited_pat_publn_id)

### filtering the edges out that are not contained in the patent data ###


intersec_edgeTarget = set_edgeID2.intersection(set_nodeID)

print(intersec_edgeTarget)

#I = cited_pat_publn_id[cited_pat_publn_id[:,1] in patent_topicDist.T[0] ]

validEdges = []
for i in range(len(cited_pat_publn_id)):
    if cited_pat_publn_id[i,1] in patent_topicDist.T[0]:
        validEdges.append(cited_pat_publn_id[i])
        #print(cited_pat_publn_id[i])

print(len(validEdges))
print(cited_pat_publn_id[0])
print(len(cited_pat_publn_id))
print(patent_topicDist.T[0])
'''
#helper = 0

ind_dict = dict((i,k) for i,k in enumerate(cited_pat_publn_id.T[1]))
#print(ind_dict)
inter = set(ind_dict).intersection(patent_topicDist.T[0])
print(inter)
indices = [ ind_dict[x] for x in inter ]
print(indices)
'''
#plainEdges = cited_pat_publn_id[helper,:]

'''
p = np.array([[1.5, 0], [1.4,1.5], [1.6, 0], [1.7, 1.8]])
print(p, "\n")
nz = (p == 0).sum(1)
print(nz)
#q = p[nz == 0, :]
#q
'''
#print(len(parent))
'''
print(len(plain.nodes))
print(plain.nodes.data)
print(plain.nodes[487838990], "\n \n")
'''
'''
G = nx.path_graph(3)
bb = nx.betweenness_centrality(G)
bb = [0, 1, 0]
print(isinstance(bb, dict))

nx.set_node_attributes(G, bb, "betweenness")
print(G.nodes[0]["betweenness"])
print(G.nodes[1]["betweenness"])
print(G.nodes[2]["betweenness"])
print(bb)
'''
'''
unique, counts = np.unique(cited_pat_publn_id, return_counts=True)
print(np.asarray((unique, counts)).T)
unique, counts = np.unique(counts, return_counts=True)
print(np.asarray((unique, counts)).T)                                   # 1910 cited id 0. They will probably be excluded
'''

