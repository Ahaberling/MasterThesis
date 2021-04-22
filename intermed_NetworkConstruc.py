import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import sys

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
    #inner_keys.append('{0}_topicID'.format(i))
    inner_keys.append('TopicID_{0}'.format(i))
    inner_keys.append('TopicName_{0}'.format(i))
    inner_keys.append('TopicCover_{0}'.format(i))
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


# 18548 - 12950

#print(cited_pat_publn_id)

### filtering the edges out that are not contained in the patent data ###


#intersec_edgeTarget = set_edgeID2.intersection(set_nodeID)

#print(len(intersec_edgeTarget))

#I = cited_pat_publn_id[cited_pat_publn_id[:,1] in patent_topicDist.T[0] ]

validEdges = []
for i in range(len(cited_pat_publn_id)):
    if cited_pat_publn_id[i,1] in patent_topicDist.T[0]:
        validEdges.append(tuple(cited_pat_publn_id[i]))
        #print(cited_pat_publn_id[i])

print('# of edges with source and target in the patent data: ', len(validEdges)) # number of edges that are present in the network

#print(validEdges)

plain.add_edges_from(validEdges)

print('# of nodes in plain: ', plain.number_of_nodes())
print('# of edges in plain: ', plain.number_of_edges())

### bipartite network ###

# node creation


bipart.add_nodes_from(patent_topicDist.T[0], bipartite=0)
nx.set_node_attributes(bipart, outer_dic)

topics = topics.to_numpy()
print(topics[0])
print(topics[0][0])
print(topics[0][1][2])
print(len(topics))

topicNode_list = ['{0}_topic'.format(i) for i in range(len(topics))]

print(topicNode_list)

for i in bipart.nodes:
    if bipart.nodes[i]['publn_title']:
        del bipart.nodes[i]['publn_title']
    if bipart.nodes[i]['publn_abstract']:
        del bipart.nodes[i]['publn_abstract']
    if bipart.nodes[i]['abstract_clean']:
        del bipart.nodes[i]['abstract_clean']




bipart.add_nodes_from(topicNode_list, bipartite=1)

print(bipart.number_of_nodes())



# edge creation

bipart_edges = np.empty((np.shape(patent_topicDist)[0],7), dtype = object)

bipart_edges = patent_topicDist[:,(0,10,11,12,13,14,15,16,17,18)]


print(bipart_edges)

#bipart_edges.T[1] = ['{0}_topic'.format(int(i)) for i in bipart_edges.T[1]]
bipart_edges.T[1] = ['{0}_topic'.format(int(i)) for i in bipart_edges.T[1]]
#bipart_edges.T[4] = ['{0}_topic'.format(int(i)) for i in bipart_edges.T[4] if i != np.nan]

c = 0
for i in bipart_edges.T[4]:
    if np.isfinite(i):
        #print(i)
        bipart_edges[c,4] = '{0}_topic'.format(int(i))
        #print(bipart_edges.T[4,i])
        #print(bipart_edges[c,4])
    c = c + 1


c = 0
for i in bipart_edges.T[7]:
    if np.isfinite(i):
        #print(i)
        bipart_edges[c,7] = '{0}_topic'.format(int(i))
        #print(bipart_edges.T[4,i])
        #print(bipart_edges[c,4])
    c = c + 1


#print(bipart_edges.T)

topic1_edges = [(i[0], i[1], {'Weight_1': i[3]}) for i in bipart_edges]
topic2_edges = [(i[0], i[4], {'Weight_2': i[6]}) for i in bipart_edges]
topic3_edges = [(i[0], i[7], {'Weight_3': i[9]}) for i in bipart_edges]
'''
print(topic1_edges)
print(topic2_edges)
print(topic3_edges)
'''

'''
bipart.add_edges_from(topic1_edges)
bipart.add_edges_from(topic2_edges)
bipart.add_edges_from(topic3_edges)
'''
#print(bipart.edges)
#print(topic1_edges)
'''
def isNan(string):
    return string != string

for i in topic3_edges:
    #print(i[1])
    if isNan(i[1]) == True:
        print('nan')
'''
#print(topic2_edges)
#print(len(topic3_edges))

topic1_edges_clear = list(filter(lambda x: x[1] == x[1], topic1_edges))
topic2_edges_clear = list(filter(lambda x: x[1] == x[1], topic2_edges))
topic3_edges_clear = list(filter(lambda x: x[1] == x[1], topic3_edges))
#print(len(topic3_edges_clear))
'''
for i in topic3_edges_clear:
    #print(i[1])
    if isNan(i[1]) == True:
        print('nan')
'''
#                 [(1, 2, {'color': 'blue'}), (2, 3, {'weight': 8})]


print(len(bipart.nodes))
print(len(bipart.edges))
bipart.add_edges_from(topic1_edges_clear)
print(len(bipart.nodes))
print(len(bipart.edges))
print(len(topic1_edges) - len(topic1_edges_clear), '\n')

bipart.add_edges_from(topic2_edges_clear)
print(len(bipart.nodes))
print(len(bipart.edges))
print(len(topic2_edges) - len(topic2_edges_clear), '\n')

bipart.add_edges_from(topic3_edges_clear)
print(len(bipart.nodes))
print(len(bipart.edges))
print(len(topic3_edges) - len(topic3_edges_clear), '\n')



print(nx.is_connected(bipart))

bottom_nodes, top_nodes = nx.algorithms.bipartite.sets(bipart)

print(bottom_nodes)
print(len(bottom_nodes))
print(top_nodes)
print(len(top_nodes))


top_nodes_check = {n for n, d in bipart.nodes(data=True) if d["bipartite"] == 1}
bottom_nodes_check = set(bipart) - top_nodes_check

print(bottom_nodes_check)
print(len(bottom_nodes_check))
print(top_nodes_check)
print(len(top_nodes_check))

print(bottom_nodes == bottom_nodes_check)
print(top_nodes == top_nodes_check)

print(round(nx.algorithms.bipartite.density(bipart, bottom_nodes_check), 2))
print(round(nx.algorithms.bipartite.density(bipart, top_nodes_check), 2))

topicSim_g_unweig = nx.algorithms.bipartite.projected_graph(bipart, bottom_nodes_check)
topic_g_unweig = nx.algorithms.bipartite.projected_graph(bipart, top_nodes_check)

#print(topicSim_g.nodes)
print(len(topicSim_g_unweig.nodes))
print(len(topicSim_g_unweig.edges))
print(round(nx.density(topicSim_g_unweig), 2))

#print(topic_g.nodes)
print(len(topic_g_unweig.nodes))
print(len(topic_g_unweig.edges))
print(round(nx.density(topic_g_unweig), 2))

'''
topicSim_g = nx.algorithms.bipartite.generic_weighted_projected_graph(bipart, bottom_nodes_check, weight_function = )
topic_g = nx.algorithms.bipartite.generic_weighted_projected_graph(bipart, top_nodes_check, weight_function = )
# Text Clustering using One-Mode Projection of Document-Word Bipartite Graphs Srivastava et al 2013
'''

#todo cant save graph, idk why. Guess: maybe it is the lda output column
#todo check if it got appended to the graph as well.

#print(plain.nodes.data())



print('\n', len(plain.nodes[487838651]))

for i in plain.nodes:
    del plain.nodes[i]['publn_title']
    del plain.nodes[i]['publn_abstract']
    del plain.nodes[i]['abstract_clean']


#del plain.nodes[487838651]['topic_list']

print(plain.nodes[487838651])
print(len(plain.nodes[487838651]))




#list(G.nodes(data=True))



#print(plain[0])
#print(plain.nodes[0][0])

#plain = nx.path_graph(4)


print(validEdges)
print(len(validEdges))
#print(np.unique(validEdges))
print(len(set(validEdges)))
#print(len(validEdges))

#todo still german patents in there


nx.write_gml(plain, r'D:\Universitaet Mannheim\MMDS 7. Semester\Master Thesis\Outline\Data\Cleaning Robots\plain.gml')
nx.write_gml(bipart, r'D:\Universitaet Mannheim\MMDS 7. Semester\Master Thesis\Outline\Data\Cleaning Robots\bipart.gml')
nx.write_gml(topicSim_g_unweig, r'D:\Universitaet Mannheim\MMDS 7. Semester\Master Thesis\Outline\Data\Cleaning Robots\topicSim_g_unweig.gml')
nx.write_gml(topic_g_unweig, r'D:\Universitaet Mannheim\MMDS 7. Semester\Master Thesis\Outline\Data\Cleaning Robots\topic_g_unweig.gml')
#nx.write_edgelist(plain, r'D:\Universitaet Mannheim\MMDS 7. Semester\Master Thesis\Outline\Data\Cleaning Robots\plain.edgelist')
#nx.write_gml(plain, "plain.gml")
#nx.write_graphml_lxml(bipart, r'D:\Universitaet Mannheim\MMDS 7. Semester\Master Thesis\Outline\Data\Cleaning Robots\bipart.graphml')

#bipart_edges.T[4] = helper1

#bipart_edges.T[7] = ['{0}_topic'.format(int(i)) for i in bipart_edges.T[7]]


#print('done')
#print(helper1)
#print(bipart_edges)
#print(bipart_edges.T[1])
#print(np.unique(bipart_edges.T[4], return_counts = True))
#print(bipart_edges.T[4].type)
#print(patent_topicDist.dtype)
#print(patent_topicDist.T[4].dtype)





'''
print(True in pd.isnull(bipart_edges.T[4]))

nanPos_dict = dict(enumerate(pd.isnull(bipart_edges.T[4])))

#print(nanPos_dict)

nanList = []
for i in nanPos_dict:
    if nanPos_dict[i] == True:
        nanList.append(i)

print(nanList)  # [292, 593, 622, 832, 858, 891, 1066, 1269, 1341, 1350, 1355, 1505, 1783, 2053, 2334, 2648, 3001, 3003, 3031, 3244, 3249, 3665, 3666, 3765, 3836]
'''
#np.set_printoptions(threshold=sys.maxsize)

#print(patent_topicDist[(292, 593, 622, 832, 858),:])



#print(bipart_edges.T[2])

#bipart_edges = bipart_edges[:,(0,)]



#print(np.shape(patent_topicDist)[0])

#print(cited_pat_publn_id[0])
#print(len(cited_pat_publn_id))
#print(patent_topicDist.T[0])
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

