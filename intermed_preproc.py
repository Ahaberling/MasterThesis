import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

patent_topicDist = pd.read_csv(r'D:\Universitaet Mannheim\MMDS 7. Semester\Master Thesis\Outline\Data\Cleaning Robots\cleaning_robot_EP_patents_07_04_topicDist.csv', quotechar='"', skipinitialspace=True)
topics = pd.read_csv(r'D:\Universitaet Mannheim\MMDS 7. Semester\Master Thesis\Outline\Data\Cleaning Robots\cleaning_robot_EP_patents_07_04_topics.csv', quotechar='"', skipinitialspace=True)
parent = pd.read_csv(r'D:\Universitaet Mannheim\MMDS 7. Semester\Master Thesis\Outline\Data\Cleaning Robots\cleaning_robot_EP_backward_citations.csv', quotechar='"', skipinitialspace=True)

patent_topicDist = patent_topicDist.to_numpy()
topics = topics.to_numpy()
parent = parent.to_numpy()

#print(patent_topicDist[:,9])
'''
['[(2, 0.018323675), (12, 0.112178914), (22, 0.034536283), (27, 0.018756341), (61, 0.04547727), (72, 0.059345122), (75, 0.047571883), (76, 0.070082165), (93, 0.04768134), (137, 0.018582573), (142, 0.06823648), (148, 0.023175726), (178, 0.037902977), (210, 0.035857208), (237, 0.07511106), (238, 0.08713831), (252, 0.05006604), (265, 0.019298442), (272, 0.09604465), (289, 0.017966142)]'
 '[(1, 0.014831993), (3, 0.051817194), (61, 0.11261249), (70, 0.07581455), (84, 0.037941262), (90, 0.018262723), (103, 0.08730816), (132, 0.20776749), (146, 0.018092591), (148, 0.018124036), (156, 0.016882196), (162, 0.015265168), (178, 0.034912385), (214, 0.017497344), (221, 0.017132074), (230, 0.017038222), (252, 0.037554085), (272, 0.117576875), (287, 0.06795255)]'
 '[(13, 0.03819017), (20, 0.04107001), (31, 0.03373102), (38, 0.019701486), (53, 0.03171818), (89, 0.09289847), (106, 0.049362075), (125, 0.017428901), (144, 0.03341596), (148, 0.05367918), (162, 0.017361969), (172, 0.01938423), (178, 0.01602213), (206, 0.017749708), (211, 0.017054759), (260, 0.2248508), (270, 0.018985232), (272, 0.11895294), (281, 0.10370684), (284, 0.018641975)]'
 ...
 '[(12, 0.020125223), (16, 0.012459388), (18, 0.06322088), (65, 0.1493339), (135, 0.02781), (141, 0.028436817), (193, 0.011546338), (216, 0.45687032), (217, 0.0140493475), (238, 0.020588199), (239, 0.03076039), (255, 0.054471254), (260, 0.011956748), (262, 0.010298248), (265, 0.05127724)]'
 '[(14, 0.037997376), (21, 0.12599283), (44, 0.020984292), (54, 0.038075365), (62, 0.04321399), (68, 0.046101157), (91, 0.018381381), (112, 0.017189967), (115, 0.06943478), (144, 0.017236684), (195, 0.03575977), (197, 0.24222249), (202, 0.041957), (244, 0.10171359), (267, 0.029422948), (278, 0.050860085), (297, 0.040751103)]'
 '[(18, 0.07167888), (25, 0.035752434), (28, 0.02642006), (36, 0.08178228), (61, 0.04808582), (69, 0.06541923), (73, 0.18013954), (87, 0.024609072), (181, 0.023846375), (195, 0.04829301), (203, 0.022806134), (210, 0.02337636), (252, 0.050114967), (267, 0.02520522), (282, 0.2503774)]']
'''

#print(patent_topicDist[0,9])
'''
[(2, 0.018323675), (12, 0.112178914), (22, 0.034536283), (27, 0.018756341), (61, 0.04547727), (72, 0.059345122), (75, 0.047571883), (76, 0.070082165), (93, 0.04768134), (137, 0.018582573), (142, 0.06823648), (148, 0.023175726), (178, 0.037902977), (210, 0.035857208), (237, 0.07511106), (238, 0.08713831), (252, 0.05006604), (265, 0.019298442), (272, 0.09604465), (289, 0.017966142)]
'''

#topic_reg_list = np.array([[]])
topic_reg_list = []

for i in range(len(patent_topicDist)):
#for i in range(0,3):
    #print(i)
    #print(patent_topicDist[i,9])
    topic_reg = re.findall("(\d*\.*?\d+)", patent_topicDist[i,9])
    #topic_reg = re.findall("\(.*\)", patent_topicDist[0,9])

    #print("topic_reg: ", topic_reg)             # topic_reg:  ['2', '0.11020917', '77', '0.15190029', '132', '0.16767512', '174', '0.14025687']
    #print("topic_reg: ", topic_reg[0])
    topic_reg_list.append(topic_reg)

#print(topic_reg_list)

list_len = [len(i) for i in topic_reg_list]
longest = max(list_len)
print('longest: ', longest)
longest = int(longest + (longest/2))
print('new longest: ', longest)


#helper = patent_topicDist[0:3,:]

patent_topicDist_test = np.empty((np.shape(patent_topicDist)[0],np.shape(patent_topicDist)[1]+longest), dtype= object)
#patent_topicDist_test = patent_topicDist_test[0:3,:]
#patent_topicDist_test[:,:-longest] = patent_topicDist[0:3,:]
patent_topicDist_test[:,:-longest] = patent_topicDist

#print(patent_topicDist[0:3,:].T,"\n")
#print(patent_topicDist_test[0:3,:].T)

'''
patent_clean_topicDist = np.empty((np.shape(patent_clean)[0],np.shape(patent_clean)[1]+1), dtype= object)
patent_clean_topicDist[:,:-1] = patent_topicDist
patent_clean_topicDist.T[9,:] = doc_affili
'''

topic_reg_list_tuple = []

c = 0
for i in topic_reg_list:
    tuple_list = []
    for j in range(0,len(i)-1,2):
        tuple = (i[j], i[j+1])
        tuple_list.append(tuple)
    tuple_list = sorted(tuple_list, key=lambda tup: tup[1], reverse=True)

    #print(tuple_list)

    l = 0
    for k in range(len(tuple_list)):
        #print(k)
        patent_topicDist_test[c,np.shape(patent_topicDist)[1]+l] = tuple_list[k][0]
        l = l + 2

        patent_topicDist_test[c,np.shape(patent_topicDist)[1]+l] = tuple_list[k][1]
        l = l + 1


    c = c + 1

print(np.shape(patent_topicDist_test))
print(patent_topicDist_test)
print(patent_topicDist_test.T)
#print('\n', patent_topicDist_test[:,51].T)

#print(sum(x is not None for x in patent_topicDist_test[:,51])) # check if all created columns are used (they are)

#print(topics)

#print(patent_topicDist_test[:,10])
#print(patent_topicDist_test[:,11])


pd.DataFrame(patent_topicDist_test).to_csv(r'D:\Universitaet Mannheim\MMDS 7. Semester\Master Thesis\Outline\Data\Cleaning Robots\cleaning_robot_EP_patents_07_04_topicNameMissing.csv', index=None)

#random note

'''
for j in len(topic_reg):
    tuple = (topic_reg[j], topic_reg[j+1])
    
    j = j + 2
'''
#print(topic_reg_list)



'''
topic_reg:  ['2', '0.11645764', '163', '0.17596386']
'''







'''
m = re.match("\[\(.*\)\]", patent_topicDist[0,9])
print('m: ', m)
if m:
        print(m.groups())
'''

'''
cited_pat_publn_id = parent[:,2]


unique, counts = np.unique(cited_pat_publn_id, return_counts=True)
print(np.asarray((unique, counts)).T)
unique, counts = np.unique(counts, return_counts=True)
print(np.asarray((unique, counts)).T)                                   # 1910 cited id 0. They will probably be excluded

plain = nx.Graph()
bipart = nx.Graph()
topic_similar = nx.Graph()
topic_net = nx.Graph()
'''