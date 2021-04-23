import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import sys
import pickle as pk

pd.set_option('display.max_columns', None)

patent_topicDist = pd.read_csv(r'D:\Universitaet Mannheim\MMDS 7. Semester\Master Thesis\Outline\Data\Cleaning Robots\cleaning_robot_EP_patents_07_04_topicNameMissing.csv', quotechar='"', skipinitialspace=True)
#topics = pd.read_csv(r'D:\Universitaet Mannheim\MMDS 7. Semester\Master Thesis\Outline\Data\Cleaning Robots\cleaning_robot_EP_patents_07_04_topics.csv', quotechar='"', skipinitialspace=True)
parent = pd.read_csv(r'D:\Universitaet Mannheim\MMDS 7. Semester\Master Thesis\Outline\Data\Cleaning Robots\cleaning_robot_EP_backward_citations.csv', quotechar='"', skipinitialspace=True)
#print(parent)

patent_topicDist = patent_topicDist.to_numpy()
parent = parent.to_numpy()

patent_time = patent_topicDist[:,3].astype('datetime64')

print(patent_topicDist[:,3])
print(min(patent_time))
print(max(patent_time))

max_timeSpan = int((max(patent_time) - min(patent_time)) / np.timedelta64(1, 'D'))
print(max_timeSpan)

val, count = np.unique(patent_time, return_counts=True)
# 6027 days days between 2001-08-01 (min) and 2018-01-31 (max)
# on 818 days publications were made -> on average every 7.36797066015 days a patent was published


#print(val)
#print(count)
print(len(val))
print(len(count))

plt.bar(val, count)
#plt.show()

### slinding window approaches ###

# 60 days window vs 90 days window & sliding by 1 day vs sliding by 7 days

#-- 90 days sliding by 1 day

'''
print(min(patent_time))
print(min(patent_time)+90)

patent_window1 = patent_topicDist[patent_topicDist[:,3].astype('datetime64')+1 <= min(patent_time)+90]

print(np.sort(patent_window1[:,3]))
print(len(patent_window1))
'''


'''
for i in range(max_timeSpan-90):
    lower_limit = i
    upper_limit = i+90

    #patent_window = patent_topicDist[patent_topicDist[:, 3].astype('datetime64') < upper_limit and patent_topicDist[:, 3].astype('datetime64') >= lower_limit]
    patent_window = patent_topicDist[patent_topicDist[:, 3].astype('datetime64') < upper_limit]
    #window90by7['window_{0}'.format(i)] =
    print(i)
    print(len(patent_window))
'''


patent_time_unique = np.unique(patent_time)                                                                                 #  818
patent_time_unique_filled = np.arange(np.min(patent_time_unique), np.max(patent_time_unique))                               # 6027
patent_time_unique_filled_90 = patent_time_unique_filled[patent_time_unique_filled <= max(patent_time_unique_filled)-90]    # 5936

'''
print(len(patent_time_unique))
print(len(patent_time_unique_filled))
print(len(patent_time_unique_filled_90))

'''

window90by1 = {}
len_window = []

c = 0
for i in patent_time_unique_filled_90:
    lower_limit = i
    upper_limit = i + 90

    patent_window = patent_topicDist[(patent_topicDist[:, 3].astype('datetime64') < upper_limit) & (patent_topicDist[:, 3].astype('datetime64') >= lower_limit)]
    len_window.append(len(patent_window))

    window90by1['window_{0}'.format(c)] = patent_window
    c = c+1

    if c % 100 == 0:
        print(c, " / ", len(patent_time_unique_filled_90))
    #if i >= 100:
        #break


#print(window90by1['window_1'])
print(len(window90by1['window_0']))

print(len(window90by1))
#print(len(patent_time_unique))
print(sum(len_window)/len(len_window))




#-- 60 days sliding by 1 day


#patent_time_unique = np.unique(patent_time)                                                     # 818
patent_time_unique_filled_60 = patent_time_unique_filled[patent_time_unique_filled <= max(patent_time_unique_filled)-60]    # 809 -> 818-809 = 9 publication dates are ommited

window60by1 = {}
len_window = []

c = 0
for i in patent_time_unique_filled_60:
    lower_limit = i
    upper_limit = i + 60

    patent_window = patent_topicDist[(patent_topicDist[:, 3].astype('datetime64') < upper_limit) & (patent_topicDist[:, 3].astype('datetime64') >= lower_limit)]
    #print(len(patent_window))
    len_window.append(len(patent_window))

    window60by1['window_{0}'.format(c)] = patent_window
    c = c+1

    if c % 100 == 0:
        print(c, " / ", len(patent_time_unique_filled_60))
    #if i >= 100:
        #break


#print(window90by1['window_1'])
print(len(window60by1['window_0']))
print(len(window60by1))

#print(len(patent_time_unique))
print(sum(len_window)/len(len_window))

#-- 90 days sliding by 7 day


patent_time_unique = np.unique(patent_time)                                                                                 #  818
patent_time_unique_filled = np.arange(np.min(patent_time_unique), np.max(patent_time_unique))                               # 6027
patent_time_unique_filled_90 = patent_time_unique_filled[patent_time_unique_filled <= max(patent_time_unique_filled)-90]    # 5936


print(len(patent_time_unique))
print(len(patent_time_unique_filled))
print(len(patent_time_unique_filled_90))



window90by7 = {}
len_window = []

c = 0
for i in patent_time_unique_filled_90:

    if c % 7 == 0:
        lower_limit = i
        upper_limit = i + 90

        patent_window = patent_topicDist[(patent_topicDist[:, 3].astype('datetime64') < upper_limit) & (patent_topicDist[:, 3].astype('datetime64') >= lower_limit)]
        len_window.append(len(patent_window))

        window90by7['window_{0}'.format(c)] = patent_window

    c = c+1

    if c % 100 == 0:
        print(c, " / ", len(patent_time_unique_filled_90))
    #if i >= 100:
        #break


#print(window90by1['window_1'])
print(len(window90by7['window_0']))

print(len(window90by7))
#print(len(patent_time_unique))
print(sum(len_window)/len(len_window))


#-- 60 days sliding by 7 day


#patent_time_unique = np.unique(patent_time)                                                                                 #  818
#patent_time_unique_filled = np.arange(np.min(patent_time_unique), np.max(patent_time_unique))                               # 6027

patent_time_unique_filled_60 = patent_time_unique_filled[patent_time_unique_filled <= max(patent_time_unique_filled)-60]    # 5936


print(len(patent_time_unique))
print(len(patent_time_unique_filled))
print(len(patent_time_unique_filled_90))



window60by7 = {}
len_window = []

c = 0
for i in patent_time_unique_filled_60:

    if c % 7 == 0:
        lower_limit = i
        upper_limit = i + 60

        patent_window = patent_topicDist[(patent_topicDist[:, 3].astype('datetime64') < upper_limit) & (patent_topicDist[:, 3].astype('datetime64') >= lower_limit)]
        len_window.append(len(patent_window))

        window60by7['window_{0}'.format(c)] = patent_window

    c = c+1

    if c % 100 == 0:
        print(c, " / ", len(patent_time_unique_filled_60))
    #if i >= 100:
        #break


#print(window90by1['window_1'])
print(len(window60by7['window_0']))

print(len(window60by7))
#print(len(patent_time_unique))
print(sum(len_window)/len(len_window))




filename = 'window90by1'
outfile = open(filename,'wb')
pk.dump(window90by1,outfile)
outfile.close()

filename = 'window60by1'
outfile = open(filename,'wb')
pk.dump(window60by1,outfile)
outfile.close()

filename = 'window90by7'
outfile = open(filename,'wb')
pk.dump(window90by7,outfile)
outfile.close()

filename = 'window60by7'
outfile = open(filename,'wb')
pk.dump(window60by7,outfile)
outfile.close()
