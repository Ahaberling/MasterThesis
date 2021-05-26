import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle as pk
import os

#--- Initialization --#

os.chdir('D:/Universitaet Mannheim/MMDS 7. Semester/Master Thesis/Outline/Data/Cleaning Robots')

#directory = 'D:/Universitaet Mannheim/MMDS 7. Semester/Master Thesis/Outline/Data/Cleaning Robots/'

#patent_lda_ipc = pd.read_csv( directory + 'patent_lda_ipc.csv', quotechar='"', skipinitialspace=True)
patent_lda_ipc = pd.read_csv('patent_lda_ipc.csv', quotechar='"', skipinitialspace=True)
#topics = pd.read_csv(r'D:\Universitaet Mannheim\MMDS 7. Semester\Master Thesis\Outline\Data\Cleaning Robots\cleaning_robot_EP_patents_07_04_topics.csv', quotechar='"', skipinitialspace=True)
#parent = pd.read_csv(r'D:\Universitaet Mannheim\MMDS 7. Semester\Master Thesis\Outline\Data\Cleaning Robots\cleaning_robot_EP_backward_citations.csv', quotechar='"', skipinitialspace=True)

patent_lda_ipc = patent_lda_ipc.to_numpy()
#parent = parent.to_numpy()

window90by1_bool = True
window60by1_bool = True
window90by7_bool = True
window60by7_bool = True


#--- Overview ---#

patent_time = patent_lda_ipc[:,3].astype('datetime64')

#print(min(patent_time))        # ealiest day with publication 2001-08-01
#print(max(patent_time))        # latest  day with publication 2018-01-31

max_timeSpan = int((max(patent_time) - min(patent_time)) / np.timedelta64(1, 'D'))
#print(max_timeSpan)            # 6027 day between earliest and latest publication

val, count = np.unique(patent_time, return_counts=True)
#print(len(val))                # 817 days publications were made -> on average every 7.37698898409 days a patent was published

#plt.bar(val, count)
#plt.show()



#--- slinding window approaches ---#

# 60 days window vs 90 days window & sliding by 1 day vs sliding by 7 days

### 90 days sliding by 1 day ###

patent_time_unique = np.unique(patent_time)                                                                                 #  817
patent_time_unique_filled = np.arange(np.min(patent_time_unique), np.max(patent_time_unique))                               # 6027
patent_time_unique_filled_90 = patent_time_unique_filled[patent_time_unique_filled <= max(patent_time_unique_filled)-90]    # 5937

if window90by1_bool == True:

    window90by1 = {}
    len_window = []

    c = 0
    for i in patent_time_unique_filled_90:
        lower_limit = i
        upper_limit = i + 90

        patent_window = patent_lda_ipc[(patent_lda_ipc[:, 3].astype('datetime64') < upper_limit) & (patent_lda_ipc[:, 3].astype('datetime64') >= lower_limit)]
        len_window.append(len(patent_window))

        window90by1['window_{0}'.format(c)] = patent_window

        if c % 100 == 0:
            print(c, " / ", len(patent_time_unique_filled_90))
        #if i >= 100:
            #break

        c = c + 1

    #print(len(window90by1))                     # 5937 windows
    #print(sum(len_window)/len(len_window))      # on average 56.253326595923866 patents per window


    filename = 'window90by1'
    outfile = open(filename,'wb')
    pk.dump(window90by1, outfile)
    outfile.close()

    '''
    with open('window90by1aaa.pickle', 'wb') as handle:
        pk.dump(window90by1, handle, protocol=pk.HIGHEST_PROTOCOL)
    '''

### 60 days sliding by 1 day ###

patent_time_unique_filled_60 = patent_time_unique_filled[patent_time_unique_filled <= max(patent_time_unique_filled)-60]

if window60by1_bool == True:

    window60by1 = {}
    len_window = []

    c = 0
    for i in patent_time_unique_filled_60:
        lower_limit = i
        upper_limit = i + 60

        patent_window = patent_lda_ipc[(patent_lda_ipc[:, 3].astype('datetime64') < upper_limit) & (patent_lda_ipc[:, 3].astype('datetime64') >= lower_limit)]
        len_window.append(len(patent_window))

        window60by1['window_{0}'.format(c)] = patent_window

        if c % 100 == 0:
            print(c, " / ", len(patent_time_unique_filled_60))
        #if i >= 100:
            #break
        c = c+1

    #print(len(window60by1))                        # 5967 windows
    #print(sum(len_window)/len(len_window))         # on average 37.50477626948215 patents per window

    filename = 'window60by1'
    outfile = open(filename,'wb')
    pk.dump(window60by1,outfile)
    outfile.close()


### 90 days sliding by 7 day ###


if window90by7_bool == True:

    window90by7 = {}
    len_window = []

    c = 0
    for i in patent_time_unique_filled_90:

        if c % 7 == 0:
            lower_limit = i
            upper_limit = i + 90

            patent_window = patent_lda_ipc[(patent_lda_ipc[:, 3].astype('datetime64') < upper_limit) & (patent_lda_ipc[:, 3].astype('datetime64') >= lower_limit)]
            len_window.append(len(patent_window))

            window90by7['window_{0}'.format(c)] = patent_window

        if c % 100 == 0:
            print(c, " / ", len(patent_time_unique_filled_90))
        #if i >= 100:
            #break
        c = c+1


    #print(len(window90by7))                         # 849 windows
    #print(sum(len_window)/len(len_window))          # on average 56.849234393404004 patents per window

    filename = 'window90by7'
    outfile = open(filename,'wb')
    pk.dump(window90by7,outfile)
    outfile.close()



### 60 days sliding by 7 day ###


if window60by7_bool == True:

    window60by7 = {}
    len_window = []

    c = 0
    for i in patent_time_unique_filled_60:

        if c % 7 == 0:
            lower_limit = i
            upper_limit = i + 60

            patent_window = patent_lda_ipc[(patent_lda_ipc[:, 3].astype('datetime64') < upper_limit) & (patent_lda_ipc[:, 3].astype('datetime64') >= lower_limit)]
            len_window.append(len(patent_window))

            window60by7['window_{0}'.format(c)] = patent_window

        if c % 100 == 0:
            print(c, " / ", len(patent_time_unique_filled_60))
        #if i >= 100:
            #break
        c = c+1

    #print(len(window60by7))                         # 853 windows
    #print(sum(len_window)/len(len_window))          # on average 39.35873388042204 patents per window

    filename = 'window60by7'
    outfile = open(filename,'wb')
    pk.dump(window60by7,outfile)
    outfile.close()
