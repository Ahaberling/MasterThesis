import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle as pk
import os
import itertools
import sys

from scipy.signal import convolve2d
from scipy.signal import convolve



import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

#--- Initialization --#



os.chdir('D:/Universitaet Mannheim/MMDS 7. Semester/Master Thesis/Outline/Data/Cleaning Robots')

#directory = 'D:/Universitaet Mannheim/MMDS 7. Semester/Master Thesis/Outline/Data/Cleaning Robots/'

#patent_lda_ipc = pd.read_csv( directory + 'patent_lda_ipc.csv', quotechar='"', skipinitialspace=True)
patent_lda_ipc = pd.read_csv('patent_lda_ipc.csv', quotechar='"', skipinitialspace=True)
topics = pd.read_csv(r'D:\Universitaet Mannheim\MMDS 7. Semester\Master Thesis\Outline\Data\Cleaning Robots\patent_topics.csv', quotechar='"', skipinitialspace=True)
og_ipc = pd.read_csv(r'D:\Universitaet Mannheim\MMDS 7. Semester\Master Thesis\Outline\Data\Cleaning Robots\cleaning_robot_EP_patents_IPC.csv', quotechar='"', skipinitialspace=True)
#parent = pd.read_csv(r'D:\Universitaet Mannheim\MMDS 7. Semester\Master Thesis\Outline\Data\Cleaning Robots\cleaning_robot_EP_backward_citations.csv', quotechar='"', skipinitialspace=True)

patent_lda_ipc = patent_lda_ipc.to_numpy()
topics = topics.to_numpy()
og_ipc = og_ipc.to_numpy()


'''
print(patent_lda_ipc)
print(np.shape(patent_lda_ipc))

print(og_ipc)
print(np.shape(og_ipc))

print(len(np.unique(og_ipc[:,0])))
print(len(np.unique(og_ipc[:,1]))) # 970 unique ipcs (and topics)

print(patent_lda_ipc[0,:])
'''
if 1 == 2:
    with open('window90by1', 'rb') as handle:
        window90by1 = pk.load(handle)

    #print(window90by1)
    print(window90by1['window_0'])
    print(np.shape(window90by1['window_0']))

    print(window90by1['window_0'][0])
    print(window90by1['window_0'][0][0])

    # now I want for each window the distribution of the ipc/topics

    window90by1_dist_ipc = {}

    #ipc_position = range(53,91,3)

    ipc_position = np.r_[range(52,91,3)]
    topic_position = np.r_[range(10,52,3)]

    window90by1_ipcs = {}
    window90by1_topics = {}

    window90by1_ipcs_allComb = {}
    window90by1_topics_allComb = {}

    window90by1_ipcs_twoComb = {}
    window90by1_topics_twoComb = {}

    window90by1_ipcs_threeComb = {}
    window90by1_topics_threeComb = {}

    c = 0

    for window in window90by1.values():
        #print('----')
        #print(window.key)
        #print(window)

        ipc_list = []
        topic_list = []

        ipc_allComb_list = []
        topic_allComb_list = []

        ipc_twoComb_list = []
        topic_twoComb_list = []

        ipc_threeComb_list = []
        topic_threeComb_list = []

        for patent in window:
            #print(patent[ipc_position4])
            #print(patent[np.r_[52,55,58]])
            #print(patent[ipc_position5])
            #print(patent[9:15])
            ipc_list.append(patent[ipc_position])
            topic_list.append(patent[topic_position])

            # ipc_allComb_list
            y = [x for x in patent[ipc_position] if x == x]             # nan elimination
            y = np.unique(y)                                            # todo probably/hopefully not neccessary (because hopefully the data is clean enough so that one paper is not classfied with the same ipc more than once)
            ipc_allComb_list.append(tuple(y))

            # topic_allComb_list
            z = [x for x in patent[topic_position] if x == x]           # nan elimination
            z = np.unique(z)                                            # todo probably/hopefully not neccessary (because hopefully the data is clean enough so that one paper is not classfied with the same topic more than once)
            topic_allComb_list.append(tuple(z))

            # ipc_twoComb_list
            ipc_twoComb_list.append(list(itertools.combinations(y, r=2)))

            # topic_twoComb_list
            topic_twoComb_list.append(list(itertools.combinations(z, r=2)))

            # ipc_threeComb_list
            ipc_threeComb_list.append(list(itertools.combinations(y, r=3)))

            # topic_threeComb_list
            topic_threeComb_list.append(list(itertools.combinations(z, r=3)))


        #print(ipc_comb_list)

        # all ipcs that occured in the window in general
        ipc_list = np.concatenate(ipc_list).ravel().tolist()
        ipc_list = [x for x in ipc_list if x == x]
        ipc_list = np.unique(ipc_list)
        window90by1_ipcs['window_{0}'.format(c)] = ipc_list

        # all topics that occured in the window in general
        topic_list = np.concatenate(topic_list).ravel().tolist()
        topic_list = [x for x in topic_list if x == x]
        topic_list = np.unique(topic_list)
        window90by1_topics['window_{0}'.format(c)] = topic_list

        # all ipcs combinations as tuple that occured in the window
        # meaning one patent -> one tuple
        #ipc_comb_list = np.unique(ipc_comb_list)                       # todo Error message, but I probably also dont want to do that in general
        window90by1_ipcs_allComb['window_{0}'.format(c)] = ipc_allComb_list

        # all topic combinations as tuple that occured in the window
        # meaning one patent -> one tuple
        #topic_comb_list = np.unique(topic_comb_list)                   # todo Error message, but I probably also dont want to do that in general
        window90by1_topics_allComb['window_{0}'.format(c)] = topic_allComb_list

        # all ipc inside a patent as pairs in the window
        # meaning one patent -> (possibly) multiple tuples of size two
        #print(window[1])
        #print(ipc_twoComb_list)                                             #todo somehow we got empty lists in here? is it for patents with only one ipc? -> no combination possible?
        ipc_twoComb_list = [item for sublist in ipc_twoComb_list for item in sublist]
        #print(ipc_twoComb_list)
        #ipc_twoComb_list = np.array(ipc_twoComb_list).ravel()
        #print(ipc_twoComb_list)
        window90by1_ipcs_twoComb['window_{0}'.format(c)] = ipc_twoComb_list


        # all topic inside a patent as pairs in the window
        # meaning one patent -> (possibly) multiple tuples of size two
        #print(topic_twoComb_list)
        topic_twoComb_list = [item for sublist in topic_twoComb_list for item in sublist]
        #print(topic_twoComb_list)
        window90by1_topics_twoComb['window_{0}'.format(c)] = topic_twoComb_list

        # all ipc inside a patent as triples in the window
        # meaning one patent -> (possibly) multiple tuples of size three
        #print(ipc_threeComb_list)
        ipc_threeComb_list = [item for sublist in ipc_threeComb_list for item in sublist]
        #print(ipc_threeComb_list)
        window90by1_ipcs_threeComb['window_{0}'.format(c)] = ipc_threeComb_list

        # all topic inside a patent as triples in the window
        # meaning one patent -> (possibly) multiple tuples of size three
        topic_threeComb_list = [item for sublist in topic_threeComb_list for item in sublist]
        window90by1_topics_threeComb['window_{0}'.format(c)] = topic_threeComb_list

        c = c + 1

    print(window90by1_ipcs_twoComb)

    filename = 'window90by1_ipcs_twoComb'
    outfile = open(filename, 'wb')
    pk.dump(window90by1_ipcs_twoComb, outfile)
    outfile.close()

if 1 == 2:
    with open('window90by1_ipcs_twoComb', 'rb') as handle:
        window90by1_ipcs_twoComb = pk.load(handle)

    #print(window90by1_ipcs_twoComb.keys())


    tuple_list = []
    for i in window90by1_ipcs_twoComb.values():

        tuple_list.append(i)

    #print(tuple_list)
    tuple_list = [item for sublist in tuple_list for item in sublist]
    #print(tuple_list)
    print('number of all tuples before taking only the unique ones', len(tuple_list))  # 1047572
    tuple_list, tuple_list_counts = np.unique(tuple_list, return_counts=True, axis=0)
    #print(tuple_list)
    print(len(tuple_list))
    #print(tuple_list_counts)        # where does the 90 and the "weird" values come from? explaination: if a combination occures in the whole timeframe only once (in one patent) then it is captures 90 times. The reason for this is the size of the sliding window of 90 and the sliding by one day. One patent will thereby be capured in 90 sliding windows (excaption: the patents in the first and last 90 days of the overall timeframe, they are capture in less then 90 sliding windows)
    #print(len(tuple_list_counts))

    window_list = window90by1_ipcs_twoComb.keys()
    #print(window_list)
    #print(len(window_list))

    '''
    np.random.seed(19680801)
    #Z = np.random.rand(len(window_list)+1, len(tuple_list)+1)  # y,x
    Z = np.random.rand(99+1, 9+1)
    
    print(Z)
    print(np.shape(Z))
    
    #x = np.arange(-0.5, len(tuple_list)+1, 1)  # len = 5445
    x = np.arange(-0.5, 9+1, 1)  # len = 10
    #y = np.arange(-0.5, len(window_list)+1, 1)  # len = 5937
    y = np.arange(-0.5, 99+1, 1)  # len = 100
    
    fig, ax = plt.subplots()
    ax.pcolormesh(x, y, Z)                  # this takes a lot of time if 5445 x 5937
    plt.show()
    
    # Alternativ: only visualize subsample (but use whole data for analysis)
    # Ok, I definitely have to visulalize only a subset e.g. 3x100 or 5x100 or 10x100
    '''
    #--- create occurenc pattern for ipc tuples ---#

    pattern = np.zeros((len(window_list), len(tuple_list)))
    print(np.shape(pattern))

    '''
    pattern = np.zeros((100, 3))
    print(np.shape(pattern))
    print(pattern)
    
    '''

    print(tuple_list)
    print(window_list)
    print(pattern)

    import tqdm

    print('--------------------------')
    print(sum(sum(pattern)))

    pbar = tqdm.tqdm(total=len(window_list))

    c_i = 0
    for i in window_list:
        c_j = 0

        for j in tuple_list:

            if tuple(j) in window90by1_ipcs_twoComb[i]:
                #pattern[c_i,c_j] = 1                                           # results in sum(sum(array)) = 869062.0
                pattern[c_i,c_j] = window90by1_ipcs_twoComb[i].count(tuple(j))

            c_j = c_j +1

        c_i = c_i +1
        pbar.update(1)

    pbar.close()

    print(sum(sum(pattern)))

    filename = 'window90by1_ipcs_twoComb_pattern'
    outfile = open(filename, 'wb')
    pk.dump(pattern, outfile)
    outfile.close()

    '''
    test_arr = np.zeros((10, len(tuple_list)))
    
    print('--------------------------')
    print(sum(sum(test_arr)))
    
    
    c_i = 0
    for i in window_list:
        c_j = 0
    
        for j in tuple_list:
    
            #print(tuple(j))
            #print(window90by1_ipcs_twoComb[i])
            if tuple(j) in window90by1_ipcs_twoComb[i]:
                #test_arr[c_i, c_j] = 1
                test_arr[c_i,c_j] = window90by1_ipcs_twoComb[i].count(tuple(j))
    
            c_j = c_j + 1
    
            #if c_j == 11:
                #break
    
        c_i = c_i + 1
    
        if c_i == 10:
            break
    
    print(sum(sum(test_arr)))
    
    np.set_printoptions(threshold=sys.maxsize)
    print(test_arr)
    '''

with open('window90by1_ipcs_twoComb_pattern', 'rb') as handle:
    pattern = pk.load(handle)

print(pattern)
print(np.amax(pattern))                                 # 15

print(pattern.size)
#print(pattern.T)

#todo transform the pattern array into an array of the same shape with 0 (combination in window does not meet window-specific threshold) and 1 (combination x in window matches the threshold. E.g. 10% of all combinations were combination x)
#todo when done on daily sliding approach, then grand some leeway so that e.g. 00001110111111110000...0001101111111000000 is treated as 2 cycle and not 4

# let's try with 5%:

window_sum = pattern.sum(axis=1)

#print(np.shape(window_sum))             # (5937,)
#print(window_sum)                       # [103. 100. 100. ... 392. 392. 392.]

'''
an_array = np.array([[1,2,3],[4,5,6]])
print(an_array)
sum_of_rows = an_array.sum(axis=1)
print(sum_of_rows)
normalized_array = an_array / sum_of_rows[:, np.newaxis]
print(normalized_array)
'''
print(np.shape(pattern))

#pattern_norm = pattern / window_sum[:, np.newaxis]
pattern_norm = pattern / window_sum[:, np.newaxis]

print(pattern_norm)
print(np.shape(pattern_norm))


window_sum_test = pattern_norm.sum(axis=1)

print(window_sum_test)
print(max(window_sum_test))

#pattern_wThreshold

pattern_wThreshold = np.where(pattern_norm < 0.01, 0, 1)

print(pattern_wThreshold)
print(np.shape(pattern_wThreshold))
print(np.amax(pattern_wThreshold))
print(sum(sum(pattern_wThreshold)))

print(np.where(pattern_wThreshold==1))                  # the indices of elements of value 1 -> of recombination candidates

#--- introduce leeway ---#


def search_sequence_numpy(arr,seq):
    """ Find sequence in an array using NumPy only.

    Parameters
    ----------
    arr    : input 1D array
    seq    : input 1D array

    Output
    ------
    Output : 1D Array of indices in the input array that satisfy the
    matching of input sequence in the input array.
    In case of no match, an empty list is returned.
    """

    # Store sizes of input array and sequence
    Na, Nseq = arr.size, seq.size

    # Range of sequence
    r_seq = np.arange(Nseq)

    # Create a 2D array of sliding indices across the entire length of input array.
    # Match up with the input sequence & get the matching starting indices.
    M = (arr[np.arange(Na-Nseq+1)[:,None] + r_seq] == seq).all(1)
    '''
    print(M)
    print(len(M))
    print('Na', Na)
    print('Nseq', Nseq)
    print('Na-Nseq+1', Na-Nseq+1)
    print('np.arange(Na-Nseq+1)', np.arange(Na-Nseq+1))
    print('shape', np.shape(np.arange(Na-Nseq+1)))
    print('np.arange(Na-Nseq+1)[:,None]', np.arange(Na-Nseq+1)[:,None])
    print('shape', np.shape(np.arange(Na-Nseq+1)[:,None]))
    print('arr[np.arange(Na-Nseq+1)[:,None]]', arr[np.arange(Na-Nseq+1)[:,None]])
    print('r_seq', r_seq)
    print('np.arange(Na-Nseq+1)[:,None] + r_seq', np.arange(Na-Nseq+1)[:,None] + r_seq)
    print('arr[np.arange(Na-Nseq+1)[:,None] + r_seq]', arr[np.arange(Na-Nseq+1)[:,None] + r_seq])
    print(np.shape(arr[np.arange(Na-Nseq+1)[:,None] + r_seq]))
    print('arr[np.arange(Na-Nseq+1)[:,None] + r_seq] == seq', arr[np.arange(Na-Nseq+1)[:,None] + r_seq] == seq)
    print('(arr[np.arange(Na-Nseq+1)[:,None] + r_seq] == seq).all(1)', (arr[np.arange(Na-Nseq+1)[:,None] + r_seq] == seq).all(1))
    print('shape', np.shape((arr[np.arange(Na-Nseq+1)[:,None] + r_seq] == seq).all(1)))
    '''
    # Get the range of those indices as final output
    if M.any() >0:
        return np.where(np.convolve(M,np.ones((Nseq),dtype=int))>0)[0]
    else:
        return []         # No match found


def replace_sequence_numpy(arr,seq, rep_seq):

    Na, Nseq = arr.size, seq.size
    r_seq = np.arange(Nseq)
    M = (arr[np.arange(Na-Nseq+1)[:,None] + r_seq] == seq).all(1)


    return np.where(np.convolve(M,np.ones((Nseq),dtype=int))>0,1,0)


'''
arr = np.array([2, 0, 0, 0, 0, 1, 0, 1, 0, 0])

seq = np.array([0,0])

print(search_sequence_numpy(arr,seq))
'''

print('-----------------')

# lets try to replace sequences of 101 within a tuple to 111

#seq = np.array([1,0,1])
seq = np.array([1,0,0,0,0,0,0,0,1])
rep_seq = np.array([1,1,1])

c = 0
for i in pattern_wThreshold.T:

    arr = i
    print(c, search_sequence_numpy(arr, seq))                       # 2747, 2847, 2860, 2936, 3060, 3138
    #print(c, replace_sequence_numpy(arr, seq, rep_seq))
    #pattern_wThreshold.T[c,:] = replace_sequence_numpy(arr, seq, rep_seq)


    c = c + 1
    #break

#print(pattern_wThreshold)




np.set_printoptions(threshold=sys.maxsize)

#print(pattern_wThreshold.T[2746])
#print('before', pattern_wThreshold.T[2747])
#print(sum(pattern_wThreshold.T[2747]))
#print('before', pattern_wThreshold.T[2709])
#print(sum(pattern_wThreshold.T[2709]))
#print(pattern_wThreshold.T[2748])


'''
c = 0
for i in pattern_wThreshold.T:

    arr = i
    #print(c, search_sequence_numpy(arr, seq))                       # 2747, 2847, 2860, 2936, 3060, 3138
    #print(c, replace_sequence_numpy(arr, seq, rep_seq))
    #pattern_wThreshold.T[c,:] = replace_sequence_numpy(arr, seq, rep_seq)

    k = seq # kernel for convolution
    i[(convolve(i, k, 'same') == 2) & (i == 0)] = 1
    print(i)
    print('convolve(i, k, \'same\')', convolve(i, k, 'same'))
    print('convolve(i, k, \'same\') == 2', convolve(i, k, 'same') == 2)
    print('i == 0', i == 0)
    print('convolve(i, k, \'same\') == 2 & (i == 0)', convolve(i, k, 'same') == 2 & (i == 0))
    
    pattern_wThreshold.T[c,:] = i

    c = c + 1
    #break
'''
print('after', pattern_wThreshold.T[2747])
#print(sum(pattern_wThreshold.T[2747]))


#todo problem 1: imputing sequences only works for 101 case, not for 100001, and so on
#todo problem 2: with only 0 and 1 a diffusion cycle is identified if the threshold is met with one set of patents, that does not change anymore for 90 days. E.g. tuple occures in x patents. x patens were all published on y (no diffusion prossible, because to little time inbetween) nevertheless the patents x might meet the thresshold for t until t+89


c = 0
for i in pattern_wThreshold.T:

    arr = i
    #print(c, search_sequence_numpy(arr, seq))                       # 2747, 2847, 2860, 2936, 3060, 3138
    #print(c, replace_sequence_numpy(arr, seq, rep_seq))
    #pattern_wThreshold.T[c,:] = replace_sequence_numpy(arr, seq, rep_seq)


    # og 0 1 1 1 1 1 1 1 1 0 0 0 1 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1

    if c == 2747:
        #k = np.array([1,0,1]) # kernel for convolution
        k = np.array([1,0,0,0,0,0,0,0,1]) # kernel for convolution
        i[(convolve(i, k, 'same') == 2) & (i == 0)] = 1

        #print('in loop 100000001', pattern_wThreshold.T[2747])
        print('convolve(i, k, same)', convolve(i, k, 'same'))
        print('convolve(i, k, same) == 2', convolve(i, k, 'same') == 2)
        print('i == 0', i == 0)
        print('(convolve(i, k, same) == 2) & (i == 0)', (convolve(i, k, 'same') == 2) & (i == 0))

        #print(i)
        #print(i == 0)

    pattern_wThreshold.T[c,:] = i

    c = c + 1
    #break

print('after after', pattern_wThreshold.T[2747])
#print(sum(pattern_wThreshold.T[2747]))


#I do find sequences like 100001 as well

#todo find recombinations in pattern_wThreshold, whenever a 1 first occures (first time in t periodes)

#todo find sequences in pattern_wThreshold to identify diffusion cycles


# I need all pair combinations that occur in the whole timeframe
# construct heatmap with  x = combination, y = window, z = increase of occurence
# for this find list with all windows
# find list with all unique pairs
# interate through dictionary and fill he dict

#todo idea: right now window90by1_ipcs_twoComb contains tuples like ('C12M   1', 'C12M   3'). If this is to fine grained (no real inovation/ recombination) then go more course graind (or fine grained)

# recombination:
# is when a combination (2+) of ipc's/topics is cited together for the first time in X
# or if the number of patents combining them cross a threshold for the first time in X

# diffusion:
# is active as long as the number of a topic/ipc or the number of a combination of them is above a certain threshold

