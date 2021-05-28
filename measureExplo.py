### This file relies on the result of v1_slidingWindows.py. It takes one of the proposed sliding window approaches and
### transforms the dictionary into arrays displaying diffusion and recombination patterns. In the resulting arrays one
### row represents a windows, while one column represents a IPC- or topic unit. In this pattern arrays diffusion and
### recombination are measured.
###
### First focusing on IPC's, the columns of the first array represent all IPC-pairs contained in the patents of the
### data set (Not all possible IPC-pairs cross patent). The cells A ij
### represent how often a pair j occured in window i.
###
###               pair_0  pair_1  pair_2  ...
###       window_0     0       1       0  ...
###       window_1     0       1       2  ...
###       window_2     1       2       5  ...
###       ...        ...     ...     ...  ...
###
### This array is then transformed in order to ease the finding of
### recombination and diffusion patterns. In the first transformation the cells are normalized by the row sum. This way
### a cell represents the fraction of how much discourse it covered in the respective window. In a second transformation
### the array is binarized by a threshold X. If a pair is responsible for X% of the discourse, it is represented as 1,
### 0 other wise. In this binary pattern array recombinations are identified and the length of their diffusion measured.
### (Side note: code for recombinations of tripples is provided as well. More then tripples is not considered relevant
### right now)
###
### Additionally a second pattern is build. While the rows are still representing the windows, the columns now represent
### singular IPC's. This way a diffusion measure independet of recombination is provided.
###
### All of the provided patterns and measures relying on them are replicated for topics and topic-pairs, instead of
### IPC's
###
### Code locating arbitrary pattern sequences like 11101101001101 is provided as well
### Code for imputing sequences like 1110111 to 1111111 is provided as well
### Code for imputing sequences like 11100111 to 11111111 is not provided yet

if __name__ == '__main__':

#--- Import Libraries ---#
    print('\n#--- Import Libraries ---#\n')

    import pandas as pd
    import numpy as np

    import matplotlib.pyplot as plt
    import pickle as pk
    import os
    import itertools
    import sys

    import tqdm

    from scipy.signal import convolve2d
    from scipy.signal import convolve

    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm
    from matplotlib.ticker import MaxNLocator

    preproc_bool = True
    plain_pattern_bool = True
    norm_pattern_bool = True
    binary_pattern_bool = True
    measures_bool = True


#--- Initialization --#
    print('\n#--- Initialization ---#\n')

    os.chdir('D:/Universitaet Mannheim/MMDS 7. Semester/Master Thesis/Outline/Data/Cleaning Robots')

    patent_lda_ipc = pd.read_csv('patent_lda_ipc.csv', quotechar='"', skipinitialspace=True)
    topics = pd.read_csv('patent_topics.csv', quotechar='"', skipinitialspace=True)
    og_ipc = pd.read_csv('cleaning_robot_EP_patents_IPC.csv', quotechar='"', skipinitialspace=True)

    patent_lda_ipc = patent_lda_ipc.to_numpy()
    topics = topics.to_numpy()
    og_ipc = og_ipc.to_numpy()

    with open('window90by1', 'rb') as handle:
        window90by1 = pk.load(handle)

#--- Overview --#
    print('\n#--- Overview ---#\n')

    #print(len(np.unique(og_ipc[:,0])))      # 3844 unique patent id's (including german and france patents)
    #print(len(np.unique(og_ipc[:,1])))      # 970 unique ipcs
    #print(len(window90by1))                 # 5937 windows

    #print(patent_lda_ipc[0])                # Locate ipc and topic positions for ipc_position and topic_position
                                             # 9-29 topics, 30-end ipc's

    #print(patent_lda_ipc[0,9:30])
    #print(patent_lda_ipc[0,30])
    #print(np.shape(patent_lda_ipc))

#--- Preprocessing for plain pattern arrays (IPC's, topics) x (pair, tripple, singular)  --#
    print('\n#--- Preprocessing for plain pattern arrays (IPC\'s, topics) x (pair, tripple, singular) ---#\n')

    if preproc_bool == True:

        ipc_position = np.r_[range(30,np.shape(patent_lda_ipc)[1]-1,3)]             # right now, this has to be adjusted manually depending on the LDA results
        topic_position = np.r_[range(9,30,3)]                                       # right now, this has to be adjusted manually depending on the LDA results

        window90by1_ipcs_single = {}
        window90by1_topics_single = {}

        window90by1_ipcs_pairs = {}
        window90by1_topics_pairs = {}

        window90by1_ipcs_tripples = {}
        window90by1_topics_tripples = {}

        c = 0

        for window in window90by1.values():

            ipc_list = []
            topic_list = []

            ipc_pair_list = []
            topic_pair_list = []

            ipc_tripple_list = []
            topic_tripple_list = []

            for patent in window:

                # collect all ipc's within patents within a window
                y = [x for x in patent[ipc_position] if x == x]             # nan elimination
                y = np.unique(y)                                            # todo probably/hopefully not neccessary (because hopefully the data is clean enough so that one paper is not classfied with the same ipc more than once)
                ipc_list.append(tuple(y))                                   # for each window I get a list of tuples. one tuple represents the ipc's within a patent of the window

                # collect all topics within patents within a window
                z = [x for x in patent[topic_position] if x == x]           # nan elimination
                z = np.unique(z)                                            # todo probably/hopefully not neccessary (because hopefully the data is clean enough so that one paper is not classfied with the same topic more than once)
                topic_list.append(tuple(z))

                # collect all possible ipc pairs and triples within a patent within a window
                ipc_pair_list.append(list(itertools.combinations(y, r=2)))
                ipc_tripple_list.append(list(itertools.combinations(y, r=3)))

                # collect all possible topic pairs and triples within a patent within a window
                topic_pair_list.append(list(itertools.combinations(z, r=2)))
                topic_tripple_list.append(list(itertools.combinations(z, r=3)))


            # dictionary with all singularly occuring ipc's within a window
            ipc_list = [item for sublist in ipc_list for item in sublist]
            ipc_list = np.unique(ipc_list)
            window90by1_ipcs_single['window_{0}'.format(c)] = ipc_list

            # dictionary with all singularly occuring topics within a window
            topic_list = [item for sublist in topic_list for item in sublist]
            topic_list = np.unique(topic_list)
            window90by1_topics_single['window_{0}'.format(c)] = topic_list

            # dictionary with all possible pairs of ipc's within patents within a window
            # meaning one patent -> (possibly) multiple tuples of size two
            ipc_pair_list = [item for sublist in ipc_pair_list for item in sublist]
            window90by1_ipcs_pairs['window_{0}'.format(c)] = ipc_pair_list

            # dictionary with all possible pairs of topics within patents within a window
            # meaning one patent -> (possibly) multiple tuples of size two
            topic_pair_list = [item for sublist in topic_pair_list for item in sublist]
            window90by1_topics_pairs['window_{0}'.format(c)] = topic_pair_list

            # dictionary with all possible tripples of ipc's within patents within a window
            # meaning one patent -> (possibly) multiple tuples of size three
            ipc_tripple_list = [item for sublist in ipc_tripple_list for item in sublist]
            window90by1_ipcs_tripples['window_{0}'.format(c)] = ipc_tripple_list

            # dictionary with all possible tripples of topics within patents within a window
            # meaning one patent -> (possibly) multiple tuples of size three
            topic_tripple_list = [item for sublist in topic_tripple_list for item in sublist]
            window90by1_topics_tripples['window_{0}'.format(c)] = topic_tripple_list

            c = c + 1


        filename = 'window90by1_ipcs_single'
        outfile = open(filename, 'wb')
        pk.dump(window90by1_ipcs_single, outfile)
        outfile.close()

        filename = 'window90by1_topics_single'
        outfile = open(filename, 'wb')
        pk.dump(window90by1_topics_single, outfile)
        outfile.close()

        filename = 'window90by1_ipcs_pairs'
        outfile = open(filename, 'wb')
        pk.dump(window90by1_ipcs_pairs, outfile)
        outfile.close()

        filename = 'window90by1_topics_pairs'
        outfile = open(filename, 'wb')
        pk.dump(window90by1_topics_pairs, outfile)
        outfile.close()

        filename = 'window90by1_ipcs_tripples'
        outfile = open(filename, 'wb')
        pk.dump(window90by1_ipcs_tripples, outfile)
        outfile.close()

        filename = 'window90by1_topics_tripples'
        outfile = open(filename, 'wb')
        pk.dump(window90by1_topics_tripples, outfile)
        outfile.close()



#--- Constructing plain pattern arrays (IPC\'s, topics) x (pair, tripple, singular)  ---#
    print('\n#--- Constructing plain pattern arrays (IPC\'s, topics) x (pair, tripple, singular)  ---#\n')

    if plain_pattern_bool == True:


    ### IPC Sigles ###
        #...
    ### IPC Pairs ###


        with open('window90by1_ipcs_pairs', 'rb') as handle:
            window90by1_ipcs_pairs = pk.load(handle)

        # Identify unique ipc pairs in the whole dictionary for the column dimention of the pattern array #

        tuple_list = []
        for i in window90by1_ipcs_pairs.values():

            tuple_list.append(i)

        tuple_list = [item for sublist in tuple_list for item in sublist]
        print('number of all tuples before taking only the unique ones: ', len(tuple_list))   # 1047572
        tuple_list, tuple_list_counts = np.unique(tuple_list, return_counts=True, axis=0)
        print('number of all tuples after taking only the unique ones (number of columns in the pattern array): ', len(tuple_list))    # 5445
        #print(tuple_list_counts)        # where does the 90 and the "weird" values come from? explaination: if a combination occures in the whole timeframe only once (in one patent) then it is captures 90 times. The reason for this is the size of the sliding window of 90 and the sliding by one day. One patent will thereby be capured in 90 sliding windows (excaption: the patents in the first and last 90 days of the overall timeframe, they are capture in less then 90 sliding windows)

        window_list = window90by1_ipcs_pairs.keys()
        print('number of all windows (number of rows in the pattent array): ', len(window_list))


        # New array, including space for occurence pattern - ipc pairs #

        pattern = np.zeros((len(window_list), len(tuple_list)))
        print(np.shape(pattern))                        # (5937, 5445)


        # Populate occurence pattern - ipc pairs #

        pbar = tqdm.tqdm(total=len(window_list))
        c_i = 0

        for i in window_list:
            c_j = 0

            for j in tuple_list:
                if tuple(j) in window90by1_ipcs_pairs[i]:
                    #pattern[c_i,c_j] = 1                                        # results in sum(sum(array)) =  869062.0
                    pattern[c_i,c_j] = window90by1_ipcs_pairs[i].count(tuple(j)) # results in sum(sum(array)) = 1047572.0

                c_j = c_j +1

            c_i = c_i +1
            pbar.update(1)

        pbar.close()


        filename = 'window90by1_ipcs_pairs_pattern'
        outfile = open(filename, 'wb')
        pk.dump(pattern, outfile)
        outfile.close()

    ### IPC Tripples ###
        # ...

    ### Topic Sigles ###
        # ...
    ### Topic Pairs ###
        # ...
    ### Topic Tripples ###
        # ...

#--- Normalized plain pattern arrays (IPC\'s, topics) x (pair, tripple, singular)  ---#
    print('\n#--- Normalized plain pattern arrays (IPC\'s, topics) x (pair, tripple, singular)  ---#\n')

    if norm_pattern_bool == True:

    ### IPC Sigles ###
        # ...
    ### IPC Pairs ###

        with open('window90by1_ipcs_pairs_pattern', 'rb') as handle:
            pattern = pk.load(handle)

        #print(np.amax(pattern))                                 # 15

        window_sum = pattern.sum(axis=1)

        pattern_norm = pattern / window_sum[:, np.newaxis]

        #window_sum_test = pattern_norm.sum(axis=1)
        #print(max(window_sum_test))

        '''
        filename = 'window90by1_ipcs_pairs_pattern_norm'
        outfile = open(filename, 'wb')
        pk.dump(pattern_norm, outfile)
        outfile.close()
        '''

    ### IPC Tripples ###
        # ...

    ### Topic Sigles ###
        # ...
    ### Topic Pairs ###
        # ...
    ### Topic Tripples ###
        # ...

#--- Binarized plain pattern arrays (IPC\'s, topics) x (pair, tripple, singular)  ---#
    print('\n#--- Binarized plain pattern arrays (IPC\'s, topics) x (pair, tripple, singular)  ---#\n')

    if binary_pattern_bool == True:

        '''
        with open('window90by1_ipcs_pairs_pattern_norm', 'rb') as handle:
            pattern_norm = pk.load(handle)
        '''
        pattern_thres = np.where(pattern_norm < 0.01, 0, 1)                    # arbitrary threshold of 0.01

        '''
        filename = 'window90by1_ipcs_pairs_pattern_thres'
        outfile = open(filename, 'wb')
        pk.dump(pattern_thres, outfile)
        outfile.close()
        '''



#--- Measuring recombination, diffusion of recombination and diffusion of singular ipc's/topics  ---#
    print('\n#--- Measuring recombination, diffusion of recombination and diffusion of singular ipc\'s/topics  ---#\n')

    # Recombination are located along the window-pair axes. Subsequently it is measured how long this recombination
    # diffuses. The resulting data structure looks like [[window, comb, duration],[w,c,d],...]
    # For further insights the w and c coordinates can be used to review the unnormalized and unbinarized pattern array.

    if measures_bool == True:

        '''
        with open('window90by1_ipcs_pairs_pattern_thres', 'rb') as handle:
            pattern_thres = pk.load(handle)
        '''

        ### finding recombinations ###

        recomb_pos = []

        c = 0
        for combinations in pattern_thres.T:
            for window_pos in range(len(combinations)):
                if window_pos != 0:
                    if combinations[window_pos] == 1:
                        if combinations[window_pos-1] == 0:
                            recomb_pos.append([window_pos, c])

            c = c + 1


        ### counting diffusion ###

        diffusion_duration_list = []

        for recomb in recomb_pos:
            diffusion = -1
            i = 0

            while pattern_thres[recomb[0]+i,recomb[1]] == 1:
                diffusion = diffusion + 1
                i = i + 1
                if recomb[0]+i == len(pattern_thres):
                    break

            diffusion_duration_list.append(diffusion)

        ### Merge both lists to get final data structure ###

        for i in range(len(recomb_pos)):
            recomb_pos[i].append(diffusion_duration_list[i])

        print(recomb_pos)
        print(recomb_pos[0])

#--- introduce leeway ---#
    '''
    
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
        ''''''
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
        ''''''
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
    
    
    ''''''
    arr = np.array([2, 0, 0, 0, 0, 1, 0, 1, 0, 0])
    
    seq = np.array([0,0])
    
    print(search_sequence_numpy(arr,seq))
    ''''''
    
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
    
    
    ''''''
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
    ''''''
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
    '''
    '''
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
    
    '''


'''
        with open('window90by1_ipcs_pairs', 'rb') as handle:
            window90by1_ipcs_pairs = pk.load(handle)



        ipc_list = []
        for i in window90by1_ipcs_pairs.values():

            ipc_list.append(i)

        ipc_list = [item for sublist in ipc_list for item in sublist]

        print('number of all tuples before taking only the unique ones', len(ipc_list))  # 1047572
        ipc_list, ipc_list_counts = np.unique(ipc_list, return_counts=True, axis=0)

        print(len(ipc_list))

        window_list = window90by1_ipcs_pairs.keys()

        pattern = np.zeros((len(window_list), len(ipc_list)))
        print(np.shape(pattern))


        print(ipc_list)
        print(window_list)
        print(pattern)

        import tqdm

        print('--------------------------')
        print(sum(sum(pattern)))

        pbar = tqdm.tqdm(total=len(window_list))

        c_i = 0
        for i in window_list:
            c_j = 0

            for j in ipc_list:

                if j in window90by1_ipcs[i]:
                    # pattern[c_i,c_j] = 1                                           # results in sum(sum(array)) = 869062.0
                    pattern[c_i, c_j] = window90by1_ipcs[i].count(j)

                c_j = c_j + 1

            c_i = c_i + 1
            pbar.update(1)

        pbar.close()

        print(sum(sum(pattern)))

        filename = 'window90by1_ipcs_pattern'
        outfile = open(filename, 'wb')
        pk.dump(pattern, outfile)
        outfile.close()
'''