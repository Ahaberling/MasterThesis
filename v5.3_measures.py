
if __name__ == '__main__':

    # --- Import Libraries ---#
    print('\n#--- Import Libraries ---#\n')

    import pandas as pd
    import numpy as np
    import pickle as pk

    import os
    import sys
    import tqdm

    import itertools
    import scipy.signal as signal

    preproc_bool = True
    pattern_bool = True
    measures_bool = True
    imputation_bool = True

    # --- Initialization --#
    print('\n#--- Initialization ---#\n')

    os.chdir('D:/Universitaet Mannheim/MMDS 7. Semester/Master Thesis/Outline/Data/Cleaning Robots')

    patent_topicDist = pd.read_csv('patent_topicDist_mallet.csv', quotechar='"', skipinitialspace=True)
    patent_IPC = pd.read_csv('cleaning_robot_EP_patents_IPC.csv', quotechar='"', skipinitialspace=True)

    patent_topicDist = patent_topicDist.to_numpy()
    patent_IPC = patent_IPC.to_numpy()

    patent_lda_ipc = patent_topicDist

    with open('slidingWindow_dict', 'rb') as handle:
        slidingWindow_dict = pk.load(handle)

    # --- Preprocessing for pattern arrays (IPC's, topics) x (singular, pair, tripple)  --#
    print('\n#--- Preprocessing for pattern arrays (IPC\'s, topics) x (singluar, pair, tripple) ---#\n')

    print(np.shape(patent_lda_ipc))

    topic_end = 23

    if preproc_bool == True:

        ipc_position = np.r_[range(30, np.shape(patent_lda_ipc)[1] - 1,
                                   3)]  # right now, this has to be adjusted manually depending on the LDA results #todo adjust
        topic_position = np.r_[
            range(9, 22, 2)]  # right now, this has to be adjusted manually depending on the LDA results #todo adjust

        slidingWindow_ipcs_single = {}
        slidingWindow_topics_single = {}

        slidingWindow_ipcs_pairs = {}
        slidingWindow_topics_pairs = {}

        slidingWindow_ipcs_triples = {}
        slidingWindow_topics_triples = {}

        c = 0
        pbar = tqdm.tqdm(total=len(slidingWindow_dict))

        for window in slidingWindow_dict.values():

            ipc_list = []
            topic_list = []

            ipc_pair_list = []
            topic_pair_list = []

            ipc_tripple_list = []
            topic_tripple_list = []

            for patent in window:
                # collect all ipc's within patents within a window
                y = [x for x in patent[ipc_position] if x == x]  # nan elimination
                y = np.unique(
                    y)  # todo probably/hopefully not neccessary (because hopefully the data is clean enough so that one paper is not classfied with the same ipc more than once)
                ipc_list.append(tuple(
                    y))  # for each window I get a list of tuples. one tuple represents the ipc's within a patent of the window

                # collect all topics within patents within a window
                z = [x for x in patent[topic_position] if x == x]  # nan elimination
                z = np.unique(
                    z)  # todo probably/hopefully not neccessary (because hopefully the data is clean enough so that one paper is not classfied with the same topic more than once)
                topic_list.append(tuple(z))

                # collect all possible ipc pairs and triples within a patent within a window
                ipc_pair_list.append(list(itertools.combinations(y, r=2)))
                ipc_tripple_list.append(list(itertools.combinations(y, r=3)))

                # collect all possible topic pairs and triples within a patent within a window
                topic_pair_list.append(list(itertools.combinations(z, r=2)))
                topic_tripple_list.append(list(itertools.combinations(z, r=3)))

            # dictionary with all singularly occuring ipc's within a window
            ipc_list = [item for sublist in ipc_list for item in sublist]
            window90by1_ipcs_single['window_{0}'.format(c)] = ipc_list

            # dictionary with all singularly occuring topics within a window
            topic_list = [item for sublist in topic_list for item in sublist]
            window90by1_topics_single['window_{0}'.format(c)] = topic_list

            # dictionary with all possible pairs of ipc's within patents within a window
            # meaning one patent -> (possibly) multiple tuples of size two
            ipc_pair_list = [item for sublist in ipc_pair_list for item in sublist]
            window90by1_ipcs_pairs['window_{0}'.format(c)] = ipc_pair_list

            # dictionary with all possible pairs of topics within patents within a window
            # meaning one patent -> (possibly) multiple tuples of size two
            topic_pair_list = [item for sublist in topic_pair_list for item in sublist]
            window90by1_topics_pairs['window_{0}'.format(c)] = topic_pair_list

            # dictionary with all possible triples of ipc's within patents within a window
            # meaning one patent -> (possibly) multiple tuples of size three
            ipc_tripple_list = [item for sublist in ipc_tripple_list for item in sublist]
            window90by1_ipcs_triples['window_{0}'.format(c)] = ipc_tripple_list

            # dictionary with all possible triples of topics within patents within a window
            # meaning one patent -> (possibly) multiple tuples of size three
            topic_tripple_list = [item for sublist in topic_tripple_list for item in sublist]
            window90by1_topics_triples['window_{0}'.format(c)] = topic_tripple_list

            c = c + 1
            pbar.update(1)

        pbar.close()

        ### Save preprocessing ###

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

        filename = 'window90by1_ipcs_triples'
        outfile = open(filename, 'wb')
        pk.dump(window90by1_ipcs_triples, outfile)
        outfile.close()

        filename = 'window90by1_topics_triples'
        outfile = open(filename, 'wb')
        pk.dump(window90by1_topics_triples, outfile)
        outfile.close()

    # --- Constructing pattern arrays (IPC\'s, topics) x (singular, pair, tripple) ---#
    print('\n#--- Constructing pattern arrays (IPC\'s, topics) x (singular, pair, tripple) ---#\n')

    if pattern_bool == True:

        ### IPC Sigles ###
        print('\n\t### IPC Sigles ###\n')

        with open('window90by1_ipcs_single', 'rb') as handle:
            window90by1_ipcs_single = pk.load(handle)

        single_list = []

        for i in window90by1_ipcs_single.values():
            single_list.append(i)

        single_list = [item for sublist in single_list for item in sublist]
        single_list, single_list_counts = np.unique(single_list, return_counts=True, axis=0)

        window_list = window90by1_ipcs_single.keys()

        # New array, including space for occurence pattern - ipc singles #

        pattern_ipc_singles = np.zeros((len(window_list), len(single_list)))
        # print(np.shape(pattern_ipc_singles))  # (5937, 953)

        # Populate occurence pattern - ipc singles #

        c_i = 0
        pbar = tqdm.tqdm(total=len(window_list))

        for i in window_list:
            c_j = 0

            for j in single_list:
                if j in window90by1_ipcs_single[i]:
                    pattern_ipc_singles[c_i, c_j] = list(window90by1_ipcs_single[i]).count(j)

                c_j = c_j + 1
            c_i = c_i + 1
            pbar.update(1)

        pbar.close()

        filename = 'window90by1_ipcs_singles_pattern'
        outfile = open(filename, 'wb')
        pk.dump(pattern_ipc_singles, outfile)
        outfile.close()

        ### IPC Pairs ###
        print('\n\t### IPC Pairs ###\n')

        with open('window90by1_ipcs_pairs', 'rb') as handle:
            window90by1_ipcs_pairs = pk.load(handle)

        # Identify unique ipc pairs in the whole dictionary for the column dimension of the pattern array #

        tuple_list = []
        for i in window90by1_ipcs_pairs.values():
            tuple_list.append(i)

        tuple_list = [item for sublist in tuple_list for item in sublist]
        # print('number of all tuples before taking only the unique ones: ', len(tuple_list))         # 1047572
        tuple_list, tuple_list_counts = np.unique(tuple_list, return_counts=True, axis=0)
        # print('number of all tuples after taking only the unique ones (number of columns in the pattern array): ', len(tuple_list))    # 5445
        # print(tuple_list_counts)        # where does the 90 and the "weird" values come from? explaination: if a combination occures in the whole timeframe only once (in one patent) then it is captures 90 times. The reason for this is the size of the sliding window of 90 and the sliding by one day. One patent will thereby be capured in 90 sliding windows (excaption: the patents in the first and last 90 days of the overall timeframe, they are capture in less then 90 sliding windows)

        window_list = window90by1_ipcs_pairs.keys()
        # print('number of all windows (number of rows in the pattent array): ', len(window_list))    # 5937

        # New array, including space for occurence pattern - ipc pairs #

        pattern_ipc_pairs = np.zeros((len(window_list), len(tuple_list)))
        # print(np.shape(pattern_ipc_pairs))                        # (5937, 5445)

        # Populate occurence pattern - ipc pairs #

        c_i = 0
        pbar = tqdm.tqdm(total=len(window_list))

        for i in window_list:
            c_j = 0

            for j in tuple_list:

                if tuple(j) in window90by1_ipcs_pairs[i]:
                    # pattern_ipc_pairs[c_i,c_j] = 1                                        # results in sum(sum(array)) =  869062.0
                    pattern_ipc_pairs[c_i, c_j] = window90by1_ipcs_pairs[i].count(
                        tuple(j))  # results in sum(sum(array)) = 1047572.0

                c_j = c_j + 1

            c_i = c_i + 1
            pbar.update(1)

        pbar.close()

        filename = 'window90by1_ipcs_pairs_pattern'
        outfile = open(filename, 'wb')
        pk.dump(pattern_ipc_pairs, outfile)
        outfile.close()

    ### IPC triples ###
    # ...
    ### Topic Sigles ###
    # ...
    ### Topic Pairs ###
    # ...
    ### Topic triples ###
    # ...

    # --- Normalized plain pattern arrays (IPC\'s, topics) x (singular, pair, triple)  ---#
    print('\n#--- Normalized plain pattern arrays (IPC\'s, topics) x (singular, pair, triple)  ---#\n')

    if measures_bool == True:

        ### IPC Sigles ###

        with open('window90by1_ipcs_singles_pattern', 'rb') as handle:
            pattern_ipc_singles = pk.load(handle)

        # print(np.amax(pattern_ipc_singles))                                 # 59

        window_sum = pattern_ipc_singles.sum(axis=1)
        pattern_ipc_singles_norm = pattern_ipc_singles / window_sum[:, np.newaxis]

        ### IPC Pairs ###

        with open('window90by1_ipcs_pairs_pattern', 'rb') as handle:
            pattern_ipc_pairs = pk.load(handle)

        # print(np.amax(pattern_ipc_pairs))                                   # 15

        window_sum = pattern_ipc_pairs.sum(axis=1)
        pattern_ipc_pairs_norm = pattern_ipc_pairs / window_sum[:, np.newaxis]

        window_sum_test = pattern_ipc_singles_norm.sum(axis=1)
        # print(max(window_sum_test))                                         # 1.0000000000000002

        ### IPC Triples ###
        # ...
        ### Topic Sigles ###
        # ...
        ### Topic Pairs ###
        # ...
        ### Topic Triples ###
        # ...

        # --- Binarized plain pattern arrays (IPC\'s, topics) x (singular, pair, triple)  ---#
        print('\n#--- Binarized plain pattern arrays (IPC\'s, topics) x (singular, pair, triple)  ---#\n')

        ### IPC singles ###

        pattern_ipc_singles_thres = np.where(pattern_ipc_singles_norm < 0.01, 0, 1)  # arbitrary threshold of 0.01

        ### IPC pairs ###

        pattern_ipc_pairs_thres = np.where(pattern_ipc_pairs_norm < 0.01, 0, 1)  # arbitrary threshold of 0.01

        # --- Measuring recombination, diffusion of recombination and diffusion of singular ipc's/topics  ---#
        print(
            '\n#--- Measuring recombination, diffusion of recombination and diffusion of singular ipc\'s/topics  ---#\n')

        # The resulting data structure looks like [[window_position, ipc_position, duration],[w,i,d],...]
        # For further insights the w and i coordinates can be used to review the unnormalized and unbinarized pattern array.

        ### IPC singles ###
        print('\n\t### IPC singles ###')
        print('\t Locating diffusion\n')

        # finding diffusion patterns #

        diff_pos = []

        c = 0
        pbar = tqdm.tqdm(total=len(pattern_ipc_singles_thres.T))

        for ipc in pattern_ipc_singles_thres.T:
            for window_pos in range(len(pattern_ipc_singles_thres)):
                if window_pos != 0:
                    if ipc[window_pos] == 1:
                        if ipc[window_pos - 1] == 0:
                            diff_pos.append([window_pos, c])

            c = c + 1
            pbar.update(1)

        pbar.close()

        # print(diff_pos)             # [[2171, 2], [2206, 2], [2213, 2], [2220, 2], [3074, 2], [3081, 2], [3088, 2], [2171, 5], [2206, 5], [2213, 5]
        #  [2220, 5], [3074, 5], [3081, 5], [3088, 5], [3383, 5], [3431, 5], [3445, 5], [3635, 5], [3747, 5], ...
        # print(len(diff_pos))        # 3095

        '''
        print(pattern_ipc_singles_thres[2170:2175, 2])
        print(pattern_ipc_singles_thres[2205:2210, 2])
        print(pattern_ipc_singles_thres[3080:3085, 5])
        print(pattern_ipc_singles_thres[232:240, 132])
        print(pattern_ipc_singles_thres[3151:3155, 126])
        print(pattern_ipc_singles_thres[4096:4100, 420])
        '''

        # counting diffusion #
        print('\t Measuring diffusion\n')

        diffusion_duration_list = []
        pbar = tqdm.tqdm(total=len(diff_pos))

        for occurrence in diff_pos:
            diffusion = -1
            i = 0

            while pattern_ipc_singles_thres[occurrence[0] + i, occurrence[1]] == 1:
                diffusion = diffusion + 1

                i = i + 1
                if occurrence[0] + i == len(pattern_ipc_singles_thres):
                    break

            diffusion_duration_list.append(diffusion)

            pbar.update(1)

        pbar.close()

        # print(diffusion_duration_list)          # [0, 0, 0, 13, 0, 0, 20, 0, 0, 0, 13, 0, 0, 20, 41, 7, 0, 89, 89, 152, 5, 229, 90, 0, 6,
        # print(len(diffusion_duration_list))     # 3095

        # Merge both lists to get final data structure #

        for i in range(len(diff_pos)):
            diff_pos[i].append(diffusion_duration_list[i])

        # print(diff_pos)
        # print(len(diff_pos))

        # print('\tPositions and duration of diffusion: \n', diff_pos)        # [[2171, 2, 0], [2206, 2, 0], [2213, 2, 0], [2220, 2, 13], [3074, 2, 0], [3081, 2, 0],
        # print('\n\tExample with first entry: \n', diff_pos[0])               # [2171, 2, 0]

        np.set_printoptions(threshold=sys.maxsize)
        # print('\n',pattern_ipc_singles_thres[2170:2175,2])                   # [0          1          0          0          0]
        # print(pattern_ipc_singles[2170:2175,2])                              # [1.         1.         1.         1.         1.]
        # print(pattern_ipc_singles_norm[2170:2175,2])                         # [0.00869565 0.01041667 0.00877193 0.00877193 0.00877193]

        '''
        print('\n',pattern_ipc_singles_thres[2205:2210,2])
        print(pattern_ipc_singles[2205:2210,2])
        print(pattern_ipc_singles_norm[2205:2210,2])
        '''
        print('\n', pattern_ipc_singles_thres[2212:2217, 2])
        print(pattern_ipc_singles[2212:2217, 2])
        print(pattern_ipc_singles_norm[2212:2217, 2])

        print('\n', pattern_ipc_singles_thres[2219:2241, 2])
        print(pattern_ipc_singles[2219:2241, 2])
        print(pattern_ipc_singles_norm[2219:2241, 2])
        '''
        print('\n',pattern_ipc_singles_thres[1093:1110,216])                # [1094, 216, 13]
        print(pattern_ipc_singles[1093:1110,216])                           # [1094, 216, 13]
        print(pattern_ipc_singles_norm[1093:1110,216])                      # [1094, 216, 13]

        print('\n',pattern_ipc_singles_thres[3654:3685,276])                # [3655, 276, 27]
        print(pattern_ipc_singles[3654:3685,276])                           # [3655, 276, 27]
        print(pattern_ipc_singles_norm[3654:3685,276])                      # [3655, 276, 27]

        print('\n',pattern_ipc_singles_thres[3745:3750,501])                # [3746, 501, 0]
        print(pattern_ipc_singles[3745:3750,501])                           # [3746, 501, 0]
        print(pattern_ipc_singles_norm[3745:3750,501])                      # [3746, 501, 0]
        '''

        # --------------------------------------------------------------------

        ### ipc pairs ###
        print('\n\t### IPC pairs ###')
        print('\t Locating recombination\n')

        recomb_pos = []

        c = 0
        pbar = tqdm.tqdm(total=len(pattern_ipc_pairs_thres))

        for combinations in pattern_ipc_pairs_thres.T:
            for window_pos in range(len(combinations)):
                if window_pos != 0:
                    if combinations[window_pos] == 1:
                        if combinations[window_pos - 1] == 0:
                            recomb_pos.append([window_pos, c])

            c = c + 1
            pbar.update(1)

        pbar.close()

        # counting diffusion #
        print('\t Measuring diffusion of recombination\n')

        diffusion_duration_list = []
        pbar = tqdm.tqdm(total=len(recomb_pos))

        for recomb in recomb_pos:
            diffusion = -1
            i = 0

            while pattern_ipc_pairs_thres[recomb[0] + i, recomb[1]] == 1:
                diffusion = diffusion + 1
                i = i + 1
                if recomb[0] + i == len(pattern_ipc_pairs_thres):
                    break

            diffusion_duration_list.append(diffusion)
            pbar.update(1)

        pbar.close()

        # Merge both lists to get final data structure #

        for i in range(len(recomb_pos)):
            recomb_pos[i].append(diffusion_duration_list[i])

        # print('\tPositions and diffusion duration of recombinations: \n', recomb_pos)       # [[2171, 7, 0], [2178, 7, 14], [2206, 7, 27], [3088, 7, 20], [2171, 8, 0],
        # print('\tExample with fourth entry: \n', recomb_pos[3])                             #                                              [3088, 7, 20]

        # print('\n', pattern_ipc_pairs_thres[3087:3110,7])               #  [0  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  0 ]
        # print(pattern_ipc_pairs[3087:3110,7])                           #  [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 0.]
        # print(pattern_ipc_pairs_norm[3087:3110,7])                      # [0.00869565 0.0125     0.0125     0.0125     0.0125     0.0125
        #  0.0125     0.0125     0.01298701 0.01162791 0.01162791 0.01162791
        #  0.01162791 0.01162791 0.01162791 0.01351351 0.0125     0.0125
        #  0.0125     0.0125     0.0125     0.0125     0.        ]

        '''
        print('\n', pattern_ipc_pairs_thres[2177:2194,8])               # [2178, 8, 14]
        print(pattern_ipc_pairs[2177:2194,8])
        print(pattern_ipc_pairs_norm[2177:2194,8])

        print('\n', pattern_ipc_pairs_thres[2835:2852,261])               # [2836, 261, 14]
        print(pattern_ipc_pairs[2835:2852,261])
        print(pattern_ipc_pairs_norm[2835:2852,261])

        print('\n', pattern_ipc_pairs_thres[3290:3294,572])               # [3291, 572, 0]
        print(pattern_ipc_pairs[3290:3294,572])
        print(pattern_ipc_pairs_norm[3290:3294,572])
        '''

    # --- introduce leeway ---#

    if imputation_bool == True:

        def search_sequence_numpy(arr, seq):
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
            M = (arr[np.arange(Na - Nseq + 1)[:, None] + r_seq] == seq).all(1)
            """
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
            """

            # Get the range of those indices as final output
            if M.any() > 0:
                return np.where(np.convolve(M, np.ones((Nseq), dtype=int)) > 0)[0]
            else:
                return []  # No match found


        ### Identifying where sequences occur ###

        seq = np.array([1, 0, 1])

        c = 0
        for i in pattern_ipc_pairs_thres.T:
            arr = i
            print(c, search_sequence_numpy(arr, seq))  # 2747, 2847, 2860, 2936, 3060, 3138

            c = c + 1
            # break

        # np.set_printoptions(threshold=sys.maxsize)
        # print(pattern_wThreshold.T[2746])
        # print('before', pattern_wThreshold.T[2747])
        # print(sum(pattern_wThreshold.T[2747]))
        # print('before', pattern_wThreshold.T[2709])
        # print(sum(pattern_wThreshold.T[2709]))
        # print(pattern_wThreshold.T[2748])

        ### Replacing sequences 101 with 111 ###

        c = 0
        for i in pattern_ipc_pairs_thres.T:
            arr = i
            # print(c, search_sequence_numpy(arr, seq))                       # 2747, 2847, 2860, 2936, 3060, 3138

            k = seq  # kernel for convolution
            i[(signal.convolve(i, k, 'same') == 2) & (i == 0)] = 1
            """
            print(i)
            print('signal.convolve(i, k, \'same\')', signal.convolve(i, k, 'same'))
            print('signal.convolve(i, k, \'same\') == 2', signal.convolve(i, k, 'same') == 2)
            print('i == 0', i == 0)
            print('signal.convolve(i, k, \'same\') == 2 & (i == 0)', signal.convolve(i, k, 'same') == 2 & (i == 0))
            """
            pattern_ipc_pairs_thres.T[c, :] = i

            c = c + 1
            # break

        # todo problem 1: imputing sequences only works for 101 case, not for 100001, and so on
        # todo problem 2: with only 0 and 1 a diffusion cycle is identified if the threshold is met with one set of patents,
        # that does not change anymore for 90 days. E.g. tuple occurs in x patents. x patents were all published on y
        # (no diffusion prossible, because to little time inbetween) nevertheless the patents x might meet the thresshold
        # for t until t+89. So diffusion might be only considered if it exceeds 90 days

        # todo idea: right now window90by1_ipcs_pairs contains tuples like ('C12M   1', 'C12M   3'). If this is to fine grained (no real inovation/ recombination) then go more course graind (or fine grained)


