import numpy as np
import tqdm
import itertools
import scipy.signal as sciSignal
from cdlib import algorithms
import networkx as nx
import operator
import copy

class ComparativeMeasures:

    @staticmethod
    def modify_arrays(array, threshold):
        row_sum = array.sum(axis=1)
        row_sum = np.where(row_sum < 1, 0.000001, row_sum)          # smoothing to avoid dividing by 0
        array_frac = array / row_sum[:, np.newaxis]

        array_threshold = np.where(array_frac < threshold, 0, 1)

        return array_threshold, array_frac

    @staticmethod
    def find_patternStart(pattern_array_thresh):

        recom_pos = []
        c = 0

        pbar = tqdm.tqdm(total=len(pattern_array_thresh.T))

        for column in pattern_array_thresh.T:
            for row in range(len(pattern_array_thresh)):
                if row != 0:
                    if column[row] == 1:
                        if column[row - 1] == 0:
                            recom_pos.append([row, c])

            c = c + 1
            pbar.update(1)

        pbar.close()
        return recom_pos

    @staticmethod
    def find_pattern_length(pattern_array_thresh, recombinationPos):

        diffu_list = []
        pbar = tqdm.tqdm(total=len(recombinationPos))

        for pos in recombinationPos:
            diffusion = -1
            i = 0

            while pattern_array_thresh[pos[0] + i, pos[1]] == 1:
                diffusion = diffusion + 1

                i = i + 1
                if pos[0] + i == len(pattern_array_thresh):
                    break

            diffu_list.append(diffusion)

            pbar.update(1)

        pbar.close()

        # print(diffusion_duration_list)          # [0, 0, 0, 13, 0, 0, 20, 0, 0, 0, 13, 0, 0, 20, 41, 7, 0, 89, 89, 152, 5, 229, 90, 0, 6,
        # print(len(diffusion_duration_list))     # 3095

        # Merge both lists to get final data structure #

        for i in range(len(recombinationPos)):
            recombinationPos[i].append(diffu_list[i])

        return recombinationPos

    @staticmethod
    def introcude_leeway(pattern_array_thresh, sequence, impute_value):

        c = 0
        for row in pattern_array_thresh.T:
            row[(sciSignal.convolve(row, sequence, 'same') == 2) & (row == 0)] = impute_value

            pattern_array_thresh.T[c, :] = row

            c = c + 1
        return pattern_array_thresh

    @staticmethod
    def check_columnLength(list_of_allArrays, diffusionArray_Topics_lp_columns):
        for i in range(len(list_of_allArrays)):
            if len(list_of_allArrays[i].T) != len((list(diffusionArray_Topics_lp_columns))):
                raise Exception("Diffusion arrays vary in their columns")
        return

    @staticmethod
    def normalized_and_binarize(list_of_allArrays, threshold, leeway):
        array_rowNorm_list = []
        array_binariz_list = []
        for i in range(len(list_of_allArrays)):
            array_toBeMod = list_of_allArrays[i]

            array_threshold, array_rowNorm = ComparativeMeasures.modify_arrays(array_toBeMod, threshold)

            if leeway == True:
                array_threshold_leeway = ComparativeMeasures.introcude_leeway(array_threshold, np.array([1, 0, 1]), 1)
                # if there is 101 this means that there was one year without a occurrence. 1001 is one year and one month ...


            array_rowNorm_list.append(array_rowNorm)
            array_binariz_list.append(array_threshold_leeway)
        return array_binariz_list, array_rowNorm_list

    @staticmethod
    def alligned_SCM_descriptives(diffusionArray_Topics_lp_columns, diffusion_length_list):
        diffu_count_per_topic = []
        diffu_duration_per_topic = []

        for topic in range(len(diffusionArray_Topics_lp_columns)):
            diff_count = 0
            diff_duration = []
            for entry in diffusion_length_list:
                if entry[1] == topic:
                    diff_count = diff_count+1
                    diff_duration.append(entry[2])

            diffu_count_per_topic.append(diff_count)
            diffu_duration_per_topic.append(diff_duration)

        return  diffu_count_per_topic, diffu_duration_per_topic

    @staticmethod
    def cosine_sim_mod(List1, List2):
        if sum(List1) == 0 or sum(List2) == 0:
            result = 0
        else:
            result = np.dot(List1, List2) / (np.linalg.norm(List1) * np.linalg.norm(List2))
        return (result)

    @staticmethod
    def manhattan_sim_mod(List1, List2):
        return 1 - sum(abs(a - b) for a, b in zip(List1, List2)) / len(List1)


    @staticmethod
    def SCM_similarities_byColumn(list_of_allArrays_names, list_of_allArrays_threshold):
        # name_list = []
        # array_list = []
        namePair_list = []
        arrayPair_list = []
        # for i in range(len(list_of_allArrays_threshold)):
        # name_list.append(name) # , array))
        # array_list.append(array)

        namePair_list.append(list(itertools.combinations(list_of_allArrays_names, r=2)))
        namePair_list = namePair_list[0]
        arrayPair_list.append(list(itertools.combinations(list_of_allArrays_threshold, r=2)))
        arrayPair_list = arrayPair_list[0]

        similarityPair_list_cosine = []
        similarityPair_list_manhattan = []

        for matrixPair in arrayPair_list:
            patternArray1 = matrixPair[0]
            patternArray2 = matrixPair[1]

            cosine_list = []
            manhattan_list = []
            for column_id in range(len(patternArray1.T)):

                # print(sum(patternArray1[:,column_id]))
                # print(sum(patternArray2[:,column_id]))
                # when both are sum != 0 -> calculate normally
                # when one is != 0 -> calculate normally (cosine = 0)
                # when both are = 0 -> do not calculate anything

                # the columns of both arrays have to sum to 0 in order to not proceed
                if not (sum(patternArray1[:, column_id]) == 0 and sum(patternArray2[:, column_id]) == 0):

                    cosine = ComparativeMeasures.cosine_sim_mod(patternArray1[:, column_id],
                                                                patternArray2[:, column_id])


                    cosine_list.append(cosine)
                manhattan = ComparativeMeasures.manhattan_sim_mod(patternArray1[:, column_id],
                                                                  patternArray2[:, column_id])
                manhattan_list.append(manhattan)

            if len(cosine_list) != 0:               # this means: if at least in one column pair both columns were not completely 0
                cosine_avg = sum(cosine_list) / len(cosine_list)
            else:
                cosine_avg = 0

            if len(manhattan_list) != 0:            # this means: if at least in one column pair both columns were not completely 0
                manhattan_avg = sum(manhattan_list) / len(manhattan_list)
            else:
                manhattan_avg = 0

            similarityPair_list_cosine.append(cosine_avg)  # here is one inside that is not working at all
            similarityPair_list_manhattan.append(manhattan_avg)

        return namePair_list, similarityPair_list_cosine, similarityPair_list_manhattan

    @staticmethod
    def SCM_topic_similarities(list_of_allArrays_names, list_of_allArrays_threshold):   #, slices_toExclude=None):
        namePair_list = []
        arrayPair_list = []
        namePair_list.append(list(itertools.combinations(list_of_allArrays_names, r=2)))
        namePair_list = namePair_list[0]
        arrayPair_list.append(list(itertools.combinations(list_of_allArrays_threshold, r=2)))
        arrayPair_list = arrayPair_list[0]

        simScores_withinTopic_list_cosine_avg = []
        simScores_withinTopic_list_manhattan_avg = []

        for column_id in range(len(list_of_allArrays_threshold[0].T)):

            simScores_withinTopic_cosine = []
            simScores_withinTopic_manhattan = []

            #namePair_list_withoutGM = list(np.delete(namePair_list, slices_toExclude, axis=0))
            #arrayPair_list_withoutGM = list(np.delete(arrayPair_list, slices_toExclude, axis=0))
            for matrixPair in arrayPair_list:
                patternArray1 = matrixPair[0]
                patternArray2 = matrixPair[1]

                if not (sum(patternArray1[:, column_id]) == 0 and sum(patternArray2[:, column_id]) == 0):
                    cosine = ComparativeMeasures.cosine_sim_mod(patternArray1[:, column_id],
                                                                patternArray2[:, column_id])
                    simScores_withinTopic_cosine.append(cosine)
                manhattan = ComparativeMeasures.manhattan_sim_mod(patternArray1[:, column_id],
                                                                  patternArray2[:, column_id])

                simScores_withinTopic_manhattan.append(manhattan)

            if len(simScores_withinTopic_cosine) != 0:
                simScores_withinTopic_list_cosine_avg.append(
                    sum(simScores_withinTopic_cosine) / len(simScores_withinTopic_cosine))
            else:
                #simScores_withinTopic_list_cosine_avg.append(-9999)
                simScores_withinTopic_list_cosine_avg.append(0)

            if len(simScores_withinTopic_manhattan) != 0:
                simScores_withinTopic_list_manhattan_avg.append(
                    sum(simScores_withinTopic_manhattan) / len(simScores_withinTopic_manhattan))
            else:
                simScores_withinTopic_list_manhattan_avg.append(-9999)
                #simScores_withinTopic_list_manhattan_avg.append(0)

        return simScores_withinTopic_list_cosine_avg, simScores_withinTopic_list_manhattan_avg

    @staticmethod
    def extend_cD_recombinationDiffuion(cd_Arrays, slidingWindow_size, cd_CCM_posStart, cd_CCM_posEnd):
        #modified_cDarrays = []
        for cd_array in cd_Arrays[cd_CCM_posStart:cd_CCM_posEnd]:

            for j in range(len(cd_array.T)):
                for i in range(len(cd_array) - 2, -1,
                               -1):  # -2 because otherwise i+1 would cause problems in the following lines

                    if cd_array[i, j] >= 1:
                        if len(cd_array[i:, j]) >= slidingWindow_size:
                            for k in range(1, slidingWindow_size):
                                cd_array[i + k, j] = cd_array[i + k, j] + cd_array[i, j]
                        else:
                            for k in range(1, len(cd_array[i:, j])):
                                cd_array[i + k, j] = cd_array[i + k, j] + cd_array[i, j]
            #modified_cDarrays.append(cd_array)
        return #modified_cDarrays

    @staticmethod
    def extend_recombination_columns(column_lists, recoArrays_threshold_list):
        all_recombs = [item for sublist in column_lists for item in sublist]
        #print(len(all_recombs))

        all_recombs = np.unique(all_recombs, axis=0)
        #print(len(all_recombs))

        all_recombs = [tuple(x) for x in all_recombs]

        extended_arrays = []

        pbar = tqdm.tqdm(total=len(column_lists))
        for i in range(len(column_lists)):
            extended_array = recoArrays_threshold_list[i]
            #recomb_pos_list = []
            #for recomb in column_lists[i]:
                #recomb_pos = all_recombs.index(tuple(recomb))
                #recomb_pos_list.append(recomb_pos)


            tuple_list = [tuple(x) for x in column_lists[i]]

            for j in range(len(all_recombs)):
                if all_recombs[j] not in tuple_list:
                    extended_array = np.c_[extended_array[:, :j], np.zeros(len(extended_array)), extended_array[:, j:]]

            extended_array = extended_array.astype(int)

            pbar.update(1)

            extended_arrays.append(extended_array)
        pbar.close()
        return extended_arrays

