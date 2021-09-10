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
    def find_recombination(pattern_array_thresh):

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
    def find_diffusion(pattern_array_thresh, recombinationPos):

        diff_list = []
        pbar = tqdm.tqdm(total=len(recombinationPos))

        for pos in recombinationPos:
            diffusion = -1
            i = 0

            while pattern_array_thresh[pos[0] + i, pos[1]] == 1:
                diffusion = diffusion + 1

                i = i + 1
                if pos[0] + i == len(pattern_array_thresh):
                    break

            diff_list.append(diffusion)

            pbar.update(1)

        pbar.close()

        # print(diffusion_duration_list)          # [0, 0, 0, 13, 0, 0, 20, 0, 0, 0, 13, 0, 0, 20, 41, 7, 0, 89, 89, 152, 5, 229, 90, 0, 6,
        # print(len(diffusion_duration_list))     # 3095

        # Merge both lists to get final data structure #

        for i in range(len(recombinationPos)):
            recombinationPos[i].append(diff_list[i])

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
    def check_dimensions(list_of_allArrays, diffusionArray_Topics_lp_columns):
        for i in range(len(list_of_allArrays)):
            if len(list_of_allArrays[i].T) != len((list(diffusionArray_Topics_lp_columns))):
                raise Exception("Diffusion arrays vary in their columns")
        return

    @staticmethod
    def normalized_and_binarize(list_of_allArrays, threshold, leeway):
        array_rowNorm_list = []
        array_binariz_list = []
        for i in range(len(list_of_allArrays)):

            array_threshold, array_rowNorm = ComparativeMeasures.modify_arrays(list_of_allArrays[i], threshold)
            if leeway == True:
                array_threshold = ComparativeMeasures.introcude_leeway(array_threshold, np.array([1, 0, 1]), 1)
                # if there is 101 this means that there was one year without a occurrence. 1001 is one year and one month ...

            array_rowNorm_list.append(array_rowNorm)
            array_binariz_list.append(array_threshold)
        return array_binariz_list, array_rowNorm_list

    @staticmethod
    def get_nonSimilarity_descriptives(diffusionArray_Topics_lp_columns, diffusion_length_list):
        diff_count_per_topic = []
        diff_duration_per_topic = []

        for topic in range(len(diffusionArray_Topics_lp_columns)):
            diff_count = 0
            diff_duration = []
            for entry in diffusion_length_list:
                if entry[1] == topic:
                    diff_count = diff_count+1
                    diff_duration.append(entry[2])

            diff_count_per_topic.append(diff_count)
            diff_duration_per_topic.append(diff_duration)

        return  diff_count_per_topic, diff_duration_per_topic

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
    def get_matrix_similarities_byTopic(list_of_allArrays_names, list_of_allArrays_threshold):
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
            # print(np.shape(matrixPair))
            patternArray2 = matrixPair[1]

            cosine_list = []
            manhattan_list = []
            for column_id in range(len(patternArray1.T)):

                # print(sum(patternArray1[:,column_id]))
                # print(sum(patternArray2[:,column_id]))
                # when both are sum != 0 -> calculate normally
                # when one is != 0 -> calculate normally (cosine = 0)
                # when both are = 0 -> do not calculate anything
                if not (sum(patternArray1[:, column_id]) == 0 and sum(patternArray2[:, column_id]) == 0):
                    cosine = ComparativeMeasures.cosine_sim_mod(patternArray1[:, column_id],
                                                                patternArray2[:, column_id])
                    manhattan = ComparativeMeasures.manhattan_sim_mod(patternArray1[:, column_id],
                                                                      patternArray2[:, column_id])

                    cosine_list.append(cosine)
                    manhattan_list.append(manhattan)

            if len(cosine_list) != 0:
                cosine_avg = sum(cosine_list) / len(cosine_list)
            else:
                cosine_avg = 0

            if len(manhattan_list) != 0:
                manhattan_avg = sum(manhattan_list) / len(manhattan_list)
            else:
                manhattan_avg = 0

            similarityPair_list_cosine.append(cosine_avg)  # here is one inside that is not working at all
            similarityPair_list_manhattan.append(manhattan_avg)

        return namePair_list, similarityPair_list_cosine, similarityPair_list_manhattan