import numpy as np
import tqdm
import itertools

class ReferenceMeasures:

    @staticmethod
    def extract_knowledgeComponent_per_window(slidingWindow_dict, kC, unit):

        if kC == 'topic':
            position = np.r_[range(9, 23, 2)]

        elif kC == 'ipc':
            position = np.r_[range(23, np.shape(slidingWindow_dict['window_0'])[1], 3)]

        else:
            raise Exception("kC must be string value 'topic' or 'ipc'")

        slidingWindow_kC_unite = {}


        pbar = tqdm.tqdm(total=len(slidingWindow_dict))

        for window_id, window in slidingWindow_dict.items():

            kC_list = []

            for patent in window:

                if kC == 'topic':
                    kC_inPatent = [int(x) for x in patent[position] if x != None]  # nan elimination
                elif kC == 'ipc':
                    kC_inPatent = [x for x in patent[position] if x != None]
                else:
                    raise Exception("kC must be string value 'topic' or 'ipc'")

                kC_inPatent = np.unique(kC_inPatent)

                if unit == 1:
                    kC_list.append(tuple(kC_inPatent))

                else:
                    kC_list.append(list(itertools.combinations(kC_inPatent, r=unit)))

            # dictionary with all singularly occuring ipc's within a window
            kC_list = [item for sublist in kC_list for item in sublist]
            #print(kC_list)
            slidingWindow_kC_unite[window_id] = kC_list

            pbar.update(1)

        pbar.close()

        return slidingWindow_kC_unite

    @staticmethod
    def create_pattern_array(knowledgeComponent_dict):

        row_list = []
        column_list = []

        for window_id, window in knowledgeComponent_dict.items():
            row_list.append(window_id)
            column_list.append(window)

        column_list = [item for sublist in column_list for item in sublist]
        column_list, column_list_counts = np.unique(column_list, return_counts=True, axis=0)

        print(type(column_list[0]))
        if np.issubdtype(type(column_list[0]), np.integer) or np.issubdtype(type(column_list[0]), np.str_):
            column_list.sort()

        else:
            ind = np.lexsort((column_list[:, 1], column_list[:, 0]))    # if 'unite' exceeds tuples, it is not sorted once more here
                                                                        # However this sort is redundant non the lest in the current version
            column_list = column_list[ind]

        pattern_array = np.zeros((len(row_list), len(column_list)))

        pbar = tqdm.tqdm(total=len(row_list))
        c_row = 0

        for row in row_list:
            c_column = 0

            for column in column_list:
                #print(tuple(column))
                #print(knowledgeComponent_dict[row])
                if tuple(column) in knowledgeComponent_dict[row]:
                    pattern_array[c_row, c_column] = list(knowledgeComponent_dict[row]).count(tuple(column))

                c_column = c_column + 1
            c_row = c_row + 1
            pbar.update(1)

        pbar.close()

        return pattern_array, column_list