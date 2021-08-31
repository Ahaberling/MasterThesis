import numpy as np
import tqdm
import itertools

class PlainMeasures:

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

                kC_inPatent = [x for x in patent[position] if x != None]  # nan elimination
                kC_inPatent = np.unique(kC_inPatent)

                if unit == 1:
                    kC_list.append(tuple(kC_inPatent))

                else:
                    kC_list.append(list(itertools.combinations(kC_inPatent, r=unit)))

            # dictionary with all singularly occuring ipc's within a window
            kC_list = [item for sublist in kC_list for item in sublist]
            slidingWindow_kC_unite[window_id] = kC_list

            pbar.update(1)

        pbar.close()

        return slidingWindow_kC_unite