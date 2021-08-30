
if __name__ == '__main__':

#--- Import Libraries ---#
    print('\n#--- Import Libraries ---#\n')

    import os
    import sys

    import numpy as np
    import pandas as pd



#--- Initialization ---#
    print('\n#--- Initialization ---#\n')

    os.chdir('D:/Universitaet Mannheim/MMDS 7. Semester/Master Thesis/Outline/Data/Cleaning Robots')

    patent_topicDist = pd.read_csv('patent_topicDist_mallet.csv', quotechar='"', skipinitialspace=True)
    patent_topicDist = patent_topicDist.to_numpy()

    patent_IPC = pd.read_csv('cleaning_robot_EP_patents_IPC.csv', quotechar='"', skipinitialspace=True)
    patent_IPC = patent_IPC.to_numpy()



#--- Transforming topic representation ---#
    print('\n#--- Transforming topic representation ---#\n')

    from utilities.my_transform_utils import Transf_misc

    topic_list_helper, max_topics = Transf_misc.max_number_topics(patent_topicDist)
    print('Maximum number of topics a patent has: ', max_topics)               # 7 is the maximum of topics abstracts have
    print('Hence, number of new columns needed: ',   max_topics*2)             # space needed = 21 (7 topic_ids + 7 topic_names + 7 topic_coverages)

    ### Prepare dataframe with columns for topics (transformation result) ###
    patent_transf = np.empty((np.shape(patent_topicDist)[0], np.shape(patent_topicDist)[1] + int(max_topics*2)), dtype=object)
    patent_transf[:, :-int(max_topics*2)] = patent_topicDist

    ### Filling the new array ###
    patent_transf = Transf_misc.fill_with_topics(patent_transf, topic_list_helper, np.shape(patent_topicDist)[1])


#--- Check transformation ---#
    print('\n#--- Check transformation ---#\n')

    #print('Shape of new array: ', np.shape(patent_transf))                                  # (3781, 30)
    #print('Are all new columns used? Number of patents with maximum number of topics: ', sum(x is not None for x in patent_transf[:,np.shape(patent_transf)[1]-1]))        # 1
    if sum(x is not None for x in patent_transf[:,np.shape(patent_transf)[1]-1]) == 0:
        raise Exception("Error: Not all created columns in patent_transf have been filled")


#--- Append IPC ---#
    print('\n#--- Append IPC ---#\n')

    # Each Patent has at least one IPC. These IPC'S are appended to the patent_transf array in order to facilitate
    # a brief, heuristic evaluation of topic modeling. Additionally they might be used to conceptualize first basic
    # recombination and diffusion measures


    ### Review patent_transf and patent_IPC ###
    print('Review patent_transf and patent_IPC:\n')

    val, count = np.unique(patent_transf[:, 0], return_counts=True)
    print('All patent ids in patent_topicDist_x are unique if: ', len(val), ' == ', len(patent_transf))


    val, count = np.unique(patent_IPC[:, 0], return_counts=True)
    print('Patent_IPC contains: ', len(val), 'unique id\'s. This is more then patent_topicDist_x, since the later was cleaned of german and fransh patents')
    print('Patent_IPC contains: ', len(patent_IPC), ' rows overall. Patents can be categorized with more than one IPC')


    ### Clean patent_IPC (remove patents with german and france abstracts) ###

    patent_IPC_clean = [i[0] for i in patent_IPC if i[0] in patent_transf[:, 0]]

    #print(len(np.unique(patent_IPC[:,1])))        #970

    val, count = np.unique(patent_IPC_clean, return_counts=True)
    # print(len(val))                             # 3781
    # print(len(patent_IPC_clean))                # 9449
    # print(max(count))                           # 13 -> patents have at most 13 IPC's

    # IPC's consist of 3 colums (IPC, subcategory, subcategory)
    # 13 * 3 = 39 additional columns are needed in the new array

    new_space_needed = max(count) * 3  # 39

    # print(np.argmax(count))                     #      3485 index of highest value in count
    # print(val[np.argmax(count)])                # 478449443 id of with highest value in count
    # print(patent_IPC[patent_IPC[:,0] == val[np.argmax(count)]])


    ### New array, including space for IPC's ###

    patent_join = np.empty((np.shape(patent_transf)[0], np.shape(patent_transf)[1] + new_space_needed), dtype=object)
    patent_join[:, :-new_space_needed] = patent_transf

    # print(np.shape(patent_transf))            # (3781, 52)
    # print(np.shape(patent_join))              # (3781, 91)

    ### Fill new array ###



    patent_join = Transf_misc.fill_with_IPC(patent_join, patent_IPC, new_space_needed)

    ### check if all created columns are used ###

    print('Are all new columns used? Number of patents with maximum number of IPC\'S: ',
          sum(x is not None for x in patent_join[:, np.shape(patent_join)[1] - 1]))

    # print(sum(x is not None for x in patent_join[:,90]))       they are
    # print(patent_join[patent_join[:,0] == 478449443])
    # print(patent_join[patent_join[:,0] == val[np.argmax(count)]])

    np.set_printoptions(threshold=sys.maxsize)

    #print(patent_join[100])

#--- Save transformation and IPC appendix---#
    print('\n#--- Save transformation and IPC appendix---#\n')

    #print('Preview of the resulting Array:\n\n', patent_join[0])                        #  [12568 'EP' 1946896.0 '2008-07-23' 15
                                                                                        # 'Method for adjusting at least one axle'
                                                                                        # 'An adjustment method for at least one axis (10) in which a robot has a control unit (12) for controlling an axis (10) via which at least two component parts (16,18) are mutually movable. The component parts (16,18) each has at least one marker (24,26) and the positions of the markers are detected by a sensor, with the actual value of a characteristic value ascertained as a relative position of the two mutually movable components (16,18). The adjustment position is repeated by comparing an actual value with a stored, desired value for the adjustment position. An independent claim is included for a device with signal processing unit.'
                                                                                        #  1 '[(139, 0.05602006688963211)]' '139' None '0.05602006688963211' None
                                                                                        #  None None None None None None None None None None None None None None
                                                                                        #  None None None 'B25J   9' 'Handling' 'Mechanical engineering' None None
                                                                                        #  None None None None None None None None None None None None None None
                                                                                        #  None None None None None None None None None None None None None None
                                                                                        #  None None None None None None]

    pd.DataFrame(patent_join).to_csv('patent_lda_ipc.csv', index=None)


