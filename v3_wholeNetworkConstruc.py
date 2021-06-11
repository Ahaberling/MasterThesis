if __name__ == '__main__':

#--- Import Libraries ---#
    print('\n#--- Import Libraries ---#\n')

    import pandas as pd
    import numpy as np

    import networkx as nx

    import os

#--- Initialization ---#
    print('\n# --- Initialization ---#\n')

    os.chdir('D:/Universitaet Mannheim/MMDS 7. Semester/Master Thesis/Outline/Data/Cleaning Robots')

    patent_lda_ipc = pd.read_csv('patent_lda_ipc.csv', quotechar='"', skipinitialspace=True)
    topics = pd.read_csv('patent_topics_mallet.csv', quotechar='"', skipinitialspace=True)
    parent = pd.read_csv('cleaning_robot_EP_backward_citations.csv', quotechar='"', skipinitialspace=True)

    patent_lda_ipc = patent_lda_ipc.to_numpy()
    topics = topics.to_numpy()
    parent = parent.to_numpy()



#--- Plain citation network - Preparing data structure for networkx ---#
    print('\n#--- Plain citation network - Preparing data structure for networkx ---#\n')


    ### Plain citation nodes ###

    node_plain = patent_lda_ipc[:,0]
    print(len(node_plain))


    ### Plain citation edges ###

    citation_source = parent[:,0]                   # pat_publn_id
    citation_target = parent[:,2]                   # cited_pat_publn_id

    #print(citation_source)                          # [468378953 285297843 336208959 ...]
    #print(len(citation_source))                     # 18548


    ### plain citation node attributes ###

    node_plain_att = ['publn_auth', 'publn_nr', 'publn_date', 'publn_claims', 'publn_title',
                      'publn_abstract', 'nb_IPC', 'abstract_clean', 'topic_list']           # pat_publn_id ommited, since it's the node id. No need to add it as attribute as well.

    #print(patent_lda_ipc[1,:])
    #print(patent_lda_ipc[1,9:30])                                      # topic columns
    #print(patent_lda_ipc[1,30:len(patent_lda_ipc)])                    # ipc columns


    number_of_topics = int(len(patent_lda_ipc[0,9:30])/3)
    #print(loop_helper)

    for i in range(1,number_of_topics+1):
        node_plain_att.append('TopicID_{0}'.format(i))
        node_plain_att.append('TopicName_{0}'.format(i))
        node_plain_att.append('TopicCover_{0}'.format(i))


    number_of_ipcs = int(len(patent_lda_ipc[0, 30:len(patent_lda_ipc)]) / 3)
    # print(loop_helper)

    for i in range(1, number_of_ipcs + 1):
        node_plain_att.append('IpcID_{0}'.format(i))
        node_plain_att.append('IpcSubCat1_{0}'.format(i))
        node_plain_att.append('IpcSubCat2_{0}'.format(i))

    #print(node_plain_att)                   # ['publn_auth', ... 'TopicID_1', ... 'IpcID_1', ... ]

    #print(len(node_plain_att))
    #print(len(patent_lda_ipc[0,:]))


    inner_dic = []
    for i in range(len(patent_topicDist_prep)):

        helper = dict(enumerate(patent_topicDist_prep[i]))
        for key, n_key in zip(helper.copy().keys(), inner_keys):
            helper[n_key] = helper.pop(key)

        inner_dic.append(helper)

    outer_dic = dict(enumerate(inner_dic))

    # 487838990: {0: 'EP', 1: 3275601.0, 2: ...}