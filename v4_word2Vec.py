if __name__ == '__main__':

#--- Import Libraries ---#
    print('\n#--- Import Libraries ---#\n')

    import pandas as pd
    import numpy as np

    #import networkx as nx

    import itertools
    import os

    from gensim.models import Word2Vec



#--- Initialization ---#
    print('\n# --- Initialization ---#\n')

    os.chdir('D:/Universitaet Mannheim/MMDS 7. Semester/Master Thesis/Outline/Data/Cleaning Robots')

    patent_lda_ipc = pd.read_csv('patent_lda_ipc.csv', quotechar='"', skipinitialspace=True)
    #topics = pd.read_csv('patent_topics_mallet.csv', quotechar='"', skipinitialspace=True)
    #parent = pd.read_csv('cleaning_robot_EP_backward_citations.csv', quotechar='"', skipinitialspace=True)

    patent_lda_ipc = patent_lda_ipc.to_numpy()

    print(patent_lda_ipc[:,6:7])

    model = Word2Vec(patent_lda_ipc[:,6:7],
                     vector_size=128,
                     window=5,
                     min_count=1,
                     #workers=4,
                     sg=1,
                     compute_loss=True)

    vector = model.wv['axis']

    print(vector)
    #words = list(model.wv.vocab)
    #print(words)

    print(model.get_latest_training_loss())


