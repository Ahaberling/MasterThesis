# This is mainly an overview and to-do file

'''

v3_textPreproc_LDA_gridSeach.py

    input:
        cleaning_robot_EP_patents.csv

    output:
        patent_topicDist.csv,
        lda_tuning_results.csv

    contains:
        text preprocessing of abstracts,
        gensim lda,
        mallet lda,
        grid search

    todos:
        talk with bwl - really cleaning robots and not just robots?
        -> adjust stopwords
        check vectorization
        maybe vectorize tokenization if too much time (so probably no)
        check bigram position and preprocessing sequence in general
        (Why) Is the result of gensim lda and mallet lda identical?
        grid search - ask/think for validation sets
        grid search - find better hyperparameters
        grid search - fancy up grid search with dm2 knowledge



v1_dataTransformation.py

    input:
        patent_topicDist.csv

    output:
        patent_lda_ipc.csv

    contains:
        unpacks/transforms topic affiliation
        unpacks/transforms ipc affiliation

    todos:
        maybe adjust for mallet lda output



v3_slidingWindows.py

    input:
        patent_lda_ipc.csv

    output:
        pickles like window90by1

    contains:
        creats dictionaries of windowed patent_lda_ipc for longitudinal analyses

    todos:
        -



v3_plainMeasureExploration.py

    input:
        pickles like window90by1

    output:


    contains:
        transforms dictionaries of windows into a window x 'ipc combination of 2' array
        Identifies patterns in this array and subsequently recombination and diffusion
        can impute pattern (e.g. 11111110111111 to 11111111111111

    todos:
        find out how to impute more then just one 0 (e.g. 1111100111111)
        recombination occurs if two topics/ips are combined the first time in x (/the first time in x the number of patents featuring the combination reach a threshold)
        is diffusion merely a  measure, for how long this recombination is popular? or a measure for how long ipcs/topics individually are popular?
          (possible critic: when is a combination of two topics only a combination and not it's own topic already. Kinda arbitrarry)
        or is diffusion something completely different?
        visualization of a sample illustrating diffusion and recombination



intermed_Network_cConstruc.py

    input:


    output:


    contains:


    todos:
        implement weigthing function for one mode projection
        implement first community detection
        implement first link prediction (py torch geo)


'''