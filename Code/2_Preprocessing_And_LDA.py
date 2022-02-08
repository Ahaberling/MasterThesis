if __name__ == '__main__':

    #--- Import Libraries ---#
    print('\n#--- Import Libraries ---#\n')

    # Utility
    import os
    import tqdm
    import itertools
    import pickle as pk
    import statistics

    # Data handling
    import numpy as np
    import pandas as pd

    # NLP
    import nltk
    import spacy
    import gensim.corpora as corpora
    import gensim.models as gensim_models

    # Custom functions
    from utilities.Data_Preparation_utils import AbstractCleaning
    from utilities.Data_Preparation_utils import LDA_functions



    #--- Initialization ---#
    print('\n#--- Initialization ---#\n')

    path = 'D:/'
    mallet_path = 'D:/'

    # Import data
    os.chdir(path)

    with open('patents_english_cleaned', 'rb') as handle:
        patents_english_cleaned = pk.load(handle)

    # Nlp misc
    nltk.download('punkt')  # nltk tokenizer
    nltk.download('stopwords')  # nltk stopwords filter

    # Specify model of choice
    mode = 'mallet'            # takes ['mallet', 'gensim', 'mallet_gridSearch', 'gensim_gridSearch']



    #--- Prepare stop word filters ---#
    print('\n#--- Prepare stop word filters ---#\n')

    # Use the nltk stop word list as basis, but exclude following words (conservative 
    # approach - they might carry relevant information for LDA)
    nltk_filter = nltk.corpus.stopwords.words('english')
    for word in ['above', 'below', 'up', 'down', 'over', 'under', 'won']:
        nltk_filter.remove(word)
 

    # Custom filters:
    numbers_filter = [
                      'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven',
                      'twelve', 'first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth',
                      'tenth', 'eleventh', 'twelfth', 'i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x', 'xi',
                      'xii']

    # High confidence, that these words can and should be filtered out
    highConfidence_filter = [
                 'also', 'therefor', 'thereto', 'additionally', 'thereof', 'minimum', 'maximum', 'multiple',
                 'thereon', 'pre', 'kind', 'extra', 'double', 'manner', 'general',
                 'previously', 'exist', 'respective', 'end', 'central', 'indirectly', 'expect', 'include', 'main',
                 'relate', 'type', 'couple', 'plurality', 'common', 'properly', 'entire', 'possible', 'multi', 'would',
                 'could', 'good', 'done', 'many', 'much', 'rather', 'right', 'even', 'may', 'some', 'preferably']

    # Medium confidence, that these words can and should be filtered out
    mediumConfidence_filter = [
                 'input', 'output', 'base', 'basic', 'directly', 'time', 'item', 'work', 'number', 'information',
                 'make', 'set', 'sequentially', 'subject', 'object', 'define', 'reference', 'give', 'feature',
                 'determine', 'workpiece', 'action', 'mode', 'function', 'relative', 'reference', 'application', 'describe',
                 'invention', 'represent', 'task', 'form', 'approach', 'independent', 'independently', 'advance', 'becomes',
                 'preform', 'parallel', 'get', 'try', 'easily', 'use', 'know', 'think', 'want', 'seem', 'robot',
                 'robotic', 'robotically', 'robotize']

    # Low confidence, that these words can and should be filtered out
    lowConfidence_filter = [
                 'machine', 'method', 'model', 'part', 'system', 'variable', 'parameter', 'structure', 'device',
                 'state', 'outer', 'device', 'present', 'remain', 'program' 'angle', 'angular', 'vertical', 'longitudinal',
                 'axis', 'position', 'operative', 'operatively', 'prepare', 'operable', 'move', 'receive', 'adapt', 'configure',
                 'movable', 'create', 'separate', 'design', 'identification', 'identify', 'joint', 'qf', 'zmp', 'llld', 'ik']



    # --- Abstract Cleaning ---#
    print('\n#--- Abstract Cleaning ---#\n')

    # Investigate smallest abstracts for suitability of the analysis
    smallestAbstracts = AbstractCleaning.get_smallest_abstracts(patents_english_cleaned[:, 6], min_abstract_size=20)

    print('Number of patents with 20 words or less: ', len(smallestAbstracts))
    print('Abstracts at question: \n')
    # all are kept (conservative approach)
    for i in smallestAbstracts:
        print(i)


    # Descriptives: report vocabulary size before preprosessing
    tokens_prePreprocessing = [nltk.word_tokenize(abstract) for abstract in patents_english_cleaned[:, 6]]
    tokens_prePreprocessing_list = []
    for i in tokens_prePreprocessing:
        for token in i:
            tokens_prePreprocessing_list.append(token)

    print('\nVocabulary size before preprocessing: ', len(np.unique(tokens_prePreprocessing_list)), '\n')


    # Remove non-alphabetic characters and single character terms; make all terms lower case
    abs_intermed_preproc = AbstractCleaning.vectorize_preprocessing(patents_english_cleaned[:, 6])


    # Report descriptives after preprocessing step 1:
    helper = []
    helper_unique = []
    abstracts_intermed = [nltk.word_tokenize(abstract) for abstract in abs_intermed_preproc]
    token_list = []

    for i in abstracts_intermed:
        for token in i:
            token_list.append(token)
        helper.append(len(i))
        i_unique = np.unique(i)
        helper_unique.append(len(np.unique(i)))

    print('Number of all tokens after removing non-alphabetic characters, single letter terms, and lowercasing: ', len(token_list))
    print('Vocabulary size after " : ', len(np.unique(token_list)), '\n')

    print('Average Number of tokens per abstract after step 1: ', sum(helper) / len(abstracts_intermed))
    print('Median Number of tokens per abstract after step 1: ', np.median(helper))
    print('Mode Number of tokens per abstract after step 1: ', statistics.mode(helper))
    print('Max Number of tokens per abstract after step 1: ', max(helper))
    print('Min Number of tokens per abstract after step 1: ', min(helper), '\n')

    print('Average Number of unique tokens per abstract after step 1: ', sum(helper_unique) / len(abstracts_intermed))
    print('Median Number of unique tokens per abstract after step 1: ', np.median(helper_unique))
    print('Mode Number of unique tokens per abstract after step 1: ', statistics.mode(helper_unique))
    print('Max Number of unique tokens per abstract after step 1: ', max(helper_unique))
    print('Min Number of unique tokens per abstract after step 1: ', min(helper_unique), '\n')


    # Apply tokenization
    abst_tokenized = [nltk.word_tokenize(abstract) for abstract in abs_intermed_preproc]


    # Apply term filters
    filter = list(itertools.chain(nltk_filter, numbers_filter, highConfidence_filter, mediumConfidence_filter))
    abst_nostops = [AbstractCleaning.remove_stopwords(abstract, filter) for abstract in abst_tokenized]
    print('Number of words in stop word filter: ', len(filter), '\n')


    # Descriptives: vocabulary size after first filter
    token_list = []
    for i in abst_nostops:
        for token in i:
            token_list.append(token)
    print('Vocabulary size after first stop word filter : ', len(np.unique(token_list)), '\n')


    # Build bigrams
    bigram = gensim_models.Phrases(abst_nostops, min_count=10, threshold=100)  # higher threshold fewer phrases.
    bigram_mod = gensim_models.phrases.Phraser(bigram)
    abst_bigrams = AbstractCleaning.make_bigrams(abst_nostops, bigram_mod)

    bigram_list = AbstractCleaning.count_bigrams(abst_bigrams)
    print('Exemplary bigrams mentioned in thesis: ', bigram_list[5], bigram_list[16])
    print('Number of unique bigrams in whole dictionary: ', len(bigram_list), '\n')


    # Apply lemmatization
    spacy_en = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
    abst_lemmatized = AbstractCleaning.lemmatization(abst_bigrams, spacy_en, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])


    # Descriptives: vocabulary size after lemmatization
    token_list = []
    for i in abst_lemmatized:
        for token in i:
            token_list.append(token)
    print('Vocabulary size after lemmatization: ', len(np.unique(token_list)), '\n')  # keep this


    # Apply term filters again on lemmatized abstracts
    abst_clean = [AbstractCleaning.remove_stopwords(abstract, filter) for abstract in abst_lemmatized]


    # Descriptives: vocabulary size after all preprocessing
    token_list = []
    for i in abst_clean:
        for token in i:
            token_list.append(token)
    print('Number of all tokens after all preprocessing: ', len(token_list))
    print('Vocabulary size after all preprocessing : ', len(np.unique(token_list)), '\n')


    # Descriptives after preprocessing
    numberOfWords_abstract_unique, numberOfWords_abstract = AbstractCleaning.number_of_word_post_preprocessing(abst_clean)
    print('Average number of tokens per abstract after preprocessing: ', sum(numberOfWords_abstract)/len(numberOfWords_abstract))
    print('Median number of tokens per abstract after preprocessing: ', np.median(numberOfWords_abstract))
    print('Mode number of tokens per abstract after preprocessing: ', statistics.mode(numberOfWords_abstract))
    print('Max number of tokens per abstract after preprocessing: ', max(numberOfWords_abstract))
    print('Min number of tokens per abstract after preprocessing: ', min(numberOfWords_abstract), '\n')

    print('Average number of unique tokens per abstract after preprocessing: ', sum(numberOfWords_abstract_unique)/len(numberOfWords_abstract_unique))
    print('Median number of unique tokens per abstract after preprocessing: ', np.median(numberOfWords_abstract_unique))
    print('Mode number of unique tokens per abstract after preprocessing: ', statistics.mode(numberOfWords_abstract_unique))
    print('Max number of unique tokens per abstract after preprocessing: ', max(numberOfWords_abstract_unique))
    print('Min number of unique tokens per abstract after preprocessing: ', min(numberOfWords_abstract_unique), '\n')



    # --- Preparing LDA ---#
    print('\n#--- Preparing LDA ---#\n')

    # Prepare dataframe with column for topic distributions (LDA result)
    patent_topicDistribution = np.empty((np.shape(patents_english_cleaned)[0], np.shape(patents_english_cleaned)[1] + 1), dtype=object)
    patent_topicDistribution[:, :-1] = patents_english_cleaned

    # Create Dictionary
    id2word = corpora.Dictionary(abst_clean)

    # Create Corpus
    corpus = [id2word.doc2bow(text) for text in abst_clean]



    # --- Compute LDA ---#
    print('\n#--- Compute LDA ---#\n')

    # if :
    # FileNotFoundError: [Errno 2] No such file or directory: '... _state.mallet.gz'

    # then:
    # updated smart-open to 5.1.0


    if mode == 'gensim':

        # src = inspect.getsource(gensim_models.LdaModel)
        '''    def __init__(self, corpus=None, num_topics=100, id2word=None,
                     distributed=False, chunksize=2000, passes=1, update_every=1,
                     alpha='symmetric', eta=None, decay=0.5, offset=1.0, eval_every=10,
                     iterations=50, gamma_threshold=0.001, minimum_probability=0.01,
                     random_state=None, ns_conf=None, minimum_phi_value=0.01,
                     per_word_topics=False, callbacks=None, dtype=np.float32):'''

        # src = inspect.getsource(gensim_models.LdaMulticore)
        '''    def __init__(self, corpus=None, num_topics=100, id2word=None, workers=None,
                     chunksize=2000, passes=1, batch=False, alpha='symmetric',
                     eta=None, decay=0.5, offset=1.0, eval_every=10, iterations=50,
                     gamma_threshold=0.001, random_state=None, minimum_probability=0.01,
                     minimum_phi_value=0.01, per_word_topics=False, dtype=np.float32):'''

        coherence_gensim, topicDistribution_gensim, topics_gensim = LDA_functions.gensim_LDA(corpus=corpus,
                                                                                             id2word=id2word,
                                                                                             num_topics=330,
                                                                                             random_state=123,
                                                                                             alpha=0.15,
                                                                                             beta=0.01,
                                                                                             per_word_topics=True,
                                                                                             abst_clean=abst_clean,
                                                                                             coherence='c_v',
                                                                                             multicore=False,
                                                                                             onlyCoherency=False)
        print('Coherency C_V of Gensim LDA: ', coherence_gensim)

        patent_topicDistribution_gensim = patent_topicDistribution
        patent_topicDistribution_gensim.T[8, :] = topicDistribution_gensim

        print('First patent and its topic affiliation and coverage: ')
        print(patent_topicDistribution_gensim[0], '\n')
        print('First topic: ')
        print(topics_gensim[0])

        pd.DataFrame(patent_topicDistribution_gensim).to_csv('patent_topicDistribution_gensim.csv', index=None)
        pd.DataFrame(topics_gensim).to_csv('patent_topics_gensim.csv', index=None)


    if mode == 'mallet':

        # src = inspect.getsource(gensim_models.wrappers.LdaMallet)
        '''    def __init__(self, mallet_path, corpus=None, num_topics=100, alpha=50, id2word=None, workers=4, prefix=None,
                     optimize_interval=0, iterations=1000, topic_threshold=0.0, random_seed=0):'''

        coherence_mallet, topicDistribution_mallet, topics_mallet = LDA_functions.mallet_LDA(mallet_path=mallet_path,
                                                                                             corpus=corpus,
                                                                                             id2word=id2word,
                                                                                             num_topics=330,
                                                                                             random_seed=123,
                                                                                             alpha=49,
                                                                                             #beta=0.01,
                                                                                             iterations=1000,
                                                                                             optimize_interval=1000,
                                                                                             abst_clean=abst_clean,
                                                                                             coherence='c_v',
                                                                                             onlyCoherency=False,
                                                                                             )

        print('Coherency C_V of Mallet LDA: ', coherence_mallet)

        patent_topicDistribution_mallet = patent_topicDistribution

        c = 0
        for i in topicDistribution_mallet:
            patent_topicDistribution_mallet.T[8, c] = list(i)
            c = c + 1

        print('First patent and its topic affiliation and coverage: ')
        print(patent_topicDistribution_mallet[0], '\n')
        print('First topic: ')
        print(topics_mallet[0])

        pd.DataFrame(patent_topicDistribution_mallet).to_csv('patent_topicDistribution_mallet.csv', index=None)
        pd.DataFrame(topics_mallet).to_csv('patent_topics_mallet.csv', index=None)


    if mode == 'mallet_gridSearch' or mode == 'gensim_gridSearch':

        # --- Grid search ---#
        print('\n#--- Grid search ---#\n')


        # specify topics range
        min_topics = 50
        max_topics = 550
        step_size = 50
        topics_range = range(min_topics, max_topics, step_size)
        topics_range_fixed = [330]

        # specify alpha range
        # small alpha = few, but fitting topics per documents
        # big alpha = more, but less fitting topics per document
        # Keep in mind: Gensim handles the 'alpha' parameter as alpha
        #               Mallet handles the 'alpha' parameter as value that is used to compute the real alpha ('alpha'/'num_topics'=alpha)

        alpha_gensim = list(np.arange(0.05, 0.55, 0.05))
        alpha_gensim.append('symmetric')
        alpha_gensim.append('asymmetric')
        # Keep in mind: 'auto' is not compatible with Multicore LDA
        alpha_gensim.append('auto')
        alpha_gensim_fixed = [0.15]

        alpha_mallet = [16, 33, 49, 66, 82, 99, 115, 132, 148, 165]  # this resamples the gensim scale with 330 topics
        alpha_mallet_fixed = [49]  # this resamples alpha = 0.15 with 330 topics

        # specify optimize_interval (mallet)
        optimize_interval = list(np.arange(0, 2100, 100))
        optimize_interval_fixed = [1000]  # same as default 0

        # specify trainings iterations (mallet)
        iterations = list(np.arange(1000, 11000, 1000))
        iterations_fixed = [1000]

        if mode == 'gensim_gridSearch':

            # prepare result file to be saved
            model_results = {'Topics': [],
                             'Alpha': [],
                             'Coherence': []
                             }

            pbar = tqdm.tqdm(total=len(alpha_gensim))  # adjust with hyperparameter change

            for k in topics_range:
                for a in alpha_gensim:
                    coherency_score = LDA_functions.gensim_LDA(corpus=corpus,
                                                               id2word=id2word,
                                                               num_topics=k,
                                                               random_state=123,
                                                               alpha=a,
                                                               beta=0.01,
                                                               abst_clean=abst_clean,
                                                               coherence='c_v',
                                                               onlyCoherency=True,
                                                               per_word_topics=True,
                                                               multicore=False
                                                               )

                    # Save the model results
                    model_results['Topics'].append(k)
                    model_results['Alpha'].append(a)
                    model_results['Coherence'].append(coherency_score)
                    pbar.update(1)

            pd.DataFrame(model_results).to_csv('gensim_gridSearch.csv', index=False)
            pbar.close()


        if mode == 'mallet_gridSearch':

            # prepare result file to be saved
            model_results = {'Topics': [],
                             'Alpha': [],
                             'optimize_interval': [],
                             'iterations': [],
                             'Coherence': []
                             }

            pbar = tqdm.tqdm(total=len(iterations))  # adjust with hyperparameter change 
            
            for k in topics_range:
                for a in alpha_mallet:
                    for op in optimize_interval:
                        for it in iterations:
                            coherency_score = LDA_functions.mallet_LDA(mallet_path=mallet_path,
                                                                       corpus=corpus,
                                                                       id2word=id2word,
                                                                       num_topics=k,
                                                                       random_seed=123,
                                                                       alpha=a,
                                                                       # beta=0.01,
                                                                       iterations=it,
                                                                       optimize_interval=op,
                                                                       abst_clean=abst_clean,
                                                                       coherence='c_v',
                                                                       onlyCoherency=True,
                                                                       )

                            # Save the model results
                            model_results['Topics'].append(k)
                            model_results['Alpha'].append(a)
                            model_results['optimize_interval'].append(op)
                            model_results['iterations'].append(it)
                            model_results['Coherence'].append(coherency_score)

                            pbar.update(1)

            pd.DataFrame(model_results).to_csv('mallet_gridSearch.csv', index=False)
            pbar.close()
