# Check v3 when doing documentation

if __name__ == '__main__':

    # --- Import Libraries ---#
    print('\n#--- Import Libraries ---#\n')

    import os
    import tqdm
    import itertools

    import numpy as np
    import pandas as pd

    import nltk
    import spacy
    import re
    import gensim.corpora as corpora
    import gensim.models as models



    # --- Initialization ---#
    print('\n#--- Initialization ---#\n')

    final_model_gensim = False
    final_model_mallet = False
    grid_search = False

    os.chdir('D:/Universitaet Mannheim/MMDS 7. Semester/Master Thesis/Outline/Data/Cleaning Robots')

    patent_raw = pd.read_csv('cleaning_robot_EP_patents.csv', quotechar='"', skipinitialspace=True)
    patent_raw = patent_raw.to_numpy()

    nltk.download('punkt')  # nltk tokenizer
    nltk.download('stopwords')  # nltk stopwords filter



    # --- Patent Cleaning ---#
    print('\n#--- Patent Cleaning ---#\n')

    from utilities.my_text_utils import PatentCleaning

    print('Number of all patents: ', len(patent_raw))
    patent_raw, number_removed_patents = PatentCleaning.remove_foreign_patents(patent_raw, language='ger', count=True)
    print('Number of german patents removed: ', number_removed_patents)
    patent_raw, number_removed_patents = PatentCleaning.remove_foreign_patents(patent_raw, language='fr', count=True)
    print('Number of french patents removed: ', number_removed_patents)

    term, number_abstracts_term = PatentCleaning.count_abstracts_with_word(patent_raw, word='clean')
    print('Number abstracts containing', term, ': ', number_abstracts_term)
    term, number_abstracts_term = PatentCleaning.count_abstracts_with_word(patent_raw, word='robot')
    print('Number abstracts containing', term, ': ', number_abstracts_term)


    ### New Array, including space for preprocessed abstracts ###

    patent_wTopics = np.empty((np.shape(patent_raw)[0], np.shape(patent_raw)[1] + 1), dtype=object)
    patent_wTopics[:, :-1] = patent_raw


    # --- Abstract Cleaning ---#

    from utilities.my_text_utils import AbstractCleaning

    abs_intermed_preproc = AbstractCleaning.vectorize_preprocessing(patent_raw[:, 6])

    ### Apply tokenization ###

    abst_tokenized = [nltk.word_tokenize(abstract) for abstract in abs_intermed_preproc]

    ### Define Stopwords ###
    # hc/mc/lc/vlc = high/middle/low/very low confidence
    # when using extended dataset, check if 'robot' is also present in every patent abstract
    # adapt if it is clear that the dataset only includes real robot things and not also pure robot programs

    nltk_filter = nltk.corpus.stopwords.words('english')
    for word in ['above', 'below', 'up', 'down', 'over', 'under', 'won']:
        nltk_filter.remove(word)

    numbers_filter = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven',
                      'twelve', 'first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth',
                      'tenth', 'eleventh', 'twelfth', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x', 'xi',
                      'xii',
                      ]  # 'i' already listed in nltk_stopWords

    hc_filter = ['also', 'therefor', 'thereto', 'additionally', 'thereof', 'minimum', 'maximum', 'multiple',
                 'pre', 'kind', 'extra', 'double', 'manner', 'general',
                 'previously', 'exist', 'respective', 'end', 'central', 'indirectly', 'expect', 'include', 'main',
                 'relate',
                 'type', 'couple', 'plurality', 'common', 'properly', 'entire', 'possible', 'multi', 'would',
                 'could', 'good', 'done', 'many', 'much', 'rather', 'right', 'even', 'may', 'some', 'preferably']

    mc_filter = ['input', 'output', 'base', 'basic', 'directly', 'time', 'item', 'work', 'number', 'information',
                 'make', 'set', 'sequentially', 'subject', 'object', 'define', 'reference', 'give', 'feature',
                 'determine',
                 'workpiece', 'action', 'mode', 'function', 'relative', 'reference', 'application', 'describe',
                 'invention',
                 'represent', 'task', 'form', 'approach', 'independent', 'independently', 'advance', 'becomes',
                 'preform', 'parallel', 'get', 'try', 'easily', 'use', 'know', 'think', 'want', 'seem', 'robot',
                 'robotic',
                 'robotically', 'robotize']

    lc_filter = ['machine', 'method', 'model', 'part', 'system', 'variable', 'parameter', 'structure', 'device',
                 'state',
                 'outer', 'device', 'present', 'remain', 'program']

    vlc_filter = ['angle', 'angular', 'vertical', 'longitudinal', 'axis', 'position', 'operative', 'operatively',
                  'prepare',
                  'operable', 'move', 'receive', 'adapt', 'configure', 'movable', 'create', 'separate', 'design',
                  'identification', 'identify', 'joint', 'qf', 'zmp', 'llld', 'ik', ]

    filter = list(itertools.chain(nltk_filter, numbers_filter, hc_filter, mc_filter))


    # why is robotic still in there??

    ### Define functions for stopwords, bigrams, trigrams and lemmatization ###

    def remove_stopwords(texts):
        clean_text = [word for word in texts if word not in filter]
        return clean_text


    def make_bigrams(texts):
        return [bigram_mod[doc] for doc in texts]


    # def make_trigrams(texts):
    # return [trigram_mod[bigram_mod[doc]] for doc in texts]

    def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        texts_out = []
        for sent in texts:
            doc = nlp(" ".join(sent))
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        return texts_out


    ### Apply functions ###
    abst_nostops = [remove_stopwords(abstract) for abstract in abst_tokenized]

    ###  Build bigram and trigram models ###

    bigram = models.Phrases(abst_nostops, min_count=5, threshold=100)  # higher threshold fewer phrases.
    # trigram = models.Phrases(bigram[tokenized_abst], threshold=100)

    bigram_mod = models.phrases.Phraser(bigram)
    # trigram_mod = models.phrases.Phraser(trigram)

    # print(trigram_mod[bigram_mod[tokenized_abst[0]]])                          # Accessing grams

    # todo: fine tune models.Phrases

    # Form Bigrams
    abst_bigrams = make_bigrams(abst_nostops)

    # Form Trigrams
    # data_words_trigrams = make_trigrams(data_words_bigrams)

    # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
    nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])

    # Do lemmatization keeping only noun, adj, vb, adv
    abst_lemmatized = lemmatization(abst_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

    # Lemma cleaning
    abst_clean = [remove_stopwords(abstract) for abstract in abst_lemmatized]

    # Remove Stop Words again
    # data_lemmatized = remove_stopwords(data_lemmatized)

    # todo check if bigrams (and everything else) at right place. Do we want 'an_independent_claim' and '
    # also_included' to be tokens? (Assuming they are not cleaned by lemmatization)

    # --- Building LDA  ---#
    print('\n#--- Building LDA ---#\n')

    ### Create Dictionary ###

    id2word = corpora.Dictionary(abst_clean)

    ### Create Corpus ###

    corpus = [id2word.doc2bow(text) for text in abst_clean]

    ### Build LDA model - Gensim ###
    if final_model_gensim == True:
        # lda_gensim = models.LdaMulticore(corpus=corpus,
        lda_gensim = models.LdaModel(corpus=corpus,
                                     id2word=id2word,
                                     num_topics=325,
                                     # adjust num_topics, alpha and eta with regard to grid search results
                                     random_state=123,
                                     # chunksize=100,
                                     passes=10,
                                     alpha='auto',
                                     eta='auto',
                                     per_word_topics=True)

        # plain                 passes=10           passes=20               passes=10 + no topics
        # -8.62154494606457     -7.233880798919008  -6.9446708245485835     -7.931440389780358
        # 0.2763968453915299    0.30734818195201263 0.3139476238900447      0.3386636130532321

        #                       alpha,eta='auto'    alpha,eta='auto'
        #                       -11.273654236990879 -11.137248612925143
        #                       0.3386636130532321  0.34062441108851377

        ### Compute Perplexity - Gensim ###

        print('Perplexity of final LDA (Gensim): ', lda_gensim.log_perplexity(corpus))  # -11.672663740370565
        # The lower the better, but heavily limited and not really useful.

        ### Compute Coherence Score - Gensim ###

        coherence_model_lda = models.CoherenceModel(model=lda_gensim, texts=abst_clean, dictionary=id2word,
                                                    coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        print('Coherence Score (c_v) of final LDA (Gensim): ', coherence_lda)  # 0.3352192192227267
        # the higher the better from 0.3 to 0.7 or 0.8.
        # source: https://stackoverflow.com/questions/54762690/what-is-the-meaning-of-coherence-score-0-4-is-it-good-or-bad

        ### Save Document-Topic affiliation - Gensim ###

        doc_affili_gensim = lda_gensim.get_document_topics(corpus, minimum_probability=0.05, minimum_phi_value=None,
                                                           per_word_topics=False)

        patent_topicDist_gensim = patent_wTopics
        patent_topicDist_gensim.T[8, :] = doc_affili_gensim

        pd.DataFrame(patent_topicDist_gensim).to_csv('patent_topicDist_gensim.csv', index=None)

        # print('Preview of the resulting Array (Gensim):\n\n', patent_topicDist_gensim[0])           #[12568 'EP' 1946896.0 '2008-07-23' 15 'Method for adjusting at least one axle'
        # 'An adjustment method for at least one axis (10) in which a robot has a control unit (12) for controlling an axis (10) via which at least two component parts (16,18) are mutually movable. The component parts (16,18) each has at least one marker (24,26) and the positions of the markers are detected by a sensor, with the actual value of a characteristic value ascertained as a relative position of the two mutually movable components (16,18). The adjustment position is repeated by comparing an actual value with a stored, desired value for the adjustment position. An independent claim is included for a device with signal processing unit.'
        # 1 list([(26, 0.12142762), (53, 0.106452435), (93, 0.08961137), (106, 0.06541982), (226, 0.12752746)])]
        # print('\nShape of the resulting Array (Gensim):', np.shape(patent_topicDist_gensim))        #(3781, 9)

        ### Save Topics - Gensim ###

        topics_gensim = lda_gensim.print_topics(num_topics=-1, num_words=8)
        topics_gensim = np.array(topics_gensim)

        pd.DataFrame(topics_gensim).to_csv('patent_topics_gensim.csv', index=None)

    if final_model_mallet == True:

        ### Build Mallet LDA model ###

        mallet_path = r'C:/mallet/bin/mallet'  # update if necessary
        '''
        lda_mallet = models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=325, id2word=id2word, random_seed=123)
        # 0.4530598854343954

    ### Compute Perplexity - Gensim ###

        # The Mallet wrapper does not seem to support log_perplexity. It is omitted here, since it is argued to be a misleading/irrelevant score anyways

        #print('Perplexity of final LDA (Mallet): ', lda_mallet.log_perplexity(corpus))
        # The lower the better, but heavily limited and not really useful.


    ### Compute Coherence Score - Mallet ###

        coherence_model_lda_mallet = models.CoherenceModel(model=lda_mallet, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
        coherence_ldamallet = coherence_model_lda_mallet.get_coherence()
        for i in range(10):
            print('Coherence Score (c_v) of final LDA (Mallet): ', coherence_ldamallet)             # 0.45689701111307507
        # the higher the better from 0.3 to 0.7 or 0.8.
        # source: https://stackoverflow.com/questions/54762690/what-is-the-meaning-of-coherence-score-0-4-is-it-good-or-bad

        lda_mallet = models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=325, id2word=id2word, alpha=65 ,random_seed=123)
        coherence_model_lda_mallet = models.CoherenceModel(model=lda_mallet, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
        coherence_ldamallet = coherence_model_lda_mallet.get_coherence()
        for i in range(10):
            print('Coherence Score (c_v) of final LDA (Mallet): ', coherence_ldamallet)  # 0.4756256448838956
        '''

        #     def __init__(self, mallet_path, corpus=None, num_topics=100, alpha=50, id2word=None, workers=4, prefix=None,
        #                  optimize_interval=0, iterations=1000, topic_threshold=0.0, random_seed=0):

        lda_mallet = models.wrappers.LdaMallet(mallet_path,
                                               corpus=corpus,
                                               num_topics=325,
                                               id2word=id2word,
                                               # alpha= 50,
                                               # beta = 0.02,
                                               # optimize_interval=0,
                                               # iterations=1000,
                                               random_seed=123)

        # plain                 no topics               iterations=10           optimize_interval=1
        # 0.4530598854343954    0.37365560343398985     0.31727626116792185     0.3722214392230059
        # 0.47429113995964317
        #                                                                       optimize_interval=10
        #                                                                       0.37720230356805884

        #                                                                       optimize_interval=100
        #                                                                       0.40265562816232037

        #                                                                       optimize_interval=1000
        #                                                                       0.4530598854343954 (same as plain)

        coherence_model_lda_mallet = models.CoherenceModel(model=lda_mallet, texts=abst_clean, dictionary=id2word,
                                                           coherence='c_v')
        coherence_ldamallet = coherence_model_lda_mallet.get_coherence()
        for i in range(10):
            print('Coherence Score (c_v) of final LDA (Mallet): ', coherence_ldamallet)  # 0.3722214392230059  # 1
            # 0.37720230356805884 # 10
            # 0.40265562816232037 # 100

        coherence_model_lda_mallet = models.CoherenceModel(model=lda_mallet, texts=abst_clean, dictionary=id2word,
                                                           coherence='c_uci')
        coherence_ldamallet = coherence_model_lda_mallet.get_coherence()
        for i in range(10):
            print('Coherence Score (c_v) of final LDA (Mallet): ', coherence_ldamallet)

        # Dirichlet hyperparameter alpha: Document-Topic Density
        # Dirichlet hyperparameter beta: Word-Topic Density

        # https://stats.stackexchange.com/questions/37405/natural-interpretation-for-lda-hyperparameters/37444#37444
        # https://people.cs.umass.edu/~wallach/talks/priors.pdf

        # I want to use the model with the best complexity
        #   For this I want to use ldaMallet in the gridsearch
        #       For this I need to know, why I cant set the beta parameter
        #           Can get the parameters of gensim lda?
        #                  If so, can I ship the mallet to gensim and read the parameters?
        #           What does optimize_interval do?
        #           Try alpha and beta as auto

        # why is there no beta?
        #   https://stackoverflow.com/questions/61870826/why-can-i-not-choose-a-beta-parameter-when-conducting-lda-with-mallet
        #   https://dragonfly.hypotheses.org/1051

        #           Can the model optimize itself? Also with regard to the number of topics?
        # Ask Jonathan about all that

        #           What is the difference in coherence scores? Which should I use? # https://towardsdatascience.com/evaluate-topic-model-in-python-latent-dirichlet-allocation-lda-7d57484bb5d0
        #               I will most likely stick with c_v

        ### Save Document-Topic affiliation - Mallet ###

        # This gives all topics, no parameer to restrict by probability/coverage threshold

        doc_affili_mallet = lda_mallet.read_doctopics(fname=lda_mallet.fdoctopics(), eps=0.05, renorm=False)
        # ldamallet.load_document_topics() reports all topic affiliations. No threshold selectable

        patent_topicDist_mallet = patent_wTopics

        c = 0
        for i in doc_affili_mallet:
            patent_topicDist_mallet.T[8, c] = list(i)
            c = c + 1

        pd.DataFrame(patent_topicDist_mallet).to_csv('patent_topicDist_mallet.csv', index=None)

        # print('Preview of the resulting Array (Mallet):\n\n', patent_topicDist_mallet[0])               # [12568 'EP' 1946896.0 '2008-07-23' 15 'Method for adjusting at least one axle'
        # 'An adjustment method for at least one axis (10) in which a robot has a control unit (12) for controlling an axis (10) via which at least two component parts (16,18) are mutually movable. The component parts (16,18) each has at least one marker (24,26) and the positions of the markers are detected by a sensor, with the actual value of a characteristic value ascertained as a relative position of the two mutually movable components (16,18). The adjustment position is repeated by comparing an actual value with a stored, desired value for the adjustment position. An independent claim is included for a device with signal processing unit.'
        # 1 list([(177, 0.05602006688963211), (306, 0.07775919732441472)])]
        # print('\nShape of the resulting Array (Mallet):', np.shape(patent_topicDist_mallet))            # (3781, 9)

        ### Save Topics - Mallet ###

        topics_mallet = lda_mallet.print_topics(num_topics=-1, num_words=8)
        topics_mallet = np.array(topics_mallet)

        pd.DataFrame(topics_mallet).to_csv('patent_topics_mallet.csv', index=None)

        # print('Preview of the topics (Mallet):\n\n', topics_mallet[0])                                  #   ['0'
        #  '0.178*"behavior" + 0.126*"response" + 0.096*"system" + 0.086*"create" + 0.049*"enable" + 0.045*"management" + 0.032*"resource" + 0.029*"execute"']

    # --- Grid search ---#
    print('\n#--- Grid search ---#\n')

    # todo: implement version with mallet lda instead of gensim lda
    # todo: implement fancier grid search (see data mining 2)

    if grid_search == True:

        mallet_path = r'C:/mallet/bin/mallet'  # update if necessary


        # supporting function
        def compute_coherence_values_gensim(corpus, dictionary, k, a, b):
            # lda_model = models.LdaModel(corpus=corpus,
            lda_model = models.LdaMulticore(corpus=corpus,
                                            id2word=dictionary,
                                            random_state=123,
                                            num_topics=k,
                                            alpha=a,
                                            eta=b,
                                            passes=10,
                                            # chunksize=100,
                                            )

            coherence_model_lda = models.CoherenceModel(model=lda_model, texts=abst_clean, dictionary=id2word,
                                                        coherence='c_v')

            return coherence_model_lda.get_coherence()


        def compute_coherence_values_mallet(corpus, dictionary, num_topics):  # , a, op): #, it): #, b):

            lda_mallet = models.wrappers.LdaMallet(mallet_path,
                                                   corpus=corpus,
                                                   id2word=dictionary,
                                                   random_seed=123,
                                                   num_topics=num_topics,
                                                   # alpha=a,
                                                   # optimize_interval=op,
                                                   # iterations=it,
                                                   )

            # print('im called')

            coherence_model_lda_mallet = models.CoherenceModel(model=lda_mallet, texts=abst_clean,
                                                               dictionary=dictionary, coherence='c_v')

            return coherence_model_lda_mallet.get_coherence()


        # Initiallizing Grid search

        # because it is unfeasable to check all the topic files of 2k ldas, first only the topic number is iterated through with the
        # default settings to arrive at a reasonable topic number (probably taking the one with the biggest increase in coherency)

        grid = {}
        grid['Validation_Set'] = {}

        # Topics range      #1      #2      #3
        min_topics = 10  # 20     #200    #300
        max_topics = 1000  # 200   #300    #420
        step_size = 10
        topics_range = range(min_topics, max_topics, step_size)

        # Alpha parameter
        # alpha = list(np.arange(0.01, 0.31, 0.03))
        # alpha.append('symmetric')
        # alpha.append('asymmetric')
        # alpha.append('auto')

        # Beta parameter
        # beta = list(np.arange(0.01, 1, 0.3))
        # beta.append('symmetric')
        # .append('auto')

        # optimize_interval = list(np.arange(100, 2100, 200))

        # iterations = list(np.arange(1, 1001, 100))

        # Validation sets
        num_of_docs = len(corpus)
        corpus_sets = [  # utils.ClippedCorpus(corpus, num_of_docs*0.25),
            # utils.ClippedCorpus(corpus, num_of_docs*0.5),
            # utils.ClippedCorpus(corpus, int(num_of_docs * 0.75)),
            corpus
        ]
        corpus_title = [  # '75% Corpus',
            '100% Mallet_Corpus'
        ]

        model_results = {'Validation_Set': [],
                         'Topics': [],
                         # 'Alpha': [],
                         # 'optimize_interval': [],
                         # 'iterations': [],
                         # 'Beta': [],
                         'Coherence': []
                         }

        # Typical value of alpha which is used in practice is 50/T where T is number of topics and value of
        # beta is 0.1 or 200/W , where W is number of words in vocabulary.

        pbar = tqdm.tqdm(total=1000)  # adjust if hyperparameters change # 21*7*6*1
        # c = 0
        # iterate through validation corpuses
        for i in range(len(corpus_sets)):
            # iterate through number of topics
            for k in topics_range:
                # iterate through alpha values
                # for a in alpha:
                # iterare through beta values
                # for b in beta:
                # for op in optimize_interval:
                # for it in iterations:
                # get the coherence score for the given parameters
                # cv = compute_coherence_values_gensim(corpus=corpus_sets[i], dictionary=id2word, k=k, a=a, b=b)

                cv = compute_coherence_values_mallet(corpus=corpus_sets[i], dictionary=id2word,
                                                     num_topics=k)  # , a=a, op=op) #, it=it) #, b=b)

                # Save the model results
                model_results['Validation_Set'].append(corpus_title[i])
                model_results['Topics'].append(k)
                # model_results['Alpha'].append(a)
                # model_results['optimize_interval'].append(op)
                # model_results['iterations'].append(it)
                # model_results['Beta'].append(b)
                model_results['Coherence'].append(cv)

                # c = c + 1
                pbar.update(1)

        # save result
        # print(c)

        pd.DataFrame(model_results).to_csv('lda_tuning_results_Mallet_default-Topics1000.csv', index=False)
        pbar.close()

    # FileNotFoundError: [Errno 2] No such file or directory: '... _state.mallet.gz' -> updated smart-open from 3.0.0 to 5.1.0

'''
# supporting function
        def compute_coherence_values(corpus, dictionary, k, a, b):
            lda_model = models.LdaMulticore(corpus=corpus,
                                            id2word=dictionary,
                                            num_topics=k,
                                            random_state=100,
                                            chunksize=100,
                                            passes=10,
                                            alpha=a,
                                            eta=b)

            coherence_model_lda = models.CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word,
                                                 coherence='c_v')

            return coherence_model_lda.get_coherence()


        # Initiallizing Grid search

        grid = {}
        grid['Validation_Set'] = {}

        # Topics range
        min_topics = 25
        max_topics = 425
        step_size = 25
        topics_range = range(min_topics, max_topics, step_size)

        # Alpha parameter
        alpha = list(np.arange(0.01, 1, 0.3))
        alpha.append('symmetric')
        alpha.append('asymmetric')

        # Beta parameter
        beta = list(np.arange(0.01, 1, 0.3))
        beta.append('symmetric')

        # Validation sets
        num_of_docs = len(corpus)
        corpus_sets = [  # gensim.utils.ClippedCorpus(corpus, num_of_docs*0.25),
            # gensim.utils.ClippedCorpus(corpus, num_of_docs*0.5),
            utils.ClippedCorpus(corpus, int(num_of_docs * 0.75)),
            corpus]
        corpus_title = ['75% Corpus', '100% Corpus']

        model_results = {'Validation_Set': [],
                         'Topics': [],
                         'Alpha': [],
                         'Beta': [],
                         'Coherence': []
                         }

        #todo check validation set differentiation between 75% and 100%

        pbar = tqdm.tqdm(total=960)                 # adjust if hyperparameters change

        # iterate through validation corpuses
        for i in range(len(corpus_sets)):
            # iterate through number of topics
            for k in topics_range:
                # iterate through alpha values
                for a in alpha:
                    # iterare through beta values
                    for b in beta:

                        # get the coherence score for the given parameters
                        cv = compute_coherence_values(corpus=corpus_sets[i], dictionary=id2word, k=k, a=a, b=b)

                        # Save the model results
                        model_results['Validation_Set'].append(corpus_title[i])
                        model_results['Topics'].append(k)
                        model_results['Alpha'].append(a)
                        model_results['Beta'].append(b)
                        model_results['Coherence'].append(cv)

                        pbar.update(1)

        # save result

        pd.DataFrame(model_results).to_csv('lda_tuning_results.csv', index=False)
        pbar.close()

'''

'''
For input string: "auto"
Traceback (most recent call last):
  File "C:\PycharmProjects\MasterThesis\v3_textPreproc_LDA_gridSeach.py", line 574, in <module>
    cv = compute_coherence_values_mallet(corpus=corpus_sets[i], dictionary=id2word, num_topics=k, a=a, op=op) #, it=it) #, b=b)
  File "C:\PycharmProjects\MasterThesis\v3_textPreproc_LDA_gridSeach.py", line 492, in compute_coherence_values_mallet
    lda_mallet = models.wrappers.LdaMallet(mallet_path,
  File "C:\PycharmProjects\thesis_venv\lib\site-packages\gensim\models\wrappers\ldamallet.py", line 131, in __init__
    self.train(corpus)
  File "C:\PycharmProjects\thesis_venv\lib\site-packages\gensim\models\wrappers\ldamallet.py", line 285, in train
    self.word_topics = self.load_word_topics()
  File "C:\PycharmProjects\thesis_venv\lib\site-packages\gensim\models\wrappers\ldamallet.py", line 343, in load_word_topics
    with utils.open(self.fstate(), 'rb') as fin:
  File "C:\PycharmProjects\thesis_venv\lib\site-packages\smart_open\smart_open_lib.py", line 222, in open
    binary = _open_binary_stream(uri, binary_mode, transport_params)
  File "C:\PycharmProjects\thesis_venv\lib\site-packages\smart_open\smart_open_lib.py", line 324, in _open_binary_stream
    fobj = submodule.open_uri(uri, mode, transport_params)
  File "C:ycharmProjects\thesis_venv\lib\site-packages\smart_open\local_file.py", line 34, in open_uri
    fobj = io.open(parsed_uri['uri_path'], mode)
FileNotFoundError: [Errno 2] No such file or directory: 'C:\\Users\\UNIVER~1\\AppData\\Local\\Temp\\7d6563_state.mallet.gz'
 21%|██▏       | 60/280 [58:01<3:32:46, 58.03s/it]
'''