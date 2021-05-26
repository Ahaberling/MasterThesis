### This file relies on patent data provided by the Mannheimer chair of organization and innovation.
### The data origins from the PATSTAT Database and is supposed to feature patents concerned with cleaning robots.
### In the following the data is first preprocessed. Afterwards LDA topic modelling is applied.
### The modelled topics and the topic affiliation of the patents are used to enrich the data set for further analyses.

### This file is mostly concerned with the abstracts of the patent (publn_abstract)

if __name__ == '__main__':

#--- Import Libraries ---#

    import re
    import nltk
    from nltk.corpus import stopwords
    import pandas as pd
    import numpy as np
    from gensim import models, utils
    from gensim.models import CoherenceModel
    from gensim.models.wrappers import LdaMallet
    import gensim.corpora as corpora
    import spacy
    import os
    from pprint import pprint
    import tqdm
    import sys

    # Specify whether you want to simply preform LDA or a grid_search for optimal LDA hyperparameters
    final_model = True
    grid_search = False


#--- Initialization ---#

    nltk.download('punkt')              # nltk tokenizer
    nltk.download('stopwords')          # nltk stopwords filter

    os.chdir('D:/Universitaet Mannheim/MMDS 7. Semester/Master Thesis/Outline/Data/Cleaning Robots')

    patent = pd.read_csv('cleaning_robot_EP_patents.csv', quotechar='"', skipinitialspace=True)
    patent = patent.to_numpy()
    #patent = patent.sample(100)



#--- Patent investigation ---#

    # The provided data set contains patents formulated in german or france. These patents are filtered out to
    # facilitate a more accurate LDA modelling based on only english patents.

    # Additionally: Data is said to be sampled around cleaning robots. If terms 'robot' and 'clean' occur in every
    # patent, then they might as well be excluded from Topic Modelling.


    # print(patent.columns)              # pat_publn_id, publn_auth, publn_nr, publn_date, publn_claims, publn_title, publn_abstract, nb_IPC
    # print(np.shape(patent))            # (3844, 8)

    ### Check for patents in german or france and for patents containing the terms 'robot' and 'clean' ###

    check_list_ger = []
    check_list_fr = []
    check_list_robot = []
    check_list_clean = []

    for i in range(len(patent[:,6])):

       # German abstract check
        regexp = re.compile(r'\sein')
        if regexp.search(patent[i,6]):
            check_list_ger.append(i)

       # France abstract check
        regexp = re.compile(r'\sune\s')
        if regexp.search(patent[i,6]):
            check_list_fr.append(i)

        # 'robot' abstract check
        regexp = re.compile('robot')
        if regexp.search(patent[i, 6]):
            check_list_robot.append(i)

       # 'clean' abstract check
        regexp = re.compile('clean')
        if regexp.search(patent[i,6]):
            check_list_clean.append(i)

    #print('Number of patents in german: ', len(removal_list_ger))              #   48/3844 german patents
    #print('Number of patents in frensh: ', len(removal_list_fr))               #   15/3844 frensh patents

    #print('Number of patents with term \'robot\': ', len(check_list_robot))    # 3844/3844 patents including 'robot'
    #print('Number of patents with term \'clean\': ', len(check_list_clean))    #  398/3844 patents including 'clean'

    #todo the data seems not solely sampled around cleaning robots, but robots in general. If only 398/3844 patents
    # refer the cleaning robots, then the term 'clean' might not be excluded from LDA, since it might enable a valid
    # topic.


#--- Patent Preprocessing ---#

    ### Removing non-english patents ###

    removal_list_all = []
    removal_list_all.append(check_list_ger)
    removal_list_all.append(check_list_fr)
    removal_list_all = [item for sublist in removal_list_all for item in sublist]

    patent = np.delete(patent, removal_list_all, 0)

    #print('Number of (non-english) patents removed: ', len(removal_list_all))      # 48 + 15 = 63 out of 3844
    #print('Number of (english) patents left: 'len(patent))                         # 3781         out of 3844


    ### New Array including space for preprocessed abstracts ###

    patent_cleanAbs = np.empty((np.shape(patent)[0],np.shape(patent)[1]+1), dtype = object)
    patent_cleanAbs[:,:-1] = patent


    ### Define functions for abstract cleaning ###

    def preprocessing(text):

        text = re.sub('[^A-Za-z]', ' ', text)           # Remove non alphabetic characters
        text = re.sub('\s[A-Za-z]\s', ' ', text)        # Remove single letters ('x', 'y', 'z' occur in abstracts, when refering to axes.
                                                        # These references are assumed to be irrelevant for now)
        text = text.lower()                             # Make all the strings lowercase and
        return text

    def vectorize(x):
        return np.vectorize(preprocessing)(x)

    # todo is this proper vectorization of a function?


    ### Apply functions for abstract cleaning ###

    semi_preprocessed = vectorize(patent[:,6])


    ### Apply tokenization ###

    tokenized_abst = []
    for i in range(len(semi_preprocessed)):
        tokenized_abst.append(nltk.word_tokenize(semi_preprocessed[i]))

    #todo maybe find a way to properly vectorize this as well


    ###  Build bigram and trigram models ###

    bigram = models.Phrases(tokenized_abst, min_count=5, threshold=100)         # higher threshold fewer phrases.
    trigram = models.Phrases(bigram[tokenized_abst], threshold=100)

    bigram_mod = models.phrases.Phraser(bigram)
    trigram_mod = models.phrases.Phraser(trigram)

    #print(trigram_mod[bigram_mod[tokenized_abst[0]]])


    ### Define Stopwords ###


    stop_words = stopwords.words('english')
    stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

    stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'not', 'would', 'say', 'could', '_', 'be', 'know', 'good',
                   'go', 'get', 'do', 'done', 'try', 'many', 'some', 'nice', 'thank', 'think', 'see', 'rather', 'easy',
                   'easily', 'lot', 'lack', 'make', 'want', 'seem', 'run', 'need', 'even', 'right', 'line', 'even',
                   'also', 'may', 'take', 'come'])

    stop_words.extend(['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve',
                 'first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth',
                 'eleventh', 'twelfth',
                 'i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x', 'xi', 'xii',
                 'robot'])

    #todo adapt when the data sample to be used for all analyses is decided


    ### Define functions for stopwords, bigrams, trigrams and lemmatization ###

    def remove_stopwords(texts):
        clean_text = [word for word in texts if word not in stop_words]
        return clean_text

    def make_bigrams(texts):
        return [bigram_mod[doc] for doc in texts]

    def make_trigrams(texts):
        return [trigram_mod[bigram_mod[doc]] for doc in texts]

    def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        """https://spacy.io/api/annotation"""
        texts_out = []
        for sent in texts:
            doc = nlp(" ".join(sent))
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        return texts_out


    ### Apply functions ###

    # Remove Stop Words
    data_words_nostops = remove_stopwords(tokenized_abst)

    # Form Bigrams
    data_words_bigrams = make_bigrams(data_words_nostops)

    # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
    nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])

    # Do lemmatization keeping only noun, adj, vb, adv
    data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

    #todo check if bigrams at right place. Do we want 'an_independent_claim' and 'also_included' to be tokens? (Assuming they are not cleaned by lemmatization)


#--- Building LDA ---#


    # Create Dictionary
    id2word = corpora.Dictionary(data_lemmatized)


    # Create Corpus
    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in data_lemmatized]


    # Build Gensim LDA model
    if final_model == True:

        lda_model = models.LdaMulticore(corpus=corpus,
                                        id2word=id2word,
                                        num_topics=325,         # adjust num_topics, alpha and eta with regard to grid search results
                                        random_state=100,
                                        chunksize=100,
                                        passes=10,
                                        per_word_topics=True)


        # Print the Keyword in the 10 topics
        #pprint(lda_model.print_topics())


    # Compute Perplexity
        #print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # The lower the better, but heavily limited and not really useful.


    # Compute Coherence Score
        #coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
        #coherence_lda = coherence_model_lda.get_coherence()
        #print('\nCoherence Score: ', coherence_lda)             # the higher the better from 0.3 to 0.7 or 0.8.
        # https://stackoverflow.com/questions/54762690/what-is-the-meaning-of-coherence-score-0-4-is-it-good-or-bad

    ### Append document topic distribution and save ###

        doc_affili = lda_model.get_document_topics(corpus, minimum_probability=0.05, minimum_phi_value=None,
                                                   per_word_topics=False)

        #for i in doc_affili:
            #print(i)
            # [(82, 0.078733206), (115, 0.09037129), (212, 0.061887488), (227, 0.10818202), (263, 0.07558539), (323, 0.06967209)]

        patent_topicDist = patent_cleanAbs
        patent_topicDist.T[8, :] = doc_affili
        #print(patent_topicDist)

        #[12878 'EP' 1947581.0 ...
        #'The method involves inputting a foil geometry in a data processing program and transmitting the data to a foil manufacturer through a data carrier and/or email and/or website. The data in the data processing program is read and a packing plane is detected. Foil portions are cut from blanks manually or by a robot and the produced packages are transported. Add-on programs are utilized, where the add-on programs are integrated in the data processing program of a customer. The cut foil portions are connected. An independent claim is also included for an article manufactured according to a method for packing foils.'
        #2 list([(27, 0.11116354), (300, 0.73837715)])]
        pd.DataFrame(patent_topicDist).to_csv('patent_topicDist.csv', index=None)

    # Build Mallet LDA model

        mallet_path = r'C:/mallet/bin/mallet' # update this path

        ldamallet = models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=325, id2word=id2word)

        # Show Topics
        #pprint(ldamallet.show_topics(formatted=False))
        '''
        # Compute Coherence Score
        coherence_model_ldamallet = CoherenceModel(model=ldamallet, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
        coherence_ldamallet = coherence_model_ldamallet.get_coherence()
        print('\nCoherence Score: ', coherence_ldamallet)
        '''

    ### Append Mallet document topic distribution and save ###

        #doc_affili_mallet = ldamallet.load_document_topics()

        #print(doc_affili_mallet)
        #for i in doc_affili_mallet:
            #print(i)
            # [(0, 0.0016542597187758413), (1, 0.0016542597187758413), (2, 0.0016542597187758413), (3, 0.0016542597187758413), (4, 0.0016542597187758413), ...

        doc_affili_mallet = ldamallet.read_doctopics(fname='C:/Users/University/AppData/Local/Temp/c6a2d9_doctopics.txt', eps= 0.05)
        #for i in doc_affili_mallet:
            #print(i)


        #print(ldamallet.read_doctopics(fname = 'C:/Users/University/AppData/Local/Temp/c6a2d9_doctopics.txt')) # same as ldamallet.load_document_topics()
        #for i in ldamallet.read_doctopics(fname = 'C:/Users/University/AppData/Local/Temp/c6a2d9_doctopics.txt'):
            #print(i)
        #print(ldamallet.fdoctopics())

        patent_topicDist_mallet = patent_cleanAbs

        c = 0
        for i in doc_affili_mallet:
            patent_topicDist_mallet.T[8, c] = list(i)
            c = c + 1

        #patent_topicDist_mallet.T[8, :] = doc_affili_mallet

        np.set_printoptions(threshold=sys.maxsize)

        print(np.shape(patent_topicDist))
        print(patent_topicDist[0])
        print(patent_topicDist[10])
        print(patent_topicDist[20])
        print(patent_topicDist[33])
        print(patent_topicDist[847])
        print(np.shape(patent_topicDist_mallet))
        print(patent_topicDist_mallet[0])
        print(patent_topicDist_mallet[10])
        print(patent_topicDist_mallet[20])
        print(patent_topicDist_mallet[33])
        print(patent_topicDist_mallet[847])

        #print(patent_topicDist_mallet)
        pd.DataFrame(patent_topicDist_mallet).to_csv('patent_topicDist_mallet.csv', index=None)

        #todo: currently, only document topic affiliations of the gensim lda are saved.
        # In future, work with the mallet results if they are better

        #doc_affili = ldamallet.get_document_topics(corpus, minimum_probability=0.05, minimum_phi_value=None,
        #                                   per_word_topics=False)
        # https://radimrehurek.com/gensim_3.8.3/models/wrappers/ldamallet.html

#--- Grid search ---#

    if grid_search == True:

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

            coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word,
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

        #todo check why validation set differentiation between 75% adn 100% is important/usefull

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
