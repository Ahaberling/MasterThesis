import re
import nltk
import pandas as pd
import numpy as np
from gensim import models, utils
import spacy
import os


if __name__ == '__main__':

#--- Initialization ---#

    nltk.download('punkt')              # tokenizer
    nltk.download('stopwords')          # stopwords filter

    os.chdir('C:/Universitaet Mannheim/MMDS 7. Semester/Master Thesis/Outline/Data/Cleaning Robots/')

#C:\Universitaet Mannheim\MMDS 7. Semester\Master Thesis\Outline\Data\Cleaning Robots

    patent = pd.read_csv('cleaning_robot_EP_patents.csv', quotechar='"', skipinitialspace=True)
    patent = patent.to_numpy()
    #patent = patent.sample(100)

    #print(patent.columns)              # pat_publn_id, publn_auth, publn_nr, publn_date, publn_claims, publn_title, publn_abstract, nb_IPC
    #print(np.shape(patent))            # (3844, 8)


#--- Patent investigation ---#

    # Identifying german and france patents and patents including the term 'robot' or 'clean'
    # Idea: Data is said to be sampled regarding cleaning robots. If these terms occur in every patent,
    # then they might as well be excluded from Topic Modelling

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

    #print(len(removal_list_ger))    #   48/3844 German patents
    #print(len(removal_list_fr))     #   15/3844 France patents

    #print(len(check_list_robot))    # 3844/3844 patents including 'robot'
    #print(len(check_list_clean))    #  398/3844 patents including 'clean'
                                     # todo maybe 'clean' should not be removed in text preprop, ask about
                                     # sampling method -> better sampling necessary?



    #--- Removing non-english patents ---#

    # Idea: non-english patents bias topic modeling based on (mostly) english abstracts

    removal_list_all = []
    removal_list_all.append(check_list_ger)
    removal_list_all.append(check_list_fr)
    removal_list_all = [item for sublist in removal_list_all for item in sublist]

    patent = np.delete(patent, removal_list_all, 0)

    #print(len(removal_list_all))    # 48 + 15 = 63 abstracts are non-english and removed 63/3844
    #print(len(patent))              # 3844 - 63 = 3781 patents remaining




    ### New Array ###
    patent_cleanAbs = np.empty((np.shape(patent)[0],np.shape(patent)[1]+1), dtype = object) # todo revisite and decide on dtype
    patent_cleanAbs[:,:-1] = patent


    # Remove punctuation

    def preprocessing(text):

        text = re.sub('[^A-Za-z]', ' ', text)           # Make all the strings lowercase and remove non alphabetic characters
        text = re.sub('\s[A-Za-z]\s', ' ', text)        # Remove single letters like x, y, z
        text = text.lower()
        #print(text)
        #text = nltk.word_tokenize(text)
        #print(text)
        #text = str(text)
        #print(text)


        return text

    def vectorize(x):
        return np.vectorize(preprocessing)(x)

    semi_preprocessed = vectorize(patent[:,6])

    #print(patent_cleanAbs)
    #print(patent_cleanAbs[0:1,8][0])
    #print(patent_cleanAbs[0:1,8][0].split())

    # need: list of list containing tokens that where previously cleaned by punctiuation removal and lower case
    # ... , 'teeth'], ['control', ... ,  'insertion', 'hole']]

    #print(patent_cleanAbs[:,8])
    #print(len(patent_cleanAbs[:,8]))

    #print(preprocessing(patent[0:1,6][0]))

    tokenized_abst = []
    for i in range(len(semi_preprocessed)):

        tokenized_abst.append(nltk.word_tokenize(semi_preprocessed[i]))

    #print(tokenized_abst)


    '''
    for i in patent[0:1,6]:

        text = re.sub('[^A-Za-z]', ' ', i)  # Make all the strings lowercase and remove non alphabetic characters
        text = re.sub('\s[A-Za-z]\s', ' ', text)  # Remove single letters like x, y, z
        text = text.lower()

        print('called')
        print(text)
        text = nltk.word_tokenize(text)
        print(text)
        #print(patent[0:1,6][0])

    text = re.sub('[^A-Za-z]', ' ', patent[0:1,6][0])  # Make all the strings lowercase and remove non alphabetic characters
    text = re.sub('\s[A-Za-z]\s', ' ', text)  # Remove single letters like x, y, z
    text = text.lower()

    print('called')
    print(text)
    text = nltk.word_tokenize(text)
    print(text)
    '''
    #print(test)

    '''

    data = papers.abstract_clean.values.tolist()
    #print(data[1])
    data_words = list(sent_to_words(data))
    #print(data_words[1])
    #print(data_words[:1][0][:30])'''

    # Build the bigram and trigram models
    bigram = models.Phrases(tokenized_abst, min_count=5, threshold=100) # higher threshold fewer phrases.
    trigram = models.Phrases(bigram[tokenized_abst], threshold=100)

    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = models.phrases.Phraser(bigram)
    trigram_mod = models.phrases.Phraser(trigram)

    print(trigram_mod[bigram_mod[tokenized_abst[0]]])
    print(trigram_mod[bigram_mod[tokenized_abst[1]]])
    print(trigram_mod[bigram_mod[tokenized_abst[2]]])
    print(trigram_mod[bigram_mod[tokenized_abst[3]]])
    print(trigram_mod[bigram_mod[tokenized_abst[4]]])
    print(trigram_mod[bigram_mod[tokenized_abst[5]]])


    # NLTK Stop words
    # import nltk
    # nltk.download('stopwords')
    from nltk.corpus import stopwords
    stop_words = stopwords.words('english')
    stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

    stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'not', 'would', 'say', 'could', '_', 'be', 'know', 'good',
                   'go', 'get', 'do', 'done', 'try', 'many', 'some', 'nice', 'thank', 'think', 'see', 'rather', 'easy',
                   'easily', 'lot', 'lack', 'make', 'want', 'seem', 'run', 'need', 'even', 'right', 'line', 'even',
                   'also', 'may', 'take', 'come']) #todo adapt

    stop_words.extend(['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve',
                 'first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth',
                 'eleventh', 'twelfth',
                 'i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x', 'xi', 'xii',
                 'robot'])


    # Define functions for stopwords, bigrams, trigrams and lemmatization

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



    import spacy
    # Remove Stop Words
    data_words_nostops = remove_stopwords(tokenized_abst)

    #print(data_words_nostops)

    # Form Bigrams
    data_words_bigrams = make_bigrams(data_words_nostops)
    #print(data_words_bigrams)

    # todo check if bigrams at right place. Do we want 'an_independent_claim' and 'also_included' to be tokens? (I assume they are not cleaned afterwards)

    # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
    nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])



    # Do lemmatization keeping only noun, adj, vb, adv
    data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    #print(data_lemmatized)


    import gensim.corpora as corpora
    # Create Dictionary
    id2word = corpora.Dictionary(data_lemmatized)

    #print(id2word)


    # Create Corpus
    texts = data_lemmatized

    #print(texts)
    #print(len(texts))



    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]
    # View
    #print(corpus)
    #print(len(corpus))

#if __name__ == '__main__':
    # Build LDA model
    lda_model = models.LdaMulticore(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=325,
                                           random_state=100,
                                           chunksize=100,
                                           passes=10,
                                           per_word_topics=True)


    from pprint import pprint
    # Print the Keyword in the 10 topics
    pprint(lda_model.print_topics())
    #doc_lda = lda_model[corpus]

    # Compute Perplexity
    print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

    from gensim.models import CoherenceModel
    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score: ', coherence_lda)     # baseline with sample(100): 0.33683232393956486


    ### mallet model ###
    from gensim.models.wrappers import LdaMallet

    mallet_path = r'C:\Universitaet Mannheim\MMDS 7. Semester\Master Thesis\Outline\Data\Mallet\mallet-2.0.8\bin\mallet' # update this path

    ldamallet = models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=20, id2word=id2word)

    # Show Topics
    pprint(ldamallet.show_topics(formatted=False))

    # Compute Coherence Score
    coherence_model_ldamallet = CoherenceModel(model=ldamallet, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
    coherence_ldamallet = coherence_model_ldamallet.get_coherence()
    print('\nCoherence Score: ', coherence_ldamallet)

    ### Grid search ###

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




    import numpy as np
    import tqdm

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

    ###### working fine untio here


    #print(len(corpus_sets), len(topics_range), len(alpha), len(beta))


    # Can take a long time to run
    if 2 == 1:
        pbar = tqdm.tqdm(total=960)

        # iterate through validation corpuses
        for i in range(len(corpus_sets)):
            # iterate through number of topics
            for k in topics_range:
                # iterate through alpha values
                for a in alpha:
                    # iterare through beta values
                    for b in beta:
                        #print(corpus_sets[i])

                        # get the coherence score for the given parameters
                        cv = compute_coherence_values(corpus=corpus_sets[i], dictionary=id2word, k=k, a=a, b=b)


                        
                        # Save the model results
                        model_results['Validation_Set'].append(corpus_title[i])
                        model_results['Topics'].append(k)
                        model_results['Alpha'].append(a)
                        model_results['Beta'].append(b)
                        model_results['Coherence'].append(cv)

                        pbar.update(1)
        pd.DataFrame(model_results).to_csv('lda_tuning_results.csv', index=False)
        pbar.close()



    '''
    lda_model = models.LdaMulticore(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=8,
                                           random_state=100,
                                           chunksize=100,
                                           passes=10,
                                           alpha=0.01,
                                           eta=0.9)

    coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word,
                                         coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score: ', coherence_lda)  # baseline with sample(100): 0.33683232393956486
    '''