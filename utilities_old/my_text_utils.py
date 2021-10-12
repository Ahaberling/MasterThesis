import re
import numpy as np
import random
import nltk

import gensim.models as gensim_models

class PatentCleaning:

    @staticmethod
    def remove_foreign_patents(patents, language, count):
        number_of_patents_removed = 0

        if language == 'ger':
            check_list_ger = []

            for i in range(len(patents[:, 6])):
                # German abstract check
                regexp = re.compile(
                    r'\sein')  # Assumption: there are no english words beginning with 'ein' worth considering
                if regexp.search(patents[i, 6]):
                    check_list_ger.append(i)

            patents_clean = np.delete(patents, check_list_ger, 0)
            number_of_patents_removed = len(check_list_ger)

        elif language == 'fr':
            check_list_fr = []

            for i in range(len(patents[:, 6])):
                # France abstract check
                regexp = re.compile(r'\sune\s')  # Assumption: the term 'une' is not used in english patent abstracts
                if regexp.search(patents[i, 6]):
                    check_list_fr.append(i)

            patents_clean = np.delete(patents, check_list_fr, 0)
            number_of_patents_removed = len(check_list_fr)

        else:
            raise Exception("Only german 'ger' and french 'fr' supported")

        if count == True:
            return patents_clean, number_of_patents_removed

        elif count == False:
            return patents_clean

        else:
            raise Exception("'Count' must be boolean")


    @staticmethod
    def count_abstracts_with_term(patents, term):
        check_list_term = []

        for i in range(len(patents[:, 6])):
            regexp = re.compile(term)
            if regexp.search(patents[i, 6]):
                check_list_term.append(i)

        return term, len(check_list_term)

    @staticmethod
    def stochastic_inestigation_IPCs(patents_english_IPC, level, identifier, sampleSize):

        if level == 'section':
            start = 0
            end = 1
        elif level == 'class':
            start = 0
            end = 3
        elif level == 'subclass':
            start = 0
            end = 4
        elif level == 'maingoup':
            start = 0
            end = -1
        else:
            raise Exception(
                "'level' must be string 'section', 'class', 'subclass', or 'maingoup'. Subgoups are not present in the dataset, yet")

        result_list = []
        for patent in patents_english_IPC:
            for ipc in range(0, len(patent[8:]), 3):
                if patent[8:][ipc] != None:
                    if patent[8:][ipc][start:end] == identifier:
                        # print(patent, '\n')
                        result_list.append((patent[8:][ipc], list(np.r_[patent[0], patent[5:7]])))
                        # result_list.append((level, identifier, patent[8:][ipc]))

        sample_list = []
        for i in range(0, sampleSize):
            rand_pos = random.randint(0, len(result_list) - 1)
            sample_list.append((result_list[rand_pos][0:])) #[0:4])) #, result_list[rand_pos][1]))

        return sample_list


class AbstractCleaning:

    @staticmethod
    def number_of_word_pre_preprocessing(patents, min_abstract_size):
        word_list = []
        numberOfWords_abstract = []
        numberOfWords_abstract_unique = []
        number_less_20_abstract = 0

        for abstract in patents:
            words_inAbstract = []
            onlyAlphabetic = re.sub('[^A-Za-z]', ' ', abstract)
            tokenized_abstract = nltk.word_tokenize(onlyAlphabetic)

            for word in tokenized_abstract:
                word_list.append(word.lower())
                words_inAbstract.append(word.lower())

            words_inAbstract_unique = np.unique(words_inAbstract)

            if len(words_inAbstract) <= min_abstract_size:
                number_less_20_abstract = number_less_20_abstract + 1

            numberOfWords_abstract.append(len(words_inAbstract))
            numberOfWords_abstract_unique.append(len(words_inAbstract_unique))

        word_list_unique = np.unique(word_list)

        return len(word_list_unique), numberOfWords_abstract, numberOfWords_abstract_unique, number_less_20_abstract

    @staticmethod
    def number_of_word_post_preprocessing(cleaned_abstracts):
        numberOfWords_abstract = []
        numberOfWords_abstract_unique = []

        for abstract in cleaned_abstracts:
            words_inAbstract = []

            for word in abstract:
                words_inAbstract.append(word)

            words_inAbstract_unique = np.unique(words_inAbstract)

            numberOfWords_abstract.append(len(words_inAbstract))
            numberOfWords_abstract_unique.append(len(words_inAbstract_unique))

        return numberOfWords_abstract_unique, numberOfWords_abstract

    @staticmethod
    def preprocessing(text):

        text = re.sub('[^A-Za-z]', ' ', text)  # Remove non alphabetic characters
        text = re.sub('\s[A-Za-z]\s', ' ',
                      text)  # Remove single letters ('x', 'y', 'z' occur in abstracts, when refering to axes.
        # These references are assumed to be irrelevant for now)
        text = text.lower()  # Make all the strings lowercase and
        return text

    @staticmethod
    def vectorize_preprocessing(x):
        return np.vectorize(AbstractCleaning.preprocessing)(x)

    @staticmethod
    def remove_stopwords(texts, filter):
        clean_text = [word for word in texts if word not in filter]
        return clean_text

    @staticmethod
    def make_bigrams(texts, bigram_mod):
        return [bigram_mod[doc] for doc in texts]

    @staticmethod
    def count_bigrams(abst_bigrams):
        bigram_list = []
        for abstract in abst_bigrams:
            for word in abstract:
                if '_' in word:
                    bigram_list.append(word)
        bigram_list = np.unique(bigram_list)

        return bigram_list

    @staticmethod
    def lemmatization(texts, spacy_en, allowed_postags):
        texts_out = []
        for sent in texts:
            doc = spacy_en(" ".join(sent))
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        return texts_out



class LDA_functions:

    @staticmethod
    def gensim_LDA(corpus, id2word, num_topics, random_state, alpha, beta, per_word_topics, abst_clean, coherence, multicore, onlyCoherency):

        if multicore == True:

            lda_gensim = gensim_models.LdaMulticore(corpus=corpus,
                                                    id2word=id2word,
                                                    num_topics=num_topics,
                                                    random_state=random_state,
                                                    alpha=alpha,
                                                    eta=beta,
                                                    per_word_topics=per_word_topics)

        elif multicore == False:

            lda_gensim = gensim_models.LdaModel(corpus=corpus,
                                                id2word=id2word,
                                                num_topics=num_topics,
                                                random_state=random_state,
                                                alpha=alpha,
                                                eta=beta,
                                                per_word_topics=per_word_topics)

        else:
            raise Exception("multicore must be True or False")

        coherence_model_lda = gensim_models.CoherenceModel(model=lda_gensim,
                                                           texts=abst_clean,
                                                           dictionary=id2word,
                                                           coherence=coherence)

        coherence_lda = coherence_model_lda.get_coherence()

        if onlyCoherency == True:

            return coherence_lda

        elif onlyCoherency == False:

            document_topic_affiliation = lda_gensim.get_document_topics(corpus,
                                                                        minimum_probability=0.05,
                                                                        minimum_phi_value=None,
                                                                        per_word_topics=False)

            topics_gensim = lda_gensim.print_topics(num_topics=-1, num_words=8)
            topics_gensim = np.array(topics_gensim)


            return coherence_lda, document_topic_affiliation, topics_gensim

        else:
            raise Exception("onlyCoherency Must be True or False")


    @staticmethod
    def mallet_LDA(corpus, id2word, num_topics, random_seed, alpha, optimize_interval, iterations, abst_clean,
                   coherence, mallet_path, onlyCoherency):
        lda_mallet = gensim_models.wrappers.LdaMallet(mallet_path=mallet_path,
                                                      corpus=corpus,
                                                      id2word=id2word,
                                                      num_topics=num_topics,
                                                      random_seed=random_seed,
                                                      alpha=alpha,
                                                      optimize_interval=optimize_interval,
                                                      iterations=iterations)

        coherence_model_lda = gensim_models.CoherenceModel(model=lda_mallet,
                                                           texts=abst_clean,
                                                           dictionary=id2word,
                                                           coherence=coherence)

        coherence_lda = coherence_model_lda.get_coherence()

        if onlyCoherency == True:

            return coherence_lda

        elif onlyCoherency == False:

            document_topic_affiliation = lda_mallet.read_doctopics(fname=lda_mallet.fdoctopics(), eps=0.05, renorm=False)

            topics_mallet = lda_mallet.print_topics(num_topics=-1, num_words=8)
            topics_mallet = np.array(topics_mallet)

            return coherence_lda, document_topic_affiliation, topics_mallet

        else:
            raise Exception("onlyCoherency Must be True or False")

