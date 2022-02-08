import numpy as np 

import re
import nltk
import gensim.models as gensim_models

import random

import tqdm



###--- Class PatentCleaning ---### ------------------------------------------------------------------------

class PatentCleaning:

    @staticmethod
    def remove_foreign_patents(patents, language, count):

        if language == 'ger':
            check_list_ger = []

            for i in range(len(patents[:, 6])):
                # German abstract check
                # Assumption: there are no english words beginning with 'ein' worth considering
                regexp = re.compile(r'\sein')
                if regexp.search(patents[i, 6]):
                    check_list_ger.append(i)

            patents_clean = np.delete(patents, check_list_ger, 0)
            number_of_patents_removed = len(check_list_ger)

        elif language == 'fr':
            check_list_fr = []

            for i in range(len(patents[:, 6])):
                # France abstract check
                # Assumption: the term 'une' is not used in english patent abstracts
                regexp = re.compile(r'\sune\s')
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
    def draw_stochastic_IPC_sample(patents_english_IPC, level, identifier, sampleSize):

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
                        result_list.append((patent[8:][ipc], list(np.r_[patent[0], patent[5:7]])))

        sample_list = []
        for i in range(0, sampleSize):
            rand_pos = random.randint(0, len(result_list) - 1)
            sample_list.append((result_list[rand_pos][0:]))

        return sample_list

    @staticmethod
    def count_abstracts_with_term(patents, term):
        check_list_term = []

        for i in range(len(patents[:, 6])):
            regexp = re.compile(term)
            if regexp.search(patents[i, 6]):
                check_list_term.append(i)

        return len(check_list_term)



###--- Class AbstractCleaning ---### ------------------------------------------------------------------------

class AbstractCleaning:

    @staticmethod
    def get_smallest_abstracts(patents, min_abstract_size):
        smallestAbstracts = []

        for abstract in patents:
            words_inAbstract = []
            onlyAlphabetic = re.sub('[^A-Za-z]', ' ', abstract)
            tokenized_abstract = nltk.word_tokenize(onlyAlphabetic)

            for word in tokenized_abstract:
                words_inAbstract.append(word.lower())

            if len(words_inAbstract) <= min_abstract_size:
                smallestAbstracts.append(abstract)

        return smallestAbstracts

    @staticmethod
    def vectorize_preprocessing(x):
        return np.vectorize(AbstractCleaning.preprocessing)(x)

    @staticmethod
    def preprocessing(text):

        text = re.sub('[^A-Za-z]', ' ', text)  # Remove non-alphabetic characters
        text = re.sub('\s[A-Za-z]\s', ' ',
                      text)  # Remove single letters ('x', 'y', 'z' occur in abstracts, when refering to axes)
        # These references are assumed to be irrelevant for now, see thesis)
        text = text.lower()  # Make all the strings lowercase and
        return text

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



###--- Class LDA_functions ---### ------------------------------------------------------------------------

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



###--- Class TransformationMisc ---### ------------------------------------------------------------------------

class TransformationMisc:

    @staticmethod
    def fill_with_IPC(array_toBeFilled, patent_IPC, max_numIPC):
        count_list = []

        for i in patent_IPC:

            if i[0] in array_toBeFilled[:,0]:  # For each row in patent_IPC, check if id in patent_join (identical to patent_transf)
                count_l = count_list.count(i[0])  # Retrieve how often the id has been seen yet (how often ipc's where appended already

                array_toBeFilled[array_toBeFilled[:, 0] == i[0], -(max_numIPC - count_l * 3)] = i[1]
                array_toBeFilled[array_toBeFilled[:, 0] == i[0], -(max_numIPC - count_l * 3 - 1)] = i[2]
                array_toBeFilled[array_toBeFilled[:, 0] == i[0], -(max_numIPC - count_l * 3 - 2)] = i[3]

            count_list.append(i[0])

        array_filled = array_toBeFilled

        return array_filled

    @staticmethod
    def max_number_topics(dataset):
        # Eliminating round brackets
        # e.g. [(45, 0.06), (145, 0.05), ...] to ['45', '0.06', '145', '0.05']

        transf_list = []
        for i in range(len(dataset)):
            topic_transf = re.findall("(\d*\.*?\d+)", dataset[i, len(dataset.T)-1])
            transf_list.append(topic_transf)

        # Identify the number of topics the abstract/s with the most topics has/have
        list_len = [len(i) for i in transf_list]
        max_topics = max(list_len) / 2

        return transf_list, max_topics

    @staticmethod
    def fill_with_topics(array_toBeFilled, topic_list, column_start):

        c = 0
        for i in topic_list:

            # Create tuple list of topic_id and coverage to sort by coverage #
            tuple_list = []

            # e.g. ['45', '0.06', '145', '0.05'] to [('45', '0.06'), ('145', '0.05'), ...]
            for j in range(0, len(i) - 1, 2):
                tuple = (i[j], i[j + 1])
                tuple_list.append(tuple)

            # Sort by coverage #
            tuple_list = sorted(tuple_list, key=lambda tup: tup[1], reverse=True)

            # Insert values ordered in new array #
            l = 0

            for k in range(len(tuple_list)):
                # + l because the new data is appended to the empty columns following the filled ones
                array_toBeFilled[c, column_start + l] = tuple_list[k][0]  # topic_id

                l = l + 1

                array_toBeFilled[c, column_start + l] = tuple_list[k][1]  # topic_coverage
                l = l + 1

            c = c + 1

        array_filled = array_toBeFilled

        return array_filled



###--- Class Transformation_SlidingWindows ---### ------------------------------------------------------------------------

class Transformation_SlidingWindows:

    @staticmethod
    def sliding_window_slizing(windowSize, slidingInterval, array_toBeSlized, max_topics):

        array_time = array_toBeSlized[:, 3].astype('datetime64')

        array_time_unique = np.unique(array_time)
        array_time_unique_filled = np.arange(np.min(array_time_unique), np.max(array_time_unique))
        array_time_unique_filled_windowSize = array_time_unique_filled[array_time_unique_filled <= max(array_time_unique_filled) - windowSize]

        slidingWindow_dict = {}
        patents_perWindow = []
        topics_perWindow = []
        topics_perWindow_unique = []

        c = 0
        pbar = tqdm.tqdm(total=len(array_time_unique_filled_windowSize))

        for i in array_time_unique_filled_windowSize:

            if c % slidingInterval == 0:
                lower_limit = i
                upper_limit = i + windowSize

                array_window = array_toBeSlized[(array_toBeSlized[:, 3].astype('datetime64') < upper_limit) & (
                        array_toBeSlized[:, 3].astype('datetime64') >= lower_limit)]
                patents_perWindow.append(len(array_window))

                topics_perWindow_helper = []
                for topic_list in array_window[:, 9:25]:                    # not dynamic yet. Adapt with max_topics
                    for column_id in range(0, len(topic_list.T), 2):
                        if topic_list[column_id] != None:
                            topics_perWindow_helper.append(topic_list[column_id])

                topics_perWindow.append(len(topics_perWindow_helper))
                topics_perWindow_unique.append(len(np.unique(topics_perWindow_helper)))

                slidingWindow_dict['window_{0}'.format(c)] = array_window

            c = c + 1
            pbar.update(1)

        pbar.close()

        return slidingWindow_dict, patents_perWindow, topics_perWindow, topics_perWindow_unique



###--- Class Transformation_Network ---### ------------------------------------------------------------------------

class Transformation_Network:

    @staticmethod
    def prepare_patentNodeAttr_Networkx(window, nodes, node_att_name):

        node_att_dic_list = []

        for i in range(len(window)):

            dic_entry = dict(enumerate(window[i]))  # Here each patent is converted into a dictionary. Dictionary keys are still numbers:
            # {0: 'EP', 1: nan, 2: '2007-10-10', ...} Note that the patent id is ommited, since it
            # serves as key for the outer dictionary encapsulating these inner once.

            for key, n_key in zip(dic_entry.copy().keys(), node_att_name):
                dic_entry[n_key] = dic_entry.pop(key)  # {'publn_auth': 'EP', 'publn_nr': nan, 'publn_date': '2009-06-17', ...}

            node_att_dic_list.append(dic_entry)

        nested_dic = dict(enumerate(node_att_dic_list))  # Here the nested (outer) dictionary is created. Each key is still
                                                         # represented as a number, each value as another dictionary

        for key, n_key in zip(nested_dic.copy().keys(),nodes):  # Here the key of the outer dictionary are renamed to the patent ids
            nested_dic[n_key] = nested_dic.pop(key)

        return nested_dic

    @staticmethod
    def prepare_topicNodes_Networkx(window, topic_position):
        topics_inWindow = []

        for patent in window:
            topics_inWindow.append(patent[topic_position])

        topics_inWindow = [item for sublist in topics_inWindow for item in sublist]
        topics_inWindow = [x for x in topics_inWindow if x is not None]
        topics_inWindow = np.unique(topics_inWindow)
        topicNode_list = ['topic_{0}'.format(int(i)) for i in topics_inWindow]

        return topicNode_list

    @staticmethod
    def prepare_edgeLists_Networkx(window, num_topics, max_topics):

        if num_topics >= max_topics + 1:
            raise Exception("num_topics must be <= max_topics")

        edges = window[:, np.r_[0, 9:(9 + (num_topics * 2))]]  # first three topics
        topic_edges_list = []

        for i in range(1, (num_topics * 2) + 1, 2):
            c = 0
            for j in edges.T[i]:
                if j != None:
                    edges[c, i] = 'topic_{0}'.format(int(j))
                c = c + 1

            topic_edges = [(j[0], j[i], {'Weight': j[i + 1]}) for j in edges]
            topic_edges_clear = list(filter(lambda x: x[1] != None, topic_edges))

            topic_edges_list.append(topic_edges_clear)

        return topic_edges_list

    @staticmethod
    def custom_projection_function(G, u, v):

        u_nbrs = set(G[u])      # Neighbors of Topic1 in set format for later intersection
        v_nbrs = set(G[v])      # Neighbors of Topic2 in set format for later intersection
        shared_nbrs = u_nbrs.intersection(v_nbrs)       # Shared neighbors of both topic nodes (intersection)

        list_of_poducts = []
        for i in shared_nbrs:

            weight1 = list(G.edges[u,i].values())[0]
            weight2 = list(G.edges[v,i].values())[0]

            list_of_poducts.append(float(weight1) * float(weight2))

        projected_weight = sum(list_of_poducts) / len(list_of_poducts)

        return projected_weight
