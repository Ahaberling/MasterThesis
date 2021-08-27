import re
import numpy as np

class PatentCleaning:

# Do I want self in here?

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


class AbstractCleaning:

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
    def lemmatization(texts, spacy_en, allowed_postags):
        texts_out = []
        for sent in texts:
            doc = spacy_en(" ".join(sent))
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        return texts_out

    '''    
    @staticmethod
    @staticmethod
    @staticmethod
    @staticmethod
    @staticmethod
    @staticmethod
    '''

