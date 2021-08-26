import re
import numpy as np

class PatentCleaning:

# Do I want self in here?

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



    def count_abstracts_with_word(patents, word):
        check_list_word = []

        for i in range(len(patents[:, 6])):
            regexp = re.compile(word)
            if regexp.search(patents[i, 6]):
                check_list_word.append(i)

        return word, len(check_list_word)


class AbstractCleaning:

    def preprocessing(text):

        text = re.sub('[^A-Za-z]', ' ', text)  # Remove non alphabetic characters
        text = re.sub('\s[A-Za-z]\s', ' ',
                      text)  # Remove single letters ('x', 'y', 'z' occur in abstracts, when refering to axes.
        # These references are assumed to be irrelevant for now)
        text = text.lower()  # Make all the strings lowercase and
        return text


    def vectorize_preprocessing(x):
        return np.vectorize(AbstractCleaning.preprocessing)(x)

