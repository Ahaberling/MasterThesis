import re
import nltk
import pandas as pd
import numpy as np
from gensim import models
from gensim.utils import simple_preprocess
import spacy



#--- Initialization ---#

directory = 'D:/Universitaet Mannheim/MMDS 7. Semester/Master Thesis/Outline/Data/Cleaning Robots/'

patent = pd.read_csv(directory + 'cleaning_robot_EP_patents.csv', quotechar='"', skipinitialspace=True)
patent = patent.to_numpy()



#--- Overview ---#

#print(patent.columns)              # pat_publn_id, publn_auth, publn_nr, publn_date, publn_claims, publn_title, publn_abstract, nb_IPC
#print(np.shape(patent))            # (3844, 8)


#--- Define cleaning functions ---#

### Remove non-alphabetic characters ###
def non_alphab_cleaner(text):
    return re.sub('[^A-Za-z]', ' ', text)

### Make all lower case ###
def lower_caser(text):
    return text.lower()

### Tokenizer ###
def tokenizer(text):
    return nltk.word_tokenize(text)

abstract_intermediate = np.array([[np.shape(patent[0])]])

for abstract in patent[0:10,6]:
    non_alpha_ab = non_alphab_cleaner(abstract)
    lowCase_ab = lower_caser(non_alpha_ab)
    tok_ab = tokenizer(lowCase_ab)

    abstract_intermediate = np.append(abstract_intermediate, tok_ab, axis = 1)


print(abstract_intermediate)

'''
# Build the bigram and trigram models
bigram = models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = models.Phrases(bigram[data_words], threshold=100)


# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = models.phrases.Phraser(bigram)
trigram_mod = models.phrases.Phraser(trigram)

'''