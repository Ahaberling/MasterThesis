import pandas as pd
import numpy as np
from gensim import corpora, models

#import pyLDAvis



#--- Initialization ---#

directory = 'D:/Universitaet Mannheim/MMDS 7. Semester/Master Thesis/Outline/Data/Cleaning Robots/'

patent_clean = pd.read_csv(directory + 'clean_patents.csv', quotechar='"', skipinitialspace=True)
patent_clean = patent_clean.to_numpy()



#--- Preparing dictionary and corpus for lda ---#

dictionary = corpora.Dictionary([d.split() for d in patent_clean[:,8]])
print(dictionary)       # 5468 unique tokens

corpus = [dictionary.doc2bow(abstract.split()) for abstract in patent_clean[:,8]]
print(len(corpus))      # 3781 corpora



#--- LDA ---#

print('lda staring')
lda_model = models.ldamodel.LdaModel(corpus, num_topics=325, id2word=dictionary, passes=15)
print('lda done')
doc_affili = lda_model.get_document_topics(corpus, minimum_probability=0.05, minimum_phi_value=None, per_word_topics=False)



#--- Coherence ---#


# Compute Coherence Score
coherence_model_lda = models.CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()

print('\nCoherence Score: ', coherence_lda)


'''
#--- Perplexity & Coherence ---#
print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

# Compute Coherence Score
coherence_model_lda = models.coherencemodel.CoherenceModel(model=lda_model, corpus=corpus, coherence='u_mass')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)

### Append document topic distribution and save ###
patent_topicDist = np.empty((np.shape(patent_clean)[0],np.shape(patent_clean)[1]+1), dtype= object)
patent_topicDist[:,:-1] = patent_clean
patent_topicDist.T[9,:] = doc_affili

pd.DataFrame(patent_topicDist).to_csv(directory + 'patent_topicDist.csv', index=None)
'''


### Save topics ###
topics = lda_model.print_topics(num_topics= -1, num_words=8)
topics_arr = np.array(topics)

#print('Number of Topics: ', 325, '\n', topics_arr, '\n\n\n')

pd.DataFrame(topics_arr).to_csv(directory + 'patent_topics.csv', index=None)


### Visualization ###

''' # requires IPython 1.0+
pyLDAvis.enable_notebook() 
vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary=lda_model.id2word)
vis
'''