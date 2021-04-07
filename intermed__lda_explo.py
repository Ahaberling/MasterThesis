import re
import nltk
import pandas as pd
import numpy as np
from gensim import corpora, models

import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)

#patent_clean = pd.read_csv(r'D:\Universitaet Mannheim\MMDS 7. Semester\Master Thesis\Outline\Data\Cleaning Robots\cleaning_robot_EP_patents_cleanAbstract.csv', quotechar='"', skipinitialspace=True)
#patent_clean = pd.read_csv(r'D:\Universitaet Mannheim\MMDS 7. Semester\Master Thesis\Outline\Data\Cleaning Robots\cleaning_robot_EP_patents_cleanAbstract_noComma.csv', quotechar='"', skipinitialspace=True)
patent_clean = pd.read_csv(r'D:\Universitaet Mannheim\MMDS 7. Semester\Master Thesis\Outline\Data\Cleaning Robots\cleaning_robot_EP_patents_cleanAbstract_noComma_noKlammer.csv', quotechar='"', skipinitialspace=True)
print(patent_clean['publn_abstract_clean'])



dictionary = corpora.Dictionary([d.split() for d in patent_clean.publn_abstract_clean])
print(dictionary)
#print(dictionary[100])



patent_cleanNP = patent_clean['publn_abstract_clean'].to_numpy()
#print(np.shape(patent_cleanNP))
print(patent_cleanNP)
#print(patent_cleanNP[0])
test = ['devic push open unit pierc movement bottom push open process reservoir contain small part remain head piec push open unit camera detect shape posit small part support head piec direct robot collect detect small part head piec includ differ contour small part differ geometri head piec vacuum support separ small part independ claim also includ method separ small part' 'method involv input foil geometri data process program transmit data foil manufactur data carrier email websit data data process program read pack plane detect foil portion cut blank manual robot produc packag transport add program util add program integr data process program custom cut foil portion connect independ claim also includ articl manufactur accord method pack foil']



#corpus = [dictionary.doc2bow(abstract) for abstract in patent_clean.publn_abstract_clean]
#corpus = [dictionary.doc2bow(patent_clean.publn_abstract_clean)]
#corpus = [dictionary.doc2bow(patent_clean['publn_abstract_clean'][0])]
corpus = [dictionary.doc2bow(abstract.split()) for abstract in patent_cleanNP]
#print(corpus)

print(len(patent_cleanNP))
print(len(corpus))
print(corpus[0])


model = models.ldamodel.LdaModel(corpus, num_topics=10, id2word=dictionary, passes=15)

doc_affili = model.get_document_topics(corpus, minimum_probability=None, minimum_phi_value=None, per_word_topics=False)
print(doc_affili)
#print(doc_affili[3843])
#print(doc_affili[3843][0])
#print(doc_affili[3844])
#print(len(doc_affili))

'''
topics = model.print_topics(num_topics= -1, num_words=8)
#for topic in topics:
    #print(topic)
print(topics[0])
print(len(topics))

'''

patent_clean['patent_topic_dist'] = 0

i = 0

for topicDist in doc_affili:
    #print(topicDist)
    #patent.publn_abstract_clean[i] = process_text(abstract)
    patent_clean.patent_topic_dist[i] = topicDist
    i = i+1
    if i % 100 == 0:
        print(i, " / ", len(patent_clean.patent_topic_dist))
    #if i >= 100:
        #break

print(patent_clean)
