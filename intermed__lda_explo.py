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
#patent_clean = pd.read_csv(r'D:\Universitaet Mannheim\MMDS 7. Semester\Master Thesis\Outline\Data\Cleaning Robots\cleaning_robot_EP_patents_cleanAbstract_noComma_noKlammer.csv', quotechar='"', skipinitialspace=True)
patent_clean = pd.read_csv(r'D:\Universitaet Mannheim\MMDS 7. Semester\Master Thesis\Outline\Data\Cleaning Robots\cleaning_robot_EP_patents_07_04.csv', quotechar='"', skipinitialspace=True)
#print(np.shape(patent_clean))
#print(patent_clean)

patent_clean = patent_clean.to_numpy()

#print(patent_clean.T)
#print(patent_clean[:,8])

dictionary = corpora.Dictionary([d.split() for d in patent_clean[:,8]])
print(dictionary)
#print(dictionary[100])



#patent_cleanNP = patent_clean['publn_abstract_clean'].to_numpy()
#print(np.shape(patent_cleanNP))
#print(patent_cleanNP)
#print(patent_cleanNP[0])
#test = ['devic push open unit pierc movement bottom push open process reservoir contain small part remain head piec push open unit camera detect shape posit small part support head piec direct robot collect detect small part head piec includ differ contour small part differ geometri head piec vacuum support separ small part independ claim also includ method separ small part' 'method involv input foil geometri data process program transmit data foil manufactur data carrier email websit data data process program read pack plane detect foil portion cut blank manual robot produc packag transport add program util add program integr data process program custom cut foil portion connect independ claim also includ articl manufactur accord method pack foil']



#corpus = [dictionary.doc2bow(abstract) for abstract in patent_clean.publn_abstract_clean]
#corpus = [dictionary.doc2bow(patent_clean.publn_abstract_clean)]
#corpus = [dictionary.doc2bow(patent_clean['publn_abstract_clean'][0])]
corpus = [dictionary.doc2bow(abstract.split()) for abstract in patent_clean[:,8]]
#print(corpus)

#print(len(patent_clean[:,8]))
print(len(corpus))
print(corpus[0])


model = models.ldamodel.LdaModel(corpus, num_topics=300, id2word=dictionary, passes=15)

doc_affili = model.get_document_topics(corpus, minimum_probability=0.05, minimum_phi_value=None, per_word_topics=False)
#print(doc_affili)
#for i in doc_affili:
    #print(i)

#print(doc_affili[3843])
#print(doc_affili[3843][0])
#print(doc_affili[3844])
#print(len(doc_affili))



patent_clean_topicDist = np.empty((np.shape(patent_clean)[0],np.shape(patent_clean)[1]+1), dtype= object)
patent_clean_topicDist[:,:-1] = patent_clean


patent_clean_topicDist.T[9,:] = doc_affili

#print(patent_clean_topicDist.T)

pd.DataFrame(patent_clean_topicDist).to_csv(r'D:\Universitaet Mannheim\MMDS 7. Semester\Master Thesis\Outline\Data\Cleaning Robots\cleaning_robot_EP_patents_07_04_topicDist.csv', index=None)


topics = model.print_topics(num_topics= -1, num_words=15)
#for topic in topics:
    #print(topic)
#print(topics[0])
#print(len(topics))

topics_arr = np.array(topics)
#print(topics_arr)
pd.DataFrame(topics_arr).to_csv(r'D:\Universitaet Mannheim\MMDS 7. Semester\Master Thesis\Outline\Data\Cleaning Robots\cleaning_robot_EP_patents_07_04_topics.csv', index=None)



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
'''