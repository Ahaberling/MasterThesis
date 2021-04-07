import re
import nltk
import pandas as pd
from gensim import corpora, models


pd.set_option('display.max_columns', None)

#patent_clean = pd.read_csv(r'D:\Universitaet Mannheim\MMDS 7. Semester\Master Thesis\Outline\Data\Cleaning Robots\cleaning_robot_EP_patents_cleanAbstract.csv', quotechar='"', skipinitialspace=True)
patent_clean = pd.read_csv(r'D:\Universitaet Mannheim\MMDS 7. Semester\Master Thesis\Outline\Data\Cleaning Robots\cleaning_robot_EP_patents_cleanAbstract_noComma.csv', quotechar='"', skipinitialspace=True)
print(patent_clean['publn_abstract_clean'][0])



dictionary = corpora.Dictionary([d.split() for d in patent_clean.publn_abstract_clean])
print(dictionary)
#print(dictionary[100])



patent_cleanNP = patent_clean['publn_abstract_clean'].to_numpy()
print(patent_cleanNP[0])
test = ['method least one axi robot control unit control', 'compon part least one marker posit marker detect sensor actual valu']

#corpus = [dictionary.doc2bow(abstract) for abstract in patent_clean.publn_abstract_clean]
#corpus = [dictionary.doc2bow(patent_clean['publn_abstract_clean'][0])]
corpus = [dictionary.doc2bow(test)]
print(corpus)

'''
model = models.ldamodel.LdaModel(corpus, num_topics=3, id2word=dictionary, passes=15)

topics = model.print_topics(num_words=3)
for topic in topics:
    print(topic)
    '''