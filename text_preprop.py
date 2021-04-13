import re
import nltk
import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

patent = pd.read_csv(r'D:\Universitaet Mannheim\MMDS 7. Semester\Master Thesis\Outline\Data\Cleaning Robots\cleaning_robot_EP_patents.csv', quotechar='"', skipinitialspace=True)
#print(patent.columns)

# pat_publn_id, publn_auth, publn_nr, publn_date, publn_claims, publn_title, publn_abstract, nb_IPC
patent = patent.to_numpy()
#print(np.shape(patent))            # (3844, 8)

patent_cleanAbs = np.empty((np.shape(patent)[0],np.shape(patent)[1]+1), dtype = object)
#todo is this dtype inefficient? should I choose something else? is it even possible,or should I just transfer the dtype later on, when dealing with columns seperatly?

patent_cleanAbs[:,:-1] = patent
print(np.shape(patent_cleanAbs))   #(3844, 9)

nltk.download('punkt')              # tokenizer
nltk.download('stopwords')          # stopwords filter

stemmer = nltk.PorterStemmer()

def process_text(text):
    # Make all the strings lowercase and remove non alphabetic characters
    text = re.sub('[^A-Za-z]', ' ', text.lower())

    # Tokenize the text; this is, separate every sentence into a list of words
    # Since the text is already split into sentences you don't have to call sent_tokenize
    tokenized_text = nltk.word_tokenize(text)

    # Remove the stopwords and stem each word to its root
    clean_text = [
        stemmer.stem(word) for word in tokenized_text
        if word not in nltk.corpus.stopwords.words('english')
    ]

    # Remember, this final output is a list of words
    return clean_text

#todo Steeming sucks (i think?)

#patent['publn_abstract_clean'] = 0
#i = 0
#print(patent)

'''
N = 10
a = np.random.rand(N,N)
print(a)
print(np.shape(a))
b = np.zeros((N,N+1))
print(b)
print(np.shape(b))
b[:,:-1] = a
print(b)
print(np.shape(b))
'''

print(patent)
print('\n ------------------------------------------------------------------------------------ \n ')
print(patent.T)
print('\n ------------------------------------------------------------------------------------ \n ')
print(patent_cleanAbs)
print('\n ------------------------------------------------------------------------------------ \n ')
print(patent_cleanAbs.T)
print('\n ------------------------------------------------------------------------------------ \n ')

'''
#test = np.random.rand(3,3)
test = np.array([[3, 3, 3],[3, 3, 3],[3, 3, 3]])
print(test)
#v = test[:, 0] ** 2
v = test[:, 0] ** 2
print(v)

v =
'''

abstracts = patent_cleanAbs.T[6]
#print(abstracts)

abstracts_clean = np.array([[]])
i = 0
for abst in abstracts:
    #print(abst)
    #print(process_text(abst))
    #patent.publn_abstract_clean[i] = process_text(abstract)
    #abstracts_clean = " ".join(process_text(abst))
    #np.append(abstracts_clean, " ".join(process_text(abst)))
    abstracts_clean = np.append(abstracts_clean, " ".join(process_text(abst)))
    i = i+1
    if i % 100 == 0:
        print(i, " / ", len(abstracts))
    #if i >= 3:
        #break

#print(patent.publn_abstract_clean)

#print(abstracts_clean)
#print('\n ------------------------------------------------------------------------------------ \n ')

#patent_cleanAbs = patent_cleanAbs[0:3,]

patent_cleanAbs.T[8:,] = abstracts_clean

#print(patent_cleanAbs)
#print('\n ------------------------------------------------------------------------------------ \n ')
#print(patent_cleanAbs.T)

#patent.to_csv(r'D:\Universitaet Mannheim\MMDS 7. Semester\Master Thesis\Outline\Data\Cleaning Robots\cleaning_robot_EP_patents_cleanAbstract.csv')
#patent.to_csv(r'D:\Universitaet Mannheim\MMDS 7. Semester\Master Thesis\Outline\Data\Cleaning Robots\cleaning_robot_EP_patents_cleanAbstract_noComma.csv')
#patent.to_csv(r'D:\Universitaet Mannheim\MMDS 7. Semester\Master Thesis\Outline\Data\Cleaning Robots\cleaning_robot_EP_patents_cleanAbstract_noComma_noKlammer.csv')
#patent.to_csv(r'D:\Universitaet Mannheim\MMDS 7. Semester\Master Thesis\Outline\Data\Cleaning Robots\cleaning_robot_EP_patents_cleanAbstract_noKlammer.csv')
#np.savetxt(r'D:\Universitaet Mannheim\MMDS 7. Semester\Master Thesis\Outline\Data\Cleaning Robots\cleaning_robot_EP_patents_07_04.csv', patent_cleanAbs, fmt='%s', delimiter=",")

pd.DataFrame(patent_cleanAbs).to_csv(r'D:\Universitaet Mannheim\MMDS 7. Semester\Master Thesis\Outline\Data\Cleaning Robots\cleaning_robot_EP_patents_07_04.csv', index=None)
