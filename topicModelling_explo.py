import re
import nltk
import pandas as pd

#from gensim import corpora

pd.set_option('display.max_columns', None)

patent = pd.read_csv(r'D:\Universitaet Mannheim\MMDS 7. Semester\Master Thesis\Outline\Data\Cleaning Robots\cleaning_robot_EP_patents.csv', quotechar='"', skipinitialspace=True)

#print(patent.publn_abstract[0])
#print(type(patent.publn_abstract))

# We need this dataset in order to use the tokenizer
nltk.download('punkt')
from nltk.tokenize import word_tokenize

# Also download the list of stopwords to filter out
nltk.download('stopwords')
from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def process_text(text):
    # Make all the strings lowercase and remove non alphabetic characters
    text = re.sub('[^A-Za-z]', ' ', text.lower())

    # Tokenize the text; this is, separate every sentence into a list of words
    # Since the text is already split into sentences you don't have to call sent_tokenize
    tokenized_text = word_tokenize(text)

    # Remove the stopwords and stem each word to its root
    clean_text = [
        stemmer.stem(word) for word in tokenized_text
        if word not in stopwords.words('english')
    ]

    # Remember, this final output is a list of words
    return clean_text




'''
patent['publn_abstract_clean'] = 0
i = 0
print(patent)

for abstract in patent.publn_abstract:
    #print(process_text(abstract))
    patent.publn_abstract_clean[i] = process_text(abstract)
    i = i+1
    if i % 100 == 0:
        print(i)

print(patent)'''