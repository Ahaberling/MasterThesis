import re
import nltk
import pandas as pd
import numpy as np



#--- Initialization ---#

directory = 'D:/Universitaet Mannheim/MMDS 7. Semester/Master Thesis/Outline/Data/Cleaning Robots/'

patent = pd.read_csv(directory + 'cleaning_robot_EP_patents.csv', quotechar='"', skipinitialspace=True)
patent = patent.to_numpy()



#--- Overview ---#

#print(patent.columns)              # pat_publn_id, publn_auth, publn_nr, publn_date, publn_claims, publn_title, publn_abstract, nb_IPC
#print(np.shape(patent))            # (3844, 8)



#--- Patent investigation ---#

# Identifying german and france patents and patents including the term 'robot' or 'clean'
# Idea: Data is said to be sampled regarding cleaning robots. If these terms occur in every patent,
# then they might as well be excluded from Topic Modelling

check_list_ger = []
check_list_fr = []
check_list_robot = []
check_list_clean = []

for i in range(len(patent[:,6])):

   # German abstract check
    regexp = re.compile(r'\sein')
    if regexp.search(patent[i,6]):
        check_list_ger.append(i)

   # France abstract check
    regexp = re.compile(r'\sune\s')
    if regexp.search(patent[i,6]):
        check_list_fr.append(i)

    # 'robot' abstract check
    regexp = re.compile('robot')
    if regexp.search(patent[i, 6]):
        check_list_robot.append(i)

   # 'clean' abstract check
    regexp = re.compile('clean')
    if regexp.search(patent[i,6]):
        check_list_clean.append(i)

#print(len(removal_list_ger))    #   48/3844 German patents
#print(len(removal_list_fr))     #   15/3844 France patents

#print(len(check_list_robot))    # 3844/3844 patents including 'robot'
#print(len(check_list_clean))    #  398/3844 patents including 'clean'
                                 # todo maybe 'clean' should not be removed in text preprop, ask about
                                 # sampling method -> better sampling necessary?



#--- Removing non-english patents ---#

# Idea: non-english patents bias topic modeling based on (mostly) english abstracts

removal_list_all = []
removal_list_all.append(check_list_ger)
removal_list_all.append(check_list_fr)
removal_list_all = [item for sublist in removal_list_all for item in sublist]

patent = np.delete(patent, removal_list_all, 0)

#print(len(removal_list_all))    # 48 + 15 = 63 abstracts are non-english and removed 63/3844
#print(len(patent))              # 3844 - 63 = 3781 patents remaining



#--- Preparing abstracts for Topic Modelling ---#

### New array ###
patent_cleanAbs = np.empty((np.shape(patent)[0],np.shape(patent)[1]+1), dtype = object) # todo revisite and decide on dtype
patent_cleanAbs[:,:-1] = patent
#print(np.shape(patent_cleanAbs))   # (3844, 9)

### Initialization ###
nltk.download('punkt')              # tokenizer
nltk.download('stopwords')          # stopwords filter
stemmer = nltk.PorterStemmer()

def process_text(text):

    text = re.sub('[^A-Za-z]', ' ', text.lower())   # Make all the strings lowercase and remove non alphabetic characters
    text = re.sub('\s[A-Za-z]\s', ' ', text)        # Remove single letters like x, y, z

    tokenized_text = nltk.word_tokenize(text)       # Tokenizing the text

    # Define custom filter
    custom_filter = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve',
                     'first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth', 'eleventh', 'twelfth',
                     'i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x', 'xi', 'xii']

    # Apply custom filter
    clean_text = [word for word in tokenized_text if word not in custom_filter]

    # Remove nltk-english-stopwords and stem each word to its root
    clean_text = [
        stemmer.stem(word) for word in clean_text if word not in nltk.corpus.stopwords.words('english')
    ]

    # Define steemed filter
    steemed_filter = ['robot']      # adjust if smapling changes and 'clean' is found in every abstract

    # Apply steemed filter
    clean_text = [word for word in clean_text if word not in steemed_filter]

    return clean_text



### Applying defined function on every abstract ###
abstracts_clean = np.array([[]])
i = 0

for abst in patent_cleanAbs.T[6]:
    abstracts_clean = np.append(abstracts_clean, ' '.join(process_text(abst)))
    i = i+1
    if i % 100 == 0:
        print(i, ' / ', len(patent_cleanAbs.T[6]))

    # Small subsample:
    #if i >= 3:
        #break
print(len(patent_cleanAbs.T[6]), ' / ', len(patent_cleanAbs.T[6]))

# Small subsample:
#patent_cleanAbs = patent_cleanAbs[0:3,]


### Append cleaned abstracts and save ###
patent_cleanAbs.T[8:,] = abstracts_clean
pd.DataFrame(patent_cleanAbs).to_csv(directory +'clean_patents.csv', index=None)
