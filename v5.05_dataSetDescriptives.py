if __name__ == '__main__':

    # --- Import Libraries ---#
    print('\n#--- Import Libraries ---#\n')

    # Utility
    import os
    import tqdm
    import itertools
    import statistics

    # Data handling
    import numpy as np
    import pandas as pd

    # NLP
    import nltk
    import spacy
    import gensim.corpora as corpora
    import gensim.models as gensim_models



    # --- Initialization ---#
    print('\n#--- Initialization ---#\n')

    # Import data
    os.chdir('D:/Universitaet Mannheim/MMDS 7. Semester/Master Thesis/Outline/Data/Cleaning Robots')

    patent_raw = pd.read_csv('cleaning_robot_EP_patents.csv', quotechar='"', skipinitialspace=True)
    patent_raw = patent_raw.to_numpy()

    patent_IPC = pd.read_csv('cleaning_robot_EP_patents_IPC.csv', quotechar='"', skipinitialspace=True)
    patent_IPC = patent_IPC.to_numpy()


    # --- Patent Cleaning - Language ---#
    print('\n#--- Patent Cleaning - Language ---#\n')

    from utilities.my_text_utils import PatentCleaning

    # Remove non-english patents
    patent_raw, number_removed_patents_ger = PatentCleaning.remove_foreign_patents(patent_raw, language='ger', count=True)
    patent_english, number_removed_patents_fr = PatentCleaning.remove_foreign_patents(patent_raw, language='fr', count=True)

    print('Number of all patents: ', len(patent_english))
    print('Number of german patents removed: ', number_removed_patents_ger)
    print('Number of french patents removed: ', number_removed_patents_fr)

    # Count patents with term
    term_clean, number_abstracts_term_clean = PatentCleaning.count_abstracts_with_term(patent_english, term='clean')
    term_robot, number_abstracts_robot = PatentCleaning.count_abstracts_with_term(patent_english, term='robot')

    print('Number abstracts containing', term_clean, ': ', number_abstracts_term_clean)
    print('Number abstracts containing', term_robot, ': ', number_abstracts_robot)

    # --- Patent Cleaning - IPCs ---#
    print('\n#--- Patent Cleaning - IPCs ---#\n')

    # Check if patent ids in patent_english are unique
    val, count = np.unique(patent_english[:, 0], return_counts=True)
    if len(val) != len(patent_english):
        raise Exception("Error: patent_english contains non-unqiue patents")

    # Get only those patent_IPC entries that have a match in patent_english (via patent id)
    patent_IPC_clean = [i[0] for i in patent_IPC if i[0] in patent_english[:, 0]]
    val, count = np.unique(patent_IPC_clean, return_counts=True)

    # Merging patent_english and patent_IPC
    new_space_needed = max(count) * 3       # x * 3 = x3 additional columns are needed in the new array
    patent_english_IPC = np.empty((np.shape(patent_english)[0], np.shape(patent_english)[1] + new_space_needed), dtype=object)
    patent_english_IPC[:, :-new_space_needed] = patent_english

    from utilities.my_transform_utils import Transf_misc
    patent_english_IPC = Transf_misc.fill_with_IPC(patent_english_IPC, patent_IPC, new_space_needed)

    # Distribution of IPCs on different levels
    ipc_list_group = []
    ipc_list_subClass = []
    ipc_list_class = []
    ipc_list_sec = []

    for patent in patent_english_IPC:
        for ipc in range(0, len(patent[8:]), 3):
            if patent[8:][ipc] != None:
                ipc_list_group.append(patent[8:][ipc])
                ipc_list_subClass.append(patent[8:][ipc][0:4])
                ipc_list_class.append(patent[8:][ipc][0:3])
                ipc_list_sec.append(patent[8:][ipc][0:1])

    print('Number of all classifications over all patents: ', len(ipc_list_group))
    ipc_list_group = np.unique(ipc_list_group)
    print('Number of unique IPCs on group level', len(ipc_list_group))
    print('Number of unique IPCs on subclass level', len(ipc_list_subClass))
    print('Number of unique IPCs on class level', len(ipc_list_class))
    print('Number of unique IPCs on section level', len(ipc_list_sec))
    print('\n')

    val, count = np.unique(ipc_list_sec, return_counts=True)
    print(val)
    print(count)

    # From a laymans point of view all sections can somewhat resonably be used to categorize cleaning robot patents.
    # However, Category 'D' seems most unfitting:
    # IPCs with section 'D' seem unfitting -> closer investigation
    print('Closer Investigation into patents with "D" IPC section: \n')
    for patent in patent_english_IPC:
        for ipc in range(0, len(patent[8:]), 3):
            if patent[8:][ipc] != None:
                if patent[8:][ipc][0] == 'D':
                    print(patent, '\n')

    # Two patents exhibiting 'D'.
    # Inclusion of patent with id 55163657 seems arguable. Patent remains in Dataset following a conservative approach
    # Patent with id 365546649 is (broadly) concerned with dishwashers and seems unfitting for the overall case of cleaning robots.

    # Exclude 365546649
    position = np.where(patent_english[:,0] == 365546649)
    patent_cleaned = np.delete(patent_english, position, 0)

    # Descriptives of cleaned IPC distribution

    # Get only those patent_IPC entries that have a match in patent_english (via patent id)
    patent_IPC_clean = [i[0] for i in patent_IPC if i[0] in patent_cleaned[:, 0]]
    val, count = np.unique(patent_IPC_clean, return_counts=True)

    # Descriptives about IPC distribution in patent_english
    print('Max number of IPCs a patent has: ', max(count))
    print('Min number of IPCs a patent has: ', min(count))
    print('Average number of IPCs a patent has: ', np.mean(count))
    print('Median number of IPCs a patent has: ', np.median(count))
    print('Mode number of IPCs a patent has: ', statistics.mode(count))



    ipc_list_sec = []
    for patent in patent_cleaned:
        for ipc in range(0, len(patent[8:]), 3):
            if patent[8:][ipc] != None:
                ipc_list_sec.append(patent[8:][ipc][0:1])

    # Visulization without odd cases
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1)
    ax.hist(sorted(ipc_list_sec), bins=8, color='darkred')
    plt.xlabel("International Patent Classification - Sections")
    plt.ylabel("Number of Patents")
    plt.show()
    plt.close()




    # --- Longitudinal descriptives ---#
    print('\n# --- Overview ---#\n')

    patent_time = patent_english[:, 3].astype('datetime64')

    print('Earliest day with publication: ', min(patent_time))  # earliest day with publication 2001-08-01
    print('Latest day with publication: ', max(patent_time))  # latest  day with publication 2018-01-31

    max_timeSpan = int((max(patent_time) - min(patent_time)) / np.timedelta64(1, 'D'))
    print('Days inbetween: ', max_timeSpan)  # 6027 day between earliest and latest publication

    val, count = np.unique(patent_time, return_counts=True)
    print('Number of days with publications: ', len(val))  # On 817 days publications were made
    # -> on average every 7.37698898409 days a patent was published

    # number of months: 5 + 16*12 + 1 = 198

    import random
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    # generate some random data (approximately over 5 years)
    data = [float(random.randint(1271517521, 1429197513)) for _ in range(1000)]


    # convert the epoch format to matplotlib date format
    mpl_data = mdates.epoch2num(data)

    print(mpl_data)
    print(patent_time)

    # plot it
    fig, ax = plt.subplots(1, 1)
    ax.hist(patent_time, bins=198, color='darkblue')
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(locator))
    #plt.title("Histogram: Monthly number of patent publications")
    plt.xlabel("Publication time span")
    plt.ylabel("Number of patents published")

    os.chdir('D:/Universitaet Mannheim/MMDS 7. Semester/Master Thesis/Outline/Plots')

    plt.savefig('hist_publications.png')
    #plt.show()

