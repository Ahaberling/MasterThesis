if __name__ == '__main__':

    # --- Import Libraries ---#
    print('\n#--- Import Libraries ---#\n')

    # Utility
    import os
    import tqdm
    import itertools

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



    # --- Patent Cleaning ---#
    print('\n#--- Patent Cleaning ---#\n')

    from utilities.my_text_utils import PatentCleaning

    # Remove non-english patents
    patent_raw, number_removed_patents_ger = PatentCleaning.remove_foreign_patents(patent_raw, language='ger', count=True)
    patent_raw, number_removed_patents_fr = PatentCleaning.remove_foreign_patents(patent_raw, language='fr', count=True)

    print('Number of all patents: ', len(patent_raw))
    print('Number of german patents removed: ', number_removed_patents_ger)
    print('Number of french patents removed: ', number_removed_patents_fr)

    # Count patents with term
    term_clean, number_abstracts_term_clean = PatentCleaning.count_abstracts_with_term(patent_raw, term='clean')
    term_robot, number_abstracts_robot = PatentCleaning.count_abstracts_with_term(patent_raw, term='robot')

    print('Number abstracts containing', term_clean, ': ', number_abstracts_term_clean)
    print('Number abstracts containing', term_robot, ': ', number_abstracts_robot)



# --- Longitudinal descriptives ---#
    print('\n# --- Overview ---#\n')

    patent_time = patent_raw[:, 3].astype('datetime64')

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

