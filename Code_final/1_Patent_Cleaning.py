if __name__ == '__main__':



    #--- Import Libraries ---#
    print('\n#--- Import Libraries ---#\n')

    # Utility
    import os
    import statistics

    # Data handling
    import numpy as np
    import pandas as pd
    import pickle as pk

    # Visualization
    import matplotlib.pyplot as plt

    # Custom functions
    from utilities.Data_Preparation_utils import PatentCleaning
    from utilities.Data_Preparation_utils import TransformationMisc



    #--- Initialization ---#
    print('\n#--- Initialization ---#\n')

    # directory
    path = 'D:/'

    # terms to be searched in the raw patent abstracts
    terms_toBeSearched = ['robot', 'clean']

    # Configurations for draw_stochastic_IPC_sample. The function draws samples from the merge
    # patents_raw and patents_IPC data set. These samples can be investigated to identify unfitting
    # IPC section/classes/subclasses/... or unfitting patents
    level = 'class'
    searched_ipc = 'A61'
    sample_size = 3

    # Adapt ids_unfitting_patents to remove identified unfitting patents and their IPCs from the data set
    ids_unfitting_patents = []



    #--- Import Data ---#
    print('\n#--- Import Data ---#\n')
    os.chdir(path)

    patents_raw = pd.read_csv('cleaning_robot_EP_patents.csv', quotechar='"', skipinitialspace=True)
    patents_raw = patents_raw.to_numpy()

    patents_IPC = pd.read_csv('cleaning_robot_EP_patents_IPC.csv', quotechar='"', skipinitialspace=True)
    patents_IPC = patents_IPC.to_numpy()

    print('Exemplary, patents_raw row: \n', patents_raw[0], '\n')
    print('Variables of interest:')
    print('Patent id: ', patents_raw[0][0])
    print('Publication date: ', patents_raw[0][3])
    print('Patent abstract: ', patents_raw[0][6], '\n')

    print('Exemplary, patents_IPC row: \n', patents_IPC[0], '\n')
    print('Variables of interest:')
    print('Patent id: ', patents_IPC[0][0])
    print('Patent IPC: ', patents_IPC[0][1], '\n')



    #--- Patent Cleaning - Language ---#
    print('\n#--- Patent Cleaning - Language ---#\n')

    print('Number of patents before language cleaning: ', len(patents_raw))

    # Remove non-english patents
    patents_raw, number_removed_patents_ger = PatentCleaning.remove_foreign_patents(patents_raw, language='ger', count=True)
    patents_english, number_removed_patents_fr = PatentCleaning.remove_foreign_patents(patents_raw, language='fr', count=True)

    print('Number of german patents removed: ', number_removed_patents_ger, 'of', len(patents_english))
    print('Number of french patents removed: ', number_removed_patents_fr, 'of', len(patents_english))

    # Count patents with term
    for term in terms_toBeSearched:
        number_abstracts_with_term = PatentCleaning.count_abstracts_with_term(patents_english, term=term)
        print('Number abstracts containing', term, ': ', number_abstracts_with_term, 'of', len(patents_english))



    #--- Appending IPCs ---#
    print('\n#--- Appending IPCs ---#\n')
    # Finding which IPCs are present in the data and to what extend.

    # Check if patent ids in patents_english are unique
    val = np.unique(patents_english[:, 0])
    if len(val) != len(patents_english):
        raise Exception("Error: patents_english contains non-unqiue patents")

    # The patents_IPC data set contains the IPCs of more patents then just the one contained in the patents_english data set
    # patents_IPC is reduced to only the patents matching with patents_english (via patent id)
    patents_IPC_clean = [patent[0] for patent in patents_IPC if patent[0] in patents_english[:, 0]]
    val, count = np.unique(patents_IPC_clean, return_counts=True)

    # Merging patents_english and patents_IPC
    new_space_needed = max(count) * 3       # x * 3 = x3 additional columns are needed in the new array

    patents_english_IPC = np.empty((np.shape(patents_english)[0], np.shape(patents_english)[1] + new_space_needed), dtype=object)
    patents_english_IPC[:, :-new_space_needed] = patents_english
    patents_english_IPC = TransformationMisc.fill_with_IPC(patents_english_IPC, patents_IPC, new_space_needed)


    #--- Investigating IPCs ---#
    print('\n#--- Investigating IPCs ---#\n')

    # Stochastic investigation of IPC fitness
    print('Randomly sampled IPC: ')
    print("Structure: ('IPC', ['patent id', 'title', 'abstract')")

    test_sample = PatentCleaning.draw_stochastic_IPC_sample(patents_english_IPC, level, searched_ipc, sample_size)

    for i in range(len(test_sample)):
        print(test_sample[i])



    #--- Excluding unfitting patents based on IPC ---#
    print('\n#--- Excluding unfitting patents based on IPC ---#\n')

    patents_english_cleaned = patents_english
    patents_english_IPC_cleaned = patents_english_IPC

    for id in ids_unfitting_patents:

        position = np.where(patents_english_cleaned[:, 0] == id)
        patents_english_cleaned = np.delete(patents_english_cleaned, position, 0)

        position2 = np.where(patents_english_IPC_cleaned[:, 0] == id)
        patents_english_IPC_cleaned = np.delete(patents_english_IPC_cleaned, position2, 0)



    #--- Generate IPC Descriptives ---#
    print('\n#--- Generate IPC Descriptives ---#\n')

    # Get only those patents_IPC entries that have a match in patents_english_IPC_cleaned (via patent id)
    patents_IPC_clean = [i[0] for i in patents_IPC if i[0] in patents_english_IPC_cleaned[:, 0]]
    val, count = np.unique(patents_IPC_clean, return_counts=True)

    print('Average number of IPCs a patent has: ', np.mean(count))
    print('Median number of IPCs a patent has: ', np.median(count))
    print('Mode number of IPCs a patent has: ', statistics.mode(count))
    print('Max number of IPCs a patent has: ', max(count))
    print('Min number of IPCs a patent has: ', min(count), '\n')

    # Get distribution of IPCs on different levels
    ipc_list_group = []
    ipc_list_subClass = []
    ipc_list_class = []
    ipc_list_section = []

    for patent in patents_english_IPC_cleaned:
        for ipc in range(0, len(patent[8:]), 3):
            if patent[8:][ipc] != None:
                ipc_list_group.append(patent[8:][ipc])
                ipc_list_subClass.append(patent[8:][ipc][0:4])
                ipc_list_class.append(patent[8:][ipc][0:3])
                ipc_list_section.append(patent[8:][ipc][0:1])

    print('Number of all classifications over all patents: ', len(ipc_list_group))

    ipc_list_group_unique = np.unique(ipc_list_group)
    ipc_list_subClass_unique = np.unique(ipc_list_subClass)
    ipc_list_class_unique = np.unique(ipc_list_class)
    ipc_list_section_unique = np.unique(ipc_list_section)
    print('Number of unique IPCs on group level', len(ipc_list_group_unique))
    print('Number of unique IPCs on subclass level', len(ipc_list_subClass_unique))
    print('Number of unique IPCs on class level', len(ipc_list_class_unique))
    print('Number of unique IPCs on section level', len(ipc_list_section_unique))
    print('\n')

    # Visualization on section level
    val, count = np.unique(ipc_list_section, return_counts=True)
    print('IPC sections present in the data set: ', val)
    print('Distribution of these sections: ', count)

    fig, ax = plt.subplots(1, 1)
    ax.hist(sorted(ipc_list_section), bins=8, color='darkred')
    plt.xlabel("International Patent Classification (IPC) - Sections")
    plt.ylabel("Frequency")
    plt.savefig('IPC_distribution.png')
    plt.close()



    #--- Generate Longitudinal Descriptives ---#
    print('\n#--- Generate Longitudinal Descriptives ---#\n')

    patent_time = patents_english_cleaned[:, 3].astype('datetime64')

    print('Earliest publication day: ', min(patent_time))
    print('Latest publication day: ', max(patent_time))

    max_timeSpan = int((max(patent_time) - min(patent_time)) / np.timedelta64(1, 'D'))
    print('Size of publication time span in days: ', max_timeSpan)

    val, count = np.unique(patent_time, return_counts=True)
    print('Number of days on which publications were made: ', len(val))
    print('Average number of days between publication dates: ', max_timeSpan / len(val))

    # Visualization of publications per time
    fig, ax = plt.subplots(1, 1)
    ax.hist(patent_time, bins=198, color='darkblue')
    plt.xlabel("Publication time span")
    plt.ylabel("Number of patents published")
    plt.savefig('hist_publications.png')
    plt.close()

    #--- Save data set ---#
    print('\n#--- Save data set ---#\n')

    # The data set saved and utilized in the next code file is the cleaned patent_raw.
    # The IPCs are added again in later steps. This is somewhat redundant, but not yet adjusted.

    filename = 'patents_english_cleaned'
    outfile = open(filename, 'wb')
    pk.dump(patents_english_cleaned, outfile)
    outfile.close()
