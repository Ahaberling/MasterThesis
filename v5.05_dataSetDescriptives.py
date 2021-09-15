if __name__ == '__main__':

    # --- Import Libraries ---#
    print('\n#--- Import Libraries ---#\n')

    # Utility
    import os
    import statistics

    # Data handling
    import numpy as np
    import pandas as pd
    import pickle as pk



    # --- Initialization ---#
    print('\n#--- Initialization ---#\n')

    # Import data
    os.chdir('D:/Universitaet Mannheim/MMDS 7. Semester/Master Thesis/Outline/Data/Cleaning Robots')

    patents_raw = pd.read_csv('cleaning_robot_EP_patents.csv', quotechar='"', skipinitialspace=True)
    patents_raw = patents_raw.to_numpy()

    patents_IPC = pd.read_csv('cleaning_robot_EP_patents_IPC.csv', quotechar='"', skipinitialspace=True)
    patents_IPC = patents_IPC.to_numpy()


    # --- Patent Cleaning - Language ---#
    print('\n#--- Patent Cleaning - Language ---#\n')

    from utilities.my_text_utils import PatentCleaning

    print('Number of all patents : ', len(patents_raw))

    # Remove non-english patents
    patents_raw, number_removed_patents_ger = PatentCleaning.remove_foreign_patents(patents_raw, language='ger', count=True)
    patents_english, number_removed_patents_fr = PatentCleaning.remove_foreign_patents(patents_raw, language='fr', count=True)

    print('Number of all english patents: ', len(patents_english))
    print('Number of german patents removed: ', number_removed_patents_ger)
    print('Number of french patents removed: ', number_removed_patents_fr)

    # Count patents with term
    term_clean, number_abstracts_term_clean = PatentCleaning.count_abstracts_with_term(patents_english, term='clean')
    term_robot, number_abstracts_robot = PatentCleaning.count_abstracts_with_term(patents_english, term='robot')

    print('Number abstracts containing', term_clean, ': ', number_abstracts_term_clean)
    print('Number abstracts containing', term_robot, ': ', number_abstracts_robot)



    # --- Patent Cleaning - IPCs ---#
    print('\n#--- Patent Cleaning - IPCs ---#\n')
    # Finding which IPCs are present in the data and to what extend.

    # Check if patent ids in patents_english are unique
    val, count = np.unique(patents_english[:, 0], return_counts=True)
    if len(val) != len(patents_english):
        raise Exception("Error: patents_english contains non-unqiue patents")

    # patents_IPC contains the IPCs of more patents then just the one listed in patents_english
    # -> Get only those patents_IPC entries that have a match in patents_english (via patent id)
    patents_IPC_clean = [patent[0] for patent in patents_IPC if patent[0] in patents_english[:, 0]]
    val, count = np.unique(patents_IPC_clean, return_counts=True)

    # Merging patents_english and patents_IPC
    new_space_needed = max(count) * 3       # x * 3 = x3 additional columns are needed in the new array
    patents_english_IPC = np.empty((np.shape(patents_english)[0], np.shape(patents_english)[1] + new_space_needed), dtype=object)
    patents_english_IPC[:, :-new_space_needed] = patents_english

    from utilities.my_transform_utils import Transf_misc
    patents_english_IPC = Transf_misc.fill_with_IPC(patents_english_IPC, patents_IPC, new_space_needed)

    # Check distribution of IPCs on section level
    ipc_list_section = []
    for patent in patents_english_IPC:
        for ipc in range(0, len(patent[8:]), 3):
            if patent[8:][ipc] != None:
                ipc_list_section.append(patent[8:][ipc][0:1])

    val, count = np.unique(ipc_list_section, return_counts=True)
    print('IPC sections present in the data set: ', val)
    print('Distribution of these sections: ', count)

    # Stochastic investigation
    print('Closer Investigation into patent/s')
    x = PatentCleaning.stochastic_inestigation_IPCs(patents_english_IPC, 'class', 'A61', 10)
    for i in range(len(x)):
        print(x[i]) #, '\n')

    # Two patents exhibiting 'D'.
    # Inclusion of patent with id 55163657 seems arguable. Patent is kept in data set, following a conservative approach.
    # Patent with id 365546649 is (broadly) concerned with dishwashers and seems unfitting for the overall case of cleaning robots.

    # Exclude of patent with id 365546649
    position = np.where(patents_english_IPC[:,0] == 365546649)
    #print(len(patents_english_IPC))
    patents_english_IPC_cleaned = np.delete(patents_english_IPC, position, 0)
    #print(len(patents_english_IPC_cleaned))

    # Revisite IPC distribution with cleaned data + other descriptives:

    # Get only those patents_IPC entries that have a match in patents_english_IPC_cleaned (via patent id)
    patents_IPC_clean = [i[0] for i in patents_IPC if i[0] in patents_english_IPC_cleaned[:, 0]]
    val, count = np.unique(patents_IPC_clean, return_counts=True)

    # Descriptives about IPC distribution in patents_english
    print('Max number of IPCs a patent has: ', max(count))
    print('Min number of IPCs a patent has: ', min(count))
    print('Average number of IPCs a patent has: ', np.mean(count))
    print('Median number of IPCs a patent has: ', np.median(count))
    print('Mode number of IPCs a patent has: ', statistics.mode(count))

    # Distribution of IPCs on different levels
    ipc_list_group = []
    ipc_list_subClass = []
    ipc_list_class = []
    ipc_list_sec = []

    for patent in patents_english_IPC_cleaned:
        for ipc in range(0, len(patent[8:]), 3):
            if patent[8:][ipc] != None:
                ipc_list_group.append(patent[8:][ipc])
                ipc_list_subClass.append(patent[8:][ipc][0:4])
                ipc_list_class.append(patent[8:][ipc][0:3])
                ipc_list_sec.append(patent[8:][ipc][0:1])

    print(len(patents_english_IPC_cleaned))
    print('Number of all classifications over all patents: ', len(ipc_list_group))

    ipc_list_group_unique = np.unique(ipc_list_group)
    ipc_list_subClass_unique = np.unique(ipc_list_subClass)
    ipc_list_class_unique = np.unique(ipc_list_class)
    ipc_list_sec_unique = np.unique(ipc_list_sec)
    print('Number of unique IPCs on group level', len(ipc_list_group_unique))
    print('Number of unique IPCs on subclass level', len(ipc_list_subClass_unique))
    print('Number of unique IPCs on class level', len(ipc_list_class_unique))
    print('Number of unique IPCs on section level', len(ipc_list_sec_unique))
    print('\n')

    # Visualization without odd cases
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1)
    ax.hist(sorted(ipc_list_sec), bins=8, color='darkred')
    plt.xlabel("International Patent Classification (IPC) - Sections")
    plt.ylabel("Number of IPCs")

    os.chdir('D:/Universitaet Mannheim/MMDS 7. Semester/Master Thesis/Outline/Plots')
    plt.savefig('IPC_distribution.png')
    os.chdir('D:/Universitaet Mannheim/MMDS 7. Semester/Master Thesis/Outline/Data/Cleaning Robots')

    plt.close()

    # Patent with id 365546649  is not only exluded for patents_english_IPC but also for patents_english.
    # The former is only created here for the descriptives. The later is used for the next step (topic modeling)
    # Ipc are appends later on in file XXX. This redundancy can be resolved in future work. Write now the redundancy is kept,
    # because I neither have the head space nor the time, to adjust the other files accordingly (list position/slicing shenanigans)
    position = np.where(patents_english[:,0] == 365546649)
    patents_english_cleaned = np.delete(patents_english, position, 0)

    filename = 'patents_english_cleaned'
    outfile = open(filename, 'wb')
    pk.dump(patents_english_cleaned, outfile)
    outfile.close()



    # --- Longitudinal Descriptives ---#
    print('\n# --- Overview ---#\n')

    patent_time = patents_english_cleaned[:, 3].astype('datetime64')

    print('Earliest day with publication: ', min(patent_time))  # earliest day with publication 2001-08-01
    print('Latest day with publication: ', max(patent_time))  # latest  day with publication 2018-01-31

    max_timeSpan = int((max(patent_time) - min(patent_time)) / np.timedelta64(1, 'D'))
    print('Days inbetween: ', max_timeSpan)  # 6027 day between earliest and latest publication

    val, count = np.unique(patent_time, return_counts=True)
    print('Number of days with publications: ', len(val))  # On 817 days publications were made
    # -> on average every 7.37698898409 days a patent was published
    print('Average publication cycle: ', max_timeSpan / len(val))

    import matplotlib.pyplot as plt

    # number of months: 5 + 16*12 + 1 = 198
    fig, ax = plt.subplots(1, 1)
    ax.hist(patent_time, bins=198, color='darkblue')
    #plt.title("Histogram: Monthly number of patent publications")
    plt.xlabel("Publication time span")
    plt.ylabel("Number of patents published")

    os.chdir('D:/Universitaet Mannheim/MMDS 7. Semester/Master Thesis/Outline/Plots')
    plt.savefig('hist_publications.png')
    os.chdir('D:/Universitaet Mannheim/MMDS 7. Semester/Master Thesis/Outline/Data/Cleaning Robots')

    plt.close()

