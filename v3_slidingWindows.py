### This file relies on the results of v3_dataTransformation.py. It takes the transformed and enriched data set and
### creates longitudinal data sets that are used for later analyses. With this longitudinal data sets insights into
### the recombination and diffusion patterns of the provided patent data is facilitated.
### The longitudinal data sets are created with the implementation of a sliding window approach. Until now, the
### following code supports a window size of 60 and 90 days and a sliding intervall of 1 or 7 days.
### The data sets are saved as dictionary in pickle form.


if __name__ == '__main__':

#--- Import Libraries ---#
    print('\n#--- Import Libraries ---#\n')

    import numpy as np
    import pandas as pd
    import pickle as pk

    import os
    import tqdm



#--- Initialization --#
    print('\n# --- Initialization ---#\n')

    os.chdir('D:/Universitaet Mannheim/MMDS 7. Semester/Master Thesis/Outline/Data/Cleaning Robots')


    patent_lda_ipc = pd.read_csv('patent_lda_ipc.csv', quotechar='"', skipinitialspace=True)

    patent_lda_ipc = patent_lda_ipc.to_numpy()


    ### Declare sliding window approach ###

    window90by1_bool = False
    window60by1_bool = False
    window90by7_bool = False
    window60by7_bool = False

    window365by30_bool = True



#--- Overview ---#
    print('\n# --- Overview ---#\n')

    patent_time = patent_lda_ipc[:,3].astype('datetime64')

    print('Earliest day with publication: ', min(patent_time))          # earliest day with publication 2001-08-01
    print('Latest day with publication: ', max(patent_time))            # latest  day with publication 2018-01-31

    max_timeSpan = int((max(patent_time) - min(patent_time)) / np.timedelta64(1, 'D'))
    print('Days inbetween: ', max_timeSpan)                             # 6027 day between earliest and latest publication

    val, count = np.unique(patent_time, return_counts=True)
    print('Number of days with publications: ', len(val))               # On 817 days publications were made
                                                                        # -> on average every 7.37698898409 days a patent was published


#--- slinding window approache 90 days by 1 day ---#
    print('\n#--- slinding window approache 90 days by 1 day ---#\n')


    ### 90 days sliding by 1 day ###

    patent_time_unique = np.unique(patent_time)                                                                                 #  817
    patent_time_unique_filled = np.arange(np.min(patent_time_unique), np.max(patent_time_unique))                               # 6027
    patent_time_unique_filled_90 = patent_time_unique_filled[patent_time_unique_filled <= max(patent_time_unique_filled)-90]    # 5937

    if window90by1_bool == True:

        window90by1 = {}
        len_window = []

        c = 0
        pbar = tqdm.tqdm(total=len(patent_time_unique_filled_90))

        for i in patent_time_unique_filled_90:
            lower_limit = i
            upper_limit = i + 90

            patent_window = patent_lda_ipc[(patent_lda_ipc[:, 3].astype('datetime64') < upper_limit) & (patent_lda_ipc[:, 3].astype('datetime64') >= lower_limit)]
            len_window.append(len(patent_window))

            window90by1['window_{0}'.format(c)] = patent_window

            pbar.update(1)
            #if i >= 100:
                #break

            c = c + 1

        pbar.close()

        #print(len(window90by1))                     # 5937 windows
        #print(sum(len_window)/len(len_window))      # on average 56.253326595923866 patents per window


        filename = 'window90by1'
        outfile = open(filename,'wb')
        pk.dump(window90by1, outfile)
        outfile.close()

        #print('Preview of the resulting Array (90by1):\n\n', window90by1)                   # 'window_3216': array([
                                                                                            # [315561860, 'EP', 2189254.0, ..., nan, nan, nan],
                                                                                            # [315562353, 'EP', 2189094.0, ..., nan, nan, nan],
                                                                                            # [315562468, 'EP', 2189252.0, ..., nan, nan, nan],
                                                                                            # ...,
                                                                                            # [323514081, 'EP', 2212064.0, ..., nan, nan, nan],
                                                                                            # [323609355, 'EP', 2214872.0, ..., nan, nan, nan],
                                                                                            # [323656643, 'EP', 2217411.0, ..., nan, nan, nan]
                                                                                            # ], dtype=object)

# --- slinding window approache 60 days by 1 day ---#
    print('\n#--- slinding window approache 60 days by 1 day ---#\n')


    patent_time_unique_filled_60 = patent_time_unique_filled[patent_time_unique_filled <= max(patent_time_unique_filled)-60]

    if window60by1_bool == True:

        window60by1 = {}
        len_window = []

        c = 0
        pbar = tqdm.tqdm(total=len(patent_time_unique_filled_60))

        for i in patent_time_unique_filled_60:
            lower_limit = i
            upper_limit = i + 60

            patent_window = patent_lda_ipc[(patent_lda_ipc[:, 3].astype('datetime64') < upper_limit) & (patent_lda_ipc[:, 3].astype('datetime64') >= lower_limit)]
            len_window.append(len(patent_window))

            window60by1['window_{0}'.format(c)] = patent_window

            pbar.update(1)
            #if i >= 100:
                #break
            c = c+1

        pbar.close()

        #print(len(window60by1))                        # 5967 windows
        #print(sum(len_window)/len(len_window))         # on average 37.50477626948215 patents per window

        filename = 'window60by1'
        outfile = open(filename,'wb')
        pk.dump(window60by1,outfile)
        outfile.close()



# --- slinding window approache 90 days by 7 day ---#
    print('\n#--- slinding window approache 90 days by 7 day ---#\n')


    if window90by7_bool == True:

        window90by7 = {}
        len_window = []

        c = 0
        pbar = tqdm.tqdm(total=len(patent_time_unique_filled_90))

        for i in patent_time_unique_filled_90:

            if c % 7 == 0:
                lower_limit = i
                upper_limit = i + 90

                patent_window = patent_lda_ipc[(patent_lda_ipc[:, 3].astype('datetime64') < upper_limit) & (patent_lda_ipc[:, 3].astype('datetime64') >= lower_limit)]
                len_window.append(len(patent_window))

                window90by7['window_{0}'.format(c)] = patent_window

            pbar.update(1)
            #if i >= 100:
                #break
            c = c+1

        pbar.close()

        #print(len(window90by7))                         # 849 windows
        #print(sum(len_window)/len(len_window))          # on average 56.849234393404004 patents per window

        filename = 'window90by7'
        outfile = open(filename,'wb')
        pk.dump(window90by7,outfile)
        outfile.close()



# --- slinding window approache 60 days by 1 day ---#
    print('\n#--- slinding window approache 60 days by 7 day ---#\n')


    if window60by7_bool == True:

        window60by7 = {}
        len_window = []

        c = 0
        pbar = tqdm.tqdm(total=len(patent_time_unique_filled_60))

        for i in patent_time_unique_filled_60:

            if c % 7 == 0:
                lower_limit = i
                upper_limit = i + 60

                patent_window = patent_lda_ipc[(patent_lda_ipc[:, 3].astype('datetime64') < upper_limit) & (patent_lda_ipc[:, 3].astype('datetime64') >= lower_limit)]
                len_window.append(len(patent_window))

                window60by7['window_{0}'.format(c)] = patent_window

            pbar.update(1)
            #if i >= 100:
                #break

            c = c+1

        pbar.close()

        #print(len(window60by7))                         # 853 windows
        #print(sum(len_window)/len(len_window))          # on average 39.35873388042204 patents per window

        filename = 'window60by7'
        outfile = open(filename,'wb')
        pk.dump(window60by7,outfile)
        outfile.close()



#--- slinding window approache 365 days by 30 days ---#
    print('\n#--- slinding window approache 365 days by 30 days ---#\n')


    ### 365 days sliding by 30 days ###

    patent_time_unique = np.unique(patent_time)                                                                                 #  817
    patent_time_unique_filled = np.arange(np.min(patent_time_unique), np.max(patent_time_unique))                               # 6027
    patent_time_unique_filled_365 = patent_time_unique_filled[patent_time_unique_filled <= max(patent_time_unique_filled)-365]    #5662

    if window365by30_bool == True:

        window365by30 = {}
        len_window = []

        c = 0
        pbar = tqdm.tqdm(total=len(patent_time_unique_filled_365))

        for i in patent_time_unique_filled_365:

            if c % 30 == 0:
                lower_limit = i
                upper_limit = i + 365

                patent_window = patent_lda_ipc[(patent_lda_ipc[:, 3].astype('datetime64') < upper_limit) & (patent_lda_ipc[:, 3].astype('datetime64') >= lower_limit)]
                len_window.append(len(patent_window))

                window365by30['window_{0}'.format(c)] = patent_window

            pbar.update(1)
            #if i >= 100:
                #break

            c = c + 1

        pbar.close()

        #print(len(window365by30))                     # 189 windows
        #print(sum(len_window)/len(len_window))      # on average 225.83597883597884 patents per window


        filename = 'window365by30'
        outfile = open(filename,'wb')
        pk.dump(window365by30, outfile)
        outfile.close()
