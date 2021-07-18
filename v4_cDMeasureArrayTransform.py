if __name__ == '__main__':
# --- Import Libraries ---#
    print('\n#--- Import Libraries ---#\n')

    import pandas as pd
    import numpy as np
    import pickle as pk

    import networkx as nx
    from cdlib import algorithms
    # import wurlitzer                   #not working for windows

    import tqdm
    import itertools
    import operator
    import os
    import sys

# --- Initialization ---#
    print('\n# --- Initialization ---#\n')

    os.chdir('D:/Universitaet Mannheim/MMDS 7. Semester/Master Thesis/Outline/Data/Cleaning Robots')

    #patent_lda_ipc = pd.read_csv('patent_lda_ipc.csv', quotechar='"', skipinitialspace=True)
    #patent_lda_ipc = patent_lda_ipc.to_numpy()

    with open('windows_topicSim', 'rb') as handle:
        topicSim = pk.load(handle)

    with open('lp_recombinations', 'rb') as handle:
        lp_recombinations = pk.load(handle)
    with open('lp_labeled', 'rb') as handle:
        lp_labeled = pk.load(handle)
    with open('lp_topD_dic', 'rb') as handle:
        lp_topD_dic = pk.load(handle)
    with open('lp_topD', 'rb') as handle:
        lp_topD = pk.load(handle)

    with open('gm_recombinations', 'rb') as handle:
        gm_recombinations = pk.load(handle)
    with open('gm_labeled', 'rb') as handle:
        gm_labeled = pk.load(handle)
    with open('gm_topD_dic', 'rb') as handle:
        gm_topD_dic = pk.load(handle)

    with open('kclique_recombinations', 'rb') as handle:
        kclique_recombinations = pk.load(handle)
    with open('kclique_labeled', 'rb') as handle:
        kclique_labeled = pk.load(handle)
    with open('kclique_topD_dic', 'rb') as handle:
        kclique_topD_dic = pk.load(handle)

    with open('lais2_recombinations', 'rb') as handle:
        lais2_recombinations = pk.load(handle)
    with open('lais2_labeled', 'rb') as handle:
        lais2_labeled = pk.load(handle)
    with open('lais2_topD_dic', 'rb') as handle:
        lais2_topD_dic = pk.load(handle)


# ---  cleaning topD_dic ---#


    # 1. cleaning topD_dic
    #
    #   go through topD_dic.
    #   if topD vanishes in t+1, but it's community id persists:
    #       get all nodes of the community on t
    #       get their life span
    #       remove community id of every entry in topD_dic, once the life span is over

    '''
    print(lp_topD['window_5640'])
    x = [item for sublist in lp_topD['window_5640'] for item in sublist]
    x = [item for sublist in x for item in sublist]
    print(len(lp_topD['window_5640']))
    print(len(x))
    '''
    # members = [288465877, 287698910, 286963357, 289531190]

    x = [1,2]
    y = [1,2,3,4,5]
    z = [6,7,8]
    '''    
    if set(x).issubset(set(y)):
        print('good')
    if set(x).issubset(set(z)):
        print('not so good')
    print(1+1)
    y.remove(x[0])
    print(y)
    print(y)
    '''

    # [[{479005092, 472829578, 483787982, 485992401, 475452405, 480127001, 478453563, 483788347}, [(475452405, 13)]], [{484343840, 476756129, 484113091, 484113352, 484100779, 484100755}, [(484100779, 12)]], [{474753794, 473894947, 474753543}, [(474753794, 4)]], [{473498661, 476102694, 481115112, 475496054, 473501239}, [(476102694, 8)]], [{483575169, 475496590, 478161808, 484341269, 474487702, 484372377, 475835169, 481118509, 475875376, 481119669, 480557238, 483275065, 475874875, 486893638, 473498184, 477742030, 474788559, 475495120, 484869216, 474789729, 484876259, 479239022, 477740025}, [(481119669, 23)]], [{476713410, 476103204, 476713455, 476714420, 484372534, 476756279, 476755323}, [(484372534, 14)]], [{483279270, 484098919, 479814028, 482968974, 477501647, 473303280, 483276855, 477286015}, [(479814028, 14)]], [{482715712, 478234059, 485992662, 473895471}, [(478234059, 3)]], [{481585217, 485155234, 484119554, 479247721, 481581899, 486220246}, [(486220246, 12)]], [{486900719, 481120338, 485151346, 475026489, 485754746, 479002811}, [(475026489, 9)]], [{479247616, 482718083, 477062788, 486574607, 475025936, 482058519, 481581344, 479238570, 478161834, 484112174, 475863733, 483790776, 483789499, 474788046, 473498062, 482715986, 484342227, 476756052, 475028565, 473499614, 477740138, 481901932, 478156654, 482968306, 484343544, 482058493}, [(483790776, 25)]], [{475453185, 485463971, 484876233, 481116587, 474754829, 484100751, 484342967, 486892440, 486573273, 472826618, 480555100}, [(484876233, 22)]], [{482055392, 483276898, 473501604, 483276681, 474486254, 476098167, 485462266, 477043100}, [(477043100, 16)]], [{482965892, 486902020, 475876229, 476756103, 473895174, 481351178, 480556173, 475495187, 478769557, 483520919, 476097175, 474199963, 481907742, 483576351, 486900260, 474788261, 475264804, 485459623, 475865265, 472830261, 482453814, 477295419, 478196923, 476786492, 473501501, 481584962, 484118852, 477473221, 483792196, 481584966, 477475019, 476595282, 480875986, 486331350, 475875799, 483522778, 474753755, 486220382, 482968927, 483277664, 486573919, 483788898, 473281634, 481353186, 474194150, 473498599, 484340587, 484340588, 484120171, 472827628, 479000944, 475863922, 484111862, 484118903, 479001081, 484876027}, [(476097175, 37)]], [{476757088, 478157316, 480757540, 477741001, 477042797, 473281391}, [(477042797, 14)]], [{478449443, 485154932, 484572502, 486218839, 481353852}, [(484572502, 11)]], [{484113185, 475026916, 486901573, 477043112, 485462156, 485459607}, [(485459607, 11)]], [{486573091, 474194377, 484370576, 473302294, 482966015}, [(474194377, 7)]], [{480557441, 479815138, 479813987, 477741988, 486374216, 486900393, 473280364, 484119352, 481902746, 485463804}, [(479813987, 16)]], [{480128481, 480129474, 484344457, 477295529, 479239115, 480556652, 472830509, 480128077, 480127151, 474488077, 475496628, 473500790, 479004855, 475264154, 486332411, 477295325, 478452287}, [(480129474, 33)]], [{472827650, 482453316, 479004849, 479001970, 486330867, 481584212, 481901875, 473280181, 477284273, 477043800, 484371193, 474486495}, [(482453316, 15)]], [{482061191, 484344337, 473302421, 481902102, 479004951, 478157472, 473280299, 481353266, 482724019, 475876159, 486218829, 485769427, 473304022, 473498213, 475876086, 473281398, 472829558, 481585530, 478200828}, [(482724019, 21)]], [{484876705, 481118389, 483280393}, [(484876705, 12)]], [{479004862, 482060921, 480555963, 482441884, 485155166}, [(479004862, 22)]], [{479815042, 486572612, 481907115, 482056907, 482715213, 474754286, 484340303, 481118123, 478452147, 479813848, 481905147, 483276734}, [(486572612, 14)]], [{476756198, 484341708, 476594003, 483793432, 484343707, 473498556}, [(484341708, 21)]], [{478449476, 479245320, 485769609, 484372722, 480127541, 484875863, 475268410}, [(478449476, 9)]], [{482968672, 484120325, 484340497, 472826707, 485155063, 472826620}, [(484120325, 13)]], [{476098439, 484341641, 476755805, 478196883, 475495129, 475025719, 479811833, 482057050, 476097051, 484877597}, [(484341641, 14)]], [{484370728, 484113514, 484113596, 485463863, 483275004, 484370686}, [(485463863, 16)]], [{475494978, 482968412, 485769614}, [(485769614, 6)]], [{482719657, 475495761, 477739794, 486219260, 475494815}, [(475494815, 6)]], [{482059289, 481180350, 473895342}, [(473895342, 7)]], [{474485986, 486572486, 477295498, 478159568, 477044473, 480877271, 483275193, 479246012, 483790175}, [(479246012, 11)]], [{486220219, 474523550, 479810615}, [(486220219, 4)]], [{486331369, 484576547, 474486532, 482965959}, [(482965959, 5)]], [{479239876, 476096981, 479237814}, [(479239876, 4)]], [{472599012, 473279492, 473301830, 473279501, 482442644, 479002356, 474523576}, [(482442644, 13)]], [{481586770, 477475323, 478779141}, [(481586770, 4)]], [{480124065, 480559642, 474199342, 483794031}, [(480124065, 6)]], [{477044008, 473302095, 478194290, 472601752, 473279550}, [(477044008, 8)]], [{480559779, 485752661, 486218758}, [(480559779, 4)]], [{483576960, 483577491, 482714365, 481584231}, [(483577491, 6)]], [{474754764, 477295573, 472601710, 478450524}, [(472601710, 13)]], [{485991960, 477043817, 483789895}, [(483789895, 11)]], [{476098424, 473499249, 473501219, 479004653}, [(479004653, 12)]]]
    # [[{479005092, 472829578, 483787982, 485992401, 475452405, 480127001, 478453563, 483788347}, [(475452405, 13)]], [{484343840, 476756129, 484113091, 484113352, 484100779, 484100755}, [(484100779, 12)]], [{474753794, 473894947, 474753543}, [(474753794, 4)]], [{473498661, 476102694, 481115112, 475496054, 473501239}, [(476102694, 8)]], [{483575169, 475496590, 478161808, 484341269, 474487702, 484372377, 475835169, 481118509, 475875376, 481119669, 480557238, 483275065, 475874875, 486893638, 473498184, 477742030, 474788559, 475495120, 484869216, 474789729, 484876259, 479239022, 477740025}, [(481119669, 23)]], [{476713410, 476103204, 476713455, 476714420, 484372534, 476756279, 476755323}, [(484372534, 14)]], [{483279270, 484098919, 479814028, 482968974, 477501647, 473303280, 483276855, 477286015}, [(479814028, 14)]], [{482715712, 478234059, 485992662, 473895471}, [(478234059, 3)]], [{481585217, 485155234, 484119554, 479247721, 481581899, 486220246}, [(486220246, 12)]], [{486900719, 481120338, 485151346, 475026489, 485754746, 479002811}, [(475026489, 9)]], [{479247616, 482718083, 477062788, 486574607, 475025936, 482058519, 481581344, 479238570, 478161834, 484112174, 475863733, 483790776, 483789499, 474788046, 473498062, 482715986, 484342227, 476756052, 475028565, 473499614, 477740138, 481901932, 478156654, 482968306, 484343544, 482058493}, [(483790776, 25)]], [{475453185, 485463971, 484876233, 481116587, 474754829, 484100751, 484342967, 486892440, 486573273, 472826618, 480555100}, [(484876233, 22)]], [{482055392, 483276898, 473501604, 483276681, 474486254, 476098167, 485462266, 477043100}, [(477043100, 16)]], [{482965892, 486902020, 475876229, 476756103, 473895174, 481351178, 480556173, 475495187, 478769557, 483520919, 476097175, 474199963, 481907742, 483576351, 486900260, 474788261, 475264804, 485459623, 475865265, 472830261, 482453814, 477295419, 478196923, 476786492, 473501501, 481584962, 484118852, 477473221, 483792196, 481584966, 477475019, 476595282, 480875986, 486331350, 475875799, 483522778, 474753755, 486220382, 482968927, 483277664, 486573919, 483788898, 473281634, 481353186, 474194150, 473498599, 484340587, 484340588, 484120171, 472827628, 479000944, 475863922, 484111862, 484118903, 479001081, 484876027}, [(476097175, 37)]], [{476757088, 478157316, 480757540, 477741001, 477042797, 473281391}, [(477042797, 14)]], [{478449443, 485154932, 484572502, 486218839, 481353852}, [(484572502, 11)]], [{484113185, 475026916, 486901573, 477043112, 485462156, 485459607}, [(485459607, 11)]], [{486573091, 474194377, 484370576, 473302294, 482966015}, [(474194377, 7)]], [{480557441, 479815138, 479813987, 477741988, 486374216, 486900393, 473280364, 484119352, 481902746, 485463804}, [(479813987, 16)]], [{480128481, 480129474, 484344457, 477295529, 479239115, 480556652, 472830509, 480128077, 480127151, 474488077, 475496628, 473500790, 479004855, 475264154, 486332411, 477295325, 478452287}, [(480129474, 33)]], [{472827650, 482453316, 479004849, 479001970, 486330867, 481584212, 481901875, 473280181, 477284273, 477043800, 484371193, 474486495}, [(482453316, 15)]], [{482061191, 484344337, 473302421, 481902102, 479004951, 478157472, 473280299, 481353266, 482724019, 475876159, 486218829, 485769427, 473304022, 473498213, 475876086, 473281398, 472829558, 481585530, 478200828}, [(482724019, 21)]], [{484876705, 481118389, 483280393}, [(484876705, 12)]], [{479004862, 482060921, 480555963, 482441884, 485155166}, [(479004862, 22)]], [{479815042, 486572612, 481907115, 482056907, 482715213, 474754286, 484340303, 481118123, 478452147, 479813848, 481905147, 483276734}, [(486572612, 14)]], [{476756198, 484341708, 476594003, 483793432, 484343707, 473498556}, [(484341708, 21)]], [{478449476, 479245320, 485769609, 484372722, 480127541, 484875863, 475268410}, [(478449476, 9)]], [{482968672, 484120325, 484340497, 472826707, 485155063, 472826620}, [(484120325, 13)]], [{476098439, 484341641, 476755805, 478196883, 475495129, 475025719, 479811833, 482057050, 476097051, 484877597}, [(484341641, 14)]], [{484370728, 484113514, 484113596, 485463863, 483275004, 484370686}, [(485463863, 16)]], [{475494978, 482968412, 485769614}, [(485769614, 6)]], [{482719657, 475495761, 477739794, 486219260, 475494815}, [(475494815, 6)]], [{482059289, 481180350, 473895342}, [(473895342, 7)]], [{474485986, 486572486, 477295498, 478159568, 477044473, 480877271, 483275193, 479246012, 483790175}, [(479246012, 11)]], [{486220219, 474523550, 479810615}, [(486220219, 4)]], [{486331369, 484576547, 474486532, 482965959}, [(482965959, 5)]], [{479239876, 476096981, 479237814}, [(479239876, 4)]], [{472599012, 473279492, 473301830, 473279501, 482442644, 479002356, 474523576}, [(482442644, 13)]], [{481586770, 477475323, 478779141}, [(481586770, 4)]], [{480124065, 480559642, 474199342, 483794031}, [(480124065, 6)]], [{477044008, 473302095, 478194290, 472601752, 473279550}, [(477044008, 8)]], [{480559779, 485752661, 486218758}, [(480559779, 4)]], [{483576960, 483577491, 482714365, 481584231}, [(483577491, 6)]], [{474754764, 477295573, 472601710, 478450524}, [(472601710, 13)]], [{485991960, 477043817, 483789895}, [(483789895, 11)]], [{476098424, 473499249, 473501219, 479004653}, [(479004653, 12)]]]

    def cleaningIndex_topD_dic(topD_dic, cd_topD):
        merging_communities_dic = {}

        for i in range(len(topD_dic)-1):

            #if i == 11:
                #print(1+1)

            window_id = 'window_{0}'.format(i * 30)
            window = topD_dic[window_id]

            next_window_id = 'window_{0}'.format((i+1) * 30)
            next_window = topD_dic[next_window_id]

            swallowed_communities = []

            if i != 0:

                #if i == 12:
                    #print(1+1)

                for topD in window.keys():

                    if topD not in next_window.keys():

                        next_community_lists = list(next_window.values())

                        #print(next_community_lists)

                        for community in next_community_lists:

                            if set(window[topD]).issubset(set(community)):

                                if len(set(community) - set(window[topD])) != 0:
                                    swallowed_communities.append([topD, window[topD]])
                                break
                        '''
                        if window[topD] in next_community_lists:
                            pos = np.where(next_community_lists == window[topD])

                            if len(next_community_lists[pos])

                        #print(window[topD])

                        window_values_flattend = [item for sublist in window.values() for item in sublist]
                        next_window_values_flattend = [item for sublist in next_window.values() for item in sublist]



                        for community_id in window[topD]:

                            next_window_values_flattend = [item for sublist in next_window.values() for item in sublist]
                        
                            if community_id in next_window_values_flattend:

                        #if set(window_values_flattend).issubset(set(next_window_values_flattend)):
                                merge = False

                                for next_topD in next_window:
                                    if community_id in next_window[next_topD]:
                                        if len(next_window[next_topD]) >= 2:
                                            merge = True

                                        break

                                if merge == True:
                                    #print(next_window_values_flattend)

                                    swallowed_communities.append([topD, community_id])
                        '''

                # get life time
                # I want for every window a list of community_ids that are swallowed, and their death time
                if len(swallowed_communities) != 0:
                    for swallowed_community in swallowed_communities:
                        #if len(swallowed_community) != 0:
                        for community in cd_topD[window_id]:

                            if swallowed_community[0] == community[1][0][0]:
                                swallowed_community.append(community[0])
                                break
                    #print(swallowed_communities)

                    for swallowed_community in swallowed_communities:
                        community_death = False
                        j = i
                        members = list(swallowed_community[2])

                        while community_death == False:                    # [288465877, 287698910, 286963357, 289531190]


                            if len(members) != 0:
                                #print(cd_topD['window_{0}'.format(j*30)])
                                all_id_in_next_window = [item for sublist in cd_topD['window_{0}'.format((j+1)*30)] for item in sublist]
                                #print(all_id_in_next_window)
                                all_id_in_next_window = [item for sublist in all_id_in_next_window for item in sublist]
                                #print(all_id_in_next_window)
                                if members[-1] not in all_id_in_next_window:
                                    #print(members)
                                    members.pop()
                                    #print(members)
                                else:
                                    j = j + 1

                                if j == 188:
                                    community_death = True

                            if len(members) == 0:
                                community_death = True
                        #print(j - i)
                        #if (j - i) >= 13:
                            #print('problem')
                        swallowed_community.append(i)   # i = last point before merge
                        swallowed_community.append(j)   # j = point of death (first row were not alive)

            merging_communities_dic[window_id] = swallowed_communities

        return merging_communities_dic


    lp_topD_dic_cleanIndex = cleaningIndex_topD_dic(lp_topD_dic, lp_topD)
    gm_topD_dic_cleanIndex = cleaningIndex_topD_dic(gm_topD_dic, gm_topD)

    '''
    print(lp_topD_dic['window_330'])
    print(lp_topD_dic['window_360'])
    print(lp_topD_dic_cleanIndex['window_330'])
    print()
    print(lp_topD_dic['window_900'])
    print(lp_topD_dic['window_930'])
    print(lp_topD_dic_cleanIndex['window_900'])
    print()
    print(lp_topD_dic['window_3000'])
    print(lp_topD_dic['window_3030'])
    print(lp_topD_dic_cleanIndex['window_3000'])
    print(1+1)
    '''


    def cleaning_topD_dic(cd_topD_dic, cd_topD_dic_cleanIndex):
        cd_topD_dic_clean = cd_topD_dic

        for i in range(len(cd_topD_dic_cleanIndex)):
            window_id = 'window_{0}'.format(i * 30)
            window = cd_topD_dic_cleanIndex[window_id]

            for cleaning_entry in window:
                community_id = cleaning_entry[1]
                last_point_before_swallowed = cleaning_entry[3]
                point_of_death = cleaning_entry[4]

                #for j2 in range(last_point_before_swallowed,point_of_death):
                    #print(cd_topD_dic_clean['window_{0}'.format(j2 * 30)])

                for j in range(point_of_death, len(cd_topD_dic_clean)):
                    cleaning_window = cd_topD_dic_clean['window_{0}'.format(j * 30)]
                    for community_toBeCleaned in cleaning_window.values():
                        if set(community_id).issubset(set(community_toBeCleaned)):
                            for id in community_id:
                                community_toBeCleaned.remove(id)

                #for j3 in range(last_point_before_swallowed, point_of_death+10):
                    #print(cd_topD_dic_clean['window_{0}'.format(j3 * 30)])

                #print(cd_topD_dic_clean['window_330'])

        return cd_topD_dic_clean

    lp_topD_dic_clean = cleaning_topD_dic(lp_topD_dic, lp_topD_dic_cleanIndex)
    '''
    print(lp_topD_dic['window_330'])
    print(lp_topD_dic['window_360'])
    print(lp_topD_dic_cleanIndex['window_330'])
    print(lp_topD_dic_clean['window_330'])
    print(lp_topD_dic_clean['window_360'])
    print(lp_topD_dic_clean['window_600'])
    print(lp_topD_dic_clean['window_630'])
    print(lp_topD_dic_clean['window_660'])
    print()
    '''

#--- Constructing Diffusion Array ---#

    #1. Compute all recombinations present in data
    #2. span np.arrays
    #3. fill np array either with count or threshold
    #4. present way to query it for long strings of ones

    def single_diffusion(cd_labeled):

        row_length = len(cd_labeled)

        only_id_dic = {}
        community_ids_all = []
        for window_id, window in cd_labeled.items():
            community_ids_window = []
            for community in window:
                community_ids_window.append(community[1][0])
                community_ids_all.append(community[1][0])

            only_id_dic[window_id] = community_ids_window

        column_length = max(community_ids_all)

        singleDiffusion_array = np.zeros((row_length, column_length), dtype=int)

        for i in range(len(singleDiffusion_array)):
            for j in range(len(singleDiffusion_array.T)):
                if j in only_id_dic['window_{0}'.format(i * 30)]:
                    #print(j)
                    #print(only_id_dic['window_{0}'.format(i * 30)])
                    singleDiffusion_array[i,j] = 1

        return singleDiffusion_array

    #lp_singleDiffusion = single_diffusion(lp_labeled)
    #gm_singleDiffusion = single_diffusion(gm_labeled)

    #kclique_singleDiffusion = single_diffusion(kclique_labeled)
    #lais2_singleDiffusion = single_diffusion(lais2_labeled)

    #print(lp_singleDiffusion)
    #print(gm_singleDiffusion)
    #print(kclique_singleDiffusion)
    #print(lais2_singleDiffusion)

    def recombination_diffusion_crips(cd_recombinations):

        row_length = len(cd_recombinations)

        recombinations_dic = {}
        recombinations_all = []
        for window_id, window in cd_recombinations.items():
            recombinations_window = []
            for recombination in window:
                community_id1 = recombination[1][0][1][0]
                community_id2 = recombination[1][1][1][0]
                recombinations_all.append((community_id1, community_id2))
                recombinations_window.append((community_id1, community_id2))

            recombinations_dic[window_id] = recombinations_window
        #print(recombinations_dic)

        recombinations_all.sort()
        column_length = len(np.unique(recombinations_all, axis=0))

        recombinationDiffusion_count = np.zeros((row_length, column_length), dtype=int)
        recombinationDiffusion_fraction = np.zeros((row_length, column_length), dtype=float)
        #recombinationDiffusion_threshold = np.zeros((row_length, column_length), dtype=float)

        for i in range(len(recombinationDiffusion_count)):
            for j in range(len(recombinationDiffusion_count.T)):
                window = recombinations_dic['window_{0}'.format(i * 30)]
                recombinationDiffusion_count[i,j] = window.count(recombinations_all[j])

        for i in range(len(topicSim)):
            all_nodes_in_window = topicSim['window_{0}'.format(i * 30)].nodes()
            recombinationDiffusion_fraction[i,:] = (recombinationDiffusion_count[i,:] / len(all_nodes_in_window) )

        recombinationDiffusion_threshold = np.where(recombinationDiffusion_fraction < 0.005, 0, 1)

        return recombinationDiffusion_count, recombinationDiffusion_fraction, recombinationDiffusion_threshold


    def recombination_diffusion_overlapping(cd_recombinations):

        row_length = len(cd_recombinations)

        recombinations_dic = {}
        recombinations_all = []
        for window_id, window in cd_recombinations.items():
            recombinations_window = []
            for recombination in window:
                community_id1 = recombination[1][0]
                community_id2 = recombination[1][1]
                recombinations_all.append((community_id1, community_id2))
                recombinations_window.append((community_id1, community_id2))

            recombinations_dic[window_id] = recombinations_window
        #print(recombinations_dic)

        recombinations_all.sort()
        column_length = len(np.unique(recombinations_all, axis=0))

        recombinationDiffusion_count = np.zeros((row_length, column_length), dtype=float)
        recombinationDiffusion_fraction = np.zeros((row_length, column_length), dtype=float)
        recombinationDiffusion_threshold = np.zeros((row_length, column_length), dtype=float)

        for i in range(len(recombinationDiffusion_count)):
            for j in range(len(recombinationDiffusion_count.T)):
                window = recombinations_dic['window_{0}'.format(i * 30)]
                recombinationDiffusion_count[i,j] = window.count(recombinations_all[j])

        for i in range(len(topicSim)):
            all_nodes_in_window = topicSim['window_{0}'.format(i * 30)].nodes()
            recombinationDiffusion_fraction[i,:] = (recombinationDiffusion_count[i,:] / len(all_nodes_in_window) )

        recombinationDiffusion_threshold = np.where(recombinationDiffusion_fraction < 0.005, 0, 1)

        return recombinationDiffusion_count, recombinationDiffusion_fraction, recombinationDiffusion_threshold


    #lp_recombinationDiffusion_count, lp_recombinationDiffusion_fraction, lp_recombinationDiffusion_threshold = recombination_diffusion_crips(lp_recombinations)
    #gm_recombinationDiffusion_count, gm_recombinationDiffusion_fraction, gm_recombinationDiffusion_threshold = recombination_diffusion_crips(gm_recombinations)

    #kclique_recombinationDiffusion_count, kclique_recombinationDiffusion_fraction, kclique_recombinationDiffusion_threshold = recombination_diffusion_overlapping(kclique_recombinations)
    #lais2_recombinationDiffusion_count, lais2_recombinationDiffusion_fraction, lais2_recombinationDiffusion_threshold = recombination_diffusion_overlapping(lais2_recombinations)

    np.set_printoptions(threshold=sys.maxsize)
    np.set_printoptions(precision=2)
    np.set_printoptions(suppress=True)
    '''
    print(lp_recombinationDiffusion_count[0:99,0:24])
    print(lp_recombinationDiffusion_fraction[0:99,0:24])
    print(lp_recombinationDiffusion_threshold[0:99,0:24])

    print(np.size(lp_recombinationDiffusion_count))
    print(sum(sum(lp_recombinationDiffusion_count)))
    '''


# Diffusion is not working like this. The reason is that a recombination of B & C is lost if afterwards B megres with A
# with A being the dominant community (giving both id A)

#--- new diffusion approach ---#



#How to solve this:

    # 2. populating diffusion array:
    #
    #   x = all t
    #   y = all recombinations (no change from what i have now)
    #   for all columns:
    #       for all rows:
    #           big_community1 = community_list where tuple[0] is present (for this window)
    #           big_community2 = community_list where tuple[1] is present (for this window)
    #           overall_count = 0
    #           for all recombinations:
    #               if recombination[0] in big_community1:
    #                   if recombination[1] in big_community2:
    #                       overall_count = overall_count + 1
    #                       break
    #               elif: recombination[0] in big_community2:
    #                   if recombination[1] in big_community1:
    #                       overall_count = overall_count + 1
    #                       break

    def single_diffusion_v2(cd_topD_dic_clean):
        row_length = len(cd_topD_dic_clean)

        all_ids = []
        for window_id, window in cd_topD_dic_clean.items():

            for community in window:

                all_ids.append(window[community])
        all_ids = [item for sublist in all_ids for item in sublist]

        column_length = max(all_ids)

        singleDiffusion_array = np.zeros((row_length, column_length), dtype=int)

        pbar = tqdm.tqdm(total=len(singleDiffusion_array))
        for i in range(len(singleDiffusion_array)):
            for j in range(len(singleDiffusion_array.T)):

                window = cd_topD_dic_clean['window_{0}'.format(i*30)]
                #print(window)
                #print(window.values())
                #print(list(window.values()))
                #print(j)

                if any(j in sublist for sublist in list(window.values())) == True:
                    # A count is not necessary, since the value can not exceed 1. Community ids are unique within a window.
                    singleDiffusion_array[i, j] = 1

                '''
                for k in range(len(list(window.values()))):
                    if j in list(window.values())[k]:
                        big_community = list(window.values())[k]
                        break
                #print(big_community)

                overall_count = 0

                big_community = 1
                '''
            pbar.update(1)

        pbar.close()

        return singleDiffusion_array


    #lp_singleDiffusion_v2 = single_diffusion_v2(lp_topD_dic_clean)
    #print(lp_singleDiffusion_v2[0:99,0:25])
    #gm_singleDiffusion_v2 = single_diffusion_v2(gm_topD_dic_clean)

    #kclique_singleDiffusion_v2 = single_diffusion_v2(kclique_topD_dic_clean)
    #lais2_singleDiffusion_v2 = single_diffusion_v2(lais2_topD_dic_clean)

    def recombination_diffusion_crip_v2(cd_topD_dic_clean, cd_recombinations):
        row_length = len(cd_recombinations)

        recombinations_dic = {}
        recombinations_all = []
        for window_id, window in cd_recombinations.items():
            recombinations_window = []
            for recombination in window:
                community_id1 = recombination[1][0][1][0]
                community_id2 = recombination[1][1][1][0]
                recombinations_all.append((community_id1, community_id2))
                recombinations_window.append((community_id1, community_id2))

            recombinations_dic[window_id] = recombinations_window
        #print(recombinations_dic)

        #print(len(recombinations_all))
        recombinations_all.sort()
        #print(recombinations_all)
        recombinations_all = np.unique(recombinations_all, axis=0)
        #print(recombinations_all)
        recombinations_all_tuple = []
        for recombination in recombinations_all:
            recombinations_all_tuple.append(tuple(recombination))
        recombinations_all = recombinations_all_tuple
        #print(len(recombinations_all))

        #print(recombinations_all)
        column_length = len(recombinations_all)

        recombinationDiffusion_count = np.zeros((row_length, column_length), dtype=int)
        #recombinationDiffusion_count = np.full((row_length, column_length), 9999999, dtype=int)
        recombinationDiffusion_fraction = np.zeros((row_length, column_length), dtype=float)
        recombinationDiffusion_threshold = np.zeros((row_length, column_length), dtype=float)

        pbar = tqdm.tqdm(total=len(recombinationDiffusion_count))
        for i in range(len(recombinationDiffusion_count)):
            #if i == 10:
                #print(1+1)
            for j in range(len(recombinationDiffusion_count.T)):


                window_topD_dic = cd_topD_dic_clean['window_{0}'.format(i * 30)]
                #print(window_topD_dic)

                # count how often a recombination appears in a window
                # Recombinations are identified over community id. These community id's are dominant.
                #print(recombinations_all[j])
                #print(recombinations_dic['window_{0}'.format(i*30)])
                recombination_count = recombinations_dic['window_{0}'.format(i*30)].count(recombinations_all[j])
                #print(recombination_count)

                if recombination_count != 0 :
                # this count has to be placed in all columns that are the same recombination under different community ids
                #(e.g. because of a community merge where the dominant id overwrite the original one used in the prior recombination

                    #print(list(window_topD_dic.values()))
                    for k in range(len(list(window_topD_dic.values()))):
                        #print(recombinations_all[j][0])
                        #print(list(window_topD_dic.values())[k])
                        if recombinations_all[j][0] in list(window_topD_dic.values())[k]:
                            big_community1 = list(window_topD_dic.values())[k]
                            #print(big_community1)
                            break

                    list(window_topD_dic.values())
                    for k in range(len(list(window_topD_dic.values()))):
                        #print(recombinations_all[j][1])
                        #print(list(window_topD_dic.values())[k])
                        if recombinations_all[j][1] in list(window_topD_dic.values())[k]:
                            big_community2 = list(window_topD_dic.values())[k]
                            #print(big_community2)
                            break

                    # find all j's where the count has to be written in as well

                    weak_recombination_list = []
                    #print(recombinations_all)
                    for h in range(len(recombinations_all)):
                        #print(recombinations_all[h][0])
                        #print(big_community1)
                        #print(big_community2)
                        if recombinations_all[h][0] in big_community1:
                            #print(recombinations_all[h][1])
                            #print(big_community2)
                            if recombinations_all[h][1] in big_community2:
                                #print(h)
                                weak_recombination_list.append(h)
                        elif recombinations_all[h][0] in big_community2:
                            #print(recombinations_all[h][1])
                            #print(big_community1)
                            if recombinations_all[h][1] in big_community1:
                                #print(h)
                                weak_recombination_list.append(h)

                    #if big_community1 != False:
                        #if big_community2 != False:

                    for weak_recombination_pos in weak_recombination_list:
                        #print(weak_recombination_pos)
                        #print(recombination_count)
                        recombinationDiffusion_count[i, weak_recombination_pos] = recombination_count
                        #print(recombinationDiffusion_count[i, weak_recombination_pos])

                    '''
                    # window_300': [[286963357, ((289337379, [5]), (287698910, [8])), 1, 1]]
                    window_community = cd_recombinations['window_{0}'.format(i*30)]
                    window_recomb = cd_recombinations['window_{0}'.format(i*30)]
                    print(window)
                    print(window.values())
                    print(list(window.values()))
                    print(j)
    
                    for k in range(len(list(window.values()))):
                        if j in list(window.values())[k]:
                            big_community = list(window.values())[k]
                            break
                    print(big_community)
    
                    overall_count = 0
    
                    community_id1 = recombinations_all[j][0]
                    community_id2 = recombinations_all[j][1]
    
                    for k in range(len(list(window_community.values()))):
                        if community_id1 in list(window_community.values())[k]:
                            big_community1 = list(window_community.values())[k]
                            break
    
                    for k in range(len(list(window_community.values()))):
                        if community_id2 in list(window_community.values())[k]:
                            big_community2 = list(window_community.values())[k]
                            break
    
                    for recombination in window_recomb:
                        if recombination[1][0][1][0] in big_community1:
                            if recombination[1][1][1][0] in big_community2:
                                overall_count = overall_count + 1
                                break
                        elif recombination[1][0][1][0] in big_community2:
                            if recombination[1][1][1][0] in big_community1:
                                overall_count = overall_count + 1
                                break
    
                    recombinationDiffusion_count[i,j] = overall_count
    
                    #for recombination in window_recomb:
    
                    #           big_community1 = community_list where tuple[0] is present (for this window)
                    #           big_community2 = community_list where tuple[1] is present (for this window)
                    #           overall_count = 0
                    #           for all recombinations:
                    #               if recombination[0] in big_community1:
                    #                   if recombination[1] in big_community2:
                    #                       overall_count = overall_count + 1
                    #                       break
                    #               elif: recombination[0] in big_community2:
                    #                   if recombination[1] in big_community1:
                    #                       overall_count = overall_count + 1
                    #                       break
    
                    '''

            pbar.update(1)
        pbar.close()

        for n in range(len(topicSim)):
            #if n == 50:
                #print(1+1)
            all_nodes_window = len(topicSim['window_{0}'.format(n * 30)].nodes())
            #print(all_nodes_window)
            #print(recombinationDiffusion_count[n,:])
            recombinationDiffusion_fraction[n,:] = recombinationDiffusion_count[n,:] / all_nodes_window
            #print(recombinationDiffusion_fraction[n,:])

        recombinationDiffusion_threshold = np.where(recombinationDiffusion_fraction < 0.005, 0, 1)


        return recombinationDiffusion_count, recombinationDiffusion_fraction, recombinationDiffusion_threshold


    def recombination_diffusion_overlapping_v2(cd_topD_dic_clean, cd_recombinations):
        row_length = len(cd_recombinations)

        recombinations_dic = {}
        recombinations_all = []
        for window_id, window in cd_recombinations.items():
            recombinations_window = []
            for recombination in window:
                community_id1 = recombination[1][0][1][0]
                community_id2 = recombination[1][1][1][0]
                recombinations_all.append((community_id1, community_id2))
                recombinations_window.append((community_id1, community_id2))

            recombinations_dic[window_id] = recombinations_window
        # print(recombinations_dic)

        # print(len(recombinations_all))
        recombinations_all.sort()
        # print(recombinations_all)
        recombinations_all = np.unique(recombinations_all, axis=0)
        # print(recombinations_all)
        recombinations_all_tuple = []
        for recombination in recombinations_all:
            recombinations_all_tuple.append(tuple(recombination))
        recombinations_all = recombinations_all_tuple
        # print(len(recombinations_all))

        # print(recombinations_all)
        column_length = len(recombinations_all)

        recombinationDiffusion_count = np.zeros((row_length, column_length), dtype=int)
        # recombinationDiffusion_count = np.full((row_length, column_length), 9999999, dtype=int)
        recombinationDiffusion_fraction = np.zeros((row_length, column_length), dtype=float)
        recombinationDiffusion_threshold = np.zeros((row_length, column_length), dtype=float)

        pbar = tqdm.tqdm(total=len(recombinationDiffusion_count))
        for i in range(len(recombinationDiffusion_count)):
            # if i == 10:
            # print(1+1)
            for j in range(len(recombinationDiffusion_count.T)):

                window_topD_dic = cd_topD_dic_clean['window_{0}'.format(i * 30)]
                # print(window_topD_dic)

                # count how often a recombination appears in a window
                # Recombinations are identified over community id. These community id's are dominant.
                # print(recombinations_all[j])
                # print(recombinations_dic['window_{0}'.format(i*30)])
                recombination_count = recombinations_dic['window_{0}'.format(i * 30)].count(recombinations_all[j])
                # print(recombination_count)

                if recombination_count != 0:
                    # this count has to be placed in all columns that are the same recombination under different community ids
                    # (e.g. because of a community merge where the dominant id overwrite the original one used in the prior recombination

                    # print(list(window_topD_dic.values()))
                    for k in range(len(list(window_topD_dic.values()))):
                        # print(recombinations_all[j][0])
                        # print(list(window_topD_dic.values())[k])
                        if recombinations_all[j][0] in list(window_topD_dic.values())[k]:
                            big_community1 = list(window_topD_dic.values())[k]
                            # print(big_community1)
                            break

                    list(window_topD_dic.values())
                    for k in range(len(list(window_topD_dic.values()))):
                        # print(recombinations_all[j][1])
                        # print(list(window_topD_dic.values())[k])
                        if recombinations_all[j][1] in list(window_topD_dic.values())[k]:
                            big_community2 = list(window_topD_dic.values())[k]
                            # print(big_community2)
                            break

                    # find all j's where the count has to be written in as well

                    weak_recombination_list = []
                    # print(recombinations_all)
                    for h in range(len(recombinations_all)):
                        # print(recombinations_all[h][0])
                        # print(big_community1)
                        # print(big_community2)
                        if recombinations_all[h][0] in big_community1:
                            # print(recombinations_all[h][1])
                            # print(big_community2)
                            if recombinations_all[h][1] in big_community2:
                                # print(h)
                                weak_recombination_list.append(h)
                        elif recombinations_all[h][0] in big_community2:
                            # print(recombinations_all[h][1])
                            # print(big_community1)
                            if recombinations_all[h][1] in big_community1:
                                # print(h)
                                weak_recombination_list.append(h)

                    # if big_community1 != False:
                    # if big_community2 != False:

                    for weak_recombination_pos in weak_recombination_list:
                        # print(weak_recombination_pos)
                        # print(recombination_count)
                        recombinationDiffusion_count[i, weak_recombination_pos] = recombination_count
                        # print(recombinationDiffusion_count[i, weak_recombination_pos])


            pbar.update(1)
        pbar.close()

        for n in range(len(topicSim)):
            # if n == 50:
            # print(1+1)
            all_nodes_window = len(topicSim['window_{0}'.format(n * 30)].nodes())
            # print(all_nodes_window)
            # print(recombinationDiffusion_count[n,:])
            recombinationDiffusion_fraction[n, :] = recombinationDiffusion_count[n, :] / all_nodes_window
            # print(recombinationDiffusion_fraction[n,:])

        recombinationDiffusion_threshold = np.where(recombinationDiffusion_fraction < 0.005, 0, 1)

        return recombinationDiffusion_count, recombinationDiffusion_fraction, recombinationDiffusion_threshold

    #print(lp_recombinations)
    #print(lp_recombinations)

    lp_recombination_diffusion_crip_count_v2, lp_recombination_diffusion_crip_fraction_v2, lp_recombination_diffusion_crip_threshold_v2 = recombination_diffusion_crip_v2(lp_topD_dic_clean, lp_recombinations)
    #print(lp_recombination_diffusion_crip_v2)
    #print(lp_recombination_diffusion_crip_v2[0:99,0:25])
    print(lp_recombination_diffusion_crip_count_v2[40:50,25:50])
    print(lp_recombination_diffusion_crip_fraction_v2[40:50,25:50])
    print(lp_recombination_diffusion_crip_threshold_v2[40:50,25:50])
    #print(lp_recombination_diffusion_crip_v2[0:99,50:75])
    #print(lp_recombination_diffusion_crip_v2[0:99,75:100])

    #for i in range(40,50):
        #print(lp_recombinations['window_{0}'.format(i*30)])
        #print(lp_topD_dic_clean['window_{0}'.format(i * 30)])

    #for i in range(40,50):
        #print(lp_topD_dic_clean['window_{0}'.format(i*30)])

    #gm_recombination_diffusion_crip_v2 = recombination_diffusion_crip_v2(gm_topD_dic_clean)

    #kclique_recombination_diffusion_overlapping_v2 = recombination_diffusion_overlapping_v2(kclique_topD_dic_clean)
    kclique_recombination_diffusion_overlapping_count_v2, kclique_recombination_diffusion_overlapping_fraction_v2, kclique_recombination_diffusion_overlapping_threshold_v2 = recombination_diffusion_overlapping_v2(kclique_topD_dic_clean, kclique_recombinations)

#lais2_recombination_diffusion_overlapping_v2 = recombination_diffusion_overlapping_v2(lais2_topD_dic_clean)

    #print(lp_singleDiffusion_v2)
    #print(gm_singleDiffusion_v2)
    #print(kclique_singleDiffusion_v2)
    #print(lais2_singleDiffusion_v2)
