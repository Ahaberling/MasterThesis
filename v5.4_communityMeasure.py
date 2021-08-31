if __name__ == '__main__':

    # --- Import Libraries ---#
    print('\n#--- Import Libraries ---#\n')

    import pickle as pk




    import tqdm
    import os

    # --- Initialization ---#
    print('\n# --- Initialization ---#\n')

    os.chdir('D:/Universitaet Mannheim/MMDS 7. Semester/Master Thesis/Outline/Data/Cleaning Robots')

    with open('patentProject_graphs', 'rb') as handle:
        patentProject_graphs = pk.load(handle)

    # --- Applying Community detection to each graph/window and populate respective dictionaries ---#

    ### Creating dictionaries to save communities ###

    from utilities.my_measure_utils import CommunityMeasures

    community_dict = CommunityMeasures.detect_communities(patentProject_graphs, cD_algorithm='greedy_modularity')


    # --- Transform data structure ---#

    # greedy_modularity #
    gm_transf = {}

    for window_id, window in community_dict.items():
        community_list = []
        focal_commu = []
        c = 0

        for patent_id, community_id in window.items():
            if community_id[0] == c:
                focal_commu.append(patent_id)
            else:
                community_list.append(focal_commu)
                focal_commu = []
                focal_commu.append(patent_id)
                c = c + 1

        gm_transf[window_id] = community_list

    # kclique #

    kclique_transf = {}

    for window_id, window in kclique_PlainCommu.items():
        community_list = []
        max_commu_counter = []

        for patent_id, community_id in window.items():
            max_commu_counter.append(len(community_id))

        if len(max_commu_counter) >= 1:
            max_commu_counter = max(max_commu_counter)

            for j in range(max_commu_counter + 1):
                focal_commu = []

                for patent_id, community_id in window.items():
                    if j in community_id:
                        focal_commu.append(patent_id)

                community_list.append(focal_commu)

        else:
            community_list.append([])

        kclique_transf[window_id] = community_list

    # lais2 #

    lais2_transf = {}

    for window_id, window in lais2_PlainCommu.items():
        community_list = []
        max_commu_counter = []

        for patent_id, community_id in window.items():
            max_commu_counter.append(len(community_id))

        max_commu_counter = max(max_commu_counter)

        for j in range(max_commu_counter + 1):
            focal_commu = []

            for patent_id, community_id in window.items():
                if j in community_id:
                    focal_commu.append(patent_id)

            community_list.append(focal_commu)

        lais2_transf[window_id] = community_list


    # --- Clean Communties ---#

    def community_cleaning(CD_dic):
        clean_CD = {}
        for window_id, window in CD_dic.items():
            clean_CD[window_id] = [x for x in window if len(x) >= 3]
        return clean_CD


    # Label Propagation #
    lp_clean = community_cleaning(lp_PlainCommu)

    # Greedy Modularity #
    gm_clean = community_cleaning(gm_transf)

    # Kclique #
    # K already discriminates for communities of size < 3. Function is
    kclique_clean = community_cleaning(kclique_transf)

    # Lais2 #
    lais2_clean = community_cleaning(lais2_transf)

    # --- Save Communties ---#

    filename = 'lp_clean'
    outfile = open(filename, 'wb')
    pk.dump(lp_clean, outfile)
    outfile.close()

    filename = 'gm_clean'
    outfile = open(filename, 'wb')
    pk.dump(gm_clean, outfile)
    outfile.close()

    filename = 'kclique_clean'
    outfile = open(filename, 'wb')
    pk.dump(kclique_clean, outfile)
    outfile.close()

    filename = 'lais2_clean'
    outfile = open(filename, 'wb')
    pk.dump(lais2_clean, outfile)
    outfile.close()
