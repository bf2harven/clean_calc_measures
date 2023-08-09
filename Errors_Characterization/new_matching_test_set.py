from utils import *
from matching_excluded_cases import fine_tuning_registration, match
from tqdm import tqdm
from notifications import notify
from tqdm.contrib.concurrent import process_map
from datetime import date
from matching_graphs import load_matching_graph


def patient_BL_and_FU(pair_name):
        pair_name = pair_name.replace('BL_', '')
        bl, fu = pair_name.split('_FU_')
        patient = '_'.join(c for c in bl.split('_') if not c.isdigit())
        return patient, bl, fu


def sort_key(name):
    split = name.split('_')
    return '_'.join(c for c in split if not c.isdigit()), int(split[-1]), int(split[-2]), int(split[-3])


def is_in_order(name1, name2):
    key1 = sort_key(name1)
    key2 = sort_key(name2)
    if key1[1] > key2[1]:
        return False
    if key1[1] == key2[1] and key1[2] > key2[2]:
        return False
    if key1[1] == key2[1] and key1[2] == key2[2] and key1[3] > key2[3]:
        return False
    return True


def diff_in_days(bl_name, fu_name):
    _, bl_y, bl_m, bl_d = sort_key(bl_name)
    _, fu_y, fu_m, fu_d = sort_key(fu_name)
    bl_date = date(bl_y, bl_m, bl_d)
    fu_date = date(fu_y, fu_m, fu_d)
    return abs((fu_date - bl_date).days)


def score_for_pair(bl_tumors: np.ndarray, fu_tumors: np.ndarray, bl_voxel_to_real_space_trans, fu_voxel_to_real_space_trans) -> float:
    return 1 / (min(get_minimum_distance_between_CCs(bl_tumors, bl_voxel_to_real_space_trans, max_points_per_CC=5000, seed=42),
                    get_minimum_distance_between_CCs(fu_tumors, fu_voxel_to_real_space_trans, max_points_per_CC=5000, seed=42)) + 1e-5)


def calc_score_for_pairs(pair_path):
    bl_tumors_file = f'{pair_path}/BL_Scan_Tumors.nii.gz'
    fu_tumors_file = f'{pair_path}/FU_Scan_Tumors.nii.gz'

    bl_liver_file = f'{pair_path}/BL_Scan_Liver.nii.gz'
    fu_liver_file = f'{pair_path}/FU_Scan_Liver.nii.gz'

    bl_tumors, bl_file = load_nifti_data(bl_tumors_file)
    fu_tumors, fu_file = load_nifti_data(fu_tumors_file)

    bl_liver, _ = load_nifti_data(bl_liver_file)
    fu_liver, _ = load_nifti_data(fu_liver_file)

    liver_vol_diff = abs((bl_liver > 0).sum() * np.prod(bl_file.header.get_zooms()) - (fu_liver > 0).sum() * np.prod(fu_file.header.get_zooms())) / 1000

    case_name = os.path.basename(pair_path)
    different_in_days = diff_in_days(*patient_BL_and_FU(case_name)[1:])

    return (case_name, liver_vol_diff, different_in_days, score_for_pair(bl_tumors, fu_tumors, bl_file.affine, fu_file.affine))


if __name__ == '__main__':


    # training_pairs = pd.read_excel('/cs/casmip/rochman/Errors_Characterization/data/all_data_pairs_data_measures.xlsx')
    # max_num_of_tumors = 40
    # min_num_of_tumors = 1
    # training_pairs = training_pairs[(training_pairs['BL_n_tumors'] <= max_num_of_tumors) & (training_pairs['FU_n_tumors'] <= max_num_of_tumors) &
    #                                 (training_pairs['BL_n_tumors'] >= min_num_of_tumors) & (training_pairs['FU_n_tumors'] >= min_num_of_tumors)]
    # training_pairs = training_pairs[training_pairs[training_pairs.columns[0]].str.startswith('BL_')][[training_pairs.columns[0], 'BL_n_tumors', 'FU_n_tumors']]
    # training_pairs = [(*patient_BL_and_FU(pair_name), int(n_bl_t), int(n_fu_t)) for (_, (pair_name, n_bl_t, n_fu_t)) in training_pairs.iterrows()]
    # training_pairs.sort(key=lambda p: (p[0], sort_key(p[1]), sort_key(p[2])))
    # training_pairs = [pair_name for pair_name in training_pairs if is_in_order(pair_name[1], pair_name[2])]
    # training_pairs = pd.DataFrame(training_pairs, columns=['patient', 'bl', 'fu', 'n_bl_t', 'n_fu_t'])
    # training_pairs['Diff in days'] = training_pairs.apply(lambda r: diff_in_days(r['bl'], r['fu']), axis=1)
    # training_pairs = training_pairs[training_pairs['Diff in days'] <= 730]
    # training_pairs['case_name'] = training_pairs.apply(lambda r: f'BL_{r["bl"]}_FU_{r["fu"]}', axis=1)
    # matching_test_set = pd.read_excel('/cs/casmip/rochman/Errors_Characterization/matching/corrected_measures_results/matching_statistics_dilate_13.xlsx')
    # training_pairs = training_pairs[~training_pairs['case_name'].isin(matching_test_set[matching_test_set.columns[0]])]
    #
    # # deleting an extremely non-contrast case
    # training_pairs = training_pairs[~training_pairs['case_name'].str.contains('A_S_S_25_08_2016')]
    # training_pairs = training_pairs[~training_pairs['case_name'].str.contains('C_A_05_12_2019')]
    #
    # # deleting a non-tumors case
    # training_pairs = training_pairs[~training_pairs['case_name'].str.contains('T_N_07_05_2019')]
    #
    # dst_dir = '/cs/casmip/rochman/Errors_Characterization/new_test_set_for_matching'
    # os.makedirs(dst_dir, exist_ok=True)
    # src_dir = '/mnt/sda1/aszeskin/Data_Followup_Full_29_4_2021'
    # for case_name in training_pairs['case_name']:
    #     if not os.path.isfile(f'{dst_dir}/{case_name}/gt_matching_graph.json'):
    #         symlink_for_inner_files_in_a_dir(f'{src_dir}/{case_name}', f'{dst_dir}/{case_name}')
    # exit(0)

    pass
    test_set_dir = '/cs/casmip/rochman/Errors_Characterization/new_test_set_for_matching'

    # # fine-tuning registration over the new test set
    # try:
    #     t = time()
    #     for pair_path in tqdm(sorted(glob(f'{test_set_dir}/BL_*'))):
    #         if not os.path.isfile(f'{pair_path}/gt_matching_graph.json'):
    #             fine_tuning_registration(pair_path)
    #     notify(f"The registration fine-tuning for the excluded-set finished successfully in: {calculate_runtime(t)} (hh:mm:ss)")
    #
    # except Exception as e:
    #     notify(f"There was an error while running the registration fine-tuning for the excluded-set: {e}",
    #            error=True)
    #     raise e
    # exit(0)

    pass

    # pairs_paths = sorted(glob(f'{test_set_dir}/BL_*'))
    # pairs_paths = [p for p in pairs_paths if not os.path.isfile(f'{p}/gt_matching_graph.json')]
    # process_map(match, pairs_paths, max_workers=os.cpu_count() - 2)
    # # list(map(match, pairs_paths))
    # # for i, pair_path in enumerate(pairs_paths):
    # #     if pair_path != '/cs/casmip/rochman/Errors_Characterization/new_test_set_for_matching/BL_B_T_18_02_2019_FU_B_T_05_05_2019':
    # #         continue
    # #     print(f'{i+1}: {pair_path},', end=' ')
    # #     t = time()
    # #     match(pair_path)
    # #     print(f'finished in {calculate_runtime(t)} (hh:mm:ss)')
    # exit(0)

    pass

    # pairs_paths = sorted(glob(f'{test_set_dir}/BL_*'))
    # scores = process_map(calc_score_for_pairs, pairs_paths, max_workers=os.cpu_count() - 2)
    # # scores = list(map(calc_score_for_pairs, pairs_paths))
    # columns = ['case_name', 'liver vol diff (CC)', 'diff in days (Days)', 'Score']
    # scores = pd.DataFrame(scores, columns=columns)
    # scores.sort_values(by='Score', ascending=False, inplace=True)
    #
    # resuls_dir_name = f'{test_set_dir}/measures_results'
    # os.makedirs(resuls_dir_name, exist_ok=True)
    #
    # if os.path.isfile(f'{resuls_dir_name}/Scores.xlsx'):
    #     previous_scores = pd.read_excel(f'{resuls_dir_name}/Scores.xlsx')
    #     previous_scores.rename(columns={previous_scores.columns[0]: 'case_name'}, inplace=True)
    #     previous_scores = previous_scores[['case_name', 'Checked', 'Richard to recheck']]
    #     scores = pd.merge(scores, previous_scores, how='left', on='case_name')
    #
    # writer = pd.ExcelWriter(f'{resuls_dir_name}/Scores.xlsx', engine='xlsxwriter')
    #
    # write_to_excel(scores, writer, columns + ['Checked', 'Richard to recheck'], 'case_name')
    # writer.save()
    #
    # exit(0)

    pass

    # for i, file_path in enumerate(sorted(glob(f'{test_set_dir}/BL_*/*_Tumors_unique_*'))):
    #     print(i+1, file_path)
    #     c, f = load_nifti_data(file_path)
    #     save(Nifti1Image(c, f.affine), file_path)
    #
    # exit(0)

    pass

    # def treat_for_prefix(pair_path, prefix):
    #     files = glob(f'{pair_path}/{prefix}*')
    #     assert len(files) in [1, 2], f'len is {len(files)}'
    #     if len(files) == 2:
    #         file1, file2 = files
    #         n1 = int(''.join([c for c in os.path.basename(file1) if c.isdigit()]))
    #         n2 = int(''.join([c for c in os.path.basename(file2) if c.isdigit()]))
    #         t1 = os.path.getmtime(file1)
    #         t2 = os.path.getmtime(file2)
    #         assert t1 != t2
    #         if t1 > t2:
    #             os.remove(file2)
    #             print(f'{prefix}: from {n2} to {n1}')
    #         else:
    #             os.remove(file1)
    #             print(f'{prefix}: from {n1} to {n2}')
    #
    #
    # pairs_paths = sorted(glob(f'/cs/casmip/rochman/Errors_Characterization/new_test_set_for_matching/BL_*'))
    # for i, p in enumerate(pairs_paths):
    #     print('-------------------------------------------------------------------')
    #     print(os.path.basename(p))
    #     treat_for_prefix(p, 'BL_Scan_Tumors_unique_')
    #     treat_for_prefix(p, 'improved_registration_BL_Scan_Tumors_unique_')
    #     treat_for_prefix(p, 'FU_Scan_Tumors_unique_')
    # exit(0)

    pass

    previous_scores = pd.read_excel(f'{test_set_dir}/measures_results/Scores.xlsx')
    previous_scores.rename(columns={previous_scores.columns[0]: 'name'}, inplace=True)
    pairs_for_Richard = previous_scores[previous_scores['Richard to recheck'] == 'yes']['name'].to_list()

    src_dir = test_set_dir
    dst_dir = '/cs/casmip/rochman/Errors_Characterization/matching_for_richard/round_5'
    for i, pair_name in enumerate(pairs_for_Richard):
        print(i+1, pair_name)
        os.makedirs(f'{dst_dir}/{pair_name}', exist_ok=True)

        fu_scan = f'{src_dir}/{pair_name}/FU_Scan_CT.nii.gz'
        assert os.path.isfile(fu_scan), pair_name
        os.symlink(fu_scan, f'{dst_dir}/{pair_name}/FU_Scan_CT.nii.gz')

        fu_tumors = glob(f'{src_dir}/{pair_name}/FU_Scan_Tumors_unique_*')
        assert len(fu_tumors) == 1, pair_name
        fu_tumors = fu_tumors[0]
        os.symlink(fu_tumors, f'{dst_dir}/{pair_name}/{os.path.basename(fu_tumors)}')

        bl_scan = f'{src_dir}/{pair_name}/improved_registration_BL_Scan_CT.nii.gz'
        assert os.path.isfile(bl_scan), pair_name
        os.symlink(bl_scan, f'{dst_dir}/{pair_name}/improved_registration_BL_Scan_CT.nii.gz')

        bl_tumors = glob(f'{src_dir}/{pair_name}/improved_registration_BL_Scan_Tumors_unique_*')
        assert len(bl_tumors) == 1, pair_name
        bl_tumors = bl_tumors[0]
        os.symlink(bl_tumors, f'{dst_dir}/{pair_name}/{os.path.basename(bl_tumors)}')

        gt_matching_graph_jpg = f'{src_dir}/{pair_name}/gt_matching_graph.jpg'
        assert os.path.isfile(gt_matching_graph_jpg), pair_name
        os.symlink(gt_matching_graph_jpg, f'{dst_dir}/{pair_name}/matching_graph.jpg')

        gt_matching_graph_json = f'{src_dir}/{pair_name}/gt_matching_graph.json'
        assert os.path.isfile(gt_matching_graph_json), pair_name
        _, _, edges, case_name, _, _, _, _, _, _ = load_matching_graph(gt_matching_graph_json)
        assert case_name == pair_name, pair_name
        df = pd.DataFrame(edges, columns=['bl tumors', 'fu tumors'])
        writer = pd.ExcelWriter(f'{dst_dir}/{pair_name}/matching.xlsx', engine='xlsxwriter')
        df.to_excel(writer)
        writer.save()




