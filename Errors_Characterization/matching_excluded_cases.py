from utils import *
from ICL_matching import prepare_dataset, execute_fast_global_registration, execute_ICP
from scipy.ndimage import affine_transform
from tqdm import tqdm
from notifications import notify
from tqdm.contrib.concurrent import process_map
from skimage.measure import centroid
from matching_graphs import draw_matching_graph, save_matching_graph


def fine_tuning_registration(pair_path):

    def preprocess_liver_and_tumors(liver, tumors):
        tumors = pre_process_segmentation(tumors)
        liver = np.logical_or(liver, tumors).astype(liver.dtype)
        liver = getLargestCC(liver)
        liver = pre_process_segmentation(liver, remove_small_obs=False)
        tumors = np.logical_and(tumors, liver).astype(tumors.dtype)
        return liver, tumors

    # loading BL data
    bl_ct, nifti_file = load_nifti_data(f'{pair_path}/BL_Scan_CT.nii.gz')
    bl_liver, _ = load_nifti_data(f'{pair_path}/BL_Scan_Liver.nii.gz')
    bl_tumors, _ = load_nifti_data(f'{pair_path}/BL_Scan_Tumors.nii.gz')

    # loading FU data
    fu_liver, _ = load_nifti_data(f'{pair_path}/FU_Scan_Liver.nii.gz')
    fu_tumors, _ = load_nifti_data(f'{pair_path}/FU_Scan_Tumors.nii.gz')

    assert is_a_scan(bl_ct)
    assert is_a_mask(bl_liver)
    assert is_a_mask(bl_tumors)
    assert is_a_mask(fu_liver)
    assert is_a_mask(fu_tumors)

    bl_liver, bl_tumors = preprocess_liver_and_tumors(bl_liver, bl_tumors)
    fu_liver, fu_tumors = preprocess_liver_and_tumors(fu_liver, fu_tumors)

    bl_tumors = get_connected_components(bl_tumors)
    fu_tumors = get_connected_components(fu_tumors)

    n_bl_tumors = np.unique(bl_tumors).size - 1
    n_fu_tumors = np.unique(fu_tumors).size - 1

    (bl_down, fu_down, bl_fpfh, fu_fpfh, bl_tumors_pc, fu_tumors_pc, relevant_bl_tumors_pc,
     relevant_fu_tumors_pc, ICP_bl_pc, ICP_fu_pc) = prepare_dataset(bl_tumors, fu_tumors,
                                                                    bl_liver=bl_liver, fu_liver=fu_liver,
                                                                    file_affine_matrix=nifti_file.affine,
                                                                    voxelspacing=nifti_file.header.get_zooms(),
                                                                    voxel_size=1,
                                                                    center_of_mass_for_RANSAC=True, n_biggest=None,
                                                                    ICP_with_liver_border=True,
                                                                    RANSAC_with_liver_border=True,
                                                                    RANSAC_with_tumors=False,
                                                                    ICP_with_tumors=False)
    result_ransac = execute_fast_global_registration(bl_down, fu_down, bl_fpfh, fu_fpfh, 1)
    result_icp = execute_ICP(ICP_bl_pc, ICP_fu_pc, 1, distance_threshold_factor=40,
                             init_transformation=result_ransac.transformation)

    # transform the BL data
    transform_inverse = np.linalg.inv(nifti_file.affine) @ np.linalg.inv(
        result_icp.transformation) @ nifti_file.affine
    transformed_bl_tumors = affine_transform(bl_tumors, transform_inverse, order=0)
    transformed_bl_liver = affine_transform(bl_liver, transform_inverse, order=0)
    transformed_bl_ct = affine_transform(bl_ct, transform_inverse)

    save(Nifti1Image(transformed_bl_ct.astype(np.float32), nifti_file.affine), f'{pair_path}/improved_registration_BL_Scan_CT.nii.gz')
    save(Nifti1Image(transformed_bl_liver.astype(np.float32), nifti_file.affine), f'{pair_path}/improved_registration_BL_Scan_Liver.nii.gz')
    save(Nifti1Image(transformed_bl_tumors.astype(np.float32), nifti_file.affine), f'{pair_path}/improved_registration_BL_Scan_Tumors_unique_{n_bl_tumors}_CC.nii.gz')

    save(Nifti1Image(bl_tumors.astype(np.float32), nifti_file.affine), f'{pair_path}/BL_Scan_Tumors_unique_{n_bl_tumors}_CC.nii.gz')
    save(Nifti1Image(fu_tumors.astype(np.float32), nifti_file.affine), f'{pair_path}/FU_Scan_Tumors_unique_{n_fu_tumors}_CC.nii.gz')


def match(pair_path):
    case_name = os.path.basename(pair_path)
    bl_tumors_file_path = glob(f'{pair_path}/improved_registration_BL_Scan_Tumors_unique_*')[0]
    fu_tumors_file_path = glob(f'{pair_path}/FU_Scan_Tumors_unique_*')[0]

    n_bl_tumors = int(''.join([c for c in os.path.basename(bl_tumors_file_path) if c.isdigit()]))
    n_fu_tumors = int(''.join([c for c in os.path.basename(fu_tumors_file_path) if c.isdigit()]))

    bl_tumors_CC_labeled, file = load_nifti_data(bl_tumors_file_path)
    fu_tumors_CC_labeled, _ = load_nifti_data(fu_tumors_file_path)

    pred_matches = match_2_cases_v3(bl_tumors_CC_labeled, fu_tumors_CC_labeled,
                                    voxelspacing=file.header.get_zooms(), max_dilate_param=9)

    pred_matches = [(int(m[0]), int(m[1])) for m in pred_matches]

    bl_weights = []
    for bl_t in range(1, n_bl_tumors + 1):
        cent = centroid(bl_tumors_CC_labeled == bl_t)
        if np.isnan(cent).any():
            bl_weights.append(-1)
            print(os.path.basename(pair_path), 'bl', f'label={bl_t}')
        else:
            bl_weights.append(int(cent[-1] + 1))

    fu_weights = []
    for fu_t in range(1, n_fu_tumors + 1):
        cent = centroid(fu_tumors_CC_labeled == fu_t)
        if np.isnan(cent).any():
            fu_weights.append(-1)
            print(os.path.basename(pair_path), 'fu', f'label={fu_t}')
        else:
            fu_weights.append(int(cent[-1] + 1))

    save_matching_graph(n_bl_tumors, n_fu_tumors, pred_matches, case_name, f'{pair_path}/pred_matching_graph.json',
                        bl_weights, fu_weights)

    draw_matching_graph(n_bl_tumors, n_fu_tumors, pred_matches, case_name, bl_weights, fu_weights,
                        saving_file_name=f'{pair_path}/pred_matching_graph.jpg')


def calculate_registration_scores(pair_path):

    # loading BL original data
    # bl_original_ct, _ = load_nifti_data(f'{pair_path}/BL_Scan_CT.nii.gz')
    bl_original_liver, nifti_file = load_nifti_data(f'{pair_path}/BL_Scan_Liver.nii.gz')
    # bl_original_tumors, _ = load_nifti_data(f'{pair_path}/BL_Scan_Tumors.nii.gz')
    assert is_a_mask(bl_original_liver)

    # loading BL improved-registration data
    # bl_improved_registration_ct, _ = load_nifti_data(f'{pair_path}/improved_registration_BL_Scan_CT.nii.gz')
    bl_improved_registration_liver, _ = load_nifti_data(f'{pair_path}/improved_registration_BL_Scan_Liver.nii.gz')
    # bl_improved_registration_tumors, _ = load_nifti_data(f'{pair_path}/improved_registration_BL_Scan_Tumors.nii.gz')
    assert is_a_mask(bl_improved_registration_liver)

    # loading FU data
    # fu_ct, _ = load_nifti_data(f'{pair_path}/FU_Scan_CT.nii.gz')
    fu_liver, _ = load_nifti_data(f'{pair_path}/FU_Scan_Liver.nii.gz')
    # fu_tumors, _ = load_nifti_data(f'{pair_path}/FU_Scan_Tumors.nii.gz')
    assert is_a_mask(fu_liver)

    voxelspacing = nifti_file.header.get_zooms()
    voxel_volume = voxelspacing[0] * voxelspacing[1] * voxelspacing[2]

    def registration_scores(_bl_liver, _fu_liver):
        liver_dice = dice(_bl_liver, _fu_liver)
        liver_assd, liver_hd = assd_and_hd(_bl_liver, _fu_liver, voxelspacing)
        liver_abs_volume_diff = voxel_volume * abs(_bl_liver.sum() - _fu_liver.sum()) / 1000
        return liver_dice, liver_assd, liver_hd, liver_abs_volume_diff

    original_res = registration_scores(bl_original_liver, fu_liver)
    improved_res = registration_scores(bl_improved_registration_liver, fu_liver)

    pair_name = os.path.basename(pair_path)

    return (pair_name, *(y for x in zip(original_res, improved_res) for y in x))


def write_registration_scores(pairs_paths: List[str], n_processes=None,
                              results_file: Optional[str] = '/cs/casmip/rochman/Errors_Characterization/excluded_set/measures_results/improved_registration_scores.xlsx'):

    if n_processes is None:
        n_processes = os.cpu_count() - 2

    res = process_map(calculate_registration_scores, pairs_paths, max_workers=n_processes)
    # res = list(map(calculate_registration_scores, pairs_paths))

    columns = ['Name',
               'Dice - Original', 'Dice - Improved',
               'ASSD - Original (mm)', 'ASSD - Improved (mm)',
               'HD - Original (mm)', 'HD - Improved (mm)',
               'ABS Volume-diff - Original (CC)', 'ABS Volume-diff - Improved (CC)']
    res = pd.DataFrame(res, columns=columns)

    res = sort_dataframe_by_key(res, 'Name', key=pairs_sort_key)

    res['Original Dice >= 0.7'] = False
    res.loc[res['Dice - Original'] >= 0.7, 'Original Dice >= 0.7'] = True

    res['Improved Dice >= 0.7'] = False
    res.loc[res['Dice - Improved'] >= 0.7, 'Improved Dice >= 0.7'] = True

    res['Original ASSD <= 9'] = False
    res.loc[res['ASSD - Original (mm)'] <= 9, 'Original ASSD <= 9'] = True

    res['Improved ASSD <= 9'] = False
    res.loc[res['ASSD - Improved (mm)'] <= 9, 'Improved ASSD <= 9'] = True

    res['Original HD <= 90'] = False
    res.loc[res['HD - Original (mm)'] <= 90, 'Original HD <= 90'] = True

    res['Improved HD <= 90'] = False
    res.loc[res['HD - Improved (mm)'] <= 90, 'Improved HD <= 90'] = True

    res['Original ABS Volume-diff <= 500'] = False
    res.loc[res['ABS Volume-diff - Original (CC)'] <= 500, 'Original ABS Volume-diff <= 500'] = True

    res['Improved ABS Volume-diff <= 500'] = False
    res.loc[res['ABS Volume-diff - Improved (CC)'] <= 500, 'Improved ABS Volume-diff <= 500'] = True

    res['Valid - Original'] = res['Original Dice >= 0.7'] & res['Original ASSD <= 9'] & res['Original HD <= 90'] & res['Original ABS Volume-diff <= 500']
    res['Valid - Improved'] = res['Improved Dice >= 0.7'] & res['Improved ASSD <= 9'] & res['Improved HD <= 90'] & res['Improved ABS Volume-diff <= 500']

    resuls_dir_name = os.path.dirname(results_file)
    os.makedirs(resuls_dir_name, exist_ok=True)

    writer = pd.ExcelWriter(results_file, engine='xlsxwriter')

    columns = ['Name',
               'Dice - Original', 'Dice - Improved',
               'ASSD - Original (mm)', 'ASSD - Improved (mm)',
               'HD - Original (mm)', 'HD - Improved (mm)',
               'ABS Volume-diff - Original (CC)', 'ABS Volume-diff - Improved (CC)',
               'Original Dice >= 0.7', 'Original ASSD <= 9', 'Original HD <= 90', 'Original ABS Volume-diff <= 500', 'Valid - Original',
               'Improved Dice >= 0.7', 'Improved ASSD <= 9', 'Improved HD <= 90', 'Improved ABS Volume-diff <= 500', 'Valid - Improved']
    write_to_excel(res, writer, columns, 'Name')
    writer.save()


if __name__ == '__main__':
    # # load registration-scores
    # registration_scores_df = pd.read_excel('/cs/casmip/rochman/Errors_Characterization/registration_scores.xlsx')
    # registration_scores_df = registration_scores_df[registration_scores_df.columns[1:]]
    # registration_scores_df = registration_scores_df[registration_scores_df['name'].str.startswith('BL_')]
    #
    # # extract the excluded pairs only
    # excluded_pairs_names = registration_scores_df[registration_scores_df['Valid'] == 0]['name'].to_list()
    pass
    excluded_set_dir = '/cs/casmip/rochman/Errors_Characterization/excluded_set'
    # excluded_set_dir = '/cs/casmip/rochman/Errors_Characterization/corrected_segmentation_for_matching'
    pass
    # # create excluded set
    # os.makedirs(excluded_set_dir, exist_ok=True)
    # for i, pair_name in enumerate(excluded_pairs_names):
    #     print(i + 1)
    #     symlink_for_inner_files_in_a_dir(f'/mnt/sda1/aszeskin/Data_Followup_Full_29_4_2021/{pair_name}',
    #                                      f'{excluded_set_dir}/{pair_name}')

    # # fine-tuning registration over excluded set
    # try:
    #     t = time()
    #     for pair_path in tqdm(sorted(glob(f'{excluded_set_dir}/BL_*'))):
    #         fine_tuning_registration(pair_path)
    #     notify(f"The registration fine-tuning for the excluded-set finished successfully in: {calculate_runtime(t)} (hh:mm:ss)")
    #
    # except Exception as e:
    #     notify(f"There was an error while running the registration fine-tuning for the excluded-set: {e}",
    #            error=True)
    #     raise e
    # exit(0)

    # # calculate and write registration-scores
    # try:
    #     write_registration_scores(sorted(glob(f'{excluded_set_dir}/BL_*')))
    #     # write_registration_scores(sorted(glob(f'{excluded_set_dir}/BL_*')), results_file='/cs/casmip/rochman/Errors_Characterization/corrected_segmentation_for_matching/measures_results/improved_registration_scores.xlsx')
    #     # notify(f"Calculating the registration-scores finished successfully")
    #
    # except Exception as e:
    #     notify(f"There was an error while calculating the registration-scores: {e}",
    #            error=True)
    #     raise e
    # exit(0)

    relevant_pairs_for_matching_dir = '/cs/casmip/rochman/Errors_Characterization/excluded_set/relevant_pairs_for_matching'

    # df = pd.read_excel(f'/cs/casmip/rochman/Errors_Characterization/excluded_set/measures_results/improved_registration_scores.xlsx')
    # df.rename(columns={df.columns[0]: 'case_name'}, inplace=True)
    # df = df[df['case_name'].str.startswith('BL_')]
    #
    # not_valid_df = df[df['Valid - Improved'] == 0].reset_index(drop=True)
    # relevant_from_not_valid = not_valid_df[not_valid_df['Improved ABS Volume-diff <= 500'] == 1].reset_index(drop=True)
    #
    #
    # os.makedirs(f'{relevant_pairs_for_matching_dir}/measures_results', exist_ok=True)
    # writer = pd.ExcelWriter(f'{relevant_pairs_for_matching_dir}/measures_results/improved_registration_scores.xlsx', engine='xlsxwriter')
    # write_to_excel(relevant_from_not_valid, writer, relevant_from_not_valid.columns, 'case_name')
    # writer.save()
    #
    # for relevant_pair in relevant_from_not_valid['case_name']:
    #     src_dir = f'{excluded_set_dir}/{relevant_pair}'
    #     dst_dir = f'{relevant_pairs_for_matching_dir}/{relevant_pair}'
    #     os.makedirs(dst_dir, exist_ok=True)
    #     symlink_for_inner_files_in_a_dir(src_dir, dst_dir)
    # exit(0)


    pairs_paths = sorted(glob(f'{relevant_pairs_for_matching_dir}/BL_*'))
    # print(pairs_paths)
    process_map(match, pairs_paths, max_workers=os.cpu_count()-2)
    # list(map(match, pairs_paths))