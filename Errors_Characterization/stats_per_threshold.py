from glob import glob
from utils import *
from os.path import basename, dirname
from multiprocessing import Manager
from tqdm.contrib.concurrent import process_map



temp_files = Manager().dict()
# temp_files = dict()


def calc_dice_for_threshold(liver_gt_pred_and_th: Tuple[str, str, str, int], max_possible_label: int = 16) -> Tuple[str, int, List[Tuple[float, Tuple[int, int, int], float]]]:
    """
    Calculating the dice for each GT tumor with all the predicted tumors by the given threshold.

    :param liver_gt_pred_and_th: a tuple in the following form: (liver_file_name, gt_tumors_file_name,
        label_pred_tumors_file_name, threshold).
    :param max_possible_label: a integer indicating the maximum possible label in the pred_tumors_file.

    :return: a tuple in the following form: (case_name, threshold, tumors_measures) where:
        • case_name is the name of the case.
        • threshold is the threshold used in the predicted tumors.
        • tumors_measures is a list containing for each GY tumor a tuple in the following form:
            (gt_tumor_diameter, gt_tumor_center_of_mass, dice) where:
                • gt_tumor_diameter is the diameter of the current GT tumor in mm.
                • gt_tumor_center_of_mass is a tuple containing the (x,y,z) coordinate of the center of mass of the
                    current GT tumor.
                • dice is the dice of the current GT tumor with the prediction tumors (by the given threshold).
    """
    case_name = basename(dirname(liver_gt_pred_and_th[0]))

    th = liver_gt_pred_and_th[-1]

    global temp_files
    if case_name in temp_files:
        liver_case, gt_tumors_unique_labels = temp_files[case_name]
    else:
        # loading the files
        (liver_case, _), (gt_tumors_case, _) = (load_nifti_data(file_name) for file_name in liver_gt_pred_and_th[:2])

        assert is_a_mask(liver_case)
        assert is_a_mask(gt_tumors_case)

        # pre-process the tumors and liver segmentations
        gt_tumors_case = pre_process_segmentation(gt_tumors_case)
        liver_case = np.logical_or(liver_case, gt_tumors_case).astype(liver_case.dtype)
        liver_case = getLargestCC(liver_case)
        liver_case = pre_process_segmentation(liver_case, remove_small_obs=False)
        gt_tumors_case = np.logical_and(liver_case, gt_tumors_case).astype(gt_tumors_case.dtype)

        gt_tumors_unique_labels = get_connected_components(gt_tumors_case)

        temp_files[case_name] = liver_case, gt_tumors_unique_labels

    pred_tumors_case, nifti_file = load_nifti_data(liver_gt_pred_and_th[2])

    assert is_a_labeled_mask(pred_tumors_case, range(max_possible_label + 1))

    # thresholding
    pred_tumors_case = (pred_tumors_case >= th).astype(pred_tumors_case.dtype)

    # correcting to ROI
    pred_tumors_case = np.logical_and(pred_tumors_case, liver_case)

    # preprocessing
    pred_tumors_case = pre_process_segmentation(pred_tumors_case, remove_small_obs=False)

    pix_dims = nifti_file.header.get_zooms()
    voxel_volume = pix_dims[0] * pix_dims[1] * pix_dims[2]

    pred_tumors_unique_labels = get_connected_components(pred_tumors_case)

    res = []

    for gt_tumor_label in np.unique(gt_tumors_unique_labels)[1:]:
        current_gt_tumor = (gt_tumors_unique_labels == gt_tumor_label).astype(gt_tumors_unique_labels.dtype)
        current_gt_tumor_diameter = approximate_diameter(current_gt_tumor.sum() * voxel_volume)
        current_gt_tumor_center_of_mass = get_center_of_mass(current_gt_tumor)
        pred_tumors_labels_touch = np.unique(current_gt_tumor * pred_tumors_unique_labels)
        pred_tumors_touch = np.zeros_like(current_gt_tumor)
        for pred_tumor_label in np.unique(pred_tumors_labels_touch)[1:]:
            current_pred_tumor = (pred_tumors_unique_labels == pred_tumor_label)
            pred_tumors_touch[current_pred_tumor] = 1
        res.append((current_gt_tumor_diameter, current_gt_tumor_center_of_mass, dice(current_gt_tumor, pred_tumors_touch)))

    return case_name, th, res


def write_stats_per_threshold(livers: List[str], gt_tumors: List[str], pred_tumors: List[str],
                              results_file: str = 'measures_results/stats_per_th.xlsx'):
    data = []
    n_cases = len(gt_tumors)
    for th in range(1, 17):
        data += list(zip(livers, gt_tumors, pred_tumors, [th] * n_cases))

    res = process_map(calc_dice_for_threshold, data)

    res = [('_-_'.join(item[0] + (str(item[2][1]),)), item[1], item[2][0], item[2][-1])
           for tup in (tuple(zip(zip([t[0]] * len(t[2]), (str(j) for j in range(1, len(t[2]) + 1))), [t[1]] * len(t[2]), t[2])) for t in res)
           for item in tup]

    columns = ['name', 'th', 'diameter', 'dice']
    df = pd.DataFrame(res, columns=columns)

    results_dir_name = os.path.dirname(results_file)
    os.makedirs(results_dir_name, exist_ok=True)

    writer = pd.ExcelWriter(results_file, engine='xlsxwriter')

    write_to_excel(df, writer, columns, 'name')
    writer.save()


if __name__ == '__main__':
    # data_path = '/cs/casmip/public/for_shalom/Tumor_segmentation/final_test_set_-_R2U_NET_standalone_as_pairwise_33_scans_-_GT_livers'
    # gt_tumors = glob(f'{data_path}/*/Scan_Tumors.nii.gz')
    # pred_tumors, livers = [], []
    # for gt_file in gt_tumors:
    #     pred_tumors.append(replace_in_file_name(gt_file, '/Scan_Tumors.nii.gz', '/Scan_Tumors_pred_label.nii.gz'))
    #     livers.append(replace_in_file_name(gt_file, '/Scan_Tumors.nii.gz', '/Scan_Liver.nii.gz'))
    # write_stats_per_threshold(livers, gt_tumors, pred_tumors)
    #
    # exit(0)

    results_file = 'measures_results/stats_per_th.xlsx'
    df = pd.read_excel(results_file).rename(columns={'Unnamed: 0': 'name'})
    df = df[~df['name'].isin(['mean', 'std', 'min', 'max', 'sum'])]

    optimal_df = df.sort_values(['dice', 'th'], ascending=False).drop_duplicates(['name']).sort_values('name')

    optimal_df.rename(columns={'name': 'tumor name', 'th': 'optimal th', 'dice': 'optimal dice'}, inplace=True)

    default_df = df[df['th'] == 8].sort_values('name')

    optimal_df['default dice (for th=8)'] = np.array(default_df['dice'])
    optimal_df['delta between optimal and default dice'] = optimal_df['optimal dice'] - optimal_df['default dice (for th=8)']
    optimal_df.sort_values(['delta between optimal and default dice', 'tumor name'], inplace=True, ascending=[False, True])
    optimal_df.reset_index(level=0, drop=True, inplace=True)

    columns = ['tumor name', 'diameter', 'optimal dice', 'default dice (for th=8)',
               'delta between optimal and default dice', 'optimal th']
    results_file = 'measures_results/optimal_dice_per_th.xlsx'
    results_dir_name = os.path.dirname(results_file)
    os.makedirs(results_dir_name, exist_ok=True)

    writer = pd.ExcelWriter(results_file, engine='xlsxwriter')

    write_to_excel(optimal_df, writer, columns, 'tumor name', sheet_name='all diameters')
    write_to_excel(optimal_df[optimal_df['diameter'] <= 5], writer, columns, 'tumor name', sheet_name='diameter <= 5')
    write_to_excel(optimal_df[(5 < optimal_df['diameter']) & (optimal_df['diameter'] <= 10)], writer, columns, 'tumor name', sheet_name='5 < diameter <= 10')
    write_to_excel(optimal_df[(10 < optimal_df['diameter']) & (optimal_df['diameter'] <= 15)], writer, columns, 'tumor name', sheet_name='10 < diameter <= 15')
    write_to_excel(optimal_df[(15 < optimal_df['diameter']) & (optimal_df['diameter'] <= 20)], writer, columns, 'tumor name', sheet_name='15 < diameter <= 20')
    write_to_excel(optimal_df[(20 < optimal_df['diameter']) & (optimal_df['diameter'] <= 30)], writer, columns, 'tumor name', sheet_name='20 < diameter <= 30')
    write_to_excel(optimal_df[30 < optimal_df['diameter']], writer, columns, 'tumor name', sheet_name='30 < diameter')
    writer.save()
