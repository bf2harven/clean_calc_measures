from datetime import date
from skimage.measure import centroid
import json
from utils import *
from matching_graphs import save_matching_graph, draw_matching_graph


def get_original_path(path):
    path = os.path.realpath(path)
    prefix = '/cs/usr/bennydv/Desktop/bennydv/'
    if path.startswith(prefix):
        path = '/cs/casmip/bennydv/' + path[len(prefix):]
    return path


def reorder_labeled_lesions(labeled_lesions: np.ndarray) -> Tuple[np.ndarray, Dict[int, int], Dict[int, int], int, bool]:

    old_labels = np.unique(labeled_lesions)
    old_labels = old_labels[old_labels != 0]

    n_labels = old_labels.size

    new_labels = np.arange(1, n_labels + 1)

    has_changed = False
    if np.any(old_labels != new_labels):
        # reordering the labels
        labeled_lesions = np.select((labeled_lesions.T.reshape([*labeled_lesions.T.shape, 1]) == old_labels.reshape([1, -1])).T, new_labels, labeled_lesions)
        has_changed = True

    to_old_labels = dict(zip(new_labels, old_labels))
    to_new_labels = dict(zip(old_labels, new_labels))

    return labeled_lesions, to_old_labels, to_new_labels, n_labels, has_changed


if __name__ == '__main__':

    # -------------------------------------------- create files -------------------------------------------- #

    results_dir = '/cs/casmip/rochman/Errors_Characterization/lung_test_set_for_matching'
    #
    # original_files = sorted(glob('/cs/casmip/bennydv/lungs_pipeline/gt_data/size_filtered/labeled_no_reg/*'))
    # registered_dir = '/cs/casmip/bennydv/lungs_pipeline/gt_data/size_filtered/labeled_pairwise'
    # longitudinal_matching_GTs_dir = '/cs/casmip/bennydv/lungs_pipeline/lesions_matching/longitudinal_gt/original'
    # n = 1
    # for j, patient_original_files in enumerate(original_files):
    #
    #     patient_name = os.path.basename(patient_original_files)
    #
    #     patient_scans = sorted(glob(f'{patient_original_files}/scan_*'), key=lambda scan_path: date(*[int(c) for c in os.path.basename(scan_path)[5:-7].split('_')[::-1]]))
    #     patient_lesions = [s.replace('/scan_', '/lesions_gt_') for s in patient_scans]
    #
    #     for i in range(len(patient_scans) - 1):
    #         t = time()
    #         case_name = f'BL_{patient_name}{os.path.basename(patient_scans[i])[5:-7]}_FU_{patient_name}{os.path.basename(patient_scans[i + 1])[5:-7]}'
    #         print(f'{n}: {case_name} (finished in ', end='')
    #         n += 1
    #
    #         fu_scan_path = patient_scans[i + 1]
    #         bl_scan_registered_path = f'{registered_dir}/{patient_name}/{os.path.basename(patient_scans[i]).replace(".nii.gz", f"_{os.path.basename(fu_scan_path)[5:]}")}'
    #
    #         fu_lesions_labeled_path = get_original_path(f'{patient_original_files}/{os.path.basename(fu_scan_path).replace("scan_", "lesions_gt_")}')
    #         bl_lesions_labeled_registered_path = get_original_path(f'{registered_dir}/{patient_name}/{os.path.basename(bl_scan_registered_path).replace("scan_", "lesions_gt_")}')
    #
    #         fu_scan_path = get_original_path(fu_scan_path)
    #         bl_scan_registered_path = get_original_path(bl_scan_registered_path)
    #
    #         # loading bl lesions
    #         bl_lesions_labeled_registered, bl_nifti_file = load_nifti_data(bl_lesions_labeled_registered_path)
    #
    #         bl_lesions_labeled_registered, bl_to_old_labels, bl_to_new_labels, n_bl_lesions, bl_has_changed = reorder_labeled_lesions(bl_lesions_labeled_registered)
    #
    #         bl_weights = []
    #         for bl_t in range(1, n_bl_lesions + 1):
    #             bl_weights.append(int(centroid(bl_lesions_labeled_registered == bl_t)[-1] + 1))
    #
    #         # loading fu lesions
    #         fu_lesions_labeled, fu_nifti_file = load_nifti_data(fu_lesions_labeled_path)
    #
    #         fu_lesions_labeled, fu_to_old_labels, fu_to_new_labels, n_fu_lesions, fu_has_changed = reorder_labeled_lesions(fu_lesions_labeled)
    #
    #         fu_weights = []
    #         for fu_t in range(1, n_fu_lesions + 1):
    #             fu_weights.append(int(centroid(fu_lesions_labeled == fu_t)[-1] + 1))
    #
    #         # create GT matching json files
    #         old_GT_json_file = f'{longitudinal_matching_GTs_dir}/{patient_name}long_gt.json'
    #         with open(old_GT_json_file) as json_file:
    #             json_string = json_file.read()
    #         gt_edges = json.loads(json_string)[i + 1]['edges']
    #
    #         if bl_has_changed or fu_has_changed:
    #             gt_edges = [[int(bl_to_new_labels[e[0]]), int(fu_to_new_labels[e[1]])] for e in gt_edges]
    #
    #         # saving files
    #         current_results_dir = f'{results_dir}/{case_name}'
    #         os.makedirs(current_results_dir, exist_ok=True)
    #
    #         new_bl_scan_registered_path = f'{current_results_dir}/BL_Scan_CT.nii.gz'
    #         new_fu_scan_path = f'{current_results_dir}/FU_Scan_CT.nii.gz'
    #         new_bl_lesions_labeled_registered_path = f'{current_results_dir}/BL_Scan_Tumors_unique_{int(n_bl_lesions)}_CC.nii.gz'
    #         new_fu_lesions_labeled_path = f'{current_results_dir}/FU_Scan_Tumors_unique_{int(n_fu_lesions)}_CC.nii.gz'
    #         new_gt_matching_graph_json_path = f'{current_results_dir}/gt_matching_graph.json'
    #         new_gt_matching_graph_jpg_path = f'{current_results_dir}/gt_matching_graph.jpg'
    #
    #         # save bl files
    #         os.symlink(bl_scan_registered_path, new_bl_scan_registered_path)
    #         if not bl_has_changed:
    #             os.symlink(bl_lesions_labeled_registered_path, new_bl_lesions_labeled_registered_path)
    #         else:
    #             save(Nifti1Image(bl_lesions_labeled_registered, bl_nifti_file.affine), new_bl_lesions_labeled_registered_path)
    #
    #         # save fu files
    #         os.symlink(fu_scan_path, new_fu_scan_path)
    #         if not fu_has_changed:
    #             os.symlink(fu_lesions_labeled_path, new_fu_lesions_labeled_path)
    #         else:
    #             save(Nifti1Image(fu_lesions_labeled, fu_nifti_file.affine), new_fu_lesions_labeled_path)
    #
    #         # save gt files
    #         save_matching_graph(n_bl_lesions, n_fu_lesions, gt_edges, case_name,
    #                             new_gt_matching_graph_json_path, bl_weights, fu_weights)
    #         if (n_bl_lesions > 0) and (n_fu_lesions > 0):
    #             draw_matching_graph(n_bl_lesions, n_fu_lesions, gt_edges, f'{case_name}_GT',
    #                                 bl_weights=bl_weights, fu_weights=fu_weights, saving_file_name=new_gt_matching_graph_jpg_path)
    #
    #         print(f'{calculate_runtime(t)} )')
    #
    # exit(0)

    # -------------------------------------------- filter files -------------------------------------------- #

    # get_patient_name = lambda case_name: '_'.join([c for c in case_name.replace('BL_', '').split('_FU_')[0].split('_') if not c.isdigit()])
    # get_bl_name = lambda case_name: case_name.replace('BL_', '').split('_FU_')[0]
    # get_fu_name = lambda case_name: case_name.replace('BL_', '').split('_FU_')[1]
    # get_date_from_name_y_m_d = lambda name: [int(f) for f in name.split('_')[-3:][::-1]]
    # get_time_interval = lambda pair: abs((date(*get_date_from_name_y_m_d(get_fu_name(pair))) - date(*get_date_from_name_y_m_d(get_bl_name(pair)))).days)
    #
    # all_pair_names_paths = sorted(glob(f'{results_dir}/*'))
    #
    # all_pair_names = [os.path.basename(p) for p in all_pair_names_paths]
    # df = pd.DataFrame(
    #     data=list(zip(all_pair_names_paths,
    #                   (get_patient_name(p) for p in all_pair_names),
    #                   (get_time_interval(p) for p in all_pair_names),
    #                   (int(''.join([c for c in os.path.basename(glob(f'{p}/BL_Scan_Tumors_unique_*')[0]) if c.isdigit()])) for p in all_pair_names_paths),
    #                   (int(''.join([c for c in os.path.basename(glob(f'{p}/FU_Scan_Tumors_unique_*')[0]) if c.isdigit()])) for p in all_pair_names_paths))),
    #     columns=['case_name', 'patient', 'time_interval', 'n_BL_tumors', 'n_FU_tumors'])
    #
    # min_num_of_lesions = 1
    # max_num_of_lesions = 40
    # max_time_interval = 730
    # max_pairs_per_patient = 20
    #
    # n_pairs = df.shape[0]
    # print(f'Original number of pairs is: {n_pairs}')
    #
    # # filter by minimum number of lesion per scan
    # df = df[(df['n_BL_tumors'] >= min_num_of_lesions) & (df['n_FU_tumors'] >= min_num_of_lesions)]
    # current_n_pairs = df.shape[0]
    # print(f'Filtering by minimum number of lesions ({min_num_of_lesions}) drops {n_pairs - current_n_pairs} pairs. Current number of pairs is: {current_n_pairs}')
    # n_pairs = current_n_pairs
    #
    # # filter by maximum number of lesion per scan
    # df = df[(df['n_BL_tumors'] <= max_num_of_lesions) & (df['n_FU_tumors'] <= max_num_of_lesions)]
    # current_n_pairs = df.shape[0]
    # print(f'Filtering by maximum number of lesions ({max_num_of_lesions}) drops {n_pairs - current_n_pairs} pairs. Current number of pairs is: {current_n_pairs}')
    # n_pairs = current_n_pairs
    #
    # # filter by maximum time interval
    # df = df[df['time_interval'] <= max_time_interval]
    # current_n_pairs = df.shape[0]
    # print(f'Filtering by maximum time interval ({max_time_interval}) drops {n_pairs - current_n_pairs} pairs. Current number of pairs is: {current_n_pairs}')
    # n_pairs = current_n_pairs
    #
    # # filter by maximum pairs per patient
    # relevant_pairs = []
    # for patient, patient_df in df.groupby('patient'):
    #     if patient_df.shape[0] > max_pairs_per_patient:
    #         patient_df = patient_df.sort_values('time_interval', ignore_index=True)
    #         patient_df = patient_df.iloc[:max_pairs_per_patient, :]
    #     relevant_pairs += patient_df['case_name'].to_list()
    # df = df[df['case_name'].isin(relevant_pairs)]
    # current_n_pairs = df.shape[0]
    # print(f'Filtering by maximum pairs per patient ({max_pairs_per_patient}) drops {n_pairs - current_n_pairs} pairs. Current number of pairs is: {current_n_pairs}')
    # n_pairs = current_n_pairs
    #
    # filtered_pair_names_paths = df['case_name'].to_list()
    #
    # pairs_to_delete = list(set(all_pair_names_paths) - set(filtered_pair_names_paths))
    #
    # import shutil
    # for p in pairs_to_delete:
    #     shutil.rmtree(p)

    # -------------------------------------------- copy lung files -------------------------------------------- #

    src_dir = '/cs/casmip/bennydv/lungs_pipeline/registration/preproc_results'
    import re
    for p in glob(f'{results_dir}/BL_*'):
        p = os.path.basename(p)
        patient = p[:re.search(r"\d", p).start()].replace('BL_', '')
        bl_date = p.split('_FU_')[0][len(patient) + 3:]
        fu_date = p.split('_FU_')[1][len(patient):]
        current_src_dir = f'{src_dir}/{patient}{bl_date}{patient}{fu_date}'
        if not os.path.isdir(current_src_dir):
            current_src_dir += '-nifti'
        if not os.path.isdir(current_src_dir):
            current_src_dir = f'{src_dir}/{patient}{bl_date}-nifti{patient}{fu_date}-nifti'
        if not os.path.isdir(current_src_dir):
            fu_d, fu_m, fu_y = fu_date.split('_')
            current_src_dir = f'{src_dir}/{patient}{bl_date}-nifti{patient}{fu_d}_{fu_m}-{fu_y}-nifti'
        if not os.path.isdir(current_src_dir):
            bl_d, bl_m, bl_y = bl_date.split('_')
            current_src_dir = f'{src_dir}/{patient}{bl_d}_{bl_m}-{bl_y}-nifti{patient}{fu_date}-nifti'
        assert os.path.isdir(current_src_dir), f'{current_src_dir} does not exists'
        bl_src_file = get_original_path(f'{current_src_dir}/BL_lung.nii.gz')
        fu_src_file = get_original_path(f'{current_src_dir}/FU_lung.nii.gz')
        assert os.path.isfile(bl_src_file), f'{bl_src_file} does not exists'
        assert os.path.isfile(fu_src_file), f'{fu_src_file} does not exists'
        bl_dst_file = f'{results_dir}/{p}/BL_Scan_Lung.nii.gz'
        fu_dst_file = f'{results_dir}/{p}/FU_Scan_Lung.nii.gz'
        if not os.path.isfile(bl_dst_file):
            os.symlink(bl_src_file, bl_dst_file)
        if not os.path.isfile(fu_dst_file):
            os.symlink(fu_src_file, fu_dst_file)