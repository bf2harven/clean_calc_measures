from utils import *
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from matching_graphs import *
from skimage.measure import centroid
from functools import partial
from notifications import notify
from ICL_matching import prepare_dataset, execute_fast_global_registration, execute_ICP
from scipy.ndimage import affine_transform
from multiprocessing import Pool
from datetime import date


def calculate_statistics_over_matching(gt_matches: List[Tuple[int, int]], pred_matches: List[Tuple[int, int]],
                                     n_bl_tumors: int, n_fu_tumors: int):
    gt_matches = set(gt_matches)
    pred_matches = set(pred_matches)

    # calculating the edges measures
    tp_edges = len(gt_matches & pred_matches)
    fp_edges = len(pred_matches - gt_matches)
    fn_edges = len(gt_matches - pred_matches)

    if tp_edges + fp_edges != 0:
        precision_edges = tp_edges / (tp_edges + fp_edges)
    else:
        precision_edges = 1

    if tp_edges + fn_edges != 0:
        recall_edges = tp_edges / (tp_edges + fn_edges)
    else:
        recall_edges = 1

    if precision_edges + recall_edges != 0:
        f1_score_edges = 2 * precision_edges * recall_edges / (precision_edges + recall_edges)
    else:
        f1_score_edges = 0

    # calculating the isolated-tumors measures
    bl_tumors = set(range(1, n_bl_tumors + 1))
    fu_tumors = set(range(1, n_fu_tumors + 1))

    if len(gt_matches) > 0:
        gt_bl_not_IS_tumors, gt_fu_not_IS_tumors = zip(*gt_matches)
    else:
        gt_bl_not_IS_tumors, gt_fu_not_IS_tumors = tuple(), tuple()
    gt_bl_not_IS_tumors = set(gt_bl_not_IS_tumors)
    gt_bl_IS_tumors = bl_tumors - gt_bl_not_IS_tumors
    gt_fu_not_IS_tumors = set(gt_fu_not_IS_tumors)
    gt_fu_IS_tumors = fu_tumors - gt_fu_not_IS_tumors

    if len(pred_matches) > 0:
        pred_bl_not_IS_tumors, pred_fu_not_IS_tumors = zip(*pred_matches)
    else:
        pred_bl_not_IS_tumors, pred_fu_not_IS_tumors = tuple(), tuple()
    pred_bl_not_IS_tumors = set(pred_bl_not_IS_tumors)
    pred_bl_IS_tumors = bl_tumors - pred_bl_not_IS_tumors
    pred_fu_not_IS_tumors = set(pred_fu_not_IS_tumors)
    pred_fu_IS_tumors = fu_tumors - pred_fu_not_IS_tumors

    tp_IS_tumors = len(gt_bl_IS_tumors & pred_bl_IS_tumors) + len(gt_fu_IS_tumors & pred_fu_IS_tumors)
    tn_IS_tumors = len(gt_bl_not_IS_tumors & pred_bl_not_IS_tumors) + len(gt_fu_not_IS_tumors & pred_fu_not_IS_tumors)
    fp_IS_tumors = len(pred_bl_IS_tumors - gt_bl_IS_tumors) + len(pred_fu_IS_tumors - gt_fu_IS_tumors)
    fn_IS_tumors = len(gt_bl_IS_tumors - pred_bl_IS_tumors) + len(gt_fu_IS_tumors - pred_fu_IS_tumors)

    if tp_IS_tumors + fp_IS_tumors != 0:
        precision_IS_tumors = tp_IS_tumors / (tp_IS_tumors + fp_IS_tumors)
    else:
        precision_IS_tumors = 1

    if tp_IS_tumors + fn_IS_tumors != 0:
        recall_IS_tumors = tp_IS_tumors / (tp_IS_tumors + fn_IS_tumors)
    else:
        recall_IS_tumors = 1

    if precision_IS_tumors + recall_IS_tumors != 0:
        f1_score_IS_tumors = 2 * precision_IS_tumors * recall_IS_tumors / (precision_IS_tumors + recall_IS_tumors)
    else:
        f1_score_IS_tumors = 0

    if tn_IS_tumors + fp_IS_tumors != 0:
        specificity_IS_tumors = tn_IS_tumors / (tn_IS_tumors + fp_IS_tumors)
    else:
        specificity_IS_tumors = 1

    if tn_IS_tumors + fn_IS_tumors != 0:
        NPV_IS_tumors = tn_IS_tumors / (tn_IS_tumors + fn_IS_tumors)
    else:
        NPV_IS_tumors = 1

    if tp_IS_tumors + tn_IS_tumors + fn_IS_tumors + fp_IS_tumors != 0:
        accuracy_IS_tumors = (tp_IS_tumors + tn_IS_tumors) / (tp_IS_tumors + tn_IS_tumors + fn_IS_tumors + fp_IS_tumors)
        positive_rate_IS_tumors = (tp_IS_tumors + fn_IS_tumors) / (tp_IS_tumors + tn_IS_tumors + fn_IS_tumors + fp_IS_tumors)
        negative_rate_IS_tumors = (tn_IS_tumors + fp_IS_tumors) / (tp_IS_tumors + tn_IS_tumors + fn_IS_tumors + fp_IS_tumors)
    else:
        accuracy_IS_tumors = 1
        positive_rate_IS_tumors = np.nan
        negative_rate_IS_tumors = np.nan

    return (precision_edges, recall_edges, f1_score_edges, tp_edges, fp_edges, fn_edges,
            precision_IS_tumors, recall_IS_tumors, f1_score_IS_tumors, specificity_IS_tumors, NPV_IS_tumors,
            accuracy_IS_tumors, tp_IS_tumors, fp_IS_tumors, fn_IS_tumors, tn_IS_tumors, positive_rate_IS_tumors,
            negative_rate_IS_tumors)


def calculate_measures_over_matching(bl_tumors_CC_labeled: np.ndarray, fu_tumors_CC_labeled: np.ndarray,
                                     gt_matches: List[Tuple[int, int]], pred_matches: List[Tuple[int, int]],
                                     pix_dims: Tuple[float, float, float]):
    gt_matches = set(gt_matches)
    pred_matches = set(pred_matches)

    not_isolated_bl_tumors_gt = set(f'{int(m[0])}_bl' for m in gt_matches)
    isolated_bl_tumors_gt = set(f'{t}_bl' for t in range(1, bl_tumors_CC_labeled.max() + 1)) - not_isolated_bl_tumors_gt
    not_isolated_fu_tumors_gt = set(f'{int(m[1])}_fu' for m in gt_matches)
    isolated_fu_tumors_gt = set(f'{t}_fu' for t in range(1, fu_tumors_CC_labeled.max() + 1)) - not_isolated_fu_tumors_gt

    not_isolated_bl_tumors_pred = set(f'{int(m[0])}_bl' for m in pred_matches)
    isolated_bl_tumors_pred = set(f'{t}_bl' for t in range(1, bl_tumors_CC_labeled.max() + 1)) - not_isolated_bl_tumors_pred
    not_isolated_fu_tumors_pred = set(f'{int(m[1])}_fu' for m in pred_matches)
    isolated_fu_tumors_pred = set(f'{t}_fu' for t in range(1, fu_tumors_CC_labeled.max() + 1)) - not_isolated_fu_tumors_pred

    tp_isolated_tumors = (isolated_bl_tumors_gt | isolated_fu_tumors_gt) & (isolated_bl_tumors_pred | isolated_fu_tumors_pred)
    fp_isolated_tumors = (isolated_bl_tumors_pred | isolated_fu_tumors_pred) - tp_isolated_tumors
    fn_isolated_tumors = (isolated_bl_tumors_gt | isolated_fu_tumors_gt) - tp_isolated_tumors

    relevant_bl_tumors = set(int(m[0]) for m in gt_matches) | set(int(m[0]) for m in pred_matches)
    not_relevant_bl_tumors = set(range(1, bl_tumors_CC_labeled.max() + 1)) - relevant_bl_tumors
    relevant_bl_tumors = list(relevant_bl_tumors)

    relevant_fu_tumors = set(int(m[1]) for m in gt_matches) | set(int(m[1]) for m in pred_matches)
    not_relevant_fu_tumors = set(range(1, fu_tumors_CC_labeled.max() + 1)) - relevant_fu_tumors
    relevant_fu_tumors = list(relevant_fu_tumors)

    # bl_tumors_CC_labeled[~np.isin(bl_tumors_CC_labeled, relevant_bl_tumors)] = 0
    # fu_tumors_CC_labeled[~np.isin(fu_tumors_CC_labeled, relevant_fu_tumors)] = 0

    all_matches = gt_matches | pred_matches
    tp_matches = gt_matches & pred_matches
    fp_matches = pred_matches - tp_matches

    voxel_volume = pix_dims[0] * pix_dims[1] * pix_dims[2]

    def get_statistic_match_stamp(m):
        assert isinstance(m, tuple)
        if m in tp_matches:
            return 1, 0, 0
        elif match in fp_matches:
            return 0, 1, 0
        else:
            return 0, 0, 1

    def get_statistic_isolated_stamp(t):
        assert isinstance(t, str)
        if t in tp_isolated_tumors:
            return 1, 0, 0, 0
        elif t in fp_isolated_tumors:
            return 0, 1, 0, 0
        elif t in fn_isolated_tumors:
            return 0, 0, 1, 0
        return 0, 0, 0, 1

    tumor_diameters = dict((f'{int(v)}_bl', (approximate_diameter((bl_tumors_CC_labeled == v).sum() * voxel_volume), *get_statistic_isolated_stamp(f'{int(v)}_bl')))
                           for v in not_relevant_bl_tumors)
    tumor_diameters.update((f'{int(v)}_fu', (approximate_diameter((fu_tumors_CC_labeled == v).sum() * voxel_volume), *get_statistic_isolated_stamp(f'{int(v)}_fu')))
                           for v in not_relevant_fu_tumors)

    res = []

    for match in all_matches:
        bl_tumor = (bl_tumors_CC_labeled == match[0]).astype(bl_tumors_CC_labeled.dtype)
        fu_tumor = (fu_tumors_CC_labeled == match[1]).astype(fu_tumors_CC_labeled.dtype)

        overlap = (bl_tumor * fu_tumor).sum()
        bl_sum = bl_tumor.sum()
        fu_sum = fu_tumor.sum()

        bl_tumor_found = bl_sum > 0
        fu_tumor_found = fu_sum > 0

        bl_tumor_name = f'{int(match[0])}_bl'
        if bl_tumor_name in tumor_diameters:
            bl_tumor_diameter, _, _, _, _ = tumor_diameters[bl_tumor_name]
        else:
            if bl_tumor_found:
                bl_tumor_diameter = approximate_diameter(bl_sum * voxel_volume)
            else:
                bl_tumor_diameter = np.nan
            tumor_diameters[bl_tumor_name] = (bl_tumor_diameter, *get_statistic_isolated_stamp(bl_tumor_name))

        fu_tumor_name = f'{int(match[1])}_fu'
        if fu_tumor_name in tumor_diameters:
            fu_tumor_diameter, _, _, _, _ = tumor_diameters[fu_tumor_name]
        else:
            if fu_tumor_found:
                fu_tumor_diameter = approximate_diameter(fu_sum * voxel_volume)
            else:
                fu_tumor_diameter = np.nan
            tumor_diameters[fu_tumor_name] = (fu_tumor_diameter, *get_statistic_isolated_stamp(fu_tumor_name))

        if fu_tumor_found and bl_tumor_found:
            overlap_with_bl = overlap/bl_sum
            overlap_with_fu = overlap/fu_sum
            IOU = overlap / (np.logical_or(bl_tumor.astype(np.bool), fu_tumor.astype(np.bool)).astype(np.int).sum())
            overlap *= voxel_volume/1000

            assd, hd, min_dist = assd_hd_and_min_distance(bl_tumor, fu_tumor, voxelspacing=pix_dims)
            dcs = dice(bl_tumor, fu_tumor)
            volume_diff = np.abs(bl_sum-fu_sum)/(bl_sum + fu_sum)

            bl_tumor_center = np.zeros_like(bl_tumors_CC_labeled)
            cent = np.round(centroid(bl_tumor)).astype(np.int)
            bl_tumor_center[cent[0], cent[1], cent[2]] = 1

            fu_tumor_center = np.zeros_like(fu_tumors_CC_labeled)
            cent = np.round(centroid(fu_tumor)).astype(np.int)
            fu_tumor_center[cent[0], cent[1], cent[2]] = 1

            # min_dist = min_distance(bl_tumor, fu_tumor, voxelspacing=pix_dims)
            center_min_dist = min_distance(bl_tumor_center, fu_tumor_center, voxelspacing=pix_dims)
        else:
            min_dist, center_min_dist, overlap, overlap_with_bl, overlap_with_fu, hd, assd, dcs, IOU, volume_diff = [np.nan] * 10

        res.append((match, bl_tumor_diameter, fu_tumor_diameter, min_dist, center_min_dist, overlap, overlap_with_bl,
                    overlap_with_fu, hd, assd, dcs, IOU, volume_diff, *get_statistic_match_stamp(match)))

    return res, tumor_diameters


def get_measures_and_statistics_of_matching(fu_tumors_file: str, read_saved_pred_matches: bool = False,
                                            save_pred_matches: bool = True, max_dilate_param: int = 5,
                                            match_func: Optional[Callable] = None,
                                            adaptive_num_of_dilations=False, liver_study=True,
                                            without_improving_registration=False):

    try:

        if not liver_study:
            assert not adaptive_num_of_dilations
            assert not without_improving_registration
            if match_func is not None:
                assert match_func != match_after_improving_registration

        if without_improving_registration and match_func is not None:
            assert match_func != match_after_improving_registration

        dir_name = os.path.dirname(fu_tumors_file)
        gt_matching_graph_file = f'{dir_name}/gt_matching_graph.json'
        assert os.path.isfile(gt_matching_graph_file)
        if without_improving_registration:
            pred_matching_graph_file = f'{dir_name}/pred_without_improving_registration_matching_graph.json'
        else:
            pred_matching_graph_file = f'{dir_name}/pred_matching_graph.json'

        pred_matches_was_loaded = False
        (n_bl_tumors, n_fu_tumors, gt_matches, case_name, bl_weights, fu_weights,
        bl_diameters, fu_diameters, bl_organ_volume, fu_organ_volume) = load_matching_graph(gt_matching_graph_file)

        if read_saved_pred_matches and os.path.isfile(pred_matching_graph_file):
            n_bl_tumors, n_fu_tumors, pred_matches, _, _, _, _, _, _, _ = load_matching_graph(pred_matching_graph_file)
            pred_matches_was_loaded = True

        if liver_study:
            if without_improving_registration:
                bl_tumors_file = glob(f'{dir_name}/BL_Scan_Tumors_unique_*')
            else:
                # bl_tumors_file = glob(f'{dir_name}/improved_registration_BL_Scan_Tumors_unique_*')
                bl_tumors_file = glob(f'{dir_name}/improved_registration_Only_ICP_BL_Scan_Tumors_unique_*')
        else:
            bl_tumors_file = glob(f'{dir_name}/BL_Scan_Tumors_unique_*')
        assert len(bl_tumors_file) == 1
        bl_tumors_file = bl_tumors_file[0]

        bl_tumors_CC_labeled, _ = load_nifti_data(bl_tumors_file)
        # bl_tumors_case, _ = load_nifti_data(bl_tumors_file)
        # bl_tumors_case = (bl_tumors_case > 0).astype(bl_tumors_case.dtype)

        # bl_tumors_CC_labeled = get_connected_components(bl_tumors_case)
        bl_tumors_CC_labeled = bl_tumors_CC_labeled.astype(np.int16)

        fu_tumors_CC_labeled, file = load_nifti_data(fu_tumors_file)
        # fu_tumors_case, file = load_nifti_data(fu_tumors_file)
        # fu_tumors_case = (fu_tumors_case > 0).astype(fu_tumors_case.dtype)

        # fu_tumors_CC_labeled = get_connected_components(fu_tumors_case)
        fu_tumors_CC_labeled = fu_tumors_CC_labeled.astype(np.int16)

        if match_func is None:
            match_func = match_2_cases_v3

        match_time = time()
        if not pred_matches_was_loaded:
            if match_func == match_after_improving_registration:
                assert '/BL_Scan_Tumors_unique_' in bl_tumors_file
                bl_organ = getLargestCC(load_nifti_data(f'{dir_name}/BL_Scan_Liver.nii.gz')[0] > 0).astype(np.float32)
                fu_organ = getLargestCC(load_nifti_data(f'{dir_name}/FU_Scan_Liver.nii.gz')[0] > 0).astype(np.float32)
                pred_matches = match_func(bl_tumors_CC_labeled, fu_tumors_CC_labeled, bl_organ, fu_organ, file.affine,
                                          voxelspacing=file.header.get_zooms(), max_dilate_param=max_dilate_param,
                                          return_iteration_indicator=True)
            else:
                if not adaptive_num_of_dilations:
                    pred_matches = match_func(bl_tumors_CC_labeled, fu_tumors_CC_labeled, voxelspacing=file.header.get_zooms(),
                                              max_dilate_param=max_dilate_param, return_iteration_indicator=True)
                else:
                    if without_improving_registration:
                        bl_organ = getLargestCC(
                            load_nifti_data(f'{dir_name}/BL_Scan_Liver.nii.gz')[0] > 0).astype(
                            np.float32)
                    else:
                        if liver_study:
                            bl_organ = getLargestCC(load_nifti_data(f'{dir_name}/improved_registration_Only_ICP_BL_Scan_Liver.nii.gz')[0] > 0).astype(
                                np.float32)
                        else:
                            bl_organ = getLargestCC(
                                load_nifti_data(f'{dir_name}/BL_Scan_Lung.nii.gz')[
                                    0] > 0).astype(
                                np.float32)
                    if liver_study:
                        fu_organ = getLargestCC(load_nifti_data(f'{dir_name}/FU_Scan_Liver.nii.gz')[0] > 0).astype(
                            np.float32)
                    else:
                        fu_organ = getLargestCC(load_nifti_data(f'{dir_name}/FU_Scan_Lung.nii.gz')[0] > 0).astype(
                            np.float32)
                    organ_dice = dice(bl_organ, fu_organ)
                    adaptive_max_dilate_param = int(np.max([1, np.round(max_dilate_param * (1 - organ_dice) / 0.2)]))
                    pred_matches = match_func(bl_tumors_CC_labeled, fu_tumors_CC_labeled,
                                              voxelspacing=file.header.get_zooms(),
                                              max_dilate_param=adaptive_max_dilate_param, return_iteration_indicator=True)
            match_time = time() - match_time
        else:
            match_time = time() - match_time

        # pred_matches = [(m[0], m[1], (int(m[2][0]), int(m[2][1]))) for m in pred_matches]
        pred_matches = [(m[0], (int(m[1][0]), int(m[1][1]))) for m in pred_matches]

        gt_data_was_updated = False
        if bl_weights is None:
            bl_weights = []
            for bl_t in range(1, n_bl_tumors + 1):
                bl_weights.append(int(centroid(bl_tumors_CC_labeled == bl_t)[-1] + 1))
            gt_data_was_updated = True

        if fu_weights is None:
            fu_weights = []
            for fu_t in range(1, n_fu_tumors + 1):
                fu_weights.append(int(centroid(fu_tumors_CC_labeled == fu_t)[-1] + 1))
            gt_data_was_updated = True

        vol = np.asarray(file.header.get_zooms()).prod() / 1000

        if bl_diameters is None:
            if liver_study and not without_improving_registration:
                bl_tumors_CC_labeled_for_diameter_checking, _ = load_nifti_data(glob(f'{dir_name}/BL_Scan_Tumors_unique_*')[0])
            else:
                bl_tumors_CC_labeled_for_diameter_checking = bl_tumors_CC_labeled
            bl_diameters = []
            for bl_t in range(1, n_bl_tumors + 1):
                bl_diameters.append(approximate_diameter((bl_tumors_CC_labeled_for_diameter_checking == bl_t).sum()*vol*1000))
            gt_data_was_updated = True

        if fu_diameters is None:
            fu_diameters = []
            for fu_t in range(1, n_fu_tumors + 1):
                fu_diameters.append(approximate_diameter((fu_tumors_CC_labeled == fu_t).sum()*vol*1000))
            gt_data_was_updated = True

        if liver_study:
            if without_improving_registration:
                bl_organ = getLargestCC(
                    load_nifti_data(f'{dir_name}/BL_Scan_Liver.nii.gz')[0] > 0)
            else:
                # bl_liver = getLargestCC(load_nifti_data(f'{dir_name}/improved_registration_BL_Scan_Liver.nii.gz')[0] > 0)
                bl_organ = getLargestCC(
                    load_nifti_data(f'{dir_name}/improved_registration_Only_ICP_BL_Scan_Liver.nii.gz')[0] > 0)
            fu_organ = getLargestCC(load_nifti_data(f'{dir_name}/FU_Scan_Liver.nii.gz')[0] > 0)
        else:
            bl_organ = getLargestCC(
                load_nifti_data(f'{dir_name}/BL_Scan_Lung.nii.gz')[0] > 0)
            fu_organ = getLargestCC(load_nifti_data(f'{dir_name}/FU_Scan_Lung.nii.gz')[0] > 0)

        bl_liver_volume = bl_organ.sum() * vol
        fu_liver_volume = fu_organ.sum() * vol

        if bl_organ_volume is None:
            bl_organ_volume = bl_liver_volume
            gt_data_was_updated = True

        if fu_organ_volume is None:
            fu_organ_volume = fu_liver_volume
            gt_data_was_updated = True

        if gt_data_was_updated:
            save_matching_graph(n_bl_tumors, n_fu_tumors, gt_matches, case_name, gt_matching_graph_file, bl_weights,
                                fu_weights, bl_diameters, fu_diameters, bl_organ_volume, fu_organ_volume)

        if save_pred_matches:
            # save_matching_graph(n_bl_tumors, n_fu_tumors, gt_matches, case_name, gt_matching_graph_file, bl_weights, fu_weights)
            save_matching_graph(n_bl_tumors, n_fu_tumors, pred_matches, case_name, pred_matching_graph_file, bl_weights,
                                fu_weights, bl_diameters, fu_diameters, bl_organ_volume, fu_organ_volume)

        # pred_matches = [m[2] for m in pred_matches if m[1] <= max_dilate_param]
        pred_matches = [m[1] for m in pred_matches if m[0] <= max_dilate_param]

        # iterations_indicators, pred_matches = zip(*pred_matches)

        matching_statistics = calculate_statistics_over_matching(gt_matches, pred_matches, n_bl_tumors, n_fu_tumors)

        matching_measures, tumor_diameters = calculate_measures_over_matching(bl_tumors_CC_labeled, fu_tumors_CC_labeled,
                                                             gt_matches, pred_matches, file.header.get_zooms())

        gt_draw_graph_file = f'{dir_name}/gt_matching_graph.jpg'
        if without_improving_registration:
            pred_draw_graph_file = f'{dir_name}/pred_without_improving_registration_matching_graph.jpg'
        else:
            pred_draw_graph_file = f'{dir_name}/pred_matching_graph.jpg'
        draw_matching_graph(n_bl_tumors, n_fu_tumors, gt_matches, f'{case_name}_GT',
                            bl_weights=bl_weights, fu_weights=fu_weights, saving_file_name=gt_draw_graph_file)
        draw_matching_graph(n_bl_tumors, n_fu_tumors, pred_matches, f'{case_name}_PRED',
                            bl_weights=bl_weights, fu_weights=fu_weights, saving_file_name=pred_draw_graph_file)

        matching_measures.sort(key=lambda t: [t[0][0], t[0][1]])
        # matching_measures = [(f'{case_name}_({int(t[0][0])},{int(t[0][1])})', *t[1:]) for t in matching_measures]
        matching_measures = [(case_name, *t) for t in matching_measures]

        tumors_names = list(tumor_diameters.keys())
        tumors_names.sort(key=lambda name: [name[-2:], int(name[:-3])])
        tumor_diameters = [(case_name, t_name, *tumor_diameters[t_name]) for t_name in tumors_names]

        bl_tumor_borden = (bl_tumors_CC_labeled > 0).sum() * vol
        fu_tumor_borden = (fu_tumors_CC_labeled > 0).sum() * vol
        tumors_abs_vol_diff = abs(bl_tumor_borden - fu_tumor_borden)

        if liver_study:
            organ_dice = dice(bl_organ, fu_organ)
            liver_assd, liver_hd = assd_and_hd(bl_organ, fu_organ, file.header.get_zooms())
            liver_abs_volume_diff = abs(bl_liver_volume - fu_liver_volume)
        else:
            organ_dice = liver_assd = liver_hd = liver_abs_volume_diff = np.nan

        n_edges = len(gt_matches)

        # calculate Time Range between scans
        bl, fu = case_name.replace('BL_', '').split('_FU_')
        bl_d, bl_m, bl_y = ([int(f) for f in bl.split('_')[-3:]])
        fu_d, fu_m, fu_y = ([int(f) for f in fu.split('_')[-3:]])
        bl_date = date(bl_y, bl_m, bl_d)
        fu_date = date(fu_y, fu_m, fu_d)
        time_range_between_scans = abs((fu_date - bl_date).days)

        return (case_name, *matching_statistics, n_bl_tumors, n_fu_tumors, n_edges, organ_dice, liver_assd, liver_hd, liver_abs_volume_diff, tumors_abs_vol_diff, time_range_between_scans, match_time), matching_measures, tumor_diameters
    except Exception as e:
        print(e)
        print(f'Given file is: {fu_tumors_file}')
        raise e


def get_measures_of_CCs_of_GT(fu_tumors_file: str):

    gt_matching_graph_file = f'{os.path.dirname(fu_tumors_file)}/gt_matching_graph.json'
    assert os.path.isfile(gt_matching_graph_file)

    _, _, gt_matches, case_name, _, _, _, _, _, _ = load_matching_graph(gt_matching_graph_file)
    gt_matches = [(int(u), int(v)) for (u, v) in gt_matches]

    bl_tumors_file = glob(f'{os.path.dirname(fu_tumors_file)}/BL_Scan_Tumors_unique_*')
    assert len(bl_tumors_file) == 1
    bl_tumors_file = bl_tumors_file[0]

    bl_tumors_case, file = load_nifti_data(bl_tumors_file)
    bl_tumors_case = (bl_tumors_case > 0).astype(bl_tumors_case.dtype)

    bl_tumors_CC_labeled = get_connected_components(bl_tumors_case)

    fu_tumors_case, file = load_nifti_data(fu_tumors_file)
    fu_tumors_case = (fu_tumors_case > 0).astype(fu_tumors_case.dtype)

    fu_tumors_CC_labeled = get_connected_components(fu_tumors_case)

    pix_dims = file.header.get_zooms()
    voxel_volume = pix_dims[0] * pix_dims[1] * pix_dims[2]

    # extracting the BL labels
    BL_labels = np.unique(bl_tumors_CC_labeled)
    BL_labels = BL_labels[BL_labels != 0]
    BL_n_tumors = BL_labels.size

    # extracting the FU labels
    FU_labels = np.unique(fu_tumors_CC_labeled)
    FU_labels = FU_labels[FU_labels != 0] + BL_n_tumors
    FU_n_tumors = FU_labels.size

    V = list(BL_labels - 1) + list(FU_labels - 1)
    visited = [False] * len(V)
    adjacency_lists = []
    for _ in range(BL_n_tumors + FU_n_tumors):
        adjacency_lists.append([])
    for (bl_v, fu_v) in gt_matches:
        fu_v += BL_n_tumors - 1
        bl_v -= 1
        adjacency_lists[bl_v].append(fu_v)
        adjacency_lists[fu_v].append(bl_v)

    def DFS(v, CC=None):
        if CC is None:
            CC = []
        visited[v] = True
        CC.append(v)
        V.remove(v)

        for u in adjacency_lists[v]:
            if not visited[u]:
                CC = DFS(u, CC)
        return CC

    is_bl_tumor = lambda v: v <= BL_n_tumors - 1

    def bl_and_fu(CC):
        bl_in_CC = []
        fu_in_CC = []
        for v in CC:
            if is_bl_tumor(v):
                bl_in_CC.append(v + 1)
            else:
                fu_in_CC.append(v + 1 - BL_n_tumors)
        return bl_in_CC, fu_in_CC

    results = []

    while len(V) > 0:
        v = V[0]
        current_CC = DFS(v)

        # in case the current tumor is a isolated
        if len(current_CC) == 1:
            continue

        bl_in_CC, fu_in_CC = bl_and_fu(current_CC)

        bl_tumor = np.isin(bl_tumors_CC_labeled, bl_in_CC).astype(bl_tumors_CC_labeled.dtype)
        fu_tumor = np.isin(fu_tumors_CC_labeled, fu_in_CC).astype(fu_tumors_CC_labeled.dtype)

        bl_tumor, fu_tumor = crop_to_relevant_joint_bbox(bl_tumor, fu_tumor)

        overlap = (bl_tumor * fu_tumor).sum()
        bl_sum = bl_tumor.sum()
        fu_sum = fu_tumor.sum()
        overlap_with_bl = overlap / bl_sum
        overlap_with_fu = overlap / fu_sum
        IOU = overlap / (np.logical_or(bl_tumor.astype(np.bool), fu_tumor.astype(np.bool)).astype(np.int).sum())
        overlap *= voxel_volume / 1000
        bl_volume = bl_sum * voxel_volume / 1000
        fu_volume = fu_sum * voxel_volume / 1000

        assd, hd, min_dist = assd_hd_and_min_distance(bl_tumor, fu_tumor, voxelspacing=pix_dims,
                                                      crop_to_relevant_scope=False)
        dcs = dice(bl_tumor, fu_tumor)
        volume_diff_percentage = np.abs(bl_sum - fu_sum) / (bl_sum + fu_sum)
        volume_diff_CC = np.abs(bl_volume - fu_volume)

        bl_tumor_center = np.zeros_like(bl_tumor)
        cent = np.round(centroid(bl_tumor)).astype(np.int)
        bl_tumor_center[cent[0], cent[1], cent[2]] = 1

        fu_tumor_center = np.zeros_like(fu_tumor)
        cent = np.round(centroid(fu_tumor)).astype(np.int)
        fu_tumor_center[cent[0], cent[1], cent[2]] = 1

        center_min_dist = min_distance(bl_tumor_center, fu_tumor_center, voxelspacing=pix_dims)
        match = f'({bl_in_CC},{fu_in_CC})'

        results.append((case_name, match, bl_volume, fu_volume, min_dist, center_min_dist, overlap,
                        overlap_with_bl, overlap_with_fu, hd, assd, dcs, IOU, volume_diff_percentage, volume_diff_CC))

    return results


if __name__ == '__main__':

    # def patient_BL_and_FU(pair_name):
    #     pair_name = pair_name.replace('BL_', '')
    #     bl, fu = pair_name.split('_FU_')
    #     patient = '_'.join(c for c in bl.split('_') if not c.isdigit())
    #     return patient, bl, fu
    #
    # def sort_key(name):
    #     split = name.split('_')
    #     return '_'.join(c for c in split if not c.isdigit()), int(split[-1]), int(split[-2]), int(split[-3])
    #
    # def is_in_order(name1, name2):
    #     key1 = sort_key(name1)
    #     key2 = sort_key(name2)
    #     if key1[1] > key2[1]:
    #         return False
    #     if key1[1] == key2[1] and key1[2] > key2[2]:
    #         return False
    #     if key1[1] == key2[1] and key1[2] == key2[2] and key1[3] > key2[3]:
    #         return False
    #     return True
    #
    # from datetime import date
    #
    # def diff_in_days(bl_name, fu_name):
    #     _, bl_y, bl_m, bl_d = sort_key(bl_name)
    #     _, fu_y, fu_m, fu_d = sort_key(fu_name)
    #     bl_date  = date(bl_y, bl_m, bl_d)
    #     fu_date  = date(fu_y, fu_m, fu_d)
    #     return abs((fu_date - bl_date).days)
    #
    # # training_pairs1 = pd.read_excel('/cs/casmip/rochman/Errors_Characterization/data/test_set_pairs_data_measures.xlsx')
    # # training_pairs2 = pd.read_excel('/cs/casmip/rochman/Errors_Characterization/data/tumors_measurements_-_th_8_-_final_test_set_-_R2U_NET_pairwise_101_pairs_-_GT_liver_without_BL_tumors_edited.xlsx')
    # # training_pairs = training_pairs1[training_pairs1['Unnamed: 0'].isin(training_pairs2['Unnamed: 0'])]
    # training_pairs = pd.read_excel('/cs/casmip/rochman/Errors_Characterization/data/all_data_pairs_data_measures.xlsx')
    # max_num_of_tumors = 30
    # min_num_of_tumors = 1
    # training_pairs = training_pairs[(training_pairs['BL_n_tumors'] <= max_num_of_tumors) & (training_pairs['FU_n_tumors'] <= max_num_of_tumors) &
    #                                 (training_pairs['BL_n_tumors'] >= min_num_of_tumors) & (training_pairs['FU_n_tumors'] >= min_num_of_tumors)]
    # training_pairs = training_pairs[training_pairs[training_pairs.columns[0]].str.startswith('BL_')][[training_pairs.columns[0], 'BL_n_tumors', 'FU_n_tumors']]
    # training_pairs = [(*patient_BL_and_FU(pair_name), int(n_bl_t), int(n_fu_t)) for (_, (pair_name, n_bl_t, n_fu_t)) in training_pairs.iterrows()]
    # training_pairs.sort(key=lambda p: (p[0], sort_key(p[1]), sort_key(p[2])))
    # training_pairs = [pair_name for pair_name in training_pairs if is_in_order(pair_name[1], pair_name[2])]
    # training_pairs = pd.DataFrame(training_pairs, columns=['patient', 'bl', 'fu', 'n_bl_t', 'n_fu_t'])
    # training_pairs['Diff in days'] = training_pairs.apply(lambda r: diff_in_days(r['bl'], r['fu']), axis=1)
    # training_pairs = training_pairs[training_pairs['Diff in days'] <= 365]
    # training_pairs['case_name'] = training_pairs.apply(lambda r: f'BL_{r["bl"]}_FU_{r["fu"]}', axis=1)
    # matching_test_set = pd.read_excel('/cs/casmip/rochman/Errors_Characterization/matching/corrected_measures_results/matching_statistics_dilate_13.xlsx')
    # training_pairs = training_pairs[~training_pairs['case_name'].isin(matching_test_set[matching_test_set.columns[0]])]
    #
    # dst_dir = '/cs/casmip/rochman/Errors_Characterization/new_test_set_for_matching'
    # os.makedirs(dst_dir, exist_ok=True)
    # src_dir = '/mnt/sda1/aszeskin/Data_Followup_Full_29_4_2021'
    # for case_name in training_pairs['case_name']:
    #     symlink_for_inner_files_in_a_dir(f'{src_dir}/{case_name}', f'{dst_dir}/{case_name}')
    # exit(0)
    #
    # final_pairs = []
    # max_num_of_pairs = 100
    # for i, (_, patient_details) in enumerate(training_pairs.groupby('patient'), start=1):
    #     if i <= max_num_of_pairs:
    #         for patient_bl, bl_details in patient_details.groupby(['patient', 'bl']):
    #             pair_name = f'BL_{patient_bl[1]}_FU_{bl_details.iloc[0, 2]}'
    #             final_pairs.append((pair_name, bl_details.iloc[0, 3], bl_details.iloc[0, 4]))
    #             break
    #
    # n_saved = 0
    # max_n_saved = 30
    # for j, (pair, n_bl_t, n_fu_t) in enumerate(final_pairs, start=1):
    #     if n_saved >= max_n_saved:
    #         break
    #     old_dir_name = f'/mnt/sda1/aszeskin/Data_Followup_Full_29_4_2021/{pair}'
    #     new_dir_name = f'matching_for_richard/round_2/{pair}'
    #     print(f'{j:02d} - {pair}', end=': ')
    #     if os.path.isdir(new_dir_name.replace('/round_2/', '/round_1/')):
    #         print('')
    #         continue
    #     n_saved += 1
    #     print(f'-- saved ({n_saved})')
    #     os.makedirs(new_dir_name, exist_ok=True)
    #     os.symlink(f'{old_dir_name}/BL_Scan_CT.nii.gz', f'{new_dir_name}/BL_Scan_CT.nii.gz')
    #     os.symlink(f'{old_dir_name}/FU_Scan_CT.nii.gz', f'{new_dir_name}/FU_Scan_CT.nii.gz')
    #     # os.symlink(f'{old_dir_name}/BL_Scan_Liver.nii.gz', f'{new_dir_name}/BL_Scan_Liver.nii.gz')
    #     # os.symlink(f'{old_dir_name}/FU_Scan_Liver.nii.gz', f'{new_dir_name}/FU_Scan_Liver.nii.gz')
    #
    #     bl_liver, _ = load_nifti_data(f'{old_dir_name}/BL_Scan_Liver.nii.gz')
    #
    #     # copyfile(f'{old_dir_name}/BL_Scan_Tumors.nii.gz', f'{new_dir_name}/BL_Scan_Tumors.nii.gz')
    #     bl_tumors, file = load_nifti_data(f'{old_dir_name}/BL_Scan_Tumors.nii.gz')
    #     bl_tumors = pre_process_segmentation(bl_tumors)
    #     bl_liver = np.logical_or(bl_liver, bl_tumors).astype(bl_liver.dtype)
    #     bl_liver = getLargestCC(bl_liver)
    #     bl_liver = pre_process_segmentation(bl_liver, remove_small_obs=False)
    #     bl_tumors = np.logical_and(bl_tumors, bl_liver).astype(bl_tumors.dtype)
    #     # save(Nifti1Image(bl_tumors, file.affine), f'{new_dir_name}/BL_Scan_Tumors.nii.gz')
    #
    #     bl_tumors = get_connected_components(bl_tumors).astype(bl_tumors.dtype)
    #     n_bl_tumors = np.unique(bl_tumors).size - 1
    #     save(Nifti1Image(bl_tumors, file.affine), f'{new_dir_name}/BL_Scan_Tumors_unique_{n_bl_tumors}_CC.nii.gz')
    #
    #     fu_liver, _ = load_nifti_data(f'{old_dir_name}/FU_Scan_Liver.nii.gz')
    #     fu_tumors, file = load_nifti_data(f'{old_dir_name}/FU_Scan_Tumors.nii.gz')
    #     fu_tumors = pre_process_segmentation(fu_tumors)
    #     fu_liver = np.logical_or(fu_liver, fu_tumors).astype(fu_liver.dtype)
    #     fu_liver = getLargestCC(fu_liver)
    #     fu_liver = pre_process_segmentation(fu_liver, remove_small_obs=False)
    #     fu_tumors = np.logical_and(fu_tumors, fu_liver).astype(fu_tumors.dtype)
    #     fu_tumors = get_connected_components(fu_tumors).astype(fu_tumors.dtype)
    #     # fu_tumors[fu_tumors != 0] += 1
    #     n_fu_tumors = np.unique(fu_tumors).size - 1
    #     save(Nifti1Image(fu_tumors, file.affine), f'{new_dir_name}/FU_Scan_Tumors_unique_{n_fu_tumors}_CC.nii.gz')
    #
    #     # assert n_bl_t == n_bl_tumors, f'n_bl_t={n_bl_t}, n_bl_tumors={n_bl_tumors}'
    #     # assert n_fu_t == n_fu_tumors, f'n_fu_t={n_fu_t}, n_fu_tumors={n_fu_tumors}'
    #
    # exit(0)

    # ------------------------------------------------------------------------------------------------------------------

    # t = time()
    # fu_tumors_files = glob('matching/*/FU_Scan_Tumors_unique_*')
    # matching_measures_of_CCs = process_map(get_measures_of_CCs_of_GT, fu_tumors_files, max_workers=os.cpu_count() - 2)
    # matching_measures_of_CCs = pd.concat([pd.DataFrame(t, columns=['Name', 'Match', 'BL Tumor Volume (CC)',
    #                                                                'FU Tumor Volume (CC)',
    #                                                                'Minimum Distance between tumors (mm)',
    #                                                                'Distance between centroid of tumors (mm)',
    #                                                                'Overlap (CC)', 'Overlap with BL (%)',
    #                                                                'Overlap with FU (%)', 'HD (mm)', 'ASSD (mm)', 'Dice',
    #                                                                'IOU', 'Volume difference (%)',
    #                                                                'Volume difference (CC)']) for t in matching_measures_of_CCs])
    # matching_measures_of_CCs = sort_dataframe_by_key(matching_measures_of_CCs, 'Name', pairs_sort_key)
    # matching_measures_of_CCs['Name'] = [f'{name}_{matching_measures_of_CCs.iloc[i, 1]}' for i, name in matching_measures_of_CCs['Name'].iteritems()]
    # del matching_measures_of_CCs['Match']
    #
    # writer = pd.ExcelWriter(f'matching/matching_measures_of_CCs_in_GT.xlsx', engine='xlsxwriter')
    # write_to_excel(matching_measures_of_CCs, writer, matching_measures_of_CCs.columns.to_list(),
    #                column_name_as_index='Name')
    # writer.save()
    #
    # print(f'\nfinished the matchings measures of CCs in GT in: {calculate_runtime(t)}')
    #
    # exit(0)

    # df = {}
    # for pred_matches_file in sorted(glob(f'matching/*/pred_matching_graph.json')):
    #     _, _, pred_matches, case_name, _, _ = load_matching_grph(pred_matches_file)
    #     gt_matches_file = os.path.dirname(pred_matches_file) + '/' + os.path.basename(pred_matches_file).replace('pred_', 'gt_')
    #     _, _, gt_matches, _, _, _ = load_matching_grph(gt_matches_file)
    #     pred_matches.sort(key=lambda m: (m[1], m[0]))
    #     print(case_name, pred_matches)
    #
    # exit(0)

    # def match_func(bl_tumors_CC_labeled, fu_tumors_CC_labeled, voxelspacing=None,
    #                max_dilate_param=13, return_iteration_and_reverse_indicator=None):
    #     bl_tumors_CC_labeled = expand_labels(bl_tumors_CC_labeled, distance=max_dilate_param, voxelspacing=voxelspacing)
    #     fu_tumors_CC_labeled = expand_labels(fu_tumors_CC_labeled, distance=max_dilate_param, voxelspacing=voxelspacing)
    #     pairs = np.hstack([bl_tumors_CC_labeled.reshape([-1, 1]), fu_tumors_CC_labeled.reshape([-1, 1])])
    #     pairs = np.unique(pairs[~np.any(pairs == 0, axis=1)], axis=0)
    #     pairs = [(0, max_dilate_param, (m[0], m[1])) for m in pairs]
    #     return pairs

    def match_after_improving_registration(bl_tumors_CC_labeled, fu_tumors_CC_labeled, bl_liver, fu_liver,
                                           file_affine_matrix, voxelspacing=None, max_dilate_param=5,
                                           return_iteration_indicator=None):
                                           # return_iteration_and_reverse_indicator=None):
        (bl_down, fu_down, bl_fpfh, fu_fpfh, bl_tumors_pc, fu_tumors_pc, relevant_bl_tumors_pc,
         relevant_fu_tumors_pc, ICP_bl_pc, ICP_fu_pc) = prepare_dataset(bl_tumors_CC_labeled, fu_tumors_CC_labeled,
                                                                        bl_liver=bl_liver, fu_liver=fu_liver,
                                                                        file_affine_matrix=file_affine_matrix,
                                                                        voxelspacing=voxelspacing, voxel_size=1,
                                                                        center_of_mass_for_RANSAC=True, n_biggest=None,
                                                                        ICP_with_liver_border=False,
                                                                        RANSAC_with_liver_border=True,
                                                                        RANSAC_with_tumors=False,
                                                                        ICP_with_tumors=True)
        result_ransac = execute_fast_global_registration(bl_down, fu_down, bl_fpfh, fu_fpfh, 1)
        result_icp = execute_ICP(ICP_bl_pc, ICP_fu_pc, 1, distance_threshold_factor=40,
                                     init_transformation=result_ransac.transformation)

        transform_inverse = np.linalg.inv(file_affine_matrix) @ np.linalg.inv(result_icp.transformation) @ file_affine_matrix
        transformed_bl_labeled_tumors = affine_transform(bl_tumors_CC_labeled, transform_inverse, order=0)

        # return match_2_cases(transformed_bl_labeled_tumors, fu_tumors_CC_labeled, voxelspacing, max_dilate_param,
        #                      return_iteration_and_reverse_indicator)
        return match_2_cases_v3(transformed_bl_labeled_tumors, fu_tumors_CC_labeled, voxelspacing, max_dilate_param,
                                return_iteration_indicator)

    try:
        all_run_t = time()
        multiprocess = True
        read_saved_pred_matches = False
        save_pred_matches = True
        match_func = match_2_cases_v5
        adaptive_num_of_dilations = False

        for liver_study in [True, False]:

            # todo delete
            if liver_study:
                continue

            for without_improving_registration in ([False, True] if liver_study else [False]):

                if liver_study:
                    _without_improving_registration = without_improving_registration
                else:
                    _without_improving_registration = False

                get_patient_name = lambda case_name: '_'.join([c for c in case_name.replace('BL_', '').split('_FU_')[0].split('_') if not c.isdigit()])
                get_bl_name = lambda case_name: case_name.replace('BL_', '').split('_FU_')[0]
                get_fu_name = lambda case_name: case_name.replace('BL_', '').split('_FU_')[1]
                get_date_from_name_y_m_d = lambda name: [int(f) for f in name.split('_')[-3:][::-1]]
                get_time_interval = lambda pair: abs((date(*get_date_from_name_y_m_d(get_fu_name(pair))) - date(*get_date_from_name_y_m_d(get_bl_name(pair)))).days)

                def filter_num_of_pairs_per_patient(pair_names: List[str], max_pairs_per_patient: int = 20, pair_name_is_a_full_path_dir: bool = False) -> List[str]:
                    if pair_name_is_a_full_path_dir:
                        pairs = [os.path.basename(p) for p in pair_names]
                    else:
                        pairs = pair_names
                    df = pd.DataFrame(data=list(zip(pair_names, (get_patient_name(p) for p in pairs), (get_time_interval(p) for p in pairs))), columns=['case_name', 'patient', 'time_interval'])
                    res = []
                    for patient, patient_df in  df.groupby('patient'):
                        if patient_df.shape[0] > max_pairs_per_patient:
                            patient_df = patient_df.sort_values('time_interval', ignore_index=True)
                            patient_df = patient_df.iloc[:max_pairs_per_patient, :]
                        res += patient_df['case_name'].to_list()
                    return res

                if liver_study:
                    # dir_results = '/cs/casmip/rochman/Errors_Characterization/corrected_segmentation_for_matching/measures_results_after_improving_registration_with_only_liver_border_at_RANSAC_and_ICP_no_tumors_+_match_algo_v3'
                    # dir_results = '/cs/casmip/rochman/Errors_Characterization/corrected_segmentation_for_matching/measures_results_after_improving_registration_with_only_liver_border_at_RANSAC_and_only_tumors_at_ICP_+_match_algo_v3'
                    # dir_results = '/cs/casmip/rochman/Errors_Characterization/corrected_segmentation_for_matching/measures_results_after_improving_registration_with_liver_for_RANSAC_and_ICP_if_liver_diff_less_300_else_liver_RANSAC_tumors_ICP+_match_algo_v3'
                    # dir_results = '/cs/casmip/rochman/Errors_Characterization/new_test_set_for_matching/all_test_set_measures_results_after_improving_registration_with_only_liver_border_at_RANSAC_and_ICP_no_tumors_+_match_algo_v3'
                    # dir_results = '/cs/casmip/rochman/Errors_Characterization/new_test_set_for_matching/all_test_set_measures_results_after_improving_registration_with_only_liver_border_at_RANSAC_and_ICP_no_tumors_+_match_algo_v5'
                    if not without_improving_registration:
                        dir_results = '/cs/casmip/rochman/Errors_Characterization/new_test_set_for_matching/all_test_set_measures_results_after_improving_registration_with_only_liver_border_at_ICP_no_tumors_and_no_RANSAC_+_match_algo_v5'
                    else:
                        dir_results = '/cs/casmip/rochman/Errors_Characterization/new_test_set_for_matching/all_test_set_measures_results_without_improving_registration_+_match_algo_v5'
                    # dir_results = '/cs/casmip/rochman/Errors_Characterization/new_test_set_for_matching/all_test_set_measures_results_after_improving_registration_with_only_liver_border_at_RANSAC_and_ICP_no_tumors_+_match_algo_v5_+_adaptive_num_of_dilations'
                    # dir_results = '/cs/casmip/rochman/Errors_Characterization/new_test_set_for_matching/all_test_set_measures_results_after_improving_registration_with_liver_border_at_RANSAC_and_tumors_and_liver_border_at_ICP_+_match_algo_v3'
                else:
                    dir_results = '/cs/casmip/rochman/Errors_Characterization/lung_test_set_for_matching/measures_results'
                os.makedirs(dir_results, exist_ok=True)
                for max_dilate_param in range((5 if not liver_study else (15 if without_improving_registration else 7)),
                                              (6 if not liver_study else (16 if without_improving_registration else 8))):
                    t = time()
                    print(f'\nstarting the matchings with {max_dilate_param} iterations')

                    if liver_study:
                        fu_tumors_files = glob('/cs/casmip/rochman/Errors_Characterization/new_test_set_for_matching/BL_*/FU_Scan_Tumors_unique_*')
                        fu_tumors_files += glob('/cs/casmip/rochman/Errors_Characterization/corrected_segmentation_for_matching/BL_*/FU_Scan_Tumors_unique_*')
                    else:
                        fu_tumors_files = glob('/cs/casmip/rochman/Errors_Characterization/lung_test_set_for_matching/BL_*/FU_Scan_Tumors_unique_*')

                    fu_tumors_files = [f for f in fu_tumors_files if os.path.isfile(f'{os.path.dirname(f)}/gt_matching_graph.json')]

                    filtered_pairs = filter_num_of_pairs_per_patient([os.path.dirname(f) for f in fu_tumors_files],
                                                                     max_pairs_per_patient=20, pair_name_is_a_full_path_dir=True)

                    fu_tumors_files = [f for f in fu_tumors_files if os.path.dirname(f) in filtered_pairs]

                    if multiprocess:
                        matching_statistics, matching_measures, tumor_diameters = zip(*process_map(
                            partial(get_measures_and_statistics_of_matching, read_saved_pred_matches=read_saved_pred_matches, save_pred_matches=save_pred_matches,
                                    max_dilate_param=max_dilate_param, match_func=match_func, adaptive_num_of_dilations=adaptive_num_of_dilations, liver_study=liver_study,
                                    without_improving_registration=_without_improving_registration),
                            fu_tumors_files, max_workers=18))
                    else:
                        matching_statistics, matching_measures, tumor_diameters = zip(*tqdm(map(
                            partial(get_measures_and_statistics_of_matching, read_saved_pred_matches=read_saved_pred_matches, save_pred_matches=save_pred_matches,
                                    max_dilate_param=max_dilate_param, match_func=match_func, adaptive_num_of_dilations=adaptive_num_of_dilations, liver_study=liver_study),
                            fu_tumors_files)))

                    tumor_diameters = pd.concat([pd.DataFrame(t, columns=['Name', 'Tumor', 'Diameter (mm)', 'Is T_IS', 'Is F_IS', 'Is F_NOT_IS', 'Is T_NOT_IS']) for t in tumor_diameters])
                    tumor_diameters = sort_dataframe_by_key(tumor_diameters, 'Name', pairs_sort_key)
                    tumor_diameters['Name'] = [f'{name}_{tumor_diameters.iloc[i, 1]}' for i, name in tumor_diameters['Name'].iteritems()]
                    del tumor_diameters['Tumor']

                    tumor_diameters_GT_IS = tumor_diameters[(tumor_diameters['Is T_IS'] == 1) | (tumor_diameters['Is F_NOT_IS'] == 1)]
                    tumor_diameters_GT_IS.rename(columns={'Is T_IS': 'Is Pred IS'}, inplace=True)
                    del tumor_diameters_GT_IS['Is F_IS']
                    del tumor_diameters_GT_IS['Is F_NOT_IS']
                    del tumor_diameters_GT_IS['Is T_NOT_IS']

                    tumor_diameters_GT_NIS = tumor_diameters[(tumor_diameters['Is T_NOT_IS'] == 1) | (tumor_diameters['Is F_IS'] == 1)]
                    tumor_diameters_GT_NIS.rename(columns={'Is T_NOT_IS': 'Is Pred Not IS'}, inplace=True)
                    del tumor_diameters_GT_NIS['Is F_IS']
                    del tumor_diameters_GT_NIS['Is F_NOT_IS']
                    del tumor_diameters_GT_NIS['Is T_IS']

                    tumor_diameters_Pred_IS = tumor_diameters[(tumor_diameters['Is T_IS'] == 1) | (tumor_diameters['Is F_IS'] == 1)]
                    tumor_diameters_Pred_IS.rename(columns={'Is T_IS': 'Is GT IS'}, inplace=True)
                    del tumor_diameters_Pred_IS['Is F_IS']
                    del tumor_diameters_Pred_IS['Is F_NOT_IS']
                    del tumor_diameters_Pred_IS['Is T_NOT_IS']

                    tumor_diameters_Pred_NIS = tumor_diameters[(tumor_diameters['Is T_NOT_IS'] == 1) | (tumor_diameters['Is F_NOT_IS'] == 1)]
                    tumor_diameters_Pred_NIS.rename(columns={'Is T_NOT_IS': 'Is GT Not IS'}, inplace=True)
                    del tumor_diameters_Pred_NIS['Is F_IS']
                    del tumor_diameters_Pred_NIS['Is F_NOT_IS']
                    del tumor_diameters_Pred_NIS['Is T_IS']

                    tumor_diameters_T_IS = tumor_diameters[tumor_diameters['Is T_IS'] == 1]
                    del tumor_diameters_T_IS['Is T_IS']
                    del tumor_diameters_T_IS['Is F_IS']
                    del tumor_diameters_T_IS['Is F_NOT_IS']
                    del tumor_diameters_T_IS['Is T_NOT_IS']

                    tumor_diameters_F_IS = tumor_diameters[tumor_diameters['Is F_IS'] == 1]
                    del tumor_diameters_F_IS['Is T_IS']
                    del tumor_diameters_F_IS['Is F_IS']
                    del tumor_diameters_F_IS['Is F_NOT_IS']
                    del tumor_diameters_F_IS['Is T_NOT_IS']

                    tumor_diameters_F_NOT_IS = tumor_diameters[tumor_diameters['Is F_NOT_IS'] == 1]
                    del tumor_diameters_F_NOT_IS['Is T_IS']
                    del tumor_diameters_F_NOT_IS['Is F_IS']
                    del tumor_diameters_F_NOT_IS['Is F_NOT_IS']
                    del tumor_diameters_F_NOT_IS['Is T_NOT_IS']

                    tumor_diameters_T_NOT_IS = tumor_diameters[tumor_diameters['Is T_NOT_IS'] == 1]
                    del tumor_diameters_T_NOT_IS['Is T_IS']
                    del tumor_diameters_T_NOT_IS['Is F_IS']
                    del tumor_diameters_T_NOT_IS['Is F_NOT_IS']
                    del tumor_diameters_T_NOT_IS['Is T_NOT_IS']

                    # bl_tumor_diameter, fu_tumor_diameter, min_dist, center_min_dist, overlap, overlap_with_bl,
                    # overlap_with_fu, hausdorff, ASSD, dcs, IOU, volume_diff

                    matching_measures = pd.concat([pd.DataFrame(t, columns=['Name', 'Match', 'BL Tumor Diameter (mm)',
                                                                            'FU Tumor Diameter (mm)',
                                                                            'Minimum Distance between tumors (mm)',
                                                                            'Distance between centroid of tumors (mm)',
                                                                            'Overlap (CC)', 'Overlap with BL (%)',
                                                                            'Overlap with FU (%)', 'HD (mm)', 'ASSD (mm)', 'Dice',
                                                                            'IOU', 'Volume difference (%)',
                                                                            'Is TC', 'Is FC', 'Is FUC']) for t in matching_measures])

                    matching_measures = sort_dataframe_by_key(matching_measures, 'Name', pairs_sort_key)
                    matching_measures['Name'] = [f'{name}_({int(matching_measures.iloc[i, 1][0])},{int(matching_measures.iloc[i, 1][1])})' for i, name in matching_measures['Name'].iteritems()]
                    del matching_measures['Match']

                    matching_measures_matched_GT = matching_measures[(matching_measures['Is TC'] == 1) | (matching_measures['Is FUC'] == 1)].copy()
                    matching_measures_matched_GT.rename(columns={'Is TC': 'Is Predicted'}, inplace=True)
                    del matching_measures_matched_GT['Is FC']
                    del matching_measures_matched_GT['Is FUC']

                    matching_measures_matched_Pred = matching_measures[(matching_measures['Is TC'] == 1) | (matching_measures['Is FC'] == 1)].copy()
                    matching_measures_matched_Pred.rename(columns={'Is TC': 'Is In GT'}, inplace=True)
                    del matching_measures_matched_Pred['Is FC']
                    del matching_measures_matched_Pred['Is FUC']

                    matching_measures_TC = matching_measures[matching_measures['Is TC'] == 1].copy()
                    del matching_measures_TC['Is TC']
                    del matching_measures_TC['Is FC']
                    del matching_measures_TC['Is FUC']

                    matching_measures_FC = matching_measures[matching_measures['Is FC'] == 1].copy()
                    del matching_measures_FC['Is TC']
                    del matching_measures_FC['Is FC']
                    del matching_measures_FC['Is FUC']

                    matching_measures_FUC = matching_measures[matching_measures['Is FUC'] == 1].copy()
                    del matching_measures_FUC['Is TC']
                    del matching_measures_FUC['Is FC']
                    del matching_measures_FUC['Is FUC']

                    matching_statistics = pd.DataFrame(matching_statistics, columns=['Name', 'Precision - Edges', 'Recall - Edges',
                                                                                     'F1-Score - Edges', 'TP - Edges', 'FP - Edges',
                                                                                     'FN - Edges', 'Precision/PPV - Isolated-Tumors',
                                                                                     'Recall/TPR/Sensitivity - Isolated-Tumors',
                                                                                     'F1-Score - Isolated-Tumors',
                                                                                     'Specificity/TNR - Isolated-Tumors',
                                                                                     'NPV - Isolated-Tumors',
                                                                                     'Accuracy - Isolated-Tumors',
                                                                                     'TP - Isolated-Tumors', 'FP - Isolated-Tumors',
                                                                                     'FN - Isolated-Tumors', 'TN - Isolated-Tumors',
                                                                                     'Positive-Rate - Isolated-Tumors',
                                                                                     'Negative-Rate - Isolated-Tumors',
                                                                                     'num BL tumors', 'num FU tumors', 'num of edges',
                                                                                     'BL&FU Liver-Dice', 'BL&FU Liver-ASSD (mm)',
                                                                                     'BL&FU Liver-HD (mm)', 'BL&FU Liver-ABS-volume-diff (CC)',
                                                                                     'BL&FU Tumors-ABS-volume-diff (CC)',
                                                                                     'Time-range between scans (Days)',
                                                                                     'Matching Time (s)'])
                    matching_statistics = sort_dataframe_by_key(matching_statistics, 'Name', pairs_sort_key)
                    writer = pd.ExcelWriter(f'{dir_results}/matching_statistics_dilate_{max_dilate_param}.xlsx', engine='xlsxwriter')
                    write_to_excel(matching_statistics, writer, ['Name', 'Precision - Edges', 'Recall - Edges', 'F1-Score - Edges',
                                                                 'TP - Edges', 'FP - Edges', 'FN - Edges', 'num BL tumors',
                                                                 'num FU tumors', 'num of edges', 'BL&FU Liver-Dice',
                                                                 'BL&FU Liver-ASSD (mm)', 'BL&FU Liver-HD (mm)',
                                                                 'BL&FU Liver-ABS-volume-diff (CC)',
                                                                 'BL&FU Tumors-ABS-volume-diff (CC)',
                                                                 'Time-range between scans (Days)', 'Matching Time (s)'],
                                   column_name_as_index='Name', sheet_name='Edges Statistics',
                                   f1_scores={'F1-Score - Edges': ('Precision - Edges', 'Recall - Edges')})
                    write_to_excel(matching_statistics, writer, ['Name', 'Precision/PPV - Isolated-Tumors',
                                                                 'Recall/TPR/Sensitivity - Isolated-Tumors',
                                                                 'F1-Score - Isolated-Tumors', 'Specificity/TNR - Isolated-Tumors',
                                                                 'NPV - Isolated-Tumors', 'Accuracy - Isolated-Tumors',
                                                                 'TP - Isolated-Tumors', 'FP - Isolated-Tumors',
                                                                 'FN - Isolated-Tumors', 'TN - Isolated-Tumors',
                                                                 'Positive-Rate - Isolated-Tumors', 'Negative-Rate - Isolated-Tumors',
                                                                 'num BL tumors', 'num FU tumors', 'num of edges',
                                                                 'BL&FU Liver-Dice', 'BL&FU Liver-ASSD (mm)',
                                                                 'BL&FU Liver-HD (mm)', 'BL&FU Liver-ABS-volume-diff (CC)',
                                                                 'BL&FU Tumors-ABS-volume-diff (CC)',
                                                                 'Time-range between scans (Days)', 'Matching Time (s)'],
                                   column_name_as_index='Name', sheet_name='Isolation Statistics',
                                   f1_scores={'F1-Score - Isolated-Tumors': ('Precision/PPV - Isolated-Tumors',
                                                                             'Recall/TPR/Sensitivity - Isolated-Tumors')})
                    writer.save()

                    # ------------

                    writer = pd.ExcelWriter(f'{dir_results}/matching_measures_dilate_{max_dilate_param}.xlsx', engine='xlsxwriter')
                    write_to_excel(matching_measures, writer, matching_measures.columns.to_list(), column_name_as_index='Name',
                                   sheet_name='All')
                    write_to_excel(matching_measures_matched_GT, writer, matching_measures_matched_GT.columns.to_list(), column_name_as_index='Name',
                                   sheet_name='GT')
                    write_to_excel(matching_measures_matched_Pred, writer, matching_measures_matched_Pred.columns.to_list(), column_name_as_index='Name',
                                   sheet_name='Predictions')
                    write_to_excel(matching_measures_TC, writer, matching_measures_TC.columns.to_list(), column_name_as_index='Name',
                                   sheet_name='TC')
                    write_to_excel(matching_measures_FC, writer, matching_measures_FC.columns.to_list(), column_name_as_index='Name',
                                   sheet_name='FC')
                    write_to_excel(matching_measures_FUC, writer, matching_measures_FUC.columns.to_list(), column_name_as_index='Name',
                                   sheet_name='FUC')
                    writer.save()

                    #------------

                    writer = pd.ExcelWriter(f'{dir_results}/isolation_measures_dilate_{max_dilate_param}.xlsx', engine='xlsxwriter')
                    write_to_excel(tumor_diameters, writer, tumor_diameters.columns.to_list(), column_name_as_index='Name',
                                   sheet_name='All')
                    write_to_excel(tumor_diameters_GT_IS, writer, tumor_diameters_GT_IS.columns.to_list(), column_name_as_index='Name',
                                   sheet_name='GT IS')
                    write_to_excel(tumor_diameters_GT_NIS, writer, tumor_diameters_GT_NIS.columns.to_list(), column_name_as_index='Name',
                                   sheet_name='GT NIS')
                    write_to_excel(tumor_diameters_Pred_IS, writer, tumor_diameters_Pred_IS.columns.to_list(), column_name_as_index='Name',
                                   sheet_name='Pred IS')
                    write_to_excel(tumor_diameters_Pred_NIS, writer, tumor_diameters_Pred_NIS.columns.to_list(), column_name_as_index='Name',
                                   sheet_name='Pred NIS')
                    write_to_excel(tumor_diameters_T_IS, writer, tumor_diameters_T_IS.columns.to_list(), column_name_as_index='Name',
                                   sheet_name='T_IS')
                    write_to_excel(tumor_diameters_F_IS, writer, tumor_diameters_F_IS.columns.to_list(), column_name_as_index='Name',
                                   sheet_name='F_IS')
                    write_to_excel(tumor_diameters_F_NOT_IS, writer, tumor_diameters_F_NOT_IS.columns.to_list(), column_name_as_index='Name',
                                   sheet_name='F_NOT_IS')
                    write_to_excel(tumor_diameters_T_NOT_IS, writer, tumor_diameters_T_NOT_IS.columns.to_list(), column_name_as_index='Name',
                                   sheet_name='T_NOT_IS')
                    writer.save()
                    print(f'\nfinished the matchings with {max_dilate_param} iterations in: {calculate_runtime(t)}')

                    from errors_characterization import write_diff_in_matching_according_to_lesion_type

                    write_diff_in_matching_according_to_lesion_type(filtered_pairs,
                                                                    results_file=f'{dir_results}/diff_in_matching_according_to_CC_type_dilate_{max_dilate_param}.xlsx',
                                                                    max_dilation_for_pred=max_dilate_param, without_improving_registration=_without_improving_registration
                                                                    )

        print(f'The matchings has ended successfully in {calculate_runtime(all_run_t)} (hh:mm:ss)')
        notify(f'The matchings has ended successfully in {calculate_runtime(all_run_t)} (hh:mm:ss)')
    except Exception as e:
        notify(f"\nThere was an error at running the matching algorithm with {max_dilate_param} iterations: {e}",
               error=True)
        raise e
    # ------------------------------------------------------------------------------------------------------------------

    # matching_statistics = pd.read_excel('matching/matching_statistics.xlsx').rename(columns={'Unnamed: 0': 'name'})
    # matching_statistics = matching_statistics[matching_statistics['name'].str.startswith('BL_')]
    # registration_errors = pd.read_excel('registration_scores.xlsx')[['name', 'dice', 'assd', 'hd', 'abs_liver_diff', 'liver_mutual_information', 'ct_mutual_information']]
    # registration_errors = registration_errors[registration_errors['name'].isin(matching_statistics['name'])]
    # matching_statistics.set_index(keys='name', drop=True, inplace=True)
    # registration_errors.set_index(keys='name', drop=True, inplace=True)
    # matching_statistics = matching_statistics.join(registration_errors, on='name')
    # matching_statistics['TPR - Edges'] = matching_statistics['TP - Edges']/(matching_statistics['TP - Edges'] + matching_statistics['FN - Edges'])
    #
    # # temp = matching_statistics.sort_values(by='F1-Score - Edges')
    # # temp.set_index('F1-Score - Edges').dice.plot()
    #
    # # matching_statistics.plot(x='TPR - Edges', y='dice', style='o')
    # # plt.show()
    # #
    # # temp = matching_statistics.sort_values(by='Recall - Edges')
    # # temp.set_index('Recall - Edges').dice.plot()
    # # plt.show()
    # #
    # #
    # #
    # # import seaborn as sns
    # #
    # # corr = matching_statistics.corr()
    # # ax = sns.heatmap(
    # #     corr,
    # #     vmin=-1, vmax=1, center=0,
    # #     cmap=sns.diverging_palette(20, 220, n=200),
    # #     square=True
    # # )
    # # ax.set_xticklabels(
    # #     ax.get_xticklabels(),
    # #     rotation=45,
    # #     horizontalalignment='right'
    # # )
    # # plt.show()
    #
    #
    # matching_measures = pd.read_excel('matching/matching_measures.xlsx').rename(columns={'Unnamed: 0': 'name'})
    # matching_measures = matching_measures[matching_measures['name'].str.startswith('BL_')]
    # matching_measures['GT'] = ((matching_measures['Is TP'] == 1) | (matching_measures['Is FN'] == 1)).astype(np.int)
    # print()

    # ------------------------------------------------------------------------------------------------------------------

    # # # todo delete
    # def measures_of_GT_set_of_matchings(fu_case_file: str):
    #     fu_name = os.path.basename(fu_case_file)
    #     n_fu_tumors = int(''.join(c for c in fu_name if c.isdigit()))
    #
    #     bl_name = os.path.basename(glob(f'{os.path.dirname(fu_case_file)}/BL_Scan_Tumors_unique_*')[0])
    #     n_bl_tumors = int(''.join(c for c in bl_name if c.isdigit()))
    #
    #     excel_file = f'{os.path.dirname(fu_case_file)}/matching{"_corrected" if "round_1" in fu_case_file else ""}.xlsx'
    #     n_matches = pd.read_excel(excel_file).shape[0]
    #
    #     case_name = os.path.basename(os.path.dirname(fu_case_file))
    #
    #     return case_name, n_bl_tumors, n_fu_tumors, n_matches
    # #
    # def get_matches_from_excel_file(excel_file: str):
    #     matches = pd.read_excel(excel_file)
    #     matches.dropna(inplace=True)
    #     matches = matches.values.tolist()
    #     return [tuple(m) for m in matches]
    #
    #
    # src_dir = '/cs/casmip/rochman/Errors_Characterization/matching_for_richard/round_4'
    # dst_dir = '/cs/casmip/rochman/Errors_Characterization/corrected_segmentation_for_matching'
    # for i, xl_file in enumerate(glob(f'{src_dir}/*/matching_corrected.xlsx')):
    #     print(i + 1)
    #     case_name = os.path.basename(os.path.dirname(xl_file))
    #     gt_matches = get_matches_from_excel_file(xl_file)
    #     n_bl_nodes, n_fu_nodes, pred_matches, case_name, bl_weights, fu_weights = load_matching_grph(f'{dst_dir}/{case_name}/pred_matching_graph.json')
    #     save_matching_graph(n_bl_nodes, n_fu_nodes, gt_matches, case_name,
    #                         f'{dst_dir}/{case_name}/gt_matching_graph.json', bl_weights, fu_weights)

    # #
    # # def extract_changes_statistics(excel_file: str):
    # #     gt_matches = get_matches_from_excel_file(excel_file)
    # #
    # #     previous_dir = os.path.dirname(excel_file).replace('/matching_for_richard/round_1/', '/matching/') + '_done'
    # #     n_bl_tumors, n_fu_tumors, pred_matches, case_name = load_matching_grph(f'{previous_dir}/gt_matching_graph.json')
    # #
    # #     matching_statistics = calculate_statistics_over_matching(gt_matches, pred_matches, n_bl_tumors, n_fu_tumors)
    # #
    # #     save_matching_graph(n_bl_tumors, n_fu_tumors, gt_matches, case_name, excel_file.replace('/matching_corrected.xlsx', '/gt_matching_graph.json'))
    # #
    # #     n_matches = len(gt_matches)
    # #
    # #     return (case_name, *matching_statistics, n_bl_tumors, n_fu_tumors, n_matches, 0)
    # #
    # # def extract_changes_statistics_in_round_2(excel_file: str):
    # #     gt_matches = get_matches_from_excel_file(excel_file)
    # #     pred_matches = get_matches_from_excel_file(replace_in_file_name(excel_file, '/matching_corrected.xlsx', '/matching.xlsx'))
    # #
    # #     case_name = os.path.basename(os.path.dirname(excel_file))
    # #     assert case_name.startswith('BL_')
    # #
    # #     bl_tumors_file = glob(f'{os.path.dirname(excel_file)}/BL_Scan_Tumors_unique_*')[0]
    # #     bl_case, _ = load_nifti_data(bl_tumors_file)
    # #     n_bl_tumors = np.unique(bl_case).size - 1
    # #
    # #     fu_tumors_file = glob(f'{os.path.dirname(excel_file)}/FU_Scan_Tumors_unique_*')[0]
    # #     fu_case, _ = load_nifti_data(fu_tumors_file)
    # #     n_fu_tumors = np.unique(fu_case).size - 1
    # #
    # #
    # #     bl_weights = []
    # #     for i in range(1, n_bl_tumors + 1):
    # #         bl_weights.append(int(centroid(bl_case == i)[-1]) + 1)
    # #
    # #     fu_weights = []
    # #     for i in range(1, n_fu_tumors + 1):
    # #         fu_weights.append(int(centroid(fu_case == i)[-1]) + 1)
    # #
    # #     matching_statistics = calculate_statistics_over_matching(gt_matches, pred_matches, n_bl_tumors, n_fu_tumors)
    # #
    # #     save_matching_graph(n_bl_tumors, n_fu_tumors, gt_matches, case_name,
    # #                         excel_file.replace('/matching_corrected.xlsx', '/gt_matching_graph.json'))
    # #     draw_matching_graph(n_bl_tumors, n_fu_tumors, gt_matches, case_name, bl_weights=bl_weights, fu_weights=fu_weights,
    # #                         saving_file_name=excel_file.replace('/matching_corrected.xlsx', '/gt_matching_graph.jpg'))
    # #
    # #     n_matches = len(gt_matches)
    # #
    # #     return (case_name, *matching_statistics, n_bl_tumors, n_fu_tumors, n_matches, 0)
    # #
    # # # import shutil
    # # # for excel_file in glob('/cs/casmip/rochman/Errors_Characterization/matching_for_richard/round_2_corrected/*/matching.xlsx'):
    # # #     shutil.copy(excel_file, excel_file.replace('/matching.xlsx', '/matching_corrected.xlsx').replace('/round_2_corrected/', '/round_2/'))
    # # # exit(0)
    # #
    # #
    # # excel_files = glob('/cs/casmip/rochman/Errors_Characterization/matching_for_richard/round_2/*/matching_corrected.xlsx')
    # # excel_files.sort()
    # #
    # # matching_statistics = process_map(extract_changes_statistics_in_round_2, excel_files, max_workers=os.cpu_count()-2)
    # # # matching_statistics = list(map(extract_changes_statistics_in_round_2, excel_files))
    # #
    # # matching_statistics = pd.DataFrame(matching_statistics, columns=['Name', 'Precision - Edges', 'Recall - Edges',
    # #                                                                  'F1-Score - Edges', 'TP - Edges', 'FP - Edges',
    # #                                                                  'FN - Edges', 'Precision/PPV - Isolated-Tumors',
    # #                                                                  'Recall/TPR/Sensitivity - Isolated-Tumors',
    # #                                                                  'F1-Score - Isolated-Tumors',
    # #                                                                  'Specificity/TNR - Isolated-Tumors',
    # #                                                                  'NPV - Isolated-Tumors',
    # #                                                                  'Accuracy - Isolated-Tumors',
    # #                                                                  'TP - Isolated-Tumors', 'FP - Isolated-Tumors',
    # #                                                                  'FN - Isolated-Tumors', 'TN - Isolated-Tumors',
    # #                                                                  'Positive-Rate - Isolated-Tumors',
    # #                                                                  'Negative-Rate - Isolated-Tumors',
    # #                                                                  'num BL tumors', 'num FU tumors', 'num of matches',
    # #                                                                  'Matching Time (s)'])
    # # matching_statistics = sort_dataframe_by_key(matching_statistics, 'Name', pairs_sort_key)
    # # writer = pd.ExcelWriter(f'matching_for_richard/round_2/changes_in_matching_statistics.xlsx', engine='xlsxwriter')
    # # write_to_excel(matching_statistics, writer, ['Name', 'Precision - Edges', 'Recall - Edges', 'F1-Score - Edges',
    # #                                              'TP - Edges', 'FP - Edges', 'FN - Edges', 'num BL tumors',
    # #                                              'num FU tumors', 'num of matches', 'Matching Time (s)'],
    # #                column_name_as_index='Name', sheet_name='Edges Statistics',
    # #                f1_scores={'F1-Score - Edges': ('Precision - Edges', 'Recall - Edges')})
    # # write_to_excel(matching_statistics, writer, ['Name', 'Precision/PPV - Isolated-Tumors',
    # #                                              'Recall/TPR/Sensitivity - Isolated-Tumors',
    # #                                              'F1-Score - Isolated-Tumors', 'Specificity/TNR - Isolated-Tumors',
    # #                                              'NPV - Isolated-Tumors', 'Accuracy - Isolated-Tumors',
    # #                                              'TP - Isolated-Tumors', 'FP - Isolated-Tumors',
    # #                                              'FN - Isolated-Tumors', 'TN - Isolated-Tumors',
    # #                                              'Positive-Rate - Isolated-Tumors', 'Negative-Rate - Isolated-Tumors',
    # #                                              'num BL tumors', 'num FU tumors', 'num of matches', 'Matching Time (s)'],
    # #                column_name_as_index='Name', sheet_name='Isolation Statistics',
    # #                f1_scores={'F1-Score - Isolated-Tumors': ('Precision/PPV - Isolated-Tumors',
    # #                                                          'Recall/TPR/Sensitivity - Isolated-Tumors')})
    # # writer.save()
    #
    # # # ------------------------------------------------------------------------------------------------------------------
    # #
    # fu_case_files = glob('/cs/casmip/rochman/Errors_Characterization/matching_for_richard/round_1/*/FU_Scan_Tumors_unique_*')
    # fu_case_files += glob('/cs/casmip/rochman/Errors_Characterization/matching_for_richard/round_2/*/FU_Scan_Tumors_unique_*')
    # fu_case_files.sort()
    #
    # GT_matching_set_measures = process_map(measures_of_GT_set_of_matchings, fu_case_files, max_workers=os.cpu_count()-2)
    # # GT_matching_set_measures = list(map(measures_of_GT_set_of_matchings, fu_case_files))
    #
    # GT_matching_set_measures = pd.DataFrame(GT_matching_set_measures, columns=['Name', 'num BL tumors',
    #                                                                            'num FU tumors', 'num of GT matches'])
    # matching_statistics = sort_dataframe_by_key(GT_matching_set_measures, 'Name', pairs_sort_key)
    # writer = pd.ExcelWriter(f'matching_for_richard/both_rounds_GT_matching_set_measures.xlsx', engine='xlsxwriter')
    # write_to_excel(matching_statistics, writer, ['Name', 'num BL tumors', 'num FU tumors', 'num of GT matches'],
    #                column_name_as_index='Name')
    # writer.save()

    # ------------------------------------------------------------------------------------------------------------------

    # import shutil
    #
    # def reorder_the_matching_files(fu_tumors_file):
    #     fu_case, file = load_nifti_data(fu_tumors_file)
    #     if not np.any(fu_case == 1):
    #         fu_case[fu_case > 0] -= 1
    #         save(Nifti1Image(fu_case, file.affine), fu_tumors_file)
    #
    #     bl_matching_file = os.path.dirname(fu_tumors_file) + '/BL_Scan_Tumors_matching.nii.gz'
    #     if os.path.isfile(bl_matching_file):
    #         os.remove(bl_matching_file)
    #
    #     old_gt_matching_graph_file = os.path.dirname(fu_tumors_file) + '/gt_matching_graph.json'
    #     # assert os.path.isfile(old_gt_matching_graph_file)
    #     new_gt_matching_graph_file = old_gt_matching_graph_file.replace('/matching/', '/matching_for_richard/round_1/').replace('_done/', '/')
    #     assert os.path.isfile(new_gt_matching_graph_file)
    #     shutil.copy(new_gt_matching_graph_file, old_gt_matching_graph_file)
    #
    #     n_bl_nodes, n_fu_nodes, edges, case_name = load_matching_grph(new_gt_matching_graph_file)
    #     bl_tumors_file = glob(f'{os.path.dirname(fu_tumors_file)}/BL_Scan_Tumors_unique_*')[0]
    #     bl_case, _ = load_nifti_data(bl_tumors_file)
    #     bl_weights = []
    #     for bl_t in range(1, n_bl_nodes + 1):
    #         bl_weights.append(int(centroid(bl_case == bl_t)[-1]) + 1)
    #     fu_weights = []
    #     for fu_t in range(1, n_fu_nodes + 1):
    #         fu_weights.append(int(centroid(fu_case == fu_t)[-1]) + 1)
    #     draw_matching_graph(n_bl_nodes, n_fu_nodes, edges, case_name + '_GT',
    #                         bl_weights=bl_weights, fu_weights=fu_weights,
    #                         saving_file_name=f'{os.path.dirname(fu_tumors_file)}/gt_matching_graph.jpg')
    #
    #     matching_excel_file1 = os.path.dirname(fu_tumors_file) + '/matching.xlsx'
    #     if os.path.isfile(matching_excel_file1):
    #         os.remove(matching_excel_file1)
    #
    #     matching_excel_file2 = os.path.dirname(fu_tumors_file) + '/matching_corrected.xlsx'
    #     if os.path.isfile(matching_excel_file2):
    #         os.remove(matching_excel_file2)
    #
    #     os.rename(os.path.dirname(fu_tumors_file), os.path.dirname(fu_tumors_file).replace('_done', ''))
    #
    # fu_tumors_files = glob(f'/cs/casmip/rochman/Errors_Characterization/matching/*_done/FU_Scan_Tumors_unique_*')
    # fu_tumors_files.sort()
    #
    # process_map(reorder_the_matching_files, fu_tumors_files, max_workers=os.cpu_count()-2)

    # ------------------------------------------------------------------------------------------------------------------

    # from shutil import copy
    #
    # cases = [dirname for dirname in glob('/cs/casmip/rochman/Errors_Characterization/matching_for_richard/round_2/*') if os.path.isdir(dirname)]
    # cases.sort()
    #
    # all_data_dir = '/mnt/sda1/aszeskin/Data_Followup_Full_29_4_2021'
    # dst_dir = '/cs/casmip/rochman/Errors_Characterization/matching'
    # for case_dir in cases:
    #     case_name = os.path.basename(case_dir)
    #     new_dir = f'{dst_dir}/{case_name}'
    #     os.makedirs(new_dir, exist_ok=True)
    #
    #     # BL_Scan_CT
    #     os.symlink(f'{all_data_dir}/{case_name}/BL_Scan_CT.nii.gz', f'{new_dir}/BL_Scan_CT.nii.gz')
    #
    #     # BL_Scan_Liver
    #     os.symlink(f'{all_data_dir}/{case_name}/BL_Scan_Liver.nii.gz', f'{new_dir}/BL_Scan_Liver.nii.gz')
    #
    #     # FU_Scan_CT
    #     os.symlink(f'{all_data_dir}/{case_name}/FU_Scan_CT.nii.gz', f'{new_dir}/FU_Scan_CT.nii.gz')
    #
    #     # FU_Scan_Liver
    #     os.symlink(f'{all_data_dir}/{case_name}/FU_Scan_Liver.nii.gz', f'{new_dir}/FU_Scan_Liver.nii.gz')
    #
    #     fu_tumors_file_old = glob(f'{case_dir}/FU_Scan_Tumors_unique_*')[0]
    #     fu_tumors_file_new = replace_in_file_name(fu_tumors_file_old, '/matching_for_richard/round_2/', '/matching/',
    #                                               dst_file_exist=False)
    #     os.rename(fu_tumors_file_old, fu_tumors_file_new)
    #     os.symlink(fu_tumors_file_new, fu_tumors_file_old)
    #
    #     bl_tumors_file_old = glob(f'{case_dir}/BL_Scan_Tumors_unique_*')[0]
    #     bl_tumors_file_new = replace_in_file_name(bl_tumors_file_old, '/matching_for_richard/round_2/', '/matching/',
    #                                               dst_file_exist=False)
    #     os.rename(bl_tumors_file_old, bl_tumors_file_new)
    #     os.symlink(bl_tumors_file_new, bl_tumors_file_old)
    #
    #     for matching_graph_file in ['gt_matching_graph.json', 'gt_matching_graph.jpg']:
    #         matching_graph_file_old = f'{case_dir}/{matching_graph_file}'
    #         matching_graph_file_new = replace_in_file_name(matching_graph_file_old, '/matching_for_richard/round_2/', '/matching/', dst_file_exist=False)
    #         copy(matching_graph_file_old, matching_graph_file_new)
