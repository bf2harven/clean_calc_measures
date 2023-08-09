from glob import glob
from typing import Optional

from utils import *
import nibabel as nib
from skimage.measure import centroid
from tqdm.contrib.concurrent import process_map
from functools import partial
import pandas as pd
from multiprocessing import cpu_count


def adaptive_threshold(pred_labels: np.ndarray, affine_matrix: np.ndarray,
                       pix_dims: Tuple[float, float, float], min_relevant_th: int = 1,
                       default_th: int = 8, n_processes: Optional[int] = None) -> np.ndarray:

    if n_processes is None:
        n_processes = cpu_count()

    pred_labels = pred_labels.astype(np.int16)

    min_th_case = (pred_labels >= min_relevant_th).astype(pred_labels.dtype)
    min_th_case = pre_process_segmentation(min_th_case)

    default_th_case = (pred_labels >= default_th).astype(pred_labels.dtype)
    default_th_case = pre_process_segmentation(default_th_case)

    min_th_unique_CC = get_connected_components(min_th_case)
    default_th_unique_CC = get_connected_components(default_th_case)

    # for label in np.unique(min_th_unique_CC)[1:]:
    #     tumor_labels = np.zeros_like(pred_labels)
    #     tumor_labels[min_th_unique_CC == label] = pred_labels[min_th_unique_CC == label]
    #
    #     max_th = 0
    #     max_dice = -np.inf
    #
    #     for th in np.unique(tumor_labels)[1:]:
    #         current_th_tumor = (tumor_labels == th).astype(pred_labels.dtype)
    #
    #         current_tumor_volume = current_th_tumor.sum() * voxel_volume
    #         current_tumor_diameter = approximate_diameter(current_tumor_volume)
    #
    #         current_tumor_centroid = centroid(current_th_tumor)
    #
    #         current_sphere = approximate_sphere(relevant_points_in_real_space,
    #                                             relevant_points_in_voxel_space,
    #                                             current_tumor_centroid, current_tumor_diameter / 2,
    #                                             affine_matrix)
    #
    #         current_dice = dice(current_th_tumor, current_sphere)
    #
    #         if current_dice > max_dice:
    #             max_th = th
    #             max_dice = current_dice
    #
    #     final_pred[tumor_labels == max_th] = 1
    #
    # final_pred = pre_process_segmentation(final_pred)
    #
    # return final_pred

    # min_th_CC_labels_and_th_values = list(product(np.unique(min_th_unique_CC)[1:], np.unique(pred_labels)[1:]))

    # computing relevant CC to consider
    default_and_min_CC_labels_intersect = np.hstack([default_th_unique_CC.reshape([-1, 1]), min_th_unique_CC.reshape([-1, 1])])
    default_and_min_CC_labels_intersect = np.unique(default_and_min_CC_labels_intersect[~np.any(default_and_min_CC_labels_intersect == 0, axis=1)], axis=0)

    min_th_unique_CC[~np.isin(min_th_unique_CC, default_and_min_CC_labels_intersect[:, 1])] = 0

    # filtering false positive
    # if min_th_for_filtering_FP is not None:
    #     df = pd.DataFrame(min_th_CC_labels_and_th_values, columns=['label', 'th'])
    #     max_th_df = df.groupby('label').max()
    #     max_th_df = max_th_df[max_th_df['th'] >= min_th_for_filtering_FP]
    #     df = df[df['label'].isin(max_th_df.index)]
    #     min_th_CC_labels_and_th_values = df.to_numpy()

    voxel_volume = pix_dims[0] * pix_dims[1] * pix_dims[2]
    # res = process_map(partial(get_best_th, pred_labels=pred_labels, min_th_unique_CC=min_th_unique_CC,
    #                   default_th_unique_CC=default_th_unique_CC, voxel_volume=voxel_volume,
    #                   affine_matrix=affine_matrix, pix_dims=pix_dims, default_th=default_th),
    #                   default_and_min_CC_labels_intersect, max_workers=n_processes)
    res = list(map(partial(get_best_th, pred_labels=pred_labels, min_th_unique_CC=min_th_unique_CC,
                           default_th_unique_CC=default_th_unique_CC, voxel_volume=voxel_volume,
                           affine_matrix=affine_matrix, pix_dims=pix_dims, default_th=default_th),
                      default_and_min_CC_labels_intersect))


    # all_dices = process_map(partial(dice_with_sphere_for_tumor_label_and_th, pred_labels=pred_labels,
    #                                 min_th_unique_CC=min_th_unique_CC, voxel_volume=voxel_volume,
    #                                 affine_matrix=affine_matrix,
    #                                 pix_dims=pix_dims), min_th_CC_labels_and_th_values, max_workers=n_processes)
    # with Pool() as pool:
    #     all_dices = pool.map(partial(dice_with_sphere_for_tumor_label_and_th, pred_labels=pred_labels,
    #                                     min_th_unique_CC=min_th_unique_CC, voxel_volume=voxel_volume,
    #                                     relevant_points_in_real_space=relevant_points_in_real_space,
    #                                     relevant_points_in_voxel_space=relevant_points_in_voxel_space),
    #                             min_th_CC_labels_and_th_values)
    # all_dices = list(map(partial(dice_with_sphere_for_tumor_label_and_th, pred_labels=pred_labels,
    #                                 min_th_unique_CC=min_th_unique_CC, voxel_volume=voxel_volume,
    #                                 affine_matrix=affine_matrix,
    #                                 pix_dims=pix_dims),
    #                         min_th_CC_labels_and_th_values))

    # best_th_s = pd.DataFrame(all_dices, columns=['label', 'th', 'dice']).sort_values('dice', ascending=False).drop_duplicates(['label']).sort_values('label')

    final_pred = np.zeros_like(pred_labels)
    for (label, th, _) in res:
        tumor_labels = np.where(min_th_unique_CC == label, pred_labels, 0)
        final_pred[tumor_labels >= th] = 1

    final_pred = pre_process_segmentation(final_pred)

    # # filtering false positive
    # if default_th is not None:
    #     final_pred_unique_CC = get_connected_components(final_pred)
    #     final_pred_unique_CC_flatten = final_pred_unique_CC.flatten()
    #     min_th_for_filtering_FP_unique_CC_flatten  = pre_process_segmentation((pred_labels >= default_th).astype(pred_labels.dtype)).flatten()
    #     relevant_final_labels = np.unique(np.hstack([final_pred_unique_CC_flatten.reshape([-1, 1]), min_th_for_filtering_FP_unique_CC_flatten.reshape([-1, 1])]), axis=0)
    #     relevant_final_labels = relevant_final_labels[~np.any(relevant_final_labels == 0, axis=1)][:, 0]
    #     final_pred = np.zeros_like(final_pred)
    #     for label in relevant_final_labels:
    #         final_pred[final_pred_unique_CC == label] = 1

    return final_pred


def get_best_th(default_th_CC_and_min_th_CC: Tuple[int, int], pred_labels: np.ndarray,
                min_th_unique_CC: np.ndarray, default_th_unique_CC: np.ndarray, voxel_volume: float,
                affine_matrix: np.ndarray, pix_dims: Tuple[float, float, float], default_th: int, eps: float = 1e-2) -> Tuple[int, int, float]:

    default_th_CC_label, min_th_CC_label = default_th_CC_and_min_th_CC

    # computing the approximate sphere
    nX, nY, nZ = pred_labels.shape
    min_p = affines.apply_affine(affine_matrix, (0, 0, 0))
    max_p = affines.apply_affine(affine_matrix, pred_labels.shape)
    relevant_points_in_real_space = np.vstack([np.repeat(np.arange(min_p[0], max_p[0], pix_dims[0]), nY * nZ),
                                               np.tile(np.repeat(np.arange(min_p[1], max_p[1], pix_dims[1]), nZ), nX),
                                               np.tile(np.arange(min_p[2], max_p[2], pix_dims[2]), nX * nY)]).T
    relevant_points_in_voxel_space = np.vstack([np.repeat(np.arange(0, nX), nY * nZ),
                                                np.tile(np.repeat(np.arange(0, nY), nZ), nX),
                                                np.tile(np.arange(0, nZ), nX * nY)]).T
    default_th_tumor = (default_th_unique_CC == default_th_CC_label).astype(default_th_unique_CC.dtype)
    center_of_mass = centroid(default_th_tumor)
    tumor_volume = default_th_tumor.sum() * voxel_volume
    tumor_diameter = approximate_diameter(tumor_volume)
    sphere = approximate_sphere(relevant_points_in_real_space,
                                relevant_points_in_voxel_space,
                                center_of_mass, tumor_diameter / 2,
                                affine_matrix)

    relevant_CC = np.where(min_th_unique_CC == min_th_CC_label, pred_labels, 0)
    relevant_th_values = np.unique(relevant_CC)
    relevant_th_values = relevant_th_values[relevant_th_values != 0]

    best_th_index = np.where(relevant_th_values == default_th)[0]
    best_dice = dice(default_th_tumor, sphere)

    improved = False
    for th_index in range(0, int(best_th_index))[::-1]:
        current_th_tumor = (relevant_CC == relevant_th_values[th_index]).astype(relevant_CC.dtype)
        current_th_tumor = pre_process_segmentation(current_th_tumor)
        current_dice = dice(current_th_tumor, sphere)
        diff = current_dice - best_dice
        if diff > 0:
            best_th_index = th_index
            best_dice = current_dice
            if diff >= eps:
                improved = True
        elif diff <= -eps:
            break
    if not improved:
        for th_index in range(int(best_th_index) + 1, relevant_th_values.size):
            current_th_tumor = (relevant_CC == relevant_th_values[th_index]).astype(relevant_CC.dtype)
            current_th_tumor = pre_process_segmentation(current_th_tumor)
            current_dice = dice(current_th_tumor, sphere)
            diff = current_dice - best_dice
            if diff > 0:
                best_th_index = th_index
                best_dice = current_dice
            elif diff <= -eps:
                break

    return min_th_CC_label, relevant_th_values[best_th_index], best_dice


def dice_with_sphere_for_tumor_label_and_th(label_th: Tuple[int, int], pred_labels: np.ndarray,
                                            min_th_unique_CC: np.ndarray, voxel_volume: float,
                                            affine_matrix: np.ndarray,
                                            pix_dims: Tuple[float, float, float]) -> Tuple[int, int, float]:

    nX, nY, nZ = pred_labels.shape
    min_p = affines.apply_affine(affine_matrix, (0, 0, 0))
    max_p = affines.apply_affine(affine_matrix, pred_labels.shape)
    relevant_points_in_real_space = np.vstack([np.repeat(np.arange(min_p[0], max_p[0], pix_dims[0]), nY * nZ),
                                               np.tile(np.repeat(np.arange(min_p[1], max_p[1], pix_dims[1]), nZ), nX),
                                               np.tile(np.arange(min_p[2], max_p[2], pix_dims[2]), nX * nY)]).T
    relevant_points_in_voxel_space = np.vstack([np.repeat(np.arange(0, nX), nY * nZ),
                                                np.tile(np.repeat(np.arange(0, nY), nZ), nX),
                                                np.tile(np.arange(0, nZ), nX * nY)]).T

    label, th = label_th

    tumor_labels = np.zeros_like(pred_labels)
    filter_slice = (min_th_unique_CC == label)
    tumor_labels[filter_slice] = pred_labels[filter_slice]

    current_th_tumor = (tumor_labels >= th).astype(pred_labels.dtype)

    if np.all(current_th_tumor == 0):
        return label, th, 0

    current_th_tumor = pre_process_segmentation(current_th_tumor, remove_small_obs=False)

    current_tumor_volume = current_th_tumor.sum() * voxel_volume
    current_tumor_diameter = approximate_diameter(current_tumor_volume)

    current_tumor_centroid = centroid(current_th_tumor)

    current_sphere = approximate_sphere(relevant_points_in_real_space,
                                        relevant_points_in_voxel_space,
                                        current_tumor_centroid, current_tumor_diameter / 2,
                                        affine_matrix)

    return label, th, dice(current_th_tumor, current_sphere)


def measures(gt_seg: np.ndarray, pred_seg: np.ndarray) -> Tuple[float, int, int, int, float, float, float]:
    gt_seg_unique_CC = get_connected_components(gt_seg)
    pred_seg_unique_CC = get_connected_components(pred_seg)
    stack = np.hstack([gt_seg_unique_CC.flatten().reshape([-1, 1]), pred_seg_unique_CC.flatten().reshape([-1, 1])])
    stack = np.unique(stack[~np.any(stack == 0, axis=1)], axis=0)
    gt_seg_TP = np.isin(gt_seg_unique_CC, stack[:, 0]).astype(gt_seg.dtype)
    pred_seg_TP = np.isin(pred_seg_unique_CC, stack[:, 1]).astype(pred_seg.dtype)

    TP = np.unique(gt_seg_unique_CC[gt_seg_TP.astype(np.bool)])
    TP = TP[TP != 0].size

    FN = np.unique(gt_seg_unique_CC)
    FN = FN[FN != 0].size - TP

    FP = np.unique(pred_seg_unique_CC)
    FP = FP[FP != 0].size - TP

    if TP + FP == 0:
        precision = 1
    else:
        precision = TP / (TP + FP)

    if TP + FN == 0:
        recall = 1
    else:
        recall = TP / (TP + FN)

    F1_score = 2 * precision * recall / (precision + recall)

    return dice(gt_seg_TP, pred_seg_TP), TP, FP, FN, precision, recall, F1_score


if __name__ == '__main__':
    data_path = '/cs/casmip/public/for_shalom/Tumor_segmentation/final_test_set_-_R2U_NET_standalone_as_pairwise_33_scans_-_GT_livers'
    gt_tumors = glob(f'{data_path}/*/Scan_Tumors.nii.gz')
    gt_tumors.sort()
    pred_label_tumors, livers, pred_default_th = [], [], []
    for gt_file in gt_tumors:
        pred_label_tumors.append(replace_in_file_name(gt_file, '/Scan_Tumors.nii.gz', '/Scan_Tumors_pred_label.nii.gz'))
        livers.append(replace_in_file_name(gt_file, '/Scan_Tumors.nii.gz', '/Scan_Liver.nii.gz'))
        pred_default_th.append(replace_in_file_name(gt_file, '/Scan_Tumors.nii.gz', '/Scan_Tumors_pred_th_8.nii.gz'))

    for i, (liver_file_name, tumors_GT_file_name, tumors_pred_label_file_name, tumors_pred_file_name)\
            in enumerate(zip(livers, gt_tumors, pred_label_tumors, pred_default_th)):

        print(f'start process {i+1}')

        t = time()

        # liver_file_name = '/cs/casmip/public/for_shalom/Tumor_segmentation/final_test_set_-_R2U_NET_standalone_as_pairwise_33_scans_-_GT_livers/A_Y_04_05_2020/Scan_Liver.nii.gz'
        # tumors_GT_file_name = '/cs/casmip/public/for_shalom/Tumor_segmentation/final_test_set_-_R2U_NET_standalone_as_pairwise_33_scans_-_GT_livers/A_Y_04_05_2020/Scan_Tumors.nii.gz'
        # # tumors_pred_file_name = '/cs/casmip/public/for_shalom/Tumor_segmentation/final_test_set_-_R2U_NET_standalone_as_pairwise_33_scans_-_GT_livers/A_Y_04_05_2020/Scan_Tumors_pred_th_8.nii.gz'
        # tumors_pred_label_file_name = '/cs/casmip/public/for_shalom/Tumor_segmentation/final_test_set_-_R2U_NET_standalone_as_pairwise_33_scans_-_GT_livers/A_Y_04_05_2020/Scan_Tumors_pred_label.nii.gz'

        liver_case, nifti_file = load_nifti_data(liver_file_name)
        tumors_gt_case, _ = load_nifti_data(tumors_GT_file_name)

        pix_dims = nifti_file.header.get_zooms()
        affine_matrix = nifti_file.affine

        # pre-process the tumors and liver segmentations
        tumors_gt_case = pre_process_segmentation(tumors_gt_case)
        liver_case = np.logical_or(liver_case, tumors_gt_case).astype(liver_case.dtype)
        liver_case = getLargestCC(liver_case)
        liver_case = pre_process_segmentation(liver_case, remove_small_obs=False)
        # tumors_gt_case = np.logical_and(liver_case, tumors_gt_case).astype(tumors_gt_case.dtype)

        tumors_pred_default_th, _ = load_nifti_data(tumors_pred_file_name)
        tumors_pred_labels, _ = load_nifti_data(tumors_pred_label_file_name)

        tumors_pred_default_th = tumors_pred_default_th * liver_case
        tumors_pred_default_th = pre_process_segmentation(tumors_pred_default_th)

        tumors_pred_labels = tumors_pred_labels * liver_case

        t1 = time()
        tumors_pred_adaptive_th = adaptive_threshold(tumors_pred_labels, affine_matrix, pix_dims, default_th=8,
                                                     n_processes=cpu_count() - 3)
        t1 = calculate_runtime(t1)

        adaptive_file_name = f'{os.path.dirname(liver_file_name)}/adaptive_th.nii.gz'
        nib.save(nib.Nifti1Image(tumors_pred_adaptive_th, affine_matrix), adaptive_file_name)

        # print(f'default th dice: {dice(tumors_gt_case, tumors_pred_default_th)}')
        # print(f'adaptive th dice: {dice(tumors_gt_case, tumors_pred_adaptive_th)}')

        print('\n-----------------------------------------------------------------')

        print('-----------------------------------------------------------------')

        print(f'making the adaptive th case: {t1}')
        print(f'overall time: {calculate_runtime(t)}')
        print('-----------------------------------------------------------------')
        dice1, TP1, FP1, FN1, precision1, recall1, F1_score1 = measures(tumors_gt_case, tumors_pred_default_th)
        dice2, TP2, FP2, FN2, precision2, recall2, F1_score2 = measures(tumors_gt_case, tumors_pred_adaptive_th)

        print('[measure_name]: [measure_of_default_th], [measure_of_adaptive_th]')
        print('-----------------------------------------------------------------')
        print(f'dice_over_TP: {dice1:.3f}, {dice2:.3f}')
        print(f'TP: {TP1}, {TP2}')
        print(f'FN: {FN1}, {FN2}')
        print(f'FP: {FP1}, {FP2}')
        print(f'precision: {precision1:.3f}, {precision2:.3f}')
        print(f'recall: {recall1:.3f}, {recall2:.3f}')
        print(f'F1 score: {F1_score1:.3f}, {F1_score2:.3f}')

        break

    # ----------------------------------------------------------------------------------------
    # tumors_pred_adaptive_th, _ = load_nifti_data('adaptive_th.nii.gz')
    #
    # dice1, TP1, FP1, FN1, precision1, recall1, F1_score1 = measures(tumors_gt_case, tumors_pred_default_th)
    # dice2, TP2, FP2, FN2, precision2, recall2, F1_score2 = measures(tumors_gt_case, tumors_pred_adaptive_th)
    #
    # print('[measure_name]: [measure_of_default_th], [measure_of_adaptive_th]')
    # print('-----------------------------------------------------------------')
    # print(f'dice_over_TP: {dice1:.3f}, {dice2:.3f}')
    # print(f'TP: {TP1}, {TP2}')
    # print(f'FN: {FN1}, {FN2}')
    # print(f'FP: {FP1}, {FP2}')
    # print(f'precision_edges: {precision1:.3f}, {precision2:.3f}')
    # print(f'recall_edges: {recall1:.3f}, {recall2:.3f}')
    # print(f'F1 score: {F1_score1:.3f}, {F1_score2:.3f}')

