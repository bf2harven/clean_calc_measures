from utils import *
from os.path import basename, dirname
from multiprocessing import Pool
from typing import Tuple, List, Callable, Optional, Union
import pandas as pd
from functools import partial
from notifications import notify
from tqdm.contrib.concurrent import process_map
# from data import test_Liver_gt
import matplotlib.pyplot as plt
from glob import glob
from matching_graphs import load_matching_graph
import networkx as nx
from datetime import date


# def get_list_of_dice_and_distance_from_border(file_names: Tuple[str, str, str], selem_radius: int = 1):
#     """
#
#     :param file_names: a tuple in the following form: (liver_file_name, tumors_gt_file_name, tumors_pred_file_name)
#     :param selem_radius: the size of the radius of the structural element used to extract the liver border.
#
#     :return: a list containing for each predicted tumor that intersect a GT tumor, a tuple in the following form:
#         (gt_tumor_diameter, dice, assd, hd, dist_from_border, case_name) where:
#         • gt_tumor_diameter is the diameter of the GT tumor the current predicted tumor intersects.
#         • dice is the Dice between the current predicted tumor and the GT tumor it intersects.
#         • assd is the Assd between the current predicted tumor and the GT tumor it intersects.
#         • hd is the Hausdorff Distance between the current predicted tumor and the GT tumor it intersects.
#         • dist_from_border is the minimum distance between the current predicted tumor and the GT liver's border.
#         • case_name is the name of the case.
#     """
#     (liver_gt_case, liver_gt_file), (tumors_gt_case, _), (tumors_pred_case, _) = (load_nifti_data(file_name) for
#                                                                                   file_name in
#                                                                                   file_names)
#
#     tumors_gt_case = np.logical_and(tumors_gt_case, liver_gt_case).astype(tumors_gt_case.dtype)
#
#     case_name = basename(dirname(file_names[0]))
#     pix_dims = liver_gt_file.header.get_zooms()
#     voxel_volume = pix_dims[0] * pix_dims[1] * pix_dims[2]
#
#     # extracting the liver border
#     liver_border = get_liver_border(liver_gt_case, selem_radius)
#
#     gt_tumors_labels = get_connected_components(tumors_gt_case)
#     pred_tumors_labels = get_connected_components(tumors_pred_case)
#
#     results = []
#     for gt_label in np.unique(gt_tumors_labels):
#         if gt_label == 0:
#             continue
#         current_gt_tumor = (gt_tumors_labels == gt_label).astype(tumors_gt_case.dtype)
#         current_gt_tumor_volume = current_gt_tumor.sum() * voxel_volume
#         current_gt_tumor_diameter = approximate_diameter(current_gt_tumor_volume)
#         pred_tumors_labels_intersect_current_gt_tumor = pred_tumors_labels * current_gt_tumor
#         for pred_label in np.unique(pred_tumors_labels_intersect_current_gt_tumor):
#             if pred_label == 0:
#                 continue
#             current_pred_tumor = (pred_tumors_labels == pred_label).astype(tumors_pred_case.dtype)
#             current_dice = dice(current_gt_tumor, current_pred_tumor)
#             # current_assd = None
#             current_assd = assd(current_pred_tumor, current_gt_tumor, voxelspacing=pix_dims, connectivity=2)
#             # current_hd = None
#             current_hd = hd(current_pred_tumor, current_gt_tumor, voxelspacing=pix_dims, connectivity=2)
#             current_dist_from_border = min_distance(current_gt_tumor, liver_border, voxelspacing=pix_dims,
#                                                     connectivity=2)
#             results.append((current_gt_tumor_diameter, current_dice, current_assd, current_hd, current_dist_from_border,
#                             case_name))
#
#     return results


def calculate_tumors_measures(file_names: Tuple[str, str, str], selem_radius: int = 1, tumor_dist_th: float = 5,
                              tumor_center_dist_th: float = 10) -> \
        Tuple[str, List[tuple]]:
    """
    Calculating measures for each tumor in a scan, given the CT scan and the liver and the tumors segmentations of the
    scan.

    :param file_names: a tuple in the following form: (ct_file_path, liver_file_path, tumors_file_path)
    :param selem_radius: the size of the radius of the structural element used to extract the liver border.
    :param tumor_dist_th: the threshold for the distance of the tumor from liver border.
    :param tumor_center_dist_th: the threshold for the distance of the center of mass of the tumor from liver border.

    :return: a tuple in the following form: (case_name, tumors_measures) where:
        • case_name is the name of the case.
        • tumors_measures is a list containing for each tumor a tuple in the following form:
            (tumor_volume, tumor_diameter, tumor_dist_from_border, tumor_center_of_mass_dist_from_border,
            is_close_to_border, touches_the_border, tumor_segment_locality, diff_from_liver_HU,
            dice_with_approximate_sphere) where:
                • tumor_volume is the volume of the current tumor in CC.
                • tumor_diameter is the diameter of the current tumor in mm.
                • tumor_dist_from_border is the minimum distance between the current tumor and the liver's border, in mm.
                • tumor_center_of_mass_dist_from_border is the minimum distance between the current tumor's center of
                    mass and the liver's border, in mm.
                • is_close_to_border is an indicator that indicates whether the current tumor's center of mass is in
                    'tumor_center_dist_th' mm from liver border and its border is in 'tumor_dist_th' mm from liver
                    border, or not.
                • touches_the_border is an indicator that indicates whether the current tumor's center of mass is in
                    'tumor_center_dist_th' mm from liver border and its border touches the liver border, or not.
                • tumor_segment_locality is either 1, 2 or 3 indicating the current tumor is in the upper_left liver
                    segment, upper_right liver segment or in the lower_left liver segment, respectively.
                • diff_from_liver_HU is the difference between the current tumor's mean of hounsfield units and the
                    liver's mean of hounsfield units.
                • dice_with_approximate_sphere is the dice between the current tumor and its approximated sphere.
    """

    # validating that the liver case and the tumors case have the same case name
    case_name = basename(dirname(file_names[0]))
    assert case_name == basename(dirname(file_names[1]))

    # loading the files
    (ct_case, _), (liver_case, liver_file), (tumors_case, _) = (load_nifti_data(file_name) for file_name in file_names)

    assert is_a_scan(ct_case)
    assert is_a_mask(liver_case)
    assert is_a_mask(tumors_case)

    # pre-process the tumors and liver segmentations
    tumors_case = pre_process_segmentation(tumors_case)
    liver_case = np.logical_or(liver_case, tumors_case).astype(liver_case.dtype)
    liver_case = getLargestCC(liver_case)
    liver_case = pre_process_segmentation(liver_case, remove_small_obs=False)
    tumors_case = np.logical_and(liver_case, tumors_case).astype(tumors_case.dtype)

    # create liver segments
    liver_segments = get_liver_segments(liver_case)

    pix_dims = liver_file.header.get_zooms()
    voxel_volume = pix_dims[0] * pix_dims[1] * pix_dims[2]

    # extracting the liver border
    liver_border = get_liver_border(liver_case, selem_radius)

    tumors_labels = get_connected_components(tumors_case)

    affine_matrix = liver_file.affine
    nX, nY, nZ = liver_case.shape
    min_p = affines.apply_affine(affine_matrix, (0, 0, 0))
    max_p = affines.apply_affine(affine_matrix, liver_case.shape)
    relevant_points_in_real_space = np.vstack([np.repeat(np.arange(min_p[0], max_p[0], pix_dims[0]), nY * nZ),
                                               np.tile(np.repeat(np.arange(min_p[1], max_p[1], pix_dims[1]), nZ), nX),
                                               np.tile(np.arange(min_p[2], max_p[2], pix_dims[2]), nX * nY)]).T
    relevant_points_in_voxel_space = np.vstack([np.repeat(np.arange(0, nX), nY * nZ),
                                               np.tile(np.repeat(np.arange(0, nY), nZ), nX),
                                               np.tile(np.arange(0, nZ), nX * nY)]).T

    results = []

    # running over each tumor
    for label in np.unique(tumors_labels):

        if label == 0:
            continue

        tumor = (tumors_labels == label).astype(tumors_case.dtype)
        tumor_volume = tumor.sum() * voxel_volume

        tumor_diameter = approximate_diameter(tumor_volume)

        tumor_dist_from_border = min_distance(tumor, liver_border, voxelspacing=pix_dims,
                                              connectivity=2)

        tumor_center_of_mass = get_center_of_mass(tumor)
        center_of_mass_case = np.zeros_like(tumors_case)
        center_of_mass_case[tumor_center_of_mass] = 1
        tumor_center_of_mass_dist_from_border = min_distance(center_of_mass_case, liver_border,
                                                             voxelspacing=pix_dims, connectivity=2)

        is_close_to_border = (tumor_center_of_mass_dist_from_border <= tumor_center_dist_th) and (tumor_dist_from_border <= tumor_dist_th)

        touches_the_border = (tumor_center_of_mass_dist_from_border <= tumor_center_dist_th) and (tumor_dist_from_border == 0)

        tumor_segment_locality = liver_segments[tumor_center_of_mass]

        tumor_diff_from_liver_HU = ct_case[tumor == 1].mean() - ct_case[liver_case == 1].mean()

        dice_with_approximate_sphere = dice(tumor, approximate_sphere(relevant_points_in_real_space,
                                                                      relevant_points_in_voxel_space,
                                                                      tumor_center_of_mass, tumor_diameter / 2,
                                                                      affine_matrix))

        results.append((tumor_volume / 1000, tumor_diameter, tumor_dist_from_border,
                        tumor_center_of_mass_dist_from_border, is_close_to_border, touches_the_border,
                        tumor_segment_locality, tumor_diff_from_liver_HU, dice_with_approximate_sphere))

    return case_name, results


def calculate_change_in_pairs(file_names: Tuple[str, str]):
    """
    Calculating changes in the tumors between a pair of scans of the same patient.

    :param file_names: a tuple in the following form: (BL_tumors_file_name, FU_tumors_file_name).

    :return: a tuple in the following form: (BL_case_name, FU_case_name, BL_n_tumors, FU_n_tumors,
        BL_total_tumors_volume, FU_total_tumors_volume, volume_diff, abs_volume_diff, volume_change, n_unique,
        n_disappear_BL, n_split_BL, n_merge_BL, n_complex_BL, n_new_FU, n_split_FU, n_merge_FU, n_complex_FU,
        n_complex_event, contains_merge, contains_split, contains_merge_|_split, contains_merge_&_split,
        contains_complex) where:
        • BL_case_name is the name of the baseline case.
        • FU_case_name is the name of the followup case.
        • BL_n_tumors is the number of tumors in the baseline case.
        • FU_n_tumors is the number of tumors in the followup case.
        • BL_total_tumors_volume is the total tumors volume of the baseline case in CC.
        • FU_total_tumors_volume is the total tumors volume of the followup case in CC.
        • volume_diff is the difference between the total tumors volume (FU_total_tumors_volume - BL_total_tumors_volume).
        • abs_volume_diff is the absolute value os the difference between the total tumors volume
            (abs(FU_total_tumors_volume - BL_total_tumors_volume)).
        • volume_change is the change in the total tumors volume between the BL and FU, in percentage
            ((FU_total_tumors_volume - BL_total_tumors_volume)/(FU_total_tumors_volume + BL_total_tumors_volume)).
        • n_unique is the number of tumors in FU that exist in BL (same as the number of tumors in BL that exist in FU).
            Namely, the number of pairs of tumors in the bipartite graph of the tumors in the cases, that share an edge
            with each other, but with nobody else.
        • n_disappear_BL is the number of disappeared tumors in BL that doesn't exist in FU (the "Disappeared-Tumors"
            definition includes only brand-disappear-tumors that aren't participating in a merge). Namely, the number of
            tumors in BL that don't share an edge with nobody in the bipartite graph of the tumors in the cases.
        • n_split_BL is the number of tumors in BL that are split in the FU. Namely, they share an edge with more then 1
            tumor in FU in the bipartite graph of the tumors in the cases.
        • n_merge_BL is the number of tumors in BL that are are participating in a merge. Namely, they share an edge
            with a tumor in FU, in the bipartite graph of the tumors in the cases, but they are not the only in BL that
            share with it.
        • n_complex_BL is the number of tumors in BL participating in a complex event. In the bipartite graph of the
            tumors in the cases, a complex event is a connected components that contains a BL/FU tumor, i, that shares
            an edge with a FU/BL tumor, j, so that j shares an edge with another BL/FU tumor that is not i. A tumor
            participates in a complex event if it is in such a connected component.
        • n_new_FU is the number of new tumors in FU that doesn't exist in BL (the "New-Tumors" definition includes
             only brand-new-tumors that aren't participating in a split). Namely, the number of tumors in FU that don't share
             an edge with nobody in the bipartite graph of the tumors in the cases.
        • n_split_FU is the number of tumors in FU that are a result of a split of a BL tumor. Namely, they share an
            edge with a tumor in BL, in the bipartite graph of the tumors in the cases, but they are not the only in FU
            that share with it.
        • n_merge_FU is the number of tumors in FU that are a resold of a merge of more then one tumor in BL. Namely,
            they share an edge with more then 1 tumor in BL in the bipartite graph of the tumors in the cases.
        • n_complex_FU is the number of tumors in FU participating in a complex event (see description of n_complex_BL).
        • n_complex_event is the number of complex events in the bipartite graph of the tumors in the cases (connected
            components with the definition as in the description of n_complex_BL).
        • contains_merge is an indicator that equals 1 iff there is a "merge" between the BL and FU.
        • contains_split is an indicator that equals 1 iff there is a "split" between the BL and FU.
        • contains_merge_|_split is an indicator that equals 1 iff there is either a "merge" or a "split" between the BL
            and FU.
        • contains_merge_&_split is an indicator that equals 1 iff there is both a "merge" and a "split" between the BL
            and FU.
        • contains_complex is an indicator that equals 1 iff the current pair has a complex event.
    """

    get_case_name = lambda file_name: basename(dirname(file_name))
    get_patient_name = lambda file_name: '_'.join(c for c in get_case_name(file_name).split('_') if not c.isdigit())

    assert get_patient_name(file_names[0]) == get_patient_name(file_names[1])
    BL_case_name = get_case_name(file_names[0])
    FU_case_name = get_case_name(file_names[1])
    # assert BL_case_name != FU_case_name

    # loading the files
    (BL_tumors_case, BL_tumors_file), (FU_tumors_case, FU_tumors_file) = (load_nifti_data(file_name) for file_name in
                                                                          file_names)

    # BL_tumors_case = (BL_tumors_case == 2).astype(BL_tumors_case.dtype)
    # FU_tumors_case = (FU_tumors_case == 2).astype(FU_tumors_case.dtype)

    BL_pix_dims = BL_tumors_file.header.get_zooms()
    BL_voxel_volume = BL_pix_dims[0] * BL_pix_dims[1] * BL_pix_dims[2]
    FU_pix_dims = FU_tumors_file.header.get_zooms()
    FU_voxel_volume = FU_pix_dims[0] * FU_pix_dims[1] * FU_pix_dims[2]

    # pre-process the tumors segmentation
    BL_tumors_case = pre_process_segmentation(BL_tumors_case)
    FU_tumors_case = pre_process_segmentation(FU_tumors_case)

    BL_tumors_labels = get_connected_components(BL_tumors_case)
    FU_tumors_labels = get_connected_components(FU_tumors_case)

    # calculating the tumors volume change
    BL_total_tumors_volume = (BL_tumors_labels != 0).sum() * BL_voxel_volume / 1000
    FU_total_tumors_volume = (FU_tumors_labels != 0).sum() * FU_voxel_volume / 1000
    volume_diff = FU_total_tumors_volume - BL_total_tumors_volume
    abs_volume_diff = abs(volume_diff)
    try:
        volume_change = 100 * float(volume_diff) / float(BL_total_tumors_volume + FU_total_tumors_volume)
    except:
        volume_change = np.nan

    # extracting the matches between the tumors in the cases
    matches = match_2_cases(BL_tumors_labels, FU_tumors_labels)

    # extracting the BL labels
    BL_labels = np.unique(BL_tumors_labels)
    BL_labels = BL_labels[BL_labels != 0]
    BL_n_tumors = BL_labels.size

    # extracting the FU labels
    FU_labels = np.unique(FU_tumors_labels)
    FU_labels = FU_labels[FU_labels != 0] + BL_n_tumors
    FU_n_tumors = FU_labels.size

    V = list(BL_labels - 1) + list(FU_labels - 1)
    visited = [False] * len(V)
    adjacency_lists = []
    for _ in range(BL_n_tumors + FU_n_tumors):
        adjacency_lists.append([])
    for (bl_v, fu_v) in matches:
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

    def n_bl_n_fu(CC):
        n_bl_in_CC = 0
        n_fu_in_CC = 0
        for v in CC:
            if is_bl_tumor(v):
                n_bl_in_CC += 1
            else:
                n_fu_in_CC += 1
        return n_bl_in_CC, n_fu_in_CC


    (n_unique, n_disappear_BL, n_split_BL, n_merge_BL, n_complex_BL, n_new_FU, n_split_FU, n_merge_FU, n_complex_FU,
     n_complex_event, contains_merge, contains_split, contains_merge_or_split, contains_merge_and_split, contains_complex) = (0,) * 15

    while len(V) > 0:
        v = V[0]
        current_CC = DFS(v)

        # in case the current tumor is a single tumor in a connected component
        if len(current_CC) == 1:
            if is_bl_tumor(v):
                n_disappear_BL += 1
            else:
                n_new_FU += 1

        # in case the current tumor is in a connected component with only two tumors, namely it is a unique case
        elif len(current_CC) == 2:
            n_unique += 1

        else:
            n_bl, n_fu = n_bl_n_fu(current_CC)

            # in case of a complex event
            if n_bl > 1 and n_fu > 1:
                contains_complex = 1
                n_complex_event += 1
                n_complex_BL += n_bl
                n_complex_FU += n_fu
            else:
                # in case of a split event
                if n_bl == 1:
                    contains_split = 1
                    contains_merge_or_split = 1
                    if contains_merge:
                        contains_merge_and_split = 1
                    n_split_BL += 1
                    n_split_FU += n_fu

                # in case of a merge event
                else:
                    contains_merge = 1
                    contains_merge_or_split = 1
                    if contains_split:
                        contains_merge_and_split = 1
                    n_merge_BL += n_bl
                    n_merge_FU += 1

    if not contains_complex:
        n_complex_FU = np.nan
        n_complex_BL = np.nan
        n_complex_event = np.nan
    if not contains_merge:
        n_merge_FU = np.nan
        n_merge_BL = np.nan
    if not contains_split:
        n_split_FU = np.nan
        n_split_BL = np.nan

    #
    #
    #
    #
    #
    # ## ???????????????????????????????????????????????????????????????????????????????????????????????????
    #
    # # def get_and_remove_item_from_matches(item):
    # #     if item in matches:
    # #         matches.remove(item)
    # #     return item
    # #
    # # def get_number_of_neighbors(v, type_of_v, all_matches):
    # #     assert type_of_v == 'BL' or type_of_v == 'FU'
    # #     if type_of_v == 'BL':
    # #         return sum(1 for m in all_matches if m[0] == v)
    # #     return sum(1 for m in all_matches if m[1] == v)
    # #
    # # # def get_all_edges_in_connected_components()
    #
    #
    #
    # original_matches = matches.copy()
    # while len(matches) > 0:
    #     current_match = matches.pop(0)
    #
    #     # updating that we found an exist tumor
    #     # n_exist += 1
    #
    #     # getting the current match's BL-neighbors (namely, other matches that share the same BL tumor)
    #     current_BL_neighbors = [get_and_remove_item_from_matches(m) for m in original_matches if
    #                             m[0] == current_match[0] and m[1] != current_match[1]]
    #
    #     # getting the current match's FU-neighbors (namely, other matches that share the same FU tumor)
    #     current_FU_neighbors = [get_and_remove_item_from_matches(m) for m in original_matches if
    #                             m[1] == current_match[1] and m[0] != current_match[0]]
    #
    #     # in case the current pair is complicated (see the description of the function)
    #     if len(current_BL_neighbors) > 0 and len(current_FU_neighbors) > 0:
    #         return (
    #         BL_case_name, FU_case_name, BL_n_tumors, FU_n_tumors, BL_total_tumors_volume, FU_total_tumors_volume,
    #         volume_diff, abs_volume_diff, volume_change, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
    #         np.nan, np.nan, np.nan, np.nan, 1)
    #
    #     # in case of a split
    #     if len(current_BL_neighbors) > 0:
    #         n_splits += 1
    #         n_split_new += len(current_BL_neighbors)
    #
    #     # in case of a merge
    #     elif len(current_FU_neighbors) > 0:
    #         n_merges += 1
    #         n_merge_disappear += len(current_FU_neighbors)
    #
    # n_new = len([t for t in FU_labels if t not in (m[1] for m in original_matches)])
    #
    # n_disappear = len([t for t in BL_labels if t not in (m[0] for m in original_matches)])
    #
    # if n_merges > 0:
    #     contains_merge = 1
    # else:
    #     n_merges = np.nan
    #     n_merge_disappear = np.nan
    #
    # if n_splits > 0:
    #     contains_split = 1
    # else:
    #     n_splits = np.nan
    #     n_split_new = np.nan
    #
    # contains_merge_and_split = contains_split * contains_merge
    # if (contains_merge == 1) or (contains_split == 1):
    #     contains_merge_or_split = 1

    return (BL_case_name, FU_case_name, BL_n_tumors, FU_n_tumors, BL_total_tumors_volume, FU_total_tumors_volume,
            volume_diff, abs_volume_diff, volume_change, n_unique, n_disappear_BL, n_split_BL, n_merge_BL, n_complex_BL,
            n_new_FU, n_split_FU, n_merge_FU, n_complex_FU, n_complex_event, contains_merge, contains_split,
            contains_merge_or_split, contains_merge_and_split, contains_complex)


def calculate_diff_in_matching_according_to_lesion_type_v1(file_names: Tuple[str, str]):
    """
    Calculating the difference in the matching between GT and Pred according to the lesion type (new, disappear, unique,
    merge, split and complex).

    :param file_names: a tuple in the following form: (GT_matching_graph_json_file_name, Pred_matching_graph_json_file_name).

    :return: a tuple in the following form: (case_name, BL_n_tumors, FU_n_tumors, n_edges, n_CCs, n_unique, n_disappear_BL, n_split_BL,
        n_merge_BL, n_complex_BL, n_new_FU, n_split_FU, n_merge_FU, n_complex_FU, n_complex_event, contains_merge,
        contains_split, contains_merge_|_split, contains_merge_&_split, contains_complex) where:
        • case_name is the name of the current pair of images.
        • BL_n_tumors is the number of tumors in the baseline case.
        • FU_n_tumors is the number of tumors in the followup case.
        • n_edges is the number of edges in the matching graph.
        • n_CCs is the number of connected components in the matching graph.
        • n_unique is the number of tumors in FU that exist in BL (same as the number of tumors in BL that exist in FU).
            Namely, the number of pairs of tumors in the bipartite graph of the tumors in the cases, that share an edge
            with each other, but with nobody else.
        • n_disappear_BL is the number of disappeared tumors in BL that doesn't exist in FU (the "Disappeared-Tumors"
            definition includes only brand-disappear-tumors that aren't participating in a merge). Namely, the number of
            tumors in BL that don't share an edge with nobody in the bipartite graph of the tumors in the cases.
        • n_split_BL is the number of tumors in BL that are split in the FU. Namely, they share an edge with more then 1
            tumor in FU in the bipartite graph of the tumors in the cases.
        • n_merge_BL is the number of tumors in BL that are are participating in a merge. Namely, they share an edge
            with a tumor in FU, in the bipartite graph of the tumors in the cases, but they are not the only in BL that
            share with it.
        • n_complex_BL is the number of tumors in BL participating in a complex event. In the bipartite graph of the
            tumors in the cases, a complex event is a connected components that contains a BL/FU tumor, i, that shares
            an edge with a FU/BL tumor, j, so that j shares an edge with another BL/FU tumor that is not i. A tumor
            participates in a complex event if it is in such a connected component.
        • n_new_FU is the number of new tumors in FU that doesn't exist in BL (the "New-Tumors" definition includes
             only brand-new-tumors that aren't participating in a split). Namely, the number of tumors in FU that don't share
             an edge with nobody in the bipartite graph of the tumors in the cases.
        • n_split_FU is the number of tumors in FU that are a result of a split of a BL tumor. Namely, they share an
            edge with a tumor in BL, in the bipartite graph of the tumors in the cases, but they are not the only in FU
            that share with it.
        • n_merge_FU is the number of tumors in FU that are a resold of a merge of more then one tumor in BL. Namely,
            they share an edge with more then 1 tumor in BL in the bipartite graph of the tumors in the cases.
        • n_complex_FU is the number of tumors in FU participating in a complex event (see description of n_complex_BL).
        • n_complex_event is the number of complex events in the bipartite graph of the tumors in the cases (connected
            components with the definition as in the description of n_complex_BL).
        • contains_merge is an indicator that equals 1 iff there is a "merge" between the BL and FU.
        • contains_split is an indicator that equals 1 iff there is a "split" between the BL and FU.
        • contains_merge_|_split is an indicator that equals 1 iff there is either a "merge" or a "split" between the BL
            and FU.
        • contains_merge_&_split is an indicator that equals 1 iff there is both a "merge" and a "split" between the BL
            and FU.
        • contains_complex is an indicator that equals 1 iff the current pair has a complex event.
    """

    GT_matching_graph_json_file_name, Pred_matching_graph_json_file_name = file_names
    BL_n_tumors, FU_n_tumors, GT_matches, case_name, _, _, _, _, _, _ = load_matching_graph(GT_matching_graph_json_file_name)
    _, _,pred_matches, pred_case_name, _, _, _, _, _, _ = load_matching_graph(Pred_matching_graph_json_file_name)
    assert case_name == pred_case_name

    GT_matches = [m[1] for m in pred_matches]

    n_edges = len(GT_matches)

    # extracting the BL labels
    BL_labels = np.arange(BL_n_tumors) + 1

    # extracting the FU labels
    FU_labels = np.arange(FU_n_tumors) + 1 + BL_n_tumors

    V = list(BL_labels - 1) + list(FU_labels - 1)
    visited = [False] * len(V)
    adjacency_lists = []
    for _ in range(BL_n_tumors + FU_n_tumors):
        adjacency_lists.append([])
    for (bl_v, fu_v) in GT_matches:
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

    def n_bl_n_fu(CC):
        n_bl_in_CC = 0
        n_fu_in_CC = 0
        for v in CC:
            if is_bl_tumor(v):
                n_bl_in_CC += 1
            else:
                n_fu_in_CC += 1
        return n_bl_in_CC, n_fu_in_CC


    (n_unique, n_disappear_BL, n_split_BL, n_merge_BL, n_complex_BL, n_new_FU, n_split_FU, n_merge_FU, n_complex_FU,
     n_complex_event, contains_merge, contains_split, contains_merge_or_split, contains_merge_and_split, contains_complex) = (0,) * 15

    n_CCs = 0
    while len(V) > 0:
        n_CCs += 1
        v = V[0]
        current_CC = DFS(v)

        # in case the current tumor is a single tumor in a connected component
        if len(current_CC) == 1:
            if is_bl_tumor(v):
                n_disappear_BL += 1
            else:
                n_new_FU += 1

        # in case the current tumor is in a connected component with only two tumors, namely it is a unique case
        elif len(current_CC) == 2:
            n_unique += 1

        else:
            n_bl, n_fu = n_bl_n_fu(current_CC)

            # in case of a complex event
            if n_bl > 1 and n_fu > 1:
                contains_complex = 1
                n_complex_event += 1
                n_complex_BL += n_bl
                n_complex_FU += n_fu
            else:
                # in case of a split event
                if n_bl == 1:
                    contains_split = 1
                    contains_merge_or_split = 1
                    if contains_merge:
                        contains_merge_and_split = 1
                    n_split_BL += 1
                    n_split_FU += n_fu

                # in case of a merge event
                else:
                    contains_merge = 1
                    contains_merge_or_split = 1
                    if contains_split:
                        contains_merge_and_split = 1
                    n_merge_BL += n_bl
                    n_merge_FU += 1

    if not contains_complex:
        n_complex_FU = np.nan
        n_complex_BL = np.nan
        n_complex_event = np.nan
    if not contains_merge:
        n_merge_FU = np.nan
        n_merge_BL = np.nan
    if not contains_split:
        n_split_FU = np.nan
        n_split_BL = np.nan

    return (case_name, BL_n_tumors, FU_n_tumors, n_edges, n_CCs, n_unique, n_disappear_BL, n_split_BL, n_merge_BL,
            n_complex_BL, n_new_FU, n_split_FU, n_merge_FU, n_complex_FU, n_complex_event, contains_merge,
            contains_split, contains_merge_or_split, contains_merge_and_split, contains_complex)


def calculate_diff_in_matching_according_to_lesion_type_v2(file_names: Tuple[str, str], max_dilation_for_pred=9):
    """
    Calculating the difference in the matching between GT and Pred according to the lesion type (new, disappear, unique,
    merge, split and complex).

    :param file_names: a tuple in the following form: (GT_matching_graph_json_file_name, Pred_matching_graph_json_file_name).

    :return: a tuple of 3 tuples in the following form: (GT_CC_types, PRED_CC_types, Delta_CC_types, Delta_lesion_type,
        Delta_edges, bl_lesion_type_conf_mat, fu_lesion_type_conf_mat, all_lesion_type_conf_mat, bl_diameters,
        fu_diameters, bl_organ_volume, fu_organ_volume), where:

        • GT/PRED_CC_types are tuples in the following form: (case_name, BL_n_tumors, FU_n_tumors, n_edges, n_CCs,
            n_unique, n_disappear_BL, n_split_BL, n_merge_BL, n_complex_BL, n_new_FU, n_split_FU, n_merge_FU,
            n_complex_FU, n_complex_event, contains_merge, contains_split, contains_merge_|_split,
            contains_merge_&_split, contains_complex) where:
            • case_name is the name of the current pair of images.
            • BL_n_tumors is the number of tumors in the baseline case.
            • FU_n_tumors is the number of tumors in the followup case.
            • n_edges is the number of edges in the matching graph.
            • n_CCs is the number of connected components in the matching graph.
            • n_unique is the number of tumors in FU that exist in BL (same as the number of tumors in BL that exist in
                FU). Namely, the number of pairs of tumors in the bipartite graph of the tumors in the cases, that share
                an edge with each other, but with nobody else.
            • n_disappear_BL is the number of disappeared tumors in BL that doesn't exist in FU (the "Disappeared-Tumors"
                definition includes only brand-disappear-tumors that aren't participating in a merge). Namely, the number of
                tumors in BL that don't share an edge with nobody in the bipartite graph of the tumors in the cases.
            • n_split_BL is the number of tumors in BL that are split in the FU. Namely, they share an edge with more then 1
                tumor in FU in the bipartite graph of the tumors in the cases.
            • n_merge_BL is the number of tumors in BL that are are participating in a merge. Namely, they share an edge
                with a tumor in FU, in the bipartite graph of the tumors in the cases, but they are not the only in BL that
                share with it.
            • n_complex_BL is the number of tumors in BL participating in a complex event. In the bipartite graph of the
                tumors in the cases, a complex event is a connected components that contains a BL/FU tumor, i, that shares
                an edge with a FU/BL tumor, j, so that j shares an edge with another BL/FU tumor that is not i. A tumor
                participates in a complex event if it is in such a connected component.
            • n_new_FU is the number of new tumors in FU that doesn't exist in BL (the "New-Tumors" definition includes
                 only brand-new-tumors that aren't participating in a split). Namely, the number of tumors in FU that don't share
                 an edge with nobody in the bipartite graph of the tumors in the cases.
            • n_split_FU is the number of tumors in FU that are a result of a split of a BL tumor. Namely, they share an
                edge with a tumor in BL, in the bipartite graph of the tumors in the cases, but they are not the only in FU
                that share with it.
            • n_merge_FU is the number of tumors in FU that are a resold of a merge of more then one tumor in BL. Namely,
                they share an edge with more then 1 tumor in BL in the bipartite graph of the tumors in the cases.
            • n_complex_FU is the number of tumors in FU participating in a complex event (see description of n_complex_BL).
            • n_complex_event is the number of complex events in the bipartite graph of the tumors in the cases (connected
                components with the definition as in the description of n_complex_BL).
            • contains_merge is an indicator that equals 1 iff there is a "merge" between the BL and FU.
            • contains_split is an indicator that equals 1 iff there is a "split" between the BL and FU.
            • contains_merge_|_split is an indicator that equals 1 iff there is either a "merge" or a "split" between the BL
                and FU.
            • contains_merge_&_split is an indicator that equals 1 iff there is both a "merge" and a "split" between the BL
                and FU.
            • contains_complex is an indicator that equals 1 iff the current pair has a complex event.

        • Delta_CC_types is a tuple in the following form: (case_name, BL_n_tumors, FU_n_tumors, delta_n_edges,
            delta_n_CCs, n_equivalent_CCs, Precision_of_CCs, Recall_of_CCs, n_equivalent_unique, n_equivalent_disappear, n_equivalent_new, n_equivalent_split,
            n_equivalent_merge, n_equivalent_complex) where:
            • case_name is the name of the current pair of images.
            • BL_n_tumors is the number of tumors in the baseline case.
            • FU_n_tumors is the number of tumors in the followup case.
            • delta_n_edges is the delta between the number of edges in the GT matching graph and the PRED matching graph.
            • delta_n_CCs is the delta between the number of connected components the GT matching graph and the PRED matching graph.
            • n_equivalent_CCs is the number of equivalent connected components in both, GT and PRED, matching graphs.
            • Precision_of_CCs is the percentage of CCs in the PRED matching graph that are equivalent to those in the GT matching graph.
            • Recall_of_CCs is the percentage of CCs in the GT matching graph that are equivalent to those in the PRED matching graph.
            • n_equivalent_unique is the number of equivalent connected components of 'unique' type in both, GT and PRED, matching graphs.
            • n_equivalent_disappear is the number of equivalent connected components of 'disappear' type in both, GT and PRED, matching graphs.
            • n_equivalent_new is the number of equivalent connected components of 'new' type in both, GT and PRED, matching graphs.
            • n_equivalent_split is the number of equivalent connected components of 'split' type in both, GT and PRED, matching graphs.
            • n_equivalent_merge is the number of equivalent connected components of 'merge' type in both, GT and PRED, matching graphs.
            • n_equivalent_complex is the number of equivalent connected components of 'complex' type in both, GT and PRED, matching graphs.

        • Delta_lesion_type is a tuple in the following form: (case_name, BL_n_tumors, FU_n_tumors, n_right_BL_labels,
            n_right_FU_labels, n_right_BL_unique_labels, n_right_FU_unique_labels, n_right_BL_disappeared_labels,
            n_right_FU_new_labels, n_right_BL_split_labels, n_right_FU_split_labels, n_right_BL_merge_labels,
            n_right_FU_merge_labels, n_right_BL_complex_labels, n_right_FU_complex_labels) where:
            • case_name is the name of the current pair of images.
            • BL_n_tumors is the number of tumors in the baseline case.
            • FU_n_tumors is the number of tumors in the followup case.
            • n_right_BL_labels is the number of tumors in BL that are labeled the same in GT and in PRED.
            • n_right_FU_labels is the number of tumors in FU that are labeled the same in GT and in PRED.
            • n_right_BL_unique_labels is the number of tumors in BL that are labeled as 'unique' in GT and in PRED.
            • n_right_FU_unique_labels is the number of tumors in FU that are labeled as 'unique' in GT and in PRED.
            • n_right_BL_disappeared_labels is the number of tumors in BL that are labeled as 'disappeared' in GT and in PRED.
            • n_right_FU_new_labels is the number of tumors in FU that are labeled as 'new' in GT and in PRED.
            • n_right_BL_split_labels is the number of tumors in BL that are labeled as 'split' in GT and in PRED.
            • n_right_FU_split_labels is the number of tumors in FU that are labeled as 'split' in GT and in PRED.
            • n_right_BL_merge_labels is the number of tumors in BL that are labeled as 'merge' in GT and in PRED.
            • n_right_FU_merge_labels is the number of tumors in FU that are labeled as 'merge' in GT and in PRED.
            • n_right_BL_complex_labels is the number of tumors in BL that are labeled as 'complex' in GT and in PRED.
            • n_right_FU_complex_labels is the number of tumors in FU that are labeled as 'complex' in GT and in PRED.

        • Delta_edges is a tuple in the following form: (case_name, BL_n_tumors, FU_n_tumors, n_edges_GT, n_edges_PRED,
            n_TP, n_FP, n_FN, Precision, Recall, F1-Score) where:
            • case_name is the name of the current pair of images.
            • BL_n_tumors is the number of tumors in the baseline case.
            • FU_n_tumors is the number of tumors in the followup case.
            • n_edges_GT is the number of edges in the GT matching graph.
            • n_edges_PRED is the number of edges in the PRED matching graph.
            • n_TP is the number of TP edges between the GT and PRED graphs.
            • n_FP is the number of FP edges between the GT and PRED graphs.
            • n_FN is the number of FN edges between the GT and PRED graphs.
            • Precision is the precision of the edges between the GT and PRED graphs.
            • Recall is the recall of the edges between the GT and PRED graphs.
            • F1-Score is the F1-score of the edges between the GT and PRED graphs.

        • bl_lesion_type_conf_mat is a confusion matrix of the BL tumors types, where the labels order is as follows:
            ['Unique', 'Disappeared', 'New', 'Split', 'Merge', 'Complex']

        • fu_lesion_type_conf_mat is a confusion matrix of the FU tumors types, where the labels order is as follows:
            ['Unique', 'Disappeared', 'New', 'Split', 'Merge', 'Complex']

        • all_lesion_type_conf_mat is a confusion matrix of all the tumors types, where the labels order is as follows:
            ['Unique', 'Disappeared', 'New', 'Split', 'Merge', 'Complex']

        • bl_diameters is a list containing for each BL lesions its diameter in mm.

        • fu_diameters is a list containing for each FU lesions its diameter in mm.

        • bl_organ_volume is the volume of the BL organ in CC.

        • fu_organ_volume is the volume of the FU organ in CC.
    """

    def get_CC_types_for_graph(graph):

        def n_bl_n_fu(CC):
            n_bl, n_fu = 0, 0
            for node in CC.nodes:
                if node.endswith('bl'):
                    n_bl += 1
                else:
                    n_fu += 1
            return n_bl, n_fu

        n_edges = len(graph.edges)

        # part the graph into connected components
        CCs = list(nx.connected_component_subgraphs(graph))
        n_CCs = len(CCs)

        dict_CC_and_type = {}
        dict_lesion_and_type = {}

        (n_unique, n_disappeared_BL, n_split_BL, n_merge_BL, n_complex_BL, n_new_FU, n_split_FU, n_merge_FU, n_complex_FU,
         n_complex_event, contains_merge, contains_split, contains_merge_or_split, contains_merge_and_split,
         contains_complex) = (0,) * 15

        for CC in CCs:
            n_bl, n_fu = n_bl_n_fu(CC)

            # in case the current CC is a 'disappear' CC
            if (n_bl == 1) and (n_fu == 0):
                n_disappeared_BL += 1
                CC_type = 'disappeared'

            # in case the current CC is a 'new' CC
            elif (n_bl == 0) and (n_fu == 1):
                n_new_FU += 1
                CC_type = 'new'

            # in case the current CC is a 'unique' CC
            elif (n_bl == 1) and (n_fu == 1):
                n_unique += 1
                CC_type = 'unique'

            # in case the current CC is a 'complex' CC
            elif (n_bl > 1) and (n_fu > 1):
                contains_complex = 1
                n_complex_event += 1
                n_complex_BL += n_bl
                n_complex_FU += n_fu
                CC_type = 'complex'

            # in case the current CC is a 'split' CC
            elif (n_bl == 1) and (n_fu > 1):
                contains_split = 1
                contains_merge_or_split = 1
                if contains_merge:
                    contains_merge_and_split = 1
                n_split_BL += 1
                n_split_FU += n_fu
                CC_type = 'split'

            # in case the current CC is a 'merge' CC
            else:
                contains_merge = 1
                contains_merge_or_split = 1
                if contains_split:
                    contains_merge_and_split = 1
                n_merge_BL += n_bl
                n_merge_FU += 1
                CC_type = 'merge'

            dict_CC_and_type[CC] = CC_type
            for tumor_name in CC.nodes:
                dict_lesion_and_type[tumor_name] = CC_type

        if not contains_complex:
            n_complex_FU = np.nan
            n_complex_BL = np.nan
            n_complex_event = np.nan
        if not contains_merge:
            n_merge_FU = np.nan
            n_merge_BL = np.nan
        if not contains_split:
            n_split_FU = np.nan
            n_split_BL = np.nan

        return dict_CC_and_type, dict_lesion_and_type, (n_edges, n_CCs, n_unique, n_disappeared_BL, n_split_BL,
                                                        n_merge_BL, n_complex_BL, n_new_FU, n_split_FU, n_merge_FU,
                                                        n_complex_FU, n_complex_event, contains_merge, contains_split,
                                                        contains_merge_or_split, contains_merge_and_split,
                                                        contains_complex)

    GT_matching_graph_json_file_name, Pred_matching_graph_json_file_name = file_names
    BL_n_tumors, FU_n_tumors, GT_matches, case_name, _, _, bl_diameters, fu_diameters, bl_organ_volume, fu_organ_volume = load_matching_graph(GT_matching_graph_json_file_name)
    _, _,pred_matches, pred_case_name, _, _, _, _, _, _ = load_matching_graph(Pred_matching_graph_json_file_name)
    assert case_name == pred_case_name

    pred_matches = [m[1] for m in pred_matches if m[0] <= max_dilation_for_pred]

    bl_tumors = [f'{t}_bl' for t in range(1, BL_n_tumors + 1)]
    fu_tumors = [f'{t}_fu' for t in range(1, FU_n_tumors + 1)]

    gt_edges = [(f'{int(e[0])}_bl', f'{int(e[1])}_fu') for e in GT_matches]
    pred_edges = [(f'{int(e[0])}_bl', f'{int(e[1])}_fu') for e in pred_matches]

    # build the GT graph
    gt_G = nx.Graph()
    gt_G.add_nodes_from(bl_tumors, bipartite='bl')
    gt_G.add_nodes_from(fu_tumors, bipartite='fu')
    gt_G.add_edges_from(gt_edges)

    # build the pred graph
    pred_G = nx.Graph()
    pred_G.add_nodes_from(bl_tumors, bipartite='bl')
    pred_G.add_nodes_from(fu_tumors, bipartite='fu')
    pred_G.add_edges_from(pred_edges)

    # extract CC types of GT and PRED graphs
    gt_CCs, gt_lesions, gt_CC_types = get_CC_types_for_graph(gt_G)
    pred_CCs, pred_lesions, pred_CC_types = get_CC_types_for_graph(pred_G)

    # --------------------------- preparing the GT/PRED_CC_types tuples ---------------------------

    n_edges_pred, n_edges_gt = pred_CC_types[0], gt_CC_types[0]
    delta_n_edges = n_edges_pred - n_edges_gt
    delta_n_CCs = pred_CC_types[1] - gt_CC_types[1]

    gt_CC_types = (case_name, BL_n_tumors, FU_n_tumors) + gt_CC_types
    pred_CC_types = (case_name, BL_n_tumors, FU_n_tumors) + pred_CC_types

    # --------------------------- preparing the delta_CC_types tuple ---------------------------

    equivalent_CCs = {'unique': 0,
                      'disappeared': 0,
                      'new': 0,
                      'split': 0,
                      'merge': 0,
                      'complex': 0}

    n_equivalent_CCs = 0

    for gt_CC, CC_type in gt_CCs.items():

        # checking if the GT CC is in PRED graph
        for pred_CC in pred_CCs.keys():
            if (gt_CC.nodes == pred_CC.nodes) and (gt_CC.edges == pred_CC.edges):
                equivalent_CCs[CC_type] += 1
                n_equivalent_CCs += 1
                del pred_CCs[pred_CC]
                break

    precision_of_CCs = 1 if len(pred_CCs) == 0 else n_equivalent_CCs / len(pred_CCs)
    recall_of_CCs = 1 if len(gt_CCs) == 0 else n_equivalent_CCs / len(gt_CCs)

    delta_CC_types = (case_name, BL_n_tumors, FU_n_tumors, delta_n_edges, delta_n_CCs, n_equivalent_CCs, precision_of_CCs, recall_of_CCs, *equivalent_CCs.values())

    # --------------------------- preparing the delta_lesion_types tuple ---------------------------

    gt_lesions = pd.DataFrame(gt_lesions.items(), columns=['tumor', 'type'])
    pred_lesions = pd.DataFrame(pred_lesions.items(), columns=['tumor', 'type'])

    lesion_types = gt_lesions.merge(pred_lesions, on='tumor', suffixes=("_gt", "_pred"))
    right_lesion_types = lesion_types[lesion_types['type_gt'] == lesion_types['type_pred']]
    right_bl_lesion_types = right_lesion_types[right_lesion_types['tumor'].str.endswith('_bl')]
    right_fu_lesion_types = right_lesion_types[right_lesion_types['tumor'].str.endswith('_fu')]

    n_right_BL_labels = right_bl_lesion_types.shape[0]
    n_right_FU_labels = right_fu_lesion_types.shape[0]

    def get_n_right_labels(scan_type, lesion_type):
        if scan_type == 'BL':
            return right_bl_lesion_types[right_bl_lesion_types['type_gt'] == lesion_type].shape[0]
        return right_fu_lesion_types[right_fu_lesion_types['type_gt'] == lesion_type].shape[0]

    n_right_BL_unique_labels = get_n_right_labels('BL', 'unique')
    n_right_FU_unique_labels = get_n_right_labels('FU', 'unique')
    n_right_BL_disappeared_labels = get_n_right_labels('BL', 'disappeared')
    n_right_FU_new_labels = get_n_right_labels('FU', 'new')
    n_right_BL_split_labels = get_n_right_labels('BL', 'split')
    n_right_FU_split_labels = get_n_right_labels('FU', 'split')
    n_right_BL_merge_labels = get_n_right_labels('BL', 'merge')
    n_right_FU_merge_labels = get_n_right_labels('FU', 'merge')
    n_right_BL_complex_labels = get_n_right_labels('BL', 'complex')
    n_right_FU_complex_labels = get_n_right_labels('FU', 'complex')

    delta_lesion_types = (case_name, BL_n_tumors, FU_n_tumors, n_right_BL_labels, n_right_FU_labels,
                         n_right_BL_unique_labels, n_right_FU_unique_labels, n_right_BL_disappeared_labels,
                         n_right_FU_new_labels, n_right_BL_split_labels, n_right_FU_split_labels,
                         n_right_BL_merge_labels, n_right_FU_merge_labels, n_right_BL_complex_labels,
                         n_right_FU_complex_labels)

    def get_lesion_type_confusion_matrix(df_lesion_types):
        labels = ['unique', 'disappeared', 'new', 'split', 'merge', 'complex']
        conf_mat = np.zeros((6, 6))
        for i, true_label in enumerate(labels):
            for j, pred_label in enumerate(labels):
                conf_mat[i, j] = df_lesion_types[(df_lesion_types['type_gt'] == true_label) & (df_lesion_types['type_pred'] == pred_label)].shape[0]
        return conf_mat

    bl_lesion_types_conf_mat = get_lesion_type_confusion_matrix(lesion_types[lesion_types['tumor'].str.endswith('_bl')])
    fu_lesion_types_conf_mat = get_lesion_type_confusion_matrix(lesion_types[lesion_types['tumor'].str.endswith('_fu')])
    all_lesion_types_conf_mat = get_lesion_type_confusion_matrix(lesion_types)

    # --------------------------- preparing the delta_edges tuple ---------------------------

    n_TP = len(set(gt_G.edges) & set(pred_G.edges))
    n_FP = len(pred_G.edges) - n_TP
    n_FN = len(gt_G.edges) - n_TP
    precision = 1 if (n_TP + n_FP == 0) else n_TP/(n_TP + n_FP)
    recall = 1 if (n_TP + n_FN == 0) else n_TP/(n_TP + n_FN)
    f1_score = 0 if (precision + recall == 0) else 2 * precision * recall / (precision + recall)

    delta_edges = (case_name, BL_n_tumors, FU_n_tumors, n_edges_gt, n_edges_pred, n_TP, n_FP, n_FN, precision, recall, f1_score)

    bl_diameters = np.array(bl_diameters) if bl_diameters is not None else None
    fu_diameters = np.array(fu_diameters) if fu_diameters is not None else None
    bl_organ_volume = bl_organ_volume if bl_organ_volume is not None else np.nan
    fu_organ_volume = fu_organ_volume if fu_organ_volume is not None else np.nan

    return (gt_CC_types, pred_CC_types, delta_CC_types, delta_lesion_types, delta_edges, bl_lesion_types_conf_mat,
            fu_lesion_types_conf_mat, all_lesion_types_conf_mat, bl_diameters, fu_diameters, bl_organ_volume, fu_organ_volume)


def write_diff_in_matching_according_to_lesion_type(pairs_dir_names: List[str], n_processes=None,
                                                    results_file: Optional[str] = 'measures_results/diff_in_matching_according_to_CC_type.xlsx',
                                                    print_summarization=True,
                                                    max_dilation_for_pred=9,
                                                    without_improving_registration=False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if n_processes is None:
        n_processes = os.cpu_count() - 2

    if without_improving_registration:
        matching_graph_json_files = [(f'{dir_name}/gt_matching_graph.json', f'{dir_name}/pred_without_improving_registration_matching_graph.json') for
                                     dir_name in pairs_dir_names]
    else:
        matching_graph_json_files = [(f'{dir_name}/gt_matching_graph.json', f'{dir_name}/pred_matching_graph.json') for dir_name in pairs_dir_names]
    for (GT_matching_graph_json_file_name, Pred_matching_graph_json_file_name) in matching_graph_json_files:
        assert os.path.isfile(GT_matching_graph_json_file_name), f'The following file does not exist: {GT_matching_graph_json_file_name}'
        assert os.path.isfile(Pred_matching_graph_json_file_name), f'The following file does not exist: {Pred_matching_graph_json_file_name}'

    matching_graph_json_files = matching_graph_json_files

    # pairs_changes = process_map(calculate_diff_in_matching_according_to_lesion_type_v1, matching_graph_json_files, max_workers=n_processes)
    # pairs_changes = process_map(calculate_diff_in_matching_according_to_lesion_type_v2, matching_graph_json_files, max_workers=n_processes)
    pairs_changes = process_map(partial(calculate_diff_in_matching_according_to_lesion_type_v2, max_dilation_for_pred=max_dilation_for_pred), matching_graph_json_files, max_workers=n_processes)
    # pairs_changes = list(map(partial(calculate_diff_in_matching_according_to_lesion_type_v2, max_dilation_for_pred=max_dilation_for_pred), matching_graph_json_files))
    # pairs_changes = list(map(_calculate_diff_in_matching_according_to_lesion_type_v1, matching_graph_json_files))

    (gt_CC_types, pred_CC_types, delta_CC_types, delta_lesion_types, delta_edges, bl_lesion_types_conf_mat,
     fu_lesion_types_conf_mat, all_lesion_types_conf_mat, bl_diameters, fu_diameters, bl_organ_volume, fu_organ_volume) = zip(*pairs_changes)

    bl_lesion_types_conf_mat = np.stack(bl_lesion_types_conf_mat).sum(axis=0).astype(np.int16)
    fu_lesion_types_conf_mat = np.stack(fu_lesion_types_conf_mat).sum(axis=0).astype(np.int16)
    all_lesion_types_conf_mat = np.stack(all_lesion_types_conf_mat).sum(axis=0).astype(np.int16)
    def get_conf_mat_str(scan_type):
        conf_mat = bl_lesion_types_conf_mat if scan_type == 'BL' else (fu_lesion_types_conf_mat if scan_type == 'FU' else all_lesion_types_conf_mat)
        classes = ['Unique', 'Disappeared', 'New', 'Split', 'Merge', 'Complex']
        df = pd.DataFrame(data=conf_mat, columns=classes, index=classes)
        if scan_type == 'BL':
            del df['New']
            df.drop('New', inplace=True)
        elif scan_type == 'FU':
            del df['Disappeared']
            df.drop('Disappeared', inplace=True)
        
        res_str = f'% ---- {scan_type} Lesions - Confusion Matrix ---- %\n\n' + '\\def\\myConfMat{{\n'
        for r in range(df.shape[0]):
            res_str += '{'
            for c in range(df.shape[1]):
                res_str += f'{df.iloc[r,c]}, '
            res_str += '},\n'
        res_str += '}}\n\n \\def\\classNames{{'
        for l in df.columns:
            res_str += f'"{l}", '
        res_str += '}}\n\n\\def\\numClasses{' + str(len(df.columns)) + '}\n\n'

        res_str += """\\def\\myScale{1.8} % 1.5 is a good scale. Values under 1 may need smaller fonts!
\\begin{tikzpicture}[
    scale = \\myScale,
    % font={\\scriptsize}, %for smaller scales, even \\tiny may be useful
    ]

\\tikzset{vertical label/.style={rotate=90,anchor=east}}   % usable styles for below
\\tikzset{diagonal label/.style={rotate=45,anchor=north east}}

\\foreach \\y in {1,...,\\numClasses} %loop vertical starting on top
{
    % Add class name on the left
    \\node [anchor=east] at (0.4,-\\y) {\\pgfmathparse{\\classNames[\\y-1]}\\pgfmathresult}; 
    
    \\foreach \\x in {1,...,\\numClasses}  %loop horizontal starting on left
    {
%---- Start of automatic calculation of colTotSamples for the column ------------   
    \\def\\colTotSamples{0}
    \\foreach \\ll in {1,...,\\numClasses}
    {
        \\pgfmathparse{\\myConfMat[\\ll-1][\\x-1]}   %fetch next element
        \\xdef\\colTotSamples{\\colTotSamples+\\pgfmathresult} %accumulate it with previous sum
        %must use \\xdef fro global effect otherwise lost in foreach loop!
    }
    \\pgfmathparse{\\colTotSamples} \\xdef\\colTotSamples{\\pgfmathresult}  % put the final sum in variable
%---- End of automatic calculation of colTotSamples ----------------
    \\node[] at (\\x,-0.25) {\\pgfmathprintnumber{\\colTotSamples}}; % add total above the columns
    
    %---- Start of automatic calculation of rowTotSamples for the rows ------------   
    \\def\\rowTotSamples{0}
    \\foreach \\ll in {1,...,\\numClasses}
    {
        \\pgfmathparse{\\myConfMat[\\y-1][\\ll-1]}   %fetch next element
        \\xdef\\rowTotSamples{\\rowTotSamples+\\pgfmathresult} %accumulate it with previous sum
        %must use \\xdef fro global effect otherwise lost in foreach loop!
    }
    \\pgfmathparse{\\rowTotSamples} \\xdef\\rowTotSamples{\\pgfmathresult}  % put the final sum in variable
%---- End of automatic calculation of rowTotSamples ----------------
    \\node[] at (\\numClasses+1,-\\y) {\\pgfmathprintnumber{\\rowTotSamples}}; % add total left to the rows
    
    \\begin{scope}[shift={(\\x,-\\y)}]
        \\def\\mVal{\\myConfMat[\\y-1][\\x-1]} % The value at index y,x (-1 because of zero indexing)
        \\pgfmathtruncatemacro{\\r}{\\mVal}   %
        \\pgfmathtruncatemacro{\\p}{round(\\r/\\colTotSamples*100)}
        \\coordinate (C) at (0,0);
        \\ifthenelse{\\p<50}{\\def\\txtcol{black}}{\\def\\txtcol{white}} %decide text color for contrast
        \\node[
            draw,                 %draw lines
            text=\\txtcol,         %text color (automatic for better contrast)
            align=center,         %align text inside cells (also for wrapping)
            fill=black!\\p,        %intensity of fill (can change base color)
            minimum size=\\myScale*10mm,    %cell size to fit the scale and integer dimensions (in cm)
            inner sep=0,          %remove all inner gaps to save space in small scales
            ] (C) {\\r\\\\\\p\\%};     %text to put in cell (adapt at will)
        %Now if last vertical class add its label at the bottom
        \\ifthenelse{\\y=\\numClasses}{
        \\node [] at ($(C)-(0,0.75)$) % can use vertical or diagonal label as option
        {\\pgfmathparse{\\classNames[\\x-1]}\\pgfmathresult};}{}
    \\end{scope}
    }
}
%Now add x and y labels on suitable coordinates
\\coordinate (yaxis) at (-1,0.325-\\numClasses/2);  %must adapt if class labels are wider!
\\coordinate (xaxis) at (0.5+\\numClasses/2, -\\numClasses-1.25); %id. for non horizontal labels!
\\node [vertical label, font=\\bfseries] at (yaxis) {True Lesion Type};
\\node [font=\\bfseries]               at (xaxis) {Predicted Lesion Type};

\\node[anchor=east, font=\\bfseries] at (0.4, -0.25) {Total}; % add y total

\\node[font=\\bfseries] at (\\numClasses+1, -\\numClasses-0.75) {Total}; % add x total

\\end{tikzpicture}\n"""
        return res_str
     
    CC_types_columns = ['name', 'BL_n_tumors', 'FU_n_tumors', 'n_edges', 'n_CCs', 'n_unique', 'n_disappeared_BL', 'n_split_BL',
                        'n_merge_BL', 'n_complex_BL', 'n_new_FU', 'n_split_FU', 'n_merge_FU', 'n_complex_FU', 'n_complex_event',
                        'contains_merge', 'contains_split', 'contains_merge_or_split', 'contains_merge_and_split', 'contains_complex']
    gt_CC_types = pd.DataFrame(data=gt_CC_types, columns=CC_types_columns)
    pred_CC_types = pd.DataFrame(data=pred_CC_types, columns=CC_types_columns)

    delta_CC_types_columns = ['name', 'BL_n_tumors', 'FU_n_tumors', 'Delta_n_edges', 'Delta_n_CCs', 'n_equivalent_CCs',
                              'Precision_of_CCs', 'Recall_of_CCs', 'n_equivalent_unique_CCs', 'n_equivalent_disappeared_CCs',
                              'n_equivalent_new_CCs', 'n_equivalent_split_CCs', 'n_equivalent_merge_CCs',
                              'n_equivalent_complex_CCs']
    delta_CC_types = pd.DataFrame(data=delta_CC_types, columns=delta_CC_types_columns)

    delta_lesion_types_columns = ['name', 'BL_n_tumors', 'FU_n_tumors', 'n_right_BL_labels', 'n_right_FU_labels',
                                  'n_right_BL_unique_labels', 'n_right_FU_unique_labels', 'n_right_BL_disappeared_labels',
                                  'n_right_FU_new_labels', 'n_right_BL_split_labels', 'n_right_FU_split_labels',
                                  'n_right_BL_merge_labels', 'n_right_FU_merge_labels', 'n_right_BL_complex_labels',
                                  'n_right_FU_complex_labels']
    delta_lesion_types = pd.DataFrame(data=delta_lesion_types, columns=delta_lesion_types_columns)

    delta_edges_columns = ['name', 'BL_n_tumors', 'FU_n_tumors', 'n_edges_GT', 'n_edges_PRED',
                           'n_TP', 'n_FP', 'n_FN', 'Precision', 'Recall', 'F1 - Score']
    delta_edges = pd.DataFrame(data=delta_edges, columns=delta_edges_columns)

    scans_measures_tmp = pd.DataFrame(data={'name': delta_lesion_types['name'], 'bl_diameters': bl_diameters,
                                        'fu_diameters': fu_diameters, 'bl_organ_volume': bl_organ_volume,
                                        'fu_organ_volume': fu_organ_volume})
    bl_names, fu_names = zip(*(pair_name.replace('BL_', '').split('_FU_') for pair_name in scans_measures_tmp['name']))
    scans_measures_tmp['bl_name'] = bl_names
    scans_measures_tmp['fu_name'] = fu_names
    scans_measures = pd.concat((scans_measures_tmp[['bl_name', 'bl_diameters', 'bl_organ_volume']].rename(columns={'bl_name': 'name', 'bl_diameters': 'diameters', 'bl_organ_volume': 'organ_volume'}),
                               scans_measures_tmp[['fu_name', 'fu_diameters', 'fu_organ_volume']].rename(columns={'fu_name': 'name', 'fu_diameters': 'diameters', 'fu_organ_volume': 'organ_volume'})))
    scans_measures.drop_duplicates('name', inplace=True, ignore_index=True)
    scans_measures['patient'] = ['_'.join([c for c in s.split('_') if not c.isdigit()]) for s in scans_measures['name']]
    scans_measures = scans_measures[['name', 'patient', 'diameters', 'organ_volume']]

    pairs_lesions_diameters = scans_measures_tmp[['name', 'bl_diameters', 'fu_diameters']]
    pairs_lesions_diameters['patient'] = ['_'.join([c for c in p.replace('BL_', '').split('_FU_')[0].split('_') if not c.isdigit()]) for p in pairs_lesions_diameters['name']]

    gt_CC_types = sort_dataframe_by_key(gt_CC_types, 'name', pairs_sort_key)
    pred_CC_types = sort_dataframe_by_key(pred_CC_types, 'name', pairs_sort_key)
    delta_CC_types = sort_dataframe_by_key(delta_CC_types, 'name', pairs_sort_key)
    delta_lesion_types = sort_dataframe_by_key(delta_lesion_types, 'name', pairs_sort_key)
    delta_edges = sort_dataframe_by_key(delta_edges, 'name', pairs_sort_key)
    scans_measures = sort_dataframe_by_key(scans_measures, 'name', scans_sort_key)
    pairs_lesions_diameters = sort_dataframe_by_key(pairs_lesions_diameters, 'name', pairs_sort_key)

    if print_summarization:

        from tabulate import tabulate
        TABLEFMT = 'grid'
        section = lambda txt: txt
        subsection = section
        sp = '\n\n--------------------------------------------------------------------------------------------------------------------------------------\n\n\n'
        prefix = ''
        suffix = ''

        latex = True
        if latex:
            TABLEFMT = 'latex'
            section = lambda txt: f'\section{{{txt}}}'
            subsection = lambda txt: f'\subsection{{{txt}}}'
            sp = '\n\n\n\n\n'
            suffix = "\n\n\\end{document}\n"
            prefix = """\\documentclass{article}

% Language setting
% Replace `english' with e.g. `spanish' to change the document language
\\usepackage[english]{babel}

% Set page size and margins
% Replace `letterpaper' with`a4paper' for UK/EU standard size
\\usepackage[letterpaper,top=2cm,bottom=2cm,left=1cm,right=1cm,marginparwidth=1.75cm]{geometry}

% Useful packages
\\usepackage{amsmath}
\\usepackage{graphicx}
\\usepackage[colorlinks=true, allcolors=blue]{hyperref}

\\usepackage{tikz}
\\usepackage{standalone}
\\usepackage{ifthen}
\\usetikzlibrary{matrix,calc}

\\title{Matching Results Summarization}
\\author{Shalom Rochman}

\\begin{document}
\\maketitle

"""

        def center_table(table_txt:str):
            if latex:
                table_txt = '\\begin{center}\n' + table_txt + '\n\end{center}'
            return table_txt

        def add_line_after_first_column(txt: str):
            if latex:
                txt_lines = txt.splitlines()
                first_line = txt_lines[0]
                assert first_line.startswith('\\begin{tabular}')
                begin_term, tabular_term, ccc_term = first_line.split('{')
                ccc_term = ccc_term[0] + '|' + ccc_term[1:]
                first_line = '{'.join([begin_term, tabular_term, ccc_term])
                txt_lines[0] = first_line
                txt = '\n'.join(txt_lines)
            return txt

        def put_tables_side_by_side(*tables, parameters_for_tabulate=None, spacing: int = 10):
            if parameters_for_tabulate is None:
                parameters_for_tabulate = {'headers': 'keys',
                                           'tablefmt': TABLEFMT,
                                           'numalign': 'center',
                                           'stralign': 'center',
                                           'showindex': False
                                           }


            if latex:
                return center_table(('\n' + ' '.join(['\quad']*spacing) + '\n').join([tabulate(t, **parameters_for_tabulate) for t in tables]))

            string_tables_split = [tabulate(t, **parameters_for_tabulate).splitlines() for t in tables]

            spacing_str = " " * spacing

            num_lines = max(map(len, string_tables_split))
            paddings = [max(map(len, s_lines)) for s_lines in string_tables_split]

            final_str = ''

            for i in range(num_lines):
                line_each_table = []
                for padding, table_lines in zip(paddings, string_tables_split):
                    if len(table_lines) <= i:
                        line_each_table.append(" " * (padding + spacing))
                    else:
                        line_table_string = table_lines[i]
                        line_len = len(line_table_string)
                        line_each_table.append(
                            line_table_string + (" " * (padding - line_len)) + spacing_str
                        )

                final_line_string = "".join(line_each_table)
                final_str += final_line_string
                final_str += '\n' if i < (num_lines - 1) else ''

            return final_str

        def separate_to_several_lines(lst_of_txt):
            assert len(lst_of_txt) > 1
            res = """\\begin{tabular}{@{}c@{}}""" + lst_of_txt[0]
            for txt in lst_of_txt[1:]:
                res += " \\\\ " + txt
            res += """\\end{tabular}"""
            return res

        def get_dataset_description():
            get_patient_name = lambda case_name: '_'.join([c for c in case_name.replace('BL_', '').split('_FU_')[0].split('_') if not c.isdigit()])
            get_bl_name = lambda case_name: case_name.replace('BL_', '').split('_FU_')[0]
            get_fu_name = lambda case_name: case_name.replace('BL_', '').split('_FU_')[1]
            get_date_from_name_y_m_d = lambda name: [int(f) for f in name.split('_')[-3:][::-1]]
            get_time_interval = lambda pair: abs((date(*get_date_from_name_y_m_d(get_fu_name(pair))) - date(*get_date_from_name_y_m_d(get_bl_name(pair)))).days)

            bl_names = set([get_bl_name(name) for name in gt_CC_types['name']])
            fu_names = set([get_fu_name(name) for name in gt_CC_types['name']])
            scan_names = bl_names | fu_names

            patients = pd.DataFrame({'p_name': [get_patient_name(name) for name in gt_CC_types['name']]}).groupby('p_name')['p_name'].count().reset_index(name="n_pairs").groupby('n_pairs')['n_pairs'].count().reset_index(name="n_patients")
            n_patients = patients['n_patients'].sum()
            pairs_partitioning_df = patients.rename(columns={'n_pairs': '#pairs', 'n_patients': '#patients'})[['#patients', '#pairs']]

            pairs_per_patient = []
            for i in range(pairs_partitioning_df.shape[0]):
                pairs_per_patient += [pairs_partitioning_df.iloc[i, 1]] * pairs_partitioning_df.iloc[i, 0]
            mean_pairs_per_patient = np.mean(pairs_per_patient)
            std_pairs_per_patient = np.std(pairs_per_patient)
            min_pairs_per_patient = np.min(pairs_per_patient)
            max_pairs_per_patient = np.max(pairs_per_patient)

            n_pairs = gt_CC_types.shape[0]

            scans_per_patient = []
            previous_patient = ''
            for s_name in sorted(list(scan_names)):
                p_name = get_patient_name(s_name)
                if p_name == previous_patient:
                    scans_per_patient[-1] += 1
                else:
                    scans_per_patient += [1]
                previous_patient = p_name
            mean_scans_per_patient = np.mean(scans_per_patient)
            std_scans_per_patient = np.std(scans_per_patient)
            min_scans_per_patient = np.min(scans_per_patient)
            max_scans_per_patient = np.max(scans_per_patient)
            total_scans = len(scan_names)

            dataset_df = pd.DataFrame({' ': ['Mean per patient', 'STD per patient', 'Min per patient', 'Max per patient', 'Total'],
                                       '#patients': ['-', '-', '-', '-', f'{int(n_patients):,}'],
                                       '#scans': [f'{mean_scans_per_patient:.2f}', f'{std_scans_per_patient:.2f}', f'{min_scans_per_patient:,}', f'{max_scans_per_patient:,}', f'{total_scans:,}'],
                                       '#pairs': [f'{mean_pairs_per_patient:.2f}', f'{std_pairs_per_patient:.2f}', f'{min_pairs_per_patient:,}', f'{max_pairs_per_patient:,}', f'{n_pairs:,}']
                                       })


            pairs = gt_CC_types['name'].to_list()
            time_intervals = np.asarray([get_time_interval(p) for p in pairs])
            mean_time_intervals_in_days = np.mean(time_intervals)
            std_time_intervals_in_days = np.std(time_intervals)
            min_time_intervals_in_days = np.min(time_intervals)
            max_time_intervals_in_days = np.max(time_intervals)
            mean_time_intervals_in_years = np.mean(time_intervals) / 365
            std_time_intervals_in_years = np.std(time_intervals / 365)
            min_time_intervals_in_years = np.min(time_intervals) / 365
            max_time_intervals_in_years = np.max(time_intervals) / 365
            mean_time_intervals_in_months = np.mean(time_intervals) / 30.4167
            std_time_intervals_in_months = np.std(time_intervals / 30.4167)
            min_time_intervals_in_months = np.min(time_intervals) / 30.4167
            max_time_intervals_in_months = np.max(time_intervals) / 30.4167
            time_intervals_df = pd.DataFrame({' ': ['Mean', 'STD', 'Min', 'Max'],
                                              'Time Interval': [f'{mean_time_intervals_in_days:.2f} days ({mean_time_intervals_in_months:.2f} months, {mean_time_intervals_in_years:.2f} years)',
                                                                f'{std_time_intervals_in_days:.2f} days ({std_time_intervals_in_months:.2f} months, {std_time_intervals_in_years:.2f} years)',
                                                                f'{min_time_intervals_in_days:,} days ({min_time_intervals_in_months:.2f} months, {min_time_intervals_in_years:.2f} years)',
                                                                f'{max_time_intervals_in_days:,} days ({max_time_intervals_in_months:.2f} months, {max_time_intervals_in_years:.2f} years)']
            })

            n_simple = gt_CC_types[(gt_CC_types['contains_merge'] == 0) & (gt_CC_types['contains_split'] == 0) & (gt_CC_types['contains_complex'] == 0)].shape[0]
            n_contain_merge = gt_CC_types['contains_merge'].sum()
            n_contain_split = gt_CC_types['contains_split'].sum()
            n_contain_merge_and_split = gt_CC_types['contains_merge_and_split'].sum()
            n_contain_merge_or_split = gt_CC_types['contains_merge_or_split'].sum()
            n_contain_complex = gt_CC_types['contains_complex'].sum()
            complexity_of_pairs_df = pd.DataFrame({
                '': ['Simple', 'Contain Merge', 'Contain Split', 'Contain Split and Merge', 'Contain Split or Merge', 'Contain Complex'],
                '#pairs': [f'{int(n_simple):,}\n({100*n_simple/n_pairs:.1f}%)',
                           f'{int(n_contain_merge):,}\n({100*n_contain_merge/n_pairs:.1f}%)',
                           f'{int(n_contain_split):,}\n({100*n_contain_split/n_pairs:.1f}%)',
                           f'{int(n_contain_merge_and_split):,}\n({100*n_contain_merge_and_split/n_pairs:.1f}%)',
                           f'{int(n_contain_merge_or_split):,}\n({100*n_contain_merge_or_split/n_pairs:.1f}%)',
                           f'{int(n_contain_complex):,}\n({100*n_contain_complex/n_pairs:.1f}%)']
            })

            def get_measures_df(measures_df):

                assert np.isin(['patient', 'diameters'], measures_df.columns).all()

                n_lesions_per_scan = np.asarray([d_s.size for d_s in measures_df['diameters']])
                n_lesions_above_5mm_per_scan = np.asarray([d_s[d_s >= 5].size for d_s in measures_df['diameters']])
                n_lesions_above_10mm_per_scan = np.asarray([d_s[d_s >= 10].size for d_s in measures_df['diameters']])
                lesions_diameters = np.concatenate(measures_df['diameters'])
                lesions_volumes = np.pi * (lesions_diameters ** 3) / 6000
                mean_n_lesions_per_patient_in_a_scan = []
                for _, patient_scans_measures in measures_df.groupby('patient'):
                    mean_n_lesions_per_patient_in_a_scan += [
                        np.mean([d_s.size for d_s in patient_scans_measures['diameters']])]

                return pd.DataFrame({
                    '': ['Mean', 'STD', 'Min', 'Max', 'Total'],
                    separate_to_several_lines(['Lesion', 'Diameter', '(mm)']): [f'{lesions_diameters.mean():.2f}',
                                                                                f'{lesions_diameters.std():.2f}',
                                                                                f'{lesions_diameters.min():.2f}',
                                                                                f'{lesions_diameters.max():.2f}',
                                                                                f'-'],
                    separate_to_several_lines(['Lesion', 'Volumes', '(CC)']): [f'{lesions_volumes.mean():.2f}',
                                                                               f'{lesions_volumes.std():.2f}',
                                                                               f'{lesions_volumes.min():.2f}',
                                                                               f'{lesions_volumes.max():.2f}',
                                                                               f'-'],
                    separate_to_several_lines(['#lesions', 'per', 'scan']): [f'{n_lesions_per_scan.mean():.2f}',
                                                                             f'{n_lesions_per_scan.std():.2f}',
                                                                             f'{n_lesions_per_scan.min():.2f}',
                                                                             f'{n_lesions_per_scan.max():.2f}',
                                                                             f'{n_lesions_per_scan.sum():.2f}'],
                    separate_to_several_lines(['#lesions', 'above 5mm', 'per scan']): [
                        f'{n_lesions_above_5mm_per_scan.mean():.2f}', f'{n_lesions_above_5mm_per_scan.std():.2f}',
                        f'{n_lesions_above_5mm_per_scan.min():.2f}', f'{n_lesions_above_5mm_per_scan.max():.2f}',
                        f'{n_lesions_above_5mm_per_scan.sum():.2f}'],
                    separate_to_several_lines(['#lesions', 'above 10mm', 'per scan']): [
                        f'{n_lesions_above_10mm_per_scan.mean():.2f}',
                        f'{n_lesions_above_10mm_per_scan.std():.2f}',
                        f'{n_lesions_above_10mm_per_scan.min():.2f}',
                        f'{n_lesions_above_10mm_per_scan.max():.2f}',
                        f'{n_lesions_above_10mm_per_scan.sum():.2f}'],
                    separate_to_several_lines(['Mean lesions', 'per patient', 'in a scan']): [
                        f'{np.mean(mean_n_lesions_per_patient_in_a_scan):.2f}',
                        f'{np.std(mean_n_lesions_per_patient_in_a_scan):.2f}',
                        f'{np.min(mean_n_lesions_per_patient_in_a_scan):.2f}',
                        f'{np.max(mean_n_lesions_per_patient_in_a_scan):.2f}',
                        '-']
                })


            prior_scans_measures_df = get_measures_df(pairs_lesions_diameters[['name', 'patient', 'bl_diameters']].rename(columns={'bl_diameters': 'diameters'}))
            current_scans_measures_df = get_measures_df(pairs_lesions_diameters[['name', 'patient', 'fu_diameters']].rename(columns={'fu_diameters': 'diameters'}))
            individual_scans_measures_df = get_measures_df(scans_measures)
            individual_scans_measures_df[separate_to_several_lines(['Organ', 'Volume', '(CC)'])] = [
                f'{scans_measures["organ_volume"].mean():.2f}',
                f'{scans_measures["organ_volume"].std():.2f}',
                f'{scans_measures["organ_volume"].min():.2f}',
                f'{scans_measures["organ_volume"].max():.2f}',
                '-'
            ]

            edges_df = pd.DataFrame({
                '': ['Mean', 'STD', 'Min', 'Max', 'Total'],
                '#edges per pair': [f'{gt_CC_types["n_edges"].mean():.2f}',
                                    f'{gt_CC_types["n_edges"].std():.2f}',
                                    f'{gt_CC_types["n_edges"].min():,}',
                                    f'{gt_CC_types["n_edges"].max():,}',
                                    f'{gt_CC_types["n_edges"].sum():,}',
                                    ]
            })

            def fix_latex_format(s):
                return s.replace('\\textbackslash{}', '\\').replace('\\{', '{').replace('\\}', '}')

            return f"{section('Dataset:')}\n\n" \
                   f"{center_table(add_line_after_first_column(tabulate(dataset_df, headers='keys', tablefmt=TABLEFMT, numalign='center', stralign='center', showindex=False)))}\n\n" \
                   f"{subsection('Patients partitioning:')}\n\n" \
                   f"{center_table(tabulate(pairs_partitioning_df, headers='keys', tablefmt=TABLEFMT, numalign='center', stralign='center', showindex=False))}\n\n" \
                   f"{subsection('Pairs time interval:')}\n\n" \
                   f"{center_table(add_line_after_first_column(tabulate(time_intervals_df, headers='keys', tablefmt=TABLEFMT, numalign='center', stralign='center', showindex=False)))}\n\n" \
                   f"{subsection('Per individual scan measures:')}\n\n" \
                   f"{center_table(add_line_after_first_column(fix_latex_format(tabulate(individual_scans_measures_df, headers='keys', tablefmt=TABLEFMT, numalign='center', stralign='center', showindex=False))))}\n\n" \
                   f"{subsection('Per pair scans measures:')}\n\n" \
                   f"{center_table(add_line_after_first_column(fix_latex_format(tabulate(prior_scans_measures_df, headers='keys', tablefmt=TABLEFMT, numalign='center', stralign='center', showindex=False))))}\n\n" \
                   f"{subsection('Per current scans measures:')}\n\n" \
                   f"{center_table(add_line_after_first_column(fix_latex_format(tabulate(current_scans_measures_df, headers='keys', tablefmt=TABLEFMT, numalign='center', stralign='center', showindex=False))))}\n\n" \
                   f"{subsection('Pairs edges measures:')}\n\n" \
                   f"{center_table(add_line_after_first_column(tabulate(edges_df, headers='keys', tablefmt=TABLEFMT, numalign='center', stralign='center', showindex=False)))}\n\n" \
                   f"{subsection('Complexity of pairs:')}\n\n" \
                   f"{center_table(add_line_after_first_column(tabulate(complexity_of_pairs_df, headers='keys', tablefmt=TABLEFMT, numalign='center', stralign='center', showindex=False)))}\n"

        def get_edges_evaluation():
            edges_evaluation_df = pd.DataFrame({
                '#Edges-in-GT': [f"{delta_edges['n_edges_GT'].sum():,}"],
                '#Edges-in-PRED': [f"{delta_edges['n_edges_PRED'].sum():,}"],
                # 'Delta #edges': [f"{delta_edges['n_edges_PRED'].sum() - delta_edges['n_edges_GT'].sum():,}"],
                '#TP-edges': [f"{delta_edges['n_TP'].sum():,}"],
                '#FP-edges': [f"{delta_edges['n_FP'].sum():,}"],
                '#FN-edges': [f"{delta_edges['n_FN'].sum():,}"],
                'Precision': [f"{delta_edges['n_TP'].sum()/(delta_edges['n_TP'].sum() + delta_edges['n_FP'].sum()):.2f}"],
                'Recall': [f"{delta_edges['n_TP'].sum()/(delta_edges['n_TP'].sum() + delta_edges['n_FN'].sum()):.2f}"],
                # 'F1-Score': [f"{delta_edges['n_TP'].sum()/(delta_edges['n_TP'].sum() + 0.5 * (delta_edges['n_FP'].sum() + delta_edges['n_FN'].sum())):.2f}"]
            })

            return f"{section('Edges evaluation:')}\n\n" \
                   f"{center_table(tabulate(edges_evaluation_df, headers='keys', tablefmt=TABLEFMT, numalign='center', stralign='center', showindex=False))}\n"

        def get_lesion_evaluation():

            def get_lesions_evaluation_df(scan_type):

                if scan_type == 'ALL':
                    #n_bl_tumors = int(delta_lesion_types["BL_n_tumors"].sum())
                    #n_bl_right_labels = int(delta_lesion_types["n_right_BL_labels"].sum())
                    #n_fu_tumors = int(delta_lesion_types["FU_n_tumors"].sum())
                    #n_fu_right_labels = int(delta_lesion_types["n_right_FU_labels"].sum())
                    #return pd.DataFrame({
                    #    "Prior lesions accuracy": [f'{n_bl_right_labels:,}/{n_bl_tumors:,} = {n_bl_right_labels/n_bl_tumors:.2f}'],
                    #    "Current lesions accuracy": [f'{n_fu_right_labels:,}/{n_fu_tumors:,} = {n_fu_right_labels/n_fu_tumors:.2f}'],
                    #    "All lesions accuracy": [f'{n_fu_right_labels + n_bl_right_labels:,}/{n_fu_tumors + n_bl_tumors:,} = {(n_fu_right_labels + n_bl_right_labels)/(n_fu_tumors + n_bl_tumors):.2f}']
                    #})

                    def get_lesion_types_weighted_precision_and_recall_and_CK_score(scan_type):
                        conf_mat = bl_lesion_types_conf_mat if scan_type == 'BL' else (fu_lesion_types_conf_mat if scan_type == 'FU' else all_lesion_types_conf_mat)
                        wei_pre = 0
                        wei_rec = 0
                        p_e = 0
                        for i in range(conf_mat.shape[0]):
                            try:
                                conf_mat_i_i = int(conf_mat[i,i])
                                conf_mat_col_i = int(conf_mat[:,i].sum())
                                conf_mat_row_i = int(conf_mat[i,:].sum())
                                wei_pre += conf_mat_i_i * conf_mat_row_i / conf_mat_col_i
                                wei_rec += conf_mat_i_i
                                p_e += conf_mat_col_i * conf_mat_row_i
                            except ZeroDivisionError:
                                pass
                        try:
                            conf_mat_sum = int(conf_mat.sum())
                            wei_pre /= conf_mat_sum
                            wei_rec /= conf_mat_sum
                            p_e /= (conf_mat_sum**2)
                        except ZeroDivisionError:
                            wei_pre = np.nan
                            wei_rec = np.nan
                            p_e = np.nan

                        p_0 = wei_rec
                        CK_score = (p_0 - p_e)/(1 - p_e)
 
                        return wei_pre, wei_rec, CK_score

                    bl_weighted_precision, bl_weighted_recall, bl_ck_score = get_lesion_types_weighted_precision_and_recall_and_CK_score('BL')
                    fu_weighted_precision, fu_weighted_recall, fu_ck_score = get_lesion_types_weighted_precision_and_recall_and_CK_score('FU')
                    all_weighted_precision, all_weighted_recall, all_ck_score = get_lesion_types_weighted_precision_and_recall_and_CK_score('ALL')

                    # return pd.DataFrame({
                    #     "Scan Type": ["Prior Lesions", "Current Lesions", "All Lesions"],
                    #     "Weighted Precision": [f'{bl_weighted_precision:.2f}', f'{fu_weighted_precision:.2f}', f'{all_weighted_precision:.2f}'],
                    #     "Weighted Recall (Accuracy)": [f'{bl_weighted_recall:.2f}', f'{fu_weighted_recall:.2f}', f'{all_weighted_recall:.2f}'],
                    #     "Cohen-Kappa Score": [f'{bl_ck_score:.2f}', f'{fu_ck_score:.2f}', f'{all_ck_score:.2f}']
                    #     })

                    return pd.DataFrame({
                        "Scan Type": ["Priors", "Currents"],
                        "Weighted Precision": [f'{bl_weighted_precision:.2f}', f'{fu_weighted_precision:.2f}'],
                        "Weighted Recall (Accuracy)": [f'{bl_weighted_recall:.2f}', f'{fu_weighted_recall:.2f}']
                    })

                New_or_Disappeared = 'New' if scan_type == 'FU' else 'Disappeared'
                new_or_disappeared = 'new' if scan_type == 'FU' else 'disappeared'

                data = {}
                for lesion_type in ['unique', new_or_disappeared, 'split', 'merge', 'complex']:
                    data[lesion_type] = {
                        'n_TP': int(delta_lesion_types[f"n_right_{scan_type}_{lesion_type}_labels"].sum()),
                        'n_GT': int(gt_CC_types[
                                        f"n_{lesion_type}{f'_{scan_type}' if lesion_type != 'unique' else ''}"].sum()),
                        'n_PRED': int(pred_CC_types[
                                          f"n_{lesion_type}{f'_{scan_type}' if lesion_type != 'unique' else ''}"].sum())
                    }

                recall = lambda type: f'{data[type]["n_TP"] / data[type]["n_GT"]:.2f}'
                precision = lambda type: f'{data[type]["n_TP"] / data[type]["n_PRED"]:.2f}'
                f1_score = lambda type: f'{data[type]["n_TP"] / (0.5 * (data[type]["n_GT"] + data[type]["n_PRED"])):.2f}'

                return pd.DataFrame({
                    'Lesion type': ['Unique', New_or_Disappeared, 'Split', 'Merge', 'Complex'],
                    '#TP-lesions': [f"{data['unique']['n_TP']:,}",
                                    f"{data[new_or_disappeared]['n_TP']:,}",
                                    f"{data['split']['n_TP']:,}",
                                    f"{data['merge']['n_TP']:,}",
                                    f"{data['complex']['n_TP']:,}"],
                    '#Lesions-in-GT': [f"{data['unique']['n_GT']:,}",
                                       f"{data[new_or_disappeared]['n_GT']:,}",
                                       f"{data['split']['n_GT']:,}",
                                       f"{data['merge']['n_GT']:,}",
                                       f"{data['complex']['n_GT']:,}"],
                    '#Lesions-in-PRED': [f"{data['unique']['n_PRED']:,}",
                                         f"{data[new_or_disappeared]['n_PRED']:,}",
                                         f"{data['split']['n_PRED']:,}",
                                         f"{data['merge']['n_PRED']:,}",
                                         f"{data['complex']['n_PRED']:,}"],
                    'Recall': [recall('unique'),
                               recall(new_or_disappeared),
                               recall('split'),
                               recall('merge'),
                               recall('complex')],
                    'Precision': [precision('unique'),
                                  precision(new_or_disappeared),
                                  precision('split'),
                                  precision('merge'),
                                  precision('complex')],
                    'F1-Score': [f1_score('unique'),
                                 f1_score(new_or_disappeared),
                                 f1_score('split'),
                                 f1_score('merge'),
                                 f1_score('complex')]
                })

            bl_lesions_evaluation_df = get_lesions_evaluation_df('BL')
            fu_lesions_evaluation_df = get_lesions_evaluation_df('FU')
            all_lesions_evaluation_df = get_lesions_evaluation_df('ALL')

            return f"{section('Lesion types evaluation:')}\n\n" \
                   f"{subsection('Prior Lesions Confusion Matrix:' if latex else '--- Prior Lesions Confusion Matrix ---')}\n\n" \
                   f"{center_table(get_conf_mat_str('BL'))}\n\n" \
                   f"{subsection('Prior Lesions Evaluation:' if latex else '--- Prior Lesions Evaluation ---')}\n\n" \
                   f"{center_table(add_line_after_first_column(tabulate(bl_lesions_evaluation_df, headers='keys', tablefmt=TABLEFMT, numalign='center', stralign='center', showindex=False)))}\n\n" \
                   f"{subsection('Current Lesions Confusion Matrix:' if latex else '--- Current Lesions Confusion Matrix ---')}\n\n" \
                   f"{center_table(get_conf_mat_str('FU'))}\n\n" \
                   f"{subsection('Current Lesions Evaluation:' if latex else '--- Current Lesions Evaluation ---')}\n\n" \
                   f"{subsection('Current Lesions Evaluation:' if latex else '--- Current Lesions Evaluation ---')}\n\n" \
                   f"{subsection('Current Lesions Evaluation:' if latex else '--- Current Lesions Evaluation ---')}\n\n" \
                   f"{center_table(add_line_after_first_column(tabulate(fu_lesions_evaluation_df, headers='keys', tablefmt=TABLEFMT, numalign='center', stralign='center', showindex=False)))}\n\n" \
                   f"{subsection('All Lesions Evaluation:' if latex else '--- All Lesions Evaluation ---')}\n\n" \
                   f"{center_table(add_line_after_first_column(tabulate(all_lesions_evaluation_df, headers='keys', tablefmt=TABLEFMT, numalign='center', stralign='center', showindex=False)))}\n" \
                   f"{center_table(get_conf_mat_str('ALL'))}\n\n" \
                   f"{center_table(tabulate(all_lesions_evaluation_df, headers='keys', tablefmt=TABLEFMT, numalign='center', stralign='center', showindex=False))}\n"
                   

        def get_CCs_evaluation():
            data = {
                'unique': {
                    'n_TP': int(delta_CC_types["n_equivalent_unique_CCs"].sum()),
                    'n_GT': int(gt_CC_types["n_unique"].sum()),
                    'n_PRED': int(pred_CC_types["n_unique"].sum())
                },
                'new': {
                    'n_TP': int(delta_CC_types["n_equivalent_new_CCs"].sum()),
                    'n_GT': int(gt_CC_types["n_new_FU"].sum()),
                    'n_PRED': int(pred_CC_types["n_new_FU"].sum())
                },
                'disappeared': {
                    'n_TP': int(delta_CC_types["n_equivalent_disappeared_CCs"].sum()),
                    'n_GT': int(gt_CC_types["n_disappeared_BL"].sum()),
                    'n_PRED': int(pred_CC_types["n_disappeared_BL"].sum())
                },
                'split': {
                    'n_TP': int(delta_CC_types["n_equivalent_split_CCs"].sum()),
                    'n_GT': int(gt_CC_types["n_split_BL"].sum()),
                    'n_PRED': int(pred_CC_types["n_split_BL"].sum())
                },
                'merge': {
                    'n_TP': int(delta_CC_types["n_equivalent_merge_CCs"].sum()),
                    'n_GT': int(gt_CC_types["n_merge_FU"].sum()),
                    'n_PRED': int(pred_CC_types["n_merge_FU"].sum())
                },
                'complex': {
                    'n_TP': int(delta_CC_types["n_equivalent_complex_CCs"].sum()),
                    'n_GT': int(gt_CC_types["n_complex_event"].sum()),
                    'n_PRED': int(pred_CC_types["n_complex_event"].sum())
                },
                'total': {
                    'n_TP': int(delta_CC_types["n_equivalent_CCs"].sum()),
                    'n_GT': int(gt_CC_types["n_CCs"].sum()),
                    'n_PRED': int(pred_CC_types["n_CCs"].sum())
                },
            }

            recall = lambda type: f'{data[type]["n_TP"]/data[type]["n_GT"]:.2f}'
            precision = lambda type: f'{data[type]["n_TP"]/data[type]["n_PRED"]:.2f}'
            f1_score = lambda type: f'{data[type]["n_TP"]/(0.5 * (data[type]["n_GT"] + data[type]["n_PRED"])):.2f}'

            CCs_evaluation_df = pd.DataFrame({
                'CC type': ['Unique', 'New', 'Disappeared', 'Split', 'Merge', 'Complex', 'All CCs'],
                '#TP-CCs': [f"{data['unique']['n_TP']:,}",
                            f"{data['new']['n_TP']:,}",
                            f"{data['disappeared']['n_TP']:,}",
                            f"{data['split']['n_TP']:,}",
                            f"{data['merge']['n_TP']:,}",
                            f"{data['complex']['n_TP']:,}",
                            f"{data['total']['n_TP']:,}"],
                '#CCs-in-GT': [f"{data['unique']['n_GT']:,}",
                               f"{data['new']['n_GT']:,}",
                               f"{data['disappeared']['n_GT']:,}",
                               f"{data['split']['n_GT']:,}",
                               f"{data['merge']['n_GT']:,}",
                               f"{data['complex']['n_GT']:,}",
                               f"{data['total']['n_GT']:,}"],
                '#CCs-in-PRED': [f"{data['unique']['n_PRED']:,}",
                                 f"{data['new']['n_PRED']:,}",
                                 f"{data['disappeared']['n_PRED']:,}",
                                 f"{data['split']['n_PRED']:,}",
                                 f"{data['merge']['n_PRED']:,}",
                                 f"{data['complex']['n_PRED']:,}",
                                 f"{data['total']['n_PRED']:,}"],
                'Recall': [recall('unique'),
                           recall('new'),
                           recall('disappeared'),
                           recall('split'),
                           recall('merge'),
                           recall('complex'),
                           recall('total')],
                'Precision': [precision('unique'),
                              precision('new'),
                              precision('disappeared'),
                              precision('split'),
                              precision('merge'),
                              precision('complex'),
                              precision('total')],
                # 'F1-Score': [f1_score('unique'),
                #              f1_score('new'),
                #              f1_score('disappeared'),
                #              f1_score('split'),
                #              f1_score('merge'),
                #              f1_score('complex'),
                #              f1_score('total')]
            }).set_index('CC type')

            return f"{section('Connected Component (CC) types evaluation:')}\n\n" \
                   f"{center_table(add_line_after_first_column(tabulate(CCs_evaluation_df, headers='keys', tablefmt=TABLEFMT, numalign='center', stralign='center')))}\n"

        def get_cases_evaluation():
            n_pairs = gt_CC_types.shape[0]
            n_perfect_pairs = delta_edges[(delta_edges['n_FP'] == 0) & (delta_edges['n_FN'] == 0)].shape[0]
            n_problematic_pairs = n_pairs - n_perfect_pairs
            cases_partitioning_df = pd.DataFrame({
                '#Pairs': [f'{n_pairs:,}'],
                '#Perfect-pairs\n(no errors)': [f'{n_perfect_pairs:,}\n({100*n_perfect_pairs/n_pairs:.1f}%)'],
                '#Problematic-pairs': [f'{n_problematic_pairs:,}\n({100*n_problematic_pairs/n_pairs:.1f}%)']
            })
            FP_df = delta_edges[(delta_edges['n_FP'] > 0) | (delta_edges['n_FN'] > 0)].groupby('n_FP')['n_FP'].count().reset_index(name='#pairs').rename(columns={'n_FP': '#FP'})[['#pairs', '#FP']]
            FN_df = delta_edges[(delta_edges['n_FP'] > 0) | (delta_edges['n_FN'] > 0)].groupby('n_FN')['n_FN'].count().reset_index(name='#pairs').rename(columns={'n_FN': '#FN'})[['#pairs', '#FN']]
            return f"{section('Cases evaluation:')}\n\n" \
                   f"{subsection('Perfect cases:')}\n\n" \
                   f"{center_table(tabulate(cases_partitioning_df, headers='keys', tablefmt=TABLEFMT, numalign='center', stralign='center', showindex=False))}\n\n" \
                   f"{subsection('Among the problematic cases, the following is the partitioning of the edge-detection errors:')}\n\n" \
                   f"{put_tables_side_by_side(FP_df, FN_df)}\n\n"


        summarization = prefix + sp.join([get_dataset_description(),
                                          get_edges_evaluation(),
                                          get_lesion_evaluation(),
                                          get_CCs_evaluation(),
                                          get_cases_evaluation(),
                                          #get_conf_mat_str('BL'),
                                          #get_conf_mat_str('FU'),
                                          #get_conf_mat_str('All')
                                          ]) + suffix

        print(summarization)

    if results_file is not None:
        resuls_dir_name = os.path.dirname(results_file)
        os.makedirs(resuls_dir_name, exist_ok=True)

        writer = pd.ExcelWriter(results_file, engine='xlsxwriter')

        write_to_excel(gt_CC_types, writer, CC_types_columns, 'name', sheet_name='GT-CC-Types')
        write_to_excel(pred_CC_types, writer, CC_types_columns, 'name', sheet_name='PRED-CC-Types')
        write_to_excel(delta_CC_types, writer, delta_CC_types_columns, 'name', sheet_name='Delta-CC-Types')
        write_to_excel(delta_lesion_types, writer, delta_lesion_types_columns, 'name', sheet_name='Delta-Lesion-Types')
        write_to_excel(delta_edges, writer, delta_edges_columns, 'name', sheet_name='Delta-Edges',
                       f1_scores={'F1 - Score': ('Precision', 'Recall')})
        writer.save()

        if print_summarization:
            with open(replace_in_file_name(results_file, '.xlsx', '.txt', dst_file_exist=False), mode='w') as f:
                f.write(summarization)

    return gt_CC_types, pred_CC_types, delta_CC_types


def get_tumors_statistics_in_a_scan(file_names: Tuple[str, str], selem_radius: int = 1,
                                    th_dist_from_border_in_mm: float = 10) -> \
        Tuple[str, int, float, int, float, float, float, float]:
    """
    Extracting tumors statistics from a scan given the liver and tumors segmentations of the scan.

    :param file_names: a tuple in the following form: (liver_file_name, tumors_file_name).
    :param selem_radius: the size of the radius of the structural element used to extract the liver border.
    :param th_dist_from_border_in_mm: the threshold of distance from liver border (in MM).

    :return: a tuple in the following form: (case_name, n_tumors, total_tumor_volume, n_tumors_close_to_liver_border,
        mean_tumor_diameter, std_tumor_diameter, min_tumor_diameter, max_tumor_diameter) where:
        • case_name is the name of the case.
        • n_tumors is the number of tumors in the scan.
        • total_tumor_volume is the total volume of all the tumors in the scan, in CC.
        • n_tumors_close_to_liver_border is the number of tumors in the scan that are close to the liver border
            (according to the given parameter 'th_dist_from_border_in_mm').
        • mean_tumor_diameter is the mean of all the tumors diameters, in mm.
        • std_tumor_diameter is the STD of all the tumors diameters, in mm.
        • min_tumor_diameter is the minimum of all the tumors diameters, in mm.
        • max_tumor_diameter is the maximum of all the tumors diameters, in mm.
    """

    # extracting the tumors measures
    case_name, tumors_measures = calculate_tumors_measures(file_names, selem_radius=selem_radius)

    if not tumors_measures:
        return case_name, 0, 0.0, 0, np.nan, np.nan, np.nan, np.nan

    tumors_measures = np.array(tumors_measures)

    n_tumors = tumors_measures.shape[0]
    total_tumor_volume = tumors_measures[:, 0].sum()
    n_tumors_close_to_liver_border = tumors_measures[:, 4].sum()
    # n_tumors_close_to_liver_border = (tumors_measures[:, 2] <= th_dist_from_border_in_mm).sum()
    tumors_diameters = tumors_measures[:, 1]
    mean_tumor_diameter = tumors_diameters.mean()
    std_tumor_diameter = tumors_diameters.std()
    min_tumor_diameter = tumors_diameters.min()
    max_tumor_diameter = tumors_diameters.max()

    return (case_name, n_tumors, total_tumor_volume, n_tumors_close_to_liver_border,
            mean_tumor_diameter, std_tumor_diameter, min_tumor_diameter, max_tumor_diameter)


def write_scans_tumors_statistics(livers_files: List[str], tumors_files: List[str], n_processes=None,
                                  results_file: str = 'measures_results/scans_tumors_statistics.xlsx'):
    if n_processes is None:
        n_processes = os.cpu_count()

    th_dist_from_border_in_mm = 5
    livers_and_tumors_file_names = zip(livers_files, tumors_files)

    with Pool(n_processes) as pool:
        scans_tumors_statistics = pool.map(
            partial(get_tumors_statistics_in_a_scan, th_dist_from_border_in_mm=th_dist_from_border_in_mm),
            livers_and_tumors_file_names)

    columns = ['file_names', 'number_of_tumors', 'total_tumor_volume (CC)',
               f'number_of_tumors_in_{th_dist_from_border_in_mm}_mm_from_liver_border', 'mean_tumor_diameter',
               'std_tumor_diameter', 'min_tumor_diameter', 'max_tumor_diameter']
    df_results = pd.DataFrame(data=scans_tumors_statistics, columns=columns)

    resuls_dir_name = os.path.dirname(results_file)
    os.makedirs(resuls_dir_name, exist_ok=True)

    writer = pd.ExcelWriter(results_file, engine='xlsxwriter')

    write_to_excel(df_results, writer, columns, 'file_names')
    writer.save()


def write_pairs_changes(BL_tumors_files: List[str], FU_tumors_files: List[str], n_processes=None,
                        results_file: Optional[str] = 'measures_results/pairs_changes.xlsx') -> pd.DataFrame:
    if n_processes is None:
        n_processes = os.cpu_count()

    BL_tumors_and_FU_tumors_file_names = zip(BL_tumors_files, FU_tumors_files)

    # with Pool(n_processes) as pool:
    #     pairs_changes = pool.map(calculate_change_in_pairs, BL_tumors_and_FU_tumors_file_names)
    #     # pairs_changes = map(calculate_change_in_pairs, BL_tumors_and_FU_tumors_file_names)
    pairs_changes = process_map(calculate_change_in_pairs, list(BL_tumors_and_FU_tumors_file_names), max_workers=n_processes)
    pairs_changes = [c[1:] for c in pairs_changes]

    columns = ['name', 'BL_n_tumors', 'FU_n_tumors', 'BL_total_tumors_volume (CC)', 'FU_total_tumors_volume (CC)',
               'volume_diff (FU-BL (CC))', 'abs_volume_diff (|FU-BL| (CC))', 'volume_change ((FU-BL)/BL (%))',
               'n_unique', 'n_disappear_BL', 'n_split_BL', 'n_merge_BL', 'n_complex_BL', 'n_new_FU', 'n_split_FU',
               'n_merge_FU', 'n_complex_FU', 'n_complex_event', 'contains_merge', 'contains_split',
               'contains_merge_or_split', 'contains_merge_and_split', 'contains_complex']
    df_results = pd.DataFrame(data=pairs_changes, columns=columns)

    if results_file is not None:
        resuls_dir_name = os.path.dirname(results_file)
        os.makedirs(resuls_dir_name, exist_ok=True)

        writer = pd.ExcelWriter(results_file, engine='xlsxwriter')

        write_to_excel(df_results, writer, columns, 'pairs_names')
        writer.save()

    return df_results


def write_tumors_measures(scan_paths: List[str], liver_paths: List[str], tumors_path: List[str], n_processes=None,
                          results_file: str = 'measures_results/tumors_measures.xlsx'):
    if n_processes is None:
        n_processes = os.cpu_count()

    results = process_map(calculate_tumors_measures, list(zip(scan_paths, liver_paths, tumors_path)), max_workers=n_processes)
    # results = list(map(calculate_tumors_measures, zip(scan_paths, liver_paths, tumors_path)))

    results = [('_-_'.join(item[0]), *item[1]) for tup in
               (tuple(zip(zip([t[0]] * len(t[1]), (str(j) for j in range(1, len(t[1]) + 1))), t[1])) for t in results)
               for item in tup]

    columns = ['tumor_ID', 'tumor_volume (CC)', 'tumor_diameter (mm)', 'tumor_dist_from_liver_border (mm)',
               'tumor_center_dist_from_liver_border (mm)', 'is_close_to_border', 'touches_the_border',
               'liver_segment', 'mean_tumor_diff_from_mean_liver (HU)', 'dice_with_approximate_sphere']
    df_results = pd.DataFrame(data=results, columns=columns)
    # tumor_dist_th = 5
    # tumor_center_dist_th = 10
    # df_results[f'tumor_in_{tumor_dist_th}_mm_from_liver_border'] = np.array(
    #     df_results['tumor_dist_from_liver_border (mm)'] <= tumor_dist_th).astype(np.int)
    # df_results[f'tumor_center_in_{tumor_center_dist_th}_mm_from_liver_border'] = np.array(
    #     df_results['tumor_center_dist_from_liver_border (mm)'] <= tumor_center_dist_th).astype(np.int)

    resuls_dir_name = os.path.dirname(results_file)
    os.makedirs(resuls_dir_name, exist_ok=True)

    writer = pd.ExcelWriter(results_file, engine='xlsxwriter')

    # columns_order = ['tumor_ID', 'tumor_volume (CC)', 'tumor_diameter (mm)', 'liver_segment',
    #                  'tumor_dist_from_liver_border (mm)', f'tumor_in_{tumor_dist_th}_mm_from_liver_border',
    #                  'tumor_center_dist_from_liver_border (mm)',
    #                  f'tumor_center_in_{tumor_center_dist_th}_mm_from_liver_border']
    write_to_excel(df_results, writer, columns, 'tumor_ID')
    writer.save()


def write_final_excel_files_over_data_set():
    train_set_data_path = '/cs/casmip/rochman/Errors_Characterization/data/train_pairs_after_validation_splitting.xlsx'
    validation_set_data_path = '/cs/casmip/rochman/Errors_Characterization/data/validation_pairs.xlsx'
    test_set_data_path = '/cs/casmip/rochman/Errors_Characterization/data/test_pairs_saved.xlsx'

    train_set_data = pd.read_excel(train_set_data_path).rename(columns={'Unnamed: 0': 'name'})
    validation_set_data = pd.read_excel(validation_set_data_path).rename(columns={'Unnamed: 0': 'name'})
    test_set_data = pd.read_excel(test_set_data_path).rename(columns={'Unnamed: 0': 'name'})

    stats = ['mean', 'std', 'min', 'max', 'sum']

    train_set_data = train_set_data[~train_set_data['name'].isin(stats)]
    validation_set_data = validation_set_data[~validation_set_data['name'].isin(stats)]
    test_set_data = test_set_data[~test_set_data['name'].isin(stats)]
    train_and_validation_set_data = pd.concat([train_set_data, validation_set_data]).reset_index(drop=True)
    all_data = pd.concat([train_and_validation_set_data, test_set_data])

    def sort_dataframe_by_key(dataframe: pd.DataFrame, column: str, key: Callable) -> pd.DataFrame:
        """ Sort a dataframe from a column using the key """
        sort_ixs = sorted(np.arange(len(dataframe)), key=lambda i: key(dataframe.iloc[i][column]))
        return pd.DataFrame(columns=list(dataframe), data=dataframe.iloc[sort_ixs].values)

    train_set_data = sort_dataframe_by_key(train_set_data, column='name', key=pairs_sort_key)
    test_set_data = sort_dataframe_by_key(test_set_data, column='name', key=pairs_sort_key)
    validation_set_data = sort_dataframe_by_key(validation_set_data, column='name', key=pairs_sort_key)
    train_and_validation_set_data = sort_dataframe_by_key(train_and_validation_set_data, column='name',
                                                          key=pairs_sort_key)
    all_data = sort_dataframe_by_key(all_data, column='name', key=pairs_sort_key)

    all_data_dir = '/mnt/sda1/aszeskin/Data_Followup_Full_29_4_2021'
    BL_tumors_files = [f'{all_data_dir}/{pair_name}/BL_Scan_Tumors.nii.gz' for pair_name in all_data['name']]
    FU_tumors_files = [f'{all_data_dir}/{pair_name}/FU_Scan_Tumors.nii.gz' for pair_name in all_data['name']]

    # todo delete this statement
    # start = 0
    # end = 10
    # BL_tumors_files, FU_tumors_files = BL_tumors_files[start: end], FU_tumors_files[start: end]
    # x = BL_tumors_files + FU_tumors_files
    # x.remove('/mnt/sda1/aszeskin/Data_Followup_Full_29_4_2021/BL_A_Ab_15_07_2018_FU_A_Ab_03_10_2018/FU_Scan_Tumors.nii.gz')
    # from tqdm import tqdm
    # for f in tqdm(x):
    #     try:
    #         load_nifti_data(f)
    #     except Exception as e:
    #         print(f'The problematic file is: {f}')
    #         raise e
    # exit(0)

    all_data_pairs_data_measures = write_pairs_changes(BL_tumors_files, FU_tumors_files,
                                                       n_processes=None, results_file=None)

    def write_individual_scans_measures(data: pd.DataFrame, result_file: str):
        bl_names, fu_names = zip(*[pair_name.replace('BL_', '').split('_FU_') for pair_name in data['name'] if
                                   pair_name.startswith('BL')])
        data['FU_name'] = fu_names
        relevant_columns = ['FU_name', 'FU_number_of_tumors', 'fu_total_tumor_volume (CC)', 'fu_mean_tumor_diameter',
                            'fu_std_tumor_diameter', 'fu_min_tumor_diameter', 'fu_max_tumor_diameter']
        data = data[relevant_columns].copy()
        data.rename(columns=lambda c: c.replace('FU_', '').replace('fu_', ''), inplace=True)
        data.drop_duplicates(inplace=True, ignore_index=True)
        data = sort_dataframe_by_key(data, column='name', key=scans_sort_key)

        resuls_dir_name = os.path.dirname(result_file)
        os.makedirs(resuls_dir_name, exist_ok=True)

        writer = pd.ExcelWriter(result_file, engine='xlsxwriter')

        write_to_excel(data, writer, [c.replace('FU_', '').replace('fu_', '') for c in relevant_columns], 'name')
        writer.save()

    def write_pairs_data_measures(data: pd.DataFrame, result_file: str):

        # relevant_columns = ['name', 'BL_number_of_tumors', 'FU_number_of_tumors', 'volume_change (FU/BL)',
        #                     'n_exist_tumors', 'n_brand_new_tumors', 'n_new_tumors', 'n_brand_disappear_tumors',
        #                     'n_disappear_tumors', 'n_merges', 'n_splits', 'is_complicated']
        relevant_columns = list(data.columns)
        # data = data[relevant_columns].copy()

        resuls_dir_name = os.path.dirname(result_file)
        os.makedirs(resuls_dir_name, exist_ok=True)

        writer = pd.ExcelWriter(result_file, engine='xlsxwriter')

        write_to_excel(data, writer, relevant_columns, 'name')
        writer.save()

    def write_pairs_registration_measures(data: pd.DataFrame, result_file: str):
        relevant_columns = ['name', 'dice', 'assd', 'hd', 'abs_liver_diff', ]
        data = data[relevant_columns].copy()

        resuls_dir_name = os.path.dirname(result_file)
        os.makedirs(resuls_dir_name, exist_ok=True)

        writer = pd.ExcelWriter(result_file, engine='xlsxwriter')

        write_to_excel(data, writer, relevant_columns, 'name')
        writer.save()

    for df, df_name in [(train_set_data, 'train_set'), (test_set_data, 'test_set'),
                        (validation_set_data, 'validation_set'),
                        (train_and_validation_set_data, 'train_and_validation_set'), (all_data, 'all_data')]:
        write_individual_scans_measures(df, f'data/{df_name}_individual_scans_data_measures.xlsx')
        write_pairs_registration_measures(df, f'data/{df_name}_pairs_registration_measures.xlsx')

        if df_name == 'all_data':
            df = all_data_pairs_data_measures
        else:
            df = all_data_pairs_data_measures[all_data_pairs_data_measures['name'].isin(df['name'])]

        write_pairs_data_measures(df, f'data/{df_name}_pairs_data_measures.xlsx')


def calculate_matching_between_GT_and_pred(file_names: Tuple[str, str, str]):
    """
    Calculating changes in the tumors between GT tumors segmentation and predicted segmentation.

    :param file_names: a tuple in the following form: (liver_file_name, GT_tumors_file_name, pred_tumors_file_name).

    :return: a list that contains for each connected component (CC) in the bipartite matching graph a tuple in the following
        form: (CC_id, GT_n_tumors, pred_n_tumors, max_GT_diameter, max_pred_diameter, GT_total_tumors_volume,
               pred_total_tumors_volume, is_TP, is_FN, is_FP, is_split, is_merge, is_complex) where:
        • CC_id is an ID in the following form:
            case_name_-_(GT_center_of_mass)_-_(pred_center_of_mass).
        • GT_n_tumors is the number of GT tumors participating in the current CC.
        • pred_n_tumors is the number of pred tumors participating in the current CC.
        • max_GT_diameter is the diameter of the biggest GT tumor in the current CC.
        • max_pred_diameter is the diameter of the biggest pred tumor in the current CC.
        • GT_total_tumors_volume is the total tumors volume of the GT tumors in the current CC.
        • pred_total_tumors_volume is the total tumors volume of the pred tumors in the current CC.
        • is_TP is an indicator that indicates whether the current CC is a TP case. A CC is a TP case when the number of
            GT and pred tumors in the CC are both 1.
        • is_FN is an indicator that indicates whether the current CC is a FN case. A CC is a FN case when the number of
            GT and pred tumors in the CC are both 1 and 0 respectively.
        • is_FP is an indicator that indicates whether the current CC is a FP case. A CC is a FP case when the number of
            GT and pred tumors in the CC are both 0 and 1 respectively.
        • is_split is an indicator that indicates whether the current CC is a split case. A CC is a split case when the
            number of GT tumors in the CC is 1 and the number of pred tumors in the CC is at least 2.
        • is_merge is an indicator that indicates whether the current CC is a merge case. A CC is a merge case when the
            number of GT tumors in the CC is at least 2 and the number of pred tumors in the CC is 1.
        • is_complex is an indicator that indicates whether the current CC is a complex case. A CC is a complex case
            when the number of GT and pred tumors in the CC are both at least 2.
    """

    get_case_name = lambda file_name: basename(dirname(file_name))
    assert get_case_name(file_names[0]) == get_case_name(file_names[1])
    assert get_case_name(file_names[0]) == get_case_name(file_names[2])
    case_name = get_case_name(file_names[0])

    # loading the files
    (liver_case, nifti_file), (gt_tumors_case, _), (pred_tumors_case, _) = (load_nifti_data(file_name) for file_name in file_names)

    assert is_a_mask(liver_case)
    assert is_a_mask(gt_tumors_case)
    assert is_a_mask(pred_tumors_case)

    # pre-process the GT tumors and liver segmentations
    gt_tumors_case = pre_process_segmentation(gt_tumors_case)
    liver_case = np.logical_or(liver_case, gt_tumors_case).astype(liver_case.dtype)
    liver_case = getLargestCC(liver_case)
    liver_case = pre_process_segmentation(liver_case, remove_small_obs=False)
    gt_tumors_case = np.logical_and(liver_case, gt_tumors_case).astype(gt_tumors_case.dtype)

    # correcting to ROI the pred tumors case
    pred_tumors_case = np.logical_and(pred_tumors_case, liver_case)

    # preprocessing the pred tumors case
    pred_tumors_case = pre_process_segmentation(pred_tumors_case)

    pix_dims = nifti_file.header.get_zooms()
    voxel_volume = pix_dims[0] * pix_dims[1] * pix_dims[2]

    gt_tumors_labels = get_connected_components(gt_tumors_case)
    pred_tumors_labels = get_connected_components(pred_tumors_case)

    # extracting the matches between the tumors in the cases (GT and pred)
    matches = match_2_cases(gt_tumors_labels, pred_tumors_labels)

    # extracting the GT labels
    GT_labels = np.unique(gt_tumors_labels)
    GT_labels = GT_labels[GT_labels != 0]
    GT_n_tumors = GT_labels.size

    # extracting the pred labels
    pred_labels = np.unique(pred_tumors_labels)
    pred_labels = pred_labels[pred_labels != 0] + GT_n_tumors
    pred_n_tumors = pred_labels.size

    V = list(GT_labels - 1) + list(pred_labels - 1)
    visited = [False] * len(V)
    adjacency_lists = []
    for _ in range(GT_n_tumors + pred_n_tumors):
        adjacency_lists.append([])
    for (gt_v, pred_v) in matches:
        pred_v += GT_n_tumors - 1
        gt_v -= 1
        adjacency_lists[gt_v].append(pred_v)
        adjacency_lists[pred_v].append(gt_v)

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

    is_gt_tumor = lambda v: v <= GT_n_tumors - 1

    def gt_list_and_pred_list(CC):
        gt_in_CC = []
        pred_in_CC = []
        for v in CC:
            if is_gt_tumor(v):
                gt_in_CC.append(v)
            else:
                pred_in_CC.append(v)
        return np.array(gt_in_CC) + 1, np.array(pred_in_CC) + 1 - GT_n_tumors

    def get_current_CC_case(tumors, label_case):
        return np.isin(label_case, tumors).astype(label_case.dtype)

    def get_max_diameter(tumors, label_case):
        return max(approximate_diameter((label_case == v).sum() * voxel_volume) for v in tumors)

    res = []

    while len(V) > 0:
        is_TP, is_FN, is_FP, is_split, is_merge, is_complex = [0]*6

        v = V[0]
        current_CC = DFS(v)

        gt_in_CC, pred_in_CC = gt_list_and_pred_list(current_CC)
        n_GT_tumors = gt_in_CC.size
        n_pred_tumors = pred_in_CC.size

        current_CC_gt_case = get_current_CC_case(gt_in_CC, gt_tumors_labels) if n_GT_tumors > 0 else None
        current_CC_pred_case = get_current_CC_case(pred_in_CC, pred_tumors_labels) if n_pred_tumors > 0 else None

        CC_id = f'{case_name}_-_{get_center_of_mass(current_CC_gt_case) if n_GT_tumors > 0 else "()"}_-_{get_center_of_mass(current_CC_pred_case) if n_pred_tumors > 0 else "()"}'

        max_GT_diameter = get_max_diameter(gt_in_CC, gt_tumors_labels) if n_GT_tumors > 0 else np.nan
        max_pred_diameter = get_max_diameter(pred_in_CC, pred_tumors_labels) if n_pred_tumors > 0 else np.nan

        GT_total_tumors_volume = current_CC_gt_case.sum() * voxel_volume / 1000 if n_GT_tumors > 0 else np.nan
        pred_total_tumors_volume = current_CC_pred_case.sum() * voxel_volume / 1000 if n_pred_tumors > 0 else np.nan

        # in case the current tumor is a single tumor in a connected component
        if len(current_CC) == 1:
            if is_gt_tumor(v):
                is_FN = 1
            else:
                is_FP = 1

        # in case the current tumor is in a connected component with only two tumors, namely it is TP case
        elif len(current_CC) == 2:
            is_TP = 1

        else:

            # in case of a complex event
            if n_GT_tumors > 1 and n_pred_tumors > 1:
                is_complex = 1
            else:
                # in case of a split
                if n_GT_tumors == 1:
                    is_split = 1
                # in case of a merge
                else:
                    is_merge = 1

        res.append((CC_id, n_GT_tumors, n_pred_tumors, max_GT_diameter, max_pred_diameter, GT_total_tumors_volume,
                    pred_total_tumors_volume, is_TP, is_FN, is_FP, is_split, is_merge, is_complex))

    return res


def write_matching_measures_between_GT_and_pred(liver_files: List[str], gt_tumors_files: List[str],
                                                pred_tumors_files: List[str], n_processes=None,
                                                results_file: Optional[str] = 'measures_results/matching_between_GT_and_pred.xlsx'):

    if n_processes is None:
        n_processes = os.cpu_count()

    file_names = list(zip(liver_files, gt_tumors_files, pred_tumors_files))

    matching_measures = process_map(calculate_matching_between_GT_and_pred, file_names, max_workers=n_processes)

    columns = ['CC_id', 'GT_n_tumors', 'pred_n_tumors', 'max_GT_diameter', 'max_pred_diameter', 'GT_total_tumors_volume',
               'pred_total_tumors_volume (CC)', 'is_TP', 'is_FN', 'is_FP', 'is_split', 'is_merge', 'is_complex']
    df_results = pd.concat([pd.DataFrame(res, columns=columns) for res in matching_measures])

    if results_file is not None:
        resuls_dir_name = os.path.dirname(results_file)
        os.makedirs(resuls_dir_name, exist_ok=True)

        writer = pd.ExcelWriter(results_file, engine='xlsxwriter')

        write_to_excel(df_results, writer, columns, 'CC_id', sheet_name='all diameters')
        write_to_excel(df_results[df_results['max_GT_diameter'] <= 5], writer, columns, 'CC_id',
                       sheet_name='max_GT_diameter <= 5')
        write_to_excel(df_results[(5 < df_results['max_GT_diameter']) & (df_results['max_GT_diameter'] <= 10)], writer, columns,
                       'CC_id', sheet_name='5 < max_GT_diameter <= 10')
        write_to_excel(df_results[(10 < df_results['max_GT_diameter']) & (df_results['max_GT_diameter'] <= 15)], writer, columns,
                       'CC_id', sheet_name='10 < max_GT_diameter <= 15')
        write_to_excel(df_results[(15 < df_results['max_GT_diameter']) & (df_results['max_GT_diameter'] <= 20)], writer, columns,
                       'CC_id', sheet_name='15 < max_GT_diameter <= 20')
        write_to_excel(df_results[(20 < df_results['max_GT_diameter']) & (df_results['max_GT_diameter'] <= 30)], writer, columns,
                       'CC_id', sheet_name='20 < max_GT_diameter <= 30')
        write_to_excel(df_results[30 < df_results['max_GT_diameter']], writer, columns, 'CC_id',
                       sheet_name='30 < max_GT_diameter')
        writer.save()

    return df_results


def triples(dataset_path: str):

    dataset_pairs = pd.read_excel(dataset_path)
    # dataset_pairs = list(dataset_pairs[~dataset_pairs['Unnamed: 0'].isin(['mean', 'std', 'min', 'max', 'sum'])]['Unnamed: 0'])

    bl_names, fu_names = zip(*[pair_name.replace('BL_', '').split('_FU_') for pair_name in dataset_pairs['Unnamed: 0'] if
                               pair_name.startswith('BL')])

    patient_names = ['_'.join(c for c in name.split('_') if not c.isdigit()) for name in bl_names]

    pairs_data = pd.DataFrame({'patient': patient_names, 'bl': bl_names, 'fu': fu_names})

    fu_data = pairs_data[['bl', 'fu']].groupby(['fu']).count()
    fu_data = fu_data[fu_data['bl'] > 1]
    fu_data.reset_index(level=0, inplace=True)
    fu_data.rename(columns={'bl': 'n_triples'}, inplace=True)
    fu_data['n_triples'] *= (fu_data['n_triples'] - 1)
    fu_data['patient'] = ['_'.join(c for c in fu.split('_') if not c.isdigit()) for fu in fu_data['fu']]

    triples_data = fu_data[['patient', 'n_triples']].groupby('patient').sum()

    return (triples_data.shape[0], triples_data['n_triples'].sum(), triples_data['n_triples'].mean(),
            triples_data['n_triples'].std(), triples_data['n_triples'].median(), triples_data['n_triples'].min(),
            triples_data['n_triples'].max())


def observer_variability(file_names: Tuple[str, str], diameter_th) -> Tuple[str, float, float, float, int, int, int]:
    get_case_name = lambda file_name: basename(dirname(file_name))
    case_name = get_case_name(file_names[0])
    assert case_name == get_case_name(file_names[1])

    # loading the files
    (old_case, nifti_file), (new_case, _)= (load_nifti_data(file_name) for file_name in file_names)

    assert is_a_mask(old_case)
    assert is_a_mask(new_case)

    old_case_labeled_CCs = get_connected_components(old_case)
    new_case_labeled_CCs = get_connected_components(new_case)

    pix_dims = nifti_file.header.get_zooms()
    voxel_volume = pix_dims[0] * pix_dims[1] * pix_dims[2]


    def threshold_tumors_by_diameter(labeled_case):
        tumor_and_diameter = np.stack(np.unique(labeled_case[labeled_case != 0], return_counts=True)).T.astype(np.float)
        tumor_and_diameter[:, 1] = approximate_diameter(tumor_and_diameter[:, 1] * voxel_volume)
        tumor_and_diameter = tumor_and_diameter[tumor_and_diameter[:, 1] >= diameter_th]
        return np.where(np.isin(labeled_case, tumor_and_diameter[:, 0]), labeled_case, 0)

    # thresholding the diameters
    old_case_labeled_CCs = threshold_tumors_by_diameter(old_case_labeled_CCs)
    new_case_labeled_CCs = threshold_tumors_by_diameter(new_case_labeled_CCs)

    # extract_TPs
    pairs_of_intersection = np.hstack([old_case_labeled_CCs.reshape([-1, 1]), new_case_labeled_CCs.reshape([-1, 1])])
    pairs_of_intersection = np.unique(pairs_of_intersection[~np.any(pairs_of_intersection == 0, axis=1)], axis=0)
    old_TPs = pairs_of_intersection[:, 0]
    new_TPs = pairs_of_intersection[:, 1]

    n_TPs = old_TPs.size
    n_FPs = np.unique(np.where(np.isin(old_case_labeled_CCs, old_TPs), 0, old_case_labeled_CCs)).size - 1
    n_FNs = np.unique(np.where(np.isin(new_case_labeled_CCs, new_TPs), 0, new_case_labeled_CCs)).size - 1

    dce = dice(np.isin(old_case_labeled_CCs, old_TPs), np.isin(new_case_labeled_CCs, new_TPs))
    if n_TPs + n_FPs != 0:
        precision = n_TPs/(n_TPs + n_FPs)
    else:
        precision = 1

    if n_TPs + n_FNs != 0:
        recall = n_TPs/(n_TPs + n_FNs)
    else:
        recall = 1

    return case_name, dce, precision, recall, n_TPs, n_FPs, n_FNs


def write_observer_variability(old_files: List[str], new_files: List[str], n_processes=None,
                               results_file: Optional[str] = 'measures_results/observer_variability.xlsx'):

    if n_processes is None:
        n_processes = os.cpu_count() - 2

    file_names = list(zip(old_files, new_files))

    observ_var_0 = process_map(partial(observer_variability, diameter_th=0), file_names, max_workers=n_processes)
    observ_var_3 = process_map(partial(observer_variability, diameter_th=3), file_names, max_workers=n_processes)
    observ_var_5 = process_map(partial(observer_variability, diameter_th=5), file_names, max_workers=n_processes)
    observ_var_10 = process_map(partial(observer_variability, diameter_th=10), file_names, max_workers=n_processes)
    # observ_var_0 = list(map(partial(observer_variability, diameter_th=0), file_names))
    # observ_var_3 = list(map(partial(observer_variability, diameter_th=3), file_names))
    # observ_var_5 = list(map(partial(observer_variability, diameter_th=5), file_names))
    # observ_var_10 = list(map(partial(observer_variability, diameter_th=10), file_names))

    columns = ['case name', 'Dice', 'Precision', 'Recall', 'n_TPs', 'n_FPs', 'n_FNs']
    df_results_0 = pd.DataFrame(observ_var_0, columns=columns)
    df_results_3 = pd.DataFrame(observ_var_3, columns=columns)
    df_results_5 = pd.DataFrame(observ_var_5, columns=columns)
    df_results_10 = pd.DataFrame(observ_var_10, columns=columns)


    resuls_dir_name = os.path.dirname(results_file)
    os.makedirs(resuls_dir_name, exist_ok=True)

    writer = pd.ExcelWriter(results_file, engine='xlsxwriter')

    write_to_excel(df_results_0, writer, columns, 'case name', sheet_name='all diameters')
    write_to_excel(df_results_3, writer, columns, 'case name', sheet_name='diameter >= 3')
    write_to_excel(df_results_5, writer, columns, 'case name', sheet_name='diameter >= 5')
    write_to_excel(df_results_10, writer, columns, 'case name', sheet_name='diameter >= 10')
    writer.save()


def vol_diff_between_GT_and_pred_for_uniques(file_names: Tuple[str, str, str]) -> List[Tuple[str, float, float,
                                                                                             float, float, float,
                                                                                             float, float, float,
                                                                                             float, float, float,
                                                                                             float, int]]:
    """
    Calculating the volume difference for each TP (in detection terms) tumor in a scan between the GT and Prediction.

    :param file_names: a tuple in the following form: (liver_file_path, GT_tumors_file_path, pred_tumors_file_path)

    :return: a list containing for each TP tumor a tuple in the following form: (case_name_and_tumor_id, GT_diameter,
        pred_diameter, gt_pred_dice, gt_pred_assd, gt_pred_hd, GT_volume, pred_volume, delta_volume, abs_delta_volume,
        percentage_delta_volume, abs_percentage_delta_volume, max_error_boundary, min_error_boundary, is_in_boundary) where:
        • case_name_and_tumor_id is the name of the case and the ID of the tumor.
        • GT_diameter is the approximated diameter of the GT tumor, in mm.
        • pred_diameter is the approximated diameter of the predicted tumor, in mm.
        • gt_pred_dice is the dice coefficient score between the GT and predicted tumor.
        • gt_pred_assd is the ASSD distance between the GT and predicted tumor, in mm.
        • gt_pred_hd is the Hausdorff distance between the GT and predicted tumor, in mm.
        • GT_volume is the volume of the GT tumor, in CC.
        • pred_volume is the volume of the predicted tumor, in CC.
        • delta_volume is the delta between the GT tumor and the predicted tumor, in CC.
        • abs_delta_volume is the absolute value of the delta between the GT tumor and the predicted tumor, in CC.
        • percentage_delta_volume is the percentage delta between the GT tumor and the predicted tumor, in CC.
        • abs_percentage_delta_volume is the absolute value of the percentage delta between the GT tumor and the
            predicted tumor, in CC.
        • max_error_boundary is the maximum volume error boundary, in CC.
        • min_error_boundary is the minimum volume error boundary, in CC.
        • is_in_boundary is 1 if pred_volume is in between the boundaries of the error, otherwise 0.
    """

    get_case_name = lambda file_name: basename(dirname(file_name))
    case_name = get_case_name(file_names[0])
    assert case_name == get_case_name(file_names[1])
    assert case_name == get_case_name(file_names[2])

    # loading the files
    (liver_case, nifti_file), (gt_tumors_case, _), (pred_tumors_case_label, _) = (load_nifti_data(file_name) for file_name in
                                                                            file_names)

    assert is_a_mask(liver_case)
    assert is_a_mask(gt_tumors_case)
    # assert is_a_mask(pred_tumors_case)

    # pre-process the GT tumors and liver segmentations
    gt_tumors_case = pre_process_segmentation(gt_tumors_case)
    liver_case = np.logical_or(liver_case, gt_tumors_case).astype(liver_case.dtype)
    liver_case = getLargestCC(liver_case)
    liver_case = pre_process_segmentation(liver_case, remove_small_obs=False)
    gt_tumors_case = np.logical_and(liver_case, gt_tumors_case).astype(gt_tumors_case.dtype)

    # correcting to ROI the pred tumors case
    pred_tumors_case_label = (pred_tumors_case_label * liver_case).astype(pred_tumors_case_label.dtype)

    res = []
    for th in range(1, 17):

        # thresholding the pred tumors case
        pred_tumors_case = (pred_tumors_case_label >= th).astype(pred_tumors_case_label.dtype)

        # preprocessing the pred tumors case
        pred_tumors_case = pre_process_segmentation(pred_tumors_case)

        pix_dims = nifti_file.header.get_zooms()
        voxel_volume = pix_dims[0] * pix_dims[1] * pix_dims[2]

        gt_tumors_labels = get_connected_components(gt_tumors_case)
        pred_tumors_labels = get_connected_components(pred_tumors_case)

        # extracting the TP tumors
        pairs = np.hstack([gt_tumors_labels.reshape([-1, 1]), pred_tumors_labels.reshape([-1, 1])])
        pairs = np.unique(pairs[~np.any(pairs == 0, axis=1)], axis=0)

        # extracting only the tumors with a unique match between GT and pred
        # unique_gt = np.stack(np.unique(pairs[:, 0], return_counts=True)).T
        # unique_gt = unique_gt[unique_gt[:, 1] == 1][:, 0]
        # unique_pred = np.stack(np.unique(pairs[:, 1], return_counts=True)).T
        # unique_pred = unique_pred[unique_pred[:, 1] == 1][:, 0]
        # pairs = pairs[np.isin(pairs[:, 0], unique_gt)]
        # pairs = pairs[np.isin(pairs[:, 1], unique_pred)]

        gt_intersections = []
        previous_gt = None
        for k, gt in enumerate(pairs[:, 0]):
            if previous_gt is not None and (gt == previous_gt[0]):
                previous_gt = (gt, previous_gt[1] + [int(pairs[k, 1])])
                gt_intersections[-1] = previous_gt
            else:
                previous_gt = (gt, [int(pairs[k, 1])])
                gt_intersections.append(previous_gt)


        for (gt, preds) in gt_intersections:
            current_GT_tumor = np.where(gt_tumors_labels == gt, 1, 0)
            # current_pred_tumor = np.where(pred_tumors_labels == pred, 1, 0)
            current_pred_tumor = np.where(np.isin(pred_tumors_labels, preds), 1, 0)

            gt_pred_dice = dice(current_GT_tumor, current_pred_tumor)
            gt_pred_assd, gt_pred_hd = assd_and_hd(current_GT_tumor, current_pred_tumor, voxelspacing=pix_dims)
            GT_volume = current_GT_tumor.sum() * voxel_volume
            GT_diameter = approximate_diameter(GT_volume)
            GT_volume /= 1000
            pred_volume = current_pred_tumor.sum() * voxel_volume
            pred_diameter = approximate_diameter(pred_volume)
            pred_volume /= 1000
            delta_volume = pred_volume - GT_volume
            abs_delta_volume = abs(delta_volume)
            percentage_delta_volume = 100 * delta_volume / (GT_volume + pred_volume)
            abs_percentage_delta_volume = abs(percentage_delta_volume)
            # min_error_boundary, max_error_boundary = get_min_max_volume_error(current_GT_tumor, voxel_volume,
            #                                                             observe_variability_of_dice=0.8)
            # is_in_boundary = 1 if (max_error_boundary >= pred_volume >= min_error_boundary) else 0
            error_boundary = 100 * np.sum(pix_dims) / (GT_diameter/2)
            min_error_boundary = -error_boundary - 24
            max_error_boundary = error_boundary + 27
            is_in_boundary = 1 if (max_error_boundary >= percentage_delta_volume >= min_error_boundary) else 0
            case_name_and_tumor_id_and_th = f'{case_name} - {gt} - {th}'

            res.append((case_name_and_tumor_id_and_th, GT_diameter, pred_diameter, gt_pred_dice, gt_pred_assd, gt_pred_hd,
                        GT_volume, pred_volume, delta_volume, abs_delta_volume, percentage_delta_volume,
                        abs_percentage_delta_volume, max_error_boundary, min_error_boundary, is_in_boundary))

    return res


from skimage.morphology import binary_dilation, binary_erosion, ball
def get_min_max_volume_error(tumor: np.ndarray, voxel_volume: float, observe_variability_of_dice: float) -> Tuple[float, float]:

    def find_boundary(type):
        morph_func = binary_dilation if type == 'max' else binary_erosion
        boundary_tumor_a = tumor
        boundary_tumor_b = morph_func(tumor, disk(1).reshape([3, 3, 1]))
        dice_a = 1
        dice_b = dice(tumor, boundary_tumor_b)
        i = 2
        while dice_b >= observe_variability_of_dice:
            boundary_tumor_a, dice_a = boundary_tumor_b, dice_b
            boundary_tumor_b = morph_func(tumor, disk(i).reshape([2*i + 1, 2*i + 1, 1]))
            dice_b = dice(tumor, boundary_tumor_b)
            i += 1
        vol_a = boundary_tumor_a.sum() * voxel_volume / 1000
        vol_b = boundary_tumor_b.sum() * voxel_volume / 1000
        # print(i - 1)
        # print(dice_a, dice_b)
        # print(vol_a, vol_b)
        # print('---------')
        dice_ratio = (observe_variability_of_dice - dice_b) / (dice_a - dice_b)
        boundary_vol = vol_b + dice_ratio * (vol_a - vol_b)

        return boundary_vol
    # print('---------------------------------------')

    min_boundary_vol = find_boundary('min')
    max_boundary_vol = find_boundary('max')

    return min_boundary_vol, max_boundary_vol


def write_vol_diff_between_GT_and_pred_for_uniques(liver_files: List[str], gt_tumors_files: List[str],
                                                   pred_tumors_files: List[str], n_processes=None,
                                                   results_file: Optional[str] = 'measures_results/vol_diff_between_GT_and_pred_for_uniques_with_observer_variability.xlsx'):

    if n_processes is None:
        n_processes = os.cpu_count() - 2

    file_names = list(zip(liver_files, gt_tumors_files, pred_tumors_files))

    vol_diffs = process_map(vol_diff_between_GT_and_pred_for_uniques, file_names, max_workers=n_processes)
    # vol_diffs = list(map(vol_diff_between_GT_and_pred_for_uniques, file_names))

    columns = ['case name - tumor id - th', 'GT diameter (mm)', 'pred diameter (mm)', 'Dice', 'ASSD (mm)', 'Hausdorff (mm)',
               'GT volume (CC)', 'pred volume (CC)', 'Delta between tumor volumes (CC)',
               'ABS Delta between tumor volumes (CC)', 'Delta between tumor volumes (%)',
               'ABS Delta between tumor volumes (%)', 'Max error boundary (CC)', 'Min error boundary (CC)',
               'Is in boundary (Boolean)']
    df_results = pd.concat([pd.DataFrame(res, columns=columns) for res in vol_diffs])

    resuls_dir_name = os.path.dirname(results_file)
    os.makedirs(resuls_dir_name, exist_ok=True)

    writer = pd.ExcelWriter(results_file, engine='xlsxwriter')

    write_to_excel(df_results, writer, columns, 'case name - tumor id - th', sheet_name='all diameters')
    write_to_excel(df_results[df_results['GT diameter (mm)'] >= 5], writer, columns, 'case name - tumor id - th',
                   sheet_name='GT_diameter >= 5')
    write_to_excel(df_results[df_results['GT diameter (mm)'] >= 10], writer, columns, 'case name - tumor id - th',
                   sheet_name='GT_diameter >= 10')
    writer.save()


if __name__ == '__main__':
    # processes_for_Adi = 0
    #
    # n=None
    # test_Liver_gt, test_Tumors_gt = test_Liver_gt[:n], test_Tumors_gt[:n]
    #
    # print('----------------- starting writing scans tumors statistics -----------------')
    # t = time()
    # write_scans_tumors_statistics(test_Liver_gt, test_Tumors_gt, n_processes=os.cpu_count()-processes_for_Adi)
    # print(f'----------------- finished writing scans tumors statistics in {calculate_runtime(t)} (hh:mm:ss) -----------------')
    #
    # n = None
    # tumors_bl_gt_niftis_paths, tumors_fu_gt_niftis_paths = tumors_bl_gt_niftis_paths[:n], tumors_fu_gt_niftis_paths[:n]
    #
    # print('----------------- starting writing pairs changes -----------------')
    # t = time()
    # write_pairs_changes(tumors_bl_gt_niftis_paths, tumors_fu_gt_niftis_paths, n_processes=os.cpu_count()-processes_for_Adi)
    # print(f'----------------- finished writing pairs changes in {calculate_runtime(t)} (hh:mm:ss) -----------------')

    # ----------------------------------------------------------------------------------------------------------------

    # test_set_pairs_path = '/cs/casmip/rochman/Errors_Characterization/data/tumors_measurements_-_th_8_-_final_test_set_-_R2U_NET_pairwise_140_pairs_-_GT_liver_without_BL_tumors_edited.xlsx'
    # test_set_pairs = pd.read_excel(test_set_pairs_path)
    # test_set_pairs = list(set(pair_name.replace("BL_", "").split("_FU_")[1] for pair_name in test_set_pairs['Unnamed: 0'] if pair_name.startswith('BL_')))
    #
    # get_case_name = lambda file_name: basename(dirname(file_name))
    # def sort_key(name):
    #     split = name.split('_')
    #     return '_'.join(c for c in split if not c.isdigit()), int(split[-1]), int(split[-2]), int(split[-3])
    # test_Liver_gt = [case for case in test_Liver_gt if get_case_name(case) in test_set_pairs]
    # test_Liver_gt.sort(key=lambda path: sort_key(get_case_name(path)))
    # test_Tumors_gt = []
    # test_CT = []
    # for case in test_Liver_gt:
    #     try:
    #         tumor_file = replace_in_file_name(case, '/liver.nii.gz', '/tumors.nii.gz')
    #     except:
    #         tumor_file = replace_in_file_name(case, '/liver_pred.nii.gz', '/tumors.nii.gz')
    #     test_Tumors_gt.append(tumor_file)
    #
    #     try:
    #         ct_file = replace_in_file_name(case, '/liver.nii.gz', '/scan.nii.gz')
    #     except:
    #         ct_file = replace_in_file_name(case, '/liver_pred.nii.gz', '/scan.nii.gz')
    #     test_CT.append(ct_file)
    #
    # processes_to_leave_free = 5
    #
    # start = None
    # end = None
    # test_Liver_gt, test_Tumors_gt, test_CT = test_Liver_gt[start:end], test_Tumors_gt[start:end], test_CT[start:end]
    #
    # print('----------------- starting writing tumors measures -----------------')
    # t = time()
    # try:
    #     write_tumors_measures(test_CT, test_Liver_gt, test_Tumors_gt, n_processes=os.cpu_count()-processes_to_leave_free)
    #     notify("The running of function 'write_tumors_measures' finished successfully")
    # except Exception as e:
    #     notify(f"There was an error while running function 'write_tumors_measures': {e}", error=True)
    #     raise e
    # print(f'----------------- finished writing tumors measures in {calculate_runtime(t)} (hh:mm:ss) -----------------')

    try:

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
            for patient, patient_df in df.groupby('patient'):
                if patient_df.shape[0] > max_pairs_per_patient:
                    patient_df = patient_df.sort_values('time_interval', ignore_index=True)
                    patient_df = patient_df.iloc[:max_pairs_per_patient, :]
                res += patient_df['case_name'].to_list()
            return res

        def filter_out_not_consecutive_pairs(pair_names: List[str], pair_name_is_a_full_path_dir: bool = False):
            if pair_name_is_a_full_path_dir:
                pairs = [os.path.basename(p) for p in pair_names]
            else:
                pairs = pair_names

            df = pd.DataFrame(data=list(zip(pair_names, pairs, (get_patient_name(p) for p in pairs), (get_bl_name(p) for p in pairs), (get_fu_name(p) for p in pairs))),
                              columns=['id', 'pair_name', 'patient', 'bl_name', 'fu_name'])

            relevant_pairs = []
            for patient, patient_df in df.groupby('patient'):
                patient_scans = sorted(list(set(patient_df['bl_name'].to_list()) | set(patient_df['fu_name'].to_list())), key=scans_sort_key)
                for i in range(len(patient_scans) - 1):
                    relevant_pairs.append(f'BL_{patient_scans[i]}_FU_{patient_scans[i+1]}')

            return df[df['pair_name'].isin(relevant_pairs)]['id'].to_list()

        liver_study = False
        without_improving_registration = False
        if not liver_study:
            without_improving_registration = False

        only_consecutive_pairs = False

        t = time()

        consecutive_pairs_suffix = '____only_consecutive_pairs' if only_consecutive_pairs else ''
        if liver_study:
            pairs_dir_names = glob(f'/cs/casmip/rochman/Errors_Characterization/corrected_segmentation_for_matching/BL_*')
            pairs_dir_names += glob(f'/cs/casmip/rochman/Errors_Characterization/new_test_set_for_matching/BL_*')
            if without_improving_registration:
                max_dilation_for_pred = 15
                results_file = f'/cs/casmip/rochman/Errors_Characterization/new_test_set_for_matching/all_test_set_measures_results_without_improving_registration_+_match_algo_v5/diff_in_matching_according_to_CC_type_dilate_{max_dilation_for_pred}.xlsx'
            else:
                max_dilation_for_pred = 7
                results_file = f'/cs/casmip/rochman/Errors_Characterization/new_test_set_for_matching/all_test_set_measures_results_after_improving_registration_with_only_liver_border_at_ICP_no_tumors_and_no_RANSAC_+_match_algo_v5/diff_in_matching_according_to_CC_type_dilate_{max_dilation_for_pred}.xlsx'
        else:
            pairs_dir_names = glob(f'/cs/casmip/rochman/Errors_Characterization/lung_test_set_for_matching/BL_*')
            max_dilation_for_pred = 5
            results_file=f'/cs/casmip/rochman/Errors_Characterization/lung_test_set_for_matching/measures_results/diff_in_matching_according_to_CC_type_dilate_{max_dilation_for_pred}.xlsx'
        results_file = results_file.replace('.xlsx', f'{consecutive_pairs_suffix}.xlsx')
        pairs_dir_names = filter_num_of_pairs_per_patient(sorted(pairs_dir_names), max_pairs_per_patient=20, pair_name_is_a_full_path_dir=True)
        if only_consecutive_pairs:
            pairs_dir_names = filter_out_not_consecutive_pairs(pairs_dir_names, pair_name_is_a_full_path_dir=True)
        write_diff_in_matching_according_to_lesion_type(pairs_dir_names,
                                                        results_file=results_file,
                                                        max_dilation_for_pred=max_dilation_for_pred,
                                                        without_improving_registration=without_improving_registration)
        # notify(f'The calculating of the difference between the matching finished in: {calculate_runtime(t)} (hh:mm:ss)')
    except Exception as e:
        # notify(f'There was an error during calculating the difference between the matching: {e}', error=True)
        raise e
    exit(0)


    # def _scans_sort_key(file, full_path_given=True):
    #     if full_path_given:
    #         file = os.path.basename(os.path.dirname(file))
    #     split = file.split('_')
    #     return '_'.join(c for c in split if not c.isdigit()), int(split[-1]), int(split[-2]), int(split[-3])
    #
    #
    # old_path = '/cs/casmip/rochman/Errors_Characterization/data_to_recheck'
    # new_path = '/cs/casmip/rochman/Errors_Characterization/data_to_recheck_corrected'
    # old_tumors = glob(f'{old_path}/*/Tumors.nii.gz')
    # old_tumors.sort(key=_scans_sort_key)
    # new_tumors = []
    # for old_tumors_file in old_tumors:
    #     new_tumors.append(replace_in_file_name(old_tumors_file, old_path, new_path))
    # write_observer_variability(old_tumors, new_tumors)
    #
    # from sys import exit
    # exit(0)

    # running time checking
    relevant_scans = [os.path.basename(s) for s in glob('/cs/casmip/rochman/Errors_Characterization/data_to_recheck/*')]
    relevant_pairs = pd.read_excel('/cs/casmip/rochman/Errors_Characterization/new_test_set_for_matching/all_test_set_measures_results_after_improving_registration_with_only_liver_border_at_ICP_no_tumors_and_no_RANSAC_+_match_algo_v5/matching_statistics_dilate_7.xlsx')
    relevant_pairs = relevant_pairs.iloc[:-5, 0].to_list()

    are_valid = lambda bl, fu: bl in relevant_scans and fu in relevant_scans
    is_valid = lambda p: are_valid(*p.replace('BL_', '').split('_FU_'))
    relevant_pairs = sorted([p for p in relevant_pairs if is_valid(p)])

    from registeration_by_folder import liver_registeration
    from ICL_matching import execute_ICP
    import open3d as o3d
    from scipy.ndimage import affine_transform
    from colorama import Fore, Style

    bl_and_fu_names = [os.path.basename(pair_name).replace('BL_', '').split('_FU_') for pair_name in relevant_pairs]

    def registration_of_pair(bl_and_fu_name):
        t = time()

        corrected_dir = f'/cs/casmip/rochman/Errors_Characterization/data_to_recheck_corrected'
        res_dir = '/cs/casmip/rochman/Errors_Characterization/running_time_checking'
        bl_name, fu_name = bl_and_fu_name

        bl_CT = f'{corrected_dir}/{bl_name}/Scan_CT.nii.gz'
        bl_liver = f'{corrected_dir}/{bl_name}/Liver.nii.gz'
        bl_tumors = f'{corrected_dir}/{bl_name}/Tumors.nii.gz'

        fu_CT = f'{corrected_dir}/{fu_name}/Scan_CT.nii.gz'
        fu_liver = f'{corrected_dir}/{fu_name}/Liver.nii.gz'
        fu_tumors = f'{corrected_dir}/{fu_name}/Tumors.nii.gz'

        register_class = liver_registeration([bl_CT], [bl_liver], [bl_tumors], [fu_CT], [fu_liver], [fu_tumors],
                                             dest_path=res_dir, bl_name=bl_name, fu_name=fu_name)
        register_class.affine_registeration()

        return time() - t

    def ICP_registration_and_matching(pair_path):

        t1 = time()

        # from here

        # load liver segmentations
        bl_liver_file = f'{pair_path}/BL_Scan_Liver.nii.gz'
        fu_liver_file = f'{pair_path}/FU_Scan_Liver.nii.gz'
        bl_liver, _ = load_nifti_data(bl_liver_file)
        fu_liver, file = load_nifti_data(fu_liver_file)

        # extract liver borders
        selem = disk(1).reshape([3, 3, 1])
        bl_liver_border_points = affines.apply_affine(file.affine, np.stack(
            np.where(np.logical_xor(binary_dilation(bl_liver, selem), bl_liver))).T)
        fu_liver_border_points = affines.apply_affine(file.affine, np.stack(
            np.where(np.logical_xor(binary_dilation(fu_liver, selem), fu_liver))).T)

        # convert to point cloud
        ICP_bl_pc = o3d.geometry.PointCloud()
        ICP_fu_pc = o3d.geometry.PointCloud()
        ICP_bl_pc.points = o3d.utility.Vector3dVector(bl_liver_border_points)
        ICP_fu_pc.points = o3d.utility.Vector3dVector(fu_liver_border_points)

        # apply ICP
        result_icp = execute_ICP(ICP_bl_pc, ICP_fu_pc, voxel_size=1, distance_threshold_factor=40,
                                 init_transformation=np.eye(4))
        transform_inverse = np.linalg.inv(file.affine) @ np.linalg.inv(result_icp.transformation) @ file.affine

        # load tumors segmentation
        bl_tumors_file = f'{pair_path}/BL_Scan_Tumors.nii.gz'
        fu_tumors_file = f'{pair_path}/FU_Scan_Tumors.nii.gz'
        bl_tumors, _ = load_nifti_data(bl_tumors_file)
        fu_tumors, _ = load_nifti_data(fu_tumors_file)

        # label the tumors
        bl_labeled_tumors = get_connected_components(bl_tumors, connectivity=1)
        fu_labeled_tumors = get_connected_components(fu_tumors, connectivity=1)

        # register the labeled tumors
        transformed_bl_labeled_tumors = affine_transform(bl_labeled_tumors, transform_inverse, order=0) # if mask order=0, else default

        # to here

        t1 = time() - t1

        t2 = time()

        # match between tumors
        matches = match_2_cases_v5(transformed_bl_labeled_tumors, fu_labeled_tumors,
                                   voxelspacing=file.header.get_zooms(), max_dilate_param=7)

        t2 = time() - t2
        return t1, t2

    reg_RTs, ICP_RTs, matching_RTs = [], [], []

    for pair_name in relevant_pairs:
        bl_and_fu_name = os.path.basename(pair_name).replace('BL_', '').split('_FU_')

        reg_RT = registration_of_pair(bl_and_fu_name)
        ICP_RT, matching_RT = ICP_registration_and_matching(f'/cs/casmip/rochman/Errors_Characterization/running_time_checking/{pair_name}')

        reg_RTs.append(reg_RT)
        ICP_RTs.append(ICP_RT)
        matching_RTs.append(matching_RT)

        print(Fore.YELLOW + f'################################# {reg_RT:.3f}, {ICP_RT:.3f}, {matching_RT:.3f} #################################')
        print(Style.RESET_ALL)

    total_RTs = np.asarray(reg_RTs) + np.asarray(ICP_RTs) + np.asarray(matching_RTs)
    np.savez('running_times.npz', np.stack([reg_RTs, ICP_RTs, matching_RTs, total_RTs]))

    print(Fore.YELLOW + f'reg_RTs: {np.mean(reg_RTs):.3f} +- {np.std(reg_RTs):.3f}')
    print(Fore.YELLOW + f'ICP_RTs: {np.mean(ICP_RTs):.3f} +- {np.std(ICP_RTs):.3f}')
    print(Fore.YELLOW + f'matching_RTs: {np.mean(matching_RTs):.3f} +- {np.std(matching_RTs):.3f}')
    print(Fore.YELLOW + f'total_RTs: {np.mean(total_RTs):.3f} +- {np.std(total_RTs):.3f}')

    exit(0)


    data_path = '/cs/casmip/public/for_shalom/Tumor_segmentation/final_test_set_-_R2U_NET_standalone_as_pairwise_33_scans_-_GT_livers'
    gt_tumors = glob(f'{data_path}/*/Scan_Tumors.nii.gz')
    gt_tumors.sort(key=_scans_sort_key)
    pred_tumors, livers = [], []
    for gt_file in gt_tumors:
        # pred_tumors.append(replace_in_file_name(gt_file, '/Scan_Tumors.nii.gz', '/Scan_Tumors_pred_th_8.nii.gz'))
        pred_tumors.append(replace_in_file_name(gt_file, '/Scan_Tumors.nii.gz', '/Scan_Tumors_pred_label.nii.gz'))
        livers.append(replace_in_file_name(gt_file, '/Scan_Tumors.nii.gz', '/Scan_Liver.nii.gz'))
    write_vol_diff_between_GT_and_pred_for_uniques(livers, gt_tumors, pred_tumors)

    from sys import exit
    exit(0)

    results_file = 'measures_results/stats_per_th.xlsx'
    df = pd.read_excel(results_file)
    # res = df[df.groupby([])]

    train_set_path = '/cs/casmip/rochman/Errors_Characterization/data/train_set_pairs_data_measures.xlsx'
    validation_set_path = '/cs/casmip/rochman/Errors_Characterization/data/validation_set_pairs_data_measures.xlsx'
    test_set_path = '/cs/casmip/rochman/Errors_Characterization/data/tumors_measurements_-_th_8_-_final_test_set_-_R2U_NET_pairwise_101_pairs_-_GT_liver_without_BL_tumors_edited.xlsx'

    for dataset, type in [(train_set_path, 'train'), (validation_set_path, 'validation'), (test_set_path, 'test')]:
        n_patients, triples_sum, triples_mean, triples_std, triples_median, triples_min, triples_max = triples(dataset)

        print(f'{type} set:')
        print(f'n_patients = {n_patients}')
        print(f'triples_sum = {triples_sum}')
        print(f'triples_mean = {triples_mean:.2f}')
        print(f'triples_std = {triples_std:.2f}')
        print(f'triples_median = {triples_median}')
        print(f'triples_min = {triples_min}')
        print(f'triples_max = {triples_max}')
        print('--------------------------------------------')

    exit(0)

    res = pd.read_excel('/cs/casmip/rochman/Errors_Characterization/measures_results/tumors_measures.xlsx')
    res = res[~res['Unnamed: 0'].isin(['mean', 'std', 'min', 'max', 'sum'])]
    # ax = res['dice_with_approximate_sphere'].plot.hist(bins=20, title='dice_with_approximate_sphere')

    #
    # index = ['tumor border\n(in/out 5 mm)', 'tumor center\n(in/out 10 mm)', 'tumor center & border\n(tumor border do/don\'t touch\n&/|\ntumor center in/out 10 mm)']
    #
    # near_liver_border = [res['tumor_in_5_mm_from_liver_border'].sum(), res['tumor_center_in_10_mm_from_liver_border'].sum(), ((res['tumor_dist_from_liver_border (mm)'] == 0) & (res['tumor_center_in_10_mm_from_liver_border'] == 1)).sum()]
    # far_from_liver_border = [(res['tumor_in_5_mm_from_liver_border'] == 0).sum(), (res['tumor_center_in_10_mm_from_liver_border'] == 0).sum(), ((res['tumor_dist_from_liver_border (mm)'] > 0) | (res['tumor_center_in_10_mm_from_liver_border'] == 0)).sum()]
    # df = pd.DataFrame({'near liver border': near_liver_border, 'far from liver border': far_from_liver_border}, index=index)
    # ax = df.plot.bar(rot=0, title='far vs near liver border')
    #
    m = res.shape[0]
    # for p in ax.patches:
    #     # label = f'{int(p.get_height())} ({100* p.get_height()/m:.0f}%)'
    #     label = f'{100* p.get_height()/m:.1f}%'
    #     if p.get_height() != 0:
    #         ax.annotate(label, (p.get_x() * 1.005, p.get_height() * 1.005))
    #
    # plt.legend(bbox_to_anchor=(1.05, 1.18))
    # plt.show()
    #
    for param, bins, show_percent, show_stats in [('tumor_diameter (mm)', [2.97,5,10,15,20,30,40,55.92], True, True),
                                      # ('tumor_dist_from_liver_border (mm)', range(0, 40, 2), True, True),
                                      # ('tumor_center_dist_from_liver_border (mm)', range(0, 40, 2), True, True),
                                      # ('liver_segment', range(1, 5, 1), True, False),
                                                  ('mean_tumor_diff_from_mean_liver (HU)', np.linspace(-72, 38, 22, endpoint=True), True, True),
                                                  ('dice_with_approximate_sphere', np.linspace(0.13, 0.92, 20, endpoint=True), True, True)]:
        current_df = res[param]
        ax = current_df.plot.hist(bins=bins, title=f'Histogram of {param}')
        ax.set_xticks(bins)
        if np.min(bins) >= 0 and np.max(bins) <= 1:
            ax.set_xticklabels(np.around(bins, decimals=2), rotation=40)
        else:
            ax.set_xticklabels(np.around(bins, decimals=1), rotation=40)
        ax.set_xlabel(param)
        if show_percent:
            sum = 0
            for p in ax.patches:
                percent = 100* p.get_height()/m
                sum += percent
                # label = f'{percent:.2f}%\n{sum:.2f}%'
                label = f'{percent:.1f}%'
                if p.get_height() != 0:
                    ax.annotate(label, (p.get_x() * 1.005, p.get_height() * 1.005))

        if show_stats:
            mean = current_df.mean()
            std = current_df.std()
            min = current_df.min()
            max = current_df.max()

            textstr = '\n'.join([f'mean={mean:.2f}',
                                 f'std={std:.2f}',
                                 f'min={min:.2f}',
                                 f'max={max:.2f}'])

            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            if param == 'dice_with_approximate_sphere':
                ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
                    verticalalignment='top', bbox=props)
            else:
                ax.text(0.73, 0.95, textstr, transform=ax.transAxes, fontsize=14,
                        verticalalignment='top', bbox=props)
        plt.show()

    pd.DataFrame.sort_values(res, by='tumor_volume (CC)').set_index('tumor_volume (CC)').dice_with_approximate_sphere.plot()
    plt.show()

    # try:
    #     write_final_excel_files_over_data_set()
    #     notify("The running of function 'write_final_excel_files_over_data_set' finished successfully")
    # except Exception as e:
    #     notify(f"There was an error while running function 'write_final_excel_files_over_data_set': {e}", error=True)
    #     raise e

    # f = '/cs/casmip/rochman/Errors_Characterization/BL_A_Ab_03_10_2018_FU_A_Ab_15_07_2018/FU_Scan_Liver.nii.gz'
    # liver_case, liver_file = load_nifti_data(f)
    # xmin, xmax, ymin, ymax, _, _ = bbox2_3D(liver_case)
    # res = np.zeros_like(liver_case)
    # res[(xmin+xmax)//2:xmax, (ymin+ymax)//2:ymax, :] = 1
    # res[xmin:(xmin+xmax)//2, (ymin+ymax)//2:ymax, :] = 2
    # res[:,ymin:(ymin+ymax)//2, :] = 3
    # res *= liver_case
    #
    # nib.save(nib.Nifti1Image(res, liver_file.affine), f.replace('FU_Scan_Liver.nii.gz', 'liver_segments.nii.gz'))

    # n_cases = 1
    # test_CT, test_Liver_gt, test_Tumors_gt = test_CT[:n_cases], test_Liver_gt[:n_cases], test_Tumors_gt[:n_cases]
    #
    # with Pool() as pool:
    #     # result = pool.map(get_list_of_dice_and_distance_from_border, zip(test_Liver_gt, test_Tumors_gt, test_Tumors_pred))
    #     result = list(map(get_list_of_dice_and_distance_from_border, zip(test_Liver_gt, test_Tumors_gt, test_Tumors_pred)))
    #
    #
    # df = pd.DataFrame([r for res in result for r in res], columns=['GT tumor diameter', 'Dice', 'Assd', 'Hausdorff', 'GT Distance from border', 'Case name'])
    #
    # print_full_df(df)

    # liver_file = '/cs/casmip/public/for_aviv/StandAlone-tumors_Jan21/new_test_set/test_set/P_A_08_06_2014/liver.nii.gz'
    # tumors_file = '/cs/casmip/public/for_aviv/StandAlone-tumors_Jan21/new_test_set/test_set/P_A_08_06_2014/tumors.nii.gz'
    # x = get_tumors_statistics((liver_file, tumors_file), th_dist_from_border_in_mm=5)
    # BL_tumors_file = '/cs/casmip/public/for_shalom/Tumor_segmentation/affine_registeration_new_fixed_test_set_-_U_NET_pairwise_26_pairs_-_GT_liver_with_BL_tumors/FU:::P_A_08_06_2014_-_BL:::P_A_21_09_2014/BL_tumors_registerated.nii.gz'
    # FU_tumors_file = '/cs/casmip/public/for_shalom/Tumor_segmentation/affine_registeration_new_fixed_test_set_-_U_NET_pairwise_26_pairs_-_GT_liver_with_BL_tumors/FU:::P_A_08_06_2014_-_BL:::P_A_21_09_2014/FU_tumors.nii.gz'
    # x = calculate_change_in_pairs((BL_tumors_file, FU_tumors_file))
    # print("")

    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    #     print(df)

    # result_dir = 'liver_border_with_tumors'
    # for i, (res, file) in enumerate(result):
    #     current_dir_name = f'{result_dir}/{basename(dirname(test_CT[i]))}/'
    #     os.makedirs(current_dir_name, exist_ok=True)
    #     copyfile(test_CT[i], current_dir_name + 'scan.nii.gz')
    #     copyfile(test_Liver_gt[i], current_dir_name + 'liver.nii.gz')
    #     copyfile(test_Tumors_gt[i], current_dir_name + 'tumors.nii.gz')
    #     save(Nifti1Image(res, file.affine), current_dir_name + 'liver_border_with_tumors.nii.gz')

    # all_dataset_path = '/mnt/sda1/aszeskin/Data_Followup_Full_29_4_2021'
    # from tqdm import tqdm
    #
    # def f(file_path):
    #     case, case_file = load_nifti_data(file_path)
    #     if (np.isin(2, case)):
    #         case[case == 1] = 0
    #         case[case == 2] = 1
    #         nib.save(nib.Nifti1Image(case, case_file.affine), file_path)
    #
    # f('/mnt/sda1/aszeskin/Data_Followup_Full_29_4_2021/BL_A_Ab_15_07_2018_FU_A_Ab_03_10_2018/BL_Scan_Tumors.nii.gz')
    # f('/mnt/sda1/aszeskin/Data_Followup_Full_29_4_2021/BL_A_Ab_15_07_2018_FU_A_Ab_03_10_2018/FU_Scan_Tumors.nii.gz')
    #
    # # with Pool(processes=10) as pool:
    # #     pool.map(f, glob(f'{all_dataset_path}/*/*_Scan_Tumors.nii.gz'))
    #
    # l = glob(f'{all_dataset_path}/*/*_Scan_Tumors.nii.gz')
    # l.remove('/mnt/sda1/aszeskin/Data_Followup_Full_29_4_2021/BL_A_Ab_15_07_2018_FU_A_Ab_03_10_2018/FU_Scan_Tumors.nii.gz')
    #
    # with Pool(processes=20) as pool:
    #     pool.map(f, l)
    #
    # # for file_path in tqdm(l):
    # #     f(file_path)
    #
    # #     case, case_file = load_nifti_data(file_path)
    # #     if np.unique(case).size == 3:
    # #         case[case == 1] = 0
    # #         case[case == 2] = 1
    # #         nib.save(nib.Nifti1Image(case, case_file.affine), file_path)
