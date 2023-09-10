import os.path
import sys
import tempfile
from typing import Union, List, Optional
from glob import glob
import numpy as np
import shutil
import concurrent.futures

from calculate_measures_utilities import aggregate_pairwise_results_per_FU, \
    write_per_tumor_analysis

# sys.path.append('/home/rochman/Documents/src')
from calculate_measures import write_stats
import nibabel as nib


def assert_same_case_ids(case_ids, test_lst):
    assert len(case_ids) == len(
        test_lst
    ), 'All the data directories should have the same number of .nii.gz files.'
    for f in test_lst:
        c_id = os.path.basename(f).replace('.nii.gz', '')
        assert c_id in case_ids, f'The case id "{c_id}" doesn\'t exist in all data directories'


def extract_case_id(f):
    return os.path.basename(f).replace('.nii.gz', '')


def nnunet_calculate_measures(gt_labels_path: str, pred_masks_paths: Union[str, List[str]],
                              roi_masks_path: Optional[str] = None, ths: Optional[List[float]] = None,
                              roi_is_gt: bool = True, n_processes: int = 10,
                              region_based_inferencing: bool = False, min_size: int = 20):
    """
    Calculating measures for nnU-Net predictions.

    Notes
    _____
    All the given directory paths must be structures as the nnU-Net `labelsTr` folder.
    See nnUNet/documentation/dataset_format.md file.

    Parameters
    ----------
    gt_labels_path : str
        Path to GT labels directory.
    pred_masks_paths : str or list of str
        Path to nnU-Net predicted labels. If a list of str is given, each directory will be measured separately. Note,
        the predicted files can be non-integer (in [0-1]), but then the `ths` parameter must be given and < 1.
    roi_masks_path str, optional
        Path to cases ROIs (in order to calculate all the measures within them). If it's set to None (by default), no
        ROI will be considered.
    ths : list of floats, optional
        Thresholds to apply on the predictions before calculating measures. If set to None (be default), the predictions
        will be threshold with 1.
        Note, each given threshold will be calculated separately.
    roi_is_gt : bool, default True
        If set to True (by default), GT labels will be considered only with in the ROI, otherwise, all the GT labels
        will be considered.
    n_processes : int, default 10
        Number of processes to use while calculating the measures. By default, 10.
    region_based_inferencing : bool, default False
        Whether the predicted masks were inferenced using a region-based trained model.

    Returns
    -------
    None
    """

    # listing gt masks files
    GT_paths = sorted(glob(f'{gt_labels_path}/*.nii.gz'))
    case_ids = [extract_case_id(f) for f in GT_paths]
    if roi_masks_path is None:
        roi_paths = [None for _ in GT_paths]
    else:
        roi_paths = sorted(glob(f'{roi_masks_path}/*.nii.gz'))
        assert_same_case_ids(case_ids, roi_paths)

    if ths is None:
        ths = [1]

    n = None
    for pp in pred_masks_paths:
        pred_paths = sorted(glob(f'{pp}/*.nii.gz'))
        assert_same_case_ids(case_ids, pred_paths)

        GT_paths, pred_paths, roi_paths = GT_paths[:n], pred_paths[:n], roi_paths[:n]

        write_stats(GT_paths, pred_paths, roi_paths, 1, extract_case_id,
                    pp, roi_is_gt=roi_is_gt, n_processes=n_processes, reduce_ram_storage=True,
                    add_th_to_excel_name=True, labels_to_consider=(2 if region_based_inferencing else None),
                    categories_to_calculate=('all', ), min_size=min_size)


