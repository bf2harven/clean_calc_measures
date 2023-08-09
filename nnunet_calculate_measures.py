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
    assert len(case_ids) == len(test_lst), f'All the data directories should have the same number of .nii.gz files.'
    for f in test_lst:
        c_id = os.path.basename(f).replace('.nii.gz', '')
        assert c_id in case_ids, f'The case id "{c_id}" doesn\'t exist in all data directories'


def extract_case_id(f):
    return os.path.basename(f).replace('.nii.gz', '')


def nnunet_calculate_measures(gt_labels_path: str, pred_masks_paths: Union[str, List[str]],
                              roi_masks_path: Optional[str] = None, ths: Optional[List[float]] = None,
                              roi_is_gt: bool = True, n_processes: int = 10,
                              region_based_inferencing: bool = False):
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
    # listing ROI masks files
    tempdir = None
    if roi_masks_path is None:
        # create tmp ROIs
        tempdir = tempfile.TemporaryDirectory()
        roi_paths = []
        for ind, f in enumerate(GT_paths):
            nifti_f = nib.load(f)
            tmp_liver = np.ones_like(nifti_f.get_fdata())
            tmp_liver_path = f'{tempdir.name}/roi_{ind}.nii.gz'
            nib.save(nib.Nifti1Image(tmp_liver, nifti_f.affine, nifti_f.header), tmp_liver_path)
            roi_paths.append(tmp_liver_path)
        cleanup_tempdir_at_the_end = True
    else:
        roi_paths = sorted(glob(f'{roi_masks_path}/*.nii.gz'))
        assert_same_case_ids(case_ids, roi_paths)
        cleanup_tempdir_at_the_end = False

    if ths is None:
        ths = [1]

    for pp in pred_masks_paths:
        pred_paths = sorted(glob(f'{pp}/*.nii.gz'))
        assert_same_case_ids(case_ids, pred_paths)

        # n = 1
        n = None

        GT_paths, pred_paths, roi_paths = GT_paths[:n], pred_paths[:n], roi_paths[:n]

        write_stats(GT_paths, pred_paths, roi_paths, ths, extract_case_id,
                    pp, roi_is_gt=roi_is_gt, n_processes=n_processes, reduce_ram_storage=True,
                    add_th_to_excel_name=True, labels_to_consider=(2 if region_based_inferencing else None),
                    categories_to_calculate=('all', ))

    if cleanup_tempdir_at_the_end:
        tempdir.cleanup()

def main_foo(gt_path,roi_path, pred_path, target_path=None, target_fname='tumors_measurements_-_th_1.xlsx'):
    data_to_calc_for = [
        (
            gt_path,
            roi_path,
            [pred_path],
            None,  # aggregration
            True,  # region based inference
            False,  # per tumor too
        ),]
    
    for gt_labels_path, roi_masks_path, pred_masks_paths, agg, region_based_inferencing, _ in data_to_calc_for:
        nnunet_calculate_measures(gt_labels_path, pred_masks_paths,
                                roi_masks_path, region_based_inferencing=region_based_inferencing)
    if agg is not None:
        for pp in pred_masks_paths:
            ex_fn = f'{pp}/tumors_measurements_-_th_1.xlsx'
            assert os.path.isfile(ex_fn)
            res_ex_fn = ex_fn.replace('.xlsx', f'_-_{agg}_agg_per_FU.xlsx')
            assert ex_fn != res_ex_fn
            aggregate_pairwise_results_per_FU(
                ex_fn=ex_fn,
                output_ex_fn=res_ex_fn,
                agg=agg
            )

    # move the results to the right place
    if target_path is not None:
        shutil.move(f'{pred_path}/tumors_measurements_-_th_1.xlsx', os.path.join(target_path, target_fname))


def main_multithreaded(gt_path, roi_path ,pred_path, target_path=None, target_fname='tumors_measurements_-_th_1.xlsx'):
    data_to_calc_for = [
        (
            gt_path,
            roi_path,
            [pred_path],
            None,  # aggregation
            False,  # region based inference
            False,  # per tumor too
        ),
    ]

    def process_data(args):
        gt_labels_path, roi_masks_path, pred_masks_paths, agg, region_based_inferencing, _ = args
        nnunet_calculate_measures(gt_labels_path, pred_masks_paths, roi_masks_path, region_based_inferencing=region_based_inferencing)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(process_data, data_to_calc_for)

    if target_path is not None:
        shutil.move(f'{pred_path}/tumors_measurements_-_th_1.xlsx', os.path.join(target_path, target_fname))




if __name__ == '__main__':
   

    # main_foo(gt_path='/cs/labs/josko/aarono/all_data/brain/edan_split/raw_split_symlinks/test/labels', 
    #          roi_path ='/cs/labs/josko/aarono/all_data/brain/edan_split/raw_split_symlinks/test/ROI',
    #      pred_path='/cs/labs/josko/aarono/all_data/brain/edan_split/edan_preds_raw',
    #      target_path='/cs/labs/josko/aarono/projects/diffunet_pt_lightning/misc_stuff/measures_calculations/xlx_pres',
    #      target_fname='edan_preds_measures_full_res_gt20_ROI.xlsx')
    

    main_foo(gt_path='/cs/labs/josko/aarono/all_data/liver/raw_data_symlinks/test/labels', 
            roi_path =None,
        pred_path='/cs/labs/josko/aarono/outputs/liver_segmentation/hyrad_no_shear/respaced_preds',
        target_path='/cs/labs/josko/aarono/outputs/liver_segmentation/hyrad_no_shear/respaced_preds',
        target_fname='diff_liver_preds_gt_20_no_rem.xlsx')

    # main_foo(gt_path='/cs/labs/josko/aarono/all_data/brain/edan_split/raw_split_symlinks/test/labels', 
    #      roi_path ='/cs/labs/josko/aarono/all_data/brain/edan_split/raw_split_symlinks/test/ROI',
    #  pred_path='/cs/labs/josko/aarono/outputs/edan_preds/respaced_cropped_preds_test/sigmoid_to_nii',
    #  target_path='/cs/labs/josko/aarono/projects/diffunet_pt_lightning/misc_stuff/measures_calculations/xlx_pres',
    #  target_fname='diff_preds_measures_full_res_gt20_ROI.xlsx')