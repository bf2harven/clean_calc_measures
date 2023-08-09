import os
import sys
import tempfile
from datetime import datetime

from tqdm.contrib.concurrent import process_map
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_PARAGRAPH_ALIGNMENT, MSO_VERTICAL_ANCHOR
from functools import partial
from glob import glob
from typing import List, Tuple, Dict, Optional
from skimage.morphology import binary_dilation, ball
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nibabel as nib
from matplotlib.ticker import PercentFormatter
# sys.path.append('/home/rochman/Documents/src/Errors_Characterization')
from Errors_Characterization.utils import (write_to_excel, load_nifti_data, get_tumors_intersections,
                                           get_connected_components, dice, min_distance, pre_process_segmentation,
                                           getLargestCC, approximate_diameter, bbox2_3D)


def aggregate_pairwise_results_per_FU(ex_fn: str, output_ex_fn: str, agg: str = 'mean'):
    """
    Aggregates `calculate measures` pairwise results per Followup case.

    Parameters
    ----------
    ex_fn : str
        Path to original pairwise `calculate measures` results Excel file.
    output_ex_fn : str
        Path to Excel file where to output the aggregated results.
    agg : {'mean', 'median'}
        Aggregation method to use: 'mean' or 'median'.

    Returns
    -------
    None
    """

    writer = pd.ExcelWriter(output_ex_fn, engine='xlsxwriter')

    # read original diameters sheets
    for d in [0, 5, 10]:
        df = pd.read_excel(ex_fn, sheet_name=f'diameter_{d}').iloc[:-5]

        # aggregate per FU
        df['FU'] = [s[1] for s in df[df.columns[0]].str.split('_FU_')]
        if agg == 'mean':
            df = df.groupby('FU')[df.columns[1:-1]].mean()
        elif agg == 'median':
            df = df.groupby('FU')[df.columns[1:-1]].median()
        else:
            raise ValueError('The `agg` parameter must be either "mean" or "median".')
        df.reset_index(names='case_name', inplace=True)

        # save results
        write_to_excel(df, writer, df.columns, 'case_name', f'diameter_{d}',
                       {'F1 Score': ('Precision', 'Recall')}
                       )

    writer.close()


def merge_excel_results_files(ex_fns: List[str], output_ex_fn: str):
    """
    Merges different datasets' `calculate measures` results.

    Parameters
    ----------
    ex_fns : list of str
        List of paths to Excel files to merge.
    output_ex_fn : str
        Path to Excel file where to output the merged results.

    Returns
    -------
    None
    """

    assert ex_fns

    writer = pd.ExcelWriter(output_ex_fn, engine='xlsxwriter')

    # read original diameters sheets
    for d in [0, 5, 10]:
        res_df = pd.DataFrame()

        # merge results
        for ex_fn in ex_fns:
            res_df = pd.concat([res_df, pd.read_excel(ex_fn, sheet_name=f'diameter_{d}').iloc[:-5]]).reset_index(drop=True)

        res_df.rename(columns={res_df.columns[0]: 'case_name'}, inplace=True)

        # save results
        write_to_excel(res_df, writer, res_df.columns, 'case_name', f'diameter_{d}',
                       {'F1 Score': ('Precision', 'Recall')}
                       )

    writer.close()


def analyze_per_tumor_measures(gt_tumors_pred_tumors_roi: Tuple[str, str, str],
                               region_based_inference: bool = False) -> Tuple[List[Tuple[int, float, float, float]], List[Tuple[float, float]]]:
    """
    Analyzing each GT tumor in the given case:

    - Classifying each GT tumor as TP or FN according to the given tumors' prediction file.
    - Calculating the Dice-Coefficient score for each of them according to the given tumors' prediction file.
    - Calculating each GT tumor's approximated diameter in mm.
    - Calculating the distance of each GT tumor to organ border.

    In addition, analyzing each FP Pred tumor in the given predicted case:

    - Calculating each FP Pred tumor's approximated diameter in mm.
    - Calculating the distance of each FP Pred tumor to organ border.

    Parameters
    ----------
    gt_tumors_pred_tumors_roi : tuple of three strs
        Paths to the GT tumors mask, the tumors' prediction mask and the organ ROI mask.
    region_based_inference : bool, default False
        Whether the predicted masks were inferenced using a region-based trained model.

    Returns
    -------
    results : list of tuple of int, and 3 floats and list of tuple of 2 floats
            1) A list containing for each GT tumor:
                1) An integer indicating if the tumor is a TP (1) or an FN (0); 2) the tumor's Dice-Coefficient score;
                3) the tumor's approximated diameter in mm, and; 4) the distance of the tumor to the organ's border.
            2) A list containing for each FP Pred tumor:
                1) the tumor's approximated diameter in mm, and; 2) the distance of the tumor to the organ's border.
    """

    # extract parameters
    gt_tumors_fn, pred_tumors_fn, roi_fn = gt_tumors_pred_tumors_roi

    # load ROI mask
    roi, _ = load_nifti_data(roi_fn)

    # load the tumors masks
    gt_tumors, nifti_f = load_nifti_data(gt_tumors_fn)
    pred_tumors, _ = load_nifti_data(pred_tumors_fn)

    if region_based_inference:
        gt_tumors = (gt_tumors == 2).astype(gt_tumors.dtype)
        pred_tumors = (pred_tumors == 2).astype(pred_tumors.dtype)

    # preprocess the GT tumors masks (1)
    gt_tumors = pre_process_segmentation(gt_tumors)

    # preprocess the ROI mask
    roi = np.logical_or(roi, gt_tumors).astype(roi.dtype)
    roi = getLargestCC(roi)
    roi = pre_process_segmentation(roi, remove_small_obs=False)

    # preprocess the GT tumors masks (2)
    gt_tumors = np.logical_and(roi, gt_tumors).astype(gt_tumors.dtype)

    # crop to ROI
    xmin, xmax, ymin, ymax, zmin, zmax = bbox2_3D(roi)
    s = np.s_[max(0, xmin - 2): min(roi.shape[0], xmax + 3),
        max(0, ymin - 2): min(roi.shape[1], ymax + 3),
        max(0, zmin - 2): min(roi.shape[2], zmax + 3)]
    roi = roi[s]
    gt_tumors = gt_tumors[s]
    pred_tumors = pred_tumors[s]

    # preprocess the Pred tumors masks
    pred_tumors = np.logical_and(roi, pred_tumors).astype(pred_tumors.dtype)
    pred_tumors = pre_process_segmentation(pred_tumors)

    roi_border = np.logical_xor(roi, binary_dilation(roi, ball(1)))

    # label tumors
    gt_tumors = get_connected_components(gt_tumors, connectivity=None)
    pred_tumors = get_connected_components(pred_tumors, connectivity=None)

    # extract all the GT tumors' labels
    all_GT_tumors = np.unique(gt_tumors)
    all_GT_tumors = all_GT_tumors[all_GT_tumors != 0]

    # extract all the pred tumors' labels
    all_pred_tumors = np.unique(pred_tumors)
    all_pred_tumors = all_pred_tumors[all_pred_tumors != 0]

    # extract for each TP tumor, a list of all pred tumors intersecting with it
    intersections = get_tumors_intersections(gt_tumors, pred_tumors)

    pred_tp_tumors = list({t for ts in intersections.values() for t in ts})

    voxelspacing = nifti_f.header.get_zooms()

    gt_res = []

    # iterating over all GT tumors
    for current_t in all_GT_tumors:

        gt_current_t = gt_tumors == current_t

        diameter = approximate_diameter(gt_current_t.sum() * voxelspacing[0] * voxelspacing[1] * voxelspacing[2])

        # if current_t is a TP tumor:
        if current_t in intersections:
            tp = 1
            dce = dice(gt_current_t, np.isin(pred_tumors, intersections[current_t]))

        # if current_t is an FN tumor:
        else:
            tp = 0
            dce = 0

        dist = min_distance(gt_current_t, roi_border, voxelspacing=voxelspacing)

        gt_res.append((tp, dce, diameter, dist))

    pred_fp_res = []

    # iterating over all PRED tumors
    for current_t in all_pred_tumors:

        if current_t not in pred_tp_tumors:

            pred_current_t = pred_tumors == current_t

            diameter = approximate_diameter(pred_current_t.sum() * voxelspacing[0] * voxelspacing[1] * voxelspacing[2])

            dist = min_distance(pred_current_t, roi_border, voxelspacing=voxelspacing)

            pred_fp_res.append((diameter, dist))

    return gt_res, pred_fp_res


def write_per_tumor_analysis(gt_tumors_dir: str, pred_tumors_dir: str, rois_dir: str,
                             region_based_inference: bool = False):

    """
    Write per tumor analysis into an Excel file.

    Parameters
    ----------
    gt_tumors_dir : str
        Path to GT tumors directory.
    pred_tumors_dir : str
        Path to PRED tumors directory.
    rois_dir : str
        Path to ROIs directory.
    region_based_inference : bool, default False
        Whether the predicted masks were inferenced using a region-based trained model.

    Returns
    -------
    None
    """

    gt_tumors_fns = sorted(glob(f'{gt_tumors_dir}/*.nii.gz'))
    pred_tumors_fns = sorted(glob(f'{pred_tumors_dir}/*.nii.gz'))
    roi_fns = sorted(glob(f'{rois_dir}/*.nii.gz'))
    output_ex_fn = f'{pred_tumors_dir}/per_tumor_analysis.xlsx'

    # n_cases = 3 # todo delete
    n_cases = None
    gt_tumors_fns, pred_tumors_fns, roi_fns = gt_tumors_fns[:n_cases], pred_tumors_fns[:n_cases], roi_fns[:n_cases]

    case_ids = []
    for i, gt_t_fn in enumerate(gt_tumors_fns):
        assert os.path.basename(gt_t_fn) == os.path.basename(pred_tumors_fns[i])
        assert os.path.basename(gt_t_fn) == os.path.basename(roi_fns[i])
        case_ids.append(os.path.basename(gt_t_fn).replace('.nii.gz', ''))

    res = process_map(partial(analyze_per_tumor_measures, region_based_inference=region_based_inference),
                      list(zip(gt_tumors_fns, pred_tumors_fns, roi_fns)), max_workers=5)
    # res = list(map(partial(analyze_per_tumor_measures, region_based_inference=region_based_inference), list(zip(gt_tumors_fns, pred_tumors_fns, roi_fns))))
    res_gts, res_fp_preds = zip(*res)

    res_gts = [(f'{case_ids[j]}_-_{i + 1}', *r_t) for j, r in enumerate(res_gts) for i, r_t in enumerate(r)]
    gts_columns = ['Case-ID', 'Is TP?', 'Dice', 'Tumor Diameter (mm)', 'Distance from organ border (mm)']
    res_gts = pd.DataFrame(res_gts, columns=gts_columns)

    os.makedirs(os.path.dirname(output_ex_fn), exist_ok=True)
    writer = pd.ExcelWriter(output_ex_fn, engine='xlsxwriter')
    write_to_excel(res_gts, writer, gts_columns, 'Case-ID', 'GT Lesions')

    res_fp_preds = [(f'{case_ids[j]}_-_{i + 1}', *r_t) for j, r in enumerate(res_fp_preds) for i, r_t in enumerate(r)]
    fp_preds_columns = ['Case-ID', 'Tumor Diameter (mm)', 'Distance from organ border (mm)']
    res_fp_preds = pd.DataFrame(res_fp_preds, columns=fp_preds_columns)

    write_to_excel(res_fp_preds, writer, fp_preds_columns, 'Case-ID', 'FP Pred Lesions')

    writer.close()


def build_comparison_presentation_slide(data: List[List[List[List[str]]]], models: List[str], datasets: List[str],
                                        categories: List[str], measures: List[str], title: Optional[str] = None,
                                        prs: Optional[Presentation] = None) -> Presentation:

    """
    Builds models comparison presentation slide.

    Parameters
    ----------
    data : List[List[List[List[str]]]]
        The input data for the table in the slide. It is structured as a nested list, where each level represents a
        model, dataset, category, and a list of measure values, respectively.
    models : list of str
        List of model names.
    datasets : list of str
        List of dataset names.
    categories : list of str
        List of category names.
    measures : list of str
        List of measure names.
    title : str, optional
        Title to add to the slide.
    prs : Presentation object, optional
        Presentation to add the slide to. If is not given a new Presentation object is initialized.

    Returns
    -------
    prs : Presentation object
        The presentation the slide was added to. If `prs` is given, the same object will be returned.
    """

    # extract number of models
    n_models = len(data)
    assert len(models) == n_models

    # extract number of datasets
    n_datasets = len(data[0]) if n_models > 0 else 0
    assert len(datasets) == n_datasets

    # extract number of categories
    n_categories = len(data[0][0]) if n_datasets > 0 else 0
    assert len(categories) == n_categories

    # extract number of measures
    n_measures = len(data[0][0][0]) if n_categories > 0 else 0
    assert len(measures) == n_measures

    # initialize presentation
    prs = Presentation() if prs is None else prs

    # add 1 slice to the presentation
    slide = prs.slides.add_slide(prs.slide_layouts[5])

    # prepare table size and place
    x_to_place, y_to_place, cell_width, cell_height = Inches(0), Inches(0), Inches(10), Inches(1.5)
    n_rows = 2 + n_measures * n_datasets
    n_columns = 2 + n_categories * n_models

    # add slide title
    if title is not None:
        tit = slide.shapes.title
        tit.text = title
        y_to_place += tit.top + tit.height

    # add table to slide
    shape = slide.shapes.add_table(n_rows, n_columns, x_to_place, y_to_place, cell_width, cell_height)
    table = shape.table

    def write_to_table(r, c, text):
        table.cell(r, c).text = text

    # insert datasets and measures names
    table.cell(0, 0).merge(table.cell(1, 0))
    table.cell(0, 1).merge(table.cell(1, 1))
    table.cell(0, 1).text = 'Measure'
    for d, dataset in enumerate(datasets):
        table.cell(2 + n_measures * d, 0).merge(table.cell(1 + n_measures * (d + 1), 0))
        write_to_table(2 + n_measures * d, 0, dataset)
        for me, meas in enumerate(measures):
            write_to_table(2 + n_measures * d + me, 1, meas)

    # iterate over all models
    for m, model in enumerate(models):

        # insert model name
        table.cell(0, 2 + n_categories * m).merge(table.cell(0, 1 + n_categories * (m + 1)))
        write_to_table(0, 2 + n_categories * m, model)

        # insert categories names
        for c, cat in enumerate(categories):
            write_to_table(1, 2 + n_categories * m + c, cat)

        # insert measures values
        for d in range(n_datasets):
            for c in range(n_categories):
                for me in range(n_measures):
                    write_to_table(2 + n_measures * d + me, 2 + n_categories * m + c, data[m][d][c][me])

    # adjust table size and font
    font_size = 8 if n_models == 5 else 10

    for row in table.rows:
        for cell in row.cells:
            cell.vertical_anchor = MSO_VERTICAL_ANCHOR.MIDDLE
            for paragraph in cell.text_frame.paragraphs:
                paragraph.alignment = PP_PARAGRAPH_ALIGNMENT.CENTER
                for run in paragraph.runs:
                    run.font.size = Pt(font_size)

    return prs


def build_per_tumor_comparison_slides(ex_data: Dict[str, Dict[str, str]], prs: Optional[Presentation] = None) -> Presentation:
    """
    Builds per tumor models comparison presentation slides.

    Parameters
    ----------
    ex_data : Dict[str, Dict[str, str]]
        A dictionary containing performance data for different models and datasets.
        The keys of the `ex_data` dictionary represent model names, while the values are dictionaries.
        Each nested dictionary contains dataset names as keys and the corresponding path to an Excel file as the value.
        The Excel file at each path contains the "per tumors" performance of the model on the respective dataset.
    prs : Presentation object, optional
        Presentation to add the slide to. If is not given a new Presentation object is initialized.

    Returns
    -------
    prs : Presentation object
        The presentation the slide was added to. If `prs` is given, the same object will be returned.
    """

    def filter_diameters(_df, _min=0, _max=np.inf):
        return _df[(_min < _df['Tumor Diameter (mm)']) & (_df['Tumor Diameter (mm)'] <= _max)]

    possible_cats = {
        'all': (0, np.inf),
        '>5': (5, np.inf),
        '<5': (0, 5),
        '>10': (10, np.inf),
        '<10': (0, 10),
        '5<&<10': (5, 10),
    }

    def get_structured_data(categories):
        data = []
        for m in sorted(ex_data):
            model = []
            for d in sorted(ex_data[m]):
                ex = ex_data[m][d]

                gt_df = pd.read_excel(ex, sheet_name='GT Lesions').iloc[:-5]
                fp_pred_df = pd.read_excel(ex, sheet_name='FP Pred Lesions').iloc[:-5]

                dataset = []
                for c in categories:
                    cat_min, cat_max = possible_cats[c]
                    gt_df_to_analyse = filter_diameters(gt_df, cat_min, cat_max)
                    fp_pred_df_to_analyse = filter_diameters(fp_pred_df, cat_min, cat_max)

                    # calculating number of lesions
                    n_lesions = len(gt_df_to_analyse)

                    # calculate Dice
                    tp_dice = gt_df_to_analyse[gt_df_to_analyse['Is TP?'] == 1]['Dice']
                    mean_dice = f'{tp_dice.mean():.2f}\n+-{tp_dice.std():.2f}'

                    # calculate detection measures
                    n_tp = len(tp_dice)
                    n_fn = n_lesions - n_tp
                    n_fp = len(fp_pred_df_to_analyse)
                    precision = f'{n_tp / (n_tp + n_fp):.2f}'
                    recall = f'{n_tp / (n_tp + n_fn):.2f}'

                    dataset.append([f'{n_lesions}', mean_dice, precision, recall, f'{n_tp}', f'{n_fn}', f'{n_fp}'])

                model.append(dataset)

            data.append(model)

        return data

    categories1 = ['all', '>5', '>10']
    categories2 = ['<5', '5<&<10', '>10']

    data1 = get_structured_data(categories1)
    data2 = get_structured_data(categories2)

    models = sorted(ex_data)
    datasets = sorted(ex_data[models[0]])
    measures = ['# of lesions', 'Mean Dice', 'Precision', 'Recall', '#TP', '#FN', '#FP']

    prs = build_comparison_presentation_slide(data1, models, datasets, categories1, measures,
                                              title='Per Tumor Comparison (1)', prs=prs)
    build_comparison_presentation_slide(data2, models, datasets, categories2, measures,
                                        title='Per Tumor Comparison (2)', prs=prs)

    return prs


def build_per_scan_comparison_slides(ex_data: Dict[str, Dict[str, str]], prs: Optional[Presentation] = None) -> Presentation:
    """
    Builds per scan models comparison presentation slides.

    Parameters
    ----------
    ex_data : Dict[str, Dict[str, str]]
        A dictionary containing performance data for different models and datasets.
        The keys of the `ex_data` dictionary represent model names, while the values are dictionaries.
        Each nested dictionary contains dataset names as keys and the corresponding path to an Excel file as the value.
        The Excel file at each path contains the "per scan" performance of the model on the respective dataset.
    prs : Presentation object, optional
        Presentation to add the slide to. If is not given a new Presentation object is initialized.

    Returns
    -------
    prs : Presentation object
        The presentation the slide was added to. If `prs` is given, the same object will be returned.
    """

    possible_cats = {
        'all': 'diameter_0',
        '>5': 'diameter_5',
        '<5': 'diameter<5',
        '>10': 'diameter_10',
        '<10': 'diameter_<10',
        '5<&<10': 'diameter_in_(5,10)',
    }

    def get_structured_data(categories):
        data = []
        for m in sorted(ex_data):
            model = []
            for d in sorted(ex_data[m]):
                ex = ex_data[m][d]
                dataset = []
                for c in categories:
                    cat_df = pd.read_excel(ex, sheet_name=possible_cats[c]).iloc[:-5]

                    # calculating number of lesions
                    n_lesions = f'{int(cat_df["Num of lesion"].sum())}'

                    # calculate Dice
                    dice = f'{cat_df["Dice"].mean():.2f}\n+-{cat_df["Dice"].std():.2f}'

                    # calculate precision
                    precision = f'{cat_df["Precision"].mean():.2f}\n+-{cat_df["Precision"].std():.2f}'
                    recall = f'{cat_df["Recall"].mean():.2f}\n+-{cat_df["Recall"].std():.2f}'

                    dataset.append([n_lesions, dice, precision, recall])

                model.append(dataset)

            data.append(model)

        return data

    categories1 = ['all', '>5', '>10']
    categories2 = ['<5', '5<&<10', '>10']

    data1 = get_structured_data(categories1)
    data2 = get_structured_data(categories2)

    models = sorted(ex_data)
    datasets = sorted(ex_data[models[0]])
    measures = ['# of lesions', 'Dice', 'Precision', 'Recall']

    prs = build_comparison_presentation_slide(data1, models, datasets, categories1, measures,
                                              title='Per Scan Comparison (1)', prs=prs)
    build_comparison_presentation_slide(data2, models, datasets, categories2, measures,
                                        title='Per Scan Comparison (2)', prs=prs)

    return prs

