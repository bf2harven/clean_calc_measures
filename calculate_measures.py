import sys
import pandas
from xlsxwriter.utility import xl_col_to_name
from Errors_Characterization.utils import assd_and_hd
from multiprocessing import Pool, cpu_count
from functools import partial
from typing import List, Tuple, Callable, Union, Optional
import pandas as pd
import numpy as np
from scipy import ndimage
from skimage import measure
import nibabel as nib
import operator
import glob
from tqdm import tqdm
from copy import deepcopy
import os
from time import time, gmtime
from skimage.morphology import remove_small_holes, remove_small_objects
from scipy.ndimage import binary_fill_holes
from tqdm.contrib.concurrent import process_map

def bbox2_3D(img):
    x = np.any(img, axis=(1, 2))
    y = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))

    xmin, xmax = np.where(x)[0][[0, -1]]
    ymin, ymax = np.where(y)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    return xmin, xmax, ymin, ymax, zmin, zmax


class tumors_statistics():
    def __init__(self, roi, gt, predictions, file_name, th, roi_is_gt=False,
                 reduce_ram_storage=False, labels_to_consider: Optional[Union[int, List[int]]] = None, min_size=20):
        # Loading 3 niftis files
        self.file_name = file_name
        gt_nifti = nib.load(gt)
        self.gt_nifti = gt_nifti
        pred_nifti = nib.load(predictions)
        self.predictions_path = predictions
        self.gt_path = gt
        

        # Getting voxel_volume
        self.pix_dims = gt_nifti.header.get_zooms()
        self.voxel_volume = self.pix_dims[0] * self.pix_dims[1] * self.pix_dims[2]
        self.affine = gt_nifti.affine

        # getting the 3 numpy arrays

        self.gt = gt_nifti.get_fdata()
        if roi is None:
            self.roi = np.ones_like(self.gt).astype(np.float32)
        else:
            self.roi = (nib.load(roi).get_fdata()>0).astype(np.float32)
        if labels_to_consider is not None:
            self.gt = np.isin(self.gt, labels_to_consider).astype(self.gt.dtype)
        
        self.gt = binary_fill_holes(self.gt, np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]).reshape([3, 3, 1]).astype(self.gt.dtype))
        self.gt = remove_small_objects(self.gt.astype(bool), min_size=min_size).astype(self.gt.dtype)

        if roi_is_gt:
            self.roi = np.logical_or(self.roi, self.gt).astype(self.roi.dtype)
        try:
            self.roi = self.getLargestCC(self.roi)
        except Exception as e:
            print(f'Error in: {self.file_name}')
            raise e
        self.roi = binary_fill_holes(self.roi, np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]).reshape([3, 3, 1]).astype(
            self.roi.dtype))

        if roi_is_gt:
            self.gt = np.logical_and(self.roi, self.gt).astype(self.gt.dtype)

        self.predictions = pred_nifti.get_fdata()
        if labels_to_consider is not None:
            self.predictions = np.isin(self.predictions, labels_to_consider).astype(self.predictions.dtype)

        if reduce_ram_storage:
            # cropping the cases to the bbox of the ROI
            xmin, xmax, ymin, ymax, zmin, zmax = bbox2_3D(np.logical_or(self.roi, self.gt))
            xmax += 1
            ymax += 1
            zmax += 1
            crop_slice = np.s_[xmin:xmax, ymin:ymax, zmin:zmax]
            self.gt = self.gt[crop_slice]
            self.roi = self.roi[crop_slice]
            self.predictions = self.predictions[crop_slice]

        self.predictions *= self.roi
        self.predictions[self.predictions < th] = 0
        self.predictions[self.predictions >= th] = 1

        # preprocessing on the prediction
        self.predictions = binary_fill_holes(self.predictions,
                                             np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]).reshape([3, 3, 1]).astype(
                                                 bool))
        self.predictions = remove_small_objects(self.predictions, min_size=min_size).astype(self.predictions.dtype)

      
        # unique lesions for gt and predictions
        self.unique_gt = self.CC(self.gt, min_size=min_size)
        self.unique_predictions = self.CC(self.predictions, min_size=min_size)

        # self.num_of_lesions = self.unique_gt[1]
        # self.dice_score = self.dice(self.gt, self.predictions)

    def calculate_statistics_by_diameter(self, diameter, oper=operator.gt, three_biggest=False,
                                         calculate_ASSD=False, calculate_HD=False):

        predict_lesions_with_diameter_TP = np.zeros(self.gt.shape)
        gt_lesions_with_diameter_TP = np.zeros(self.gt.shape)
        predict_lesions_TP = np.zeros(self.gt.shape)

        # calculate diameter for each lesion in GT
        tumors_with_diameter_gt_unique, tumors_with_diameter_gt_matrix, tumors_with_diameter_gt = \
                self.mask_by_diameter(self.unique_gt, diameter, oper)


        # Find 3 biggest GT
        if three_biggest:
            tumors_with_diameter_gt_unique, tumors_with_diameter_gt_matrix, tumors_with_diameter_gt = \
                    self.find_3_biggest_tumors(tumors_with_diameter_gt_unique)
       

        # calculate diameter for each lesion in Predictions
        tumors_with_diameter_predictions_matrix_unique, tumors_with_diameter_predictions_matrix, tumors_with_diameter_predictions = \
                self.mask_by_diameter(self.unique_predictions, diameter, oper)

        # Find 3 biggest GT
        if three_biggest:
            tumors_with_diameter_predictions_matrix_unique, tumors_with_diameter_predictions_matrix, tumors_with_diameter_predictions = \
                    self.find_3_biggest_tumors(tumors_with_diameter_predictions_matrix_unique)


        # Find predicted tumor that touches 1 tumor of the predicition
        # and calculating ASSDs ans Hausdorff metrices
        ASSDs: List[float] = []
        HDs: List[float] = []
        for i in tumors_with_diameter_gt:
            current_1_tumor_gt = (self.unique_gt[0] == i)
            unique_predictions_touch_current_gt_tumor = np.unique((current_1_tumor_gt * self.unique_predictions[0]))
            unique_predictions_touch_current_gt_tumor = list(unique_predictions_touch_current_gt_tumor[unique_predictions_touch_current_gt_tumor != 0])
            if unique_predictions_touch_current_gt_tumor:
                gt_lesions_with_diameter_TP[current_1_tumor_gt] = 1
            for j in unique_predictions_touch_current_gt_tumor:
                current_1_tumor_pred = (self.unique_predictions[0] == j)
                if calculate_HD and calculate_ASSD:
                    _assd, _hd = assd_and_hd(current_1_tumor_gt, current_1_tumor_pred,
                                             voxelspacing=self.pix_dims, connectivity=2)
                    ASSDs += [_assd]
                    HDs += [_hd]
                predict_lesions_TP[current_1_tumor_pred] = 1
    
        for i in tumors_with_diameter_predictions:
            current_1_tumor_pred = (self.unique_predictions[0] == i)
            unique_gt_touch_current_pred_tumor = np.unique((current_1_tumor_pred * self.unique_gt[0]))
            unique_gt_touch_current_pred_tumor = list(unique_gt_touch_current_pred_tumor[unique_gt_touch_current_pred_tumor != 0])
            if unique_gt_touch_current_pred_tumor:
                predict_lesions_with_diameter_TP[current_1_tumor_pred] = 1

        mean_ASSDs = float(format(np.mean(ASSDs), '.3f')) if ASSDs else np.nan
        mean_HDs = float(format(np.mean(HDs), '.3f')) if HDs else np.nan
        max_HDs = float(format(np.max(HDs), '.3f')) if HDs else np.nan

        # Segmentation statistics

        seg_TP, seg_FP, seg_FN = \
                self.Segmentation_statistics(tumors_with_diameter_gt_matrix, predict_lesions_with_diameter_TP,
                                         tumors_with_diameter_predictions_matrix, gt_lesions_with_diameter_TP,
                                         debug=False)

        Total_tumor_GT = float(format((tumors_with_diameter_gt_matrix > 0).sum() * self.voxel_volume * 0.001, '.3f'))
        Total_tumor_pred = float(
            format((tumors_with_diameter_predictions_matrix > 0).sum() * self.voxel_volume * 0.001, '.3f'))
        Total_tumor_GT_without_FN = float(format((gt_lesions_with_diameter_TP > 0).sum() * self.voxel_volume * 0.001, '.3f'))
        Total_tumor_pred_without_FP = float(format((predict_lesions_with_diameter_TP > 0).sum() * self.voxel_volume * 0.001, '.3f'))
        Liver_cc = self.roi.sum() * self.voxel_volume * 0.001

        if (Total_tumor_GT + Total_tumor_pred) == 0:
            delta_percentage = 0
        else:
            delta_percentage = ((Total_tumor_GT - Total_tumor_pred) / (Total_tumor_GT + Total_tumor_pred)) * 100

        if (Total_tumor_GT_without_FN + Total_tumor_pred_without_FP) == 0:
            delta_percentage_TP_only = 0
        else:
            delta_percentage_TP_only = ((Total_tumor_GT_without_FN - Total_tumor_pred_without_FP) / (Total_tumor_GT_without_FN + Total_tumor_pred_without_FP)) * 100

        # Detection statistics

        detection_TP, detection_FP, detection_FN, precision, recall = \
                self.Detection_statistics(predict_lesions_with_diameter_TP, tumors_with_diameter_gt_unique,
                                      tumors_with_diameter_predictions_matrix_unique, three_biggest, debug=False)
        try:
            f1_score = 2 * precision * recall / (precision + recall)
        except:
            f1_score = 0

        return {'Filename': self.file_name,
                'Num of lesion': len(tumors_with_diameter_gt),
                'Dice': self.dice(gt_lesions_with_diameter_TP, predict_lesions_TP),
                # 'Dice with FP and FN': self.dice(tumors_with_diameter_gt, tumors_with_diameter_predictions_matrix),
                'Dice with FN': self.dice(tumors_with_diameter_gt_matrix, predict_lesions_TP),
                'Segmentation TP (cc)': float(format(seg_TP.sum() * self.voxel_volume * 0.001, '.3f')),
                'Segmentation FP (cc)': float(format(seg_FP.sum() * self.voxel_volume * 0.001, '.3f')),
                'Segmentation FN (cc)': float(format(seg_FN.sum() * self.voxel_volume * 0.001, '.3f')),
                'Total tumor volume GT (cc)': Total_tumor_GT,
                'Total tumor volume Predictions (cc)': Total_tumor_pred,
                'Delta between total tumor volumes (cc)': Total_tumor_GT - Total_tumor_pred,
                'Delta between total tumor volumes (%)': delta_percentage,
                'Delta between total tumor volumes (TP only) (cc)': Total_tumor_GT_without_FN - Total_tumor_pred_without_FP,
                'ABS Delta between total tumor volumes (TP only) (cc)': abs(Total_tumor_GT_without_FN - Total_tumor_pred_without_FP),
                'Delta between total tumor volumes (TP only) (%)': delta_percentage_TP_only,
                'ABS Delta between total tumor volumes (TP only) (%)': abs(delta_percentage_TP_only),
                'Tumor Burden GT (%)': float(format(Total_tumor_GT / Liver_cc, '.3f')) * 100,
                'Tumor Burden Pred (%)': float(format(Total_tumor_pred / Liver_cc, '.3f')) * 100,
                'Tumor Burden Delta (%)': float(format((Total_tumor_GT - Total_tumor_pred) / Liver_cc, '.3f')) * 100,
                'Detection TP (per lesion)': detection_TP,
                'Detection FP (per lesion)': detection_FP,
                'Detection FN (per lesion)': detection_FN,
                'Precision': float(format(precision, '.3f')),
                'Recall': float(format(recall, '.3f')),
                'F1 Score': float(format(f1_score, '.3f')),
                'Mean ASSD (mm)': mean_ASSDs,
                'Mean Hausdorff (mm)': mean_HDs,
                'Max Hausdorff (mm)': max_HDs}
        # 'diameter': diameter,
        # 'oper': oper}

    def Segmentation_statistics(self, tumors_with_diameter_gt_matrix, predict_lesions_with_diameter_TP, tumors_with_diameter_predictions_matrix, gt_lesions_with_diameter_TP, debug=False):
        seg_TP = (tumors_with_diameter_gt_matrix * self.predictions)
        # seg_FP = (predict_lesions_with_diameter_TP - (tumors_with_diameter_gt_matrix * predict_lesions_with_diameter_TP))
        seg_FP = tumors_with_diameter_predictions_matrix * (1 - self.gt)
        # seg_FN = (tumors_with_diameter_gt_matrix - (tumors_with_diameter_gt_matrix * predict_lesions_with_diameter_TP))
        seg_FN = tumors_with_diameter_gt_matrix * (1 - self.predictions)
        if debug:
            unique_gt = nib.Nifti1Image(seg_FP, self.affine)
            nib.save(unique_gt, 'FP.nii.gz')
            unique_gt = nib.Nifti1Image(seg_FN, self.affine)
            nib.save(unique_gt, 'FN.nii.gz')

        return seg_TP, seg_FP, seg_FN

    def mask_by_diameter(self, labeled_unique, diameter, oper):
        tumors_with_diameter_list = []
        debug = []
        tumors_with_diameter_mask = np.zeros(self.gt.shape)
        for i in range(1, labeled_unique[1] + 1):
            current_1_tumor = (labeled_unique[0] == i)
            num_of_voxels = current_1_tumor.sum()
            tumor_volume = num_of_voxels * self.voxel_volume
            approx_diameter = self.approximate_diameter(tumor_volume)
            if oper(approx_diameter, diameter):
                tumors_with_diameter_list.append(i)
                debug.append(num_of_voxels)
                tumors_with_diameter_mask[current_1_tumor] = 1
        tumors_with_diameter_labeled = measure.label(tumors_with_diameter_mask)
        tumors_with_diameter_labeled = (
            tumors_with_diameter_labeled,
            tumors_with_diameter_labeled.max(),
        )
        return tumors_with_diameter_labeled, tumors_with_diameter_mask, tumors_with_diameter_list

    def find_3_biggest_tumors(self, tumors_with_diameter_labeling):
        tumors_with_diameter_list = []
        tumors_with_diameter_labeling_copy = deepcopy(tumors_with_diameter_labeling)
        three_biggest = np.bincount(tumors_with_diameter_labeling_copy[0].flatten())
        three_biggest[0] = 0
        three_biggest = np.argsort(three_biggest)
        three_biggest = three_biggest[1:]
        if three_biggest.__len__() >= 3:
            three_biggest = three_biggest[-3:]

        tumors_with_diameter_mask = np.zeros(self.gt.shape)
        for i in three_biggest:
            tumors_with_diameter_list.append(i)
            tumors_with_diameter_mask[tumors_with_diameter_labeling[0] == i] = 1
        tumors_with_diameter_mask_labeled = measure.label(tumors_with_diameter_mask)
        tumors_with_diameter_mask_labeled = (
            tumors_with_diameter_mask_labeled,
            tumors_with_diameter_mask_labeled.max(),
        )
        return tumors_with_diameter_mask_labeled, tumors_with_diameter_mask, tumors_with_diameter_list


    def Detection_statistics(self, predict_lesions_with_diameter_TP, tumors_with_diameter_gt_unique,
                             tumors_with_diameter_predictions_matrix_unique, three_biggest, debug=False, ):
        all_gt_tumors = (self.unique_gt[0] >= 1)
        # detection_TP = len(list(np.unique((predict_lesions_with_diameter_TP * tumors_with_diameter_gt_unique[0])))) - 1
        detection_TP = len(list(np.unique((tumors_with_diameter_gt_unique[0] * self.predictions)))) - 1

        detection_FP = int(tumors_with_diameter_predictions_matrix_unique[1] - \
                       (len(list(np.unique(
                           ( all_gt_tumors * tumors_with_diameter_predictions_matrix_unique[0])))) - 1))
        detection_FN = int(tumors_with_diameter_gt_unique[1] - detection_TP)

        try:
            precision = detection_TP / (detection_TP + detection_FP)
        except:
            precision = 1
        try:
            recall = detection_TP / (detection_FN + detection_TP)
        except:
            recall = 1

        return detection_TP, detection_FP, detection_FN, precision, recall

    @staticmethod
    def getLargestCC(segmentation):
        labels = measure.label(segmentation, connectivity=1)
        assert (labels.max() != 0)  # assume at least 1 CC
        return labels == np.argmax(np.bincount(labels.flat)[1:]) + 1

    @staticmethod
    def CC(Map, min_size=20):
        """
        Remove Small connected component
        :param Map:
        :return:
        """
        label_img = measure.label(Map)
        cc_num = label_img.max()
        cc_areas = ndimage.sum(Map, label_img, range(cc_num + 1))
        area_mask = (cc_areas <= min_size)
        label_img[area_mask[label_img]] = 0
        return_value = measure.label(label_img)
        return return_value, return_value.max()

    @staticmethod
    def dice(gt_seg, prediction_seg):
        """
        compute dice coefficient
        :param gt_seg:
        :param prediction_seg:
        :return: dice coefficient between gt and predictions
        """
        seg1 = np.asarray(gt_seg).astype(bool)
        seg2 = np.asarray(prediction_seg).astype(bool)

        # Compute Dice coefficient
        intersection = np.logical_and(seg1, seg2)
        if seg1.sum() + seg2.sum() == 0:
            return 1
        return float(format(2. * intersection.sum() / (seg1.sum() + seg2.sum()), '.3f'))

    @staticmethod
    def approximate_diameter(tumor_volume):
        r = ((3 * tumor_volume) / (4 * np.pi)) ** (1 / 3)
        return 2 * r


def write_to_excel(sheet_name, df, writer):
    columns_order = ['Num of lesion',
                     'Dice',
                     'Dice with FN',
                     'Mean ASSD (mm)',
                     'Mean Hausdorff (mm)',
                     'Max Hausdorff (mm)',
                     'Segmentation TP (cc)',
                     'Segmentation FP (cc)',
                     'Segmentation FN (cc)',
                     'Total tumor volume GT (cc)',
                     'Total tumor volume Predictions (cc)',
                     'Delta between total tumor volumes (cc)',
                     'Delta between total tumor volumes (%)',
                     'Delta between total tumor volumes (TP only) (cc)',
                     'ABS Delta between total tumor volumes (TP only) (cc)',
                     'Delta between total tumor volumes (TP only) (%)',
                     'ABS Delta between total tumor volumes (TP only) (%)',
                     'Tumor Burden GT (%)',
                     'Tumor Burden Pred (%)',
                     'Tumor Burden Delta (%)',
                     'Detection TP (per lesion)',
                     'Detection FP (per lesion)',
                     'Detection FN (per lesion)',
                     'Precision',
                     'Recall',
                     'F1 Score']
    df = df.set_index('Filename')
    # df = df.append(df.agg(['mean', 'std', 'min', 'max', 'sum']))
    df = pd.concat([df, df.agg(['mean', 'std', 'min', 'max', 'sum'])], ignore_index=False)
    workbook = writer.book
    # cell_format = workbook.add_format()
    cell_format = workbook.add_format({'num_format': '#,##0.00'})
    cell_format.set_font_size(16)

    df.to_excel(writer, sheet_name=sheet_name, columns=columns_order, startrow=1, startcol=1, header=False, index=False)
    header_format = workbook.add_format({
        'bold': True,
        'text_wrap': True,
        'font_size': 16,
        'valign': 'top',
        'border': 1})

    max_format = workbook.add_format({
        'font_size': 16,
        'bg_color': '#E6FFCC'})
    min_format = workbook.add_format({
        'font_size': 16,
        'bg_color': '#FFB3B3'})
    last_format = workbook.add_format({
        'font_size': 16,
        'bg_color': '#C0C0C0',
        'border': 1,
        'num_format': '#,##0.00'})

    worksheet = writer.sheets[sheet_name]
    worksheet.freeze_panes(1, 1)
    worksheet.conditional_format(
        f'$B$2:${xl_col_to_name(len(columns_order))}${str(len(df.axes[0]) - 4)}',
        {
            'type': 'formula',
            'criteria': f'=B2=B${len(df.axes[0])}',
            'format': max_format,
        },
    )

    worksheet.conditional_format(
        f'$B$2:${xl_col_to_name(len(columns_order))}${str(len(df.axes[0]) - 4)}',
        {
            'type': 'formula',
            'criteria': f'=B2=B${str(len(df.axes[0]) - 1)}',
            'format': min_format,
        },
    )

    n = df.shape[0] - 5
    for col in np.arange(len(columns_order)) + 1:
        for i, measure in enumerate(['AVERAGE', 'STDEV', 'MIN', 'MAX', 'SUM'], start=1):
            col_name = xl_col_to_name(col)
            worksheet.write(f'{col_name}{n + i + 1}', f'{{={measure}({col_name}2:{col_name}{n + 1})}}')

    # fixing the f1-score to be harmonic mean
    f1_col_name = xl_col_to_name(columns_order.index('F1 Score') + 1)
    p_col_name = xl_col_to_name(columns_order.index('Precision') + 1)
    r_col_name = xl_col_to_name(columns_order.index('Recall') + 1)
    worksheet.write(f'{f1_col_name}{n + 2}', f'{{=HARMEAN({p_col_name}{n + 2}:{r_col_name}{n + 2})}}')
    for i in range(1, 5):
        worksheet.write(f'{f1_col_name}{n + 2 + i}', " ")

    for i in range(len(df.axes[0]) - 4, len(df.axes[0]) + 1):
        worksheet.set_row(i, None, last_format)

    for col_num, value in enumerate(columns_order):
        worksheet.write(0, col_num + 1, value, header_format)
    for row_num, value in enumerate(df.axes[0].astype(str)):
        worksheet.write(row_num + 1, 0, value, header_format)

    # Fix first column
    column_len = df.axes[0].astype(str).str.len().max() + df.axes[0].astype(str).str.len().max() * 0.5
    worksheet.set_column(0, 0, column_len, cell_format)

    # Fix all  the rest of the columns
    for i, col in enumerate(columns_order):
        # find length of column i
        column_len = df[col].astype(str).str.len().max()
        # Setting the length if the column header is larger
        # than the max column value length
        column_len = max(column_len, len(col))
        column_len += column_len * 0.5
        # set the column length
        worksheet.set_column(i + 1, i + 1, column_len, cell_format)


def calculate_runtime(t):
    t2 = gmtime(time() - t)
    return f'{t2.tm_hour:02.0f}:{t2.tm_min:02.0f}:{t2.tm_sec:02.0f}'


def replace_in_file_name(file_name, old_part, new_part):
    if old_part not in file_name:
        raise Exception(f'The following file doesn\'t contain the part "{old_part}": {file_name}')
    new_file_name = file_name.replace(old_part, new_part)
    if not os.path.isfile(new_file_name):
        raise Exception(f'It looks like the following file doesn\'t exist: {new_file_name}')
    return new_file_name


def calc_stats(files: Tuple[str, str, str], th: int, get_case_name: Callable[[str], str], roi_is_gt=True,
               reduce_ram_storage=False, labels_to_consider: Optional[Union[int, List[int]]] = None,
               categories_to_calculate: Tuple[str, ...] = ('>0', '>5', '>10'), min_size: int = 20):

    assert isinstance(categories_to_calculate, tuple)

    cat_to_diameters_and_oper = {
        '>0': (0, operator.gt),
        '>5': (5, operator.gt),
        '>10': (10, operator.gt),
        '<5': (5, operator.le),
        '<10': (10, operator.le),
        '5<&<10': ([5, 10], lambda c, a_b: a_b[0] < c <= a_b[1])
    }

    if categories_to_calculate == ('all',):
        categories_to_calculate = tuple(cat_to_diameters_and_oper.keys())

    assert all(t in cat_to_diameters_and_oper for t in categories_to_calculate)

    GT_path, pred_path, liver_path = files

    file_name = get_case_name(GT_path)

    one_case = tumors_statistics(liver_path, GT_path, pred_path, file_name, th, roi_is_gt=roi_is_gt,
                                 reduce_ram_storage=reduce_ram_storage, labels_to_consider=labels_to_consider, min_size=min_size)

    calculate_ASSD = True
    calculate_HD = True


    res = []
    for cat in categories_to_calculate:
        diameters, oper = cat_to_diameters_and_oper[cat]
        res.append(one_case.calculate_statistics_by_diameter(diameters, oper, three_biggest=False, calculate_ASSD=calculate_ASSD,
                                                  calculate_HD=calculate_HD))

    return tuple(res), categories_to_calculate


def write_stats(GT_paths: List[str], pred_paths: List[str], liver_paths: Union[List[None], List[str]], ths: Union[int, List[int]],
                get_case_name: Callable[[str], str], results_dir: str, excel_suffix: str = '', roi_is_gt=True,
                n_processes: Optional[int] = None, reduce_ram_storage=False, add_th_to_excel_name: bool = True,
                labels_to_consider: Optional[Union[int, List[int]]] = None,
                categories_to_calculate: Tuple[str, ...] = ('>0', '>5', '>10'), min_size=20):
    """
    Supported categories are:
        - '>0'
        - '>5'
        - '>10'
        - '<5'
        - '<10'
        - '5<&<10'
        - 'all'
    """

    assert isinstance(categories_to_calculate, tuple)

    cat_to_sheet_name = {
        '>0': 'diameter_0',
        '>5': 'diameter_5',
        '>10': 'diameter_10',
        '<5': 'diameter<5',
        '<10': 'diameter<10',
        '5<&<10': 'diameter_in_(5,10)'
    }

    if categories_to_calculate == ('all',):
        categories_to_calculate = tuple(cat_to_sheet_name.keys())

    assert all(t in cat_to_sheet_name for t in categories_to_calculate)

    files = list(zip(GT_paths, pred_paths, liver_paths))

    if not isinstance(ths, list):
        assert isinstance(ths, int)
        ths = [ths]

    os.makedirs(results_dir, exist_ok=True)

    for th in ths:

        t = time()

        print(f'Calculating threshold: {str(th)}')

        if n_processes is None:
            n_processes = cpu_count() - 2

        results = process_map(partial(calc_stats, th=th, get_case_name=get_case_name, roi_is_gt=roi_is_gt,
                                      reduce_ram_storage=reduce_ram_storage, labels_to_consider=labels_to_consider,
                                      categories_to_calculate=categories_to_calculate, min_size=min_size), files, max_workers=n_processes)
 
        if add_th_to_excel_name:
            res_file_path = f'{results_dir}/tumors_measurements_-_th_{th}{"_-_" if excel_suffix != "" else ""}{excel_suffix}.xlsx'
        else:
            res_file_path = f'{results_dir}/tumors_measurements{"_-_" if excel_suffix != "" else ""}{excel_suffix}.xlsx'
        writer = pd.ExcelWriter(res_file_path, engine='xlsxwriter')

        dfs = dict([(cat, pd.DataFrame()) for cat in categories_to_calculate])

        for res, categories_to_calculate in results:
            for k, cat in enumerate(categories_to_calculate):
                dfs[cat] = pd.concat([dfs[cat], pd.DataFrame(res[k], index=[0])], ignore_index=True)

        for cat in categories_to_calculate:
            write_to_excel(cat_to_sheet_name[cat], dfs[cat], writer)


        writer.close()
        print(f'Finished th={th} in {calculate_runtime(t)} hh:mm:ss')


def scans_sort_key(file, full_path_given=True):
    if full_path_given:
        file = os.path.basename(os.path.dirname(file))
    split = file.split('_')
    return '_'.join(c for c in split if not c.isdigit()), int(split[-1]), int(split[-2]), int(split[-3])


def pairs_sort_key(file):
    file = os.path.basename(os.path.dirname(file))
    file = file.replace('BL_', '')
    bl_name, fu_name = file.split('_FU_')
    return (*scans_sort_key(bl_name, full_path_given=False), *scans_sort_key(fu_name, full_path_given=False))

