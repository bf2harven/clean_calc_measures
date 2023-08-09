import nibabel as nib
import pandas as pd
from data import *
from utils import *
from os.path import isfile
from typing import Tuple, List
from medpy.metric import hd, assd
from skimage import measure
from copy import deepcopy
from multiprocessing import Pool
import operator

old_dir = '/cs/casmip/public/for_aviv/StandAlone-tumors_Jan21/old'
old_CT_cases, old_GT_liver, old_GT_tumors = [], [], []
new_CT_cases, new_GT_liver, new_GT_tumors = [], [], []
for i, ct_file in enumerate(test_CT):
    case_name = basename(dirname(ct_file))
    old_ct = f'{old_dir}/{case_name}/scan.nii.gz'
    old_liver = f'{old_dir}/{case_name}/liver.nii.gz'
    old_tumors = f'{old_dir}/{case_name}/tumors.nii.gz'
    if all(isfile(f) for f in (old_ct, old_liver, old_tumors)):
        old_CT_cases.append(old_ct)
        old_GT_liver.append(old_liver)
        old_GT_tumors.append(old_tumors)

        new_CT_cases.append(ct_file)
        new_GT_liver.append(test_Liver_gt[i])
        new_GT_tumors.append(test_Tumors_gt[i])


class tumors_statistics():
    def __init__(self, new_roi, old_roi, gt, predictions, file_name, th):
        # Loading 3 niftis files
        self.file_name = file_name
        self.gt_nifti = nib.load(gt)
        pred_nifti = nib.load(predictions)
        new_roi_nifti = nib.load(new_roi)
        old_roi_nifti = nib.load(old_roi)

        # Getting voxel_volume
        self.pix_dims = self.gt_nifti.header.get_zooms()
        self.voxel_volume = self.pix_dims[0] * self.pix_dims[1] * self.pix_dims[2]

        # getting the 3 numpy arrays
        new_roi_temp = new_roi_nifti.get_fdata()
        self.new_roi = self.getLargestCC(new_roi_temp)
        old_roi_temp = old_roi_nifti.get_fdata()
        self.old_roi = self.getLargestCC(old_roi_temp)
        self.gt = self.gt_nifti.get_fdata() #* self.roi
        self.predictions = pred_nifti.get_fdata() #* self.roi
        self.predictions[self.predictions < th] = 0
        self.predictions[self.predictions >= th] = 1

        # unique_gt = nib.Nifti1Image(self.predictions, self.gt_nifti.affine)
        # nib.save(unique_gt, predictions.replace('Prediction', 'TH'))
        # nib.save(unique_gt, predictions.replace('Predicition', 'TH'))

        # unique lesions for gt and predictions
        self.unique_gt = self.CC(self.gt)
        self.unique_predictions = self.CC(self.predictions)

        self.num_of_lesions = self.unique_gt[1]
        self.dice_score = self.dice(self.gt, self.predictions)

    def calculate_statistics_by_diameter(self, diameter, oper=operator.gt, three_biggest=False,
                                         calculate_ADDS=False, calculate_HD=False):

        predict_lesions_touches = np.zeros(self.gt.shape)

        # calculate diameter for each lesion in GT
        tumors_with_diameter_gt_unique, tumors_with_diameter_gt, tumors_with_diameter = \
            self.mask_by_diameter(self.unique_gt, diameter, oper)

        # unique_gt = nib.Nifti1Image(tumors_with_diameter_gt_unique[0], self.gt_nifti.affine)

        # Find 3 biggest GT
        if three_biggest:
            tumors_with_diameter_gt_unique, tumors_with_diameter_gt, tumors_with_diameter = \
                self.find_3_biggest_tumors(tumors_with_diameter_gt_unique)
            unique_gt = nib.Nifti1Image(tumors_with_diameter_gt_unique[0], self.gt_nifti.affine)
            # nib.save(unique_gt, 'GT_unique.nii.gz')

        # calculate diameter for each lesion in Predictions
        tumors_with_diameter_predictions_matrix_unique, tumors_with_diameter_predictions_matrix, tumors_with_diameter_predictions = \
            self.mask_by_diameter(self.unique_predictions, diameter, oper)

        # Find 3 biggest GT
        if three_biggest:
            tumors_with_diameter_predictions_matrix_unique, tumors_with_diameter_predictions_matrix, tumors_with_diameter_predictions = \
                self.find_3_biggest_tumors(tumors_with_diameter_predictions_matrix_unique)
            unique_gt = nib.Nifti1Image(tumors_with_diameter_predictions_matrix_unique[0], self.gt_nifti.affine)
            # nib.save(unique_gt, 'Pred_unique.nii.gz')

        # Find predicted tumor that touches 1 tumor of the predicition
        # and calculating ASSDs ans Hausdorff metrices
        ASSDs: List[float] = []
        HDs: List[float] = []
        for i in tumors_with_diameter:
            current_1_tumor = (self.unique_gt[0] == i)
            unique_predictions = list(np.unique((current_1_tumor * tumors_with_diameter_predictions_matrix_unique[0])))
            unique_predictions.pop(0)
            for j in unique_predictions:
                predict_lesions_touches[tumors_with_diameter_predictions_matrix_unique[0] == j] = 1
                if calculate_ADDS:
                    ASSDs += [assd(current_1_tumor, tumors_with_diameter_predictions_matrix_unique[0] == j,
                                   voxelspacing=self.pix_dims, connectivity=2)]
                if calculate_HD:
                    HDs += [hd(current_1_tumor, tumors_with_diameter_predictions_matrix_unique[0] == j,
                               voxelspacing=self.pix_dims, connectivity=2)]
        mean_ASSDs = float(format(np.mean(ASSDs), '.3f')) if ASSDs else np.nan
        mean_HDs = float(format(np.mean(HDs), '.3f')) if HDs else np.nan
        max_HDs = float(format(np.max(HDs), '.3f')) if HDs else np.nan

        # Segmentation statistics

        seg_TP, seg_FP, seg_FN = \
            self.Segmentation_statistics(tumors_with_diameter_gt, predict_lesions_touches, debug=False)

        Total_tumor_GT = float(format((tumors_with_diameter_gt > 0).sum() * self.voxel_volume * 0.001, '.3f'))
        Total_tumor_pred = float(
            format((tumors_with_diameter_predictions_matrix > 0).sum() * self.voxel_volume * 0.001, '.3f'))
        new_Liver_cc = self.new_roi.sum() * self.voxel_volume * 0.001
        old_Liver_cc = self.old_roi.sum() * self.voxel_volume * 0.001

        if Total_tumor_GT == 0:
            delta_percentage = 0
        else:
            delta_percentage = ((Total_tumor_GT - Total_tumor_pred) / (Total_tumor_GT + Total_tumor_pred)) * 100
        # Detection statistics
        try:
            detection_TP, detection_FP, detection_FN, precision, recall = \
                self.Detection_statistics(predict_lesions_touches, tumors_with_diameter_gt_unique,
                                          tumors_with_diameter_predictions_matrix_unique, three_biggest)
        except:
            print('im here')

        return {'Filename': self.file_name,
                'Num_of_lesion': len(tumors_with_diameter),
                'Num_of_lesion_in_pred': len(tumors_with_diameter_predictions),
                'Dice': self.dice(tumors_with_diameter_gt, predict_lesions_touches),
                'Segmentation TP (cc)': float(format(seg_TP.sum() * self.voxel_volume * 0.001, '.3f')),
                'Segmentation FP (cc)': float(format(seg_FP.sum() * self.voxel_volume * 0.001, '.3f')),
                'Segmentation FN (cc)': float(format(seg_FN.sum() * self.voxel_volume * 0.001, '.3f')),
                'Total tumor volume GT (cc)': Total_tumor_GT,
                'Total tumor volume Predictions (cc)': Total_tumor_pred,
                'Delta between total tumor volumes (cc)': Total_tumor_GT - Total_tumor_pred,
                'Delta between total tumor volumes (%)': delta_percentage,
                'Tumor Burden GT (%)': float(format(Total_tumor_GT / new_Liver_cc, '.3f')) * 100,
                'Tumor Burden Pred (%)': float(format(Total_tumor_pred / old_Liver_cc, '.3f')) * 100,
                'Tumor Burden Delta (%)': float(format(Total_tumor_GT / new_Liver_cc - Total_tumor_pred / old_Liver_cc, '.3f')) * 100,
                'Detection TP (per lesion)': detection_TP,
                'Detection FP (per lesion)': detection_FP,
                'Detection FN (per lesion)': detection_FN,
                'Precision': float(format(precision, '.3f')),
                'Recall': float(format(recall, '.3f')),
                'Mean ASSD (mm)': mean_ASSDs,
                'Mean Hausdorff (mm)': mean_HDs,
                'Max Hausdorff (mm)': max_HDs,
                'Num of new tumor voxels out liver': np.logical_and(self.gt, np.logical_not(self.new_roi)).sum(),
                'Num of old tumor voxels out liver': np.logical_and(self.predictions, np.logical_not(self.old_roi)).sum()}
        # 'diameter': diameter,
        # 'oper': oper}

    def Segmentation_statistics(self, tumors_with_diameter_gt, predict_lesions_touches, debug=False):
        seg_TP = (tumors_with_diameter_gt * predict_lesions_touches)
        seg_FP = (predict_lesions_touches - (tumors_with_diameter_gt * predict_lesions_touches))
        seg_FN = (tumors_with_diameter_gt - (tumors_with_diameter_gt * predict_lesions_touches))
        if debug:
            unique_gt = nib.Nifti1Image(seg_FP, self.gt_nifti.affine)
            nib.save(unique_gt, 'FP.nii.gz')
            unique_gt = nib.Nifti1Image(seg_FN, self.gt_nifti.affine)
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
        tumors_with_diameter_labeled = tuple((tumors_with_diameter_labeled, tumors_with_diameter_labeled.max()))
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
        tumors_with_diameter_mask_labeled = tuple(
            (tumors_with_diameter_mask_labeled, tumors_with_diameter_mask_labeled.max()))
        return tumors_with_diameter_mask_labeled, tumors_with_diameter_mask, tumors_with_diameter_list

    @staticmethod
    def Detection_statistics(predict_lesions_touches, tumors_with_diameter_gt_unique,
                             tumors_with_diameter_predictions_matrix_unique, three_biggest, debug=False, ):

        detection_TP = len(list(np.unique((predict_lesions_touches * tumors_with_diameter_gt_unique[0])))) - 1
        if three_biggest:
            detection_FP = tumors_with_diameter_predictions_matrix_unique[1] - detection_TP
        else:
            detection_FP = tumors_with_diameter_predictions_matrix_unique[1] - \
                           (len(list(np.unique(
                               (predict_lesions_touches * tumors_with_diameter_predictions_matrix_unique[0])))) - 1)
        detection_FN = tumors_with_diameter_gt_unique[1] - detection_TP

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
        labels = measure.label(segmentation)
        assert (labels.max() != 0)  # assume at least 1 CC
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
        return largestCC

    @staticmethod
    def CC(Map):
        """
        Remove Small connected component
        :param Map:
        :return:
        """
        label_img = measure.label(Map)
        cc_num = label_img.max()
        cc_areas = ndimage.sum(Map, label_img, range(cc_num + 1))
        area_mask = (cc_areas <= 10)
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
        seg1 = np.asarray(gt_seg).astype(np.bool)
        seg2 = np.asarray(prediction_seg).astype(np.bool)

        # Compute Dice coefficient
        intersection = np.logical_and(seg1, seg2)
        if seg1.sum() + seg2.sum() == 0:
            return 1
        return float(format(2. * intersection.sum() / (seg1.sum() + seg2.sum()), '.3f'))

    @staticmethod
    def approximate_diameter(tumor_volume):
        r = ((3 * tumor_volume) / (4 * np.pi)) ** (1 / 3)
        diameter = 2 * r
        return diameter


class liver_statistics():
    def __init__(self, roi, gt, predictions, file_name, th):
        # Loading 3 niftis files
        self.file_name = file_name
        self.gt_nifti = nib.load(gt)
        pred_nifti = nib.load(predictions)
        roi_nifti = nib.load(roi)

        # Getting voxel_volume
        self.pix_dims = self.gt_nifti.header.get_zooms()
        self.voxel_volume = self.pix_dims[0] * self.pix_dims[1] * self.pix_dims[2]

        # getting the 3 numpy arrays
        roi_temp = roi_nifti.get_fdata()
        self.roi = self.getLargestCC(roi_temp)
        self.gt = self.getLargestCC(self.gt_nifti.get_fdata())  # * self.roi
        self.predictions = pred_nifti.get_fdata()  # * self.roi
        self.predictions[self.predictions < th] = 0
        self.predictions[self.predictions >= th] = 1
        self.predictions = self.getLargestCC(self.predictions)

        # unique_gt = nib.Nifti1Image(self.predictions, self.gt_nifti.affine)
        # nib.save(unique_gt, predictions.replace('Predicition', 'TH'))

        # unique lesions for gt and predictions
        self.unique_gt = self.CC(self.gt)
        self.unique_predictions = self.CC(self.predictions)

        self.num_of_lesions = self.unique_gt[1]
        self.dice_score = self.dice(self.gt, self.predictions)

    def calculate_statistics_by_diameter(self, diameter, oper=operator.gt, three_biggest=False, calculate_ADDS=False,
                                         calculate_HD=False):

        predict_lesions_touches = np.zeros(self.gt.shape)

        # calculate diameter for each lesion in GT
        tumors_with_diameter_gt_unique, tumors_with_diameter_gt, tumors_with_diameter = \
            self.mask_by_diameter(self.unique_gt, diameter, oper)

        # unique_gt = nib.Nifti1Image(tumors_with_diameter_gt_unique[0], self.gt_nifti.affine)
        # nib.save(unique_gt, 'FU_adi.nii.gz')

        # Find 3 biggest GT
        if three_biggest:
            tumors_with_diameter_gt_unique, tumors_with_diameter_gt, tumors_with_diameter = \
                self.find_3_biggest_tumors(tumors_with_diameter_gt_unique)
            unique_gt = nib.Nifti1Image(tumors_with_diameter_gt_unique[0], self.gt_nifti.affine)
            # nib.save(unique_gt, 'GT_unique.nii.gz')

        # calculate diameter for each lesion in Predictions
        tumors_with_diameter_predictions_matrix_unique, tumors_with_diameter_predictions_matrix, tumors_with_diameter_predictions = \
            self.mask_by_diameter(self.unique_predictions, diameter, oper)

        # Find 3 biggest GT
        if three_biggest:
            tumors_with_diameter_predictions_matrix_unique, tumors_with_diameter_predictions_matrix, tumors_with_diameter_predictions = \
                self.find_3_biggest_tumors(tumors_with_diameter_predictions_matrix_unique)
            unique_gt = nib.Nifti1Image(tumors_with_diameter_predictions_matrix_unique[0], self.gt_nifti.affine)
            # nib.save(unique_gt, 'Pred_unique.nii.gz')

        # Find predicted tumor that touches 1 tumor of the predicition
        # and calculating ASSDs ans Hausdorff metrices
        ASSDs = []
        HDs = []
        for i in tumors_with_diameter:
            current_1_tumor = (self.unique_gt[0] == i)
            unique_predictions = list(np.unique((current_1_tumor * tumors_with_diameter_predictions_matrix_unique[0])))
            unique_predictions.pop(0)
            for j in unique_predictions:
                predict_lesions_touches[tumors_with_diameter_predictions_matrix_unique[0] == j] = 1
                if calculate_ADDS:
                    ASSDs += [assd(current_1_tumor, tumors_with_diameter_predictions_matrix_unique[0] == j,
                                   voxelspacing=self.pix_dims, connectivity=2)]
                if calculate_HD:
                    HDs += [hd(current_1_tumor, tumors_with_diameter_predictions_matrix_unique[0] == j,
                               voxelspacing=self.pix_dims, connectivity=2)]
        mean_ASSDs = float(format(np.mean(ASSDs), '.3f')) if ASSDs else np.nan
        mean_HDs = float(format(np.mean(HDs), '.3f')) if HDs else np.nan
        max_HDs = float(format(np.max(HDs), '.3f')) if HDs else np.nan

        # Segmentation statistics

        seg_TP, seg_FP, seg_FN = \
            self.Segmentation_statistics(tumors_with_diameter_gt, predict_lesions_touches, debug=False)

        Total_tumor_GT = float(format((tumors_with_diameter_gt > 0).sum() * self.voxel_volume * 0.001, '.3f'))
        Total_tumor_pred = float(
            format((tumors_with_diameter_predictions_matrix > 0).sum() * self.voxel_volume * 0.001, '.3f'))
        Liver_cc = self.roi.sum() * self.voxel_volume * 0.001

        if Total_tumor_GT == 0:
            delta_percentage = 0
        else:
            delta_percentage = ((Total_tumor_GT - Total_tumor_pred) / (Total_tumor_GT + Total_tumor_pred)) * 100
        # Detection statistics
        try:
            detection_TP, detection_FP, detection_FN, precision, recall = \
                self.Detection_statistics(predict_lesions_touches, tumors_with_diameter_gt_unique,
                                          tumors_with_diameter_predictions_matrix_unique, three_biggest)
        except:
            print('im here')

        return {'Filename': self.file_name,
                'Num_of_lesion': len(tumors_with_diameter),
                'Dice': self.dice(tumors_with_diameter_gt, predict_lesions_touches),
                'Segmentation TP (cc)': float(format(seg_TP.sum() * self.voxel_volume * 0.001, '.3f')),
                'Segmentation FP (cc)': float(format(seg_FP.sum() * self.voxel_volume * 0.001, '.3f')),
                'Segmentation FN (cc)': float(format(seg_FN.sum() * self.voxel_volume * 0.001, '.3f')),
                'Total tumor volume GT (cc)': Total_tumor_GT,
                'Total tumor volume Predictions (cc)': Total_tumor_pred,
                'Delta between total Liver volumes (cc)': Total_tumor_GT - Total_tumor_pred,
                'Delta between total Liver volumes (%)': delta_percentage,
                'Tumor Burden GT (%)': float(format(Total_tumor_GT / Liver_cc, '.3f')) * 100,
                'Tumor Burden Pred (%)': float(format(Total_tumor_pred / Liver_cc, '.3f')) * 100,
                'Tumor Burden Delta (%)': float(format((Total_tumor_GT - Total_tumor_pred) / Liver_cc, '.3f')) * 100,
                'Detection TP': detection_TP,
                'Detection FP': detection_FP,
                'Detection FN': detection_FN,
                'Precision': float(format(precision, '.3f')),
                'Recall': float(format(recall, '.3f')),
                'Mean ASSD (mm)': mean_ASSDs,
                'Mean Hausdorff (mm)': mean_HDs,
                'Max Hausdorff (mm)': max_HDs}
        # 'diameter': diameter,
        # 'oper': oper}

    def Segmentation_statistics(self, tumors_with_diameter_gt, predict_lesions_touches, debug=False):
        seg_TP = (tumors_with_diameter_gt * predict_lesions_touches)
        seg_FP = (predict_lesions_touches - (tumors_with_diameter_gt * predict_lesions_touches))
        seg_FN = (tumors_with_diameter_gt - (tumors_with_diameter_gt * predict_lesions_touches))
        if debug:
            unique_gt = nib.Nifti1Image(seg_FP, self.gt_nifti.affine)
            nib.save(unique_gt, 'FP.nii.gz')
            unique_gt = nib.Nifti1Image(seg_FN, self.gt_nifti.affine)
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
        tumors_with_diameter_labeled = tuple((tumors_with_diameter_labeled, tumors_with_diameter_labeled.max()))
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
        tumors_with_diameter_mask_labeled = tuple(
            (tumors_with_diameter_mask_labeled, tumors_with_diameter_mask_labeled.max()))
        return tumors_with_diameter_mask_labeled, tumors_with_diameter_mask, tumors_with_diameter_list

    @staticmethod
    def Detection_statistics(predict_lesions_touches, tumors_with_diameter_gt_unique,
                             tumors_with_diameter_predictions_matrix_unique, three_biggest, debug=False, ):

        detection_TP = len(list(np.unique((predict_lesions_touches * tumors_with_diameter_gt_unique[0])))) - 1
        if three_biggest:
            detection_FP = tumors_with_diameter_predictions_matrix_unique[1] - detection_TP
        else:
            detection_FP = tumors_with_diameter_predictions_matrix_unique[1] - \
                           (len(list(np.unique(
                               (predict_lesions_touches * tumors_with_diameter_predictions_matrix_unique[0])))) - 1)
        detection_FN = tumors_with_diameter_gt_unique[1] - detection_TP

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
        labels = measure.label(segmentation)
        assert (labels.max() != 0)  # assume at least 1 CC
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
        return largestCC

    @staticmethod
    def CC(Map):
        """
        Remove Small connected component
        :param Map:
        :return:
        """
        label_img = measure.label(Map)
        cc_num = label_img.max()
        cc_areas = ndimage.sum(Map, label_img, range(cc_num + 1))
        area_mask = (cc_areas <= 10)
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
        seg1 = np.asarray(gt_seg).astype(np.bool)
        seg2 = np.asarray(prediction_seg).astype(np.bool)

        # Compute Dice coefficient
        intersection = np.logical_and(seg1, seg2)
        if seg1.sum() + seg2.sum() == 0:
            return 1
        return float(format(2. * intersection.sum() / (seg1.sum() + seg2.sum()), '.3f'))

    @staticmethod
    def approximate_diameter(tumor_volume):
        r = ((3 * tumor_volume) / (4 * np.pi)) ** (1 / 3)
        diameter = 2 * r
        return diameter


def write_to_excel(sheet_name, df, writer):
    # df.rename(columns={'Mean Hausdorff (mm)': 'Hausdorff (mm)',
    #            'Mean ASSD (mm)': 'ASSD (mm)',
    #            'Num_of_lesion': 'Num_of_Connected_Components'}, inplace=True)
    # df = df[['Filename', 'Num_of_Connected_Components', 'Dice', 'ASSD (mm)', 'Hausdorff (mm)']]
    # columns_order = ['Num_of_Connected_Components',
    #                  'Dice',
    #                  'ASSD (mm)',
    #                  # 'Mean ASSD (mm)',
    #                  'Hausdorff (mm)',
    #                  # 'Mean Hausdorff (mm)',
    #                  # 'Max Hausdorff (mm)',
    #                  # 'Segmentation TP (cc)',
    #                  # 'Segmentation FP (cc)',
    #                  # 'Segmentation FN (cc)',
    #                  # 'Total tumor volume GT (cc)',
    #                  # 'Total tumor volume Predictions (cc)',
    #                  # 'Delta between total Liver volumes (cc)',
    #                  # 'Delta between total Liver volumes (%)',
    #                  # 'Tumor Burden GT (%)',
    #                  # 'Tumor Burden Pred (%)',
    #                  # 'Tumor Burden Delta (%)',
    #                  # 'Detection TP',
    #                  # 'Detection FP',
    #                  # 'Detection FN',
    #                  # 'Precision',
    #                  # 'Recall'
    #                  ]
    columns_order = ['Liver Dice',
                     'New liver volume (cc)',
                     'Old liver volume (cc)',
                     'Delta between Liver volumes (cc)',
                     'Delta between Liver volumes (%)',
                     'New tumors num of lesions',
                     'Old tumors num of lesions',
                     'Delta between num of lesions',
                     'Tumors Dice',
                     'New tumors volume (cc)',
                     'Old tumors volume (cc)',
                     'Delta between Tumors volumes (cc)',
                     'Delta between Tumors volumes (%)',
                     'New Tumor Burden (%)',
                     'Old Tumor Burden (%)',
                     'Delta Tumor Burden (%)',
                     'Num of new tumor voxels out liver',
                     'Num of old tumor voxels out liver']
    df = df.set_index('Filename')
    # df = df.append(df.agg(['mean', 'std', 'min', 'max', 'sum']))
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

    # max_format = workbook.add_format({
    #     'font_size': 16,
    #     'bg_color': '#E6FFCC'})
    # min_format = workbook.add_format({
    #     'font_size': 16,
    #     'bg_color': '#FFB3B3'})
    # last_format = workbook.add_format({
    #     'font_size': 16,
    #     'bg_color': '#C0C0C0',
    #     'border': 1,
    #     'num_format': '#,##0.00'})

    worksheet = writer.sheets[sheet_name]
    worksheet.freeze_panes(1, 1)
    # worksheet.conditional_format('$B$2:$E$' + str(len(df.axes[0]) - 4), {'type': 'formula',
    #                                                                      'criteria': '=B2=B$' + str(len(df.axes[0])),
    #                                                                      'format': max_format})

    # worksheet.conditional_format('$B$2:$E$' + str(len(df.axes[0]) - 4), {'type': 'formula',
    #                                                                      'criteria': '=B2=B$' + str(
    #                                                                          len(df.axes[0]) - 1),
    #                                                                      'format': min_format})

    # for i in range(len(df.axes[0]) - 4, len(df.axes[0]) + 1):
    #     worksheet.set_row(i, None, last_format)

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


if __name__ == '__main__':

    n = 4
    old_CT_cases, old_GT_liver, old_GT_tumors ,new_CT_cases, new_GT_liver, new_GT_tumors = \
        old_CT_cases[:n], old_GT_liver[:n], old_GT_tumors[:n], new_CT_cases[:n], new_GT_liver[:n], new_GT_tumors[:n]

    # -------------------------------------------------------------------------------------------------------
    # new_dir = old_dir.replace('/old', '/new')
    # for i, new_liver_file_name in enumerate(new_GT_liver):
    #     case_name = basename(dirname(new_liver_file_name))
    #     os.makedirs(f'{new_dir}/{case_name}', exist_ok=True)
    #     new_liver_case, new_liver_file = load_nifti_data(new_liver_file_name)
    #     new_tumors_case, _ = load_nifti_data(new_GT_tumors[i])
    #     new_liver_and_tumors_case = new_liver_case.copy()
    #     new_liver_and_tumors_case[new_tumors_case == 1] = 2
    #     new_liver_and_tumors_case[np.logical_and(new_tumors_case, np.logical_not(new_liver_case))] = 3
    #     nib.save(nib.Nifti1Image(new_liver_and_tumors_case, new_liver_file.affine), f'{new_dir}/{case_name}/liver_and_tumors.nii.gz')
    # exit(0)
    # -------------------------------------------------------------------------------------------------------

    def calculate_stats(file_names: str):
        old_liver, old_tumors, new_liver, new_tumors = file_names
        case_name = basename(dirname(old_liver))

        tumors_stats = tumors_statistics(new_liver, old_liver, new_tumors, old_tumors, case_name, 1)
        liver_stats = liver_statistics(new_liver, new_liver, old_liver, case_name, 1)

        liver_stats_res = liver_stats.calculate_statistics_by_diameter(0, calculate_ADDS=False, calculate_HD=False)
        tumors_stats_res = tumors_stats.calculate_statistics_by_diameter(0, three_biggest=False, calculate_ADDS=False, calculate_HD=False)

        res = {'Filename': case_name}
        res['Liver Dice'] = liver_stats_res['Dice']
        res['New liver volume (cc)'] = liver_stats_res['Total tumor volume GT (cc)']
        res['Old liver volume (cc)'] = liver_stats_res['Total tumor volume Predictions (cc)']
        res['Delta between Liver volumes (cc)'] = liver_stats_res['Delta between total Liver volumes (cc)']
        res['Delta between Liver volumes (%)'] = liver_stats_res['Delta between total Liver volumes (%)']
        res['New tumors num of lesions'] = tumors_stats_res['Num_of_lesion']
        res['Old tumors num of lesions'] = tumors_stats_res['Num_of_lesion_in_pred']
        res['Delta between num of lesions'] = res['New tumors num of lesions'] - res['Old tumors num of lesions']
        res['Tumors Dice'] = tumors_stats_res['Dice']
        res['New tumors volume (cc)'] = tumors_stats_res['Total tumor volume GT (cc)']
        res['Old tumors volume (cc)'] = tumors_stats_res['Total tumor volume Predictions (cc)']
        res['Delta between Tumors volumes (cc)'] = tumors_stats_res['Delta between total tumor volumes (cc)']
        res['Delta between Tumors volumes (%)'] = tumors_stats_res['Delta between total tumor volumes (%)']
        res['New Tumor Burden (%)'] = tumors_stats_res['Tumor Burden GT (%)']
        res['Old Tumor Burden (%)'] = tumors_stats_res['Tumor Burden Pred (%)']
        res['Delta Tumor Burden (%)'] = tumors_stats_res['Tumor Burden Delta (%)']
        res['Num of new tumor voxels out liver'] = tumors_stats_res['Num of new tumor voxels out liver']
        res['Num of old tumor voxels out liver'] = tumors_stats_res['Num of old tumor voxels out liver']

        return res

    with Pool() as pool:
        results = pool.map(calculate_stats, zip(old_GT_liver, old_GT_tumors, new_GT_liver, new_GT_tumors))

    final_results = pd.DataFrame()
    for res in results:
        final_results = final_results.append(res, ignore_index=True)

    writer = pd.ExcelWriter('changes_in_GT.xlsx', engine='xlsxwriter')
    write_to_excel('sheet_1', final_results, writer)
    writer.save()