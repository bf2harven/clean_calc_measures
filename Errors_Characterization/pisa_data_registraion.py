import os
from glob import glob
import pandas as pd
import open3d as o3d
from scipy.ndimage import affine_transform
from time import time
import numpy as np
from skimage.morphology import disk
from nibabel import affines
from skimage.morphology import binary_dilation
# from clinical_data_study import get_gt_tumors, extract_case_name_func
import nibabel as nib
from ICP_registration_step import ICP_registration

import sys
sys.path.append('/cs/casmip/rochman/Errors_Characterization')
from registeration_by_folder import liver_registeration
from ICL_matching import execute_ICP
from utils import load_nifti_data, get_connected_components, match_2_cases_v5, replace_in_file_name
from matching_graphs import save_matching_graph

if __name__ == '__main__':

    patients_cases = sorted(glob('/home/rochman/Documents/src/data/pisa_data_for_retraining/*'))

    # todo delete
    # patients_cases = [p for p in patients_cases if p.endswith('Antonio Goddi - 1')]

    def liver_volume_in_CC(liver_f):
        liver, f = load_nifti_data(liver_f)

        pix_dims = f.header.get_zooms()
        voxel_volume = pix_dims[0] * pix_dims[1] * pix_dims[2]

        return liver.sum() * voxel_volume / 1000

    def create_pair(bl_folder, fu_folder):
        def create_data(folder):
            return (os.path.join(folder, 'scan.nii.gz'),
                    os.path.join(folder, 'liver_pred.nii.gz'),
                    os.path.join(folder, 'tumors.nii.gz'),
                    "_-_".join(folder.split('/')[-2:]))

        return (create_data(bl_folder), create_data(fu_folder))

    pairs = []
    for p in patients_cases:
        p_cs = glob(f'{p}/*')
        for i in range(len(p_cs) - 1):
            for j in range(i + 1, len(p_cs)):
                pairs.append(create_pair(p_cs[i], p_cs[j]))
                pairs.append(create_pair(p_cs[j], p_cs[i]))

    print('filtering pairs...')
    filtered_pairs = []
    for i, p in enumerate(pairs):
        (_, bl_liver, _, bl_name), (_, fu_liver, _, fu_name) = p

        # todo delete
        # if f'BL_{bl_name}_FU_{fu_name}' != 'BL_Antonio Goddi - 1_-_2015-01-27_FU_Antonio Goddi - 1_-_2015-08-19':
        #     continue

        bl_liver_volume = liver_volume_in_CC(bl_liver)
        fu_liver_volume = liver_volume_in_CC(fu_liver)
        if abs(bl_liver_volume - fu_liver_volume) >= 500:
            print(i + 1, f'({bl_name}, {fu_name}) failed')
        else:
            filtered_pairs.append(p)
            print(i + 1, f'({bl_name}, {fu_name}) OK')
    pairs = filtered_pairs
    print(f'\nThere are {len(pairs)} relevant pairs\n')

    # register
    print('\n------------------ Applying basic registration ------------------\n')
    res_dir = '/home/rochman/Documents/src/data/pisa_data_pairs_for_retraining'
    for p in pairs:
        os.makedirs(res_dir, exist_ok=True)
        (bl_CT, bl_liver, bl_tumors, bl_name), (fu_CT, fu_liver, fu_tumors, fu_name) = p

        register_class = liver_registeration([bl_CT], [bl_liver], [bl_tumors], [fu_CT], [fu_liver], [fu_tumors],
                                             dest_path=res_dir, bl_name=bl_name, fu_name=fu_name)

        register_class.affine_registeration(stop_before_bspline=True)

        print('-----------------')

    print('\n------------------ Applying ICP registration ------------------\n')

    def ICP_reg(pair_path):

        # load liver segmentations
        bl_liver_file = f'{pair_path}/BL_Scan_Liver.nii.gz'
        fu_liver_file = f'{pair_path}/FU_Scan_Liver.nii.gz'
        bl_liver, _ = load_nifti_data(bl_liver_file)
        fu_liver, file = load_nifti_data(fu_liver_file)

        # load tumors segmentation
        bl_tumors_file = f'{pair_path}/BL_Scan_Tumors.nii.gz'
        bl_labeled_tumors, _ = load_nifti_data(bl_tumors_file)

        # load the BL ct
        bl_ct, _ = load_nifti_data(f'{pair_path}/BL_Scan_CT.nii.gz')

        (transformed_bl_liver, transformed_bl_ct, transformed_bl_labeled_tumors), _ = ICP_registration(bl_liver, fu_liver, file.affine, bl_ct, bl_labeled_tumors)

        # save registered files and labels
        nib.save(nib.Nifti1Image(transformed_bl_ct, file.affine), f'{pair_path}/improved_registration_BL_Scan_CT.nii.gz')
        nib.save(nib.Nifti1Image(transformed_bl_liver, file.affine), f'{pair_path}/improved_registration_BL_Scan_Liver.nii.gz')
        nib.save(nib.Nifti1Image(transformed_bl_labeled_tumors, file.affine), f'{pair_path}/improved_registration_BL_Scan_Tumors_labeled.nii.gz')

    pair_paths = glob(f'{res_dir}/*')

    # todo delete
    # pair_paths = [p for p in pair_paths if p.endswith('BL_Antonio Goddi - 1_-_2015-01-27_FU_Antonio Goddi - 1_-_2015-08-19')]

    # ICP and matching
    for i, p_path in enumerate(pair_paths):
        print(i + 1, f'ICP for {os.path.basename(p_path)}')
        ICP_reg(p_path)