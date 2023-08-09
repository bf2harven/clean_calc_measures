from utils import *
from matching_excluded_cases import fine_tuning_registration, match
from tqdm import tqdm
from notifications import notify
from tqdm.contrib.concurrent import process_map
from datetime import date
from matching_graphs import load_matching_graph


def patient_BL_and_FU(pair_name):
        pair_name = pair_name.replace('BL_', '')
        bl, fu = pair_name.split('_FU_')
        patient = '_'.join(c for c in bl.split('_') if not c.isdigit())
        return patient, bl, fu


def sort_key(name):
    split = name.split('_')
    return '_'.join(c for c in split if not c.isdigit()), int(split[-1]), int(split[-2]), int(split[-3])


def is_in_order(name1, name2):
        key1 = sort_key(name1)
        key2 = sort_key(name2)
        if key1[1] > key2[1]:
            return False
        if key1[1] == key2[1]:
                if key1[2] > key2[2]:
                        return False
                if key1[2] == key2[2] and key1[3] > key2[3]:
                        return False
        return True


def diff_in_days(bl_name, fu_name):
    _, bl_y, bl_m, bl_d = sort_key(bl_name)
    _, fu_y, fu_m, fu_d = sort_key(fu_name)
    bl_date = date(bl_y, bl_m, bl_d)
    fu_date = date(fu_y, fu_m, fu_d)
    return abs((fu_date - bl_date).days)


def score_for_pair(bl_tumors: np.ndarray, fu_tumors: np.ndarray, bl_voxel_to_real_space_trans, fu_voxel_to_real_space_trans) -> float:
    return 1 / (min(get_minimum_distance_between_CCs(bl_tumors, bl_voxel_to_real_space_trans, max_points_per_CC=5000, seed=42),
                    get_minimum_distance_between_CCs(fu_tumors, fu_voxel_to_real_space_trans, max_points_per_CC=5000, seed=42)) + 1e-5)


def calc_score_for_pairs(pair_path):
    bl_tumors_file = f'{pair_path}/BL_Scan_Tumors.nii.gz'
    fu_tumors_file = f'{pair_path}/FU_Scan_Tumors.nii.gz'

    bl_liver_file = f'{pair_path}/BL_Scan_Liver.nii.gz'
    fu_liver_file = f'{pair_path}/FU_Scan_Liver.nii.gz'

    bl_tumors, bl_file = load_nifti_data(bl_tumors_file)
    fu_tumors, fu_file = load_nifti_data(fu_tumors_file)

    bl_liver, _ = load_nifti_data(bl_liver_file)
    fu_liver, _ = load_nifti_data(fu_liver_file)

    liver_vol_diff = abs((bl_liver > 0).sum() * np.prod(bl_file.header.get_zooms()) - (fu_liver > 0).sum() * np.prod(fu_file.header.get_zooms())) / 1000

    case_name = os.path.basename(pair_path)
    different_in_days = diff_in_days(*patient_BL_and_FU(case_name)[1:])

    return (case_name, liver_vol_diff, different_in_days, score_for_pair(bl_tumors, fu_tumors, bl_file.affine, fu_file.affine))

