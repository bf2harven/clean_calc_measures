from glob import glob
from utils import load_nifti_data
import nibabel as nib
from os.path import isfile, isdir
import os
from shutil import copyfile
from tqdm.contrib.concurrent import process_map
from time import sleep

if __name__ == '__main__':
    # new_test_set_dir = '/cs/casmip/public/for_aviv/StandAlone-tumors_Jan21/rechecked_test_set'
    # old_test_set_dir1 = '/cs/casmip/public/for_aviv/StandAlone-tumors_Jan21/test_set'
    # old_test_set_dir2 = '/cs/casmip/public/for_aviv/StandAlone-tumors_Jan21/pairwise'
    # for dir in glob('/cs/casmip/public/for_aviv/StandAlone-tumors_Jan21/Sosna/Test_validation/*'):
    #     # for kind in ['liver', 'tumors']:
    #     #     file_name = f'{dir}/combined_{kind}.nii.gz'
    #     #     if isfile(file_name):
    #     #         case, file = load_nifti_data(file_name)
    #     #         case[case == 3] = 0
    #     #         case[case == 2] = 1
    #     #         nib.save(nib.Nifti1Image(case, file.affine), f'{dir}/{kind}.nii.gz')
    #
    #
    #
    #
    #     case_name = os.path.basename(dir)
    #     old_dir = f'{old_test_set_dir1}/{case_name}'
    #     if not isdir(old_dir):
    #         old_dir = f'{old_test_set_dir2}/{case_name}'
    #     new_dir = f'{new_test_set_dir}/{case_name}'
    #     os.makedirs(new_dir, exist_ok=True)
    #     for kind in ['liver', 'tumors']:
    #         old_fixed_file_name = f'{dir}/{kind}.nii.gz'
    #         new_file_name = f'{new_dir}/{kind}.nii.gz'
    #         if isfile(old_fixed_file_name):
    #             copyfile(old_fixed_file_name, new_file_name)
    #         else:
    #             copyfile(f'{old_dir}/{kind}.nii.gz', new_file_name)
    #     copyfile(f'{old_dir}/scan.nii.gz', f'{new_dir}/scan.nii.gz')
    pass

    def f(i):
        s = 0
        for j in range(1000000*i):
            s += j**8
        return s

    process_map(f, range(50))


