import os
import sys
sys.path.append('/cs/casmip/public/adi/site-packages/SimpleITK-2.0.0rc2.dev908+g8244e-py3.6-linux-x86_64.egg')
import SimpleITK as sitk

from os import path


import nibabel as nib
import numpy as np

import glob
from tqdm.contrib.concurrent import process_map


def load_nifti(nifti_file_path, min_clip=0, max_clip=1,segmentation = False):
    try:
        nifti_file = nib.load(nifti_file_path)
    except:
        nifti_file = nib.load(nifti_file_path.replace('liver','liver_pred'))


    nifti_file = nib.as_closest_canonical(nifti_file)

    case = nifti_file.get_fdata()
    if segmentation:
        if case.max()>1:
            case = np.clip(case, min_clip, max_clip)
            nifti_file = nib.Nifti1Image(case, nifti_file.affine)
            nib.save(nifti_file,nifti_file_path)
    else:
        case = np.clip(case, min_clip, max_clip)
    return case


def bbox2_3D(img):
    x = np.any(img, axis=(1, 2))
    y = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))

    xmin, xmax = np.where(x)[0][[0, -1]]
    ymin, ymax = np.where(y)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    return xmin, xmax, ymin, ymax, zmin, zmax



# writing to csv file
def sorting(string):
    return string[-4:] + string[-7:-5]


class liver_registeration():
    def __init__(self,CT_BL, Livers_BL, Tumors_BL, CT_FU, Livers_FU,Tumors_FU, dest_path='/tmp/', bl_name='', fu_name='' ,reversed=False):
        """
        :param CT_BL:
        :param Livers_BL:
        :param Tumors_BL:
        :param CT_FU:
        :param Livers_FU:
        :param dest_path:
        :param reversed:
        """
        self.CT_BL = CT_BL
        self.Livers_BL = Livers_BL
        self.Tumors_BL = Tumors_BL
        self.CT_FU = CT_FU
        self.Livers_FU = Livers_FU
        self.Tumors_FU = Tumors_FU
        self.dest_path= dest_path
        self.bl_name = bl_name
        self.fu_name = fu_name
        self.reversed=reversed
        self.case_name = 'BL_' + self.bl_name + '_FU_' + self.fu_name

    def registeration(self,trans_type='translation',liver_mask=True):
        for i, BL_Scan in enumerate(self.CT_BL):
            Baseline = BL_Scan
            # Baseline = '/cs/labs/josko/aszeskin/case_01/BL_Cropped_CT_Canonical.nii.gz'
            FollowUp = self.CT_FU[i]
            # if self.reversed:
            #     case_name = Baseline.split("/")[-2] + '_Reversed'
            # else:
            #     case_name = FollowUp.split("/")[-2]

            case_name = self.case_name
            print(f'{case_name} - {trans_type}')


            Baseline_Liver = self.Livers_BL[i]
            FollowUp_Liver = self.Livers_FU[i]
            Baseline_Tumors = self.Tumors_BL[i]
            FollowUp_Tumors = self.Tumors_FU[i]

            if trans_type=='affine' or trans_type=='bspline':
                self.fix_origin(FollowUp, Baseline)
                self.fix_origin(FollowUp, Baseline_Liver)
                self.fix_origin(FollowUp, Baseline_Tumors)


            elastixImageFilter = sitk.ElastixImageFilter()
            elastixImageFilter.LogToConsoleOff()
            elastixImageFilter.SetFixedImage(sitk.ReadImage(FollowUp))
            if liver_mask:
                elastixImageFilter.SetFixedMask(sitk.ReadImage(FollowUp_Liver, sitk.sitkUInt8))
            movingLiver = sitk.ReadImage(Baseline_Liver, sitk.sitkUInt8)
            movingTumors = sitk.ReadImage(Baseline_Tumors, sitk.sitkUInt8)



            elastixImageFilter.SetMovingImage(sitk.ReadImage(Baseline))
            if liver_mask:
                elastixImageFilter.SetMovingMask(sitk.ReadImage(Baseline_Liver,sitk.sitkUInt8))

            parameterMap = sitk.GetDefaultParameterMap(trans_type)
            if trans_type == 'affine' or trans_type=='bspline':
                parameterMap['AutomaticTransformInitialization'] = ['false']
            elastixImageFilter.SetParameterMap(parameterMap)

            elastixImageFilter.LogToFileOn()
            elastixImageFilter.SetOutputDirectory('/tmp/')
            elastixImageFilter.Execute()
            temp = elastixImageFilter.GetTransformParameterMap()
            temp[0]["ResampleInterpolator"] = ["FinalNearestNeighborInterpolator"]
            resultLiver = sitk.Transformix(movingLiver, temp)
            resultTumors = sitk.Transformix(movingTumors, temp)

            os.makedirs(self.dest_path + "/" + case_name + "/", exist_ok=True)

            sitk.WriteImage(elastixImageFilter.GetResultImage(), self.dest_path + '/' + case_name + '/BL_Scan_CT.nii.gz')

            sitk.WriteImage(resultLiver, self.dest_path + '/' + case_name + '/BL_Scan_Liver.nii.gz')
            sitk.WriteImage(resultTumors, self.dest_path + '/' + case_name+ '/BL_Scan_Tumors.nii.gz')


            try:
                # shutil.copyfile(FollowUp, self.dest_path + '/' + case_name+ '/FU_Scan_CT.nii.gz')
                # shutil.copyfile(FollowUp_Liver, self.dest_path+ '/' + case_name + '/FU_Scan_Liver.nii.gz')
                # shutil.copyfile(FollowUp_Tumors, self.dest_path + '/' + case_name+ '/FU_Scan_Tumors.nii.gz')
                fu_scan_file = self.dest_path + '/' + case_name + '/FU_Scan_CT.nii.gz'
                if not os.path.isfile(fu_scan_file):
                    os.symlink(FollowUp, fu_scan_file)
                fu_liver_file = self.dest_path + '/' + case_name + '/FU_Scan_Liver.nii.gz'
                if not os.path.isfile(fu_liver_file):
                    os.symlink(FollowUp_Liver, fu_liver_file)
                fu_tumors_file = self.dest_path + '/' + case_name + '/FU_Scan_Tumors.nii.gz'
                if not os.path.isfile(fu_tumors_file):
                    os.symlink(FollowUp_Tumors, fu_tumors_file)
            except:
                print('Cannot overwrite file')


            return ([self.dest_path + '/' + case_name + '/BL_Scan_CT.nii.gz'],
                    [self.dest_path + '/' + case_name + '/BL_Scan_Liver.nii.gz'],
                    [self.dest_path + '/' + case_name+ '/BL_Scan_Tumors.nii.gz'],
                    [self.dest_path + '/' + case_name+ '/FU_Scan_CT.nii.gz'],
                    [self.dest_path+ '/' + case_name + '/FU_Scan_Liver.nii.gz'],
                    [self.dest_path+ '/' + case_name + '/FU_Scan_Tumors.nii.gz'],
                    self.dest_path,
                    self.bl_name,
                    self.fu_name,
                    self.reversed)



    def affine_registeration(self, stop_before_bspline: bool = False):
        try:
            # Starting with non-liver rigid registeration
            affine_input = liver_registeration(*self.registeration(liver_mask=False))
        except Exception as e:
            print('Couldnt register by transalation with out liver mask perhaps there is something wrong with the data')
            raise e
        try:
            # Starting with Liver affine registeration
            affine_input.registeration(liver_mask=True, trans_type='affine')
            return 'affine'

        except Exception as e:
            print(e)
            print('Couldnt register by affine, but manage to regsiter by the entire image')

            if stop_before_bspline:
                return 'translation'

            try:
                print('Trying to register by bspline')
                affine_input.registeration(liver_mask=True, trans_type='bspline')
                return 'bspline'
            except Exception as e:
                print(e)
                print('Couldnt register by bspline, but manage to regsiter by the entire image')
                return 'translation'

    def bspline_registeration(self):
        try:
            # Starting with non-liver rigid registeration
            affine_input = liver_registeration(*self.registeration(liver_mask=False))
        except Exception as e:
            print('Couldnt register by transalation with out liver mask perhaps there is something wrong with the data')
            raise e
        try:
            # Starting with Liver affine registeration
            affine_input.registeration(liver_mask=True, trans_type='bspline')
            return 'bspline'
        except:
            print('Couldnt register by bspline, but manage to regsiter by the entire image')

    def translation_registeration(self):
        try:
            self.registeration(liver_mask=True, trans_type='translation')
            return 'translation'
        except:
            print('Couldnt register by transalation, trying to register by affine registeration')
            self.affine_registeration()





    @staticmethod
    def fix_origin(origin,dest):
        origin_nifti = nib.load(origin)
        dest_nifti = nib.load(dest)
        if (np.around(origin_nifti.affine, decimals=2) == np.around(dest_nifti.affine, decimals=2)).all():
            nifti_file = nib.Nifti1Image(dest_nifti.get_fdata(), origin_nifti.affine)
            nib.save(nifti_file,dest)


def multiprocess(folder):
    case = folder.split('/')[-1]
    case_name_letters = case.split('_')[:-3]
    case_name = ''

    for i in case_name_letters:
        case_name = case_name+ i +'_'
    strings_with_substring = [string for string in all_folders2 if (case_name in string and case not in string)]

    Baseline_folder = folder
    Baseline_name = case

    for i in strings_with_substring:

        Followup_folder = i
        Followup_name = Followup_folder.split('/')[-1]
        print(Baseline_name + ' ' +Followup_name)


        CT_1 = [Baseline_folder+'/scan.nii.gz']
        if path.exists(Baseline_folder + '/liver.nii.gz'):
            CT_1_Liver = [Baseline_folder + '/liver.nii.gz']
        else:
            CT_1_Liver = [Baseline_folder + '/liver_pred.nii.gz']
        CT_1_Lesions = [Baseline_folder + '/merged.nii.gz']

        CT_2 = [Followup_folder + '/scan.nii.gz']
        if path.exists(Followup_folder + '/liver.nii.gz'):
            CT_2_Liver = [Followup_folder + '/liver.nii.gz']
        else:
            CT_2_Liver = [Followup_folder + '/liver_pred.nii.gz']
        CT_2_Lesions = [Followup_folder + '/merged.nii.gz']




        register_class = liver_registeration(CT_1, CT_1_Liver, CT_1_Lesions, CT_2, CT_2_Liver, CT_2_Lesions ,dest_path , Baseline_name ,Followup_name)
        register_class.affine_registeration()





if __name__ == '__main__':

    dest_path = '/mnt/local1/aszeskin/Data_Followup_Full_29_4_2021/'
    for paths in ['/cs/labs/josko/public/for_aviv/all_data/Ein_Kerem/K_E_*']: #['/cs/casmip/public/for_aviv/all_data/Har_Hatzofim/*', '/cs/casmip/public/for_aviv/all_data/Ein_Kerem/*']: #
        all_folders = glob.glob(paths)
        all_folders2 = glob.glob(paths)
        for folder in all_folders:
            case = folder.split('/')[-1]
            case_name_letters = case.split('_')[:-3]
            case_name = ''
            Baseline_folder = folder
            Baseline_name = case


            CT_1 = [Baseline_folder+'/scan.nii.gz']

            CT_1_Liver = [Baseline_folder + '/liver.nii.gz']
            CT_1_Lesions = [Baseline_folder + '/tumors.nii.gz']


            for i,j in enumerate(CT_1):
                try:
                    load_nifti(j)

                    merged = load_nifti(CT_1_Liver[i],segmentation=True)

                    merged[load_nifti(CT_1_Lesions[i],segmentation=True)>0] = 2

                    try:
                        nifti_file = nib.load(CT_1_Liver[i])
                    except:
                        nifti_file = nib.load(CT_1_Liver[i].replace('liver','liver_pred'))

                    nifti_file = nib.Nifti1Image(merged, nifti_file.affine)
                    nib.save(nifti_file,Baseline_folder + '/merged.nii.gz')

                except Exception as e:
                    print(j)
                    print(e)
                    try:
                        all_folders2.remove(Baseline_folder)
                    except:
                        print('Already removed')

        process_map(multiprocess, all_folders2, max_workers=20)
        # for folder in all_folders2:
        #     case = folder.split('/')[-1]
        #     case_name_letters = case.split('_')[:-3]
        #     case_name = ''
        #     for i in case_name_letters:
        #         case_name = case_name+ i +'_'
        #     strings_with_substring = [string for string in all_folders2 if (case_name in string and case not in string)]
        #
        #     Baseline_folder = folder
        #     Baseline_name = case
        #
        #     for i in strings_with_substring:
        #         print(Baseline_name + ' ' +Followup_name)
        #         Followup_folder = i
        #         Followup_name = Followup_folder.split('/')[-1]
        #
        #
        #         CT_1 = [Baseline_folder+'/scan.nii.gz']
        #
        #         CT_1_Liver = [Baseline_folder + '/merged.nii.gz']
        #         CT_1_Lesions = [Baseline_folder + '/merged.nii.gz']
        #
        #         CT_2 = [Followup_folder + '/scan.nii.gz']
        #         CT_2_Liver = [Followup_folder + '/merged.nii.gz']
        #         CT_2_Lesions = [Followup_folder + '/merged.nii.gz']
        #
        #
        #
        #
        #         register_class = liver_registeration(CT_1, CT_1_Liver, CT_1_Lesions, CT_2, CT_2_Liver, CT_2_Lesions ,dest_path , Baseline_name ,Followup_name)
        #         register_class.affine_registeration()







    # CT_1 = ['/cs/casmip/public/for_aviv/all_data/Ein_Kerem/L_E_20_06_2020/scan.nii.gz']
    # CT_1_Liver = ['/cs/casmip/public/for_aviv/all_data/Ein_Kerem/L_E_20_06_2020/liver.nii.gz']
    # CT_1_Lesions = ['/cs/casmip/public/for_aviv/all_data/Ein_Kerem/L_E_20_06_2020/tumors.nii.gz']
    #
    # CT_2 = ['/cs/casmip/public/for_aviv/all_data/Ein_Kerem/L_E_02_09_2020/scan.nii.gz']
    # CT_2_Liver = ['/cs/casmip/public/for_aviv/all_data/Ein_Kerem/L_E_02_09_2020/liver.nii.gz']
    # CT_2_Lesions = ['/cs/casmip/public/for_aviv/all_data/Ein_Kerem/L_E_02_09_2020/tumors.nii.gz']
    #
    # register_class = liver_registeration(CT_1, CT_1_Liver, CT_1_Lesions, CT_2, CT_2_Liver, CT_2_Lesions ,dest_path)
    # register_class.affine_registeration()






