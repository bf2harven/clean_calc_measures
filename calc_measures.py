from nnunet_calculate_measures import nnunet_calculate_measures
from typing import Union
import shutil
import os
class CalcMeasures:
    def __init__(self)->None:
        pass

    def calculate_measures(self, gt_labels_path:str, 
                                pred_masks_paths:str,
                                roi_masks_path:Union[str , None], 
                                region_based_inferencing:bool=False,
                                ths=None,
                                roi_is_gt: bool = True,
                                n_processes: int = 10,
                                target_path:Union[str, None]=None,
                                target_fname:str='tumors_measurements_-_th_1.xlsx',
                                min_size:int=20):
        

        nnunet_calculate_measures(gt_labels_path=gt_labels_path, 
                                pred_masks_paths=[pred_masks_paths],
                                roi_masks_path=roi_masks_path, 
                                region_based_inferencing=region_based_inferencing,
                                ths=ths,
                                roi_is_gt=roi_is_gt,
                                n_processes=n_processes,
                                min_size=min_size)
        
        if target_path is not None:
            shutil.move(f'{pred_masks_paths}/tumors_measurements_-_th_1.xlsx', os.path.join(target_path, target_fname))





if __name__ == '__main__':
    c = CalcMeasures()
    c.calculate_measures(gt_labels_path='/cs/labs/josko/aarono/all_data/brain/edan_split/raw_longitudinal_split/labels',
                                pred_masks_paths='/cs/labs/josko/aarono/projects/brain_diff_last/temp/sigmoid/sigmoid_to_nii',
                                roi_masks_path='/cs/labs/josko/aarono/all_data/brain/edan_split/raw_longitudinal_split/ROI',
                                region_based_inferencing=False,
                                target_path='/cs/labs/josko/aarono/projects/brain_diff_last/temp/sigmoid/sigmoid_to_nii',
                                target_fname='output.xlsx',
                                min_size=20)

        
