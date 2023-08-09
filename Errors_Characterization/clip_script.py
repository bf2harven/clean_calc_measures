from utils import load_nifti_data
from nibabel import save, Nifti1Image
import sys
from numpy import clip

_, file_path_to_load, file_path_to_save, min_clip_val, max_clip_val = sys.argv
min_clip_val, max_clip_val = int(min_clip_val), int(max_clip_val)
case, nifti = load_nifti_data(file_path_to_load)
case = clip(case, min_clip_val, max_clip_val).astype(case.dtype)
save(Nifti1Image(case, nifti.affine), file_path_to_save)
