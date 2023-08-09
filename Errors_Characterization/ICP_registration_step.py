from typing import Optional, Tuple
from skimage.morphology import disk
from nibabel import affines
from skimage.morphology import binary_dilation
from scipy.ndimage import affine_transform

import open3d as o3d
import numpy as np


def execute_ICP(bl_pc, fu_pc, voxel_size, distance_threshold_factor, init_transformation):

    distance_threshold = voxel_size * distance_threshold_factor

    result = o3d.pipelines.registration.registration_icp(bl_pc, fu_pc, distance_threshold, init_transformation,
                                                         o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                                                         o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=5000))

    return result


def extract_liver_contour_as_PC(liver_seg: np.ndarray, affine_matrix: np.ndarray) -> o3d.geometry.PointCloud:
    """
    Extract the liver contour as a point cloud
    """

    # extract liver borders
    selem = disk(1).reshape([3, 3, 1])
    liver_border_points = affines.apply_affine(affine_matrix, np.stack(np.where(np.logical_xor(binary_dilation(liver_seg, selem), liver_seg))).T)

    # convert to point cloud
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(liver_border_points)

    return pc


def apply_affine_registration(A: np.ndarray, transform_inverse: np.ndarray, is_A_a_mask: bool) -> np.ndarray:
    return affine_transform(A, transform_inverse, order=0 if is_A_a_mask else 3)


def ICP_registration(bl_liver_seg: np.ndarray, fu_liver_seg: np.ndarray, affine_matrix: np.ndarray,
                     bl_ct_scan: Optional[np.ndarray] = None, bl_tumors_seg: Optional[np.ndarray] = None) -> Tuple[Tuple[np.ndarray,
                                                                                                                         Optional[np.ndarray],
                                                                                                                         Optional[np.ndarray]],
                                                                                                                   np.ndarray]:
    """
    Applies ICP registration to between bl_liver_seg and fu_liver_seg based on their contour.
    Optional: Applies it on the bl_ct_scan and/or bl_tumors_seg too.
    The given affine_matrix is that of the fu.

    Returns the registered bl_liver_seg (also the registered bl_ct_scan and/or bl_tumors_seg, if they are given).
    In addition, returns the rigid registration parameters (4D matrix).
    """

    bl_pc = extract_liver_contour_as_PC(bl_liver_seg, affine_matrix)
    fu_pc = extract_liver_contour_as_PC(fu_liver_seg, affine_matrix)

    # apply ICP
    result_icp = execute_ICP(bl_pc, fu_pc, voxel_size=1, distance_threshold_factor=40, init_transformation=np.eye(4))

    transform_inverse = np.linalg.inv(affine_matrix) @ np.linalg.inv(result_icp.transformation) @ affine_matrix

    # registration of the bl liver
    transformed_bl_liver_seg = apply_affine_registration(bl_liver_seg, transform_inverse, is_A_a_mask=True)

    # registration of the bl ct scan
    transformed_bl_ct_scan = None
    if bl_ct_scan is not None:
        transformed_bl_ct_scan = apply_affine_registration(bl_ct_scan, transform_inverse, is_A_a_mask=False)

    # registration of the bl tumors
    transformed_bl_tumors_seg = None
    if bl_tumors_seg is not None:
        transformed_bl_tumors_seg = apply_affine_registration(bl_tumors_seg, transform_inverse, is_A_a_mask=True)

    return (transformed_bl_liver_seg, transformed_bl_ct_scan, transformed_bl_tumors_seg), transform_inverse
