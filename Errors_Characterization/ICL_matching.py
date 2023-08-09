from utils import *
from skimage.measure import centroid
import open3d as o3d
import copy
from copy import deepcopy
from skimage.transform import AffineTransform, warp
from scipy.ndimage import affine_transform
from skimage.morphology import binary_erosion, ball, binary_dilation
from sklearn.cluster import SpectralClustering
from skimage.transform import resize
import cv2
import shutil
from typing import Tuple, Optional


def get_COMs_of_tumors(labeled_tumors: np.ndarray) -> np.ndarray:
    pc = []
    for t in np.unique(labeled_tumors):
        if t == 0:
            continue
        centroid_in_voxel_space = centroid(labeled_tumors == t)
        pc.append(centroid_in_voxel_space)
    return np.asarray(pc)


def get_COMs_of_tumors_for_each_slice(labeled_tumors: np.ndarray) -> np.ndarray:
    pc = []
    for t in np.unique(labeled_tumors):
        if t == 0:
            continue
        current_tumor_points = np.stack(np.where(labeled_tumors == t)).T
        for z in np.unique(current_tumor_points[:, 2]):
            relevant_points = current_tumor_points[current_tumor_points[:, 2] == z]
            current_tumor_and_slice = np.zeros_like(labeled_tumors)
            current_tumor_and_slice[relevant_points[:, 0], relevant_points[:, 1], relevant_points[:, 2]] = 1
            centroid_in_voxel_space = centroid(current_tumor_and_slice)
            pc.append(centroid_in_voxel_space)
    return np.asarray(pc)


def get_n_biggest_tumors(labeled_tumors: np.ndarray, n_biggest: int) -> np.ndarray:
    tumors_sizes = np.stack(np.unique(labeled_tumors[labeled_tumors != 0], return_counts=True)).T
    tumors_sizes = tumors_sizes[np.argsort(tumors_sizes[:, 1])]
    relevant_tumors = tumors_sizes[-n_biggest:, 0]
    return np.where(np.isin(labeled_tumors, relevant_tumors), labeled_tumors, 0)


def draw_registration_result(source, target, transformation=None, window_name='Open3D', dynamic_rotation=(0, 0)):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])

    if transformation is not None:
        source_temp.transform(transformation)
    # o3d.visualization.draw_geometries([source_temp, target_temp], window_name=window_name)

    def rotate_view(vis):
        ctr = vis.get_view_control()
        ctr.rotate(*dynamic_rotation)
        return False

    o3d.visualization.draw_geometries_with_animation_callback([source_temp, target_temp],
                                                              rotate_view, window_name=window_name)


def preprocess_point_cloud(pcd, voxel_size):

    # print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    # print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    # print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


# def load_points_clouds(bl_labeled_tumors_file, fu_labeled_tumors_file, center_of_mass_for_RANSAC=False,
#                        n_biggest=None):
#     print(":: Load two point clouds.")
#     bl_labeled_tumors, file = load_nifti_data(bl_labeled_tumors_file)
#     fu_labeled_tumors, _ = load_nifti_data(fu_labeled_tumors_file)
#
#     if n_biggest is not None:
#         bl_tumors = get_n_biggest_tumors(bl_labeled_tumors, n_biggest=n_biggest)
#         fu_tumors = get_n_biggest_tumors(fu_labeled_tumors, n_biggest=n_biggest)
#     else:
#         bl_tumors = bl_labeled_tumors
#         fu_tumors = fu_labeled_tumors
#
#     bl_points = affines.apply_affine(file.affine, np.stack(np.where(bl_tumors > 0)).T)
#     fu_points = affines.apply_affine(file.affine, np.stack(np.where(fu_tumors > 0)).T)
#
#     if center_of_mass_for_RANSAC:
#         # extract center of mass
#         bl_p = affines.apply_affine(file.affine, get_COMs_of_tumors_as_PC(bl_tumors))
#         fu_p = affines.apply_affine(file.affine, get_COMs_of_tumors_as_PC(fu_tumors))
#     else:
#         bl_p = bl_points
#         fu_p = fu_points
#
#     working_bl_pc = o3d.geometry.PointCloud()
#     working_bl_pc.points = o3d.utility.Vector3dVector(bl_p)
#     working_fu_pc = o3d.geometry.PointCloud()
#     working_fu_pc.points = o3d.utility.Vector3dVector(fu_p)
#
#     if center_of_mass_for_RANSAC:
#         bl_pc = o3d.geometry.PointCloud()
#         bl_pc.points = o3d.utility.Vector3dVector(bl_points)
#         fu_pc = o3d.geometry.PointCloud()
#         fu_pc.points = o3d.utility.Vector3dVector(fu_points)
#     else:
#         bl_pc = deepcopy(working_bl_pc)
#         fu_pc = deepcopy(working_fu_pc)
#
#     return working_bl_pc, working_fu_pc, bl_pc, fu_pc, bl_points, fu_points, file
#
#
# def prepare_dataset(bl_pc, fu_pc, voxel_size, downsample=True):
#     bl_down, bl_fpfh = preprocess_point_cloud(bl_pc, voxel_size, downsample=downsample)
#     fu_down, fu_fpfh = preprocess_point_cloud(fu_pc, voxel_size, downsample=downsample)
#     return bl_down, fu_down, bl_fpfh, fu_fpfh


def pairwise_preprocessing_by_erosion(bl_labeled_tumors: np.ndarray, fu_labeled_tumors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

    def erosion_for_given_tumors(tumors_labeled_case, tumors_to_reduce):
        relevant_tumors_case = np.isin(tumors_labeled_case, tumors_to_reduce).astype(tumors_labeled_case.dtype)

        relevant_tumors_case = binary_erosion(relevant_tumors_case, ball(5))

        result = np.where(relevant_tumors_case, tumors_labeled_case, 0)
        result = np.where(~np.isin(tumors_labeled_case, tumors_to_reduce), tumors_labeled_case, result)

        deleted_tumors = tumors_to_reduce[~np.isin(tumors_to_reduce, np.unique(result))]

        if deleted_tumors.size > 0:
            result = np.where(np.isin(tumors_labeled_case, deleted_tumors), tumors_labeled_case, result)

        result = get_connected_components((result > 0).astype(result.dtype), 1).astype(tumors_labeled_case.dtype)

        return result


    # check intersections
    intersections = np.hstack([bl_labeled_tumors.reshape([-1, 1]), fu_labeled_tumors.reshape([-1, 1])])
    intersections = np.unique(intersections[~np.any(intersections == 0, axis=1)], axis=0)

    # find bl tumors that intersect more than 1 tumor in fu
    bl_tumors_unique_with_number_of_intersections = np.stack(np.unique(intersections[:, 0], return_counts=True)).T
    too_big_bl_tumors = bl_tumors_unique_with_number_of_intersections[bl_tumors_unique_with_number_of_intersections[:, 1] > 1][:, 0]

    if too_big_bl_tumors.size > 0:
        bl_result = erosion_for_given_tumors(bl_labeled_tumors, too_big_bl_tumors)
    else:
        bl_result = bl_labeled_tumors.copy()

    # find fu tumors that intersect more than 1 tumor in bl
    fu_tumors_unique_with_number_of_intersections = np.stack(np.unique(intersections[:, 1], return_counts=True)).T
    too_big_fu_tumors = fu_tumors_unique_with_number_of_intersections[fu_tumors_unique_with_number_of_intersections[:, 1] > 1][:, 0]

    if too_big_fu_tumors.size > 0:
        fu_result = erosion_for_given_tumors(fu_labeled_tumors, too_big_fu_tumors)
    else:
        fu_result = fu_labeled_tumors.copy()

    save(Nifti1Image(bl_result, file.affine), replace_in_file_name(bl_labeled_tumors_file, '.nii.gz', '_after_erosion_you_can_delete.nii.gz', dst_file_exist=False))
    save(Nifti1Image(fu_result, file.affine), replace_in_file_name(fu_labeled_tumors_file, '.nii.gz', '_after_erosion_you_can_delete.nii.gz', dst_file_exist=False))

    exit(0)

    return bl_result, fu_result


def pairwise_preprocessing_by_distance(bl_labeled_tumors: np.ndarray, fu_labeled_tumors: np.ndarray, voxelspacing) -> Tuple[np.ndarray, np.ndarray]:

    def get_relevant_centroids(source_tumors, source_type, target_labeled_tumors):
        source_tumors_centroids = {}
        source_index = 0 if source_type == 'bl' else 1
        for source_t in source_tumors:
            relevant_intersections = (intersections[:, source_index] == source_t)
            target_t_that_intersect_with_size_of_intersection = np.stack([intersections[relevant_intersections][:, abs(source_index - 1)], counts[relevant_intersections]]).T
            target_t_that_intersect_with_size_of_intersection = target_t_that_intersect_with_size_of_intersection[target_t_that_intersect_with_size_of_intersection[:, 0].argsort()]
            target_t_that_intersect_with_original_size = np.stack(np.unique(target_labeled_tumors[np.isin(target_labeled_tumors, target_t_that_intersect_with_size_of_intersection[:, 0])],
                                                                            return_counts=True)).T
            ratio = target_t_that_intersect_with_size_of_intersection.copy()
            ratio[:, 1] /= target_t_that_intersect_with_original_size[:, 1]
            relevant_target_intersection_tumors = ratio[ratio[:, 1] >= 0.15][:, 0]
            centroids = []
            for target_t in relevant_target_intersection_tumors:
                centroids.append(centroid(target_labeled_tumors == target_t))
            if len(centroids) > 1:
                source_tumors_centroids[source_t] = centroids
        return source_tumors_centroids

    def divide_given_tumors(labeled_tumors, too_big_tumors_centroids):
        res = np.zeros_like(labeled_tumors)
        for t, centroids in too_big_tumors_centroids.items():
            centroids = np.asarray(centroids)
            centroids = centroids.round().astype(np.int)
            centroids_in_grid = np.zeros_like(labeled_tumors)
            centroids_in_grid[centroids[:, 0], centroids[:, 1], centroids[:, 2]] = np.unique(labeled_tumors).size + np.arange(centroids.shape[0]) + 1
            _, nearest_label_coords = distance_transform_edt(centroids_in_grid == 0, return_indices=True, sampling=voxelspacing)
            res = np.where(labeled_tumors == t, centroids_in_grid[tuple(nearest_label_coords)], res)
        res = np.where(res > 0, res, labeled_tumors)
        return res

    # check intersections
    intersections = np.hstack([bl_labeled_tumors.reshape([-1, 1]), fu_labeled_tumors.reshape([-1, 1])])
    intersections, counts = np.unique(intersections[~np.any(intersections == 0, axis=1)], axis=0, return_counts=True)

    # find bl tumors that intersect more than 1 tumor in fu
    bl_tumors_unique_with_number_of_intersections = np.stack(np.unique(intersections[:, 0], return_counts=True)).T
    too_big_bl_tumors = bl_tumors_unique_with_number_of_intersections[bl_tumors_unique_with_number_of_intersections[:, 1] > 1][:, 0]

    # find fu tumors that intersect more than 1 tumor in bl
    fu_tumors_unique_with_number_of_intersections = np.stack(np.unique(intersections[:, 1], return_counts=True)).T
    too_big_fu_tumors = fu_tumors_unique_with_number_of_intersections[fu_tumors_unique_with_number_of_intersections[:, 1] > 1][:, 0]

    if too_big_bl_tumors.size > 0:
        too_big_bl_tumors_centroids = get_relevant_centroids(too_big_bl_tumors, source_type='bl',
                                                             target_labeled_tumors=fu_labeled_tumors)
        bl_result = divide_given_tumors(bl_labeled_tumors, too_big_bl_tumors_centroids)
    else:
        bl_result = bl_labeled_tumors.copy()

    if too_big_fu_tumors.size > 0:
        too_big_fu_tumors_centroids = get_relevant_centroids(too_big_fu_tumors, source_type='fu',
                                                             target_labeled_tumors=bl_labeled_tumors)
        fu_result = divide_given_tumors(fu_labeled_tumors, too_big_fu_tumors_centroids)
    else:
        fu_result = fu_labeled_tumors.copy()

    save(Nifti1Image(bl_result, file.affine), replace_in_file_name(bl_labeled_tumors_file, '.nii.gz', '_after_erosion_you_can_delete.nii.gz', dst_file_exist=False).replace('BL_Scan', 'bl_scan'))
    save(Nifti1Image(fu_result, file.affine), replace_in_file_name(fu_labeled_tumors_file, '.nii.gz', '_after_erosion_you_can_delete.nii.gz', dst_file_exist=False).replace('FU_Scan', 'fu_scan'))
    exit(0)

    return bl_result, fu_result


def pairwise_preprocessing_by_clustering(bl_labeled_tumors: np.ndarray, fu_labeled_tumors: np.ndarray, voxelspacing,
                                         file_affine_matrix) -> Tuple[np.ndarray, np.ndarray]:

    def get_relevant_num_of_clusters(source_tumors, source_type, source_labeled_tumors, target_labeled_tumors):
        source_tumors_num_of_clusters = {}
        source_index = 0 if source_type == 'bl' else 1
        for source_t in source_tumors:
            source_t_vol = (source_labeled_tumors == source_t).sum() * voxelspacing[0] * voxelspacing[1] * voxelspacing[2] / 1000
            if source_t_vol < 15:
                continue
            relevant_intersections = (intersections[:, source_index] == source_t)
            target_t_that_intersect_with_size_of_intersection = np.stack([intersections[relevant_intersections][:, abs(source_index - 1)], counts[relevant_intersections]]).T
            target_t_that_intersect_with_size_of_intersection = target_t_that_intersect_with_size_of_intersection[target_t_that_intersect_with_size_of_intersection[:, 0].argsort()]
            target_t_that_intersect_with_original_size = np.stack(np.unique(target_labeled_tumors[np.isin(target_labeled_tumors, target_t_that_intersect_with_size_of_intersection[:, 0])],
                                                                            return_counts=True)).T
            ratio = target_t_that_intersect_with_size_of_intersection.copy()
            ratio[:, 1] /= target_t_that_intersect_with_original_size[:, 1]
            relevant_target_intersection_tumors = ratio[ratio[:, 1] >= 0.15][:, 0]
            if relevant_target_intersection_tumors.size > 1:
                source_tumors_num_of_clusters[source_t] = relevant_target_intersection_tumors.size
        return source_tumors_num_of_clusters

    def cluster_given_tumors(labeled_tumors, too_big_tumors_num_of_clusters):
        if len(too_big_tumors_num_of_clusters) > 0:
            resized_labeled_tumors = resize(labeled_tumors, np.asarray(labeled_tumors.shape)//4, order=0,
                                            mode='constant', anti_aliasing=False)
            res = np.zeros_like(resized_labeled_tumors)
            n_tumors = np.unique(labeled_tumors).size
            copy_of_labeled_tumors = labeled_tumors.copy()
            for t, n_clusters in too_big_tumors_num_of_clusters.items():
                print('Start clustering')
                tm = time()
                copy_of_labeled_tumors[copy_of_labeled_tumors == t] = 0
                points = np.stack(np.where(resized_labeled_tumors == t)).T.astype(np.int32)
                X = affines.apply_affine(file_affine_matrix, points).astype(np.float32)
                clustering = SpectralClustering(n_clusters=n_clusters, random_state=42, assign_labels='discretize',
                                                n_jobs=-1).fit(X)
                res[points[:, 0], points[:, 1], points[:, 2]] = clustering.labels_ + n_tumors + 1
                print(f'finished clustering in {calculate_runtime(tm)}')
            res = resize(res, labeled_tumors.shape, order=0, mode='constant', anti_aliasing=False)
            res = np.where(res > 0, res * (labeled_tumors > 0), copy_of_labeled_tumors)
        else:
            res = labeled_tumors.copy()
        return res

    # check intersections
    intersections = np.hstack([bl_labeled_tumors.reshape([-1, 1]), fu_labeled_tumors.reshape([-1, 1])])
    intersections, counts = np.unique(intersections[~np.any(intersections == 0, axis=1)], axis=0, return_counts=True)

    # find bl tumors that intersect more than 1 tumor in fu
    bl_tumors_unique_with_number_of_intersections = np.stack(np.unique(intersections[:, 0], return_counts=True)).T
    too_big_bl_tumors = bl_tumors_unique_with_number_of_intersections[bl_tumors_unique_with_number_of_intersections[:, 1] > 1][:, 0]

    # find fu tumors that intersect more than 1 tumor in bl
    fu_tumors_unique_with_number_of_intersections = np.stack(np.unique(intersections[:, 1], return_counts=True)).T
    too_big_fu_tumors = fu_tumors_unique_with_number_of_intersections[fu_tumors_unique_with_number_of_intersections[:, 1] > 1][:, 0]

    if too_big_bl_tumors.size > 0:
        too_big_bl_tumors_centroids = get_relevant_num_of_clusters(too_big_bl_tumors, source_type='bl',
                                                                   source_labeled_tumors=bl_labeled_tumors,
                                                                   target_labeled_tumors=fu_labeled_tumors)
        bl_result = cluster_given_tumors(bl_labeled_tumors, too_big_bl_tumors_centroids)
    else:
        bl_result = bl_labeled_tumors.copy()

    if too_big_fu_tumors.size > 0:
        too_big_fu_tumors_centroids = get_relevant_num_of_clusters(too_big_fu_tumors, source_type='fu',
                                                                   source_labeled_tumors=fu_labeled_tumors,
                                                                   target_labeled_tumors=bl_labeled_tumors)
        fu_result = cluster_given_tumors(fu_labeled_tumors, too_big_fu_tumors_centroids)
    else:
        fu_result = fu_labeled_tumors.copy()

    save(Nifti1Image(bl_result, file.affine), replace_in_file_name(bl_labeled_tumors_file, '.nii.gz', '_after_erosion_you_can_delete.nii.gz', dst_file_exist=False).replace('BL_Scan', 'bl_scan'))
    save(Nifti1Image(fu_result, file.affine), replace_in_file_name(fu_labeled_tumors_file, '.nii.gz', '_after_erosion_you_can_delete.nii.gz', dst_file_exist=False).replace('FU_Scan', 'fu_scan'))
    exit(0)

    return bl_result, fu_result


def prepare_dataset(bl_labeled_tumors: np.ndarray, fu_labeled_tumors: np.ndarray, bl_liver: np.ndarray, fu_liver: np.ndarray,
                    file_affine_matrix: np.ndarray, voxelspacing: Tuple[float, float, float], voxel_size: float, center_of_mass_for_RANSAC: bool=False,
                    n_biggest: Optional[int]=None, ICP_with_liver_border: bool = False, RANSAC_with_liver_border: bool = False,
                    RANSAC_with_tumors: bool = True, ICP_with_tumors: bool = True):

    relevant_bl_labeled_tumors = bl_labeled_tumors
    relevant_fu_labeled_tumors = fu_labeled_tumors
    if n_biggest is not None:
        relevant_bl_labeled_tumors = get_n_biggest_tumors(relevant_bl_labeled_tumors, n_biggest=n_biggest)
        relevant_fu_labeled_tumors = get_n_biggest_tumors(relevant_fu_labeled_tumors, n_biggest=n_biggest)

    bl_tumors_pc = o3d.geometry.PointCloud()
    bl_tumors_pc.points = o3d.utility.Vector3dVector(affines.apply_affine(file_affine_matrix, np.stack(np.where(bl_labeled_tumors > 0)).T))
    fu_tumors_pc = o3d.geometry.PointCloud()
    fu_tumors_pc.points = o3d.utility.Vector3dVector(affines.apply_affine(file_affine_matrix, np.stack(np.where(fu_labeled_tumors > 0)).T))

    relevant_bl_tumors_pc = bl_tumors_pc
    relevant_fu_tumors_pc = fu_tumors_pc
    if n_biggest is not None:
        relevant_bl_tumors_pc = o3d.geometry.PointCloud()
        relevant_bl_tumors_pc.points = o3d.utility.Vector3dVector(affines.apply_affine(file_affine_matrix, np.stack(np.where(relevant_bl_labeled_tumors > 0)).T))
        relevant_fu_tumors_pc = o3d.geometry.PointCloud()
        relevant_fu_tumors_pc.points = o3d.utility.Vector3dVector(affines.apply_affine(file_affine_matrix, np.stack(np.where(relevant_fu_labeled_tumors > 0)).T))

    if RANSAC_with_tumors:
        global_registration_working_bl_pc = relevant_bl_tumors_pc
        global_registration_working_fu_pc = relevant_fu_tumors_pc
        if center_of_mass_for_RANSAC:
            # processed_relevant_bl_labeled_tumors, processed_relevant_fu_labeled_tumors = \
            #     pairwise_preprocessing_by_clustering(relevant_bl_labeled_tumors, relevant_fu_labeled_tumors, voxelspacing, file_affine_matrix)
            processed_relevant_bl_labeled_tumors, processed_relevant_fu_labeled_tumors = (relevant_bl_labeled_tumors, relevant_fu_labeled_tumors)
            global_registration_working_bl_pc = o3d.geometry.PointCloud()
            global_registration_working_bl_pc.points = o3d.utility.Vector3dVector(affines.apply_affine(file_affine_matrix, get_COMs_of_tumors(processed_relevant_bl_labeled_tumors)))
            # global_registration_working_bl_pc.points = o3d.utility.Vector3dVector(affines.apply_affine(file_affine_matrix, get_COMs_of_tumors_for_each_slice(processed_relevant_bl_labeled_tumors)))
            global_registration_working_fu_pc = o3d.geometry.PointCloud()
            global_registration_working_fu_pc.points = o3d.utility.Vector3dVector(affines.apply_affine(file_affine_matrix, get_COMs_of_tumors(processed_relevant_fu_labeled_tumors)))
            # global_registration_working_fu_pc.points = o3d.utility.Vector3dVector(affines.apply_affine(file_affine_matrix, get_COMs_of_tumors_for_each_slice(processed_relevant_fu_labeled_tumors)))
    else:
        global_registration_working_bl_pc = o3d.geometry.PointCloud()
        global_registration_working_fu_pc = o3d.geometry.PointCloud()

    # adding liver border points
    if ICP_with_liver_border or RANSAC_with_liver_border:

        selem = disk(1).reshape([3, 3, 1])
        # selem = ball(1)
        bl_liver_border_points = affines.apply_affine(file_affine_matrix, np.stack(np.where(np.logical_xor(binary_dilation(bl_liver, selem), bl_liver))).T)
        fu_liver_border_points = affines.apply_affine(file_affine_matrix, np.stack(np.where(np.logical_xor(binary_dilation(fu_liver, selem), fu_liver))).T)

    if ICP_with_tumors:
        ICP_bl_pc = deepcopy(relevant_bl_tumors_pc)
        ICP_fu_pc = deepcopy(relevant_fu_tumors_pc)
    else:
        ICP_bl_pc = o3d.geometry.PointCloud()
        ICP_fu_pc = o3d.geometry.PointCloud()

    if ICP_with_liver_border:
        ICP_bl_pc.points = o3d.utility.Vector3dVector(np.concatenate([bl_liver_border_points, np.asarray(ICP_bl_pc.points)]))
        ICP_fu_pc.points = o3d.utility.Vector3dVector(np.concatenate([fu_liver_border_points, np.asarray(ICP_fu_pc.points)]))

    # if ICP_with_liver_border:
    #     ICP_bl_pc = o3d.geometry.PointCloud()
    #     ICP_bl_pc.points = o3d.utility.Vector3dVector(np.concatenate([bl_liver_border_points, np.asarray(relevant_bl_tumors_pc.points)]))
    #     ICP_fu_pc = o3d.geometry.PointCloud()
    #     ICP_fu_pc.points = o3d.utility.Vector3dVector(np.concatenate([fu_liver_border_points, np.asarray(relevant_fu_tumors_pc.points)]))
    # else:
    #     ICP_bl_pc = relevant_bl_tumors_pc
    #     ICP_fu_pc = relevant_fu_tumors_pc

    if RANSAC_with_liver_border:
        global_registration_working_bl_pc.points = o3d.utility.Vector3dVector(np.concatenate([bl_liver_border_points, np.asarray(global_registration_working_bl_pc.points)]))
        global_registration_working_fu_pc.points = o3d.utility.Vector3dVector(np.concatenate([fu_liver_border_points, np.asarray(global_registration_working_fu_pc.points)]))

    if np.asarray(global_registration_working_bl_pc.points).shape[0] > 0:
        bl_down, bl_fpfh = preprocess_point_cloud(global_registration_working_bl_pc, voxel_size)
        fu_down, fu_fpfh = preprocess_point_cloud(global_registration_working_fu_pc, voxel_size)
    else:
        bl_down = o3d.geometry.PointCloud()
        fu_down = o3d.geometry.PointCloud()
        bl_fpfh = o3d.pybind.pipelines.registration.Feature()
        fu_fpfh = o3d.pybind.pipelines.registration.Feature()


    return bl_down, fu_down, bl_fpfh, fu_fpfh, bl_tumors_pc, fu_tumors_pc, relevant_bl_tumors_pc, relevant_fu_tumors_pc, ICP_bl_pc, ICP_fu_pc


def execute_global_registration(bl_down, fu_down, bl_fpfh,
                                fu_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        bl_down, fu_down, bl_fpfh, fu_fpfh, distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 4, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500))
    return result


def execute_fast_global_registration(bl_down, fu_down, bl_fpfh,
                                     fu_fpfh, voxel_size):
    if (np.asarray(bl_down.points).shape[0] < 3) or (np.asarray(fu_down.points).shape[0] < 3):
        result = o3d.pipelines.registration.RegistrationResult()
        result.transformation = np.eye(4)
    else:
        distance_threshold = voxel_size * 0.5
        # print(":: Apply fast global registration with distance threshold %.3f" \
        #         % distance_threshold)
        # result = o3d.pipelines.registration.registration_fast_based_on_feature_matching(
        #     bl_down, fu_down, bl_fpfh, fu_fpfh,
        #     o3d.pipelines.registration.FastGlobalRegistrationOption(
        #         maximum_correspondence_distance=distance_threshold), seed=42)

        result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
            bl_down, fu_down, bl_fpfh, fu_fpfh,
            o3d.pipelines.registration.FastGlobalRegistrationOption(
                maximum_correspondence_distance=distance_threshold, seed=42))
                # maximum_correspondence_distance=distance_threshold, seed=None))
        # result = o3d.pipelines.registration.registration_fast_based_on_feature_matching(
        #     bl_down, fu_down, bl_fpfh, fu_fpfh,
        #     o3d.pipelines.registration.FastGlobalRegistrationOption(
        #         maximum_correspondence_distance=distance_threshold))
    return result


def execute_ICP(bl_pc, fu_pc, voxel_size, distance_threshold_factor, init_transformation):
    distance_threshold = voxel_size * distance_threshold_factor
    # print(":: Point-to-plane ICP registration is applied on original point")
    # print(":: Point-to-point ICP registration is applied on original point")
    # print("   clouds to refine the alignment. This time we use a strict")
    # print("   distance threshold %.3f." % distance_threshold)

    # radius_normal = voxel_size * 2
    # print(":: Estimate normal with search radius %.3f." % radius_normal)
    # bl_pc.estimate_normals(
    #     o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    # fu_pc.estimate_normals(
    #     o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    # result = o3d.pipelines.registration.registration_icp(
    #     bl_pc, fu_pc, distance_threshold, init_transformation,
    #     o3d.pipelines.registration.TransformationEstimationPointToPlane())
    result = o3d.pipelines.registration.registration_icp(bl_pc, fu_pc, distance_threshold, init_transformation,
                                                         o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                                                         o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=5000))
    return result


def run_for_visualization_ICP(bl_pc_run, fu_pc_run, bl_pc_visualize, fu_pc_visualize, voxel_size, distance_threshold_factor, init_transformation,
                             icp_iterations, pre_vis_iterations=5.5e+2, post_vis_iterations=1e+1, save_as_video_at=None):
    distance_threshold = voxel_size * distance_threshold_factor

    source_temp_run = copy.deepcopy(bl_pc_run)
    target_temp_run = copy.deepcopy(fu_pc_run)
    # source_temp_run.transform(init_transformation)

    source_temp_visualize = copy.deepcopy(bl_pc_visualize)
    target_temp_visualize = copy.deepcopy(fu_pc_visualize)
    # source_temp_visualize.transform(init_transformation)
    source_temp_visualize.paint_uniform_color([1, 0.706, 0])
    target_temp_visualize.paint_uniform_color([0, 0.651, 0.929])

    # todo delete
    # source_temp_run.paint_uniform_color([0, 0, 1], alpha=0.5)
    # target_temp_run.paint_uniform_color([0, 1, 0], alpha=0.5)
    # todo delete


    # Initialize Visualizer class
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(source_temp_visualize)
    vis.add_geometry(target_temp_visualize)


    # todo delete
    # vis.add_geometry(source_temp_run)
    # vis.add_geometry(target_temp_run)
    # todo delete

    save_video = save_as_video_at is not None
    if save_video:
        os.makedirs('/tmp/ICP_video', exist_ok=True)

    # pre visualize
    for j in range(int(pre_vis_iterations)):
        if save_video:
            vis.capture_screen_image("/tmp/ICP_video/temp1_%04d.jpg" % j)
        ctr = vis.get_view_control()
        ctr.rotate(5, 1)
        vis.poll_events()
        vis.update_renderer()

    # Transform geometry and visualize it
    identity = np.eye(4)
    reg_p2l = o3d.pipelines.registration.RegistrationResult()
    reg_p2l.transformation = identity
    for i in range(icp_iterations):
        if save_video:
            vis.capture_screen_image("/tmp/ICP_video/temp2_%04d.jpg" % i)
        if i % 20 == 0:
            reg_p2l = o3d.pipelines.registration.registration_icp(
                source_temp_run, target_temp_run, distance_threshold, np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1))
            source_temp_visualize.transform(reg_p2l.transformation)
            source_temp_run.transform(reg_p2l.transformation)
            vis.update_geometry(source_temp_visualize)

            # todo delete
            # vis.update_geometry(source_temp_run)
            # todo delete

        ctr = vis.get_view_control()
        ctr.rotate(5, 1)
        vis.poll_events()
        vis.update_renderer()
        if np.isclose(reg_p2l.transformation, identity, rtol=1e-1, atol=1e-1).all():
            break


    # post visualize
    for k in range(int(post_vis_iterations)):
        if save_video:
            vis.capture_screen_image("/tmp/ICP_video/temp3_%04d.jpg" % k)
        ctr = vis.get_view_control()
        ctr.rotate(5, 1)
        vis.poll_events()
        vis.update_renderer()

    vis.destroy_window()

    if save_video:
        print(':: Saving ICP video')
        vis.capture_screen_image("/tmp/ICP_video/temp_last.jpg")
        images = sorted(glob('/tmp/ICP_video/*.jpg'), key=os.path.getmtime)
        frame = cv2.imread(images[0])
        height, width, layers = frame.shape

        video = cv2.VideoWriter(save_as_video_at, 0, 1, (width, height))

        for image in images:
            video.write(cv2.imread(image))

        cv2.destroyAllWindows()
        video.release()

        shutil.rmtree('/tmp/ICP_video')

    return reg_p2l


if __name__ == "__main__":

    debug = False
    visualize_ICP = False
    n_biggest = None
    center_of_mass_for_RANSAC = True
    RANSAC_with_liver_border = False
    RANSAC_with_tumors = False
    ICP_with_liver_border = True
    ICP_with_tumors = False
    save_results = True
    # case_name = 'BL_A_Y_22_01_2020_FU_A_Y_04_05_2020'
    # case_name = 'BL_A_Ac_21_12_2020_FU_A_Ac_30_12_2020'
    # case_name = 'BL_A_D_04_11_2019_FU_A_D_16_12_2019'
    # case_name = 'BL_A_J_25_12_2019_FU_A_J_24_03_2020'
    # case_name = 'BL_A_S_H_01_09_2019_FU_A_S_H_06_01_2020'
    # case_name = 'BL_H_G_06_10_2019_FU_H_G_24_11_2019'
    # case_name = 'BL_A_K_08_09_2015_FU_A_K_25_11_2015'
    # case_name = 'BL_A_W_10_07_2019_FU_A_W_26_09_2019'
    # case_name = 'BL_C_R_A_27_07_2018_FU_C_R_A_05_09_2018'
    # case_name = 'BL_B_S_Ya_18_02_2019_FU_B_S_Ya_10_02_2020'
    # case_name = 'BL_B_T_18_01_2019_FU_B_T_18_02_2019'
    # case_name = 'BL_Z_Y_18_03_2020_FU_Z_Y_06_05_2020'
    # case_name = 'BL_B_A_D_04_03_2014_FU_B_A_D_02_06_2014'
    # case_name = 'BL_B_B_S_06_08_2019_FU_B_B_S_27_10_2019'
    # case_name = 'BL_F_M_02_06_2019_FU_F_M_30_01_2020'
    # case_name = 'BL_E_N_03_11_2019_FU_E_N_17_12_2019'
    # case_name = 'BL_G_Eb_22_12_2013_FU_G_Eb_29_01_2014'
    # case_name = 'BL_N_Na_06_03_2018_FU_N_Na_05_06_2018'
    # case_name = 'BL_P_A_08_06_2014_FU_P_A_21_09_2014'
    # case_name = 'BL_S_M_Y_05_12_2019_FU_S_M_Y_05_04_2020'
    # case_name = 'BL_K_Mb_28_01_2020_FU_K_Mb_26_03_2020'
    # case_name = 'BL_G_G_17_06_2020_FU_G_G_16_08_2020'
    # case_name = 'BL_A_Ad_17_07_2019_FU_A_Ad_05_04_2021'
    # case_name = 'BL_S_H_31_12_2013_FU_A_S_H_20_02_2020'
    # case_name = 'BL_S_N_07_05_2017_FU_S_N_22_01_2019'
    # case_name = 'BL_S_I_24_11_2019_FU_S_I_22_01_2020'
    # case_name = 'BL_N_M_15_10_2019_FU_N_M_01_03_2020'

    case_names = sorted([os.path.basename(f) for f in glob(f'/cs/casmip/rochman/Errors_Characterization/corrected_segmentation_for_matching/BL_*')] +
                        [os.path.basename(f) for f in glob(f'/cs/casmip/rochman/Errors_Characterization/new_test_set_for_matching/BL_*')])
    # case_names = [case_name]
    # case_names = ['BL_G_G_17_06_2020_FU_G_G_16_08_2020', 'BL_K_Mb_28_01_2020_FU_K_Mb_26_03_2020', 'BL_S_M_Y_05_12_2019_FU_S_M_Y_05_04_2020']
    for i, case_name in enumerate(case_names, start=1):
        tm = time()

        dir_name = f'/cs/casmip/rochman/Errors_Characterization/corrected_segmentation_for_matching/{case_name}'
        if not os.path.isdir(dir_name):
            dir_name = f'/cs/casmip/rochman/Errors_Characterization/new_test_set_for_matching/{case_name}'
            assert os.path.isdir(dir_name)

        try:
            bl_labeled_tumors_file = glob(f'{dir_name}/BL_Scan_Tumors_unique_*')[0]
            # bl_labeled_tumors_file = glob(f'{dir_name}/improved_registration_BL_Scan_Tumors_unique_*')[0]
        except:
            bl_labeled_tumors_file = f'{dir_name}/BL_Scan_Tumors.nii.gz'
            assert os.path.isfile(bl_labeled_tumors_file)

        try:
            fu_labeled_tumors_file = glob(f'{dir_name}/FU_Scan_Tumors_unique_*')[0]
        except:
            fu_labeled_tumors_file = f'{dir_name}/FU_Scan_Tumors.nii.gz'
            assert os.path.isfile(fu_labeled_tumors_file)

        bl_liver_file = f'{dir_name}/BL_Scan_Liver.nii.gz'
        fu_liver_file = f'{dir_name}/FU_Scan_Liver.nii.gz'

        # load the data
        if debug:
            print(':: Loading the data')

        bl_labeled_tumors, file = load_nifti_data(bl_labeled_tumors_file)
        fu_labeled_tumors, _ = load_nifti_data(fu_labeled_tumors_file)
        if ICP_with_liver_border or RANSAC_with_liver_border:
            bl_liver, _ = load_nifti_data(bl_liver_file)
            fu_liver, _ = load_nifti_data(fu_liver_file)
            bl_liver = pre_process_segmentation(bl_liver)
            fu_liver = pre_process_segmentation(fu_liver)
        else:
            bl_liver, fu_liver = None, None


        # TODO delete if not necessary
        # vol = np.asarray(file.header.get_zooms()).prod() / 1000
        # if (bl_liver is None) or (fu_liver is None) or (abs((bl_liver > 0).sum() - (fu_liver > 0).sum()) * vol < 300) or \
        #     (np.unique(bl_labeled_tumors).size < 4) or (np.unique(fu_labeled_tumors).size < 4):
        #     continue
        # TODO till here

        # prepare the data for global registration
        voxel_size = 1  # means 5cm for the dataset
        if debug:
            print(':: Preparing the data')
        bl_down, fu_down, bl_fpfh, fu_fpfh, bl_tumors_pc_real_space_o3d, fu_tumors_pc, relevant_bl_tumors_pc, relevant_fu_tumors_pc, ICP_bl_pc, ICP_fu_pc = \
            prepare_dataset(bl_labeled_tumors, fu_labeled_tumors, bl_liver, fu_liver, file.affine, file.header.get_zooms(),
                            voxel_size, center_of_mass_for_RANSAC=center_of_mass_for_RANSAC, n_biggest=n_biggest,
                            ICP_with_liver_border=ICP_with_liver_border, RANSAC_with_liver_border=RANSAC_with_liver_border,
                            RANSAC_with_tumors=RANSAC_with_tumors, ICP_with_tumors=ICP_with_tumors)

        # working_bl_pc, working_fu_pc, bl_pc, fu_pc, bl_points, fu_points, file = load_points_clouds(bl_labeled_tumors_file, fu_labeled_tumors_file,
        #                                                                                             center_of_mass_for_RANSAC=extract_center_of_mass, n_biggest=None)
        #
        # # draw the points
        # # draw_registration_result(bl_pc, fu_pc)
        # # exit(0)
        #
        #
        # bl_down, fu_down, bl_fpfh, fu_fpfh = prepare_dataset(working_bl_pc, working_fu_pc, voxel_size, downsample=(not extract_center_of_mass))

        if debug:
            # draw_registration_result(relevant_bl_tumors_pc, relevant_fu_tumors_pc, window_name='Original')
            draw_registration_result(relevant_bl_tumors_pc, relevant_fu_tumors_pc, window_name='Original', dynamic_rotation=(5, 1))
            # draw_registration_result(ICP_bl_pc, ICP_fu_pc, window_name='Original', dynamic_rotation=(5, 1))
            # draw_registration_result(bl_down, fu_down, window_name='Original', dynamic_rotation=(5, 1))
            # draw_registration_result(bl_down, fu_down)

        # apply global registration on the data
        # result_ransac = execute_global_registration(bl_down, fu_down, bl_fpfh, fu_fpfh, voxel_size)
        result_ransac = execute_fast_global_registration(bl_down, fu_down, bl_fpfh, fu_fpfh, voxel_size)

        if debug:
            print('-------------------------------------------------------------')
            print(result_ransac)
            print('***********************')
            print(result_ransac.transformation)
            print('-------------------------------------------------------------')

        if debug and (RANSAC_with_liver_border or RANSAC_with_tumors):
            # draw_registration_result(relevant_bl_tumors_pc, relevant_fu_tumors_pc, result_ransac.transformation, 'After RANSAC')
            # draw_registration_result(bl_down, fu_down, result_ransac.transformation)
            # draw_registration_result(ICP_bl_pc, ICP_fu_pc, transformation=result_ransac.transformation, window_name='After RANSAC', dynamic_rotation=(5, 1))
            pass

        if ((np.unique(bl_labeled_tumors) != 0).sum() < 2) or ((np.unique(fu_labeled_tumors) != 0).sum() < 2):
            result_icp = result_ransac
        else:
            # result_icp = execute_ICP(relevant_bl_tumors_pc, relevant_fu_tumors_pc, voxel_size, distance_threshold_factor=40,
            #                          init_transformation=result_ransac.transformation)
            # result_icp = execute_ICP(bl_down, fu_down, voxel_size, distance_threshold_factor=40,
            #                          init_transformation=result_ransac.transformation)
            # ICP_bl_pc = ICP_bl_pc.voxel_down_sample(voxel_size)
            # ICP_fu_pc = ICP_fu_pc.voxel_down_sample(voxel_size)
            result_icp = execute_ICP(ICP_bl_pc, ICP_fu_pc, voxel_size, distance_threshold_factor=40,
                                     init_transformation=result_ransac.transformation)
            # result_icp = execute_ICP(ICP_bl_pc, ICP_fu_pc, voxel_size, distance_threshold_factor=40,
            #                          init_transformation=np.eye(4))

            if debug and visualize_ICP:
                run_for_visualization_ICP(ICP_bl_pc, ICP_fu_pc, relevant_bl_tumors_pc, relevant_fu_tumors_pc, voxel_size,
                                          distance_threshold_factor=40, init_transformation=result_ransac.transformation,
                                          icp_iterations=5000, save_as_video_at=None)
                # run_for_visualization_ICP(ICP_bl_pc, ICP_fu_pc, ICP_bl_pc, ICP_fu_pc,
                #                           voxel_size,
                #                           distance_threshold_factor=40,
                #                           init_transformation=result_ransac.transformation,
                #                           icp_iterations=5000, save_as_video_at=None)
        if debug:
            print('-------------------------------------------------------------')
            print(result_icp)
            print('***********************')
            print(result_icp.transformation)
            print('-------------------------------------------------------------')

            draw_registration_result(relevant_bl_tumors_pc, relevant_fu_tumors_pc, result_icp.transformation, window_name='After ICP',
                                     dynamic_rotation=(5, 1))
            # draw_registration_result(ICP_bl_pc, ICP_fu_pc, result_icp.transformation,
            #                          window_name='After ICP',
            #                          dynamic_rotation=(5, 1))
            # draw_registration_result(bl_down, fu_down, result_icp.transformation,
            #                          window_name='After ICP',
            #                          dynamic_rotation=(5, 1))
            if n_biggest is not None:
                draw_registration_result(bl_tumors_pc_real_space_o3d, fu_tumors_pc, result_icp.transformation, window_name='After ICP (All Tumors)',
                                         dynamic_rotation=(5, 1))

        if save_results:
            # transform = AffineTransform(np.linalg.pinv(file.affine) @ result_icp.transformation @ file.affine)
        # transformed_bl_labeled_tumors = warp(bl_labeled_tumors, transform.inverse, order=0)

            # todo test maybe you can delete
            # import SimpleITK as sitk
            # transform = np.linalg.inv(file.affine) @ result_icp.transformation @ file.affine
            # aff = sitk.AffineTransform(3)
            # aff.SetMatrix(transform[:-1, :-1].flatten())
            # aff.SetTranslation(transform[:-1, -1].flatten())
            # bl_tumors_sitk_img = sitk.ReadImage(bl_labeled_tumors_file)
            # elastixImageFilter = sitk.ElastixImageFilter()
            # elastixImageFilter.SetFixedImage(sitk.ReadImage(f'{dir_name}/FU_Scan_CT.nii.gz'))
            # elastixImageFilter.SetMovingImage(sitk.ReadImage(f'{dir_name}/BL_Scan_CT.nii.gz'))
            # elastixImageFilter.Execute()
            # # bl_tumors_sitk_img = sitk.GetImageFromArray(bl_labeled_tumors.astype(np.int16))
            # # bl_tumors_sitk_img.SetSpacing(tuple(float(s) for s in file.header.get_zooms()))
            # transform_parameter_map = sitk.GetDefaultParameterMap(aff)
            # sitk.GridImageSource()
            # temp = elastixImageFilter.GetTransformParameterMap()
            # temp[0]["ResampleInterpolator"] = ["FinalNearestNeighborInterpolator"]
            # resultLiver = sitk.Transformix(movingLiver, temp)
            # resultTumors = sitk.Transformix(movingTumors, temp)
            # print()

            def transform_label_image(image, transformation, voxel_to_real_space_transform):
                # extract a point cloud of the label points
                pc_voxel_space = np.stack(np.where(image > 0)).T

                # transfer the points to real space
                pc_real_space_o3d = o3d.geometry.PointCloud()
                pc_real_space_o3d.points = o3d.utility.Vector3dVector(affines.apply_affine(voxel_to_real_space_transform, pc_voxel_space))

                # apply transformation
                pc_real_space_o3d.transform(transformation)

                # transfer moved points to voxel space
                real_to_voxel_space_transform = np.linalg.inv(voxel_to_real_space_transform)
                pc_voxel_space_moved = affines.apply_affine(real_to_voxel_space_transform, np.asarray(pc_real_space_o3d.points))

                # round moved points to integers
                pc_voxel_space_moved = np.round(pc_voxel_space_moved).astype(np.int16)

                # filter relevant points
                relevant_points_filter = np.all(pc_voxel_space_moved < np.asarray(image.shape), axis=1)
                pc_voxel_space_moved = pc_voxel_space_moved[relevant_points_filter]
                pc_voxel_space = pc_voxel_space[relevant_points_filter]

                # insert moved points to result image
                res = np.zeros_like(image)
                res[pc_voxel_space_moved[:, 0], pc_voxel_space_moved[:, 1], pc_voxel_space_moved[:, 2]] = image[pc_voxel_space[:, 0], pc_voxel_space[:, 1], pc_voxel_space[:, 2]]

                from skimage.morphology import closing, ball, disk
                res = closing(res, disk(2).reshape(5, 5, 1))
                # res = binary_fill_holes(res, np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]).reshape([3, 3, 1]).astype(np.bool))

                return res




            transform_inverse = np.linalg.inv(file.affine) @ np.linalg.inv(result_icp.transformation) @ file.affine

            transformed_bl_labeled_tumors = affine_transform(bl_labeled_tumors, transform_inverse, order=0)
            # transformed_bl_labeled_tumors = transform_label_image(bl_labeled_tumors, result_icp.transformation, file.affine)

            bl_liver_file = f'{dir_name}/BL_Scan_Liver.nii.gz'
            bl_CT_file = f'{dir_name}/BL_Scan_CT.nii.gz'

            bl_liver_case, _ = load_nifti_data(bl_liver_file)
            bl_CT_case, _ = load_nifti_data(bl_CT_file)


            transformed_bl_liver_case = affine_transform(bl_liver_case, transform_inverse, order=0)
            # transformed_bl_liver_case = transform_label_image(bl_liver_case, result_icp.transformation, file.affine)

            transformed_bl_CT_case = affine_transform(bl_CT_case, transform_inverse)

            add_to_file_name = lambda f: os.path.dirname(f) + f'/improved_registration_Only_ICP_point_to_plane_' + os.path.basename(f)
            # add_to_file_name = lambda f: os.path.dirname(f) + f'/3you_can_delete___transformed_' + os.path.basename(f)

            save(Nifti1Image(transformed_bl_CT_case, file.affine), add_to_file_name(bl_CT_file))
            save(Nifti1Image(transformed_bl_liver_case, file.affine), add_to_file_name(bl_liver_file))
            save(Nifti1Image(transformed_bl_labeled_tumors, file.affine), add_to_file_name(bl_labeled_tumors_file))

        print(f'{i}: finished registration of {case_name} in {calculate_runtime(tm)} (hh:mm:ss)')



    # -------------------------------------------------------------------------------------------------------

    # threshold = 0.0000002
    # # trans_init = np.eye(4)
    # draw_registration_result(bl_pc, fu_pc)
    # trans_init = np.array([[ 0.98412936,  0.17314167,  0.03887618, 16.6820229],
    #                        [-0.17217298,  0.98469545, -0.02704311, 10.18500152],
    #                        [-0.04296349,  0.01992049,  0.99887803, 7.58492001],
    #                        [-0.,         -0.,         -0.,         1.        ]])
    #
    # print('start registration')
    # reg_p2p = o3d.pipelines.registration.registration_icp(
    #     bl_pc, fu_pc, threshold, trans_init,
    #     o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    #     o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=5000))
    #
    # transformed_bl_pc = copy.deepcopy(bl_pc)
    # transformed_bl_pc.transform(reg_p2p.transformation)
    # transformed_bl_points = np.asarray(transformed_bl_pc.points)
    # print(np.all(np.isclose(transformed_bl_points, bl_points)))
    #
    # draw_registration_result(bl_pc, fu_pc, reg_p2p.transformation)
    #
    # print()

