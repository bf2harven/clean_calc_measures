import os
from glob import glob
from typing import Tuple, List, Collection, Callable, Optional, Dict, Union, Iterator
import numpy as np
import pandas as pd
from nibabel import load, as_closest_canonical, affines, save, Nifti1Image
from scipy.ndimage import _ni_support
from scipy.ndimage.morphology import distance_transform_edt, binary_erosion,\
    generate_binary_structure
from scipy import ndimage
from skimage.measure import label
from skimage.morphology import disk
from scipy.ndimage import binary_fill_holes
from skimage.morphology import remove_small_objects, ball
from xlsxwriter.utility import xl_col_to_name
from time import time, gmtime
from medpy.metric import hd, assd
from multiprocessing.pool import Pool
from functools import partial
from skimage.measure import regionprops


def _surface_distances(result, reference, voxelspacing=None, connectivity=1):
    """
    This function is copied from medpy.metric version 0.3.0

    The distances between the surface voxel of binary objects in result and their
    nearest partner surface voxel of a binary object in reference.
    """
    result = np.atleast_1d(result.astype(np.bool_))
    reference = np.atleast_1d(reference.astype(np.bool_))
    if voxelspacing is not None:
        voxelspacing = _ni_support._normalize_sequence(voxelspacing, result.ndim)
        voxelspacing = np.asarray(voxelspacing, dtype=np.float64)
        if not voxelspacing.flags.contiguous:
            voxelspacing = voxelspacing.copy()

    # binary structure
    footprint = generate_binary_structure(result.ndim, connectivity)

    # test for emptiness
    if 0 == np.count_nonzero(result):
        raise RuntimeError('The first supplied array does not contain any binary object.')
    if 0 == np.count_nonzero(reference):
        raise RuntimeError('The second supplied array does not contain any binary object.')

        # extract only 1-pixel border line of objects
    result_border = result ^ binary_erosion(result, structure=footprint, iterations=1)
    reference_border = reference ^ binary_erosion(reference, structure=footprint, iterations=1)

    # compute average surface distance
    # Note: scipys distance transform is calculated only inside the borders of the
    #       foreground objects, therefore the input has to be reversed
    dt = distance_transform_edt(~reference_border, sampling=voxelspacing)
    sds = dt[result_border]

    return sds


def crop_to_relevant_joint_bbox(result, reference):
    relevant_joint_case = np.logical_or(result, reference)
    slc = get_slice_of_cropped_relevant_bbox(relevant_joint_case)
    return result[slc], reference[slc]


def get_slice_of_cropped_relevant_bbox(case: np.ndarray):

    if case.ndim == 3:
        xmin, xmax, ymin, ymax, zmin, zmax = bbox2_3D(case)
        xmax += 1
        ymax += 1
        zmax += 1
        slc = np.s_[xmin: xmax, ymin: ymax, zmin: zmax]
    else:
        xmin, xmax, ymin, ymax = bbox2_2D(case)
        xmax += 1
        ymax += 1
        slc = np.s_[xmin: xmax, ymin: ymax]

    return slc


def min_distance(result, reference, voxelspacing=None, connectivity: int = 1, crop_to_relevant_scope: bool = True):
    """
    The concept is taken from medpy.metric.hd version 0.3.0

    Minimum Distance.

    Computes the (symmetric) Minimum Distance between the binary objects in two images. It is defined as the minimum
    surface distance between the objects (Hausdorff Distance however, is defined as the maximum surface distance between
    the objects).

    Parameters
    ----------
    result : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    reference : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    voxelspacing : float or sequence of floats, optional
        The voxelspacing in a distance unit i.e. spacing of elements
        along each dimension. If a sequence, must be of length equal to
        the input rank; if a single number, this is used for all axes. If
        not specified, a grid spacing of unity is implied.
    connectivity : int
        The neighbourhood/connectivity considered when determining the surface
        of the binary objects. This value is passed to
        `scipy.ndimage.morphology.generate_binary_structure` and should usually be :math:`> 1`.
        Note that the connectivity influences the result in the case of the Hausdorff distance.
    crop_to_relevant_scope : bool
        If set to True (by default) the two cases holding the objects will be cropped to the relevant region
        to save running time.

    Returns
    -------
    min_distance : float
        The symmetric Minimum Distance between the object(s) in ```result``` and the
        object(s) in ```reference```. The distance unit is the same as for the spacing of
        elements along each dimension, which is usually given in mm.

    Notes
    -----
    This is a real metric. The binary images can therefore be supplied in any order.
    """
    if crop_to_relevant_scope:
        result, reference = crop_to_relevant_joint_bbox(result, reference)
    if np.any(np.logical_and(result, reference)):
        md = np.float64(0)
    else:
        md = _surface_distances(result, reference, voxelspacing, connectivity).min()
    return md


def Hausdorff(result, reference, voxelspacing=None, connectivity: int = 1, crop_to_relevant_scope: bool = True):
    """
    The concept is taken from medpy.metric.hd

    Hausdorff Distance.

    Computes the (symmetric) Hausdorff Distance (HD) between the binary objects in two
    images. It is defined as the maximum surface distance between the objects.

    Parameters
    ----------
    result : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    reference : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    voxelspacing : float or sequence of floats, optional
        The voxelspacing in a distance unit i.e. spacing of elements
        along each dimension. If a sequence, must be of length equal to
        the input rank; if a single number, this is used for all axes. If
        not specified, a grid spacing of unity is implied.
    connectivity : int
        The neighbourhood/connectivity considered when determining the surface
        of the binary objects. This value is passed to
        `scipy.ndimage.morphology.generate_binary_structure` and should usually be :math:`> 1`.
        Note that the connectivity influences the result in the case of the Hausdorff distance.
    crop_to_relevant_scope : bool
        If set to True (by default) the two cases holding the objects will be cropped to the relevant region
        to save running time.

    Returns
    -------
    hd : float
        The symmetric Hausdorff Distance between the object(s) in ```result``` and the
        object(s) in ```reference```. The distance unit is the same as for the spacing of
        elements along each dimension, which is usually given in mm.

    Notes
    -----
    This is a real metric. The binary images can therefore be supplied in any order.
    """
    if crop_to_relevant_scope:
        result, reference = crop_to_relevant_joint_bbox(result, reference)
    return hd(result, reference, voxelspacing, connectivity)


def ASSD(result, reference, voxelspacing=None, connectivity: int = 1, crop_to_relevant_scope: bool = True):
    """
    The concept is taken from medpy.metric.assd

    Average symmetric surface distance.

    Computes the average symmetric surface distance (ASD) between the binary objects in
    two images.

    Parameters
    ----------
    result : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    reference : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    voxelspacing : float or sequence of floats, optional
        The voxelspacing in a distance unit i.e. spacing of elements
        along each dimension. If a sequence, must be of length equal to
        the input rank; if a single number, this is used for all axes. If
        not specified, a grid spacing of unity is implied.
    connectivity : int
        The neighbourhood/connectivity considered when determining the surface
        of the binary objects. This value is passed to
        `scipy.ndimage.morphology.generate_binary_structure` and should usually be :math:`> 1`.
        The decision on the connectivity is important, as it can influence the results
        strongly. If in doubt, leave it as it is.
    crop_to_relevant_scope : bool
        If set to True (by default) the two cases holding the objects will be cropped to the relevant region
        to save running time.

    Returns
    -------
    assd : float
        The average symmetric surface distance between the object(s) in ``result`` and the
        object(s) in ``reference``. The distance unit is the same as for the spacing of
        elements along each dimension, which is usually given in mm.

    Notes
    -----
    This is a real metric, obtained by calling and averaging

    >>> asd(result, reference)

    and

    >>> asd(reference, result)

    The binary images can therefore be supplied in any order.
    """
    if crop_to_relevant_scope:
        result, reference = crop_to_relevant_joint_bbox(result, reference)
    return assd(result, reference, voxelspacing, connectivity)


def assd_and_hd(result, reference, voxelspacing=None, connectivity: int = 1, crop_to_relevant_scope: bool = True):
    """
    The concept is taken from medpy.metric.assd and medpy.metric.hd

    Average symmetric surface distance and Hausdorff Distance.

    Computes the average symmetric surface distance (ASD) and the (symmetric) Hausdorff Distance (HD) between the binary objects in
    two images.

    Parameters
    ----------
    result : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    reference : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    voxelspacing : float or sequence of floats, optional
        The voxelspacing in a distance unit i.e. spacing of elements
        along each dimension. If a sequence, must be of length equal to
        the input rank; if a single number, this is used for all axes. If
        not specified, a grid spacing of unity is implied.
    connectivity : int
        The neighbourhood/connectivity considered when determining the surface
        of the binary objects. This value is passed to
        `scipy.ndimage.morphology.generate_binary_structure` and should usually be :math:`> 1`.
        The decision on the connectivity is important, as it can influence the results
        strongly. If in doubt, leave it as it is.
    crop_to_relevant_scope : bool
        If set to True (by default) the two cases holding the objects will be cropped to the relevant region
        to save running time.

    Returns
    -------
    (assd, hd) : Tuple(float, float)
        The average symmetric surface distance and The symmetric Hausdorff Distance between the object(s) in ``result`` and the
        object(s) in ``reference``. The distance unit is the same as for the spacing of
        elements along each dimension, which is usually given in mm.

    Notes
    -----
    These are real metrics. The binary images can therefore be supplied in any order.
    """

    if crop_to_relevant_scope:
        result, reference = crop_to_relevant_joint_bbox(result, reference)

    sds1 = _surface_distances(result, reference, voxelspacing, connectivity)
    sds2 = _surface_distances(reference, result, voxelspacing, connectivity)

    assd_res = np.mean((sds1.mean(), sds2.mean()))
    hd_res = max(sds1.max(), sds2.max())

    return assd_res, hd_res


def assd_hd_and_min_distance(result, reference, voxelspacing=None, connectivity: int = 1, crop_to_relevant_scope: bool = True):
    """
    The concept is taken from medpy.metric.assd and medpy.metric.hd

    Average symmetric surface distance, Hausdorff Distance and Minimum Distance.

    Computes the average symmetric surface distance (ASD), the (symmetric) Hausdorff Distance (HD) and the (symmetric)
    Minimum Distance between the binary objects in two images.

    Parameters
    ----------
    result : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    reference : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    voxelspacing : float or sequence of floats, optional
        The voxelspacing in a distance unit i.e. spacing of elements
        along each dimension. If a sequence, must be of length equal to
        the input rank; if a single number, this is used for all axes. If
        not specified, a grid spacing of unity is implied.
    connectivity : int
        The neighbourhood/connectivity considered when determining the surface
        of the binary objects. This value is passed to
        `scipy.ndimage.morphology.generate_binary_structure` and should usually be :math:`> 1`.
        The decision on the connectivity is important, as it can influence the results
        strongly. If in doubt, leave it as it is.
    crop_to_relevant_scope : bool
        If set to True (by default) the two cases holding the objects will be cropped to the relevant region
        to save running time.

    Returns
    -------
    (assd, hd) : Tuple(float, float)
        The average symmetric surface distance and The symmetric Hausdorff Distance between the object(s) in ``result`` and the
        object(s) in ``reference``. The distance unit is the same as for the spacing of
        elements along each dimension, which is usually given in mm.

    Notes
    -----
    These are real metrics. The binary images can therefore be supplied in any order.
    """

    if crop_to_relevant_scope:
        result, reference = crop_to_relevant_joint_bbox(result, reference)

    sds1 = _surface_distances(result, reference, voxelspacing, connectivity)
    sds2 = _surface_distances(reference, result, voxelspacing, connectivity)

    assd_res = np.mean((sds1.mean(), sds2.mean()))
    hd_res = max(sds1.max(), sds2.max())
    if np.any(np.logical_and(result, reference)):
        md_res = np.float64(0)
    else:
        md_res = sds1.min()

    return assd_res, hd_res, md_res


def dice(gt_seg, prediction_seg):
    """
    compute dice coefficient
    :param gt_seg:
    :param prediction_seg:
    :return: dice coefficient between gt and predictions
    """
    seg1 = np.asarray(gt_seg).astype(np.bool_)
    seg2 = np.asarray(prediction_seg).astype(np.bool_)

    # Compute Dice coefficient
    intersection = np.logical_and(seg1, seg2)
    denominator_of_res = seg1.sum() + seg2.sum()
    if denominator_of_res == 0:
        return 1
    return 2. * intersection.sum() / denominator_of_res


def approximate_diameter(volume):
    r = ((3 * volume) / (4 * np.pi)) ** (1 / 3)
    diameter = 2 * r
    return diameter


def get_connected_components(Map, connectivity=None):
    """
    Remove Small connected component
    """
    label_img = label(Map, connectivity=connectivity)
    cc_num = label_img.max()
    cc_areas = ndimage.sum(Map, label_img, range(cc_num + 1))
    area_mask = (cc_areas <= 10)
    label_img[area_mask[label_img]] = 0
    return_value = label(label_img)
    return return_value


def getLargestCC(segmentation, connectivity=1):
    labels = label(segmentation, connectivity=connectivity)
    assert (labels.max() != 0)  # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    return largestCC.astype(segmentation.dtype)


def load_nifti_data(nifti_file_name: str):
    """
    Loading data from a nifti file.

    :param nifti_file_name: The path to the desired nifti file.

    :return: A tuple in the following form: (data, file), where:
        • data is a ndarray containing the loaded data.
        • file is the file object that was loaded.
    """

    # loading nifti file
    nifti_file = load(nifti_file_name)
    nifti_file = as_closest_canonical(nifti_file)

    # extracting the data of the file
    data = nifti_file.get_fdata().astype(np.float32)


    return data, nifti_file


def get_liver_border(liver_case: np.ndarray, selem_radius: int = 1) -> np.ndarray:
    return np.logical_xor(liver_case, binary_erosion(liver_case, np.expand_dims(disk(selem_radius), 2))).astype(
        liver_case.dtype)


def expand_labels(label_image, distance=1, voxelspacing=None, distance_cache=None, return_distance_cache=False):
    """

    This function is based on the same named function in skimage.segmentation version 0.18.3

    expand_labels is derived from code that was
    originally part of CellProfiler, code licensed under BSD license.
    Website: http://www.cellprofiler.org

    Copyright (c) 2020 Broad Institute
    All rights reserved.

    Original authors: CellProfiler team


    Expand labels in label image by ``distance`` pixels without overlapping.
    Given a label image, ``expand_labels`` grows label regions (connected components)
    outwards by up to ``distance`` pixels without overflowing into neighboring regions.
    More specifically, each background pixel that is within Euclidean distance
    of <= ``distance`` pixels of a connected component is assigned the label of that
    connected component.
    Where multiple connected components are within ``distance`` pixels of a background
    pixel, the label value of the closest connected component will be assigned (see
    Notes for the case of multiple labels at equal distance).
    Parameters
    ----------
    label_image : ndarray of dtype int
        label image
    distance : float
        Euclidean distance in pixels by which to grow the labels. Default is one.
    voxelspacing : float or sequence of floats, optional
        The voxelspacing in a distance unit i.e. spacing of elements
        along each dimension. If a sequence, must be of length equal to
        the input rank; if a single number, this is used for all axes. If
        not specified, a grid spacing of unity is implied.
    distance_cache : a tuple with 2 ndarrays, optional
        This two ndarrays are distances calculated earlyer to use in the current calculation
        This is used, for example, if you want to run this function several times while changing only
        the ``distance`` parameter. The calculation will be more optimized.
    return_distance_cache : bool, optional
        If this is set to True, the distances cache will be returned too. By default it's False.
        See distance_cache decumentation.
    Returns
    -------
    enlarged_labels : ndarray of dtype int
        Labeled array, where all connected regions have been enlarged
    distance_cache : a tuple with 2 ndarrays
        This will be returned only if return_distance_cache is set to True.
        See distance_cache decumentation.
    Notes
    -----
    Where labels are spaced more than ``distance`` pixels are apart, this is
    equivalent to a morphological dilation with a disc or hyperball of radius ``distance``.
    However, in contrast to a morphological dilation, ``expand_labels`` will
    not expand a label region into a neighboring region.
    This implementation of ``expand_labels`` is derived from CellProfiler [1]_, where
    it is known as module "IdentifySecondaryObjects (Distance-N)" [2]_.
    There is an important edge case when a pixel has the same distance to
    multiple regions, as it is not defined which region expands into that
    space. Here, the exact behavior depends on the upstream implementation
    of ``scipy.ndimage.distance_transform_edt``.
    See Also
    --------
    :func:`skimage.measure.label`, :func:`skimage.segmentation.watershed`, :func:`skimage.morphology.dilation`
    References
    ----------
    .. [1] https://cellprofiler.org
    .. [2] https://github.com/CellProfiler/CellProfiler/blob/082930ea95add7b72243a4fa3d39ae5145995e9c/cellprofiler/modules/identifysecondaryobjects.py#L559
    Examples
    --------
    >>> labels = np.array([0, 1, 0, 0, 0, 0, 2])
    >>> expand_labels(labels, distance=1)
    array([1, 1, 1, 0, 0, 2, 2])
    Labels will not overwrite each other:
    >>> expand_labels(labels, distance=3)
    array([1, 1, 1, 1, 2, 2, 2])
    In case of ties, behavior is undefined, but currently resolves to the
    label closest to ``(0,) * ndim`` in lexicographical order.
    >>> labels_tied = np.array([0, 1, 0, 2, 0])
    >>> expand_labels(labels_tied, 1)
    array([1, 1, 1, 2, 2])
    >>> labels2d = np.array(
    ...     [[0, 1, 0, 0],
    ...      [2, 0, 0, 0],
    ...      [0, 3, 0, 0]]
    ... )
    >>> expand_labels(labels2d, 1)
    array([[2, 1, 1, 0],
           [2, 2, 0, 0],
           [2, 3, 3, 0]])
    """
    if distance_cache is None:
        distances, nearest_label_coords = distance_transform_edt(
            label_image == 0, return_indices=True, sampling=voxelspacing
        )
    else:
        distances, nearest_label_coords = distance_cache
    labels_out = np.zeros_like(label_image)
    dilate_mask = distances <= distance
    # build the coordinates to find nearest labels,
    # in contrast to [1] this implementation supports label arrays
    # of any dimension
    masked_nearest_label_coords = [
        dimension_indices[dilate_mask]
        for dimension_indices in nearest_label_coords
    ]
    nearest_labels = label_image[tuple(masked_nearest_label_coords)]
    labels_out[dilate_mask] = nearest_labels
    if return_distance_cache:
        return labels_out, (distances, nearest_label_coords)
    return labels_out


def expand_per_label(label_image: np.ndarray, dists_to_expand: Union[float, np.ndarray] = 1, max_relevant_distances: Union[float, np.ndarray] = None,
                     voxelspacing=None, distance_cache=None, return_distance_cache=False):

    if max_relevant_distances is None or not return_distance_cache:
        max_relevant_distances = dists_to_expand

    # in case the distance to expand is equivalent for all labels
    if np.unique(dists_to_expand).size == 1:
        return expand_labels(label_image,
                             dists_to_expand if np.asarray(dists_to_expand).size == 1 else dists_to_expand[0],
                             voxelspacing, distance_cache, return_distance_cache)

    unique_labels = np.unique(label_image)
    unique_labels = unique_labels[unique_labels != 0]

    dists_to_expand = np.asarray(dists_to_expand)
    max_relevant_distances = np.asarray(max_relevant_distances)

    assert dists_to_expand.size == unique_labels.size
    assert max_relevant_distances.size == unique_labels.size

    # calculating the distances
    if distance_cache is None:
        distances = np.empty((unique_labels.size,) + label_image.shape, dtype=np.float32)
        for i in range(unique_labels.size):
            distances[i] = distance_transform_edt_for_certain_label((unique_labels[i], max_relevant_distances[i]), label_image, voxelspacing, return_indices=False)
    else:
        distances = distance_cache

    dilate_mask = (distances <= dists_to_expand.reshape([-1, *([1] * (distances.ndim - 1))]))
    treated_distances = np.where(dilate_mask, distances, np.inf)

    labels_out_ind = np.argmin(treated_distances, axis=0)
    min_dist_between_label_val = np.squeeze(np.take_along_axis(treated_distances, np.expand_dims(labels_out_ind, 0), 0), axis=0)
    labels_out_ind[min_dist_between_label_val == np.inf] = -1
    labels_out_ind += 1
    
    labels_out = np.concatenate([[0], unique_labels])[labels_out_ind].astype(np.float32)
    
    if return_distance_cache:
        return labels_out, distances
    return labels_out


def distance_transform_edt_for_certain_label(label_and_max_relevant_dist, label_image, voxelspacing, return_indices):
    label, max_relevant_dist = label_and_max_relevant_dist
    return fast_distance_transform_edt(label_image == label, voxelspacing, max_relevant_dist, return_indices=return_indices)


def fast_distance_transform_edt(input, voxelspacing, max_relevant_dist, return_indices):

    input_shape = input.shape

    size_to_extend = np.ceil(max_relevant_dist / np.asarray(voxelspacing)).astype(np.int16)

    assert input.ndim in [2, 3]

    if input.ndim == 3:
        xmin, xmax, ymin, ymax, zmin, zmax = bbox2_3D(input)
    else:
        xmin, xmax, ymin, ymax = bbox2_2D(input)

    xmax += 1
    ymax += 1

    xmin = max(0, xmin - size_to_extend[0])
    xmax = min(input.shape[0], xmax + size_to_extend[0])

    ymin = max(0, ymin - size_to_extend[1])
    ymax = min(input.shape[1], ymax + size_to_extend[1])

    if input.ndim == 3:
        zmax += 1
        zmin = max(0, zmin - size_to_extend[2])
        zmax = min(input.shape[2], zmax + size_to_extend[2])

        slc = np.s_[xmin: xmax, ymin: ymax, zmin: zmax]
    else:
        slc = np.s_[xmin: xmax, ymin: ymax]

    # cropping the input image to the relevant are
    input = input[slc]

    if return_indices:
        distances, nearest_label_coords = distance_transform_edt(input == 0, return_indices=return_indices, sampling=voxelspacing)

        # extending the results to the input image shape
        nearest_label_coords[0] += xmin
        nearest_label_coords[1] += ymin
        if input.ndim == 3:
            nearest_label_coords[2] += zmin

        extended_nearest_label_coords = np.zeros((input.ndim,) + input_shape)
        if input.ndim == 3:
            extended_nearest_label_coords[:, slc[0], slc[1], slc[2]] = nearest_label_coords
        else:
            extended_nearest_label_coords[:, slc[0], slc[1]] = nearest_label_coords

    else:
        distances = distance_transform_edt(input == 0, return_indices=return_indices, sampling=voxelspacing)

    # extending the results to the input image shape
    extended_distances = np.ones(input_shape) * np.inf
    extended_distances[slc] = distances

    if return_indices:
        return extended_distances, extended_nearest_label_coords

    return extended_distances

# def shrink_labels(label_image, distance=1, voxelspacing=None, distance_cache=None, return_distance_cache=False,
#                   is_the_label_image_binary=False):
#     """
#
#     This function is based on the function expand_labels in skimage.segmentation version 0.18.3, with changes made to
#     be a shrink version of it.
#
#     See decumentation of the funcion expand_labels here.
#
#     Shrink labels in label image by ``distance`` pixels.
#     Given a label image, ``shrink_labels`` decreas label regions (connected components)
#     inwards by up to ``distance`` pixels.
#
#     Parameters
#     ----------
#     label_image : ndarray of dtype int
#         label image
#     distance : float
#         Euclidean distance in pixels by which to grow the labels. Default is one.
#     voxelspacing : float or sequence of floats, optional
#         The voxelspacing in a distance unit i.e. spacing of elements
#         along each dimension. If a sequence, must be of length equal to
#         the input rank; if a single number, this is used for all axes. If
#         not specified, a grid spacing of unity is implied.
#     distance_cache : ndarray of dtype float, optional
#         The distances calculated earlier to use in the current calculation
#         This is used, for example, if you want to run this function several times while changing only
#         the ``distance`` parameter. The calculation will be more optimized.
#     return_distance_cache : bool, optional
#         If this is set to True, the distances cache will be returned too. By default it's False.
#         See distance_cache documentation.
#     is_the_label_image_binary : bool, optional
#         If this is set to True, the function assumes that the given label_image is consisted of only 0,1 labels, and
#         calculates some calculations faster.
#     Returns
#     -------
#     reduced_labels : ndarray of dtype int
#         Labeled array, where all connected regions have been reduced
#     distance_cache : ndarray of dtype float
#         This will be returned only if return_distance_cache is set to True.
#         See distance_cache documentation.
#     """
#
#     if distance_cache is None:
#         # extract borders
#         morph_func = binary_erosion if is_the_label_image_binary else erosion
#         # borders = np.logical_and(label_image, np.logical_not(morph_func(label_image, ball(1)))).astype(label_image.dtype)
#         borders = np.logical_and(label_image, np.logical_not(morph_func(label_image, np.array(
#             [[0, 1, 0], [1, 1, 1], [0, 1, 0]]).reshape([3, 3, 1]).astype(np.bool_)))).astype(label_image.dtype)
#         borders = label_image * borders
#
#         distances = distance_transform_edt(borders == 0, sampling=voxelspacing)
#     else:
#         distances = distance_cache
#     labels_out = np.zeros_like(label_image)
#     dilate_mask = distances >= distance
#
#     labels_out[dilate_mask] = 1
#     labels_out *= label_image
#     if return_distance_cache:
#         return labels_out, distances
#     return labels_out


# def find_pairs(baseline_moved_labeled, followup_labeled, reverse=False, voxelspacing=None, max_dilate_param=5,
#                return_iteration_indicator=False):
#     list_of_pairs = []
#     # Hyper-parameter for sensitivity of the matching (5 means that 10 pixels between the scans will be same)
#     for dilate in range(max_dilate_param):
#
#         # find intersection areas between baseline and followup
#         matched = np.logical_and((baseline_moved_labeled > 0), followup_labeled > 0)
#
#         # iterate over tumors in baseline that intersect tumors in followup
#         for i in np.unique((matched * baseline_moved_labeled).flatten()):
#             if i == 0:
#                 continue
#
#             # find intersection between current BL tumor and FU tumors
#             overlap = ((baseline_moved_labeled == i) * followup_labeled)
#
#             # get the labels of the FU tumors that intersect the current BL tumor
#             follow_up_num = np.unique(overlap.flatten())
#
#             # in case there is more than 1 FU tumor that intersect the current BL tumor
#             if follow_up_num.shape[0] > 2:
#                 sum = 0
#                 # iterate over the FU tumors that intersect the current BL tumor
#                 for j in follow_up_num[1:]:
#                     # in case the current FU tumor has the biggest found overlap with current BL tumor,
#                     # till this iteration
#                     if sum < (overlap == j).sum():
#                         sum = (overlap == j).sum()
#                         biggest = j
#
#                     # in case the overlap of the current FU tumor with the current BL tumor
#                     # is grader than 10% of the BL or FU tumor
#                     elif ((overlap == j).sum() / (followup_labeled == j).sum()) > 0.1 or (
#                             (overlap == j).sum() / (baseline_moved_labeled == i).sum()) > 0.1:
#                         # a match was found
#                         if reverse:
#                             if return_iteration_indicator:
#                                 list_of_pairs.append((dilate + 1, (j, i)))
#                             else:
#                                 list_of_pairs.append((j, i))
#                         else:
#                             if return_iteration_indicator:
#                                 list_of_pairs.append((dilate + 1, (i, j)))
#                             else:
#                                 list_of_pairs.append((i, j))
#                         # zero the current FU tumor and the current BL tumor
#                         baseline_moved_labeled[baseline_moved_labeled == i] = 0
#                         followup_labeled[followup_labeled == j] = 0
#                 # marking the FU tumor with the biggest overlap with the BL tumor as a found match
#                 if reverse:
#                     if return_iteration_indicator:
#                         list_of_pairs.append((dilate + 1, (biggest, i)))
#                     else:
#                         list_of_pairs.append((biggest, i))
#                 else:
#                     if return_iteration_indicator:
#                         list_of_pairs.append((dilate + 1, (i, biggest)))
#                     else:
#                         list_of_pairs.append((i, biggest))
#
#                 # zero the current BL tumor and the FU tumor that has jost been
#                 # marked as a match with the current BL tumor
#                 baseline_moved_labeled[baseline_moved_labeled == i] = 0
#                 followup_labeled[followup_labeled == biggest] = 0
#
#             # in case there is only 1 FU tumor that intersect the current BL tumor
#             elif follow_up_num.shape[0] > 1:
#                 # a match was found
#                 if reverse:
#                     if return_iteration_indicator:
#                         list_of_pairs.append((dilate + 1, (follow_up_num[-1], i)))
#                     else:
#                         list_of_pairs.append((follow_up_num[-1], i))
#                 else:
#                     if return_iteration_indicator:
#                         list_of_pairs.append((dilate + 1, (i, follow_up_num[-1])))
#                     else:
#                         list_of_pairs.append((i, follow_up_num[-1]))
#                 # zero the current BL tumor and the FU tumor that intersects it
#                 baseline_moved_labeled[baseline_moved_labeled == i] = 0
#                 followup_labeled[followup_labeled == follow_up_num[-1]] = 0
#
#         # dilate the BL and FU tumors cases
#         # if dilate % 5 == 3:
#         #     plus = ndimage.generate_binary_structure(3, 1)
#         #     plus[1, 1, 0] = False
#         #     plus[1, 1, 2] = False
#         #     baseline_moved_labeled = dilation(baseline_moved_labeled, selem=ndimage.generate_binary_structure(3, 1))
#         #     followup_labeled = dilation(followup_labeled, selem=plus)
#         # else:
#         #     plus = ndimage.generate_binary_structure(3, 1)
#         #     plus[1, 1, 0] = False
#         #     plus[1, 1, 2] = False
#         #     baseline_moved_labeled = dilation(baseline_moved_labeled, selem=plus)
#         #     followup_labeled = dilation(followup_labeled, selem=plus)
#
#         # dilation without overlap and considering resolution
#         baseline_moved_labeled = expand_labels(baseline_moved_labeled, distance=1, voxelspacing=voxelspacing)
#         followup_labeled = expand_labels(followup_labeled, distance=1, voxelspacing=voxelspacing)
#
#     return (list_of_pairs)

def find_pairs(baseline_moved_labeled, followup_labeled, reverse=False, voxelspacing=None, max_dilate_param=5,
               return_iteration_and_reverse_indicator=False):

    working_baseline_moved_labeled = baseline_moved_labeled
    working_followup_labeled = followup_labeled

    distance_cache_bl, distance_cache_fu = None, None

    bl_matched_tumors, fu_matched_tumors = [], []

    list_of_pairs = []
    # Hyper-parameter for sensitivity of the matching (5 means that 10 pixels between the scans will be same)
    for dilate in range(max_dilate_param):

        # find intersection areas between baseline and followup
        matched = np.logical_and((working_baseline_moved_labeled > 0), working_followup_labeled > 0)

        # iterate over tumors in baseline that intersect tumors in followup
        for i in np.unique((matched * working_baseline_moved_labeled).flatten()):
            if i == 0:
                continue

            # find intersection between current BL tumor and FU tumors
            overlap = ((working_baseline_moved_labeled == i) * working_followup_labeled)

            # get the labels of the FU tumors that intersect the current BL tumor
            follow_up_num = np.unique(overlap.flatten())

            # in case there is more than 1 FU tumor that intersect the current BL tumor
            if follow_up_num.shape[0] > 2:
                sum = 0
                # iterate over the FU tumors that intersect the current BL tumor
                for j in follow_up_num[1:]:
                    # in case the current FU tumor has the biggest found overlap with current BL tumor,
                    # till this iteration
                    if sum < (overlap == j).sum():
                        sum = (overlap == j).sum()
                        biggest = j

                    # in case the overlap of the current FU tumor with the current BL tumor
                    # is grader than 10% of the BL or FU tumor
                    elif ((overlap == j).sum() / (working_followup_labeled == j).sum()) > 0.1 or (
                            (overlap == j).sum() / (working_baseline_moved_labeled == i).sum()) > 0.1:
                        # a match was found
                        if reverse:
                            if return_iteration_and_reverse_indicator:
                                list_of_pairs.append((int(reverse), dilate + 1, (j, i)))
                            else:
                                list_of_pairs.append((j, i))
                        else:
                            if return_iteration_and_reverse_indicator:
                                list_of_pairs.append((int(reverse), dilate + 1, (i, j)))
                            else:
                                list_of_pairs.append((i, j))
                        # zero the current FU tumor and the current BL tumor
                        bl_matched_tumors.append(i)
                        fu_matched_tumors.append(j)
                        working_baseline_moved_labeled[working_baseline_moved_labeled == i] = 0
                        working_followup_labeled[working_followup_labeled == j] = 0
                # marking the FU tumor with the biggest overlap with the current BL tumor as a found match
                if reverse:
                    if return_iteration_and_reverse_indicator:
                        list_of_pairs.append((int(reverse), dilate + 1, (biggest, i)))
                    else:
                        list_of_pairs.append((biggest, i))
                else:
                    if return_iteration_and_reverse_indicator:
                        list_of_pairs.append((int(reverse), dilate + 1, (i, biggest)))
                    else:
                        list_of_pairs.append((i, biggest))

                # zero the current BL tumor and the FU tumor that has jost been
                # marked as a match with the current BL tumor
                bl_matched_tumors.append(i)
                fu_matched_tumors.append(biggest)
                working_baseline_moved_labeled[working_baseline_moved_labeled == i] = 0
                working_followup_labeled[working_followup_labeled == biggest] = 0

            # in case there is only 1 FU tumor that intersect the current BL tumor
            elif follow_up_num.shape[0] > 1:
                # a match was found
                if reverse:
                    if return_iteration_and_reverse_indicator:
                        list_of_pairs.append((int(reverse), dilate + 1, (follow_up_num[-1], i)))
                    else:
                        list_of_pairs.append((follow_up_num[-1], i))
                else:
                    if return_iteration_and_reverse_indicator:
                        list_of_pairs.append((int(reverse), dilate + 1, (i, follow_up_num[-1])))
                    else:
                        list_of_pairs.append((i, follow_up_num[-1]))

                # zero the current BL tumor and the FU tumor that intersects it
                bl_matched_tumors.append(i)
                fu_matched_tumors.append(follow_up_num[-1])
                working_baseline_moved_labeled[working_baseline_moved_labeled == i] = 0
                working_followup_labeled[working_followup_labeled == follow_up_num[-1]] = 0

        if dilate == (max_dilate_param - 1) or np.all(working_baseline_moved_labeled == 0) or np.all(working_followup_labeled == 0):
            break

        # dilation without overlap and considering resolution
        working_baseline_moved_labeled, distance_cache_bl = expand_labels(baseline_moved_labeled, distance=dilate+1,
                                                                          voxelspacing=voxelspacing,
                                                                          distance_cache=distance_cache_bl,
                                                                          return_distance_cache=True)
        working_baseline_moved_labeled[np.isin(working_baseline_moved_labeled, bl_matched_tumors)] = 0
        working_followup_labeled, distance_cache_fu = expand_labels(followup_labeled, distance=dilate+1,
                                                                    voxelspacing=voxelspacing,
                                                                    distance_cache=distance_cache_fu,
                                                                    return_distance_cache=True)
        working_followup_labeled[np.isin(working_followup_labeled, fu_matched_tumors)] = 0

    return (list_of_pairs)


def match_2_cases(BL_tumors_labels, FU_tumors_labels, voxelspacing=None, max_dilate_param=5,
                  return_iteration_and_reverse_indicator=False):
    """
    • This version removes the tumors immediately after their match, not at the end of the iteration (as a result,
        the number of the tumors may affect the final results. Additionally, it requires one check as (bl=BL, fu=FU) and
        one check as (bl=FU, fu=BL)).
    • This version works with python 'for' iterations and not with numpy optimizations.
    """
    first = BL_tumors_labels.copy()
    second = FU_tumors_labels.copy()

    list_of_pairs = find_pairs(first, second, voxelspacing=voxelspacing, max_dilate_param=max_dilate_param,
                               return_iteration_and_reverse_indicator=return_iteration_and_reverse_indicator)

    first = BL_tumors_labels.copy()
    second = FU_tumors_labels.copy()
    list_of_pairs2 = find_pairs(second, first, reverse=True, voxelspacing=voxelspacing,
                                max_dilate_param=max_dilate_param,
                                return_iteration_and_reverse_indicator=return_iteration_and_reverse_indicator)

    resulting_list = list(list_of_pairs)
    if return_iteration_and_reverse_indicator:
        if len(resulting_list) > 0:
            _, _, resulting_matches = zip(*resulting_list)
            resulting_list.extend(x for x in list_of_pairs2 if x[2] not in resulting_matches)
        else:
            resulting_list = list(list_of_pairs2)
    else:
        resulting_list.extend(x for x in list_of_pairs2 if x not in resulting_list)

    return resulting_list


def match_2_cases_v2(baseline_moved_labeled, followup_labeled, voxelspacing=None, max_dilate_param=5,
                  return_iteration_indicator=False):
    """
    • This version removes the tumors only at the end of the iterations.
    • This version works with minimum python 'for' iterations and mostly with numpy optimizations.
    """
    working_baseline_moved_labeled = baseline_moved_labeled
    working_followup_labeled = followup_labeled

    distance_cache_bl, distance_cache_fu = None, None

    if return_iteration_indicator:
        pairs = np.array([]).reshape([0, 3])
    else:
        pairs = np.array([]).reshape([0, 2])
    # Hyper-parameter for sensitivity of the matching (5 means that 10 pixels between the scans will be same)
    for dilate in range(max_dilate_param):

        # find pairs of intersection of tumors
        pairs_of_intersection = np.stack([working_baseline_moved_labeled, working_followup_labeled]).astype(np.int16)
        pairs_of_intersection, overlap_vol  = np.unique(pairs_of_intersection[:, np.all(pairs_of_intersection != 0, axis=0)].T, axis=0, return_counts=True)

        if pairs_of_intersection.size > 0:

            relevant_bl_tumors, relevant_bl_tumors_vol = np.unique(working_baseline_moved_labeled[np.isin(working_baseline_moved_labeled, pairs_of_intersection[:, 0])], return_counts=True)
            relevant_fu_tumors, relevant_fu_tumors_vol = np.unique(working_followup_labeled[np.isin(working_followup_labeled, pairs_of_intersection[:, 1])], return_counts=True)

            # intersection_matrix_overlap_vol[i,j] = #voxels_in_the_overlap_of_relevant_bl_tumors[i]_and_relevant_fu_tumors[j]
            intersection_matrix_overlap_vol = np.zeros((relevant_bl_tumors.size, relevant_fu_tumors.size))
            intersection_matrix_overlap_vol[np.searchsorted(relevant_bl_tumors, pairs_of_intersection[:, 0]),
                                            np.searchsorted(relevant_fu_tumors, pairs_of_intersection[:, 1])] = overlap_vol

            # intersection_matrix_overlap_with_bl_ratio[i,j] = #voxels_in_the_overlap_of_relevant_bl_tumors[i]_and_relevant_fu_tumors[j] / #voxels_in_relevant_bl_tumors[i]
            intersection_matrix_overlap_with_bl_ratio = intersection_matrix_overlap_vol / relevant_bl_tumors_vol.reshape([-1, 1])

            # intersection_matrix_overlap_with_fu_ratio[i,j] = #voxels_in_the_overlap_of_relevant_bl_tumors[i]_and_relevant_fu_tumors[j] / #voxels_in_relevant_fu_tumors[j]
            intersection_matrix_overlap_with_fu_ratio = intersection_matrix_overlap_vol / relevant_fu_tumors_vol.reshape([1, -1])

            # take as a match any maximum intersection with bl tumors
            current_pairs_inds = np.stack([np.arange(relevant_bl_tumors.size), np.argmax(intersection_matrix_overlap_vol, axis=1)]).T

            # take as a match any maximum intersection with fu tumors
            current_pairs_inds = np.unique(np.concatenate([current_pairs_inds, np.stack([np.argmax(intersection_matrix_overlap_vol, axis=0), np.arange(relevant_fu_tumors.size)]).T]), axis=0)

            # take as a match any intersection with a overlap ratio with either bl of fu above 10 percent
            current_pairs_inds = np.unique(np.concatenate([current_pairs_inds, np.stack(np.where(intersection_matrix_overlap_with_bl_ratio > 0.1)).T]), axis=0)
            current_pairs_inds = np.unique(np.concatenate([current_pairs_inds, np.stack(np.where(intersection_matrix_overlap_with_fu_ratio > 0.1)).T]), axis=0)

            if return_iteration_indicator:
                current_pairs = np.stack([relevant_bl_tumors[current_pairs_inds[:, 0]], relevant_fu_tumors[current_pairs_inds[:, 1]], np.repeat(dilate + 1, current_pairs_inds.shape[0])]).T
            else:
                current_pairs = np.stack([relevant_bl_tumors[current_pairs_inds[:, 0]], relevant_fu_tumors[current_pairs_inds[:, 1]]]).T

            pairs = np.concatenate([pairs, current_pairs])

        if dilate == (max_dilate_param - 1):
            break

        # dilation without overlap, and considering resolution
        working_baseline_moved_labeled, distance_cache_bl = expand_labels(baseline_moved_labeled, distance=dilate+1,
                                                                          voxelspacing=voxelspacing,
                                                                          distance_cache=distance_cache_bl,
                                                                          return_distance_cache=True)
        working_followup_labeled, distance_cache_fu = expand_labels(followup_labeled, distance=dilate+1,
                                                                    voxelspacing=voxelspacing,
                                                                    distance_cache=distance_cache_fu,
                                                                    return_distance_cache=True)

        # zero the BL tumor and the FU tumor in the matches
        working_baseline_moved_labeled[np.isin(working_baseline_moved_labeled, pairs[:, 0])] = 0
        working_followup_labeled[np.isin(working_followup_labeled, pairs[:, 1])] = 0

        if np.all(working_baseline_moved_labeled == 0) or np.all(working_followup_labeled == 0):
            break

    if return_iteration_indicator:
        return [(p[2], (p[0], p[1])) for p in pairs]
    return [tuple(p) for p in pairs]


def match_2_cases_v3(baseline_moved_labeled, followup_labeled, voxelspacing=None, max_dilate_param=5,
                     return_iteration_indicator=False):
    """
    • This version removes the tumors only at the end of the iterations.
    • This version works with minimum python 'for' iterations and mostly with numpy optimizations.
    • This version's match criteria is only the ratio of intersection between tumors, and it doesn't take a maximum
        intersection as a match.
    """
    working_baseline_moved_labeled = baseline_moved_labeled
    working_followup_labeled = followup_labeled

    distance_cache_bl, distance_cache_fu = None, None

    pairs = np.array([]).reshape([0, 3 if return_iteration_indicator else 2])
    # Hyper-parameter for sensitivity of the matching (5 means that 10 pixels between the scans will be same)
    for dilate in range(max_dilate_param):

        # find pairs of intersection of tumors
        pairs_of_intersection = np.stack([working_baseline_moved_labeled, working_followup_labeled]).astype(np.int16)
        pairs_of_intersection, overlap_vol  = np.unique(pairs_of_intersection[:, np.all(pairs_of_intersection != 0, axis=0)].T, axis=0, return_counts=True)

        if pairs_of_intersection.size > 0:

            relevant_bl_tumors, relevant_bl_tumors_vol = np.unique(working_baseline_moved_labeled[np.isin(working_baseline_moved_labeled, pairs_of_intersection[:, 0])], return_counts=True)
            relevant_fu_tumors, relevant_fu_tumors_vol = np.unique(working_followup_labeled[np.isin(working_followup_labeled, pairs_of_intersection[:, 1])], return_counts=True)

            # intersection_matrix_overlap_vol[i,j] = #voxels_in_the_overlap_of_relevant_bl_tumors[i]_and_relevant_fu_tumors[j]
            intersection_matrix_overlap_vol = np.zeros((relevant_bl_tumors.size, relevant_fu_tumors.size))
            intersection_matrix_overlap_vol[np.searchsorted(relevant_bl_tumors, pairs_of_intersection[:, 0]),
                                            np.searchsorted(relevant_fu_tumors, pairs_of_intersection[:, 1])] = overlap_vol

            # intersection_matrix_overlap_with_bl_ratio[i,j] = #voxels_in_the_overlap_of_relevant_bl_tumors[i]_and_relevant_fu_tumors[j] / #voxels_in_relevant_bl_tumors[i]
            intersection_matrix_overlap_with_bl_ratio = intersection_matrix_overlap_vol / relevant_bl_tumors_vol.reshape([-1, 1])

            # intersection_matrix_overlap_with_fu_ratio[i,j] = #voxels_in_the_overlap_of_relevant_bl_tumors[i]_and_relevant_fu_tumors[j] / #voxels_in_relevant_fu_tumors[j]
            intersection_matrix_overlap_with_fu_ratio = intersection_matrix_overlap_vol / relevant_fu_tumors_vol.reshape([1, -1])

            # take as a match any maximum intersection with bl tumors
            # current_pairs_inds = np.stack([np.arange(relevant_bl_tumors.size), np.argmax(intersection_matrix_overlap_vol, axis=1)]).T

            # take as a match any maximum intersection with fu tumors
            # current_pairs_inds = np.unique(np.concatenate([current_pairs_inds, np.stack([np.argmax(intersection_matrix_overlap_vol, axis=0), np.arange(relevant_fu_tumors.size)]).T]), axis=0)

            current_pairs_inds = np.array([], dtype=np.int16).reshape([0, 2])
            # take as a match any intersection with a overlap ratio with either bl of fu above 10 percent
            current_pairs_inds = np.unique(np.concatenate([current_pairs_inds, np.stack(np.where(intersection_matrix_overlap_with_bl_ratio > 0.1)).T]), axis=0)
            current_pairs_inds = np.unique(np.concatenate([current_pairs_inds, np.stack(np.where(intersection_matrix_overlap_with_fu_ratio > 0.1)).T]), axis=0)

            if return_iteration_indicator:
                current_pairs = np.stack([relevant_bl_tumors[current_pairs_inds[:, 0]], relevant_fu_tumors[current_pairs_inds[:, 1]], np.repeat(dilate + 1, current_pairs_inds.shape[0])]).T
            else:
                current_pairs = np.stack([relevant_bl_tumors[current_pairs_inds[:, 0]], relevant_fu_tumors[current_pairs_inds[:, 1]]]).T

            pairs = np.concatenate([pairs, current_pairs])

        if dilate == (max_dilate_param - 1):
            break

        # dilation without overlap, and considering resolution
        working_baseline_moved_labeled, distance_cache_bl = expand_labels(baseline_moved_labeled, distance=dilate + 1,
                                                                          voxelspacing=voxelspacing,
                                                                          distance_cache=distance_cache_bl,
                                                                          return_distance_cache=True)
        working_followup_labeled, distance_cache_fu = expand_labels(followup_labeled, distance=dilate+1,
                                                                    voxelspacing=voxelspacing,
                                                                    distance_cache=distance_cache_fu,
                                                                    return_distance_cache=True)

        # zero the BL tumor and the FU tumor in the matches
        working_baseline_moved_labeled[np.isin(working_baseline_moved_labeled, pairs[:, 0])] = 0
        working_followup_labeled[np.isin(working_followup_labeled, pairs[:, 1])] = 0

        if np.all(working_baseline_moved_labeled == 0) or np.all(working_followup_labeled == 0):
            break

    if return_iteration_indicator:
        return [(p[2], (p[0], p[1])) for p in pairs]
    return [tuple(p) for p in pairs]


def match_2_cases_v4(baseline_moved_labeled, followup_labeled, voxelspacing=None, max_dilate_param=5,
                     return_iteration_indicator=False):
    """
    • This version removes the tumors only at the end of the iterations.
    • This version works with minimum python 'for' iterations and mostly with numpy optimizations.
    • This version's match criteria is only the ratio of intersection between tumors, and it doesn't take a maximum
        intersection as a match.
    • This version expands the tumors after each iteration relative to the tumors size.
    """

    def dists_to_expand(diameters: np.ndarray, i_1: float = 6, i_10: float = 3) -> np.ndarray:
        """
        The distance to expand is a function of the diameter.

        i_1 is the number of mm to expand for a tumor with a diameter of 1
        i_10 is the number of mm to expand for a tumor with a diameter of 10
        See here the visual graph: https://www.desmos.com/calculator/lznykxikim
        """

        b = (10 * i_10 - i_1) / 9
        c = ((i_1**10)/i_10)**(1/9)

        if isinstance(diameters, (int, float)):
            diameters = np.array([diameters])

        diameters = np.clip(diameters, 1, 20)
        dists = np.empty(diameters.shape, np.float32)

        # for tumors with diameter less or equal to 10, the distance to expand
        # as a function of the diameter is (i_1 - b)/x + b, where x is the diameter, and b and i_1 are defined above.
        dists[diameters <= 10] = ((i_1 - b) / diameters[diameters <= 10]) + b

        # for tumors with diameter greater than 10, the distance to expand
        # as a function of the diameter is c * (i_1/c)^x, where x is the diameter, and c and i_1 are defined above.
        dists[diameters > 10] = c * ((i_1 / c) ** diameters[diameters > 10])

        return dists

    def dists_to_expand_v2(diameters: np.ndarray, i: float = 5, j: float = 3, k: float = 0.05) -> np.ndarray:
        """
        The distance (number of mm) to expand is a function of the diameter. The following functions are in use:

        func_1: a/x + b, where x is the diameter.
        func_2: c/e^(dx), where x is the diameter.

        For a diameter between 1 and 10, func_1 is used, and for a diameter between 10 and 20, func_2 is used.
        All the diameters is clipped to range [1,20]

        The parameters, {a,b} for func_1 and {c,d} for func_2 is decided due to the given arguments.

        i is the number of mm to expand for a tumor with a diameter of 1
        j is the number of mm to expand for a tumor with a diameter of 10
        k is the number of mm to expand for a tumor with a diameter of 20
        See here the visual graph: https://www.desmos.com/calculator/dvokawlytl
        """

        if isinstance(diameters, (int, float)):
            diameters = np.array([diameters])

        diameters = np.clip(diameters, 1, 20)
        dists = np.empty(diameters.shape, np.float32)

        # for tumors with diameter less or equal to 10, the distance to expand
        # as a function of the diameter is 10(i-j)/9x + (10j-i)/9, where x is the diameter, and i and j are defined above.
        dists[diameters <= 10] = 10 * (i - j) / (9 * diameters[diameters <= 10]) + (10 * j - i) / 9

        # for tumors with diameter greater than 10, the distance to expand
        # as a function of the diameter is j * (j/k)^((10-x)/10), where x is the diameter, and j and k are defined above.
        dists[diameters > 10] = j * ((j / k) ** ((10 - diameters[diameters > 10]) / 10))

        return dists

    def extract_diameters(tumors_labeled_case: np.ndarray) -> np.ndarray:
        tumors_labels, tumors_vol = np.unique(tumors_labeled_case, return_counts=True)
        tumors_vol = tumors_vol[tumors_labels > 0] * np.asarray(voxelspacing).prod()
        return (6 * tumors_vol / np.pi) ** (1/3)

    bl_dists_to_expand = dists_to_expand_v2(extract_diameters(baseline_moved_labeled))
    bl_max_relevant_distances = max_dilate_param * bl_dists_to_expand

    fu_dists_to_expand = dists_to_expand_v2(extract_diameters(followup_labeled))
    fu_max_relevant_distances = max_dilate_param * fu_dists_to_expand

    # working_baseline_moved_labeled = baseline_moved_labeled
    # working_followup_labeled = followup_labeled

    distance_cache_bl, distance_cache_fu = None, None

    pairs = np.array([]).reshape([0, 3 if return_iteration_indicator else 2])
    # Hyper-parameter for sensitivity of the matching (5 means that 10 pixels between the scans will be same)
    for dilate in range(max_dilate_param):

        # dilation without overlap, and considering resolution
        working_baseline_moved_labeled, distance_cache_bl = expand_per_label(baseline_moved_labeled,
                                                                             dists_to_expand=(dilate + 1) * bl_dists_to_expand,
                                                                             max_relevant_distances=bl_max_relevant_distances,
                                                                             voxelspacing=voxelspacing,
                                                                             distance_cache=distance_cache_bl,
                                                                             return_distance_cache=True)
        working_followup_labeled, distance_cache_fu = expand_per_label(followup_labeled,
                                                                       dists_to_expand=(dilate + 1) * fu_dists_to_expand,
                                                                       max_relevant_distances=fu_max_relevant_distances,
                                                                       voxelspacing=voxelspacing,
                                                                       distance_cache=distance_cache_fu,
                                                                       return_distance_cache=True)

        # zero the BL tumor and the FU tumor in the matches
        working_baseline_moved_labeled[np.isin(working_baseline_moved_labeled, pairs[:, 0])] = 0
        working_followup_labeled[np.isin(working_followup_labeled, pairs[:, 1])] = 0

        # find pairs of intersection of tumors
        pairs_of_intersection = np.stack([working_baseline_moved_labeled, working_followup_labeled]).astype(np.int16)
        pairs_of_intersection, overlap_vol = np.unique(pairs_of_intersection[:, np.all(pairs_of_intersection != 0, axis=0)].T, axis=0, return_counts=True)

        if pairs_of_intersection.size > 0:

            relevant_bl_tumors, relevant_bl_tumors_vol = np.unique(working_baseline_moved_labeled[np.isin(working_baseline_moved_labeled, pairs_of_intersection[:, 0])], return_counts=True)
            relevant_fu_tumors, relevant_fu_tumors_vol = np.unique(working_followup_labeled[np.isin(working_followup_labeled, pairs_of_intersection[:, 1])], return_counts=True)

            # intersection_matrix_overlap_vol[i,j] = #voxels_in_the_overlap_of_relevant_bl_tumors[i]_and_relevant_fu_tumors[j]
            intersection_matrix_overlap_vol = np.zeros((relevant_bl_tumors.size, relevant_fu_tumors.size))
            intersection_matrix_overlap_vol[np.searchsorted(relevant_bl_tumors, pairs_of_intersection[:, 0]),
                                            np.searchsorted(relevant_fu_tumors, pairs_of_intersection[:, 1])] = overlap_vol

            # intersection_matrix_overlap_with_bl_ratio[i,j] = #voxels_in_the_overlap_of_relevant_bl_tumors[i]_and_relevant_fu_tumors[j] / #voxels_in_relevant_bl_tumors[i]
            intersection_matrix_overlap_with_bl_ratio = intersection_matrix_overlap_vol / relevant_bl_tumors_vol.reshape([-1, 1])

            # intersection_matrix_overlap_with_fu_ratio[i,j] = #voxels_in_the_overlap_of_relevant_bl_tumors[i]_and_relevant_fu_tumors[j] / #voxels_in_relevant_fu_tumors[j]
            intersection_matrix_overlap_with_fu_ratio = intersection_matrix_overlap_vol / relevant_fu_tumors_vol.reshape([1, -1])

            # take as a match any maximum intersection with bl tumors
            # current_pairs_inds = np.stack([np.arange(relevant_bl_tumors.size), np.argmax(intersection_matrix_overlap_vol, axis=1)]).T

            # take as a match any maximum intersection with fu tumors
            # current_pairs_inds = np.unique(np.concatenate([current_pairs_inds, np.stack([np.argmax(intersection_matrix_overlap_vol, axis=0), np.arange(relevant_fu_tumors.size)]).T]), axis=0)

            current_pairs_inds = np.array([], dtype=np.int16).reshape([0, 2])
            # take as a match any intersection with a overlap ratio with either bl of fu above 10 percent
            current_pairs_inds = np.unique(np.concatenate([current_pairs_inds, np.stack(np.where(intersection_matrix_overlap_with_bl_ratio > 0.1)).T]), axis=0)
            current_pairs_inds = np.unique(np.concatenate([current_pairs_inds, np.stack(np.where(intersection_matrix_overlap_with_fu_ratio > 0.1)).T]), axis=0)

            if return_iteration_indicator:
                current_pairs = np.stack([relevant_bl_tumors[current_pairs_inds[:, 0]], relevant_fu_tumors[current_pairs_inds[:, 1]], np.repeat(dilate + 1, current_pairs_inds.shape[0])]).T
            else:
                current_pairs = np.stack([relevant_bl_tumors[current_pairs_inds[:, 0]], relevant_fu_tumors[current_pairs_inds[:, 1]]]).T

            pairs = np.concatenate([pairs, current_pairs])

        if np.all(working_baseline_moved_labeled == 0) or np.all(working_followup_labeled == 0):
            break

    if return_iteration_indicator:
        return [(p[2], (p[0], p[1])) for p in pairs]
    return [tuple(p) for p in pairs]


def match_2_cases_v5(baseline_moved_labeled, followup_labeled, voxelspacing=None, max_dilate_param=5,
                     return_iteration_indicator=False):
    """
    • This version removes the tumors only at the end of the iterations.
    • This version works with minimum python 'for' iterations and mostly with numpy optimizations.
    • This version's match criteria is only the ratio of intersection between tumors, and it doesn't take a maximum
        intersection as a match.
    • This version dilates the images ones in the beginning.
    """

    if np.all(baseline_moved_labeled == 0) or np.all(followup_labeled == 0):
        return []

    distance_cache_bl, distance_cache_fu = None, None

    pairs = np.array([]).reshape([0, 3 if return_iteration_indicator else 2])
    # Hyper-parameter for sensitivity of the matching (5 means that 10 pixels between the scans will be same)
    for dilate in range(max_dilate_param):

        # dilation without overlap, and considering resolution
        working_baseline_moved_labeled, distance_cache_bl = expand_labels(baseline_moved_labeled, distance=dilate + 1,
                                                                          voxelspacing=voxelspacing,
                                                                          distance_cache=distance_cache_bl,
                                                                          return_distance_cache=True)
        working_followup_labeled, distance_cache_fu = expand_labels(followup_labeled, distance=dilate + 1,
                                                                    voxelspacing=voxelspacing,
                                                                    distance_cache=distance_cache_fu,
                                                                    return_distance_cache=True)

        if dilate > 0:
            # zero the BL tumor and the FU tumor in the matches
            working_baseline_moved_labeled[np.isin(working_baseline_moved_labeled, pairs[:, 0])] = 0
            working_followup_labeled[np.isin(working_followup_labeled, pairs[:, 1])] = 0

            if np.all(working_baseline_moved_labeled == 0) or np.all(working_followup_labeled == 0):
                break

        # find pairs of intersection of tumors
        pairs_of_intersection = np.stack([working_baseline_moved_labeled, working_followup_labeled]).astype(np.int16)
        pairs_of_intersection, overlap_vol = np.unique(pairs_of_intersection[:, np.all(pairs_of_intersection != 0, axis=0)].T, axis=0, return_counts=True)

        if pairs_of_intersection.size > 0:

            relevant_bl_tumors, relevant_bl_tumors_vol = np.unique(working_baseline_moved_labeled[np.isin(working_baseline_moved_labeled, pairs_of_intersection[:, 0])], return_counts=True)
            relevant_fu_tumors, relevant_fu_tumors_vol = np.unique(working_followup_labeled[np.isin(working_followup_labeled, pairs_of_intersection[:, 1])], return_counts=True)

            # intersection_matrix_overlap_vol[i,j] = #voxels_in_the_overlap_of_relevant_bl_tumors[i]_and_relevant_fu_tumors[j]
            intersection_matrix_overlap_vol = np.zeros((relevant_bl_tumors.size, relevant_fu_tumors.size))
            intersection_matrix_overlap_vol[np.searchsorted(relevant_bl_tumors, pairs_of_intersection[:, 0]),
                                            np.searchsorted(relevant_fu_tumors, pairs_of_intersection[:, 1])] = overlap_vol

            # intersection_matrix_overlap_with_bl_ratio[i,j] = #voxels_in_the_overlap_of_relevant_bl_tumors[i]_and_relevant_fu_tumors[j] / #voxels_in_relevant_bl_tumors[i]
            intersection_matrix_overlap_with_bl_ratio = intersection_matrix_overlap_vol / relevant_bl_tumors_vol.reshape([-1, 1])

            # intersection_matrix_overlap_with_fu_ratio[i,j] = #voxels_in_the_overlap_of_relevant_bl_tumors[i]_and_relevant_fu_tumors[j] / #voxels_in_relevant_fu_tumors[j]
            intersection_matrix_overlap_with_fu_ratio = intersection_matrix_overlap_vol / relevant_fu_tumors_vol.reshape([1, -1])

            current_pairs_inds = np.array([], dtype=np.int16).reshape([0, 2])
            # take as a match any intersection with a overlap ratio with either bl of fu above 10 percent
            current_pairs_inds = np.unique(np.concatenate([current_pairs_inds, np.stack(np.where(intersection_matrix_overlap_with_bl_ratio > 0.1)).T]), axis=0)
            current_pairs_inds = np.unique(np.concatenate([current_pairs_inds, np.stack(np.where(intersection_matrix_overlap_with_fu_ratio > 0.1)).T]), axis=0)

            if return_iteration_indicator:
                current_pairs = np.stack([relevant_bl_tumors[current_pairs_inds[:, 0]], relevant_fu_tumors[current_pairs_inds[:, 1]], np.repeat(dilate + 1, current_pairs_inds.shape[0])]).T
            else:
                current_pairs = np.stack([relevant_bl_tumors[current_pairs_inds[:, 0]], relevant_fu_tumors[current_pairs_inds[:, 1]]]).T

            pairs = np.concatenate([pairs, current_pairs])

    if return_iteration_indicator:
        return [(p[2], (p[0], p[1])) for p in pairs]
    return [tuple(p) for p in pairs]


def pre_process_segmentation(seg, remove_small_obs=True):

    # fill holes over 2D slices
    res = binary_fill_holes(seg, np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]).reshape([3, 3, 1]).astype(np.bool_))

    # removing small objects
    if remove_small_obs:
        res = remove_small_objects(res, min_size=20)

    return res.astype(seg.dtype)


def replace_in_file_name(file_name, old_part, new_part, dir_file_name=False, dst_file_exist=True):
    if old_part not in file_name:
        raise Exception(f'The following file/dir doesn\'t contain the part "{old_part}": {file_name}')
    new_file_name = file_name.replace(old_part, new_part)
    check_if_exist = os.path.isdir if dir_file_name else os.path.isfile
    if dst_file_exist and (not check_if_exist(new_file_name)):
        raise Exception(f'It looks like the following file/dir doesn\'t exist: {new_file_name}')
    return new_file_name


def write_to_excel(df, writer, columns_order, column_name_as_index, sheet_name='Sheet1',
                   f1_scores: Optional[Dict[str, Tuple[str, str]]] = None):
    df = df.set_index(column_name_as_index)
    df = pd.concat([df, df.agg(['mean', 'std', 'min', 'max', 'sum'])], ignore_index=False)
    workbook = writer.book
    # cell_format = workbook.add_format()
    cell_format = workbook.add_format({'num_format': '#,##0.00'})
    cell_format.set_font_size(16)

    columns_order = [c for c in columns_order if c != column_name_as_index]
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

    n = df.shape[0] - 5
    for col in np.arange(len(columns_order)) + 1:
        for i, measure in enumerate(['AVERAGE', 'STDEV', 'MIN', 'MAX', 'SUM'], start=1):
            col_name = xl_col_to_name(col)
            worksheet.write(f'{col_name}{n + i + 1}', f'{{={measure}({col_name}2:{col_name}{n + 1})}}')

    if f1_scores is not None:
        for col_name in f1_scores:
            f1_col_name = xl_col_to_name(columns_order.index(col_name) + 1)
            p_col_name = xl_col_to_name(columns_order.index(f1_scores[col_name][0]) + 1)
            r_col_name = xl_col_to_name(columns_order.index(f1_scores[col_name][1]) + 1)
            worksheet.write(f'{f1_col_name}{n + 2}', f'{{=HARMEAN({p_col_name}{n + 2}:{r_col_name}{n + 2})}}')
            for i in range(1, 5):
                worksheet.write(f'{f1_col_name}{n + 2 + i}', " ")

    worksheet.conditional_format(f'$B$2:${xl_col_to_name(len(columns_order))}$' + str(len(df.axes[0]) - 4), {'type': 'formula',
                                                                         'criteria': '=B2=B$' + str(len(df.axes[0])),
                                                                         'format': max_format})

    worksheet.conditional_format(f'$B$2:${xl_col_to_name(len(columns_order))}$' + str(len(df.axes[0]) - 4), {'type': 'formula',
                                                                         'criteria': '=B2=B$' + str(
                                                                             len(df.axes[0]) - 1),
                                                                         'format': min_format})



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


def bbox2_3D(img):
    x = np.any(img, axis=(1, 2))
    y = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))

    xmin, xmax = np.where(x)[0][[0, -1]]
    ymin, ymax = np.where(y)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    return xmin, xmax, ymin, ymax, zmin, zmax


def bbox2_2D(img):
    x = np.any(img, axis=1)
    y = np.any(img, axis=0)

    xmin, xmax = np.where(x)[0][[0, -1]]
    ymin, ymax = np.where(y)[0][[0, -1]]

    return xmin, xmax, ymin, ymax


def get_liver_segments(liver_case: np.ndarray) -> np.ndarray:
    xmin, xmax, ymin, ymax, _, _ = bbox2_3D(liver_case)
    res = np.zeros_like(liver_case)
    res[(xmin + xmax) // 2:xmax, (ymin + ymax) // 2:ymax, :] = 1
    res[xmin:(xmin + xmax) // 2, (ymin + ymax) // 2:ymax, :] = 2
    res[:, ymin:(ymin + ymax) // 2, :] = 3
    res *= liver_case
    return res


def get_center_of_mass(img: np.ndarray) -> Tuple[int, int, int]:
    xmin, xmax, ymin, ymax, zmin, zmax = bbox2_3D(img)
    return (xmin + xmax) // 2, (ymin + ymax) // 2, (zmin + zmax) // 2


def print_full_df(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')


def approximate_sphere(relevant_points_in_real_space: np.ndarray, relevant_points_in_voxel_space: np.ndarray,
                       center_of_mass_in_voxel_space: Tuple[int, int, int], approximate_radius_in_real_space: float,
                       affine_matrix: np.ndarray):
    center_of_mass_in_real_space = affines.apply_affine(affine_matrix, center_of_mass_in_voxel_space)
    final_points_in_voxel_space = relevant_points_in_voxel_space[((relevant_points_in_real_space - center_of_mass_in_real_space)**2).sum(axis=1) <= approximate_radius_in_real_space**2]
    sphere = np.zeros(relevant_points_in_voxel_space[-1] + 1)
    sphere[final_points_in_voxel_space[:, 0], final_points_in_voxel_space[:, 1], final_points_in_voxel_space[:, 2]] = 1
    return sphere


def is_a_scan(case: np.ndarray) -> bool:
    if np.unique(case).size <= 2:
        return False
    return True


def is_a_mask(case: np.ndarray) -> bool:
    if np.any((case != 0) & (case != 1)):
        return False
    return True


def is_a_labeled_mask(case: np.ndarray, relevant_labels: Collection) -> bool:
    if np.all(np.isin(case, relevant_labels)):
        return True
    return False


def symlink_for_inner_files_in_a_dir(src: str, dst: str, map_file_basename: Callable = None,
                                     filter_file_basename: Callable = None):
    """makes a symbolic link of files in a directory"""
    if not os.path.isdir(src):
        raise Exception("symlink_for_inner_files works only for directories")
    if src.endswith('/'):
        src = src[:-1]
    if dst.endswith('/'):
        dst = dst[:-1]
    os.makedirs(dst, exist_ok=True)
    map_file_basename = (lambda x: x) if map_file_basename is None else map_file_basename
    filter_file_basename = (lambda x: True) if filter_file_basename is None else filter_file_basename
    for file in glob(f'{src}/*'):
        file_basename = os.path.basename(file)
        if os.path.isdir(file):
            symlink_for_inner_files_in_a_dir(file, f'{dst}/{file_basename}')
        else:
            if filter_file_basename(file_basename):
                os.symlink(file, f'{dst}/{map_file_basename(file_basename)}')


def scans_sort_key(name, full_path_is_given=False):
    if full_path_is_given:
        name = os.path.basename(name)
    split = name.split('_')
    return '_'.join(c for c in split if not c.isdigit()), int(split[-1]), int(split[-2]), int(split[-3])


def pairs_sort_key(name, full_path_is_given=False):
    if full_path_is_given:
        name = os.path.basename(name)
    name = name.replace('BL_', '')
    bl_name, fu_name = name.split('_FU_')
    return (*scans_sort_key(bl_name), *scans_sort_key(fu_name))


def sort_dataframe_by_key(dataframe: pd.DataFrame, column: str, key: Callable) -> pd.DataFrame:
    """ Sort a dataframe from a column using the key """
    sort_ixs = sorted(np.arange(len(dataframe)), key=lambda i: key(dataframe.iloc[i][column]))
    return pd.DataFrame(columns=list(dataframe), data=dataframe.iloc[sort_ixs].values)


def get_minimum_distance_between_CCs(mask: np.ndarray, voxel_to_real_space_trans: Optional[np.ndarray] = None,
                                     max_points_per_CC: Optional[int] = None, seed: Optional[int] = None) -> float:
    """
    Get the minimum distance between every 2 connected components in a binary image
    """

    rand = np.random.RandomState(seed)

    def dist_between_PCs_squared(pc1, pc2):
        return np.min(((np.expand_dims(pc1, -1) - np.expand_dims(pc2.T, 0)) ** 2).sum(axis=1))

    def choose_n_random_points(pc: np.ndarray, n) -> np.ndarray:
        perm = rand.permutation(pc.shape[0])
        idx = perm[:n]
        return pc[idx, :]

    def filter_and_transfer_points(pc: np.ndarray):
        if (max_points_per_CC is not None) and (pc.shape[0] > max_points_per_CC):
            pc = choose_n_random_points(pc, max_points_per_CC)
        if voxel_to_real_space_trans is not None:
            pc = affines.apply_affine(voxel_to_real_space_trans, pc)
        return pc

    mask = get_connected_components(mask > 0)

    list_of_PCs = [filter_and_transfer_points(r.coords) for r in regionprops(mask)]

    n_CCs = len(list_of_PCs)

    if n_CCs >= 2:
        return np.sqrt(np.min([dist_between_PCs_squared(list_of_PCs[i], list_of_PCs[j])
                               for i in range(n_CCs) for j in range(i + 1, n_CCs)]))

    return np.inf


def get_tumors_intersections(gt: np.ndarray, pred: np.ndarray, unique_intersections_only: bool = False) -> Dict[int, List[int]]:

    """
    Get intersections of tumors between GT and PRED

    :param gt: GT tumors case
    :param pred: PRED tumors case
    :param unique_intersections_only: If considering only unique intersections

    :return: a dict containing for each relevant GT tumor (key) a list with the relevant intersections (value)
    """

    # extract intersection pairs of tumors
    pairs = np.hstack([gt.reshape([-1, 1]), pred.reshape([-1, 1])])
    pairs = np.unique(pairs[~np.any(pairs == 0, axis=1)], axis=0)

    if unique_intersections_only:

        # filter out unique connections
        unique_gt = np.stack(np.unique(pairs[:, 0], return_counts=True)).T
        unique_gt = unique_gt[unique_gt[:, 1] == 1][:, 0]
        unique_pred = np.stack(np.unique(pairs[:, 1], return_counts=True)).T
        unique_pred = unique_pred[unique_pred[:, 1] == 1][:, 0]
        pairs = pairs[np.isin(pairs[:, 0], unique_gt)]
        pairs = pairs[np.isin(pairs[:, 1], unique_pred)]

    intersections = []
    previous_gt = None
    for k, gt in enumerate(pairs[:, 0]):
        if previous_gt is not None and (gt == previous_gt[0]):
            previous_gt = (gt, previous_gt[1] + [int(pairs[k, 1])])
            intersections[-1] = previous_gt
        else:
            previous_gt = (gt, [int(pairs[k, 1])])
            intersections.append(previous_gt)

    return dict(intersections)


if __name__ == '__main__':

    x = np.array([[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 0, 0], [0, 1 ,0 ,0]])

    print(min_distance(x, x))
    exit()

    # todo creating files for Kobi
    excel_file_path = '/cs/casmip/rochman/Errors_Characterization/measures_results/tumors_measures.xlsx'
    df = pd.read_excel(excel_file_path)
    df.rename(columns={df.columns[0]: 'name'}, inplace=True)
    df = df.iloc[:-5, :]
    scans = [n[1].split('_-_')[0] for n in df['name'].iteritems()]
    df['scan'] = scans
    df = df[['name', 'scan', *df.columns[1:-1]]]

    total_min_number_of_tumors = 25
    total_max_number_of_tumors = 35

    sections_and_min_max_tumors_per_section = [((0, 6), (8, 12)),
                                               ((6, 11), (8, 12)),
                                               ((11, 100), (8, 12))]

    n_scans = 5
    min_num_of_patients = n_scans
    min_num_of_tumors_per_scan = 3
    max_num_of_tumors_per_scan = 10

    lst_scans = []
    for s in scans:
        if s not in lst_scans:
            lst_scans += [s]

    rand = np.random.RandomState(42)
    for _ in range(1500):
        relevant_scans = rand.choice(lst_scans, n_scans)

        s_s, counts = np.unique(scans, return_counts=True)
        flag = True
        for s, c in zip(s_s, counts):
            if s in relevant_scans and not (min_num_of_tumors_per_scan <= c <= max_num_of_tumors_per_scan):
                flag = False
                break
        if not flag:
            continue

        n_patients = len(set(['_'.join([c for c in s.split('_') if not c.isdigit()]) for s in relevant_scans]))

        if n_patients < min_num_of_patients:
            continue

        relevant_df = df[df['scan'].isin(relevant_scans)]

        n_tumors = relevant_df.shape[0]

        if not (total_min_number_of_tumors <= n_tumors <= total_max_number_of_tumors):
            continue

        flag = True
        res = []
        for ((lower_limit, higher_limit), (min_number_of_tumors_per_section, max_number_of_tumors_per_section)) in sections_and_min_max_tumors_per_section:
            n_tumors_in_current_section = relevant_df[(relevant_df['tumor_diameter (mm)'] > lower_limit) & (relevant_df['tumor_diameter (mm)'] <= higher_limit)].shape[0]
            if not (min_number_of_tumors_per_section <= n_tumors_in_current_section <= max_number_of_tumors_per_section):
                flag = False
                break
            res += [n_tumors_in_current_section]

        if flag:
            print(relevant_scans)
            for s, c in zip(s_s, counts):
                if s in relevant_scans:
                    print(f'{s}: {c} tumors')
            print(f'n_patients={n_patients}, n_tumors={n_tumors}, n_small_tumors={res[0]}, n_medium_tumors={res[1]}, n_large_tumors={res[2]}')
            print(f'small_tumors_range=({sections_and_min_max_tumors_per_section[0][0][0]}, {sections_and_min_max_tumors_per_section[0][0][1]}), '
                  f'medium_tumors_range=({sections_and_min_max_tumors_per_section[1][0][0]}, {sections_and_min_max_tumors_per_section[1][0][1]}) '
                  f'large_tumors_range=({sections_and_min_max_tumors_per_section[2][0][0]}, {sections_and_min_max_tumors_per_section[2][0][1]})')
            print('--------------------------------------------------------------------------------------------------------------------')

    exit(0)


    # tumor_case, file = load_nifti_data('/cs/casmip/rochman/Errors_Characterization/liver_border_with_tumors/A_K_25_11_2015/tumors.nii.gz')
    # p = (361, 178, 73)
    #
    # tumor_case_unique_CC = get_connected_components(tumor_case)
    # res = np.zeros_like(tumor_case)
    # res[tumor_case_unique_CC == tumor_case_unique_CC[p]] = 1

    # ex1 = np.array([[0, 0, 0, 0, 0],
    #                 [0, 0, 0, 0, 0],
    #                 [0, 0, 1, 0, 0],
    #                 [0, 0, 0, 0, 0],
    #                 [0, 0, 0, 0, 0]])


    # bl_tumors_file = '/cs/casmip/rochman/Errors_Characterization/matching/BL_H_G_06_10_2019_FU_H_G_24_11_2019/BL_Scan_Tumors_unique_22_CC.nii.gz'
    # fu_tumors_file = '/cs/casmip/rochman/Errors_Characterization/matching/BL_H_G_06_10_2019_FU_H_G_24_11_2019/FU_Scan_Tumors_unique_23_CC.nii.gz'
    # bl_liver_file = '/cs/casmip/rochman/Errors_Characterization/matching/BL_H_G_06_10_2019_FU_H_G_24_11_2019/BL_Scan_Liver.nii.gz'
    # fu_liver_file = '/cs/casmip/rochman/Errors_Characterization/matching/BL_H_G_06_10_2019_FU_H_G_24_11_2019/FU_Scan_Liver.nii.gz'
    #
    # bl_tumors_case, file = load_nifti_data(bl_tumors_file)
    #
    # # todo you can delete this section of code
    # # ---------------------------------
    # # shrinked = shrink_labels(bl_tumors_case, distance=2.5, voxelspacing=file.header.get_zooms())
    # # save(Nifti1Image(shrinked, file.affine), '/cs/casmip/rochman/Errors_Characterization/matching/BL_H_G_06_10_2019_FU_H_G_24_11_2019/test_you_can_delete.nii.gz')
    # # exit(0)
    # # ---------------------------------
    # fu_tumors_case, _ = load_nifti_data(fu_tumors_file)
    # bl_liver_case, _ = load_nifti_data(bl_liver_file)
    # fu_liver_case, _ = load_nifti_data(fu_liver_file)
    #
    # expand_bl_tumors_case = expand_labels(bl_tumors_case, distance=13, voxelspacing=file.header.get_zooms())
    # expand_fu_tumors_case = expand_labels(fu_tumors_case, distance=13, voxelspacing=file.header.get_zooms())
    #
    # bl_liver_case = getLargestCC(bl_liver_case)
    # bl_liver_case = pre_process_segmentation(bl_liver_case, remove_small_obs=False)
    #
    # fu_liver_case = getLargestCC(fu_liver_case)
    # fu_liver_case = pre_process_segmentation(fu_liver_case, remove_small_obs=False)
    #
    # from skimage.morphology import binary_dilation, ball
    #
    # # selem = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]).reshape([3, 3, 1]).astype(np.bool_)
    # bl_liver_border = np.logical_and(binary_dilation(bl_liver_case, ball(1)), np.logical_not(bl_liver_case)).astype(bl_liver_case.dtype)
    # fu_liver_border = np.logical_and(binary_dilation(fu_liver_case, ball(1)), np.logical_not(fu_liver_case)).astype(fu_liver_case.dtype)
    #
    # # save(Nifti1Image(bl_liver_border, file.affine), os.path.dirname(bl_liver_file) + '/bl_liver_border_you_can_delete.nii.gz')
    # # save(Nifti1Image(fu_liver_border, file.affine), os.path.dirname(fu_liver_file) + '/fu_liver_border_you_can_delete.nii.gz')
    # #
    # # exit(0)
    #
    # pairs = np.hstack([expand_bl_tumors_case.reshape([-1, 1]), expand_fu_tumors_case.reshape([-1, 1])])
    # pairs = np.unique(pairs[~np.any(pairs == 0, axis=1)], axis=0)
    #
    # def get_distance_from_border_for_each_tumor(tumors, border, voxelspacing):
    #     distances, nearest_label_coords = distance_transform_edt(
    #         border == 0, return_indices=True, sampling=voxelspacing
    #     )
    #     unique_tumors = np.unique(tumors)
    #     unique_tumors = unique_tumors[unique_tumors > 0]
    #     res = {}
    #     for t in unique_tumors:
    #         current_t = (tumors == t)
    #         max_dist_coords = np.unravel_index(np.argmax(np.where(current_t, distances, -np.inf)), tumors.shape)
    #         min_dist_coords = np.unravel_index(np.argmin(np.where(current_t, distances, np.inf)), tumors.shape)
    #
    #         max_dist_nearest_label_coords = np.array([nearest_label_coords[0][max_dist_coords],
    #                                                  nearest_label_coords[1][max_dist_coords],
    #                                                  nearest_label_coords[2][max_dist_coords]])
    #
    #         min_dist_nearest_label_coords = np.array([nearest_label_coords[0][min_dist_coords],
    #                                                  nearest_label_coords[1][min_dist_coords],
    #                                                  nearest_label_coords[2][min_dist_coords]])
    #
    #         volume = np.sum(current_t) * voxelspacing[0] * voxelspacing[1] * voxelspacing[2]
    #
    #         res[t] = (distances[min_dist_coords], min_dist_nearest_label_coords,
    #                   distances[max_dist_coords], max_dist_nearest_label_coords, volume)
    #
    #     return res
    #
    # voxelspacing = file.header.get_zooms()
    # bl_measures = get_distance_from_border_for_each_tumor(np.where(np.isin(bl_tumors_case, pairs[:, 0]), bl_tumors_case, 0), bl_liver_border, voxelspacing)
    # fu_measures = get_distance_from_border_for_each_tumor(np.where(np.isin(fu_tumors_case, pairs[:, 1]), fu_tumors_case, 0), fu_liver_border, voxelspacing)
    #
    # voxelspacing = np.asarray(voxelspacing)
    # pairs_measurs = []
    # for bl_index, fu_index in pairs:
    #     bl_min_dist, bl_min_dist_nearest_label_coords, bl_max_dist, bl_max_dist_nearest_label_coords, bl_vol = bl_measures[bl_index]
    #     fu_min_dist, fu_min_dist_nearest_label_coords, fu_max_dist, fu_max_dist_nearest_label_coords, fu_vol = fu_measures[fu_index]
    #
    #     diff_between_min_dist = abs(bl_min_dist - fu_min_dist)
    #     diff_between_max_dist = abs(bl_max_dist - fu_max_dist)
    #
    #     dist_between_min_dist_nearest_label = np.sqrt(np.sum((voxelspacing ** 2) *
    #                                                          ((bl_min_dist_nearest_label_coords - fu_min_dist_nearest_label_coords)**2)))
    #
    #     dist_between_max_dist_nearest_label = np.sqrt(np.sum((voxelspacing ** 2) *
    #                                                          ((bl_max_dist_nearest_label_coords - fu_max_dist_nearest_label_coords) ** 2)))
    #
    #     vol_diff = np.abs(bl_vol - fu_vol)/(bl_vol + fu_vol)
    #
    #     # pairs_measurs.append(((bl_index, fu_index), diff_between_min_dist, dist_between_min_dist_nearest_label,
    #     #                       diff_between_max_dist, dist_between_max_dist_nearest_label, vol_diff))
    #
    #     pairs_measurs.append((f'({bl_index}, {fu_index})', diff_between_min_dist, dist_between_min_dist_nearest_label,
    #                           diff_between_max_dist, dist_between_max_dist_nearest_label, vol_diff))
    #
    # df = pd.DataFrame(pairs_measurs, columns=['pair', 'diff_between_min_dist (mm)', 'dist_between_min_dist_nearest_label (mm)',
    #                                           'diff_between_max_dist (mm)', 'dist_between_max_dist_nearest_label (mm)',
    #                                           'vol_diff (%)'])
    #
    # df['norm'] = np.linalg.norm(df.to_numpy()[:, 1:], axis=1)
    #
    # print()
    #
    #
    #
    #
    #
    # # bl_liver_case, _ =  load_nifti_data(bl_liver_file)
    # # fu_liver_case, _ =  load_nifti_data(fu_liver_file)
    # #
    # # from skimage.transform import resize
    # #
    # # def f(bl_tumors_case, fu_tumors_case, bl_liver_case, fu_liver_case, voxelspacing):
    # #
    # #     bl_tumors_case = expand_labels(bl_tumors_case, distance=15, voxelspacing=voxelspacing)
    # #     fu_tumors_case = expand_labels(fu_tumors_case, distance=15, voxelspacing=voxelspacing)
    # #
    # #     bl_tumors_case = np.where(bl_tumors_case > 0, bl_tumors_case + 1, 0)
    # #     fu_tumors_case = np.where(fu_tumors_case > 0, fu_tumors_case + 1, 0)
    # #
    # #     bl_merged = np.where(bl_tumors_case > 1, bl_tumors_case, bl_liver_case)
    # #     fu_merged = np.where(fu_tumors_case > 1, fu_tumors_case, fu_liver_case)
    # #
    # #     bl_vec = bl_merged[bl_merged > 0] - 1
    # #     fu_vec = fu_merged[fu_merged > 0] - 1
    # #
    # #     if bl_vec.size > fu_vec.size:
    # #         fu_vec = resize(fu_vec, (bl_vec.size, ), order=0, anti_aliasing=False)
    # #     elif bl_vec.size < fu_vec.size:
    # #         bl_vec = resize(bl_vec, (fu_vec.size, ), order=0, anti_aliasing=False)
    # #
    # #     pairs = np.hstack([bl_vec.reshape([-1, 1]), fu_vec.reshape([-1, 1])])
    # #     filtered_pairs = np.unique(pairs[~np.any(pairs == 0, axis=1)], axis=0)
    # #
    # #     print()
    # #
    # # f(bl_tumors_case, fu_tumors_case, bl_liver_case, fu_liver_case, file.header.get_zooms())
    # #
    # # exit(0)
    #
    # # pred_matches = match_2_cases(bl_tumors_case, fu_tumors_case, file.header.get_zooms(), 60, True)
    # # print()
    #
    # # for dilate in range(16):
    #     # if dilate % 5 == 3:
    #     #     plus = ndimage.generate_binary_structure(3, 1)
    #     #     plus[1, 1, 0] = False
    #     #     plus[1, 1, 2] = False
    #     #     bl_tumors_case = dilation(bl_tumors_case, selem=ndimage.generate_binary_structure(3, 1))
    #     #     fu_tumors_case = dilation(fu_tumors_case, selem=plus)
    #     # else:
    #     #     plus = ndimage.generate_binary_structure(3, 1)
    #     #     plus[1, 1, 0] = False
    #     #     plus[1, 1, 2] = False
    #     #     bl_tumors_case = dilation(bl_tumors_case, selem=plus)
    #     #     fu_tumors_case = dilation(fu_tumors_case, selem=plus)
    #
    #     # bl_tumors_case = expand_labels(bl_tumors_case, distance=1, voxelspacing=file.header.get_zooms())
    #     # fu_tumors_case = expand_labels(fu_tumors_case, distance=1, voxelspacing=file.header.get_zooms())
    #
    # bl_tumors_case = expand_labels(bl_tumors_case, distance=13, voxelspacing=file.header.get_zooms())
    # fu_tumors_case = expand_labels(fu_tumors_case, distance=13, voxelspacing=file.header.get_zooms())
    #
    # save(Nifti1Image(bl_tumors_case, file.affine), os.path.dirname(bl_tumors_file) + '/' + 'bl_dilated_13_you_can_delete.nii.gz')
    # save(Nifti1Image(fu_tumors_case, file.affine), os.path.dirname(fu_tumors_file) + '/' + 'fu_dilated_13_you_can_delete.nii.gz')
    #
    # exit(0)
    #
    # file_name = '/cs/casmip/rochman/Errors_Characterization/matching/BL_A_Z_15_07_2020_FU_A_Z_12_08_2020/FU_Scan_Tumors_unique_12_CC.nii.gz'
    #
    # case, file = load_nifti_data(file_name)
    #
    # # temp = case.copy()
    # # case[case == 12] = 4
    # # case[temp == 4] = 12
    # # from skimage.morphology import ball
    # # dilate = 10
    # # t = time()
    # # case1 = expand_labels(case, dilate)
    # # print(calculate_runtime(t))
    # # t = time()
    # # case2 = expand_labels(case, dilate, file.header.get_zooms())
    # # print(calculate_runtime(t))
    # # print(file.header.get_zooms())
    # # t = time()
    # # case3 = dilation(case, selem=ball(dilate))
    # # print(calculate_runtime(t))
    # # os.makedirs('test_you_can_delete', exist_ok=True)
    # # save(Nifti1Image(case, file.affine), 'test_you_can_delete/original.nii.gz')
    # # save(Nifti1Image(case1, file.affine), 'test_you_can_delete/expand.nii.gz')
    # # save(Nifti1Image(case2, file.affine), 'test_you_can_delete/expand_with_spacing.nii.gz')
    # # save(Nifti1Image(case3, file.affine), 'test_you_can_delete/dilation.nii.gz')
    # #
    # # from shutil import copy
    # # copy('matching/BL_A_Z_15_07_2020_FU_A_Z_12_08_2020/FU_Scan_CT.nii.gz', 'test_you_can_delete/FU_Scan_CT.nii.gz')
    # #
    # # # from skimage.morphology import disk
    # # a = np.array(([0, 0, 0, 0, 0, 0, 0, 0],
    # #               [0, 0, 0, 0, 0, 0, 0, 0],
    # #               [0, 0, 1, 0, 0, 0, 0, 0],
    # #               [0, 0, 0, 0, 0, 0, 0, 0],
    # #               [0, 0, 0, 0, 0, 0, 0, 0],
    # #               [0, 0, 0, 0, 0, 0, 0, 0],
    # #               [0, 0, 0, 0, 0, 0, 0, 0],
    # #               [0, 0, 0, 0, 0, 0, 0, 0]))
    # # print(a)
    # # print('------------------------')
    # # a_ = a.copy()
    # # a_ = expand_labels(a_, 1)
    # # # a_ = expand_labels(a_, 1)
    # # # a_ = expand_labels(a_, 1)
    # # # a_ = expand_labels(a_, 1)
    # # # a_ = expand_labels(a_, 1)
    # # # a_ = expand_labels(a_, 1)
    # # print(a_)
    # # print('------------------------')
    # # print(expand_labels(a, 1, voxelspacing=(1, 0.5)))
    # # # a_ = a.copy()
    # # # a_ = dilation(a_, selem=disk(1))
    # # # a_ = dilation(a_, selem=disk(1))
    # # # a_ = dilation(a_, selem=disk(1))
    # # # a_ = dilation(a_, selem=disk(1))
    # # # a_ = dilation(a_, selem=disk(1))
    # # # a_ = dilation(a_, selem=disk(1))
    # # # print(a_)
    # # exit(0)
    # match_2_cases(case, case, voxelspacing=file.header.get_zooms(), return_iteration_and_reverse_indicator=True)
    #
    # pix_dims = file.header.get_zooms()
    #
    # case1 = (case == 5).astype(case.dtype)
    # case2 = (case == 7).astype(case.dtype)
    #
    # t1 = time()
    # res_assd1 = assd(case1, case2, voxelspacing=pix_dims, connectivity=2)
    # res_hd1 = hd(case1, case2, voxelspacing=pix_dims, connectivity=2)
    # res_md1 = min_distance(case1, case2, voxelspacing=pix_dims, connectivity=2, crop_to_relevant_scope=False)
    # t1 = time() - t1
    #
    # t2 = time()
    # # res_assd2, res_hd2, res_md2 = assd_hd_and_min_distance(case1, case2, voxelspacing=pix_dims, connectivity=2, crop_to_relevant_scope=True)
    # res_assd2 = ASSD(case1, case2, voxelspacing=pix_dims, connectivity=2, crop_to_relevant_scope=True)
    # res_hd2 = Hausdorff(case1, case2, voxelspacing=pix_dims, connectivity=2, crop_to_relevant_scope=True)
    # res_md2 = min_distance(case1, case2, voxelspacing=pix_dims, connectivity=2, crop_to_relevant_scope=True)
    # t2 = time() - t2
    #
    # print(res_assd1, res_hd1, res_md1, t1)
    # print(res_assd2, res_hd2, res_md2, t2)
    # print(res_assd1 == res_assd2, res_hd1 == res_hd2, res_md1 == res_md2)
    #
    #
    # # def bbox2_2D(img):
    # #     x = np.any(img, axis=1)
    # #     y = np.any(img, axis=0)
    # #
    # #     xmin, xmax = np.where(x)[0][[0, -1]]
    # #     ymin, ymax = np.where(y)[0][[0, -1]]
    # #
    # #     return xmin, xmax, ymin, ymax
    # #
    # #
    # # ex1 = np.array([[0, 0, 0, 0, 0],
    # #                 [0, 1, 1, 1, 0],
    # #                 [0, 1, 1, 1, 0],
    # #                 [0, 1, 1, 1, 0],
    # #                 [0, 0, 0, 0, 0]])
    # #
    # # ex2 = np.ones_like(ex1)
    # #
    # # print(min_distance(ex1, ex2))
    #
    # #
    # #
    # # print(ex1, ex2, sep='\n---------------------------\n', end='\n---------------------------\n')
    # #
    # # shape = (512, 512)
    # # ex1_expand = np.zeros(shape)
    # # ex2_expand = np.zeros(shape)
    # # ex1_expand[shape[0]//2:shape[0]//2+ex1.shape[0], shape[1]//2: shape[1]//2+ex1.shape[1]] = ex1
    # # ex2_expand[shape[0]//2:shape[0]//2+ex2.shape[0], shape[1]//2: shape[1]//2+ex2.shape[1]] = ex2
    # #
    # # t = time()
    # # res1 = hd(ex1_expand, ex2_expand)
    # # print(res1, time()-t)
    # #
    # # t = time()
    # # xmin, xmax, ymin, ymax = bbox2_2D(np.logical_or(ex1_expand, ex2_expand))
    # # ex1_cropped = ex1_expand[xmin:xmax+1, ymin:ymax+1]
    # # ex2_cropped = ex2_expand[xmin:xmax+1, ymin:ymax+1]
    # # res2 = hd(ex1_cropped, ex2_cropped)
    # # print(res2, time()-t, end='\n---------------------------\n')
    # #
    # # print(ex1_cropped, ex2_cropped, sep='\n---------------------------\n', end='\n---------------------------\n')

    # all_tumors_files = sorted(glob('/mnt/sda1/aszeskin/Data_Followup_Full_29_4_2021/BL_*/*_Scan_Tumors.nii.gz'))
    # def f(tumors_file):
    #     tumors_case, nifti_file = load_nifti_data(tumors_file)
    #
    #     if np.isin(2, tumors_case):
    #         tumors_case[tumors_case == 1] = 0
    #         tumors_case[tumors_case == 2] -= 1
    #
    #     assert np.all(np.unique(tumors_case) == [0, 1])
    #
    #     save(Nifti1Image(tumors_case, nifti_file.affine), tumors_file)
    #
    #
    # from tqdm.contrib.concurrent import process_map
    # process_map(f, all_tumors_files, max_workers=os.cpu_count()-2)
    #
    # exit(0)




    # corrected_dir = f'/cs/casmip/rochman/data_to_recheck'
    # old_dir = f'/cs/casmip/rochman/Errors_Characterization/data_to_recheck'
    # dest_dir = f'/cs/casmip/rochman/Errors_Characterization/data_to_recheck_corrected'
    # i = 1
    # for current_corrected_case in sorted(glob(f'{corrected_dir}/*')):
    #     current_dest_case_dir = f'{dest_dir}/{os.path.basename(current_corrected_case)}'
    #     os.makedirs(current_dest_case_dir, exist_ok=True)
    #
    #     corrected_tumors_file = f'{current_corrected_case}/Tumors.nii.gz'
    #     old_tumors_file = f'{old_dir}/{os.path.basename(current_corrected_case)}/Tumors.nii.gz'
    #
    #     corrected_tumors_case, _ = load_nifti_data(corrected_tumors_file)
    #     old_tumors_case, _ = load_nifti_data(old_tumors_file)
    #
    #     if (corrected_tumors_case != old_tumors_case).any():
    #         print('---------------------------------------------------')
    #         os.system(f'\cp {corrected_tumors_file} {current_dest_case_dir}/Tumors.nii.gz')
    #         print(f'{i}: Find change in tumors case of {os.path.basename(current_corrected_case)}')
    #         i += 1
    #     else:
    #         os.symlink(os.readlink(old_tumors_file), f'{current_dest_case_dir}/Tumors.nii.gz')
    #
    #     os.symlink(os.readlink(f'{old_dir}/{os.path.basename(current_corrected_case)}/Liver.nii.gz'), f'{current_dest_case_dir}/Liver.nii.gz')
    #     os.symlink(os.readlink(f'{old_dir}/{os.path.basename(current_corrected_case)}/Scan_CT.nii.gz'), f'{current_dest_case_dir}/Scan_CT.nii.gz')
    #
    # exit(0)

    # def preprocess_liver_and_tumors_gt_segmentation(tumors, liver):
    #     # pre-process the tumors and liver segmentations
    #     tumors = pre_process_segmentation(tumors)
    #     liver = np.logical_or(liver, tumors).astype(liver.dtype)
    #     liver = getLargestCC(liver)
    #     liver = pre_process_segmentation(liver, remove_small_obs=False)
    #     tumors = np.logical_and(liver, tumors).astype(tumors.dtype)
    #     return tumors, liver

    # def color_tumors(case_path):
    #     bl_tumor_file = f'{case_path}/BL_Scan_Tumors.nii.gz'
    #     fu_tumor_file = f'{case_path}/FU_Scan_Tumors.nii.gz'
    #
    #     bl_liver_file = f'{case_path}/BL_Scan_Liver.nii.gz'
    #     fu_liver_file = f'{case_path}/FU_Scan_Liver.nii.gz'
    #
    #     bl_tumor_case, file = load_nifti_data(bl_tumor_file)
    #     fu_tumor_case, _ = load_nifti_data(fu_tumor_file)
    #
    #     bl_liver_case, _ = load_nifti_data(bl_liver_file)
    #     fu_liver_case, _ = load_nifti_data(fu_liver_file)
    #
    #     bl_tumor_case, bl_liver_case = preprocess_liver_and_tumors_gt_segmentation(bl_tumor_case, bl_liver_case)
    #     fu_tumor_case, fu_liver_case = preprocess_liver_and_tumors_gt_segmentation(fu_tumor_case, fu_liver_case)
    #
    #     bl_tumor_case = get_connected_components(bl_tumor_case).astype(bl_tumor_case.dtype)
    #     fu_tumor_case = get_connected_components(fu_tumor_case).astype(fu_tumor_case.dtype)
    #
    #     n_bl_tumors = np.unique(bl_tumor_case).size - 1
    #     n_fu_tumors = np.unique(fu_tumor_case).size - 1
    #
    #     os.rename(bl_tumor_file, f'corrected_segmentation_for_matching/test/{os.path.basename(case_path)}_BL_Scan_Tumors.nii.gz')
    #     os.rename(fu_tumor_file, f'corrected_segmentation_for_matching/test/{os.path.basename(case_path)}_FU_Scan_Tumors.nii.gz')
    #
    #     save(Nifti1Image(bl_tumor_case, file.affine), f'{case_path}/BL_Scan_Tumors_unique_{n_bl_tumors}_CC.nii.gz')
    #     save(Nifti1Image(fu_tumor_case, file.affine), f'{case_path}/FU_Scan_Tumors_unique_{n_fu_tumors}_CC.nii.gz')
    pass
    # todo you can delete
    case_name = 'BL_B_T_18_02_2019_FU_B_T_05_05_2019'
    if os.path.isdir(f'/cs/casmip/rochman/Errors_Characterization/new_test_set_for_matching/{case_name}'):
        bl_tumors, file = load_nifti_data(glob(f'/cs/casmip/rochman/Errors_Characterization/new_test_set_for_matching/{case_name}/improved_registration_BL_Scan_Tumors_unique_*')[0])
        fu_tumors, _ = load_nifti_data(glob(f'/cs/casmip/rochman/Errors_Characterization/new_test_set_for_matching/{case_name}/FU_Scan_Tumors_unique_*')[0])
    else:
        bl_tumors, file = load_nifti_data(glob(f'/cs/casmip/rochman/Errors_Characterization/corrected_segmentation_for_matching/{case_name}/improved_registration_BL_Scan_Tumors_unique_*')[0])
        fu_tumors, _ = load_nifti_data(glob(f'/cs/casmip/rochman/Errors_Characterization/corrected_segmentation_for_matching/{case_name}/FU_Scan_Tumors_unique_*')[0])

    # test, test_file = load_nifti_data('/cs/casmip/rochman/Errors_Characterization/new_test_set_for_matching/BL_A_S_H_01_09_2019_FU_A_S_H_06_04_2020/BL_Scan_Tumors_unique_12_CC.nii.gz')
    #
    #
    # t = time()
    # print(get_minimum_distance_between_CCs(fu_tumors, voxel_to_real_space_trans=file.affine,
    #                                        max_points_per_CC=5000, seed=42))
    # # print(get_minimum_distance_between_CCs(test, voxel_to_real_space_trans=test_file.affine,
    # #                                        max_points_per_CC=5000, seed=42))
    # print(calculate_runtime(t))
    # exit(0)

    # t = time()
    # matches1 = match_2_cases(bl_tumors, fu_tumors, file.header.get_zooms(), max_dilate_param=5, return_iteration_and_reverse_indicator=False)
    # print(calculate_runtime(t))
    t = time()
    matches3 = match_2_cases_v3(bl_tumors, fu_tumors, file.header.get_zooms(), max_dilate_param=9, return_iteration_indicator=False)
    print(calculate_runtime(t))
    t = time()
    matches5 = match_2_cases_v5(bl_tumors, fu_tumors, file.header.get_zooms(), max_dilate_param=9, return_iteration_indicator=False)
    print(calculate_runtime(t))
    # t = time()
    # matches4 = match_2_cases_v4(bl_tumors, fu_tumors, file.header.get_zooms(), max_dilate_param=9,
    #                             return_iteration_indicator=False)
    # print(calculate_runtime(t))
    # matches1.sort(key=lambda m: m)
    # matches2.sort(key=lambda m: m)
    matches3.sort(key=lambda m: m)
    # matches4.sort(key=lambda m: m)
    matches5.sort(key=lambda m: m)
    # print(matches1)
    # print(matches2)
    print(matches3)
    # print(matches4)
    print(matches5)
    print('Matches3: ', end='')
    for m in matches3:
         if m not in matches5:
             print(m, end=', ')
    print('\nMatches5: ', end='')
    for m in matches5:
         if m not in matches3:
             print(m, end=', ')
    exit(0)

    # todo you can delete
    tumors_case_file = '/cs/casmip/rochman/Errors_Characterization/corrected_segmentation_for_matching/BL_B_A_D_04_03_2014_FU_B_A_D_02_06_2014/improved_registration_BL_Scan_Tumors_unique_21_CC.nii.gz'
    tumors_case, file = load_nifti_data(tumors_case_file)
    tumors_case = (tumors_case > 0).astype(tumors_case.dtype)
    tumors_case = get_connected_components(tumors_case, connectivity=1).astype(tumors_case.dtype)
    save(Nifti1Image(tumors_case, file.affine), '/cs/casmip/rochman/Errors_Characterization/corrected_segmentation_for_matching/BL_B_A_D_04_03_2014_FU_B_A_D_02_06_2014/you_can_delete_connectivity_1_improved_registration_BL_Scan_Tumors_unique_21_CC.nii.gz')
    exit(0)



    from skimage.morphology import disk, binary_erosion
    liver_file = '/cs/casmip/rochman/Errors_Characterization/matching/BL_A_Ac_21_12_2020_FU_A_Ac_30_12_2020/FU_Scan_Liver.nii.gz'
    liver_case, file = load_nifti_data(liver_file)
    liver_border = np.logical_xor(liver_case, binary_erosion(liver_case, disk(1).reshape([3, 3, 1]))).astype(liver_case.dtype)
    save(Nifti1Image(liver_border, file.affine), '/cs/casmip/rochman/Errors_Characterization/matching/BL_A_Ac_21_12_2020_FU_A_Ac_30_12_2020/you_can_delete_fu_Liver_border.nii.gz')
    exit(0)


    def matching_for_richard(case_path):
        case_name = os.path.basename(case_path)
        src_dir = case_path
        dest_dir = f'/cs/casmip/rochman/Errors_Characterization/matching_for_richard/round_3/{case_name}'
        os.makedirs(dest_dir)

        os.symlink(f'{src_dir}/improved_registration_BL_Scan_CT.nii.gz', f'{dest_dir}/improved_registration_BL_Scan_CT.nii.gz')
        bl_tumors_file_basename = os.path.basename(glob(f'{src_dir}/improved_registration_BL_Scan_Tumors_unique_*')[0])
        os.symlink(f'{src_dir}/{bl_tumors_file_basename}', f'{dest_dir}/{bl_tumors_file_basename}')

        os.symlink(f'{src_dir}/FU_Scan_CT.nii.gz', f'{dest_dir}/FU_Scan_CT.nii.gz')
        fu_tumors_file_basename = os.path.basename(glob(f'{src_dir}/FU_Scan_Tumors_unique_*')[0])
        os.symlink(f'{src_dir}/{fu_tumors_file_basename}', f'{dest_dir}/{fu_tumors_file_basename}')

        os.symlink(f'{src_dir}/pred_matching_graph.jpg', f'{dest_dir}/pred_matching_graph.jpg')
        os.symlink(f'{src_dir}/matching.xlsx', f'{dest_dir}/matching.xlsx')

    cases_paths = sorted(glob('/cs/casmip/rochman/Errors_Characterization/corrected_segmentation_for_matching/BL_*'))
    from tqdm.contrib.concurrent import process_map
    os.makedirs('corrected_segmentation_for_matching/test', exist_ok=True)
    process_map(matching_for_richard, cases_paths, max_workers=os.cpu_count() - 2)
    # list(map(color_tumors, cases_paths))
    exit(0)







    # from registeration_by_folder import liver_registeration
    # bl_and_fu_names = [os.path.basename(pair_name).replace('BL_', '').split('_FU_') for pair_name in glob(f'matching/BL_*')]
    #
    # def registration_of_pair(bl_and_fu_name):
    #     corrected_dir = f'/cs/casmip/rochman/Errors_Characterization/data_to_recheck_corrected'
    #     res_dir = '/cs/casmip/rochman/Errors_Characterization/corrected_segmentation_for_matching'
    #     bl_name, fu_name = bl_and_fu_name
    #
    #     bl_CT = f'{corrected_dir}/{bl_name}/Scan_CT.nii.gz'
    #     bl_liver = f'{corrected_dir}/{bl_name}/Liver.nii.gz'
    #     bl_tumors = f'{corrected_dir}/{bl_name}/Tumors.nii.gz'
    #
    #     fu_CT = f'{corrected_dir}/{fu_name}/Scan_CT.nii.gz'
    #     fu_liver = f'{corrected_dir}/{fu_name}/Liver.nii.gz'
    #     fu_tumors = f'{corrected_dir}/{fu_name}/Tumors.nii.gz'
    #
    #     register_class = liver_registeration([bl_CT], [bl_liver], [bl_tumors], [fu_CT], [fu_liver], [fu_tumors],
    #                                          dest_path=res_dir, bl_name=bl_name, fu_name=fu_name)
    #     register_class.affine_registeration()
    #
    #
    # from tqdm.contrib.concurrent import process_map
    # process_map(registration_of_pair, bl_and_fu_names, max_workers=os.cpu_count()-2)
    # # list(map(registration_of_pair, bl_and_fu_names[:1]))




