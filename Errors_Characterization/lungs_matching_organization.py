from datetime import date
from skimage.measure import centroid
import json
from utils import *
from matching_graphs import save_matching_graph, draw_matching_graph


def get_original_path(path):
    path = os.path.realpath(path)
    prefix = '/cs/usr/bennydv/Desktop/bennydv/'
    if path.startswith(prefix):
        path = f'/cs/casmip/bennydv/{path[len(prefix):]}'
    return path


def reorder_labeled_lesions(labeled_lesions: np.ndarray) -> Tuple[np.ndarray, Dict[int, int], Dict[int, int], int, bool]:

    old_labels = np.unique(labeled_lesions)
    old_labels = old_labels[old_labels != 0]

    n_labels = old_labels.size

    new_labels = np.arange(1, n_labels + 1)

    has_changed = False
    if np.any(old_labels != new_labels):
        # reordering the labels
        labeled_lesions = np.select((labeled_lesions.T.reshape([*labeled_lesions.T.shape, 1]) == old_labels.reshape([1, -1])).T, new_labels, labeled_lesions)
        has_changed = True

    to_old_labels = dict(zip(new_labels, old_labels))
    to_new_labels = dict(zip(old_labels, new_labels))

    return labeled_lesions, to_old_labels, to_new_labels, n_labels, has_changed
