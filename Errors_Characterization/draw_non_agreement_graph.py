from matching_graphs import _draw_matching_graph, load_matching_graph
import sys
import networkx as nx
from matplotlib import pyplot as plt
from os.path import isdir

_, folder_name, gt_save_fig_path, pred_save_fig_path, max_dilate = sys.argv
# folder_name, gt_save_fig_path, pred_save_fig_path, max_dilate = 'BL_H_G_06_10_2019_FU_H_G_24_11_2019', '/tmp/gt_diff.jpg', '/tmp/pred_diff.jpg', '5'

gt_save_fig_path = None
pred_save_fig_path = None

liver_study = False

if liver_study:
    dir_path = f'/cs/casmip/rochman/Errors_Characterization/new_test_set_for_matching/{folder_name}'
    if not isdir(dir_path):
        dir_path = f'/cs/casmip/rochman/Errors_Characterization/corrected_segmentation_for_matching/{folder_name}'
else:
    dir_path = f'/cs/casmip/rochman/Errors_Characterization/lung_test_set_for_matching/{folder_name}'

BL_n_tumors, FU_n_tumors, gt_edges, case_name, bl_weights, fu_weights, _, _, _, _ = load_matching_graph(
    f'{dir_path}/gt_matching_graph.json')
_, _, pred_edges, _, _, _, _, _, _, _ = load_matching_graph(f'{dir_path}/pred_matching_graph.json')
pred_edges = [x[1] for x in pred_edges if x[0] <= int(max_dilate)]
# pred_edges = [x[2] for x in pred_edges if x[1] <= int(max_dilate)]

# # extracting the BL labels
# BL_labels = np.arange(1, BL_n_tumors)
#
# # extracting the FU labels
# FU_labels = np.arange(1, FU_n_tumors)
# FU_labels += BL_n_tumors
#
# V = list(BL_labels - 1) + list(FU_labels - 1)
# visited = [False] * len(V)
# adjacency_lists = []
# for _ in range(BL_n_tumors + FU_n_tumors):
#     adjacency_lists.append([])
# for (bl_v, fu_v) in gt_matches:
#     fu_v += BL_n_tumors - 1
#     bl_v -= 1
#     adjacency_lists[bl_v].append(fu_v)
#     adjacency_lists[fu_v].append(bl_v)
#
#
# def DFS(v, CC=None):
#     if CC is None:
#         CC = []
#     visited[v] = True
#     CC.append(v)
#     V.remove(v)
#
#     for u in adjacency_lists[v]:
#         if not visited[u]:
#             CC = DFS(u, CC)
#     return CC
#
#
# is_bl_tumor = lambda v: v <= BL_n_tumors - 1
#
#
# def bl_and_fu(CC):
#     bl_in_CC = []
#     fu_in_CC = []
#     for v in CC:
#         if is_bl_tumor(v):
#             bl_in_CC.append(v + 1)
#         else:
#             fu_in_CC.append(v + 1 - BL_n_tumors)
#     return bl_in_CC, fu_in_CC
#
#
# results = []
#
# while len(V) > 0:
#     v = V[0]
#     current_CC = DFS(v)
#
#     # in case the current tumor is a isolated
#     if len(current_CC) == 1:
#         continue
#
#     bl_in_CC, fu_in_CC = bl_and_fu(current_CC)

bl_tumors = [f'{t}_bl' for t in range(1, BL_n_tumors + 1)]
fu_tumors = [f'{t}_fu' for t in range(1, FU_n_tumors + 1)]

gt_edges = [(f'{int(e[0])}_bl', f'{int(e[1])}_fu') for e in gt_edges]
pred_edges = [(f'{int(e[0])}_bl', f'{int(e[1])}_fu') for e in pred_edges]

# build the GT graph
gt_G = nx.Graph()
gt_G.add_nodes_from(bl_tumors, bipartite='bl')
gt_G.add_nodes_from(fu_tumors, bipartite='fu')
gt_G.add_edges_from(gt_edges)

# build the pred graph
pred_G = nx.Graph()
pred_G.add_nodes_from(bl_tumors, bipartite='bl')
pred_G.add_nodes_from(fu_tumors, bipartite='fu')
pred_G.add_edges_from(pred_edges)

gt_CCs = list(nx.connected_component_subgraphs(gt_G))
pred_CCs = list(nx.connected_component_subgraphs(pred_G))

def get_diff_a_from_b(a_CCs, b_CCs):
    a_CCs = a_CCs.copy()
    b_CCs = b_CCs.copy()

    diff = nx.Graph()

    for sub_a in a_CCs:
        found_equal = False
        for sub_b in b_CCs:
            if sub_a.nodes == sub_b.nodes and sub_a.edges == sub_b.edges:
                found_equal = True
                b_CCs.remove(sub_b)
                break
        if not found_equal:
            diff.add_nodes_from(sub_a.nodes)
            diff.add_edges_from(sub_a.edges)

    return diff

gt_diff_G = get_diff_a_from_b(gt_CCs, pred_CCs)
pred_diff_G = get_diff_a_from_b(pred_CCs, gt_CCs)

# diff_G = pred_diff_G
diff_G = gt_diff_G
# bl_tumors = sorted([v for v in diff_G.nodes if 'bl' in v], key=lambda v: int(v.split('_')[0]))
bl_tumors = [v for v in diff_G.nodes if 'bl' in v]
# fu_tumors = sorted([v for v in diff_G.nodes if 'fu' in v], key=lambda v: int(v.split('_')[0]))
fu_tumors = [v for v in diff_G.nodes if 'fu' in v]

gt_edges = sorted([e if 'bl' in e[0] else e[::-1] for e in gt_diff_G.edges], key=lambda e: (int(e[0].split('_')[0]), int(e[1].split('_')[0])))
pred_edges = sorted([e if 'bl' in e[0] else e[::-1] for e in pred_diff_G.edges], key=lambda e: (int(e[0].split('_')[0]), int(e[1].split('_')[0])))

BL_n_tumors, FU_n_tumors = len(bl_tumors), len(fu_tumors)

bl_new_weights = []
for bl in bl_tumors:
    bl_new_weights.append(bl_weights[int(bl.split('_')[0]) - 1])

fu_new_weights = []
for fu in fu_tumors:
    fu_new_weights.append(fu_weights[int(fu.split('_')[0]) - 1])

gt_edges = [(int(e[0].split('_')[0]), int(e[1].split('_')[0])) for e in gt_edges]
pred_edges = [(int(e[0].split('_')[0]), int(e[1].split('_')[0])) for e in pred_edges]

_draw_matching_graph(bl_tumors, fu_tumors, len(bl_tumors), len(fu_tumors), gt_edges, case_name + '_GT', bl_new_weights,
                     fu_new_weights, saving_file_name=gt_save_fig_path, show=False, close_fig_at_end=False)

_draw_matching_graph(bl_tumors, fu_tumors, len(bl_tumors), len(fu_tumors), pred_edges, case_name + '_PRED', bl_new_weights,
                     fu_new_weights, saving_file_name=pred_save_fig_path, show=False, close_fig_at_end=False)

plt.show()



