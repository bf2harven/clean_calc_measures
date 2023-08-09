from typing import List, Tuple, Optional, Any
import json
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import collections
import warnings
from utils import *


class default_dict(collections.UserDict):
    def __init__(self, default_factory=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not callable(default_factory) and default_factory is not None:
            raise TypeError('first argument must be callable or None')
        self.default_factory = default_factory

    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        if key not in self:
            self[key] = self.default_factory()
        return self[key]


def get_itk_colors(src: str = '/cs/casmip/rochman/Errors_Characterization/label_descriptions.txt'):
    with open(src) as f:
        colors = f.readlines()
    colors = colors[15:]
    colors = [[c for c in l.split(' ') if c != ''] for l in colors]
    return default_dict(lambda: (0, 0, 0), ((int(l[0]), (int(l[1]) / 255, int(l[2]) / 255, int(l[3]) / 255)) for l in colors))


def draw_matching_graph(n_bl_nodes: int, n_fu_nodes: int, edges: List[Tuple[int, int]], case_name: str,
                        bl_weights: Optional[list] = None, fu_weights: Optional[list] = None,
                        edges_weights: Optional[dict] = None,
                        saving_file_name: Optional[str] = None, show: bool = False):

    bl_tumors = [f'{t}_bl' for t in range(1, n_bl_nodes + 1)]
    fu_tumors = [f'{t}_fu' for t in range(1, n_fu_nodes + 1)]

    _draw_matching_graph(bl_tumors, fu_tumors, n_bl_nodes, n_fu_nodes, edges, case_name, bl_weights, fu_weights,
                         edges_weights, saving_file_name, show)


def _draw_matching_graph(bl_tumors: List[str], fu_tumors: List[str], n_bl_nodes: int, n_fu_nodes: int,
                         edges: List[Tuple[int, int]], case_name: str, bl_weights: Optional[list] = None,
                         fu_weights: Optional[list] = None, edges_weights: Optional[dict] = None,
                         saving_file_name: Optional[str] = None, show: bool = False,
                         close_fig_at_end: bool = True):

    if bl_weights is not None:
        assert len(bl_weights) == n_bl_nodes, f'the bl_weights list have to contain {n_bl_nodes} weights and it contains {len(bl_weights)} weights: case_name = "{case_name}"'
        temp_bl_weights = dict()
        for i, bl_tumor in enumerate(bl_tumors):
            temp_bl_weights[bl_tumor] = bl_weights[i]
        bl_weights = temp_bl_weights

    if fu_weights is not None:
        assert len(fu_weights) == n_fu_nodes, f'the fu_weights list have to contain {n_fu_nodes} weights and it contains {len(fu_weights)} weights: case_name = "{case_name}"'
        temp_fu_weights = dict()
        for i, fu_tumor in enumerate(fu_tumors):
            temp_fu_weights[fu_tumor] = fu_weights[i]
        fu_weights = temp_fu_weights

    edges = [(f'{int(e[0])}_bl', f'{int(e[1])}_fu') for e in edges]

    matched_tumors = [n for n in bl_tumors if n in (e[0] for e in edges)]
    matched_tumors += [n for n in fu_tumors if n in (e[1] for e in edges)]

    # build the graph
    G = nx.Graph()
    G.add_nodes_from(bl_tumors, bipartite='bl')
    G.add_nodes_from(fu_tumors, bipartite='fu')
    G.add_edges_from(edges)

    # define the position of the nodes
    max_range_to_draw = max(n_bl_nodes, n_fu_nodes)
    delta = 0.47 if bl_weights is not None else 0
    pos = dict(zip(bl_tumors, zip([delta] * n_bl_nodes, (max_range_to_draw * i / (n_bl_nodes - 1) for i in range(n_bl_nodes)[::-1]) if n_bl_nodes > 1 else (max_range_to_draw,))))
    delta = 0.47 if fu_weights is not None else 0
    pos.update(
        dict(zip(fu_tumors, zip([1 - delta] * n_fu_nodes, (max_range_to_draw * i / (n_fu_nodes - 1) for i in range(n_fu_nodes)[::-1]) if n_fu_nodes > 1 else (max_range_to_draw,)))))

    # define the nodes' labels and colors
    nodelist = bl_tumors + fu_tumors
    nodes_labels = dict((n, int(n.split('_')[0])) for n in nodelist)
    colors = get_itk_colors()
    node_color = [colors[nodes_labels[n]] for n in nodelist]

    # draw the graph
    x = 12.4
    y = 18.8
    fig = plt.figure(figsize=(x, y))
    plt.title(f'{case_name}\n\nMatching Graph', fontsize=25)

    cf = plt.gcf()
    cf.set_facecolor("w")
    if cf._axstack() is None:
        ax = cf.add_axes((0, 0, 1, 1))
    else:
        ax = cf.gca()

    ax.axis('off')

    if n_bl_nodes > 0 or n_fu_nodes > 0:

        nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=nodelist, node_color=node_color, node_size=1000)
        m = y/max_range_to_draw
        c = 1
        x = c / m

        if bl_weights is not None and fu_weights is None:

            plt.text(pos[bl_tumors[0]][0] - 0.0555, pos[bl_tumors[0]][1] + x, '$_{(slice)}$ BL', fontsize=20)
            plt.text(pos[fu_tumors[0]][0] - 0.012, pos[fu_tumors[0]][1] + x, 'FU', fontsize=20)

            bl_weights_pos = dict((v, (pos[v][0] - 0.032, pos[v][1])) for v in pos if v.endswith('bl'))
            nx.draw_networkx_labels(G, bl_weights_pos, labels=bl_weights, ax=ax, font_size=15)

        elif bl_weights is None and fu_weights is not None:

            plt.text(pos[bl_tumors[0]][0] - 0.012, pos[bl_tumors[0]][1] + x, 'BL', fontsize=20)
            plt.text(pos[fu_tumors[0]][0] - 0.012, pos[fu_tumors[0]][1] + x, 'FU $_{(slice)}$', fontsize=20)

            fu_weights_pos = dict((v, (pos[v][0] + 0.032, pos[v][1])) for v in pos if v.endswith('fu'))
            nx.draw_networkx_labels(G, fu_weights_pos, labels=fu_weights, ax=ax, font_size=15)

        elif bl_weights is not None and fu_weights is not None:

            plt.text(pos[bl_tumors[0]][0] - 0.009, pos[bl_tumors[0]][1] + x, '$_{(slice)}$ BL', fontsize=20)
            plt.text(pos[fu_tumors[0]][0] - 0.0022, pos[fu_tumors[0]][1] + x, 'FU $_{(slice)}$', fontsize=20)

            bl_weights_pos = dict((v, (pos[v][0] - 0.005, pos[v][1])) for v in pos if v.endswith('bl'))
            nx.draw_networkx_labels(G, bl_weights_pos, labels=bl_weights, ax=ax, font_size=15)

            fu_weights_pos = dict((v, (pos[v][0] + 0.005, pos[v][1])) for v in pos if v.endswith('fu'))
            nx.draw_networkx_labels(G, fu_weights_pos, labels=fu_weights, ax=ax, font_size=15)

        else:

            plt.text(pos[fu_tumors[0]][0] - 0.024, pos[fu_tumors[0]][1] + x, 'FU', fontsize=20)
            plt.text(pos[bl_tumors[0]][0] - 0.024, pos[bl_tumors[0]][1] + x, 'BL', fontsize=20)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nx.draw_networkx_edges(G, pos, ax=ax, edge_color='r')

            if edges_weights is not None:
                nx.draw_networkx_edge_labels(G, pos, edge_labels=edges_weights, ax=ax, label_pos=0.15, rotate=False,
                                             font_size=9)

        texts = nx.draw_networkx_labels(G, pos, ax=ax, labels=nodes_labels, font_family='fantasy', font_size=22,
                                        font_color='w')
        for text in texts.values():
            text.set_path_effects([PathEffects.Stroke(linewidth=1, foreground='black')])

    if saving_file_name is not None:
        plt.savefig(saving_file_name)

    if show:
        # plt.show(bbox_inches='tight')
        # mng = plt.get_current_fig_manager()
        # mng.resize(*mng.window.maxsize())
        plt.show()

    if close_fig_at_end:
        plt.close(fig)


def save_matching_graph(n_bl_nodes: int, n_fu_nodes: int, edges: List[Tuple[int, int]],
                        case_name: str, saving_file_name: str, bl_weights: Optional[List[int]] = None,
                        fu_weights: Optional[List[int]] = None, bl_diameters: Optional[List[float]] = None,
                        fu_diameters: Optional[List[float]] = None, bl_organ_volume: Optional[float] = None,
                        fu_organ_volume: Optional[float] = None):

    if bl_weights is None:
        bl_weights = []

    if fu_weights is None:
        fu_weights = []

    if bl_diameters is None:
        bl_diameters = []

    if fu_diameters is None:
        fu_diameters = []

    if bl_organ_volume is None:
        bl_organ_volume = -1

    if fu_organ_volume is None:
        fu_organ_volume = -1

    graph_dict = {'case_name': case_name, 'n_bl_nodes': n_bl_nodes, 'n_fu_nodes': n_fu_nodes, 'edges': edges,
                  'bl_weights': bl_weights, 'fu_weights': fu_weights, 'bl_diameters': bl_diameters,
                  'fu_diameters': fu_diameters, 'bl_organ_volume': bl_organ_volume, 'fu_organ_volume': fu_organ_volume}

    json_string = json.dumps(graph_dict)
    with open(saving_file_name, "w") as json_file:
        json_file.write(json_string)


def load_matching_graph(file_name: str):
    with open(file_name) as json_file:
        json_string = json_file.read()
    graph_dict = json.loads(json_string)
    n_bl_nodes = graph_dict['n_bl_nodes']
    n_fu_nodes = graph_dict['n_fu_nodes']
    edges = [tuple(e) for e in graph_dict['edges']]
    case_name = graph_dict['case_name']
    bl_weights = graph_dict.get('bl_weights', None)
    fu_weights = graph_dict.get('fu_weights', None)
    bl_diameters = graph_dict.get('bl_diameters', None)
    fu_diameters = graph_dict.get('fu_diameters', None)
    bl_organ_volume = graph_dict.get('bl_organ_volume', None)
    fu_organ_volume = graph_dict.get('fu_organ_volume', None)

    if (bl_weights is not None) and (len(bl_weights) != n_bl_nodes):
        bl_weights = None

    if (fu_weights is not None) and (len(fu_weights) != n_fu_nodes):
        fu_weights = None

    if (bl_diameters is not None) and (len(bl_diameters) != n_bl_nodes):
        bl_diameters = None

    if (fu_diameters is not None) and (len(fu_diameters) != n_fu_nodes):
        fu_diameters = None

    if (bl_organ_volume is not None) and (bl_organ_volume <= 0):
        bl_organ_volume = None

    if (fu_organ_volume is not None) and (fu_organ_volume <= 0):
        fu_organ_volume = None

    return (n_bl_nodes, n_fu_nodes, edges, case_name, bl_weights, fu_weights,
            bl_diameters, fu_diameters, bl_organ_volume, fu_organ_volume)


if __name__ == '__main__':
    from skimage.measure import centroid
    def first_matching(case_path):
        bl_tumors_file = glob(f'{case_path}/improved_registration_BL_Scan_Tumors_unique_*')[0]
        fu_tumors_file = glob(f'{case_path}/FU_Scan_Tumors_unique_*')[0]

        bl_tumors_labeled_case, file = load_nifti_data(bl_tumors_file)
        fu_tumors_labeled_case, _ = load_nifti_data(fu_tumors_file)

        matches = match_2_cases(bl_tumors_labeled_case, fu_tumors_labeled_case, voxelspacing=file.header.get_zooms(),
                                max_dilate_param=5)

        matches = [(int(m[0]), int(m[1])) for m in matches]

        n_bl_tumors = int(''.join([c for c in os.path.basename(bl_tumors_file) if c.isdigit()]))
        n_fu_tumors = int(''.join([c for c in os.path.basename(fu_tumors_file) if c.isdigit()]))

        bl_weights = []
        for bl_t in range(1, n_bl_tumors + 1):
            bl_weights.append(int(centroid(bl_tumors_labeled_case == bl_t)[-1] + 1))

        fu_weights = []
        for fu_t in range(1, n_fu_tumors + 1):
            fu_weights.append(int(centroid(fu_tumors_labeled_case == fu_t)[-1] + 1))

        draw_matching_graph(n_bl_tumors, n_fu_tumors, matches, os.path.basename(case_path), bl_weights, fu_weights,
                            saving_file_name=f'{case_path}/pred_matching_graph.jpg')

        save_matching_graph(n_bl_tumors, n_fu_tumors, matches, os.path.basename(case_path),
                            saving_file_name=f'{case_path}/pred_matching_graph.json', bl_weights=bl_weights,
                            fu_weights=fu_weights)

        df = pd.DataFrame(data=matches, columns=['bl tumors', 'fu tumors'])
        writer = pd.ExcelWriter(f'{case_path}/matching.xlsx')
        df.to_excel(writer, index=False)
        writer.save()


    cases_paths = sorted(glob('/cs/casmip/rochman/Errors_Characterization/corrected_segmentation_for_matching/BL_*'))
    from tqdm.contrib.concurrent import process_map
    os.makedirs('corrected_segmentation_for_matching/test', exist_ok=True)
    process_map(first_matching, cases_paths, max_workers=os.cpu_count() - 2)
    # list(map(first_matching, cases_paths))
    exit(0)
    pass
    # n_bl_nodes = 4
    # n_fu_nodes = 1
    # # edges = [(1, 1), (2, 2), (3, 3), (7, 3), (5, 4), (6, 5), (8, 7), (10, 8), (9, 9), (11, 10), (12, 11), (25, 27), (29, 30)]
    # edges = [(1, 1)]
    # # nodes_weights = dict((f'{n}_bl', n+0.0) for n in range(1, n_bl_nodes + 1))
    # bl_weights = [(i*100) % 1000 for i in range(1, n_bl_nodes + 1)]
    # fu_weights = [(i*100) % 1000 for i in range(1, n_fu_nodes + 1)]
    # # nodes_weights.update((f'{n}_fu', n+0.0) for n in range(1, n_fu_nodes + 1))
    # # # nodes_weights = None
    # fu_weights = bl_weights = None
    # # edges_weights = dict(((f'{int(e[0])}_bl', f'{int(e[1])}_fu'), f'{e[0] + 0.0}|{e[1] + 0.0}') for e in edges)
    # case_name = 'BL_A_Ab_03_10_2018_FU_A_Ab_15_07_2018'
    # # json_file_name = 'matching_graph.json'
    # draw_matching_graph(n_bl_nodes, n_fu_nodes, edges, case_name, bl_weights=bl_weights, fu_weights=fu_weights,
    #                     show=True, saving_file_name='matching_graph_test.jpg')
    # # # save_matching_graph(n_bl_nodes, n_fu_nodes, edges, case_name, json_file_name)
    # # # x_n_bl_nodes, x_n_fu_nodes, x_edges, x_case_name = load_matching_grph(json_file_name)
    # # # print(x_n_bl_nodes, x_n_fu_nodes, x_edges, x_case_name, sep='\n')
    # exit(0)

    # ------------------------------------------------------------------------------------------------------------------

    # fu_tumors_files = glob('/cs/casmip/rochman/Errors_Characterization/matching/*_done/FU_Scan_Tumors_unique_*')
    # for fu_tumors_file in fu_tumors_files:
    #     current_dir = f'matching_for_richard/{os.path.basename(os.path.dirname(fu_tumors_file))[:-5]}'
    #     os.makedirs(current_dir, exist_ok=True)
    #     fu_case, file = load_nifti_data(fu_tumors_file)
    #     fu_case[fu_case > 0] -= 1
    #     save(Nifti1Image(fu_case, file.affine), f'{current_dir}/{os.path.basename(fu_tumors_file)}')
    #
    #     bl_tumors_file = glob(f'{os.path.dirname(fu_tumors_file)}/BL_Scan_Tumors_unique_*')[0]
    #     os.symlink(bl_tumors_file, f'{current_dir}/{os.path.basename(bl_tumors_file)}')
    #
    #     os.symlink(f'{os.path.dirname(fu_tumors_file)}/BL_Scan_CT.nii.gz', f'{current_dir}/BL_Scan_CT.nii.gz')
    #     os.symlink(f'{os.path.dirname(fu_tumors_file)}/FU_Scan_CT.nii.gz', f'{current_dir}/FU_Scan_CT.nii.gz')
    #     os.symlink(f'{os.path.dirname(fu_tumors_file)}/gt_matching_graph.jpg', f'{current_dir}/gt_matching_graph.jpg')
    #
    #     n_bl_nodes, n_fu_nodes, edges, case_name = load_matching_grph(f'{os.path.dirname(fu_tumors_file)}/gt_matching_graph.json')
    #
    #     df = pd.DataFrame(data=edges, columns=['bl tumors', 'fu tumors'])
    #     writer = pd.ExcelWriter(f'{current_dir}/matching.xlsx')
    #     df.to_excel(writer, index=False)
    #     writer.save()

    # ------------------------------------------------------------------------------------------------------------------

    # from tqdm import tqdm
    # from tqdm.contrib.concurrent import process_map
    # from skimage.measure import centroid
    #
    # def f(fu_tumors_file):
    #     current_dir = os.path.dirname(fu_tumors_file)
    #     case_name = os.path.basename(current_dir)
    #
    #     fu_case, _ = load_nifti_data(fu_tumors_file)
    #     n_fu_nodes = np.unique(fu_case).size - 1
    #
    #     bl_tumors_file = glob(f'{current_dir}/BL_Scan_Tumors_unique_*')[0]
    #
    #     bl_case, _ = load_nifti_data(bl_tumors_file)
    #     n_bl_nodes = np.unique(bl_case).size - 1
    #
    #     pred_matches = match_2_cases(bl_case, fu_case, max_dilate_param=25)
    #
    #     # save_matching_graph(n_bl_nodes, n_fu_nodes, pred_matches, case_name, f'{current_dir}/pred_matching_graph.json')
    #
    #     bl_weights = []
    #     for i in range(1, n_bl_nodes + 1):
    #         bl_weights.append(int(centroid(bl_case == i)[-1]) + 1)
    #
    #     fu_weights = []
    #     for i in range(1, n_fu_nodes + 1):
    #         fu_weights.append(int(centroid(fu_case == i)[-1]) + 1)
    #
    #     draw_matching_graph(n_bl_nodes, n_fu_nodes, pred_matches, case_name, bl_weights=bl_weights,
    #                         fu_weights=fu_weights, saving_file_name=f'{current_dir}/pred_matching_graph.jpg')
    #
    #     df = pd.DataFrame(data=pred_matches, columns=['bl tumors', 'fu tumors'])
    #     writer = pd.ExcelWriter(f'{current_dir}/matching.xlsx')
    #     df.to_excel(writer, index=False)
    #     writer.save()
    #
    # fu_tumors_files = glob('/cs/casmip/rochman/Errors_Characterization/matching_for_richard/round_2/*/FU_Scan_Tumors_unique_*')
    # fu_tumors_files.sort()
    # # for fu_tumors_file in tqdm(fu_tumors_files):
    # #     f(fu_tumors_file)
    # process_map(f, fu_tumors_files, max_workers=os.cpu_count()-2)

    # ------------------------------------------------------------------------------------------------------------------

    # from shutil import copyfile
    #
    # excel_files = glob('/cs/casmip/rochman/Errors_Characterization/matching_for_richard/round_1_corrected/*/matching.xlsx')
    #
    # for excel_file in excel_files:
    #     copyfile(excel_file, excel_file.replace('/round_1_corrected/', '/round_1/').replace('/matching.xlsx',
    #                                                                                         '/matching_corrected.xlsx'))

    # ------------------------------------------------------------------------------------------------------------------

    # get_bl_and_fu_names = lambda pair_name_dir: os.path.basename(pair_name_dir).replace('BL_', '').split('_FU_')
    #
    # all_data_dirs = [dirname for dirname in glob(f'/mnt/sda1/aszeskin/Data_Followup_Full_29_4_2021/*') if os.path.isdir(dirname) and os.path.basename(dirname).startswith('BL_')]
    # all_data_dirs.sort()
    #
    # matching_dirs = [dirname for dirname in glob(f'/cs/casmip/rochman/Errors_Characterization/matching/*') if os.path.isdir(dirname) and os.path.basename(dirname).startswith('BL_')]
    #
    # all_data_fu = set(get_bl_and_fu_names(pair_name_dir)[1] for pair_name_dir in all_data_dirs)
    #
    # matching_bl = set(get_bl_and_fu_names(pair_name_dir)[0] for pair_name_dir in matching_dirs)
    #
    # missing_bl = matching_bl - all_data_fu
    #
    # print(missing_bl)

    # ------------------------------------------------------------------------------------------------------------------

    # get_bl_and_fu_names = lambda pair_name_dir: os.path.basename(pair_name_dir).replace('BL_', '').split('_FU_')
    #
    # dst_dir = '/cs/casmip/rochman/Errors_Characterization/data_to_recheck'
    # relevant_pairs_dirs = [dirname for dirname in glob(f'/cs/casmip/rochman/Errors_Characterization/matching/*') if os.path.isdir(dirname) and os.path.basename(dirname).startswith('BL_')]
    # all_data_dir = '/mnt/sda1/aszeskin/Data_Followup_Full_29_4_2021'
    # all_pairs_dirs = [dirname for dirname in glob(f'{all_data_dir}/*') if os.path.isdir(dirname) and os.path.basename(dirname).startswith('BL_')]
    #
    # for pair_dir in relevant_pairs_dirs:
    #     bl, fu = get_bl_and_fu_names(pair_dir)
    #
    #     # copy fu files
    #     current_fu_dir = f'{dst_dir}/{fu}'
    #     os.makedirs(current_fu_dir, exist_ok=True)
    #     src_dir = f'{all_data_dir}/{os.path.basename(pair_dir)}'
    #     os.symlink(f'{src_dir}/FU_Scan_CT.nii.gz', f'{current_fu_dir}/Scan_CT.nii.gz')
    #     os.symlink(f'{src_dir}/FU_Scan_Liver.nii.gz', f'{current_fu_dir}/Liver.nii.gz')
    #     os.symlink(f'{src_dir}/FU_Scan_Tumors.nii.gz', f'{current_fu_dir}/Tumors.nii.gz')
    #
    #     # copy bl files
    #     current_bl_dir = f'{dst_dir}/{bl}'
    #     os.makedirs(current_bl_dir, exist_ok=True)
    #     src_dir = [f for f in all_pairs_dirs if f.endswith(f'_FU_{bl}')][0]
    #     os.symlink(f'{src_dir}/FU_Scan_CT.nii.gz', f'{current_bl_dir}/Scan_CT.nii.gz')
    #     os.symlink(f'{src_dir}/FU_Scan_Liver.nii.gz', f'{current_bl_dir}/Liver.nii.gz')
    #     os.symlink(f'{src_dir}/FU_Scan_Tumors.nii.gz', f'{current_bl_dir}/Tumors.nii.gz')

    # ------------------------------------------------------------------------------------------------------------------

    bl_tumors_file = '/cs/casmip/rochman/Errors_Characterization/matching/BL_H_G_06_10_2019_FU_H_G_24_11_2019/BL_Scan_Tumors_unique_22_CC.nii.gz'
    fu_tumors_file = '/cs/casmip/rochman/Errors_Characterization/matching/BL_H_G_06_10_2019_FU_H_G_24_11_2019/FU_Scan_Tumors_unique_23_CC.nii.gz'
    matching_graph = '/cs/casmip/rochman/Errors_Characterization/matching/BL_H_G_06_10_2019_FU_H_G_24_11_2019/gt_matching_graph.json'
    # bl_liver_file = '/cs/casmip/rochman/Errors_Characterization/matching/BL_A_Ac_21_12_2020_FU_A_Ac_30_12_2020/BL_Scan_Liver.nii.gz'
    # fu_liver_file = '/cs/casmip/rochman/Errors_Characterization/matching/BL_A_Ac_21_12_2020_FU_A_Ac_30_12_2020/FU_Scan_Liver.nii.gz'
    #
    bl_tumors_case, file = load_nifti_data(bl_tumors_file)
    fu_tumors_case, _ = load_nifti_data(fu_tumors_file)

    bl_tumors_case[bl_tumors_case > 0] += 23

    n_bl_nodes, n_fu_nodes, edges, case_name, bl_weights, fu_weights, _, _, _, _ = load_matching_graph(matching_graph)

    for bl, fu in edges:
        bl_tumors_case[bl_tumors_case == bl + 23] = fu

    save(Nifti1Image(bl_tumors_case, file.affine), '/cs/casmip/rochman/Errors_Characterization/matching/BL_H_G_06_10_2019_FU_H_G_24_11_2019/bl_constracted_tumors_you_can_delete.nii.gz')


