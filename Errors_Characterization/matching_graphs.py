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

