from typing import Optional, Dict, Tuple, List

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from math import sqrt, ceil
import numpy as np
from scipy.stats import hmean


class read_compare_excel:
    def __init__(self, excel_path, data_name, sheet_name, id='ID', hues_to_add_to_file=None, hues_name=None):
        self.ID = id
        self.file_path = excel_path
        self.sheet_name = sheet_name
        self.excel_data = self.read_excel(self.file_path, data_name)

        self.hues_name = hues_name
        self.hues_was_added = False
        if hues_to_add_to_file is not None:
            assert self.hues_name is not None
            self.excel_data[self.hues_name] = hues_to_add_to_file
            self.hues_was_added = True

    def add_excel(self, excel_path, data_name, hues_to_add_to_file=None):
        new_excel_file = self.read_excel(excel_path, data_name)
        if self.hues_was_added:
            assert hues_to_add_to_file is not None
            new_excel_file[self.hues_name] = hues_to_add_to_file
        self.excel_data = pd.concat([self.excel_data, new_excel_file], ignore_index=True)

    def read_excel(self, excel_path, data_name):
        excel_file = pd.read_excel(excel_path, sheet_name=self.sheet_name)[:-5]
        excel_file.rename(columns=lambda c: c.replace(' - Isolated-Tumors', '').replace(' - Edges', ''), inplace=True)
        excel_file[self.ID] = data_name
        return excel_file

    @staticmethod
    def plot_sub_plot(data, x, y, subplotid, fig, plot_type='barplot', f1_scores: Optional[Tuple[str, str]] = None,
                      hues_was_added: bool = False, hues_name=None):

        ax1 = fig.add_subplot(*subplotid)
        # Draw a nested barplot by species and sex
        if f1_scores is None:

            if plot_type == 'barplot':
                if hues_was_added:
                    assert hues_name is not None
                    g = sns.barplot(hue=hues_name,
                                    data=data,
                                    x=x, y=y,
                                    ci="sd", palette="dark", alpha=.6, ax=ax1
                                    )
                else:
                    g = sns.barplot(
                        data=data,
                        x=x, y=y,
                        ci="sd", palette="dark", alpha=.6, ax=ax1
                    )
                for i, p in enumerate(g.patches):
                    height = p.get_height()
                    # g.text(p.get_x() + p.get_width() / 2.,
                    #        height / 2,
                    #        '{:1.2f}\n({:1.2f})'.format(height, (
                    #                g.lines[i].get_ydata()[1] - g.lines[i].get_ydata()[0]) / 2),
                    #        ha="center", fontsize=14, fontweight='bold')
                    current_mean = f'{height:.2f}'[1:]
                    current_std = f'{(g.lines[i].get_ydata()[1] - g.lines[i].get_ydata()[0]) / 2:.2f}'[1:]
                    g.text(p.get_x() + p.get_width() / 2.,
                           height / 2, f'{current_mean}\n({current_std})',
                           ha="center", fontsize=16, fontweight='bold')
            if plot_type == 'violinplot':
                g = sns.violinplot(data=data, x=x, y=y, bw=1, inner=None, cut=0)
                sns.stripplot(data=data, x=x, y=y, color="k", size=4)

            x = ' / '.join(x.split('/'))
            y = ' / '.join(y.split('/'))
            plt.xlabel(x, fontsize=20, fontweight='bold')
            plt.ylabel(y, fontsize=20, fontweight='bold')

        else:
            # data_helper = data.groupby(x).mean().reset_index()
            # g = sns.barplot(
            #     data=pd.concat([pd.DataFrame({x: [r[x], r[x]], y: [r[f1_scores[0]], r[f1_scores[1]]]}) for (_, r) in data_helper.iterrows()]),
            #     x=x, y=y,
            #     ci=None, palette="dark", alpha=.6, ax=ax1, estimator=hmean
            # )
            # for i, p in enumerate(g.patches):
            #     height = p.get_height()
            #     g.text(p.get_x() + p.get_width() / 2.,
            #            height / 2,
            #            'F1: {:1.2f}'.format(height),
            #            ha="center")
            # plt.ylim(0, 1)
            precision_label, recall_label = f1_scores
            # for _, case_data in data.groupby('Unnamed: 0'):
            #     ax1.plot(case_data[recall_label], case_data[precision_label], lw=1, alpha=0.75)

            if hues_was_added:
                assert hues_name is not None
                unique_hues = np.unique(data[hues_name])
            else:
                unique_hues = ['bla_bla']

            point_labels = []
            x_values = []
            y_values = []

            for i, hue in enumerate(unique_hues):

                if hues_was_added:
                    working_data = data[data[hues_name] == hue]
                else:
                    working_data = data

                mean_per_id = working_data.groupby(x).mean()
                mean_precision = mean_per_id[precision_label]
                mean_recall = mean_per_id[recall_label]
                # f1_s = hmean(np.vstack([mean_precision, mean_recall]))/50
                point_labels.append(list(range(1, np.unique(working_data[x]).size + 1)))

                # argsort = np.argsort(mean_recall.to_numpy())
                mean_precision = mean_precision.to_numpy()
                mean_recall = mean_recall.to_numpy()
                # f1_s = f1_s[argsort]
                # point_labels = np.array(point_labels)[argsort]

                # plt.errorbar(mean_recall, mean_precision, yerr=f1_s, fmt='ro-', ecolor='b', elinewidth=3, capsize=5)
                ax1.plot(mean_recall, mean_precision, 'ro-', color='b' if i == 0 else 'r', label=hue)
                # ax1.plot(mean_recall, mean_precision, color='b', lw=2, alpha=.8)

                x_values.append(mean_recall)
                y_values.append(mean_precision)

            all_xticks = sorted(list(set(v for s in x_values for v in s)))
            last_xtick_inserted = 0
            xticks = [all_xticks[0]]
            for k in range(1, len(all_xticks)):
                if np.abs(all_xticks[k] - all_xticks[last_xtick_inserted]) >= 0.01:
                    last_xtick_inserted = k
                    xticks.append(all_xticks[k])
            xlabels = [f'{i:.2f}' for i in xticks]
            plt.xticks(ticks=xticks, labels=xlabels, rotation=20)

            all_yticks = sorted(list(set(v for s in y_values for v in s)))
            last_ytick_inserted = 0
            yticks = [all_yticks[0]]
            for k in range(1, len(all_yticks)):
                if np.abs(all_yticks[k] - all_yticks[last_ytick_inserted]) >= 0.01:
                    last_ytick_inserted = k
                    yticks.append(all_yticks[k])
            # yticks = [all_yticks[0]] + list(all_yticks[1:-1:6]) + [all_yticks[-1]]
            ylabels = [f'{i:.2f}' for i in yticks]
            plt.yticks(ticks=yticks, labels=ylabels, rotation=25)

            plt.ylim(np.min(all_yticks) - .02, np.max(all_yticks) + .02)
            plt.xlim(np.min(all_xticks) - .02, np.max(all_xticks) + .02)

            lim = .005

            for j, point_labels_set in enumerate(point_labels):
                last_marked = 0
                for i, label in enumerate(point_labels_set):
                    if not (i > 0 and (np.abs(x_values[j][i] - x_values[j][last_marked]) < lim) and (
                            np.abs(y_values[j][i] - y_values[j][last_marked]) < lim)):
                        last_marked = i
                        ax1.annotate(label, (x_values[j][i], y_values[j][i]),
                                     (x_values[j][i] + 0.001, y_values[j][i] + 0.001), fontsize=20, fontweight='bold',
                                     color='b' if j == 0 else 'r')

            recall_label = ' / '.join(recall_label.split('/'))
            precision_label = ' / '.join(precision_label.split('/'))
            plt.xlabel(recall_label, fontsize=20, fontweight='bold')
            plt.ylabel(precision_label, fontsize=20, fontweight='bold')

            # plt.text((mean_recall.max() + mean_recall.min())/2, (mean_precision.max() + mean_precision.min())/2,
            #          '$Precision = \\frac{TP}{TP+FP}$')
            # plt.text(.9*mean_recall.max() + .1*mean_recall.min(), .9*mean_precision.max() + .1*mean_precision.min(),
            #          "$Precision = \\frac{TP}{TP+FP}$\n\n$Recall = \\frac{TP}{TP+FN}$", size=20,
            #          ha="center", va="center",
            #          bbox=dict(boxstyle="round",
            #                    ec=(1., 0.5, 0.5),
            #                    fc=(1., 0.8, 0.8),
            #                    )
            #          )

            # ax1.fill_between(mean_recall, mean_precision + f1_s, mean_precision - f1_s, color='grey', alpha=.2,
            #                 label=r'$\pm$ 1 std. dev.')

        # plt.ylabel(y, fontsize=10)
        if hues_was_added:
            plt.legend(prop={'size': 15})
        plt.xticks(fontsize=15, fontweight='bold')
        plt.yticks(fontsize=15, fontweight='bold')

    def plot(self, figure_num, key, measures,
             f1_scores: Optional[Dict[str, Tuple[str, str]]] = None,
             violinplot: Optional[List[str]] = None):

        if f1_scores is None:
            f1_scores = dict()

        if violinplot is None:
            violinplot = []

        sns.set_theme(style="whitegrid")

        # keys_list = [m for m in self.excel_data.keys() if m in measures]
        keys_list = measures

        for j, i in enumerate(range(0, len(keys_list), 2)):

            fig = plt.figure(f'{figure_num}_{j}')
            # For 16 graphs view
            fig.subplots_adjust(top=0.961, bottom=0.05, left=0.03, right=0.993)
            fig.suptitle(f'{key} ({j + 1})', fontsize=16, y=0.995)
            num_of_rows = 2  # ceil(len(keys_list)/4)
            num_of_columns = 1  # ceil(len(keys_list) / sqrt(len(keys_list)))
            plot_num = 1  # if i < len(keys_list) - 1 else 2
            # for k in (keys_list[i: i+2] if i < len(keys_list) - 1 else keys_list[i: i+1]):
            for k in keys_list[i: i + 2]:
                if k in f1_scores:
                    self.plot_sub_plot(self.excel_data, self.ID, k, (num_of_rows, num_of_columns, plot_num), fig,
                                       f1_scores=f1_scores[k], hues_was_added=self.hues_was_added,
                                       hues_name=self.hues_name)
                elif k in violinplot:
                    self.plot_sub_plot(self.excel_data, self.ID, k, (num_of_rows, num_of_columns, plot_num), fig,
                                       hues_was_added=self.hues_was_added, hues_name=self.hues_name,
                                       plot_type='violinplot')
                else:
                    self.plot_sub_plot(self.excel_data, self.ID, k, (num_of_rows, num_of_columns, plot_num), fig,
                                       hues_was_added=self.hues_was_added, hues_name=self.hues_name)
                plot_num += 1

                mng = plt.get_current_fig_manager()
                # mng.resize(*mng.window.maxsize())

            # Closing opened plots
            # for i in range(2, len(keys_list) + 1):
            #     plt.close(i)
            # mng = plt.get_current_fig_manager()
            # mng.window.showMaximized()

            # plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    def plot_4_barplots(self, figure_num, key):
        sns.set_theme(style="whitegrid")

        fig = plt.figure(figure_num)
        # For 4 graphs view
        # fig.subplots_adjust(top=0.88,bottom=0.11,left=0.04,right=0.97,wspace=0.12,hspace=0.2)
        fig.suptitle(key, fontsize=16)
        keys_list = list(self.excel_data.keys())[2:-1]
        keys_list = list([self.excel_data.keys()[2], self.excel_data.keys()[3], self.excel_data.keys()[4],
                          self.excel_data.keys()[-3], self.excel_data.keys()[-2]])
        num_of_rows = ceil(len(keys_list) / 2)
        num_of_columns = 2
        plot_num = 1

        for i in keys_list:
            self.plot_sub_plot(self.excel_data, self.ID, i, (num_of_rows, num_of_columns, plot_num), fig)
            plot_num += 1

        # Closing opened plots
        # for i in range(2, len(keys_list) + 1):
        #     plt.close(i)
        mng = plt.get_current_fig_manager()
        # mng.window.showMaximized()

        # plt.tight_layout()


# for i,key in enumerate(['diameter_0', 'diameter_5', 'diameter_10']):
#     print(i)
#     #test = read_compare_excel('test_results_of_FU.xlsx', 'One Scan', sheet_name=key)
#     #test.add_excel('test_results_th_6_With_BL_Scan_and_FU_Scan.xlsx', 'BL Scan FU Scan')
#     #test.add_excel('test_results_th_7_With_BL_GT_and_FU_Scan.xlsx', 'BL GT FU Scan')
#     #test.add_excel('test_results_th_5_With_BL_GT_Plus_Scan_and_FU_Scan.xlsx', 'BL GT + Scan FU Scan')
#     test = read_compare_excel('/cs/casmip/public/for_shalom/Tumor_segmentation/final_validation_set_-_R2U_NET_pairwise_35_pairs_-_GT_liver_with_BL_tumors/measures_results/tumors_measurements_-_th_1_-_final_validation_set_-_R2U_NET_pairwise_35_pairs_-_GT_liver_with_BL_tumors.xlsx', '1', sheet_name=key)
#     test.add_excel('/cs/casmip/public/for_shalom/Tumor_segmentation/final_validation_set_-_R2U_NET_pairwise_35_pairs_-_GT_liver_with_BL_tumors/measures_results/tumors_measurements_-_th_2_-_final_validation_set_-_R2U_NET_pairwise_35_pairs_-_GT_liver_with_BL_tumors.xlsx', '2')
#     test.add_excel('/cs/casmip/public/for_shalom/Tumor_segmentation/final_validation_set_-_R2U_NET_pairwise_35_pairs_-_GT_liver_with_BL_tumors/measures_results/tumors_measurements_-_th_3_-_final_validation_set_-_R2U_NET_pairwise_35_pairs_-_GT_liver_with_BL_tumors.xlsx', '3')
#     test.add_excel('/cs/casmip/public/for_shalom/Tumor_segmentation/final_validation_set_-_R2U_NET_pairwise_35_pairs_-_GT_liver_with_BL_tumors/measures_results/tumors_measurements_-_th_4_-_final_validation_set_-_R2U_NET_pairwise_35_pairs_-_GT_liver_with_BL_tumors.xlsx', '4')
#     test.add_excel('/cs/casmip/public/for_shalom/Tumor_segmentation/final_validation_set_-_R2U_NET_pairwise_35_pairs_-_GT_liver_with_BL_tumors/measures_results/tumors_measurements_-_th_5_-_final_validation_set_-_R2U_NET_pairwise_35_pairs_-_GT_liver_with_BL_tumors.xlsx', '5')
#     test.add_excel('/cs/casmip/public/for_shalom/Tumor_segmentation/final_validation_set_-_R2U_NET_pairwise_35_pairs_-_GT_liver_with_BL_tumors/measures_results/tumors_measurements_-_th_6_-_final_validation_set_-_R2U_NET_pairwise_35_pairs_-_GT_liver_with_BL_tumors.xlsx', '6')
#     test.add_excel('/cs/casmip/public/for_shalom/Tumor_segmentation/final_validation_set_-_R2U_NET_pairwise_35_pairs_-_GT_liver_with_BL_tumors/measures_results/tumors_measurements_-_th_7_-_final_validation_set_-_R2U_NET_pairwise_35_pairs_-_GT_liver_with_BL_tumors.xlsx', '7')
#     test.add_excel('/cs/casmip/public/for_shalom/Tumor_segmentation/final_validation_set_-_R2U_NET_pairwise_35_pairs_-_GT_liver_with_BL_tumors/measures_results/tumors_measurements_-_th_8_-_final_validation_set_-_R2U_NET_pairwise_35_pairs_-_GT_liver_with_BL_tumors.xlsx', '8')
#     test.add_excel('/cs/casmip/public/for_shalom/Tumor_segmentation/final_validation_set_-_R2U_NET_pairwise_35_pairs_-_GT_liver_with_BL_tumors/measures_results/tumors_measurements_-_th_9_-_final_validation_set_-_R2U_NET_pairwise_35_pairs_-_GT_liver_with_BL_tumors.xlsx', '9')
#     test.add_excel('/cs/casmip/public/for_shalom/Tumor_segmentation/final_validation_set_-_R2U_NET_pairwise_35_pairs_-_GT_liver_with_BL_tumors/measures_results/tumors_measurements_-_th_10_-_final_validation_set_-_R2U_NET_pairwise_35_pairs_-_GT_liver_with_BL_tumors.xlsx', '10')
#     test.add_excel('/cs/casmip/public/for_shalom/Tumor_segmentation/final_validation_set_-_R2U_NET_pairwise_35_pairs_-_GT_liver_with_BL_tumors/measures_results/tumors_measurements_-_th_11_-_final_validation_set_-_R2U_NET_pairwise_35_pairs_-_GT_liver_with_BL_tumors.xlsx', '11')
#     test.add_excel('/cs/casmip/public/for_shalom/Tumor_segmentation/final_validation_set_-_R2U_NET_pairwise_35_pairs_-_GT_liver_with_BL_tumors/measures_results/tumors_measurements_-_th_12_-_final_validation_set_-_R2U_NET_pairwise_35_pairs_-_GT_liver_with_BL_tumors.xlsx', '12')
#     test.add_excel('/cs/casmip/public/for_shalom/Tumor_segmentation/final_validation_set_-_R2U_NET_pairwise_35_pairs_-_GT_liver_with_BL_tumors/measures_results/tumors_measurements_-_th_13_-_final_validation_set_-_R2U_NET_pairwise_35_pairs_-_GT_liver_with_BL_tumors.xlsx', '13')
#     test.add_excel('/cs/casmip/public/for_shalom/Tumor_segmentation/final_validation_set_-_R2U_NET_pairwise_35_pairs_-_GT_liver_with_BL_tumors/measures_results/tumors_measurements_-_th_14_-_final_validation_set_-_R2U_NET_pairwise_35_pairs_-_GT_liver_with_BL_tumors.xlsx', '14')
#     test.add_excel('/cs/casmip/public/for_shalom/Tumor_segmentation/final_validation_set_-_R2U_NET_pairwise_35_pairs_-_GT_liver_with_BL_tumors/measures_results/tumors_measurements_-_th_15_-_final_validation_set_-_R2U_NET_pairwise_35_pairs_-_GT_liver_with_BL_tumors.xlsx', '15')
#     test.add_excel('/cs/casmip/public/for_shalom/Tumor_segmentation/final_validation_set_-_R2U_NET_pairwise_35_pairs_-_GT_liver_with_BL_tumors/measures_results/tumors_measurements_-_th_16_-_final_validation_set_-_R2U_NET_pairwise_35_pairs_-_GT_liver_with_BL_tumors.xlsx', '16')
#
#
#     # example = test.excel_data
#     test.plot_barplots(i,key)
#     test.plot_4_barplots(i+4,key)
# plt.show()

if __name__ == '__main__':

    dir_results = 'new_test_set_for_matching/all_test_set_measures_results_without_improving_registration_+_match_algo_v5'
    n_iters = 25
    exel_files = [f'/cs/casmip/rochman/Errors_Characterization/{dir_results}/matching_statistics_dilate_{i}.xlsx' for i
                  in range(1, n_iters + 1)]
    precisions, recalls, f1_scores = [], [], []
    for f in exel_files:
        df = pd.read_excel(f).iloc[:-5, :][['TP - Edges', 'FP - Edges', 'FN - Edges']]
        tp = df['TP - Edges'].sum()
        fp = df['FP - Edges'].sum()
        fn = df['FN - Edges'].sum()
        precisions.append(tp / (tp + fp))
        recalls.append(tp / (tp + fn))
        f1_scores.append(2 * precisions[-1] * recalls[-1] / (precisions[-1] + recalls[-1]))

    # precision_recall_curve = []
    # for i in range(n_iters):
    #     precision_recall_curve.append((recalls[i], precisions[i]))

    plt.plot(recalls, precisions, '-o')
    # plt.axis([0.4, 0.9, 0.4, 0.9])
    for i, (rec, pre) in enumerate(zip(recalls, precisions), start=1):
        plt.text(rec, pre, f'({rec * 100:.0f}, {pre * 100:.0f})')
        plt.text(rec, pre - 0.005, i)
    plt.text(0.675, 0.895, f'max f1_score={np.max(f1_scores)*100:.0f} (i={np.argmax(f1_scores) + 1})')
    plt.show()
    exit(0)

    for i, (key, measures, f1_scores) in enumerate([('Edges Statistics',
                                                     ['Precision',
                                                      'Recall',
                                                      'F1-Score',
                                                      'TP',
                                                      'FP',
                                                      'FN'],
                                                     {'F1-Score': ('Precision', 'Recall')}),

                                                    ('Isolation Statistics',
                                                     ['Precision/PPV',
                                                      'Recall/TPR/Sensitivity',
                                                      'F1-Score',
                                                      'Specificity/TNR',
                                                      'NPV',
                                                      'F1',
                                                      'Accuracy',
                                                      'TP',
                                                      'FP',
                                                      'FN',
                                                      'TN'],
                                                     {'F1-Score': ('Precision/PPV', 'Recall/TPR/Sensitivity'),
                                                      'F1': ('NPV', 'Specificity/TNR')})]):
        print(i)
        # test = read_compare_excel('/cs/casmip/rochman/Errors_Characterization/matching/matching_statistics_dilate_1.xlsx', '01 dilate', sheet_name=key)
        # # test = read_compare_excel('/cs/casmip/rochman/Errors_Characterization/matching/matching_statistics_dilate_2.xlsx', '02 dilate', sheet_name=key)
        # # test = read_compare_excel('/cs/casmip/rochman/Errors_Characterization/matching/matching_statistics_dilate_3.xlsx', '03 dilate', sheet_name=key)
        # # test = read_compare_excel('/cs/casmip/rochman/Errors_Characterization/matching/matching_statistics_dilate_4.xlsx', '04 dilate', sheet_name=key)
        # # test = read_compare_excel('/cs/casmip/rochman/Errors_Characterization/matching/matching_statistics_dilate_5.xlsx', '05 dilate', sheet_name=key)
        # # test = read_compare_excel('/cs/casmip/rochman/Errors_Characterization/matching/matching_statistics_dilate_6.xlsx', '06 dilate', sheet_name=key)
        # # test = read_compare_excel('/cs/casmip/rochman/Errors_Characterization/matching/matching_statistics_dilate_7.xlsx', '07 dilate', sheet_name=key)
        # # test = read_compare_excel('/cs/casmip/rochman/Errors_Characterization/matching/matching_statistics_dilate_8.xlsx', '08 dilate', sheet_name=key)
        # # test = read_compare_excel('/cs/casmip/rochman/Errors_Characterization/matching/matching_statistics_dilate_9.xlsx', '09 dilate', sheet_name=key)
        # # test = read_compare_excel('/cs/casmip/rochman/Errors_Characterization/matching/matching_statistics_dilate_10.xlsx', '10 dilate', sheet_name=key)
        # # test = read_compare_excel('/cs/casmip/rochman/Errors_Characterization/matching/matching_statistics_dilate_11.xlsx', '11 dilate', sheet_name=key)
        # test.add_excel('/cs/casmip/rochman/Errors_Characterization/matching/matching_statistics_dilate_2.xlsx', '02 dilate')
        # test.add_excel('/cs/casmip/rochman/Errors_Characterization/matching/matching_statistics_dilate_3.xlsx', '03 dilate')
        # test.add_excel('/cs/casmip/rochman/Errors_Characterization/matching/matching_statistics_dilate_4.xlsx', '04 dilate')
        # test.add_excel('/cs/casmip/rochman/Errors_Characterization/matching/matching_statistics_dilate_5.xlsx', '05 dilate')
        # test.add_excel('/cs/casmip/rochman/Errors_Characterization/matching/matching_statistics_dilate_6.xlsx', '06 dilate')
        # test.add_excel('/cs/casmip/rochman/Errors_Characterization/matching/matching_statistics_dilate_7.xlsx', '07 dilate')
        # test.add_excel('/cs/casmip/rochman/Errors_Characterization/matching/matching_statistics_dilate_8.xlsx', '08 dilate')
        # test.add_excel('/cs/casmip/rochman/Errors_Characterization/matching/matching_statistics_dilate_9.xlsx', '09 dilate')
        # test.add_excel('/cs/casmip/rochman/Errors_Characterization/matching/matching_statistics_dilate_10.xlsx', '10 dilate')
        # test.add_excel('/cs/casmip/rochman/Errors_Characterization/matching/matching_statistics_dilate_11.xlsx', '11 dilate')
        # test.add_excel('/cs/casmip/rochman/Errors_Characterization/matching/matching_statistics_dilate_12.xlsx', '12 dilate')
        # test.add_excel('/cs/casmip/rochman/Errors_Characterization/matching/matching_statistics_dilate_13.xlsx', '13 dilate')
        # test.add_excel('/cs/casmip/rochman/Errors_Characterization/matching/matching_statistics_dilate_14.xlsx', '14 dilate')
        # test.add_excel('/cs/casmip/rochman/Errors_Characterization/matching/matching_statistics_dilate_15.xlsx', '15 dilate')
        comparing_versions = False
        if not comparing_versions:
            # dir_results = 'matching/corrected_measures_results'
            # dir_results = 'matching/final_corrected_measures_results'
            # dir_results = 'matching/measures_results_after_improving_registration'
            # dir_results = 'matching/measures_results_after_improving_registration_with_liver_border_at_RANSAC'
            # dir_results = 'matching/measures_results_after_improving_registration_with_liver_border_at_RANSAC_and_ICP'
            # dir_results = 'corrected_segmentation_for_matching/measures_results_after_improving_registration_with_only_liver_border_at_RANSAC_and_ICP_no_tumors_+_match_algo_v2'
            # dir_results = 'corrected_segmentation_for_matching/measures_results_after_improving_registration_with_only_liver_border_at_RANSAC_and_ICP_no_tumors_+_match_algo_v3'
            # dir_results = 'corrected_segmentation_for_matching/measures_results_after_improving_registration_with_only_liver_border_at_RANSAC_and_only_tumors_at_ICP_+_match_algo_v3'
            # dir_results = 'corrected_segmentation_for_matching/measures_results_after_improving_registration_with_liver_for_RANSAC_and_ICP_if_liver_diff_less_300_else_liver_RANSAC_tumors_ICP+_match_algo_v4_with_i1_3_i10_2'
            # dir_results = 'corrected_segmentation_for_matching/measures_results_after_improving_registration_with_liver_for_RANSAC_and_ICP_if_liver_diff_less_300_else_liver_RANSAC_tumors_ICP+_match_algo_v4_with_i1_6_i10_3'
            # dir_results = 'corrected_segmentation_for_matching/measures_results_after_improving_registration_with_liver_for_RANSAC_and_ICP_if_liver_diff_less_300_else_liver_RANSAC_tumors_ICP+_match_algo_v4_with_i_5_j_3_k_0_05'
            # dir_results = 'new_test_set_for_matching/all_test_set_measures_results_after_improving_registration_with_liver_border_at_RANSAC_and_tumors_at_ICP_+_match_algo_v3'
            # dir_results = 'new_test_set_for_matching/all_test_set_measures_results_after_improving_registration_with_liver_border_at_RANSAC_and_tumors_and_liver_border_at_ICP_+_match_algo_v3'
            # dir_results = 'new_test_set_for_matching/all_test_set_measures_results_after_improving_registration_with_only_liver_border_at_RANSAC_and_ICP_no_tumors_+_match_algo_v3'
            # dir_results = 'new_test_set_for_matching/all_test_set_measures_results_after_improving_registration_with_only_liver_border_at_RANSAC_and_ICP_no_tumors_+_match_algo_v5'
            # dir_results = 'new_test_set_for_matching/all_test_set_measures_results_after_improving_registration_with_only_liver_border_at_RANSAC_and_ICP_no_tumors_+_match_algo_v5_+_adaptive_num_of_dilations'
            dir_results = 'new_test_set_for_matching/all_test_set_measures_results_without_improving_registration_+_match_algo_v5'
            # dir_results = 'matching/measures_results_after_improving_registration_with_only_liver_border_at_ICP_no_tumors_and_no_RANSAC'
            # test = read_compare_excel(f'/cs/casmip/rochman/Errors_Characterization/{dir_results}/matching_statistics_dilate_1.xlsx', 1,
            #     sheet_name=key, id='Number of Dilations', hues_to_add_to_file='old', hues_name='Version')
            test = read_compare_excel(
                f'/cs/casmip/rochman/Errors_Characterization/{dir_results}/matching_statistics_dilate_1.xlsx', 1,
                sheet_name=key, id='Number of Dilations')
            n_iters = 25
            for j in range(2, n_iters + 1):
                # test.add_excel(f'/cs/casmip/rochman/Errors_Characterization/{dir_results}/matching_statistics_dilate_{j}.xlsx', j,
                #                hues_to_add_to_file='old')
                test.add_excel(
                    f'/cs/casmip/rochman/Errors_Characterization/{dir_results}/matching_statistics_dilate_{j}.xlsx', j)

            # dir_results = 'matching/final_corrected_measures_results'
            # dir_results = 'matching/measures_results_after_improving_registration'
            # dir_results = 'matching/measures_results_after_improving_registration_with_liver_border_at_RANSAC'
            # dir_results = 'matching/measures_results_after_improving_registration_with_liver_border_at_RANSAC_and_ICP'
            # dir_results = 'matching/measures_results_after_improving_registration_with_only_liver_border_at_RANSAC_and_ICP_no_tumors'
            # dir_results = 'matching/measures_results_after_improving_registration_with_only_liver_border_at_ICP_no_tumors_and_no_RANSAC'
            # dir_results = 'corrected_segmentation_for_matching/measures_results_after_improving_registration_with_only_liver_border_at_RANSAC_and_ICP_no_tumors_+_match_algo_v3'
            # for j in range(1, n_iters + 1):
            #     test.add_excel(f'/cs/casmip/rochman/Errors_Characterization/{dir_results}/matching_statistics_dilate_{j}.xlsx', j,
            #                    hues_to_add_to_file='new')
        else:
            # todo only for comparing versions
            test = read_compare_excel(
                f'/cs/casmip/rochman/Errors_Characterization/new_test_set_for_matching/all_test_set_measures_results_after_improving_registration_with_liver_border_at_RANSAC_and_tumors_and_liver_border_at_ICP_+_match_algo_v3/matching_statistics_dilate_9.xlsx',
                1,
                sheet_name=key, id='Number of Dilations')
            test.add_excel(
                f'/cs/casmip/rochman/Errors_Characterization/new_test_set_for_matching/all_test_set_measures_results_after_improving_registration_with_only_liver_border_at_RANSAC_and_ICP_no_tumors_+_match_algo_v3/matching_statistics_dilate_9.xlsx',
                2)
            # test.add_excel(
            #     f'/cs/casmip/rochman/Errors_Characterization/corrected_segmentation_for_matching/measures_results_after_improving_registration_with_only_liver_border_at_RANSAC_and_ICP_no_tumors_+_match_algo_v3/matching_statistics_dilate_9.xlsx',
            #     3)
        # example = test.excel_data
        test.plot(i, key, measures=measures, f1_scores=f1_scores, violinplot=['TP', 'TN', 'FP', 'FN'])

        # test.plot_4_barplots(i+4, key)
    plt.show()

    # for i,(key, measures, hues) in enumerate([('All',
    #                                            ['Minimum Distance between tumors (mm)'],
    #                                            ['Is TC', 'Is FC', 'Is FUC'])
    #                                           ]):
    #     print(i)
    #     test = read_compare_excel('/cs/casmip/rochman/Errors_Characterization/matching/matching_measures_dilate_3.xlsx', '3 dilate', sheet_name=key)
    #     test.add_excel('/cs/casmip/rochman/Errors_Characterization/matching/matching_measures_dilate_4.xlsx', '4 dilate')
    #     test.add_excel('/cs/casmip/rochman/Errors_Characterization/matching/matching_measures_dilate_5.xlsx', '5 dilate')
    #     test.add_excel('/cs/casmip/rochman/Errors_Characterization/matching/matching_measures_dilate_6.xlsx', '6 dilate')
    #     test.add_excel('/cs/casmip/rochman/Errors_Characterization/matching/matching_measures_dilate_7.xlsx', '7 dilate')
    #
    #     # example = test.excel_data
    #     test.plot_barplots(i, key, measures=measures, hues=hues)
    #
    #     # test.plot_4_barplots(i+4, key)
    # plt.show()
    exit(0)

    train_df = pd.read_excel('/cs/casmip/rochman/Errors_Characterization/matching/matching_train_test_split.xlsx',
                             sheet_name='train set')
    train_df.rename(columns={'Unnamed: 0': 'Name'}, inplace=True)
    train_df = train_df[train_df['Name'].str.startswith('BL_')]
    train_names = train_df['Name'].to_list()

    df = pd.read_excel('/cs/casmip/rochman/Errors_Characterization/matching/matching_measures_dilate_25.xlsx')
    df.rename(columns={'Unnamed: 0': 'Name'}, inplace=True)
    df = df[df['Name'].str.startswith('BL_')]

    df = df[df.apply(func=lambda r: r['Name'].split('(')[0][:-1] in train_names, axis=1)]

    df['Diff in Diameter (mm)'] = np.abs(df['BL Tumor Diameter (mm)'] - df['FU Tumor Diameter (mm)'])


    # bl_num, fu_num = zip(*[name.split('_')[-1].split(',') for name in df['Name']])
    # bl_num = [int(n[1:]) for n in bl_num]
    # fu_num = [int(n[:-1]) for n in fu_num]
    # df['bl_tumor'] = bl_num
    # df['fu_tumor'] = fu_num

    def plot_scatter(df, x, y):
        ax = df[df['Is TC'] == 1].plot.scatter(x=x, y=y, c='g', label='TP')
        df[df['Is FUC'] == 1].plot.scatter(x=x, y=y, c='b', label='FN', ax=ax)
        df[df['Is FC'] == 1].plot.scatter(x=x, y=y, c='r', label='FP', ax=ax)

        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())


    def plot_corr_matrix(df, title=''):
        df = df[[c for c in df.columns if c not in ['Is TC', 'Is FC', 'Is FUC', 'Name']]]
        corr = df.corr()
        plt.figure()
        ax = sns.heatmap(
            corr, annot=True,
            vmin=-1, vmax=1, center=0,
            cmap=sns.diverging_palette(20, 220, n=200),
            square=True
        )
        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=45,
            horizontalalignment='right'
        )
        plt.title(title)

        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())


    def plot_diff_in_corr_matrix(df, a, b, title=''):

        def apply_helper(raw, columns):
            for c in columns:
                if raw[c] == 1:
                    return True
            return False

        if isinstance(a, str):
            df_a = df[df[a] == 1]
        else:
            df_a = df[df.apply(func=apply_helper, axis=1, columns=a)]
        if isinstance(b, str):
            df_b = df[df[b] == 1]
        else:
            df_b = df[df.apply(func=apply_helper, axis=1, columns=b)]
        df_a = df_a[[c for c in df_a.columns if c not in ['Is TC', 'Is FC', 'Is FUC', 'Name']]]
        df_b = df_b[[c for c in df_b.columns if c not in ['Is TC', 'Is FC', 'Is FUC', 'Name']]]
        corr_a = df_a.corr()
        corr_b = df_b.corr()
        corr = corr_a - corr_b

        plt.figure()
        ax = sns.heatmap(
            corr, annot=True,
            vmin=-1, vmax=1, center=0,
            cmap=sns.diverging_palette(20, 220, n=200),
            square=True
        )
        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=45,
            horizontalalignment='right'
        )
        plt.title(title)

        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())

        # ------------------------------------------------

        diff = np.abs(corr).sum(axis=1).to_frame().rename(columns={0: 'Sum Of ABS'})
        plt.figure()
        sns.heatmap(
            diff, annot=True,
            vmin=float(diff.min()), vmax=float(diff.max()), center=0,
            cmap=sns.diverging_palette(20, 220, n=200)
        )
        plt.title(title + ' - Sum Of ABS')
        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())


    def plot_kde(df, column):
        fig = plt.figure()
        succeeded = False
        try:
            df[df['Is TC'] == 1][column].plot.kde(c='g', label='TP')
            succeeded = True
        except:
            pass
        try:
            df[df['Is FC'] == 1][column].plot.kde(c='r', label='FP')
            succeeded = True
        except:
            pass
        try:
            df[df['Is FUC'] == 1][column].plot.kde(c='b', label='FN')
            succeeded = True
        except:
            pass

        if not succeeded:
            raise Exception()

        plt.xlabel(column)
        plt.legend()

        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())


    # x = 'Diff in Diameter (mm)'
    # plot_scatter(x=x, y='ASSD (mm)')
    # plot_scatter(x=x, y='Dice')
    # plot_scatter(x=x, y='HD (mm)')
    # plot_scatter(x=x, y='Minimum Distance between tumors (mm)')
    # plot_scatter(x=x, y='Distance between centroid of tumors (mm)')

    # df['dist_center/diff_diameter'] = df['Distance between centroid of tumors (mm)']/df['Diff in Diameter (mm)']
    df['diff_diameter/dist_center'] = df['Diff in Diameter (mm)'] / df['Distance between centroid of tumors (mm)']
    df['diff_diameter*dist_center'] = df['Diff in Diameter (mm)'] * df['Distance between centroid of tumors (mm)']
    df['diff_diameter*HD'] = df['Diff in Diameter (mm)'] * df['HD (mm)']
    df['diff_diameter*ASSD'] = df['Diff in Diameter (mm)'] * df['ASSD (mm)']
    df['diff_diameter*dist_center*HD*ASSD'] = df['Diff in Diameter (mm)'] * df[
        'Distance between centroid of tumors (mm)'] * df['ASSD (mm)'] * df['HD (mm)']

    original_TP = df["Is TC"].sum()
    original_FP = df["Is FC"].sum()
    original_FN = df["Is FUC"].sum()
    original_all = df.shape[0]

    df = df[df['Overlap (CC)'] == 0]
    # df = df[df['Overlap (CC)'] > 0]
    # df = df[df['Volume difference (%)'] > .75]
    # df = df[df['diff_diameter*dist_center*HD*ASSD'] > 7000]

    df = df[[c for c in df.columns if c not in ['Overlap (CC)',
                                                'Overlap with BL (%)',
                                                'Overlap with FU (%)',
                                                'Dice',
                                                'IOU']]]

    # df_ = df[['diff_diameter*dist_center*HD*ASSD', 'Is TC', 'Is FC', 'Is FUC']]

    plot_corr_matrix(df[df['Is TC'] == 1], 'TP')
    plot_corr_matrix(df[df['Is FC'] == 1], 'FP')
    plot_corr_matrix(df[df['Is FUC'] == 1], 'FN')
    plot_diff_in_corr_matrix(df, 'Is TC', 'Is FC', 'TP - FP')
    plot_diff_in_corr_matrix(df, 'Is TC', 'Is FUC', 'TP - FN')
    plot_diff_in_corr_matrix(df, 'Is FC', 'Is FUC', 'FP - FN')

    # plot_corr_matrix(df[(df['Is TC'] == 1) | (df['Is FUC'] == 1)], 'T')
    # plot_corr_matrix(df[df['Is FC'] == 1], 'F(P)')
    # plot_diff_in_corr_matrix(df, ['Is TC', 'Is FUC'], 'Is FC', 'T - F(P)')

    #
    # plot_scatter(df, x='BL Tumor Diameter (mm)', y='FU Tumor Diameter (mm)')
    plot_scatter(df, x='Diff in Diameter (mm)', y='Distance between centroid of tumors (mm)')
    plot_scatter(df, x='Diff in Diameter (mm)', y='HD (mm)')
    plot_scatter(df, x='Diff in Diameter (mm)', y='ASSD (mm)')
    # plot_scatter(df, x='Volume difference (%)', y='Minimum Distance between tumors (mm)')
    # plot_scatter(df, x='Volume difference (%)', y='Distance between centroid of tumors (mm)')
    # plot_scatter(df, x='Overlap with BL (%)', y='BL Tumor Diameter (mm)')
    # plot_scatter(df, x='Overlap with BL (%)', y='Dice')
    # plot_scatter(df, x='Overlap with FU (%)', y='Dice')
    # plot_scatter(df, x='Minimum Distance between tumors (mm)', y='ASSD (mm)')
    # plot_scatter(df, x='Diff in Diameter (mm)', y='Distance between centroid of tumors (mm)')
    # plot_scatter(df, x='Overlap (CC)', y='Distance between centroid of tumors (mm)')
    # plot_scatter(df, x='Overlap (CC)', y='HD (mm)')
    # plot_scatter(df, x='Overlap (CC)', y='ASSD (mm)')
    # plot_scatter(df, x='Overlap (CC)', y='Diff in Diameter (mm)')
    # plot_scatter(df, x='Overlap (CC)', y='Overlap with FU (%)')

    # plot_kde(df, 'Minimum Distance between tumors (mm)')
    # plot_kde(df, 'Distance between centroid of tumors (mm)')
    # plot_kde(df, 'HD (mm)')
    # plot_kde(df, 'ASSD (mm)')

    for c in df.columns:
        if c not in ['Is TC', 'Is FC', 'Is FUC', 'Name']:
            print(c, end=': ')
            try:
                plot_kde(df, c)
                print('succeeded')
            except:
                plt.close()
                print('not succeeded')

    TP = df["Is TC"].sum()
    FP = df["Is FC"].sum()
    FN = df["Is FUC"].sum()
    all = df.shape[0]
    print('\n\n###################################################################################')
    print(f'Original TP = {original_TP}/{original_all} = {100 * original_TP / original_all:.2f}%')
    print(f'Original FP = {original_FP}/{original_all} = {100 * original_FP / original_all:.2f}%')
    print(f'Original FN = {original_FN}/{original_all} = {100 * original_FN / original_all:.2f}%')
    print(f'Original Precision = {original_TP / (original_TP + original_FP):.3f}')
    print(f'Original Recall = {original_TP / (original_TP + original_FN):.3f}')
    print('###################################################################################')
    print(f'Percentage of the data: {all}/{original_all} = {100 * all / original_all:.2f}')
    print(f'TP = {TP}/{all} = {100 * TP / all:.2f}%')
    print(f'FP = {FP}/{all} = {100 * FP / all:.2f}%')
    print(f'FN = {FN}/{all} = {100 * FN / all:.2f}%')
    print(f'Precision = {TP / (TP + FP):.2f}')
    print(f'Recall = {TP / (TP + FN):.2f}')
    print('###################################################################################')

    plt.show()
