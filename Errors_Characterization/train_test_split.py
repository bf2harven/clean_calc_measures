from typing import List, Tuple, Optional, Generator
from kennard_stone import train_test_split as ks_train_test_split
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split as random_train_test_split
from utils import write_to_excel
import numpy as np
import pandas as pd
import os
from utils import *
from scipy.cluster.vq import kmeans2, whiten


# def train_test_split(registration_stats_file_path: str, standalone_stats_file_path: str, pairs_stats_file_path: str,
#                      test_set_size_per_distribution_parameter: float = 0.25, max_tumors_in_scan: int = 50,
#                      distribution_parameters: List[str] = ('dice', 'abs_liver_diff'),
#                      n_bins_to_divide_distribution: int = 10, seed: int = None,
#                      max_number_of_scans_per_patient_in_test_set: int = 6):
#
#     # loading registration, standalone and pairs statistics
#     relevant_scans = pd.read_excel(standalone_stats_file_path)
#     relevant_pairs = pd.read_excel(registration_stats_file_path)
#     relevant_pairs.drop('Unnamed: 0', axis='columns', inplace=True)
#     relevant_pairs = relevant_pairs[relevant_pairs['Valid'] == 1].reset_index(drop=True)
#     relevant_pairs = pd.merge(left=relevant_pairs, right=pd.read_excel(pairs_stats_file_path), how='left',
#                                        left_on='name', right_on='Unnamed: 0', left_index=True, sort=True, validate='one_to_many').drop('Unnamed: 0', axis='columns')
#
#     # merge pairs and standalone information
#     relevant_pairs = merge_pairs_and_standalone_information(relevant_pairs, relevant_scans)
#
#     # filter out complicated pairs (and all pairs of the same patient)
#     patients_not_to_appear_in_test_set = relevant_pairs[relevant_pairs['is_complicated'] == 1]['patient_name']
#     relevant_pairs_for_test = relevant_pairs[~relevant_pairs['patient_name'].isin(patients_not_to_appear_in_test_set)]
#
#     # filter out scans with too much tumors (and all scans of the same patient)
#     patients_not_to_appear_in_test_set = relevant_pairs_for_test[(relevant_pairs_for_test['bl_number_of_tumors'] > max_tumors_in_scan) | (relevant_pairs_for_test['fu_number_of_tumors'] > max_tumors_in_scan)]['patient_name']
#     relevant_pairs_for_test = relevant_pairs_for_test[~relevant_pairs_for_test['patient_name'].isin(patients_not_to_appear_in_test_set)]
#
#     # todo test
#     # filter out pairs of patients with too much scans
#     # temp = relevant_pairs_for_test[['fu_name', 'patient_name']].drop_duplicates().groupby('patient_name').size().reset_index(level=0)
#     # patients_not_to_appear_in_test_set = temp[temp[0] > max_number_of_scans_per_patient_in_test_set]['patient_name']
#     # relevant_pairs_for_test = relevant_pairs_for_test[~relevant_pairs_for_test['patient_name'].isin(patients_not_to_appear_in_test_set)]
#
#     test_patients = extract_test_patients(relevant_pairs_for_test, n_bins_to_divide_distribution,
#                                           distribution_parameters, test_set_size_per_distribution_parameter, seed)
#
#     test_pairs = relevant_pairs[relevant_pairs['patient_name'].isin(test_patients['patient_name'])]
#     train_pairs = relevant_pairs[~relevant_pairs['patient_name'].isin(test_patients['patient_name'])]
#
#     return train_pairs, test_pairs


def train_validation_split(train_pairs_file_path: str, test_set_size_per_distribution_parameter: float = 0.15,
                           distribution_parameters: List[str] = ('dice', 'abs_liver_diff'),
                           n_bins_to_divide_distribution: int = 10, seed: int = None):

    # loading data
    train_pairs = pd.read_excel(train_pairs_file_path)

    train_pairs.rename(columns={'Unnamed: 0': 'name'}, inplace=True)

    train_pairs = train_pairs[train_pairs['name'].str.startswith('BL_')]

    bl_names, fu_names = zip(*[pair_name.replace('BL_', '').split('_FU_') for pair_name in train_pairs['name'] if pair_name.startswith('BL')])
    patient_names = ['_'.join(c for c in name.split('_') if not c.isdigit()) for name in bl_names]

    # merging the information
    train_pairs.insert(loc=train_pairs.shape[1], column='patient_name', value=patient_names)
    train_pairs.insert(loc=train_pairs.shape[1], column='bl_name', value=bl_names)
    train_pairs.insert(loc=train_pairs.shape[1], column='fu_name', value=fu_names)

    validation_patients = extract_test_patients(train_pairs, n_bins_to_divide_distribution, distribution_parameters,
                          test_set_size_per_distribution_parameter, seed)

    validation_pairs = train_pairs[train_pairs['patient_name'].isin(validation_patients['patient_name'])]
    train_pairs = train_pairs[~train_pairs['patient_name'].isin(validation_patients['patient_name'])]

    return train_pairs, validation_pairs


def extract_test_patients(relevant_pairs_for_test, n_bins_to_divide_distribution, distribution_parameters,
                          test_set_size_per_distribution_parameter, seed):
    relevant_patients_for_test = relevant_pairs_for_test.groupby('patient_name').mean()
    relevant_patients_for_test.reset_index(level=0, inplace=True)
    test_patients = relevant_patients_for_test.copy()
    for param in distribution_parameters:
        current_param_test_patients = pd.DataFrame(columns=relevant_patients_for_test.columns)

        bins_limits = np.linspace(relevant_patients_for_test[param].min(), relevant_patients_for_test[param].max(),
                                  n_bins_to_divide_distribution + 1, endpoint=True)
        bins_limits[-1] += 1e-4

        for i in range(n_bins_to_divide_distribution):
            current_bin_relevant_patients = relevant_patients_for_test[
                (relevant_patients_for_test[param] >= bins_limits[i]) & (
                            relevant_patients_for_test[param] < bins_limits[i + 1])]
            _, current_bin_test_patients = random_train_test_split(current_bin_relevant_patients,
                                                                   test_size=test_set_size_per_distribution_parameter,
                                                                   shuffle=True, random_state=seed)
            current_param_test_patients = pd.concat([current_param_test_patients, current_bin_test_patients])

        test_patients = pd.merge(test_patients, current_param_test_patients, how='inner')
    return test_patients


def merge_pairs_and_standalone_information(pairs_info: pd.DataFrame, standalone_info: pd.DataFrame):
    if 'Unnamed: 0' in standalone_info.columns:
        standalone_info.rename(columns={'Unnamed: 0': 'name'}, inplace=True)

    # filter out overall statistic info
    pairs_info = pairs_info[~pairs_info['name'].isin(['mean', 'std', 'min', 'max', 'sum', 'median'])]
    standalone_info = standalone_info[~standalone_info['name'].isin(['mean', 'std', 'min', 'max', 'sum', 'median'])]

    bl_names, fu_names = zip(*[pair_name.replace('BL_', '').split('_FU_') for pair_name in pairs_info['name'] if pair_name.startswith('BL')])
    patient_names = ['_'.join(c for c in name.split('_') if not c.isdigit()) for name in bl_names]

    # separate standalone information for BL and FU
    bl_standalone_info = standalone_info[standalone_info['name'].isin(bl_names)].rename(columns=lambda col: f'bl_{col}')
    fu_standalone_info = standalone_info[standalone_info['name'].isin(fu_names)].rename(columns=lambda col: f'fu_{col}')

    # merging the information
    pairs_info.insert(loc=pairs_info.shape[1], column='patient_name', value=patient_names)
    pairs_info.insert(loc=pairs_info.shape[1], column='bl_name', value=bl_names)
    pairs_info.insert(loc=pairs_info.shape[1], column='fu_name', value=fu_names)

    pairs_info = pd.merge(left=pairs_info, right=bl_standalone_info, how='left',
                          on='bl_name', left_index=True, sort=True, validate='many_to_one')
    pairs_info = pd.merge(left=pairs_info, right=fu_standalone_info, how='left',
                          on='fu_name', left_index=True, sort=True, validate='many_to_one')

    return pairs_info


def train_test_split(df: pd.DataFrame, n_bins_to_divide_distribution: int, test_set_size_per_distribution_parameter: float,
                     distribution_parameters: Optional[List[str]] = None, seed: int = None,
                     extract_for_train: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:

    train_df = None
    if extract_for_train is not None:
        # move exclude samples to train set
        train_df = df[extract_for_train]
        df = df[~extract_for_train]

    if distribution_parameters is None:
        distribution_parameters = list(df.columns)

    df.reset_index(inplace=True)

    firs_iteration = True
    for param in distribution_parameters:
        current_param_test_df = pd.DataFrame(columns=df.columns)

        bins_limits = np.linspace(df[param].min(), df[param].max(),
                                  n_bins_to_divide_distribution + 1, endpoint=True)
        bins_limits[-1] += 1e-4

        for i in range(n_bins_to_divide_distribution):
            current_bin_df = df[(df[param] >= bins_limits[i]) & (df[param] < bins_limits[i + 1])]
            _, current_bin_test_df = random_train_test_split(current_bin_df,
                                                             test_size=test_set_size_per_distribution_parameter,
                                                             shuffle=True, random_state=seed)
            current_param_test_df = pd.concat([current_param_test_df, current_bin_test_df])

        if firs_iteration:
            test_df = current_param_test_df
            firs_iteration = False
        else:
            test_df = pd.merge(test_df, current_param_test_df, how='inner')

    test_df.set_index('index', inplace=True)
    df.set_index('index', inplace=True)

    train_df = pd.concat([train_df, df[~df.index.isin(test_df.index)]])

    return train_df, test_df


def k_means_train_test_split(data: np.ndarray, test_set_size_per_cluster: float, k_clusters: Optional[int] = 10,
                             iters: Optional[int] = 10, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """

    Splits the data into train and test sets using K-means clustering over the feature-space in order to result in the
    same feature distribution in both train and test set.

    Parameters
    ----------
    data : ndarray
        A ‘M’ by ‘N’ array of ‘M’ observations in ‘N’ dimensions or a length ‘M’ array of ‘M’ 1-D observations.
    test_set_size_per_cluster : float
        Should be between 0.0 and 1.0 and represent the proportion of the test-set to include in each cluster test split.
    k_clusters : int, optional
        The number of clusters to form as well as the number of centroids to generate. Default value is set to 10 clusters.
    iters : int, optional
        Number of iterations of the k-means algorithm to run.
    seed : {None, int, numpy.random.Generator, numpy.random.RandomState}, optional
        Seed for initializing the pseudo-random number generator. If seed is None (or numpy.random),
        the numpy.random.RandomState singleton is used. If seed is an int, a new RandomState instance is used,
        seeded with seed. If seed is already a Generator or RandomState instance then that instance is used.
        The default is None.

    Returns
    -------
    train_data : List[int]
        A list of indexes of the observations in data to consider them as the train-set.
    train_data : List[int]
        A list of indexes of the observations in data to consider them as the test-set.
    """

    def _kpp(data, k, rng):
        """ Picks k points in the data based on the kmeans++ method.
        Parameters
        ----------
        data : ndarray
            Expect a rank 1 or 2 array. Rank 1 is assumed to describe 1-D
            data, rank 2 multidimensional data, in which case one
            row is one observation.
        k : int
            Number of samples to generate.
        rng : `numpy.random.Generator` or `numpy.random.RandomState`
            Random number generator.
        Returns
        -------
        init : ndarray
            A 'k' by 'N' containing the initial centroids.
        References
        ----------
        .. [1] D. Arthur and S. Vassilvitskii, "k-means++: the advantages of
           careful seeding", Proceedings of the Eighteenth Annual ACM-SIAM Symposium
           on Discrete Algorithms, 2007.
        """

        def rng_integers(gen, low, high=None, size=None, dtype='int64',
                         endpoint=False):
            """
            Return random integers from low (inclusive) to high (exclusive), or if
            endpoint=True, low (inclusive) to high (inclusive). Replaces
            `RandomState.randint` (with endpoint=False) and
            `RandomState.random_integers` (with endpoint=True).
            Return random integers from the "discrete uniform" distribution of the
            specified dtype. If high is None (the default), then results are from
            0 to low.
            Parameters
            ----------
            gen : {None, np.random.RandomState, np.random.Generator}
                Random number generator. If None, then the np.random.RandomState
                singleton is used.
            low : int or array-like of ints
                Lowest (signed) integers to be drawn from the distribution (unless
                high=None, in which case this parameter is 0 and this value is used
                for high).
            high : int or array-like of ints
                If provided, one above the largest (signed) integer to be drawn from
                the distribution (see above for behavior if high=None). If array-like,
                must contain integer values.
            size : array-like of ints, optional
                Output shape. If the given shape is, e.g., (m, n, k), then m * n * k
                samples are drawn. Default is None, in which case a single value is
                returned.
            dtype : {str, dtype}, optional
                Desired dtype of the result. All dtypes are determined by their name,
                i.e., 'int64', 'int', etc, so byteorder is not available and a specific
                precision may have different C types depending on the platform.
                The default value is np.int_.
            endpoint : bool, optional
                If True, sample from the interval [low, high] instead of the default
                [low, high) Defaults to False.
            Returns
            -------
            out: int or ndarray of ints
                size-shaped array of random integers from the appropriate distribution,
                or a single such random int if size not provided.
            """
            if isinstance(gen, Generator):
                return gen.integers(low, high=high, size=size, dtype=dtype,
                                    endpoint=endpoint)
            else:
                if gen is None:
                    # default is RandomState singleton used by np.random.
                    gen = np.random.mtrand._rand
                if endpoint:
                    # inclusive of endpoint
                    # remember that low and high can be arrays, so don't modify in
                    # place
                    if high is None:
                        return gen.randint(low + 1, size=size, dtype=dtype)
                    if high is not None:
                        return gen.randint(low, high=high + 1, size=size, dtype=dtype)

                # exclusive
                return gen.randint(low, high=high, size=size, dtype=dtype)

        dims = data.shape[1] if len(data.shape) > 1 else 1
        init = np.ndarray((k, dims))

        for i in range(k):
            if i == 0:
                init[i, :] = data[rng_integers(rng, data.shape[0])]

            else:
                D2 = cdist(init[:i, :], data, metric='sqeuclidean').min(axis=0)
                probs = D2 / D2.sum()
                cumprobs = probs.cumsum()
                r = rng.uniform()
                init[i, :] = data[np.searchsorted(cumprobs, r)]

        return init

    def check_random_state(seed):
        """Turn `seed` into a `np.random.RandomState` instance.
        Parameters
        ----------
        seed : {None, int, `numpy.random.Generator`,
                `numpy.random.RandomState`}, optional
            If `seed` is None (or `np.random`), the `numpy.random.RandomState`
            singleton is used.
            If `seed` is an int, a new ``RandomState`` instance is used,
            seeded with `seed`.
            If `seed` is already a ``Generator`` or ``RandomState`` instance then
            that instance is used.
        Returns
        -------
        seed : {`numpy.random.Generator`, `numpy.random.RandomState`}
            Random number generator.
        """
        import numbers
        if seed is None or seed is np.random:
            return np.random.mtrand._rand
        if isinstance(seed, (numbers.Integral, np.integer)):
            return np.random.RandomState(seed)
        if isinstance(seed, np.random.RandomState):
            return seed
        try:
            # Generator is only available in numpy >= 1.17
            if isinstance(seed, np.random.Generator):
                return seed
        except AttributeError:
            pass
        raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                         ' instance' % seed)

    data = whiten(data)

    rng = check_random_state(seed)

    init_points = _kpp(data, k_clusters, rng)

    _, labels = kmeans2(data, k=init_points, iter=iters, minit='matrix', missing='raise')
    print()

    # todo complete the implementation of the function


if __name__ == '__main__':

    # #
    # # train_pairs, test_pairs = train_test_split(registration_stats_file_path, standalone_stats_file_path,
    # #                                            pairs_stats_file_path, test_set_size_per_distribution_parameter=0.30,
    # #                                            seed=62)
    #
    # columns = ['name', 'dice', 'assd', 'hd', 'abs_liver_diff',
    #            'liver_mutual_information', 'ct_mutual_information',
    #            'volume_change (FU/BL)', 'n_exist_tumors', 'n_brand_new_tumors', 'n_new_tumors',
    #            'n_brand_disappear_tumors', 'n_disappear_tumors', 'n_merges', 'n_splits', 'is_complicated', 'BL_number_of_tumors',
    #            'FU_number_of_tumors', 'bl_total_tumor_volume (CC)', 'fu_total_tumor_volume (CC)',
    #            'bl_number_of_tumors_in_5_mm_from_liver_border', 'fu_number_of_tumors_in_5_mm_from_liver_border',
    #            'bl_mean_tumor_diameter', 'fu_mean_tumor_diameter', 'bl_std_tumor_diameter', 'fu_std_tumor_diameter',
    #            'bl_min_tumor_diameter', 'fu_min_tumor_diameter', 'bl_max_tumor_diameter', 'fu_max_tumor_diameter']
    #
    # dir_name = 'data'
    # os.makedirs(dir_name, exist_ok=True)
    #
    # # train_file = f'{dir_name}/train_pairs.xlsx'
    # # test_file = f'{dir_name}/test_pairs.xlsx'
    #
    # # writer = pd.ExcelWriter(train_file, engine='xlsxwriter')
    # # write_to_excel(train_pairs, writer, columns, 'name')
    # # writer.save()
    # #
    # # writer = pd.ExcelWriter(test_file, engine='xlsxwriter')
    # # write_to_excel(test_pairs, writer, columns, 'name')
    # # writer.save()
    #
    # # -----------------------------------------------------------------------------------------------------------------
    #
    # train_pairs_file = '/cs/casmip/rochman/Errors_Characterization/data/train_pairs_saved.xlsx'
    #
    # train_pairs, validation_pairs = train_validation_split(train_pairs_file,
    #                                                        test_set_size_per_distribution_parameter=0.15, seed=2)
    #
    # train_file_after_validation_splitting = f'{dir_name}/train_pairs_after_validation_splitting.xlsx'
    # validation_file = f'{dir_name}/validation_pairs.xlsx'
    #
    # writer = pd.ExcelWriter(train_file_after_validation_splitting, engine='xlsxwriter')
    # write_to_excel(train_pairs, writer, columns, 'name')
    # writer.save()
    #
    # writer = pd.ExcelWriter(validation_file, engine='xlsxwriter')
    # write_to_excel(validation_pairs, writer, columns, 'name')
    # writer.save()
    #
    # # print()
    # #
    # #
    # # for seed in range(201):
    # #     train_pairs, validation_pairs = train_validation_split(train_pairs_file,
    # #                                                      test_set_size_per_distribution_parameter=0.15, seed=seed)
    # #
    # #     n_train_patients = np.unique(train_pairs["patient_name"]).size
    # #     n_test_patients = np.unique(validation_pairs["patient_name"]).size
    # #     n_patients = n_train_patients + n_test_patients
    # #     patients_ratio = n_test_patients / n_patients
    # #
    # #     n_train_pairs = train_pairs.shape[0]
    # #     n_test_pairs = validation_pairs.shape[0]
    # #     n_pairs = n_train_pairs + n_test_pairs
    # #     pairs_ratio = n_test_pairs / n_pairs
    # #
    # #     print(f'n_test_patients/n_patients: {n_test_patients}/{n_patients} = {patients_ratio}')
    # #     print(f'n_test_pairs/n_pairs: {n_test_pairs}/{n_pairs} = {pairs_ratio}')
    # #     print('------------------------------------------')
    # #     if (0.09 <= patients_ratio <= 0.11) and (0.09 <= pairs_ratio <= 0.11):
    # #         print(f'last_seed={seed}')
    # #         break
    # #     # if (0.09 <= patients_ratio <= 0.11) and (0.09 <= pairs_ratio <= 0.11):
    # #     #     print(f'seed={seed}')
    # #     #     print(f'n_test_patients/n_patients: {n_test_patients}/{n_patients} = {patients_ratio}')
    # #     #     print(f'n_test_pairs/n_pairs: {n_test_pairs}/{n_pairs} = {pairs_ratio}')
    # #     #     print('------------------------------------------')
    # #     # else:
    # #     #     print(f'seed={seed} is not a seed that achieves the goal')
    # # else:
    # #     print('there is not a seed that achieves the goal')

    # splitting test and validation matching set
    isolation_statistics_df = pd.read_excel('/cs/casmip/rochman/Errors_Characterization/matching/matching_statistics_dilate_25.xlsx', sheet_name=1)
    matching_measures_df = pd.read_excel('/cs/casmip/rochman/Errors_Characterization/matching/matching_measures_dilate_25.xlsx')

    # loading and treating isolation_statistics
    isolation_statistics_df.rename(columns={'Unnamed: 0': 'name'}, inplace=True)
    isolation_statistics_df = isolation_statistics_df[['name',
                                                       'Positive-Rate - Isolated-Tumors',
                                                       'num BL tumors',
                                                       'num FU tumors']]
    isolation_statistics_df = isolation_statistics_df[isolation_statistics_df['name'].str.startswith('BL_')]
    isolation_statistics_df.set_index(keys='name', inplace=True)

    # loading and treating matching_measures
    matching_measures_df.rename(columns={'Unnamed: 0': 'name'}, inplace=True)
    matching_measures_df = matching_measures_df[(matching_measures_df['Is TC'] == 1) | (matching_measures_df['Is FUC'] == 1)]
    matching_measures_df['Diff in Diameter (mm)'] = np.abs(matching_measures_df['BL Tumor Diameter (mm)'] - matching_measures_df['FU Tumor Diameter (mm)'])
    matching_measures_df = matching_measures_df[matching_measures_df['name'].str.startswith('BL_')]
    matching_measures_df['name'] = matching_measures_df['name'].apply(lambda n: n.split('(')[0][:-1])
    matching_measures_df = matching_measures_df[['name',
                                                 'Minimum Distance between tumors (mm)',
                                                 'Distance between centroid of tumors (mm)',
                                                 'Overlap (CC)',
                                                 'Overlap with BL (%)',
                                                 'Overlap with FU (%)',
                                                 'HD (mm)',
                                                 'ASSD (mm)',
                                                 'Dice',
                                                 'IOU',
                                                 'Volume difference (%)',
                                                 'Diff in Diameter (mm)']]
    num_of_matches = matching_measures_df.groupby('name').count()[['HD (mm)']]
    num_of_matches.rename(columns={'HD (mm)': 'Num of Matches'}, inplace=True)
    matching_measures_df = matching_measures_df.groupby('name').mean()
    matching_measures_df.rename(columns=lambda c: f'Mean {c}', inplace=True)

    df = pd.concat([isolation_statistics_df, num_of_matches,  matching_measures_df], axis=1)

    # for seed in range(50):
    #     train_df, test_df = train_test_split(df, n_bins_to_divide_distribution=10,
    #                                          test_set_size_per_distribution_parameter=0.8, seed=seed,
    #                                          extract_for_train=df.isna().any(axis=1))
    #     if test_df.shape[0] == 10:
    #         print('##############################')
    #         print(f'seed = {seed:02d}: test tile is {test_df.shape[0]}/{df.shape[0]} = {test_df.shape[0]/df.shape[0]:.4f}')

    # old_way = True
    #
    # if old_way:
    #
    #     train_df, test_df = train_test_split(df, n_bins_to_divide_distribution=10,
    #                                          test_set_size_per_distribution_parameter=0.8, seed=42,
    #                                          extract_for_train=df.isna().any(axis=1))
    #
    #     train_df = train_df.reset_index().rename(columns={'index': 'name'})
    #     test_df = test_df.reset_index().rename(columns={'index': 'name'})
    #
    #     train_df = sort_dataframe_by_key(train_df, column='name', key=pairs_sort_key)
    #     test_df = sort_dataframe_by_key(test_df, column='name', key=pairs_sort_key)
    #
    # # else:
    # #
    # #     # extract_for_train = df.isna().any(axis=1)
    # #     # extract_for_train_df = df[extract_for_train]
    # #     # df = df[~extract_for_train]
    # #
    # #
    # #     names = matching_measures_df['name'].to_list()
    # #     data = matching_measures_df.set_index('name').to_numpy().astype(float)
    # #     # data = df.to_numpy().astype(float)
    # #
    # #     train_X, test_X = k_means_train_test_split(data, test_set_size_per_cluster=0.2, k_clusters=10, iters=10, seed=42)
    # #     print()
    #
    # else:
    extract_for_train = df.isna().any(axis=1)
    extract_for_train_df = df[extract_for_train]
    df = df[~extract_for_train]

    train_df, test_df = ks_train_test_split(df, test_size=10)

    train_df = pd.concat([train_df, extract_for_train_df])

    train_df = train_df.reset_index().rename(columns={'index': 'name'})
    test_df = test_df.reset_index().rename(columns={'index': 'name'})

    train_df = sort_dataframe_by_key(train_df, column='name', key=pairs_sort_key)
    test_df = sort_dataframe_by_key(test_df, column='name', key=pairs_sort_key)

    # train_X = train_df.drop(columns=['name']).to_numpy().astype(float)
    # test_X = test_df.drop(columns=['name']).to_numpy().astype(float)
    #
    # mu_train = np.nanmean(train_X, axis=0)
    # mu_test = np.nanmean(test_X, axis=0)
    #
    # # cov_train = np.cov(train_X, rowvar=False)
    # cov_train = np.asarray(np.ma.cov(np.ma.masked_invalid(train_X), rowvar=False))
    # cov_test = np.asarray(np.ma.cov(np.ma.masked_invalid(test_X), rowvar=False))
    #
    # print(f'Distance between mean: {np.linalg.norm(mu_test - mu_train)}')
    # print(f'Distance between covariance matrix: {np.linalg.norm(cov_test.flatten() - cov_train.flatten())}')

    writer = pd.ExcelWriter(f'matching/matching_train_test_split.xlsx', engine='xlsxwriter')
    write_to_excel(train_df, writer, train_df.columns, column_name_as_index='name', sheet_name='train set')
    write_to_excel(test_df, writer, test_df.columns, column_name_as_index='name', sheet_name='test set')
    writer.save()

