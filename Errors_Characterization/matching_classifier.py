from utils import *
from xgboost import XGBClassifier
from xgboost import plot_importance
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupKFold
import pandas as pd

if __name__ == '__main__':
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

    df['y'] = df['Is TC'] + 2*df['Is FUC'] + 3*df['Is FC']
    groups = df['Name'].to_list()

    X = df[[c for c in df.columns if c not in ['Name', 'Is TC', 'Is FUC', 'Is FC']]].to_numpy()
    y = X[:, -1]
    X = X[:, :-1]

    groups = ['_'.join(c for c in g.replace('BL_', '').split('_FU_')[0].split('_') if not c.isdigit()) for g in groups]

    y = (y < 3).astype(np.int)

    gkf = GroupKFold(n_splits=3)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=7)

    # fit model no training data
    model = XGBClassifier()
    model.fit(X, y)
    # plot feature importance
    plot_importance(model)
    plt.show()
