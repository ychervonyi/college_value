import pandas
from collections import OrderedDict
import numpy as np
import os
import glob


class Feature(object):
    def __init__(self, name, datatype='float64', onehot=False, replace_with=-1.0):
        self._name = name
        self._datatype = datatype
        self._onehot = onehot
        self._replace_with = replace_with

    @property
    def name(self):
        return self._name

    @property
    def datatype(self):
        return self._datatype

    @property
    def onehot(self):
        return self._onehot

    @property
    def replace_with(self):
        return self._replace_with


def compute_average_earnings(dataframe):
    # Average earnings by male after 6 years of entry
    male_earnings = dataframe['MN_EARN_WNE_MALE1_P6'].values
    # Male count
    male_count = dataframe['COUNT_WNE_MALE1_P6'].values
    # Average earnings by male after 6 years of entry
    female_earnings = dataframe['MN_EARN_WNE_MALE0_P6'].values
    # Female count
    female_count = dataframe['COUNT_WNE_MALE0_P6'].values

    # Death rate
    # print_death_rate(df, year)

    # Average earnings
    average_earnings = (male_earnings * male_count + female_earnings * female_count)/(male_count + female_count)
    return average_earnings


def df_feature_into_onehot(df, feature_name):
    if feature_name not in df:
        return df, None
    # Get index of the feature
    one_hot_index = df.columns.get_loc(feature_name)
    # Split dataframe into 3
    dfs = np.split(df, [one_hot_index, one_hot_index + 1], axis=1)
    # Transform feature into str type
    f = dfs[1].astype(np.int8).astype(str)
    # Transform feature into onehot
    f_onehot = pandas.get_dummies(f, prefix='', prefix_sep='')
    n_columns = len(f_onehot.columns)
    one_hot_columns = []
    for i in range(n_columns):
        one_hot_columns.append('%s_%s' % (feature_name, i))
    f_onehot.columns = one_hot_columns
    return pandas.concat([dfs[0], f_onehot, dfs[2]], axis=1), len(f_onehot.columns)


def get_data(features, path):
    df_year = OrderedDict({})

    cwd = os.getcwd()
    files = sorted(glob.glob(cwd + path))
    # files = [cwd + '/CollegeScorecard_Raw_Data/MERGED2013_14_PP.csv']
    for file in files:
        # Extract year from file name
        year = file.split('/')[-1][6:-7]
        print("Year: %s" % year)

        # Consider only data from 2xxx (previous years don't have the data we are interested in)
        if year.startswith('1'):
            continue
        df = pandas.read_csv(file, delimiter=',', error_bad_lines=False)

        feature_names = []
        for feature in features:
            feature_names.append(feature.name)
        df = df.filter(feature_names, axis=1)
        # Missing values and types
        for feature in features:
            df[feature.name].replace('PrivacySuppressed', feature.replace_with, inplace=True)
            df[feature.name].fillna(feature.replace_with, inplace=True)
            # df[feature.name].astype(feature.datatype)

        # Filter missing values
        for feature in features:
            df = df[df[feature.name] != feature.replace_with]

        for feature in features:
            df[feature.name] = df[feature.name].astype(feature.datatype)

        df = df.assign(EARNINGS=compute_average_earnings(df))

        # Encode into one hots
        for feature in features:
            if feature.onehot:
                df = df_feature_into_onehot(df, feature.name)

        # Skip years with no data
        if df.values.size > 0:
            df_year[year] = df
    return df_year


if __name__ == '__main__':
    earnings_features = [Feature(name='MN_EARN_WNE_MALE1_P6'),
                         Feature(name='COUNT_WNE_MALE1_P6'),
                         Feature(name='MN_EARN_WNE_MALE0_P6'),
                         Feature(name='COUNT_WNE_MALE0_P6')]
    features_student = [Feature(name='SAT_AVG_ALL'),
                        Feature(name='SATVRMID'),
                        Feature(name='SATMTMID')]
    college_features = [Feature(name='INSTNM', datatype='str', replace_with='')]
    features_all = earnings_features + features_student + college_features
    print("Reading data...")
    dataset = get_data(features_all, path='/CollegeScorecard_Raw_Data/*.csv')