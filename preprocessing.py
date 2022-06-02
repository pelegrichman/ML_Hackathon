from typing import Set, Dict

import pandas as pd
import numpy as np
import json
# from pandas_profiling import ProfileReport

# import generes_data
# import productions_data
# import keywords_data
# import cast_data
# import crew_data
# import collections_data

from itertools import chain
from collections import Counter
from pandas import DataFrame
from sklearn.impute import SimpleImputer
import ast

# _------------------------------------------------------
# ----- HARDCODE any encoding so we can handle well new ones (ignore them)
# _------------------------------------------------------

def add_missing_encodings(encodings: pd.DataFrame, cols_names: set):
    cols = set(encodings.columns)
    missing = cols_names.difference(cols)
    for col in missing:
        encodings[col] = 0
    return encodings


def parse_column(data: str):
    if not data:
        return data

    try:
        # d = data.replace('\'', '"')  # might need a better solution
        # obj = json.loads(d)
        obj = ast.literal_eval(data)
        return pd.DataFrame([obj]) if isinstance(obj, dict) else pd.DataFrame(obj)
    except Exception as e:
        return data


def get_val(obj, v, default=None, first=False):
    try:
        return obj[v] if not first else obj[v][0]
    except Exception as e:
        return default


def list_col_name(dataframe, col):
    try:
        return dataframe[col].tolist()
    except Exception as e:
        return np.NAN


SEED = 13

HIS_DIAG_DICT: Dict[str, int] = {"INTRADUCT AND LOBULAR CARCINOMA IN SITU": 0,
                 "INTRADUCTAL CARCINOMA": 0,
                 "LOBULAR CARCINOMA IN SITU": 0,
                 "INFILTRATING DUCT CARCINOMA": 1,
                 "LOBULAR INFILTRATING CARCINOMA": 1,
                 "INFILTRATING DUCTULAR CARCINOMA WITH DCIS": 1}

HER2_DICT: Dict[str, int] = {'NEGATIVE PER FISH': -1,
             'NEG': -1, 'neg': -1, '-': -1, 'FISH -': -1, '0': -1,
             '1': -1, '2': -1, '(-)': -1, 'Neg': -1, '+2 FISH(-)': -1, '+2 Fish NEG': -1,
            'fish neg': -1, 'negative': -1, '2+': -1, 'NEGATIVE': -1, '+2 FISH-neg': -1,
            'HER2/NEU-NEG': -1, 'Neg (FISH non amplified)': -1, 'Neg vs +2': -1, '/NEU-NEG': -1,
             '2 non amplified': 0, ' ': 0, '': 0, '?': 0,
             'FISH POS': 1, '+2 FISH-pos': 1, '+': 1, 'POS': 1, 'pos': 1, 'Pos': 1,
            'POSITIVE': 1, 'positive': 1, '/NEU-POSITIVE+3': 1, '3': 1, '+3': 1,
            '+2, FISH חיובי': 1, '100': 1, 'POSITIVE +3': 1, 'FISH+': 1, 'Positive by FISH': 1,
             'POS +3': 1, '+1, +2 FISH pos at ': 1, 'surgery': 1, '+2 IHC': 1}

HIS_DEG_DICT: Dict[str, int] = {'GX': 0,
                                'Nu': 0,
                                'G1': 1,
                                'G2': 2,
                                'G3': 3,
                                'G4': 4}

LVI_DICT: Dict[str, int] = {'none': 0,
                            '-': 0,
                            '(-)': 0,
                            'No': 0,
                            'NO': 0,
                            'no': 0,
                            'neg': 0,
                            'not': 0,
                            'MICROPAPILLARY VARIANT': 1,
                            '+': 1,
                            'extensive': 1,
                            'yes': 1,
                            '(+)': 1,
                            'pos': 1}

LYM_PEN_DICT: Dict[str, int] = {'L0': 0,
                            'Nu': 0,
                            'L1': 1,
                            'L2': 1,
                            'LI': 1}

STAGE_DICT: Dict[str, int] = {'Stage0': 0,
                              'Stage0a': 1,
                              'Stage0is': 2,
                              'Stage1': 3,
                              'Stage1a': 4,
                              'Stage1b': 5,
                              'Stage1c': 6,
                              'Stage2': 7,
                              'Stage2a': 8,
                              'Stage2b': 9,
                              'Stage3': 10,
                              'Stage3a': 11,
                              'Stage3b': 12,
                              'Stage3c': 13,
                              'Stage4': 14}
def to_names_list(sub_df, allowed=None):
    if isinstance(sub_df, pd.DataFrame) and not sub_df.empty:
        if 'name' in sub_df.columns:
            vals = set(sub_df['name'].tolist())
            return vals.intersection(allowed) if allowed else vals
    return []


class Preprocessing:

    @staticmethod
    def generate_frequent_values(df: pd.DataFrame, col: str, thresh):
        vals = df[col].apply(to_names_list)
        v = pd.Series(Counter(chain.from_iterable(vals)))
        return {ind for ind in v.index if (v > thresh)[ind]}

    def __init__(self, src, encoding=None):
        self.df: pd.DataFrame = pd.read_csv(src, header=0, encoding=encoding)
        self.__process()
        self.df = self.df.fillna(0)

    ############################## Processing ##############################

    def __process(self):
        # change names with Hebrew letters + drop useless
        self.__remove_useless()
        self.__rename()

        # process the columns that stay
        self.__process_age()
        self.__process_basic_stage()  # Heli
        self.__process_Her2()  # Asif
        self.__process_histological_diagnosis()  # Peleg - done
        self.__process_histological_degree()  # Peleg
        self.__process_lvi_lymphovascular_invasion() # Peleg
        self.__process_meta_mark()
        self.__process_margin_type()
        self.__process_side()
        self.__process_stage()
        self.__process_surgery_name1()
        self.__process_surgery_sum()
        self.__process_tumor_mark()
        self.__process_tumor_width()
        self.__process_actual_activity()

    def __rename(self):
        self.df.rename(columns=
                       {'אבחנה-Age': 'Age',
                        'אבחנה-Basic stage': 'Basic stage',
                        'אבחנה-Her2': 'Her2',
                        'אבחנה-Histological diagnosis': 'Histological diagnosis',
                        'אבחנה-Histopatological degree': 'Histopatological degree',
                        'אבחנה-Lvi -Lymphovascular invasion': 'Lvi',
                        'אבחנה-M -metastases mark (TNM)': 'M mark (TNM)',
                        'אבחנה-Margin Type': 'Margin Type',
                        'אבחנה-Side': 'Side',
                        'אבחנה-Stage': 'Stage',
                        'אבחנה-Surgery name1': 'Surgery name1',
                        'אבחנה-Surgery sum': 'Surgery sum',
                        'אבחנה-T -Tumor mark (TNM)': 'Tumor mark',
                        'אבחנה-Tumor width': 'Tumor width',
                        'surgery before or after-Actual activity': 'Actual activity'
                        }, inplace=True)

    def __remove_useless(self):
        self.df.drop([' Form Name', ' Hospital', 'User Name',
                      'אבחנה-Diagnosis date', 'אבחנה-Lymphatic penetration', 'אבחנה-Lymphatic penetration',
                      'אבחנה-N -lymph nodes mark (TNM)', 'אבחנה-Nodes exam',
                      'אבחנה-Positive nodes', 'אבחנה-Surgery date1', 'אבחנה-Surgery date2',
                      'אבחנה-Surgery date3', 'אבחנה-Surgery name2', 'אבחנה-Surgery name3',
                      'אבחנה-Tumor depth', 'surgery before or after-Activity date',
                      'אבחנה-KI67 protein', 'אבחנה-er', 'אבחנה-pr'], axis=1,
                     inplace=True)

    def save(self, path):
        self.df.to_csv(path)

    @staticmethod
    def to_test_results(data: str, test_res_dict: Dict[str, int]) -> int:
        """Gets a data value, and a dict,
        than map each data value to test results that loaded in dict"""
        encoded_res: int = test_res_dict.get(data)
        if encoded_res is None:
            return 0
        return encoded_res

    def __process_age(self):  # TODO: remove?
        self.df = self.df[self.df['Age'] > 0]
        self.df = self.df[self.df['Age'] < 110]

    def __process_basic_stage(self):
        self.df = self.df.dropna(subset=['Basic stage'])  # TODO - Check it's ok
        self.df = pd.get_dummies(self.df, columns=['Basic Stage'],
                                 drop_first=True)

    def __process_Her2(self):
        # Denote -1 for neg, 0 for undefined, 1 for positive
        self.df['Her2'].apply(Preprocessing.to_test_results, args=HER2_DICT)

    def __process_histological_diagnosis(self):  # TODO - implement: parse
        # Denote 1 for invasive or 0 for non-invasive
        def to_invasive_non_invasive(his_diag_res: str) -> int:
            res: int = HIS_DIAG_DICT.get(his_diag_res, default='No diag')
            if res == "No diag":
                return int("INFILTRATING" in his_diag_res)
            return res

        self.df['Histological diagnosis'].apply(to_invasive_non_invasive) \
        .rename(columns={'Histological diagnosis': 'Invasiveness'})

    def __process_histological_degree(self):
        # Denote 0-4 according to degree, null|GX gets 0.
        self.df['Histopatological degree'].apply(lambda x: Preprocessing.to_test_results(x[:2], HIS_DEG_DICT))

    def __process_lvi_lymphovascular_invasion(self):
        # Denote 1 if positive 0 if negative.
        self.df['Lvi'].apply(Preprocessing.to_test_results, args=LVI_DICT)

    def __process_meta_mark(self):
        self.df.loc[self.df['M mark (TNM)'] == '' or \
                    self.df['M mark (TNM)'] == 'Not yet Established', 'M mark (TNM)'] = 'MX'
        self.df = pd.get_dummies(self.df, columns=['M mark (TNM)'])

    def __process_margin_type(self):
        self.df = pd.get_dummies(self.df, columns=['Margin Type'])

    def __process_side(self):
        imp_most_freq: SimpleImputer = SimpleImputer(missing_values=[pd.NA, ''], strategy='most_frequent')
        self.df['Side'] = imp_most_freq.transform(self.df['Side'])
        self.df = pd.get_dummies(self.df, columns=['Side'])

    def __process_stage(self):
        self.df.loc[self.df['Stage'] == '' or \
                    self.df['Stage'] == 'Not yet Established', 'Stage'] = 'Unknown'
        self.df = pd.get_dummies(self.df, columns=['Stage'])

    def __process_surgery_name1(self):
        self.df.loc[self.df['Surgery name1'] == '' or \
                    self.df['Surgery name1'] == 'Not yet Established', 'Surgery name1'] = 'Unknown'
        self.df = pd.get_dummies(self.df, columns=['Surgery name1'])

    def __process_surgery_sum(self):
        imp_constant: SimpleImputer = SimpleImputer(missing_values=[pd.NA, ''], strategy='constant', fill_value=0)
        self.df['Surgery sum'] = imp_constant.transform(self.df['Surgery sum'])

    def __process_tumor_mark(self):
        self.df.loc[self.df['Tumor mark'] == '' or \
                    self.df['Tumor mark'] == 'Not yet Established', 'Tumor mark'] = 'Unknown'
        self.df = pd.get_dummies(self.df, columns=['Tumor mark'])

    def __process_tumor_width(self):
        imp_mean: SimpleImputer = SimpleImputer(missing_values=[pd.NA, ''], strategy='mean')
        self.df['Tumor width'] = imp_mean.transform(self.df['Tumor width'])

    def __process_actual_activity(self):
        self.df.loc[self.df['Actual activity'] == '' or \
                    str(self.df['Actual activity'])[0].isdigit(), 'Actual activity'] = 'Unknown'
        self.df = pd.get_dummies(self.df, columns=['Actual activity'])


if __name__ == '__main__':
    src = 'Data and Supplementary Material-20220601/Mission 2 - Breast Cancer/train.feats.csv'
    # df = pd.read_csv(src, header=0)
    # prof = ProfileReport(df)
    # prof.to_file(output_file='output.html')

    # pre = Preprocessing('train_0.7.csv')
    # pre.save('./init.csv')
