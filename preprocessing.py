import pandas as pd
import numpy as np
import json
from pandas_profiling import ProfileReport
from itertools import chain
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer

import ast
# TODO :
# TODO 1 : Figure out what to do with the id column


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


JSON_COLS = []
CONVERTERS = {key: parse_column for key in JSON_COLS}

SEED = 13


def to_names_list(sub_df, allowed=None):
    if isinstance(sub_df, pd.DataFrame) and not sub_df.empty:
        if 'name' in sub_df.columns:
            vals = set(sub_df['name'].tolist())
            return vals.intersection(allowed) if allowed else vals
    return []


class Preprocessing:

    def __init__(self, src, encoding=None):
        self.df: pd.DataFrame = pd.read_csv(src, converters=CONVERTERS, header=0, encoding=encoding)
        self.__process()
        self.df = self.df.fillna(0)

    ############################## Processing ##############################

    def __process(self):
        # change names with Hebrew letters + drop useless
        self.__rename()
        self.__remove_useless()

        # process the columns that stay
        self.__process_age()
        self.__process_basic_stage() # Heli
        self.__process_Her2() # Asif
        self.__process_histological_diagnosis() # Peleg - done
        self.__process_histological_degree() # Peleg
        self.__process_ivi_lymphovascular_invasion()
        self.__process_KI67_protein()
        self.__process_meta_mark()
        self.__process_margin_type()
        self.__process_side()
        self.__process_stage()
        self.__process_surgery_name1()
        self.__process_surgery_sum()
        self.__process_tumor_mark()
        self.__process_tumor_depth()
        self.__process_er_pr()
        self.__process_actual_activity()

    def __rename(self):
        self.df.rename(columns=
                  {'אבחנה-Age': 'Age',
                   'אבחנה-Basic stage': 'Basic stage',
                   'אבחנה-Her2': 'Her2',
                   'אבחנה-Histological diagnosis': 'Histological diagnosis',
                   'אבחנה-Histopatological degree': 'Histopatological degree',
                   'אבחנה-Ivi -Lymphovascular invasion': 'Ivi -Lymphovascular invasion',
                   'אבחנה-KI67 protein': 'KI67 protein',
                   'אבחנה-M -metastases mark (TNM)': 'M -metastases mark (TNM)',
                   'אבחנה-Margin Type': 'Margin Type',
                   'אבחנה-Side': 'Side',
                   'אבחנה-Stage': 'Stage',
                   'אבחנה-Surgery name1': 'Surgery name1',
                   'אבחנה-Surgery sum': 'Surgery sum',
                   'אבחנה-T -Tumor mark (TNM)': 'T -Tumor mark (TNM)',
                   'אבחנה-Tumor depth': 'Tumor depth',
                   'אבחנה-er': 'er',
                   'אבחנה-pr': 'pr',
                   }, inplace=True)

    def __remove_useless(self):
        self.df.drop(['Form Name', 'Hospital', 'User Name',
                      'אבחנה-Diagnosis date', 'אבחנה-Lymphatic penetration', 'אבחנה-Lymphatic penetration',
                      'אבחנה-N -lymph nodes mark (TNM)', 'אבחנה-Nodes exam',
                      'אבחנה-Positive nodes', 'אבחנה-Surgery date1', 'אבחנה-Surgery date2',
                      'אבחנה-Surgery date3', 'אבחנה-Surgery name2', 'אבחנה-Surgery name3',
                      'אבחנה-Tumor width', 'surgery before or after-Activity date',
                      ''], axis=1,
                     inplace=True)

    def __process_age(self): # TODO: remove?
        self.df = self.df[self.df['Age'] > 0]
        self.df = self.df[self.df['Age'] < 110]

    def __process_basic_stage(self):
        self.df = self.df.dropna(subset=['Basic stage']) # TODO - Check it's ok
        self.df = pd.get_dummies(self.df, columns=['Basic Stage'],
                                 drop_first=True)

    def __process_Her2(self):
        pass

    def __process_histological_diagnosis(self):
        pass

    def __process_histological_degree(self):
        pass

    def __process_ivi_lymphovascular_invasion(self):
        pass

    def __process_KI67_protein(self):
        pass

    def __process_meta_mark(self):
        pass

    def __process_margin_type(self):
        pass

    def __process_side(self):
        pass

    def __process_stage(self):
        pass

    def __process_surgery_name1(self):
        pass

    def __process_surgery_sum(self):
        pass

    def __process_tumor_mark(self):
        pass

    def __process_tumor_depth(self):
        pass

    def __process_er_pr(self):
        pass

    def __process_actual_activity(self):
        pass


if __name__ == '__main__':

    pre = Preprocessing('Data and Supplementary Material-20220601/Mission 2 - Breast Cancer/train.feats.csv')

    # show data with pandas profiling
    src = 'Data and Supplementary Material-20220601/Mission 2 - Breast Cancer/train.feats.csv'
    df = pd.read_csv(src, header=0)
    prof = ProfileReport(df)
    prof.to_file(output_file='output.html')

