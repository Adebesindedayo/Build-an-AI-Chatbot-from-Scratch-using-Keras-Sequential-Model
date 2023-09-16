import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from .preprocess import Utt


class PrepareData:
    '''Class for preparing data for training.'''

    def __init__(self, json_data, intents):
        '''Args:
            json_data (str): Variable for path to json intents file.
            intents (:obj: `list`): List of intents.
        '''

        # load intents data into dataframe
        self.df = pd.DataFrame.from_dict(json_data)

        self.intents = intents

        self.df_reformat = self._reformat_intents_df()
        self.df_preprocessed = self._preprocess_utts()
        df_one_hot_labels = self._encode_labels()
        self.df_concat = pd.concat([self.df_preprocessed, df_one_hot_labels], axis=1)
        self.df_train, self.df_val, self.df_test = self._get_train_val_test()
        self.train_x, self.val_x, self.test_x, self.train_y, self.val_y, self.test_y = self._get_input_output()

    def _reformat_intents_df(self):
        '''Takes data frame, removes response column
        and expands query list to individual rows with matching intent.'''

        df_T =  self.df.T.reset_index()
        df_T.columns = ["intent", "query", "response"]
        df_drop_response = df_T.drop(columns="response")
        df_drop_response.intent = df_drop_response.intent.astype('category')
        df_explode = df_drop_response.explode('query')
        return df_explode

    def _preprocess_utts(self):
        '''Passes utterances through preprocessing
        and deduplicates utterances of the same intent label.'''

        self.df_reformat['query_preprocessed'] = self.df_reformat.apply(lambda row: Utt(row['query'], query=True).parse_utt(), axis = 1)
        df_filter = self.df_reformat.dropna()
        df_dedup = df_filter.drop_duplicates(subset=['intent', 'query'], keep='first')
        return df_dedup

    def _encode_labels(self):
        '''One hot encodes labels.'''

        df_one_hot_intents = pd.get_dummies(self.df_preprocessed.intent)
        return df_one_hot_intents

    def _get_train_val_test(self):
        '''Split randomized data into train/eval/test with 80:10:10 ratio.'''

        df_train, df_val, df_test = np.split(self.df_concat.sample(frac=1, random_state=42),
                                             [int(.8*len(self.df_concat)), int(.9*len(self.df_concat))])
        return df_train, df_val, df_test

    def _get_input_output(self):
        '''Converts input and output taxt and labels to numpy arrays.'''

        train_queries_list = self.df_train['query_preprocessed'].tolist()
        val_queries_list = self.df_val['query_preprocessed'].tolist()
        test_queries_list = self.df_test['query_preprocessed'].tolist()

        train_x = np.array(train_queries_list, dtype=object)[:, np.newaxis]
        val_x = np.array(val_queries_list, dtype=object)[:, np.newaxis]
        test_x = np.array(test_queries_list, dtype=object)[:, np.newaxis]

        df_train_y = self.df_train.iloc[:,3:]
        df_val_y = self.df_val.iloc[:,3:]
        df_test_y = self.df_test.iloc[:,3:]

        train_y = df_train_y.to_numpy()
        val_y = df_val_y.to_numpy()
        test_y = df_test_y.to_numpy()

        return train_x, val_x, test_x, train_y, val_y, test_y
