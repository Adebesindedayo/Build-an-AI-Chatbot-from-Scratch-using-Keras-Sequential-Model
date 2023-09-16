import chatintents
import pandas as pd
import tensorflow_hub as hub
from chatintents import ChatIntents
from hyperopt import hp


class ClusterIntents:
    '''Class for clustering utterances and generating labels for clusters.'''

    def __init__(self, data, label_file, module_url, sorted_labels=False):
        '''Args:
            data (:obj:`dataframe`): Tabularised chat transcript data.
            label_file (str): Variable for path to save output file with labelled utterances.
            module_url (str): Variable for url to pretrained Universal
            Sentence Encoder model.
            sorted_labels (bool, optional): Option to sort output file by labels instead of
            by original chat order.
        '''

        embedder = hub.load(module_url)
        print(f"module {module_url} loaded")

        self.data = data
        self.label_file = label_file

        # list of utterances
        all_intents = self.data['text'].tolist()

        #numpy array of document embeddings
        embeddings = embedder(all_intents)

        # convert embeddings array to sentence embeddings using USE
        model = ChatIntents(embeddings, 'use')

        # hyperparameter search configuration
        hspace = {
            "n_neighbors": hp.choice('n_neighbors', range(3,16)),
            "n_components": hp.choice('n_components', range(3,16)),
            "min_cluster_size": hp.choice('min_cluster_size', range(23,38)),
            "min_samples": None,
            "random_state": 42
        }

        # label number upper and lower bounds
        label_lower = 30
        label_upper = 100

        # maximum number of search runs
        max_evals = 25

        # run bayesian hyperparameter search
        model.bayesian_search(space=hspace,
                              label_lower=label_lower,
                              label_upper=label_upper,
                              max_evals=max_evals)

        # print the best model parameters found search
        model.best_params

        # hyperparameter optimized instance attribute
        self.model_final = model

        # generate summary dataframe and labeled utts dataframe
        self.df_summary, self.labeled_utts = self.model_final.apply_and_summarize_labels(data[['text']])

        # original data with appended labels data
        self.labeled_data = self._get_labeled_data()

        if sorted_labels:
            self.labeled_data = self.get_sorted_labels()

        # output csv file containing extended data input
        self._get_data_csv()
        print(f'Labelled utterances successfuly written to {label_file}')

    def get_model_best_params(self):
        '''Prints the final best parameters derived through search.'''

        return self.model_final.trials.best_trial

    def get_cluster_plot(self):
        '''Plot the clusters found from clustering.'''

        return self.model_final.plot_best_clusters()

    def get_labels_summary(self, n: int):
        '''Print n slice of labels summary.'''

        return self.df_summary.head(n)

    def get_labeled_utts(self, n: int):
        '''Print n slice of labelled utterances.'''

        return self.labeled_utts.head(n)

    def _get_labeled_data(self):
        '''Takes original data input adds label column and then removes duplicate
        utterances that have the same label.'''

        data_copy = self.data.copy()
        data_copy['label'] = self.labeled_utts['label']
        data_copy_dedup = data_copy.drop_duplicates(subset=['original_text', 'text', 'label'], keep='first')
        return data_copy_dedup

    def _get_sorted_labels(self):
        '''Sort extended data table by label.'''

        return self.labeled_data.sort_values(by=['label'])

    def _get_data_csv(self):
        '''Exports extended data to csv file.'''

        return self.labeled_data.to_csv(self.label_file)
