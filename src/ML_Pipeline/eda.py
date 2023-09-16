import nltk
from nltk.probability import FreqDist
import matplotlib.pyplot as plt


class ExploreData:
    '''Class for exploring features of data in dataframe.'''

    def __init__(self, data):
        '''Args:
            data (:obj:`dataframe`): Tabularised chat transcript data.
        '''

        # concatenated text data
        self.text = ' '.join(list(data['text'].values))

        # list of utterances
        self.sents = data["text"].tolist()

        # list of all tokens
        self.tokens = self.text.split()

    def get_token_frequency_dist(self):
        '''Takes list of tokens and prints frequency distribution of tokens.'''

        print(FreqDist(self.tokens))

    def get_top_n_tokens(self, n: int):
        '''Takens integer as n and returns top n high frequency tokens.'''

        return FreqDist(self.tokens).most_common(n)

    def plot_dist_curve(self):
        '''Plots frequency (zipf) curve of tokens.'''

        fig, ax = plt.subplots(figsize=(12,8))
        FreqDist(self.tokens).plot(20, cumulative=False)
        plt.show()

    def get_token_length_visualisations(self):
        '''Plot histogram of token lengths.'''

        lengths = [len(i) for i in self.tokens]
        plt.figure(figsize=(13,6))
        plt.hist(lengths, bins = 40)
        plt.title("Length of tokens in text")
        plt.show()

    def get_sent_length_visuals(self):
        '''Plot histogram of sentence lengths.'''

        lengths = [len(i) for i in self.sents]
        plt.figure(figsize=(13,6))
        plt.hist(lengths, bins = 40)
        plt.title("Length of sentences")
        plt.show()
