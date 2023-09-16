import os
import pandas as pd
from .preprocess import Utt


def parse_files(directory):
    '''Takes a directory containing raw chat logs as input and returns data frame of
    preprocessed text with speaker number.'''

    data = []
    for filename in os.listdir(directory):
        if filename.endswith(".text"):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith(r'('):
                        try:
                            utt_object = Utt(line, query=False)
                            participant, utt, parsed_utt = utt_object.parse_utt()
                            if parsed_utt:
                                data.append([participant, utt, parsed_utt])
                        except TypeError:
                            pass

    if data:
        df = pd.DataFrame(data, columns=['participant', 'original_text', 'text'])
        df.participant = df.participant.astype('category')
        return df


def parse_data_csv(data_csv):
    '''Parse human reviewed data to json file.'''

    df = pd.DataFrame(data)
    df = pd.read_csv(data_csv)
    df_sorted = df.sort_values(by=['label_number']).copy()

    intents_dict = {}

    for idx, row in df_sorted.iterrows():
        if row['label'] not in intents_dict:
            intents_dict[row['label']] = {}
        if row['participant'] and row['participant'] not in intents_dict[row['label']]:
            intents_dict[row['label']][row['participant']] = []
        if row['participant'] == '1':
            if row['original_text'] not in intents_dict[row['label']][row['participant']]:
                intents_dict[row['label']][row['participant']].append(row['original_text'])
        elif row['participant'] == '2':
            if row['original_text'] not in intents_dict[row['label']][row['participant']]:
                intents_dict[row['label']][row['participant']].append(row['original_text'])

    return intents_dict
