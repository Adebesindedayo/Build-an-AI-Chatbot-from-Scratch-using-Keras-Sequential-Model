import random
import json
import numpy as np
from .preprocess import Utt


class Predict:
    '''Class for model inference.'''

    def __init__(self, model, intents, intent_data, utt):
        '''Args:
            model (str): Variable name for model path.
            intents (:obj:`list`): Ordered list of intent labels.
            intent_data (:obj:, `dict`): Dictionary containing intents as keys
            and and dictionaries of queries and responses as values.
            utt (str): Utterance to be used to predict intent.
        '''

        self.intent_data = intent_data

        self.intents = intents

        self.model = model

        # instantiate Utt object
        parsed_utt = Utt(utt, query=True)

        # preprocess utterance
        self.utt = parsed_utt.parse_utt()

        if self.utt:
            self.predicted_intent = self._predict_intent()

            if not self.predicted_intent:
                self.response = "Sorry, I didn't understand that. Please can you rephrase your message."
            else:
                self.response = self._get_response()
        else:
            self.response = "Sorry, I didn't understand that. Please can you rephrase your message."

    def _predict_intent(self):
        '''Given utterance, predicts intent label.'''

        query = np.array([self.utt], dtype=object)
        predict = self.model.predict(query, batch_size=32, verbose=0)
        predict_logit = predict.argmax(axis=1)
        predicted_label = self.intents[predict_logit[0]]
        if predicted_label:
            return predicted_label
        return None

    def _get_response(self):
        '''Given label, generates random response from list of available responses.'''

        response_list = self.intent_data[self.predicted_intent]['2']
        return random.choice(response_list)
