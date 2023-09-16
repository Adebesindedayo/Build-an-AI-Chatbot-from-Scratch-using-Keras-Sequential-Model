import nltk
import re
import spacy
from nltk.corpus import stopwords

nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])


class Utt:
    '''Class for processing and normalizing raw text.'''

    def __init__(self, text, query=False):
        '''Args:
            text (str): Raw text from chat transcript.
            query (bool, optional): If True, the output of the class will be
            retricted to only the text.
        '''

        self.text = text

        # dictionary of shorthand tokens that require expanding
        self.norm_dict = {
           "btw":"by the way",
           "aint":"is not",
           "arent":"are not",
           "cant":"cannot",
           "cause":"because",
           "couldve":"could have",
           "couldnt":"could not",
           "didnt":"did not",
           "doesnt":"does not",
           "dont":"do not",
           "hadnt":"had not",
           "hasnt":"has not",
           "havent":"have not",
           "hed":"he would",
           "hell":"he will",
           "hes":"he is",
           "howd":"how did",
           "howdy":"how do you",
           "howll":"how will",
           "hows":"how is",
           "Id":"I would",
           "Idve":"I would have",
           "Ill":"I will",
           "Illve":"I will have",
           "Im":"I am",
           "Ive":"I have",
           "id":"i would",
           "idve":"i would have",
           "ill":"i will",
           "illve":"i will have",
           "im":"i am",
           "ive":"i have",
           "isnt":"is not",
           "itd":"it would",
           "itdve":"it would have",
           "itll":"it will",
           "itllve":"it will have",
           "its":"it is",
           "lets":"let us",
           "maam":"madam",
           "maynt":"may not",
           "mightve":"might have",
           "mightnt":"might not",
           "mightntve":"might not have",
           "mustve":"must have",
           "mustnt":"must not",
           "mustntve":"must not have",
           "neednt":"need not",
           "needntve":"need not have",
           "ok": "okay",
           "oclock":"of the clock",
           "oughtnt":"ought not",
           "oughtntve":"ought not have",
           "shant":"shall not",
           "shant":"shall not",
           "shantve":"shall not have",
           "shed":"she would",
           "shedve":"she would have",
           "shell":"she will",
           "shellve":"she will have",
           "shes":"she is",
           "shouldve":"should have",
           "shouldnt":"should not",
           "shouldntve":"should not have",
           "sove":"so have",
           "sos":"so as",
           "thiss":"this is",
           "thatd":"that would",
           "thatdve":"that would have",
           "thats":"that is",
           "thered":"there would",
           "theredve":"there would have",
           "theres":"there is",
           "heres":"here is",
           "theyd":"they would",
           "theydve":"they would have",
           "theyll":"they will",
           "theyllve":"they will have",
           "theyre":"they are",
           "theyve":"they have",
           "tove":"to have",
           "wasnt":"was not",
           "wed":"we would",
           "wedve":"we would have",
           "well":"we will",
           "wellve":"we will have",
           "were":"we are",
           "weve":"we have",
           "werent":"were not",
           "whatll":"what will",
           "whatllve":"what will have",
           "whatre":"what are",
           "whats":"what is",
           "whatve":"what have",
           "whens":"when is",
           "whenve":"when have",
           "whered":"where did",
           "wheres":"where is",
           "whereve":"where have",
           "wholl":"who will",
           "whollve":"who will have",
           "whos":"who is",
           "whove":"who have",
           "whys":"why is",
           "whyve":"why have",
           "willve":"will have",
           "wont":"will not",
           "wontve":"will not have",
           "wouldve":"would have",
           "wouldnt":"would not",
           "wouldntve":"would not have",
           "yall":"you all",
           "yalld":"you all would",
           "yalldve":"you all would have",
           "yallre":"you all are",
           "yallve":"you all have",
           "youd":"you would",
           "youdve":"you would have"
        }

        self.query = query

        # process text
        if not self.query:
            self.participant, self.utt_preprocess, self.utt = self._preprocess_text()
        else:
            self.utt_preprocess = self.text.lstrip().lower()

        # clean utt
        self.utt_clean = self._clean_utt()

    def _preprocess_text(self):
        '''Takes transcript line, removes dates and returns participant number
        and lowered text.'''

        text = re.sub(r'\(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\) ', '', self.text)
        try:
            participant, text = text.split(':', 1)
            if text:
                if participant.startswith('Visitor'):
                    participant = 1
                else:
                    participant = 2
                return participant, text.lstrip().lower(), text
        except ValueError:
            pass

    def _clean_utt(self):
        '''Cleans text of urls and everything by alpha characters
        and single-wide whitespace'''

        utt_clean = re.sub(r'http.*\b', '', self.utt_preprocess)
        utt_clean = utt_clean.replace('-', ' ')
        utt_clean = re.sub(r"[^a-z ]", '', utt_clean)
        self.utt_clean = re.sub(r' {2,}', ' ', utt_clean)

        return self.utt_clean

    def _norm_utt(self):
        '''Replaces contracted tokens with expanded forms in text'''

        for token, repl in self.norm_dict.items():
            utt_norm = self.utt_clean.replace(token, repl)

        return utt_norm

    def _limit_utt_length(self, utt):
        '''Checks if utterance is within is certain length range.
        Returns utt if in range or returns None'''

        if len(utt) > 50:
            utt = None
        if len(utt) < 1:
            utt = None
        return utt

    def parse_utt(self):
        '''Process clean utt to: Normalize contractions, lemmatize tokens,
        check utterance within lenth range. If self.query is True returns utterance
        else returns participant numner, original utt, and preprocessed utt.'''

        if self.utt_clean:
            utt_norm = self._norm_utt()
            doc = nlp(utt_norm)
            list_lemma = [token.lemma_ for token in doc if len(token) > 1]
            utt = self._limit_utt_length(list_lemma)
            if utt:
                utt = ' '.join(utt)
                if not self.query:
                    return self.participant, self.utt, utt
                return utt
