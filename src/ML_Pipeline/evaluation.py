import sklearn
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report


class Eval:
    '''Class for model evaluation.'''

    def __init__(self, model, td, intents, test_x, test_y):
        '''Args:
            model (str): Variable name for model path.
            td (:obj:, `dataframe`): Dataframe from prapare data object.
            intents (:obj:`list`): Ordered list of intent labels.
            test_x (array): Numpy array of test utterances.
            test_y (array): Numpy array of one hot encoded test labels.
        '''

        self.test_x = test_x
        self.test_y = test_y

        # get column containing labels of test data
        self.df_test = td.df_test.iloc[:, 0:1]

        # get label predictions of test inputs
        predicts = model.predict(self.test_x, batch_size=32)

        # Get predicted labels that are above 0.5 probability
        self.predict_logits = predicts.argmax(axis=1)

        self.test_y_rounded = np.argmax(self.test_y, axis=1)
        self.model = model
        self.intents = intents

    def get_test_loss_acc(self):
        '''Get accuracy and loss of test data.'''

        test_scores = self.model.evaluate(self.test_x, self.test_y, verbose=2)
        print("Test loss:", test_scores[0])
        print("Test accuracy:", test_scores[1])


    def get_accuracy_plot(self, history):
        '''Plot test accuracy against training accuracy.'''

        plt.plot(history['accuracy'])
        plt.plot(history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()

    def get_loss_plot(self, history):
        '''Plot test loss against training loss.'''

        plt.plot(history['loss'])
        plt.plot(history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()

    def compare_predicted_intents(self):
        '''Get table of predicted vs actuals intent labels.'''

        predicted_labels = [self.intents[logit] for logit in self.predict_logits]
        self.df_test['predicted_intent'] = predicted_labels
        print(self.df_test)

    def get_fscore(self):
        '''Get f-score of predicted labels.'''

        return pd.DataFrame(classification_report(self.test_y_rounded, self.predict_logits, output_dict=True)).T

    def get_confusion_matrix(self):
        '''Get confusion matrix of predicted vs actual labels.'''

        cm = confusion_matrix(self.test_y_rounded, self.predict_logits)
        cm_display = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix = cm)
        cm_display.plot()
        plt.show()
